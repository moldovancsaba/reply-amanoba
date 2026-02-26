from pathlib import Path
import csv
import json
import re
import threading
import time
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import requests

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

from app.config import settings
from app.db import Database
from app.embeddings import Embedder
from app.llm import OllamaLLM
from app.rag import read_pdf_text, reindex_docs, retrieve
from app.schemas import (
    AskRequest,
    AskResponse,
    CompareRequest,
    EnrichmentEscalateRequest,
    EditorResponseRequest,
    EnrichmentAnswerRequest,
    EnrichmentGenerateRequest,
    LanguagePolicyRequest,
    ModelSelectRequest,
    QAExportRequest,
    TicketDismissRequest,
    ChatHistoryRequest,
    ChatMessageRequest,
    ChatSessionCreateRequest,
)

app = FastAPI(title="{reply}")

db = Database(settings.db_path)
embedder = Embedder(settings.embed_model)
llm = OllamaLLM(settings.ollama_url, settings.ollama_model)
web_dir = Path(__file__).resolve().parent / "web"
web_path = web_dir / "index.html"
allowed_upload_exts = {".pdf", ".md", ".txt"}
active_model = {"name": settings.ollama_model}
tunnel_log_path = Path("/tmp/helpbot-tunnel.log")
supported_languages = ["hu", "en", "ro", "es", "fr", "de", "it", "pt"]
language_markers = {
    "hu": [" mi ", " vagy ", " és ", " hogy ", " az ", " egy ", "kérdés", "válasz"],
    "ro": [" și ", " este ", " sunt ", " ce ", " pentru ", " întrebare", "răspuns"],
    "es": [" el ", " la ", " de ", " que ", " para ", " pregunta", "respuesta"],
    "fr": [" le ", " la ", " de ", " et ", " pour ", " question", "réponse"],
    "de": [" der ", " die ", " und ", " für ", " ist ", " frage", "antwort"],
    "it": [" il ", " la ", " e ", " per ", " che ", " domanda", "risposta"],
    "pt": [" o ", " a ", " e ", " para ", " que ", " pergunta", "resposta"],
    "en": [" the ", " and ", " what ", " is ", " are ", " question", "answer"],
}

app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

cors_origins = settings.chat_allowed_origins or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False if "*" in cors_origins else True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

chat_rate_window: dict[str, list[float]] = {}
chat_rate_lock = threading.Lock()


def normalize_lang(code: str) -> str:
    c = (code or "").strip().lower()
    return c if c in supported_languages else "en"


def load_language_policy() -> dict:
    primary = db.get_setting("language_primary") or settings.language_primary
    allowed_raw = db.get_setting("language_allowed")
    if allowed_raw:
        allowed = [normalize_lang(x) for x in allowed_raw.split(",") if x.strip()]
    else:
        allowed = [normalize_lang(x) for x in settings.language_allowed if x.strip()]

    primary = normalize_lang(primary)
    allowed = [x for x in allowed if x in supported_languages]
    if not allowed:
        allowed = [primary]
    if primary not in allowed:
        allowed.insert(0, primary)
    # Stable unique order.
    deduped = []
    seen = set()
    for x in allowed:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    return {"primary": primary, "allowed": deduped}


def save_language_policy(primary: str, allowed: list[str]) -> dict:
    p = normalize_lang(primary)
    clean = [normalize_lang(x) for x in allowed if normalize_lang(x) in supported_languages]
    if p not in clean:
        clean.insert(0, p)
    deduped = []
    seen = set()
    for x in clean:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    db.set_setting("language_primary", p)
    db.set_setting("language_allowed", ",".join(deduped))
    return {"primary": p, "allowed": deduped}


language_policy = load_language_policy()


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)
    return cleaned or "uploaded_document.txt"


def unique_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    i = 1
    while True:
        candidate = base_path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def ollama_health() -> dict:
    info = {
        "provider": settings.model_provider,
        "url": settings.ollama_url,
        "model": settings.ollama_model,
        "reachable": False,
        "model_available": False,
        "error": None,
    }
    try:
        resp = requests.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
        info["reachable"] = True
        payload = resp.json()
        models = [m.get("model", "") for m in payload.get("models", [])]
        info["model_available"] = settings.ollama_model in models
    except Exception as exc:
        info["error"] = str(exc)
    return info


def ollama_models() -> list[str]:
    try:
        resp = requests.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        return [m.get("model", "") for m in payload.get("models", []) if m.get("model")]
    except Exception:
        return []


def docs_annotation_status() -> list[dict]:
    settings.docs_path.mkdir(parents=True, exist_ok=True)
    indexed = {r["source"]: int(r["chunk_count"]) for r in db.doc_chunks_by_source()}

    statuses: list[dict] = []
    for path in sorted(settings.docs_path.glob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in allowed_upload_exts:
            continue
        chunks = indexed.get(path.name, 0)
        statuses.append(
            {
                "filename": path.name,
                "format": ext,
                "size_bytes": path.stat().st_size,
                "annotated": chunks > 0,
                "indexed_chunks": chunks,
            }
        )

    return statuses


def current_public_url() -> str | None:
    if not tunnel_log_path.exists():
        return None
    raw = tunnel_log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"https://[-a-z0-9]+\.trycloudflare\.com", raw)
    if not matches:
        return None
    return matches[-1]


def chat_origin_allowed(origin: str | None) -> bool:
    allowed = settings.chat_allowed_origins or ["*"]
    if "*" in allowed:
        return True
    if not origin:
        return True
    return origin in allowed


def chat_token_valid(request: Request) -> bool:
    required = settings.chat_api_token.strip()
    if not required:
        return True
    auth = (request.headers.get("authorization") or "").strip()
    header_token = (request.headers.get("x-api-token") or "").strip()
    bearer = ""
    if auth.lower().startswith("bearer "):
        bearer = auth[7:].strip()
    provided = bearer or header_token
    return bool(provided) and provided == required


def check_chat_rate_limit(request: Request) -> None:
    max_per_min = max(1, int(settings.chat_rate_limit_per_min))
    now = time.time()
    ip = request.client.host if request.client else "unknown"
    with chat_rate_lock:
        hits = chat_rate_window.get(ip, [])
        hits = [t for t in hits if now - t < 60.0]
        if len(hits) >= max_per_min:
            chat_rate_window[ip] = hits
            raise HTTPException(status_code=429, detail="Chat rate limit exceeded. Try again in a minute.")
        hits.append(now)
        chat_rate_window[ip] = hits


def require_chat_access(request: Request, apply_rate_limit: bool = False) -> None:
    if not settings.chat_enabled:
        raise HTTPException(status_code=403, detail="Webchat is disabled.")
    if not chat_token_valid(request):
        raise HTTPException(status_code=401, detail="Invalid or missing chat API token.")
    origin = request.headers.get("origin")
    if not chat_origin_allowed(origin):
        raise HTTPException(status_code=403, detail="Origin is not allowed for webchat.")
    if apply_rate_limit:
        check_chat_rate_limit(request)


def export_name(ext: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"qa-export-{stamp}.{ext}"


def write_qa_jsonl(records: list[dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_qa_csv(records: list[dict], out_path: Path) -> None:
    fields = ["id", "user_id", "question", "answer", "status", "citations", "source", "created_at"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            line = dict(row)
            line["citations"] = ", ".join(line.get("citations") or [])
            writer.writerow({k: line.get(k) for k in fields})


def write_qa_md(records: list[dict], out_path: Path) -> None:
    lines = ["# Q/A Export", ""]
    for row in records:
        lines.append(f"## #{row.get('id')} - {row.get('status', 'n/a')}")
        lines.append(f"- User: {row.get('user_id', '')}")
        lines.append(f"- Source: {row.get('source', '')}")
        lines.append(f"- Time: {row.get('created_at', '')}")
        citations = row.get("citations") or []
        lines.append(f"- Citations: {', '.join(citations) if citations else 'none'}")
        lines.append("")
        lines.append("### Question")
        lines.append(str(row.get("question", "")).strip() or "(empty)")
        lines.append("")
        lines.append("### Answer")
        lines.append(str(row.get("answer", "")).strip() or "(none)")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wrap_text_to_width(text: str, width: int = 115) -> list[str]:
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    if not words or words == [""]:
        return [""]
    lines: list[str] = []
    current = ""
    for w in words:
        proposal = f"{current} {w}".strip()
        if len(proposal) <= width:
            current = proposal
            continue
        if current:
            lines.append(current)
        current = w
    if current:
        lines.append(current)
    return lines


def write_qa_pdf(records: list[dict], out_path: Path) -> None:
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(status_code=500, detail="PDF export requires reportlab. Install dependencies and retry.")

    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    x = 40
    y = height - 40
    line_height = 14

    def put_line(text: str, bold: bool = False) -> None:
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
        c.drawString(x, y, text[:2000])
        y -= line_height
        if y < 50:
            c.showPage()
            y = height - 40

    put_line("Q/A Export", bold=True)
    put_line("")
    for row in records:
        put_line(f"#{row.get('id')} [{row.get('status', 'n/a')}] {row.get('created_at', '')}", bold=True)
        put_line(f"User: {row.get('user_id', '')} | Source: {row.get('source', '')}")
        citations = row.get("citations") or []
        put_line(f"Citations: {', '.join(citations) if citations else 'none'}")
        put_line("Question:", bold=True)
        for ln in _wrap_text_to_width(str(row.get("question", ""))):
            put_line(ln)
        put_line("Answer:", bold=True)
        for ln in _wrap_text_to_width(str(row.get("answer", "")) or "(none)"):
            put_line(ln)
        put_line("")

    c.save()


def detect_language(text: str) -> str:
    t = f" {text.lower()} "
    for lang, words in language_markers.items():
        if any(w in t for w in words):
            return lang
    return "en"


def language_profile(text: str) -> dict[str, int]:
    t = f" {text.lower()} "
    scores: dict[str, int] = {}
    for lang, words in language_markers.items():
        score = sum(1 for w in words if w in t)
        scores[lang] = score
    return scores


def is_mixed_language(text: str) -> bool:
    scores = language_profile(text)
    positives = [s for s in scores.values() if s > 0]
    if len(positives) <= 1:
        return False
    top_two = sorted(positives, reverse=True)[:2]
    # Mixed if at least two languages have meaningful signal.
    return top_two[1] >= 1


def is_language_compliant(text: str, target_lang: str) -> bool:
    if not text.strip():
        return False
    detected = normalize_lang(detect_language(text))
    mixed = is_mixed_language(text)
    return detected == normalize_lang(target_lang) and not mixed


def translate_to_language(text: str, target_lang: str, model_name: str) -> str:
    prompt = f"""
Translate the following text to language code '{target_lang}'.
Rules:
- Preserve citations like [doc:...#...]
- Keep technical terms if needed
- Return only translated text

TEXT:
{text}
""".strip()
    resp = requests.post(
        f"{settings.ollama_url.rstrip('/')}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


def enforce_language_policy(text: str, model_name: str, preferred_lang: str | None = None) -> str:
    if not text.strip():
        return text
    target_lang = normalize_lang(preferred_lang or language_policy["primary"])
    if target_lang not in language_policy["allowed"]:
        target_lang = language_policy["primary"]
    detected = normalize_lang(detect_language(text))
    mixed = is_mixed_language(text)
    if detected == target_lang and not mixed:
        return text
    try:
        translated = translate_to_language(text, target_lang, model_name)
        return translated or text
    except Exception:
        return text


def normalize_or_none(text: str, model_name: str, target_lang: str) -> str | None:
    target = normalize_lang(target_lang)
    if is_language_compliant(text, target):
        return text
    try:
        translated = translate_to_language(text, target, model_name)
    except Exception:
        return None
    if not translated.strip():
        return None
    if not is_language_compliant(translated, target):
        return None
    return translated


def switch_notice(lang: str, old_model: str, new_model: str) -> str:
    messages = {
        "hu": f"Megvaltoztattam az aktiv AI agentet ({old_model} -> {new_model}), hogy valaszolni tudjak.",
        "ro": f"Am schimbat agentul AI activ ({old_model} -> {new_model}) pentru a-ti putea raspunde.",
        "es": f"Cambie el agente de IA activo ({old_model} -> {new_model}) para poder responderte.",
        "fr": f"J'ai change l'agent IA actif ({old_model} -> {new_model}) pour pouvoir te repondre.",
        "de": f"Ich habe den aktiven KI-Agenten gewechselt ({old_model} -> {new_model}), um dir zu antworten.",
        "it": f"Ho cambiato l'agente AI attivo ({old_model} -> {new_model}) per poterti rispondere.",
        "pt": f"Mudei o agente de IA ativo ({old_model} -> {new_model}) para conseguir te responder.",
        "en": f"I switched the active AI agent ({old_model} -> {new_model}) to answer your request.",
    }
    return messages.get(lang, messages["en"])


def extract_doc_citations(answer: str) -> set[str]:
    return set(re.findall(r"\[(doc:[^\]]+)\]", answer or ""))


def has_valid_citation(answer: str, allowed_citations: list[str]) -> bool:
    cited = extract_doc_citations(answer)
    if not cited:
        return False
    allowed = set(allowed_citations)
    return bool(cited.intersection(allowed))


def compute_confidence(hits: list, citation_ok: bool) -> float:
    if not hits:
        return 0.0
    top = max(0.0, min(1.0, float(hits[0].score)))
    avg = max(0.0, min(1.0, float(sum(h.score for h in hits[:3]) / max(1, len(hits[:3])))))
    retrieval_conf = (0.7 * top) + (0.3 * avg)
    citation_conf = 1.0 if citation_ok else 0.0
    score = (0.8 * retrieval_conf) + (0.2 * citation_conf)
    return round(max(0.0, min(1.0, score)), 4)


def stale_sources(sources: list[str]) -> list[str]:
    if settings.stale_doc_policy == "off":
        return []
    now = datetime.now(timezone.utc)
    max_age_days = max(0, int(settings.stale_doc_days))
    stale: list[str] = []
    seen = set()
    for source in sources:
        if source in seen:
            continue
        seen.add(source)
        p = settings.docs_path / source
        if not p.exists() or not p.is_file():
            # Knowledge entries that are not physical docs are treated as fresh.
            continue
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        except Exception:
            continue
        age_days = (now - mtime).days
        if age_days > max_age_days:
            stale.append(source)
    return stale


def answer_with_fallback(question: str, context_blocks: list[str], selected_model: str) -> tuple[str, str, str | None]:
    # Returns answer, model_used, switched_from_model
    try:
        answer = llm.answer_with_context(question, context_blocks, model=selected_model)
        return (answer or "").strip(), selected_model, None
    except Exception:
        candidates = [m for m in ollama_models() if m != selected_model]
        for fallback_model in candidates:
            try:
                answer = llm.answer_with_context(question, context_blocks, model=fallback_model)
                active_model["name"] = fallback_model
                return (answer or "").strip(), fallback_model, selected_model
            except Exception:
                continue
        raise


def parse_json_list(raw: str) -> list[str]:
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass

    match = re.search(r"\[(.*?)\]", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads("[" + match.group(1) + "]")
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass

    # Fallback for numbered/bulleted lists.
    lines = []
    for ln in raw.splitlines():
        item = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", ln).strip()
        if len(item) >= 8 and item.endswith("?"):
            lines.append(item)
    return lines


def read_uploaded_doc(source_doc: str) -> str:
    p = settings.docs_path / source_doc
    if not p.exists() or not p.is_file():
        return ""
    ext = p.suffix.lower()
    if ext in {".md", ".txt"}:
        return p.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        return read_pdf_text(p)
    return ""


def generate_clarification_questions(source_doc: str, model_name: str, questions_per_doc: int = 5) -> list[str]:
    text = read_uploaded_doc(source_doc).strip()
    if not text:
        return []

    sample = re.sub(r"\s+", " ", text)[:8000]
    prompt = f"""
You are helping create an internal knowledge base from corporate documents.
Read the DOCUMENT and produce exactly {questions_per_doc} clarification questions.

Goal:
- Ask only high-impact questions where details are missing/ambiguous.
- Questions should help employees get more precise, policy-safe answers later.
- Keep questions concise and specific.
- Use ONLY language code '{language_policy["primary"]}'.
- Return ONLY a JSON array of strings.

DOCUMENT_SOURCE: {source_doc}
DOCUMENT:
{sample}
""".strip()

    resp = requests.post(
        f"{settings.ollama_url.rstrip('/')}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = (resp.json().get("response") or "").strip()
    questions = parse_json_list(raw)

    # Keep only unique, non-empty first N.
    deduped: list[str] = []
    seen = set()
    for q in questions:
        q = normalize_or_none(q, model_name, language_policy["primary"])
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
        if len(deduped) >= questions_per_doc:
            break
    return deduped


def add_enrichment_questions_unique(source_doc: str, questions: list[str]) -> int:
    created = 0
    for q in questions:
        text = q.strip()
        if not text:
            continue
        if db.find_open_enrichment_by_question(text):
            continue
        db.create_enrichment_question(source_doc=source_doc, question=text)
        created += 1
    return created


def generate_strategic_questions_from_context(
    model_name: str, desired_count: int = 5, include_general_knowledge: bool = True
) -> list[str]:
    docs = [d["filename"] for d in docs_annotation_status()][:20]
    ticket_questions = [t["question"] for t in db.open_tickets()][:30]
    if not docs and not ticket_questions:
        return []

    knowledge_mode = "Use broad domain best practices to ask stronger clarifications." if include_general_knowledge else "Use only the provided context."
    prompt = f"""
You are a knowledge engineer preparing follow-up questions for an internal help bot.
Create {desired_count} high-value clarification questions that improve answer quality.

Focus areas:
1) industry context
2) product specifics
3) workflows / process steps
4) value proposition / business outcomes
5) policy or terminology ambiguities seen in user tickets

Rules:
- Questions must be practical and answerable by the editor.
- Prioritize gaps implied by OPEN_TICKETS.
- Avoid duplicates and vague wording.
- Use ONLY language code '{language_policy["primary"]}'.
- Return ONLY a JSON array of strings.

{knowledge_mode}

DOCUMENTS:
{json.dumps(docs, ensure_ascii=True)}

OPEN_TICKETS:
{json.dumps(ticket_questions, ensure_ascii=True)}
""".strip()

    resp = requests.post(
        f"{settings.ollama_url.rstrip('/')}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = (resp.json().get("response") or "").strip()
    items = parse_json_list(raw)

    deduped: list[str] = []
    seen = set()
    for q in items:
        q = normalize_or_none(q, model_name, language_policy["primary"])
        if not q:
            continue
        k = q.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        deduped.append(q.strip())
        if len(deduped) >= desired_count:
            break
    return deduped


def ensure_enrichment_backlog(min_open: int = 5, include_general_knowledge: bool = True) -> int:
    open_count = len(db.open_enrichment_questions())
    if open_count >= min_open:
        return 0

    need = min_open - open_count
    created = 0

    # First, from docs directly.
    for doc in [d["filename"] for d in docs_annotation_status()]:
        if created >= need:
            break
        try:
            qs = generate_clarification_questions(
                source_doc=doc,
                model_name=active_model["name"],
                questions_per_doc=min(need - created, 5),
            )
        except Exception:
            qs = []
        created += add_enrichment_questions_unique(source_doc=doc, questions=qs)

    # Then strategic ticket+domain questions.
    if created < need:
        try:
            strategic = generate_strategic_questions_from_context(
                model_name=active_model["name"],
                desired_count=need - created,
                include_general_knowledge=include_general_knowledge,
            )
        except Exception:
            strategic = []
        created += add_enrichment_questions_unique(source_doc="_SYSTEM_CONTEXT_", questions=strategic)

    return created


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/")
def web_ui() -> FileResponse:
    return FileResponse(web_path)


@app.post("/ingest/reindex")
def ingest_reindex() -> dict:
    count = reindex_docs(db=db, embedder=embedder, docs_path=settings.docs_path)
    return {"indexed_chunks": count, "docs_path": str(settings.docs_path)}


@app.get("/dashboard/health")
def dashboard_health() -> dict:
    docs = db.docs_summary()
    tickets = db.tickets_summary()
    interactions = db.interactions_summary()
    enrichment = db.enrichment_summary()
    status_by_doc = docs_annotation_status()

    total_interactions = interactions["total_count"]
    answer_rate = (
        round((interactions["answered_count"] / total_interactions) * 100, 2)
        if total_interactions
        else 0.0
    )

    return {
        "system": {
            "api": "ok",
            "embedding_model": settings.embed_model,
            "embedding_loaded": embedder.model is not None,
            "strict_citation_gate": settings.strict_citation_gate,
            "confidence_threshold": settings.confidence_threshold,
            "hybrid_vector_weight": settings.hybrid_vector_weight,
            "stale_doc_policy": settings.stale_doc_policy,
            "stale_doc_days": settings.stale_doc_days,
            "language_primary": language_policy["primary"],
            "language_allowed": language_policy["allowed"],
            "llm": ollama_health(),
            "active_model": active_model["name"],
        },
        "annotation": {
            "docs_folder": str(settings.docs_path),
            "supported_formats": sorted(allowed_upload_exts),
            "uploaded_docs_count": len(status_by_doc),
            "indexed_chunks_total": int(docs["total_chunks"] or 0),
            "indexed_sources_total": int(docs["total_sources"] or 0),
            "last_indexed_at": docs.get("last_indexed_at"),
            "document_status": status_by_doc,
        },
        "operations": {
            "tickets": tickets,
            "enrichment": enrichment,
            "interactions": interactions,
            "answer_rate_percent": answer_rate,
        },
    }


@app.get("/models")
def models() -> dict:
    return {
        "active_model": active_model["name"],
        "available_models": ollama_models(),
    }


@app.get("/admin/public-url")
def admin_public_url() -> dict:
    url = current_public_url()
    return {"public_url": url}


@app.get("/admin/language-policy")
def admin_language_policy() -> dict:
    return {
        "primary_language": language_policy["primary"],
        "allowed_languages": language_policy["allowed"],
        "supported_languages": supported_languages,
    }


@app.post("/admin/language-policy")
def admin_set_language_policy(payload: LanguagePolicyRequest) -> dict:
    global language_policy
    language_policy = save_language_policy(payload.primary_language, payload.allowed_languages)
    return {
        "status": "ok",
        "primary_language": language_policy["primary"],
        "allowed_languages": language_policy["allowed"],
    }


@app.post("/models/select")
def select_model(payload: ModelSelectRequest) -> dict:
    available = ollama_models()
    if payload.model not in available:
        raise HTTPException(status_code=400, detail=f"Model not installed. Available: {available}")
    active_model["name"] = payload.model
    return {"status": "ok", "active_model": active_model["name"]}


@app.get("/admin/webchat/snippet")
def admin_webchat_snippet(request: Request) -> dict:
    base_url = current_public_url() or str(request.base_url).rstrip("/")
    token_hint = "<CHAT_API_TOKEN>"
    snippet = (
        "<script>\n"
        f"  window.REPLY_WEBCHAT_CONFIG = {{ baseUrl: '{base_url}', token: '{token_hint}', userId: 'employee-1' }};\n"
        "  const s = document.createElement('script');\n"
        "  s.src = `${window.REPLY_WEBCHAT_CONFIG.baseUrl}/static/webchat.js`;\n"
        "  document.head.appendChild(s);\n"
        "</script>"
    )
    return {
        "status": "ok",
        "chat_enabled": settings.chat_enabled,
        "allowed_origins": settings.chat_allowed_origins,
        "rate_limit_per_min": settings.chat_rate_limit_per_min,
        "snippet": snippet,
    }


@app.get("/chat/config")
def chat_config() -> dict:
    return {
        "enabled": settings.chat_enabled,
        "allowed_origins": settings.chat_allowed_origins,
        "rate_limit_per_min": settings.chat_rate_limit_per_min,
        "auth_required": bool(settings.chat_api_token.strip()),
    }


@app.post("/chat/session")
def chat_create_session(payload: ChatSessionCreateRequest, request: Request) -> dict:
    require_chat_access(request, apply_rate_limit=False)
    session_id = db.create_chat_session(
        user_id=payload.user_id,
        source=payload.source,
        metadata=payload.metadata,
    )
    return {"status": "ok", "session_id": session_id}


@app.post("/chat/history")
def chat_history(payload: ChatHistoryRequest, request: Request) -> dict:
    require_chat_access(request, apply_rate_limit=False)
    session = db.get_chat_session(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    messages = db.chat_history(payload.session_id, limit=payload.limit)
    return {"status": "ok", "session": session, "messages": messages}


@app.post("/chat/message")
def chat_message(payload: ChatMessageRequest, request: Request) -> dict:
    require_chat_access(request, apply_rate_limit=True)
    session = db.get_chat_session(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    db.append_chat_message(payload.session_id, role="user", content=message, status="received")
    ask_result = ask(AskRequest(user_id=session["user_id"], question=message, model=payload.model, source="webchat"))

    if ask_result.status == "answered":
        assistant_text = ask_result.answer or ""
    else:
        assistant_text = (
            f"Escalated to editor. Ticket #{ask_result.ticket_id}\n"
            f"Reason: {ask_result.reason or 'Insufficient context.'}"
        )

    db.append_chat_message(
        payload.session_id,
        role="assistant",
        content=assistant_text,
        status=ask_result.status,
        citations=ask_result.citations,
    )

    return {
        "status": "ok",
        "session_id": payload.session_id,
        "result": ask_result.model_dump(),
        "assistant_message": assistant_text,
    }


@app.post("/chat/stream")
def chat_stream(payload: ChatMessageRequest, request: Request) -> StreamingResponse:
    require_chat_access(request, apply_rate_limit=True)
    session = db.get_chat_session(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    db.append_chat_message(payload.session_id, role="user", content=message, status="received")
    ask_result = ask(AskRequest(user_id=session["user_id"], question=message, model=payload.model, source="webchat"))
    assistant_text = ask_result.answer or ""
    if ask_result.status != "answered":
        assistant_text = (
            f"Escalated to editor. Ticket #{ask_result.ticket_id}\n"
            f"Reason: {ask_result.reason or 'Insufficient context.'}"
        )
    db.append_chat_message(
        payload.session_id,
        role="assistant",
        content=assistant_text,
        status=ask_result.status,
        citations=ask_result.citations,
    )

    payload_json = {
        "status": "ok",
        "session_id": payload.session_id,
        "result": ask_result.model_dump(),
        "assistant_message": assistant_text,
    }

    def event_stream():
        yield "event: message\n"
        yield f"data: {json.dumps(payload_json, ensure_ascii=False)}\n\n"
        yield "event: done\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/upload")
async def upload_docs(files: list[UploadFile] = File(...), reindex: bool = True, enrich: bool = True) -> dict:
    settings.docs_path.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    rejected: list[dict] = []

    for file in files:
        original = file.filename or "uploaded_document.txt"
        ext = Path(original).suffix.lower()
        if ext not in allowed_upload_exts:
            rejected.append({"filename": original, "reason": "Unsupported format"})
            await file.close()
            continue

        dest = unique_path(settings.docs_path / safe_filename(original))
        payload = await file.read()
        dest.write_bytes(payload)
        saved.append(dest.name)
        await file.close()

    result = {"saved": saved, "rejected": rejected}
    if reindex and saved:
        result["indexed_chunks"] = reindex_docs(db=db, embedder=embedder, docs_path=settings.docs_path)
    if enrich and saved:
        generated_count = 0
        for doc_name in saved:
            try:
                questions = generate_clarification_questions(
                    source_doc=doc_name,
                    model_name=active_model["name"],
                    questions_per_doc=5,
                )
            except Exception:
                questions = []
            generated_count += add_enrichment_questions_unique(source_doc=doc_name, questions=questions)
        result["generated_enrichment_questions"] = generated_count

    return result


@app.post("/enrichment/generate")
def enrichment_generate(payload: EnrichmentGenerateRequest) -> dict:
    docs = payload.source_docs or [d["filename"] for d in docs_annotation_status()]
    created = 0
    per_doc: dict[str, int] = {}
    for source_doc in docs:
        try:
            questions = generate_clarification_questions(
                source_doc=source_doc,
                model_name=active_model["name"],
                questions_per_doc=payload.questions_per_doc,
            )
        except Exception:
            questions = []
        added = add_enrichment_questions_unique(source_doc=source_doc, questions=questions)
        created += added
        per_doc[source_doc] = per_doc.get(source_doc, 0) + added

    if payload.include_ticket_based:
        try:
            strategic = generate_strategic_questions_from_context(
                model_name=active_model["name"],
                desired_count=payload.questions_per_doc,
                include_general_knowledge=payload.include_general_knowledge,
            )
        except Exception:
            strategic = []
        added = add_enrichment_questions_unique(source_doc="_SYSTEM_CONTEXT_", questions=strategic)
        created += added
        per_doc["_SYSTEM_CONTEXT_"] = per_doc.get("_SYSTEM_CONTEXT_", 0) + added

    return {"status": "ok", "created_questions": created, "by_doc": per_doc}


@app.get("/enrichment/open")
def enrichment_open() -> dict:
    return {"questions": db.open_enrichment_questions()}


@app.get("/enrichment/improve-me")
def enrichment_improve_me() -> dict:
    def pick_next_compliant() -> dict | None:
        open_questions = db.open_enrichment_questions()
        target = language_policy["primary"]
        for q in open_questions:
            norm = normalize_or_none(q["question"], active_model["name"], target)
            if norm is None:
                db.escalate_enrichment_question(q["id"], "Language policy violation: could not normalize question")
                continue
            if norm != q["question"]:
                db.update_enrichment_question_text(q["id"], norm)
                q["question"] = norm
            return q
        return None

    q = pick_next_compliant()
    if q:
        return {"status": "ok", "question": q}

    created = ensure_enrichment_backlog(min_open=5, include_general_knowledge=True)
    q = pick_next_compliant()
    if not q:
        return {"status": "empty", "question": None, "auto_generated": created}
    return {"status": "ok", "question": q, "auto_generated": created}


@app.post("/enrichment/answer")
def enrichment_answer(payload: EnrichmentAnswerRequest) -> dict:
    q = db.get_enrichment_question(payload.question_id)
    if not q or q["status"] != "open":
        raise HTTPException(status_code=404, detail="Open enrichment question not found")

    db.answer_enrichment_question(payload.question_id, payload.answer)

    materialized = f"Source document: {q['source_doc']}\nClarification question: {q['question']}\nApproved answer: {payload.answer}"
    vec = embedder.embed_text(materialized)
    db.insert_doc_chunk(
        source=payload.source_label,
        chunk_index=0,
        content=materialized,
        embedding=vec,
    )
    return {"status": "answered_and_indexed", "question_id": payload.question_id}


@app.post("/enrichment/escalate")
def enrichment_escalate(payload: EnrichmentEscalateRequest) -> dict:
    q = db.get_enrichment_question(payload.question_id)
    if not q or q["status"] != "open":
        raise HTTPException(status_code=404, detail="Open enrichment question not found")

    ticket_id = db.create_ticket(payload.user_id, q["question"])
    db.escalate_enrichment_question(payload.question_id, payload.reason)
    db.log_interaction(
        user_id=payload.user_id,
        question=q["question"],
        answer=None,
        status="escalated",
        citations=[],
        ticket_id=ticket_id,
        source="enrichment",
    )
    return {
        "status": "escalated_to_ticket",
        "question_id": payload.question_id,
        "ticket_id": ticket_id,
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    source = re.sub(r"[^a-z0-9_-]+", "-", (payload.source or "ask").strip().lower())[:32] or "ask"
    hits = retrieve(
        db=db,
        embedder=embedder,
        question=payload.question,
        top_k=settings.top_k,
        vector_weight=settings.hybrid_vector_weight,
        keyword_min_token_length=settings.keyword_min_token_length,
    )

    if not hits:
        ticket_id = db.create_ticket(payload.user_id, payload.question)
        db.log_interaction(
            user_id=payload.user_id,
            question=payload.question,
            answer=None,
            status="escalated",
            citations=[],
            ticket_id=ticket_id,
            source=source,
        )
        return AskResponse(
            status="escalated",
            ticket_id=ticket_id,
            reason="No documentation indexed.",
        )

    top_score = hits[0].score
    if top_score < settings.min_similarity:
        ticket_id = db.create_ticket(payload.user_id, payload.question)
        db.log_interaction(
            user_id=payload.user_id,
            question=payload.question,
            answer=None,
            status="escalated",
            citations=[],
            ticket_id=ticket_id,
            source=source,
        )
        return AskResponse(
            status="escalated",
            ticket_id=ticket_id,
            reason=f"Insufficient supporting context (top score={top_score:.3f}).",
        )

    context_blocks = []
    citations = []
    sources = []
    for h in hits:
        cid = f"doc:{h.source}#{h.chunk_index}"
        citations.append(cid)
        sources.append(h.source)
        context_blocks.append(f"[{cid}] {h.content}")

    selected_model = payload.model or active_model["name"]
    switched_from_model: str | None = None
    model_used = selected_model
    try:
        answer, model_used, switched_from_model = answer_with_fallback(payload.question, context_blocks, selected_model)
    except Exception as exc:
        ticket_id = db.create_ticket(payload.user_id, payload.question)
        db.log_interaction(
            user_id=payload.user_id,
            question=payload.question,
            answer=None,
            status="escalated",
            citations=citations,
            ticket_id=ticket_id,
            source=source,
        )
        return AskResponse(
            status="escalated",
            ticket_id=ticket_id,
            reason=f"Model request failed: {exc}",
            active_model=active_model["name"],
        )

    if not answer:
        ticket_id = db.create_ticket(payload.user_id, payload.question)
        db.log_interaction(
            user_id=payload.user_id,
            question=payload.question,
            answer=None,
            status="escalated",
            citations=citations,
            ticket_id=ticket_id,
            source=source,
        )
        return AskResponse(
            status="escalated",
            ticket_id=ticket_id,
            reason="Model returned an empty answer.",
            active_model=active_model["name"],
            switched_from_model=switched_from_model,
        )

    if "INSUFFICIENT_CONTEXT" in answer.upper():
        ticket_id = db.create_ticket(payload.user_id, payload.question)
        db.log_interaction(
            user_id=payload.user_id,
            question=payload.question,
            answer=None,
            status="escalated",
            citations=citations,
            ticket_id=ticket_id,
            source=source,
        )
        return AskResponse(
            status="escalated",
            ticket_id=ticket_id,
            reason="Model could not answer from retrieved context.",
            active_model=active_model["name"],
            switched_from_model=switched_from_model,
        )

    preferred_output_lang = normalize_lang(detect_language(payload.question))
    if preferred_output_lang not in language_policy["allowed"]:
        preferred_output_lang = language_policy["primary"]
    normalized_answer = normalize_or_none(answer, model_used, preferred_output_lang)
    if normalized_answer is None:
        ticket_id = db.create_ticket(payload.user_id, payload.question)
        db.log_interaction(
            user_id=payload.user_id,
            question=payload.question,
            answer=None,
            status="escalated",
            citations=citations,
            ticket_id=ticket_id,
            source=source,
        )
        return AskResponse(
            status="escalated",
            ticket_id=ticket_id,
            reason=f"Language policy enforcement failed (required: {preferred_output_lang}).",
            active_model=active_model["name"],
            switched_from_model=switched_from_model,
        )
    answer = normalized_answer

    citation_ok = has_valid_citation(answer, citations)
    confidence = compute_confidence(hits, citation_ok)

    if settings.strict_citation_gate and not citation_ok:
        ticket_id = db.create_ticket(payload.user_id, payload.question)
        db.log_interaction(
            user_id=payload.user_id,
            question=payload.question,
            answer=None,
            status="escalated",
            citations=citations,
            ticket_id=ticket_id,
            source=source,
        )
        return AskResponse(
            status="escalated",
            ticket_id=ticket_id,
            reason="Answer lacked explicit citations.",
            active_model=active_model["name"],
            switched_from_model=switched_from_model,
            confidence=confidence,
        )

    stale = stale_sources(sources)
    if stale and settings.stale_doc_policy == "escalate":
        ticket_id = db.create_ticket(payload.user_id, payload.question)
        db.log_interaction(
            user_id=payload.user_id,
            question=payload.question,
            answer=None,
            status="escalated",
            citations=citations,
            ticket_id=ticket_id,
            source=source,
        )
        return AskResponse(
            status="escalated",
            ticket_id=ticket_id,
            reason=f"Retrieved sources appear outdated (>{settings.stale_doc_days} days): {', '.join(stale[:3])}",
            active_model=active_model["name"],
            switched_from_model=switched_from_model,
            confidence=confidence,
        )

    if confidence < settings.confidence_threshold:
        ticket_id = db.create_ticket(payload.user_id, payload.question)
        db.log_interaction(
            user_id=payload.user_id,
            question=payload.question,
            answer=None,
            status="escalated",
            citations=citations,
            ticket_id=ticket_id,
            source=source,
        )
        return AskResponse(
            status="escalated",
            ticket_id=ticket_id,
            reason=f"Low confidence ({confidence:.2f} < {settings.confidence_threshold:.2f}).",
            active_model=active_model["name"],
            switched_from_model=switched_from_model,
            confidence=confidence,
        )

    if switched_from_model:
        lang = detect_language(payload.question)
        notice = switch_notice(lang, switched_from_model, model_used)
        answer = f"{notice}\n\n{answer}"

    if stale and settings.stale_doc_policy == "warn":
        answer = (
            f"{answer}\n\nWarning: some cited sources may be outdated "
            f"(>{settings.stale_doc_days} days): {', '.join(stale[:3])}"
        )

    if not settings.strict_citation_gate and "[doc:" not in answer and citations:
        answer = f"{answer}\n\nSources: " + ", ".join(f"[{c}]" for c in citations[:3])

    db.log_interaction(
        user_id=payload.user_id,
        question=payload.question,
        answer=answer,
        status="answered",
        citations=citations,
        ticket_id=None,
        source=source,
    )
    return AskResponse(
        status="answered",
        answer=answer,
        citations=citations,
        confidence=confidence,
        active_model=active_model["name"],
        switched_from_model=switched_from_model,
    )


@app.post("/ask/compare")
def ask_compare(payload: CompareRequest) -> dict:
    models = [m.strip() for m in payload.models if m.strip()]
    if len(models) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two models for comparison.")

    available = set(ollama_models())
    missing = [m for m in models if m not in available]
    if missing:
        raise HTTPException(status_code=400, detail=f"These models are not installed: {missing}")

    hits = retrieve(
        db=db,
        embedder=embedder,
        question=payload.question,
        top_k=settings.top_k,
        vector_weight=settings.hybrid_vector_weight,
        keyword_min_token_length=settings.keyword_min_token_length,
    )
    if not hits:
        return {
            "status": "insufficient_context",
            "reason": "No documentation indexed.",
            "results": [],
        }

    top_score = hits[0].score
    if top_score < settings.min_similarity:
        return {
            "status": "insufficient_context",
            "reason": f"Insufficient supporting context (top score={top_score:.3f}).",
            "results": [],
        }

    context_blocks = []
    citations = []
    sources = []
    for h in hits:
        cid = f"doc:{h.source}#{h.chunk_index}"
        citations.append(cid)
        sources.append(h.source)
        context_blocks.append(f"[{cid}] {h.content}")
    stale = stale_sources(sources)

    results = []
    for model_name in models:
        try:
            answer = (llm.answer_with_context(payload.question, context_blocks, model=model_name) or "").strip()
        except Exception as exc:
            results.append(
                {
                    "model": model_name,
                    "status": "error",
                    "reason": str(exc),
                    "answer": None,
                    "citations": citations,
                }
            )
            continue

        if not answer:
            results.append(
                {
                    "model": model_name,
                    "status": "escalated",
                    "reason": "Model returned an empty answer.",
                    "answer": None,
                    "citations": citations,
                }
            )
            continue

        if "INSUFFICIENT_CONTEXT" in answer.upper():
            results.append(
                {
                    "model": model_name,
                    "status": "escalated",
                    "reason": "Model could not answer from retrieved context.",
                    "answer": None,
                    "citations": citations,
                }
            )
            continue

        preferred_output_lang = normalize_lang(detect_language(payload.question))
        if preferred_output_lang not in language_policy["allowed"]:
            preferred_output_lang = language_policy["primary"]
        normalized_answer = normalize_or_none(answer, model_name, preferred_output_lang)
        if normalized_answer is None:
            results.append(
                {
                    "model": model_name,
                    "status": "escalated",
                    "reason": f"Language policy enforcement failed (required: {preferred_output_lang}).",
                    "answer": None,
                    "citations": citations,
                }
            )
            continue
        answer = normalized_answer
        citation_ok = has_valid_citation(answer, citations)
        confidence = compute_confidence(hits, citation_ok)

        if settings.strict_citation_gate and not citation_ok:
            results.append(
                {
                    "model": model_name,
                    "status": "escalated",
                    "reason": "Answer lacked explicit citations.",
                    "answer": None,
                    "citations": citations,
                    "confidence": confidence,
                }
            )
            continue

        if stale and settings.stale_doc_policy == "escalate":
            results.append(
                {
                    "model": model_name,
                    "status": "escalated",
                    "reason": f"Retrieved sources appear outdated (>{settings.stale_doc_days} days): {', '.join(stale[:3])}",
                    "answer": None,
                    "citations": citations,
                    "confidence": confidence,
                }
            )
            continue

        if confidence < settings.confidence_threshold:
            results.append(
                {
                    "model": model_name,
                    "status": "escalated",
                    "reason": f"Low confidence ({confidence:.2f} < {settings.confidence_threshold:.2f}).",
                    "answer": None,
                    "citations": citations,
                    "confidence": confidence,
                }
            )
            continue

        if not settings.strict_citation_gate and "[doc:" not in answer and citations:
            answer = f"{answer}\n\nSources: " + ", ".join(f"[{c}]" for c in citations[:3])
        if stale and settings.stale_doc_policy == "warn":
            answer = (
                f"{answer}\n\nWarning: some cited sources may be outdated "
                f"(>{settings.stale_doc_days} days): {', '.join(stale[:3])}"
            )

        results.append(
            {
                "model": model_name,
                "status": "answered",
                "reason": None,
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
            }
        )

    return {
        "status": "ok",
        "question": payload.question,
        "results": results,
    }


@app.get("/tickets/open")
def tickets_open() -> dict:
    return {"tickets": db.open_tickets()}


@app.post("/editor/respond")
def editor_respond(payload: EditorResponseRequest) -> dict:
    open_ids = {t["id"] for t in db.open_tickets()}
    if payload.ticket_id not in open_ids:
        raise HTTPException(status_code=404, detail="Open ticket not found")

    db.resolve_ticket(payload.ticket_id, payload.editor_answer)

    # Feed approved answer back into retrieval index.
    vec = embedder.embed_text(payload.editor_answer)
    db.insert_doc_chunk(
        source=payload.source_label,
        chunk_index=0,
        content=payload.editor_answer,
        embedding=vec,
    )

    return {"status": "resolved_and_indexed", "ticket_id": payload.ticket_id}


@app.post("/tickets/dismiss")
def tickets_dismiss(payload: TicketDismissRequest) -> dict:
    open_ids = {t["id"] for t in db.open_tickets()}
    if payload.ticket_id not in open_ids:
        raise HTTPException(status_code=404, detail="Open ticket not found")
    db.dismiss_ticket(payload.ticket_id, payload.reason)
    return {"status": "dismissed", "ticket_id": payload.ticket_id}


@app.get("/qa/documents")
def qa_documents(limit: int = 5000) -> dict:
    rows = db.list_qa_documents(limit=limit)
    return {"count": len(rows), "documents": rows}


@app.get("/qa/exports")
def qa_exports() -> dict:
    settings.exports_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(settings.exports_path.glob("qa-export-*"), reverse=True):
        if not p.is_file():
            continue
        rows.append(
            {
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "created_at": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
                "download_url": f"/qa/export/{p.name}",
            }
        )
    return {"count": len(rows), "exports": rows}


@app.post("/qa/export")
def qa_export(payload: QAExportRequest) -> dict:
    rows = db.list_qa_documents(limit=payload.limit)
    settings.exports_path.mkdir(parents=True, exist_ok=True)
    fmt = payload.format.lower().strip()
    filename = export_name(fmt)
    out_path = settings.exports_path / filename

    if fmt == "jsonl":
        write_qa_jsonl(rows, out_path)
    elif fmt == "csv":
        write_qa_csv(rows, out_path)
    elif fmt == "md":
        write_qa_md(rows, out_path)
    elif fmt == "pdf":
        write_qa_pdf(rows, out_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")

    return {
        "status": "ok",
        "format": fmt,
        "count": len(rows),
        "path": str(out_path),
        "download_url": f"/qa/export/{filename}",
    }


@app.get("/qa/export/{filename}")
def qa_export_download(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.startswith("qa-export-"):
        raise HTTPException(status_code=400, detail="Invalid export filename")
    target = (settings.exports_path / safe_name).resolve()
    base = settings.exports_path.resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Invalid export path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Export file not found")
    return FileResponse(target)
