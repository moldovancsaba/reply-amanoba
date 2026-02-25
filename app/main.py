from pathlib import Path
import json
import re

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests

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
    ModelSelectRequest,
    TicketDismissRequest,
)

app = FastAPI(title="Internal Help Bot")

db = Database(settings.db_path)
embedder = Embedder(settings.embed_model)
llm = OllamaLLM(settings.ollama_url, settings.ollama_model)
web_dir = Path(__file__).resolve().parent / "web"
web_path = web_dir / "index.html"
allowed_upload_exts = {".pdf", ".md", ".txt"}
active_model = {"name": settings.ollama_model}
tunnel_log_path = Path("/tmp/helpbot-tunnel.log")

app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


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


def detect_language(text: str) -> str:
    t = f" {text.lower()} "
    markers = {
        "hu": [" mi ", " vagy ", " és ", " hogy ", " az ", " egy ", "kérdés", "válasz"],
        "ro": [" și ", " este ", " sunt ", " ce ", " pentru ", " întrebare", "răspuns"],
        "es": [" el ", " la ", " de ", " que ", " para ", " pregunta", "respuesta"],
        "fr": [" le ", " la ", " de ", " et ", " pour ", " question", "réponse"],
        "de": [" der ", " die ", " und ", " für ", " ist ", " frage", "antwort"],
        "it": [" il ", " la ", " e ", " per ", " che ", " domanda", "risposta"],
        "pt": [" o ", " a ", " e ", " para ", " que ", " pergunta", "resposta"],
    }
    for lang, words in markers.items():
        if any(w in t for w in words):
            return lang
    return "en"


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
- Use the same language as the document.
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


@app.post("/models/select")
def select_model(payload: ModelSelectRequest) -> dict:
    available = ollama_models()
    if payload.model not in available:
        raise HTTPException(status_code=400, detail=f"Model not installed. Available: {available}")
    active_model["name"] = payload.model
    return {"status": "ok", "active_model": active_model["name"]}


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
    open_questions = db.open_enrichment_questions()
    if not open_questions:
        created = ensure_enrichment_backlog(min_open=5, include_general_knowledge=True)
        open_questions = db.open_enrichment_questions()
        if not open_questions:
            return {"status": "empty", "question": None, "auto_generated": created}
        return {"status": "ok", "question": open_questions[0], "auto_generated": created}
    return {"status": "ok", "question": open_questions[0]}


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
    )
    return {
        "status": "escalated_to_ticket",
        "question_id": payload.question_id,
        "ticket_id": ticket_id,
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    hits = retrieve(db=db, embedder=embedder, question=payload.question, top_k=settings.top_k)

    if not hits:
        ticket_id = db.create_ticket(payload.user_id, payload.question)
        db.log_interaction(
            user_id=payload.user_id,
            question=payload.question,
            answer=None,
            status="escalated",
            citations=[],
            ticket_id=ticket_id,
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
        )
        return AskResponse(
            status="escalated",
            ticket_id=ticket_id,
            reason=f"Insufficient supporting context (top score={top_score:.3f}).",
        )

    context_blocks = []
    citations = []
    for h in hits:
        cid = f"doc:{h.source}#{h.chunk_index}"
        citations.append(cid)
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
        )
        return AskResponse(
            status="escalated",
            ticket_id=ticket_id,
            reason="Model could not answer from retrieved context.",
            active_model=active_model["name"],
            switched_from_model=switched_from_model,
        )

    if switched_from_model:
        lang = detect_language(payload.question)
        notice = switch_notice(lang, switched_from_model, model_used)
        answer = f"{notice}\n\n{answer}"

    if "[doc:" not in answer and citations:
        answer = f"{answer}\n\nSources: " + ", ".join(f"[{c}]" for c in citations[:3])

    db.log_interaction(
        user_id=payload.user_id,
        question=payload.question,
        answer=answer,
        status="answered",
        citations=citations,
        ticket_id=None,
    )
    return AskResponse(
        status="answered",
        answer=answer,
        citations=citations,
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

    hits = retrieve(db=db, embedder=embedder, question=payload.question, top_k=settings.top_k)
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
    for h in hits:
        cid = f"doc:{h.source}#{h.chunk_index}"
        citations.append(cid)
        context_blocks.append(f"[{cid}] {h.content}")

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

        if "[doc:" not in answer and citations:
            answer = f"{answer}\n\nSources: " + ", ".join(f"[{c}]" for c in citations[:3])

        results.append(
            {
                "model": model_name,
                "status": "answered",
                "reason": None,
                "answer": answer,
                "citations": citations,
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
