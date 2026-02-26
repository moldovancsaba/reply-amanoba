from dataclasses import dataclass
from pathlib import Path
import re
import math
from collections import Counter
import numpy as np
from pypdf import PdfReader

from app.db import Database
from app.embeddings import Embedder


@dataclass
class RetrievedChunk:
    doc_id: int
    source: str
    chunk_index: int
    content: str
    score: float
    vector_score: float
    keyword_score: float


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []

    chunks = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def load_docs_from_folder(path: Path) -> list[tuple[str, str]]:
    docs = []
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue

        ext = p.suffix.lower()
        if ext in {".txt", ".md"}:
            docs.append((str(p.name), p.read_text(encoding="utf-8", errors="ignore")))
            continue

        if ext == ".pdf":
            text = read_pdf_text(p)
            if text.strip():
                docs.append((str(p.name), text))
    return docs


def read_pdf_text(path: Path) -> str:
    pages: list[str] = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def reindex_docs(db: Database, embedder: Embedder, docs_path: Path) -> int:
    docs = load_docs_from_folder(docs_path)
    db.clear_docs()

    count = 0
    for source, text in docs:
        chunks = chunk_text(text)
        if not chunks:
            continue
        vecs = embedder.embed_many(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
            db.insert_doc_chunk(source=source, chunk_index=i, content=chunk, embedding=vec)
            count += 1

    return count


def tokenize(text: str, min_len: int = 2) -> list[str]:
    parts = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [p for p in parts if len(p) >= min_len]


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        if hi <= 0:
            return [0.0 for _ in values]
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def retrieve(
    db: Database,
    embedder: Embedder,
    question: str,
    top_k: int,
    vector_weight: float = 0.7,
    keyword_min_token_length: int = 2,
) -> list[RetrievedChunk]:
    vector_weight = max(0.0, min(1.0, float(vector_weight)))
    q = np.array(embedder.embed_text(question), dtype=np.float32)
    q_tokens = tokenize(question, min_len=keyword_min_token_length)
    rows = db.list_doc_chunks()
    if not rows:
        return []

    n_docs = len(rows)
    chunk_tf: list[Counter[str]] = []
    df: Counter[str] = Counter()
    vector_raw: list[float] = []
    keyword_raw: list[float] = []

    # Build term statistics once for keyword relevance.
    for row in rows:
        tf = Counter(tokenize(row["content"], min_len=keyword_min_token_length))
        chunk_tf.append(tf)
        for term in tf.keys():
            df[term] += 1

    for i, row in enumerate(rows):
        emb = np.array(eval_embedding(row["embedding"]), dtype=np.float32)
        vec_score = float(np.dot(q, emb))
        vector_raw.append(vec_score)

        tf = chunk_tf[i]
        kw_score = 0.0
        for term in q_tokens:
            if term not in tf:
                continue
            # Smooth IDF with +1 so frequent terms still contribute minimally.
            idf = math.log((1.0 + n_docs) / (1.0 + df.get(term, 0))) + 1.0
            kw_score += float(tf[term]) * idf
        keyword_raw.append(kw_score)

    vector_norm = normalize_scores(vector_raw)
    keyword_norm = normalize_scores(keyword_raw)

    scored: list[RetrievedChunk] = []
    for i, row in enumerate(rows):
        combined = (vector_weight * vector_norm[i]) + ((1.0 - vector_weight) * keyword_norm[i])
        scored.append(
            RetrievedChunk(
                doc_id=row["id"],
                source=row["source"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                score=float(combined),
                vector_score=float(vector_norm[i]),
                keyword_score=float(keyword_norm[i]),
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def eval_embedding(raw: str) -> list[float]:
    # Stored as JSON, parse safely and keep numeric values only.
    import json

    data = json.loads(raw)
    return [float(x) for x in data]
