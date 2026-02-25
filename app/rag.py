from dataclasses import dataclass
from pathlib import Path
import re
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


def retrieve(db: Database, embedder: Embedder, question: str, top_k: int) -> list[RetrievedChunk]:
    q = np.array(embedder.embed_text(question), dtype=np.float32)
    rows = db.list_doc_chunks()
    if not rows:
        return []

    scored: list[RetrievedChunk] = []
    for row in rows:
        emb = np.array(eval_embedding(row["embedding"]), dtype=np.float32)
        score = float(np.dot(q, emb))
        scored.append(
            RetrievedChunk(
                doc_id=row["id"],
                source=row["source"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                score=score,
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def eval_embedding(raw: str) -> list[float]:
    # Stored as JSON, parse safely and keep numeric values only.
    import json

    data = json.loads(raw)
    return [float(x) for x in data]
