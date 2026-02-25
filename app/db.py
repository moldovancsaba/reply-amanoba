import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                editor_answer TEXT
            );

            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                status TEXT NOT NULL,
                citations TEXT,
                ticket_id INTEGER,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS enrichment_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_doc TEXT NOT NULL,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                answer TEXT,
                created_at TEXT NOT NULL,
                answered_at TEXT
            );
            """
        )
        self.conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def clear_docs(self) -> None:
        self.conn.execute("DELETE FROM docs")
        self.conn.commit()

    def insert_doc_chunk(self, source: str, chunk_index: int, content: str, embedding: list[float]) -> None:
        self.conn.execute(
            "INSERT INTO docs(source, chunk_index, content, embedding, created_at) VALUES(?, ?, ?, ?, ?)",
            (source, chunk_index, content, json.dumps(embedding), self._now()),
        )
        self.conn.commit()

    def list_doc_chunks(self) -> list[dict[str, Any]]:
        cur = self.conn.execute("SELECT id, source, chunk_index, content, embedding FROM docs")
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def docs_summary(self) -> dict[str, Any]:
        cur = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_chunks,
                COUNT(DISTINCT source) AS total_sources,
                MAX(created_at) AS last_indexed_at
            FROM docs
            """
        )
        row = cur.fetchone()
        return dict(row) if row else {"total_chunks": 0, "total_sources": 0, "last_indexed_at": None}

    def doc_chunks_by_source(self) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT source, COUNT(*) AS chunk_count, MAX(created_at) AS last_indexed_at
            FROM docs
            GROUP BY source
            ORDER BY source ASC
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def create_ticket(self, user_id: str, question: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO tickets(user_id, question, status, created_at) VALUES (?, ?, 'open', ?)",
            (user_id, question, self._now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def open_tickets(self) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT id, user_id, question, status, created_at FROM tickets WHERE status='open' ORDER BY id ASC"
        )
        return [dict(r) for r in cur.fetchall()]

    def tickets_summary(self) -> dict[str, int]:
        cur = self.conn.execute(
            """
            SELECT
                SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS open_count,
                SUM(CASE WHEN status='resolved' THEN 1 ELSE 0 END) AS resolved_count,
                SUM(CASE WHEN status='dismissed' THEN 1 ELSE 0 END) AS dismissed_count,
                COUNT(*) AS total_count
            FROM tickets
            """
        )
        row = cur.fetchone()
        if not row:
            return {"open_count": 0, "resolved_count": 0, "dismissed_count": 0, "total_count": 0}
        return {
            "open_count": int(row["open_count"] or 0),
            "resolved_count": int(row["resolved_count"] or 0),
            "dismissed_count": int(row["dismissed_count"] or 0),
            "total_count": int(row["total_count"] or 0),
        }

    def resolve_ticket(self, ticket_id: int, editor_answer: str) -> None:
        self.conn.execute(
            "UPDATE tickets SET status='resolved', editor_answer=?, resolved_at=? WHERE id=?",
            (editor_answer, self._now(), ticket_id),
        )
        self.conn.commit()

    def dismiss_ticket(self, ticket_id: int, reason: str) -> None:
        self.conn.execute(
            "UPDATE tickets SET status='dismissed', editor_answer=?, resolved_at=? WHERE id=?",
            (reason, self._now(), ticket_id),
        )
        self.conn.commit()

    def log_interaction(
        self,
        user_id: str,
        question: str,
        answer: str | None,
        status: str,
        citations: list[str] | None = None,
        ticket_id: int | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO interactions(user_id, question, answer, status, citations, ticket_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                question,
                answer,
                status,
                json.dumps(citations or []),
                ticket_id,
                self._now(),
            ),
        )
        self.conn.commit()

    def interactions_summary(self) -> dict[str, int]:
        cur = self.conn.execute(
            """
            SELECT
                SUM(CASE WHEN status='answered' THEN 1 ELSE 0 END) AS answered_count,
                SUM(CASE WHEN status='escalated' THEN 1 ELSE 0 END) AS escalated_count,
                COUNT(*) AS total_count
            FROM interactions
            """
        )
        row = cur.fetchone()
        if not row:
            return {"answered_count": 0, "escalated_count": 0, "total_count": 0}
        return {
            "answered_count": int(row["answered_count"] or 0),
            "escalated_count": int(row["escalated_count"] or 0),
            "total_count": int(row["total_count"] or 0),
        }

    def create_enrichment_question(self, source_doc: str, question: str) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO enrichment_questions(source_doc, question, status, created_at)
            VALUES (?, ?, 'open', ?)
            """,
            (source_doc, question, self._now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def open_enrichment_questions(self) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT id, source_doc, question, status, created_at
            FROM enrichment_questions
            WHERE status='open'
            ORDER BY id ASC
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def answer_enrichment_question(self, question_id: int, answer: str) -> None:
        self.conn.execute(
            """
            UPDATE enrichment_questions
            SET status='answered', answer=?, answered_at=?
            WHERE id=? AND status='open'
            """,
            (answer, self._now(), question_id),
        )
        self.conn.commit()

    def escalate_enrichment_question(self, question_id: int, reason: str) -> None:
        self.conn.execute(
            """
            UPDATE enrichment_questions
            SET status='escalated', answer=?, answered_at=?
            WHERE id=? AND status='open'
            """,
            (reason, self._now(), question_id),
        )
        self.conn.commit()

    def get_enrichment_question(self, question_id: int) -> dict[str, Any] | None:
        cur = self.conn.execute(
            """
            SELECT id, source_doc, question, status, answer, created_at, answered_at
            FROM enrichment_questions
            WHERE id=?
            """,
            (question_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def enrichment_summary(self) -> dict[str, int]:
        cur = self.conn.execute(
            """
            SELECT
                SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS open_count,
                SUM(CASE WHEN status='answered' THEN 1 ELSE 0 END) AS answered_count,
                COUNT(*) AS total_count
            FROM enrichment_questions
            """
        )
        row = cur.fetchone()
        if not row:
            return {"open_count": 0, "answered_count": 0, "total_count": 0}
        return {
            "open_count": int(row["open_count"] or 0),
            "answered_count": int(row["answered_count"] or 0),
            "total_count": int(row["total_count"] or 0),
        }

    def find_open_enrichment_by_question(self, question: str) -> dict[str, Any] | None:
        cur = self.conn.execute(
            """
            SELECT id, source_doc, question, status, created_at
            FROM enrichment_questions
            WHERE status='open' AND LOWER(TRIM(question)) = LOWER(TRIM(?))
            LIMIT 1
            """,
            (question,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
