import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid


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

            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                actor TEXT NOT NULL,
                role TEXT NOT NULL,
                action TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                details TEXT NOT NULL
            );

            CREATE TRIGGER IF NOT EXISTS audit_events_no_update
            BEFORE UPDATE ON audit_events
            BEGIN
                SELECT RAISE(ABORT, 'audit_events is immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
            BEFORE DELETE ON audit_events
            BEGIN
                SELECT RAISE(ABORT, 'audit_events is immutable');
            END;

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_activity_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                citations TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS qa_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                status TEXT NOT NULL,
                citations TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL
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
        source: str = "ask",
    ) -> None:
        now = self._now()
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
                now,
            ),
        )
        self.conn.execute(
            """
            INSERT INTO qa_documents(user_id, question, answer, status, citations, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                question,
                answer,
                status,
                json.dumps(citations or []),
                source,
                now,
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

    def update_enrichment_question_text(self, question_id: int, question: str) -> None:
        self.conn.execute(
            """
            UPDATE enrichment_questions
            SET question=?
            WHERE id=?
            """,
            (question, question_id),
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

    def get_setting(self, key: str) -> str | None:
        cur = self.conn.execute("SELECT value FROM app_settings WHERE key=? LIMIT 1", (key,))
        row = cur.fetchone()
        return str(row["value"]) if row else None

    def set_setting(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO app_settings(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, value, self._now()),
        )
        self.conn.commit()

    def list_interactions(self, limit: int = 2000, offset: int = 0) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT id, user_id, question, answer, status, citations, ticket_id, created_at
            FROM interactions
            ORDER BY id ASC
            LIMIT ? OFFSET ?
            """,
            (int(limit), int(offset)),
        )
        rows = [dict(r) for r in cur.fetchall()]
        for row in rows:
            try:
                row["citations"] = json.loads(row.get("citations") or "[]")
            except Exception:
                row["citations"] = []
        return rows

    def create_chat_session(self, user_id: str, source: str, metadata: dict[str, Any] | None = None) -> str:
        session_key = str(uuid.uuid4())
        now = self._now()
        self.conn.execute(
            """
            INSERT INTO chat_sessions(session_key, user_id, source, metadata, created_at, last_activity_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_key, user_id, source, json.dumps(metadata or {}), now, now),
        )
        self.conn.commit()
        return session_key

    def get_chat_session(self, session_key: str) -> dict[str, Any] | None:
        cur = self.conn.execute(
            """
            SELECT session_key, user_id, source, metadata, created_at, last_activity_at
            FROM chat_sessions
            WHERE session_key=?
            LIMIT 1
            """,
            (session_key,),
        )
        row = cur.fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["metadata"] = json.loads(data.get("metadata") or "{}")
        except Exception:
            data["metadata"] = {}
        return data

    def append_chat_message(
        self,
        session_key: str,
        role: str,
        content: str,
        status: str = "ok",
        citations: list[str] | None = None,
    ) -> None:
        now = self._now()
        self.conn.execute(
            """
            INSERT INTO chat_messages(session_key, role, content, status, citations, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_key, role, content, status, json.dumps(citations or []), now),
        )
        self.conn.execute(
            "UPDATE chat_sessions SET last_activity_at=? WHERE session_key=?",
            (now, session_key),
        )
        self.conn.commit()

    def chat_history(self, session_key: str, limit: int = 200) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT id, session_key, role, content, status, citations, created_at
            FROM chat_messages
            WHERE session_key=?
            ORDER BY id ASC
            LIMIT ?
            """,
            (session_key, int(limit)),
        )
        rows = [dict(r) for r in cur.fetchall()]
        for row in rows:
            try:
                row["citations"] = json.loads(row.get("citations") or "[]")
            except Exception:
                row["citations"] = []
        return rows

    def create_qa_document(
        self,
        user_id: str,
        question: str,
        answer: str | None,
        status: str,
        citations: list[str] | None,
        source: str,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO qa_documents(user_id, question, answer, status, citations, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, question, answer, status, json.dumps(citations or []), source, self._now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_qa_documents(self, limit: int = 5000) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT id, user_id, question, answer, status, citations, source, created_at
            FROM qa_documents
            ORDER BY id ASC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = [dict(r) for r in cur.fetchall()]
        for row in rows:
            try:
                row["citations"] = json.loads(row.get("citations") or "[]")
            except Exception:
                row["citations"] = []
        return rows

    def log_audit(
        self,
        actor: str,
        role: str,
        action: str,
        target_type: str,
        target_id: str,
        details: dict[str, Any] | None = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO audit_events(occurred_at, actor, role, action, target_type, target_id, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._now(),
                actor or "unknown",
                role or "unknown",
                action,
                target_type,
                target_id,
                json.dumps(details or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recent_audit_events(self, limit: int = 200) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT id, occurred_at, actor, role, action, target_type, target_id, details
            FROM audit_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = [dict(r) for r in cur.fetchall()]
        for row in rows:
            try:
                row["details"] = json.loads(row.get("details") or "{}")
            except Exception:
                row["details"] = {}
        return rows
