import importlib
import sqlite3
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest


def _purge_app_modules() -> None:
    for name in list(sys.modules.keys()):
        if name == "app" or name.startswith("app."):
            sys.modules.pop(name, None)


def _load_app(monkeypatch, tmp_path: Path, auth_enabled: bool = True):
    docs = tmp_path / "docs"
    exports = tmp_path / "exports"
    docs.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setenv("DOCS_PATH", str(docs))
    monkeypatch.setenv("EXPORTS_PATH", str(exports))
    monkeypatch.setenv("LANGUAGE_PRIMARY", "en")
    monkeypatch.setenv("LANGUAGE_ALLOWED", "en")
    monkeypatch.setenv("EMBEDDER_FAKE", "1")
    monkeypatch.setenv("LLM_FAKE", "1")
    monkeypatch.setenv("STRICT_CITATION_GATE", "0")
    monkeypatch.setenv("CHAT_ENABLED", "1")
    monkeypatch.setenv("CHAT_API_TOKEN", "chat-token")
    monkeypatch.setenv("CHAT_ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("AUTH_ENABLED", "1" if auth_enabled else "0")
    monkeypatch.setenv("ADMIN_API_TOKEN", "admin-token")
    monkeypatch.setenv("EDITOR_API_TOKEN", "editor-token")
    monkeypatch.setenv("EMPLOYEE_API_TOKEN", "employee-token")

    _purge_app_modules()
    return importlib.import_module("app.main")


def test_rbac_and_immutable_audit(monkeypatch, tmp_path):
    main = _load_app(monkeypatch, tmp_path, auth_enabled=True)
    client = TestClient(main.app)

    # No token => blocked.
    r = client.get("/admin/language-policy")
    assert r.status_code == 401

    # Wrong role token => blocked.
    r = client.get("/admin/language-policy", headers={"Authorization": "Bearer employee-token"})
    assert r.status_code == 403

    admin_headers = {"Authorization": "Bearer admin-token", "X-Actor": "admin-ci"}
    r = client.post(
        "/admin/language-policy",
        json={"primary_language": "en", "allowed_languages": ["en"]},
        headers=admin_headers,
    )
    assert r.status_code == 200, r.text

    audit = client.get("/admin/audit?limit=20", headers=admin_headers)
    assert audit.status_code == 200, audit.text
    events = audit.json()["events"]
    assert any(e["action"] == "set_language_policy" for e in events)

    # DB-level immutability.
    with pytest.raises(sqlite3.DatabaseError):
        main.db.conn.execute("UPDATE audit_events SET actor='tamper' WHERE id=1")
        main.db.conn.commit()


def test_chat_and_export_flow_with_tokens(monkeypatch, tmp_path):
    main = _load_app(monkeypatch, tmp_path, auth_enabled=True)
    client = TestClient(main.app)

    chat_headers = {"Authorization": "Bearer chat-token"}
    admin_headers = {"Authorization": "Bearer admin-token", "X-Actor": "admin-ci"}

    session = client.post(
        "/chat/session",
        json={"user_id": "u1", "source": "webchat", "metadata": {}},
        headers=chat_headers,
    )
    assert session.status_code == 200, session.text
    session_id = session.json()["session_id"]

    msg = client.post(
        "/chat/message",
        json={"session_id": session_id, "message": "What is the process?"},
        headers=chat_headers,
    )
    assert msg.status_code == 200, msg.text
    assert msg.json()["status"] == "ok"

    export = client.post(
        "/qa/export",
        json={"format": "jsonl", "limit": 100},
        headers=admin_headers,
    )
    assert export.status_code == 200, export.text
    download_url = export.json()["download_url"]

    dl = client.get(download_url, headers=admin_headers)
    assert dl.status_code == 200
    assert dl.text.strip() != ""

    audit = client.get("/admin/audit?limit=100", headers=admin_headers)
    assert audit.status_code == 200
    actions = [e["action"] for e in audit.json()["events"]]
    assert "chat_create_session" in actions
    assert "chat_message" in actions
    assert "export_qa_documents" in actions
