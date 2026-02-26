# {reply} - Local-First Corporate Help Bot

`{reply}` is an offline-first, citation-gated help bot for internal company knowledge.

It supports:
- employee Q/A from uploaded docs (`.pdf`, `.md`, `.txt`)
- strict escalation when confidence/citations are not enough
- editor-in-the-loop ticket resolution and annotation
- admin dashboard (model health, indexing, operations)
- embeddable webchat API/module for corporate intranets
- export of all Q/A records as `jsonl`, `csv`, `md`, `pdf`
- multi-instance lifecycle with `replyctl` (install/start/stop/update/backup)

## Current Version
- `VERSION`: `0.1.0`

## Quick Start (Current Project)
```bash
cd /Users/moldovancsaba/Projects/reply-amanoba
./start_helpbot.command
```

## Manual Run
```bash
cd /Users/moldovancsaba/Projects/reply-amanoba
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Multi-Company Install/Update (`replyctl`)
Main script:
- `tools/scripts/replyctl`

Example manifest:
- `deploy/company.yaml.example`

Install dependencies + instance:
```bash
cd /Users/moldovancsaba/Projects/reply-amanoba
tools/scripts/replyctl install --manifest deploy/company.yaml.example --with-deps
```

Start instance:
```bash
tools/scripts/replyctl start example-corp
```

Start with public tunnel (if enabled in manifest):
```bash
tools/scripts/replyctl start example-corp --public
```

Status:
```bash
tools/scripts/replyctl status example-corp
```

Backup:
```bash
tools/scripts/replyctl backup example-corp
```

Update one instance:
```bash
tools/scripts/replyctl update example-corp
```

Update all instances:
```bash
tools/scripts/replyctl update --all
```

Stop:
```bash
tools/scripts/replyctl stop example-corp
```

Uninstall:
```bash
tools/scripts/replyctl uninstall example-corp --yes
```

Dependency checks:
```bash
tools/scripts/replyctl doctor
tools/scripts/replyctl deps-verify
tools/scripts/replyctl deps-install
tools/scripts/replyctl deps-upgrade
```

## Webchat API (Embed on Internal Website)
Endpoints:
- `POST /chat/session`
- `POST /chat/message`
- `POST /chat/stream`
- `POST /chat/history`
- `GET /chat/config`
- `GET /admin/webchat/snippet`

Auth:
- If `CHAT_API_TOKEN` is set, send:
  - `Authorization: Bearer <token>` or `X-API-Token: <token>`

CORS:
- Controlled by `CHAT_ALLOWED_ORIGINS`.

Rate limit:
- Controlled by `CHAT_RATE_LIMIT_PER_MIN`.

### Minimal Embed
Use Admin tool output (`/admin/webchat/snippet`) or this template:
```html
<script>
  window.REPLY_WEBCHAT_CONFIG = {
    baseUrl: "https://YOUR-URL",
    token: "YOUR_CHAT_TOKEN",
    userId: "employee-1"
  };
  const s = document.createElement("script");
  s.src = window.REPLY_WEBCHAT_CONFIG.baseUrl + "/static/webchat.js";
  document.head.appendChild(s);
</script>
```

## Q/A Archive and Export
All `ask`/`webchat`/`enrichment escalation` interactions are stored in `qa_documents`.

Endpoints:
- `GET /qa/documents`
- `GET /qa/exports`
- `POST /qa/export`
- `GET /qa/export/{filename}`

Formats:
- `jsonl`
- `csv`
- `md`
- `pdf` (requires `reportlab`, already in `requirements.txt`)

Storage:
- `EXPORTS_PATH` (default: `./data/exports`)

## Environment Variables
See `.env.example` for full list.

Key runtime vars:
- `OLLAMA_URL`, `OLLAMA_MODEL`, `EMBED_MODEL`
- `DOCS_PATH`, `DB_PATH`
- `EXPORTS_PATH`
- `STRICT_CITATION_GATE`, `CONFIDENCE_THRESHOLD`
- `LANGUAGE_PRIMARY`, `LANGUAGE_ALLOWED`
- `CHAT_ENABLED`, `CHAT_API_TOKEN`, `CHAT_ALLOWED_ORIGINS`, `CHAT_RATE_LIMIT_PER_MIN`

## Core API Surface
- `GET /health`
- `POST /upload`
- `POST /ingest/reindex`
- `POST /ask`
- `POST /ask/compare`
- `GET /tickets/open`
- `POST /editor/respond`
- `POST /tickets/dismiss`
- `POST /enrichment/generate`
- `GET /enrichment/open`
- `GET /enrichment/improve-me`
- `POST /enrichment/answer`
- `POST /enrichment/escalate`
- `GET /dashboard/health`
- `GET /models`
- `POST /models/select`
- `GET /admin/public-url`
- `GET /admin/language-policy`
- `POST /admin/language-policy`
- `GET /admin/webchat/snippet`
- `POST /chat/session`
- `POST /chat/message`
- `POST /chat/stream`
- `POST /chat/history`
- `GET /chat/config`
- `GET /qa/documents`
- `GET /qa/exports`
- `POST /qa/export`
- `GET /qa/export/{filename}`

## Notes
- Offline-first and open-source runtime (`Ollama` + local embeddings).
- Escalate over hallucinate (confidence + citation gates).
- Editor answers are re-annotated into retrieval index.
