# {reply} - Internal Help Bot

Local-first, grounded enterprise help bot with editor-in-the-loop escalation.

## What This System Does
- Answers employee questions using only annotated internal documentation.
- Refuses low-confidence / unsupported answers and escalates to editor tickets.
- Lets editors resolve tickets and enrich knowledge through clarifications.
- Re-indexes approved editor answers so future answers improve.
- Runs locally with open-source models via Ollama.

## Core Principles
- Grounded answering from retrieved context only.
- Citation-first responses where possible.
- Escalate instead of hallucinate.
- Human-approved knowledge continuously feeds back into RAG.

## Current Stack
- Backend: FastAPI
- Storage: SQLite (`helpbot.sqlite3`)
- LLM runtime: Ollama
- Default LLM: `qwen2.5:3b`
- Embeddings: `intfloat/multilingual-e5-small`
- Frontend: server-rendered static HTML + global CSS (`/static/styles.css`)

## Main Capabilities Implemented
- Drag-and-drop and file upload (`.pdf`, `.md`, `.txt`)
- Reindex pipeline from uploaded docs
- Strict RAG answer path with escalation
- Open ticket queue
- Editor answer + annotation flow
- Ticket dismiss/delete flow
- Clarification question generation (`Improve Me`)
- Clarification answer annotation back into KB
- Persona-separated UI:
  - Employee: ask only
  - Editor: upload + improve + ticket handling
  - Admin: models + dashboard + compare + public URL
- Model management:
  - list installed local models
  - select active model
  - compare models on same question
  - automatic fallback model switch on failures
- Dashboard:
  - annotation status
  - doc/chunk counters
  - ticket/enrichment/answer-rate metrics
  - Ollama reachability + active model
- Public tunnel URL surfaced in Admin panel

## API Endpoints (Current)
- `GET /health`
- `GET /`
- `POST /ingest/reindex`
- `GET /dashboard/health`
- `GET /models`
- `POST /models/select`
- `GET /admin/public-url`
- `POST /upload`
- `POST /ask`
- `POST /ask/compare`
- `GET /tickets/open`
- `POST /editor/respond`
- `POST /tickets/dismiss`
- `POST /enrichment/generate`
- `GET /enrichment/open`
- `GET /enrichment/improve-me`
- `POST /enrichment/answer`

## Data Model (SQLite)
- `docs`: indexed chunks + embeddings
- `tickets`: open/resolved/dismissed escalations + editor answer
- `interactions`: answered/escalated history
- `enrichment_questions`: clarification queue + approved answers

## Run (Non-Developer)
Use launcher:
- `/Users/moldovancsaba/Projects/reply-amanoba/start_helpbot.command`

What it does:
- creates/uses `.venv`
- installs requirements
- ensures model availability
- starts local server
- starts Cloudflare tunnel
- opens local + public URLs

## Manual Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5:3b
ollama serve
uvicorn app.main:app --reload
```

## Frontend Notes
- UI title changed to `{reply}`.
- Styles are centralized in `app/web/styles.css`.
- Static assets served from `/static`.
- Persona view is client-side role filtering.

## Current Known Issues / Risks
- In `app/main.py`:
  - duplicated `timeout=120` in one `requests.post(...)` call
  - duplicated `created += added` in enrichment generation path
- Authentication/authorization is not implemented (MVP only).
- Public tunnel is convenient for testing but not production-safe by default.

## Security / Production Gaps (Not Yet Implemented)
- Authn/Authz (role-enforced server-side, not only UI)
- Audit logs and immutable action trail
- Rate limiting and abuse protection
- Data retention policy and PII controls
- Background job queue for heavy indexing/enrichment

## Repo Layout
- `app/main.py` - API, orchestration, routing
- `app/db.py` - SQLite access layer
- `app/rag.py` - retrieval/indexing pipeline
- `app/llm.py` - Ollama integration
- `app/embeddings.py` - embedding model wrapper
- `app/schemas.py` - request/response models
- `app/web/index.html` - UI markup + behavior
- `app/web/styles.css` - global design system
- `start_helpbot.command` - unified local+public launcher
- `start_helpbot_public.command` - public launcher helper

## Operational Guidance
- Reindex after embedding-model change.
- Keep docs in `data/docs` or upload through UI.
- Prefer editor resolution over forcing low-confidence answers.
- Stop public tunnel when not needed.
