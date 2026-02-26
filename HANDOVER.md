# Handover - `reply-amanoba`

## Repository
- Path: `/Users/moldovancsaba/Projects/reply-amanoba`
- Branch: `main`

## What Is Implemented Now
1. Multi-instance lifecycle CLI (`replyctl`):
- install/start/stop/status/uninstall
- backup/restore
- update/update-all with migration + rollback path
- dependency operations (`doctor`, `deps-install`, `deps-verify`, `deps-upgrade`)
- manifest schema: `deploy/company.yaml.example`

2. Webchat runtime API:
- `POST /chat/session`
- `POST /chat/message`
- `POST /chat/stream` (SSE)
- `POST /chat/history`
- `GET /chat/config`
- `GET /admin/webchat/snippet`
- token auth + CORS + per-IP rate limit

3. Q/A document archival + export:
- `log_interaction` now also writes to `qa_documents`
- export endpoints:
  - `GET /qa/documents`
  - `GET /qa/exports`
  - `POST /qa/export` (`jsonl|csv|md|pdf`)
  - `GET /qa/export/{filename}`
- exports folder: `EXPORTS_PATH` (`./data/exports` default)

4. UI updates (Admin):
- Q/A export controls + export list
- webchat snippet viewer/copy
- no full-page reload for these actions

5. Embeddable widget:
- `app/web/webchat.js`
- injected via snippet and uses chat API

## Updated Config/Dependencies
- `.env.example` extended with docs/export/chat/language settings
- `requirements.txt` added:
  - `PyYAML==6.0.3`
  - `reportlab==4.2.5`
- `VERSION` file added (`0.1.0`)
- `.gitignore` now ignores `data/exports/`

## Validation Performed
- Python compile checks passed:
  - `app/main.py`, `app/db.py`, `app/schemas.py`, `app/config.py`, `app/rag.py`, `replyctl/*.py`
- Shell syntax checks passed:
  - `start_helpbot.command`
  - `start_helpbot_public.command`
  - `tools/scripts/replyctl`
- Runtime sanity (via direct function calls) passed for:
  - chat session create/history
  - Q/A export generation
  - chat message path

## Main Files Changed
- `app/main.py`
- `app/db.py`
- `app/schemas.py`
- `app/web/index.html`
- `app/web/webchat.js` (new)
- `replyctl/cli.py`
- `tools/scripts/replyctl`
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `README.md`
- `HANDOVER.md`
- `VERSION` (new)

## Known Next Steps
1. Add server-side RBAC (currently token-based chat auth only).
2. Add immutable audit trail for admin/editor critical actions.
3. Add CI tests for chat/export/replyctl lifecycle flows.
4. Extend dependency adapters beyond local Homebrew paths.
5. Add import/export contract tests for `qa-export-*` files.
