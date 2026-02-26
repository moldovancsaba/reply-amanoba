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

3. Server-side RBAC + immutable audit:
- role-token enforcement for protected backend endpoints (`employee`/`editor`/`admin`)
- append-only `audit_events` table with DB triggers blocking update/delete
- admin audit endpoint: `GET /admin/audit`

4. Q/A document archival + export:
- `log_interaction` now also writes to `qa_documents`
- export endpoints:
  - `GET /qa/documents`
  - `GET /qa/exports`
  - `POST /qa/export` (`jsonl|csv|md|pdf`)
  - `GET /qa/export/{filename}`
- exports folder: `EXPORTS_PATH` (`./data/exports` default)

5. UI updates:
- Q/A export controls + export list
- webchat snippet viewer/copy
- global actor/token fields to satisfy RBAC without page reload

6. Embeddable widget:
- `app/web/webchat.js`
- injected via snippet and uses chat API

## Updated Config/Dependencies
- `.env.example` extended with docs/export/chat/language/auth settings
- `requirements.txt` added:
  - `PyYAML==6.0.3`
  - `reportlab==4.2.5`
  - `pytest==8.3.4`
  - `httpx==0.27.2`
- `VERSION` file set to `0.2.0`
- `.gitignore` now ignores `data/exports/`
- GitHub Actions CI: `.github/workflows/ci.yml`

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
- Pytest suite added for:
  - RBAC/audit immutability
  - chat + export flow
  - replyctl lifecycle

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
1. Extend dependency adapters beyond local Homebrew paths.
2. Add stronger token/session management (rotation and expiration).
3. Add API contract tests for remaining admin/editor endpoints.
4. Add signed/hashed audit export bundles for tamper-evidence off-host.
