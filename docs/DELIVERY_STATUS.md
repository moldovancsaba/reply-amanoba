# Delivery Status (Current Pass)

## Delivered
- Platform lifecycle foundation:
  - `replyctl install/start/stop/status/uninstall`
  - `replyctl backup/restore/update/update --all`
  - dependency management commands (`doctor`, `deps-install`, `deps-verify`, `deps-upgrade`)
- Tenant manifest and instance layout:
  - `deploy/company.yaml.example`
  - per-instance config/data/log/run/backups
- Webchat module foundation:
  - API endpoints (`/chat/*`)
  - embeddable script (`/static/webchat.js`)
  - admin snippet endpoint (`/admin/webchat/snippet`)
  - chat auth (token), CORS allowlist, rate limit
- Server-side RBAC:
  - role-token enforcement for protected endpoints
  - configurable via `AUTH_ENABLED`, `ADMIN_API_TOKEN`, `EDITOR_API_TOKEN`, `EMPLOYEE_API_TOKEN`
- Immutable audit trail:
  - `audit_events` table
  - DB triggers reject update/delete
  - admin endpoint: `GET /admin/audit`
- Q/A archival and export:
  - every logged interaction saved into `qa_documents`
  - export endpoints for `jsonl`, `csv`, `md`, `pdf`
  - admin UI actions to create/download exports
- CI gates on push/PR:
  - `.github/workflows/ci.yml`
  - dependency install + compile checks + pytest

## In Progress / Next
- Fleet-grade update orchestration and rollback verification tests.
- Better dependency adapters for Linux/Windows package managers.
- Stronger token/session management (rotation/expiration).
- Signed audit export bundles for off-host tamper evidence.

## Operational Notes
- Exports are written under `EXPORTS_PATH` (`./data/exports` by default).
- Chat token is optional; when set, requests must include bearer or `X-API-Token`.
- `replyctl` prefers project `.venv` Python when available.
