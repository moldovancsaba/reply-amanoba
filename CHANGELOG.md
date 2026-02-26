# Changelog

## v0.2.0 - 2026-02-26
- Added server-side RBAC with role tokens (`employee`, `editor`, `admin`).
- Added immutable `audit_events` with DB-level no-update/no-delete triggers.
- Added admin audit endpoint: `GET /admin/audit`.
- Added CI workflow on push/PR with compile checks and pytest gates.
- Added automated tests for:
  - RBAC and audit immutability
  - chat + export flow
  - replyctl lifecycle install/update/uninstall
- Updated UI with actor/token fields and authenticated export downloads.

## v0.1.0 - 2026-02-26
- Runtime foundation (`replyctl`, webchat API/widget, Q/A export pipeline).
