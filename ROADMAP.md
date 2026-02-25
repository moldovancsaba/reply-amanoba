# {reply} Roadmap

## Priority Scale
- `P0` = critical now (quality/safety blocker)
- `P1` = high impact next
- `P2` = important optimization
- `P3` = strategic later

## 1) Retrieval Quality (Hybrid + Filtering + Rerank)
Priority: `P0`

### User Story 1.1
As an employee, I want the bot to find the most relevant policy passages so answers are accurate.
- Subtasks:
1. Add BM25/keyword search on top of vector search.
2. Merge vector + keyword results with weighted scoring.
3. Return top-k combined candidates before generation.
- Agent prompt:
`Implement hybrid retrieval in this FastAPI app by combining embedding similarity and BM25 keyword scoring. Add a configurable weight in settings, update retrieval pipeline, and include tests for relevance ordering.`

### User Story 1.2
As an employee, I want answers scoped to my context (department/country/version) so I don’t get wrong policy.
- Subtasks:
1. Add metadata schema for doc chunks (`department`, `locale`, `effective_date`, `version`).
2. Save metadata during ingestion.
3. Add filter fields to `/ask` request and retrieval logic.
- Agent prompt:
`Extend ingestion and retrieval to support metadata filters (department, locale, effective_date, version). Update schemas, DB, and /ask endpoint so retrieval applies filters before ranking.`

### User Story 1.3
As an employee, I want cleaner final context selection so answers are less noisy.
- Subtasks:
1. Add reranker step over top N retrieved chunks.
2. Keep best M chunks for prompt context.
3. Log reranker scores for debugging.
- Agent prompt:
`Add a reranking stage after initial retrieval and before generation. Keep top M reranked chunks, expose debug scores in logs, and keep backward compatibility with existing ask flow.`

## 2) Answer Reliability (Citation Gate + Confidence)
Priority: `P0`

### User Story 2.1
As an employee, I want only cited answers so I can trust outputs.
- Subtasks:
1. Add strict citation validator for generated responses.
2. If missing citations, auto-escalate ticket.
3. Return clear reason to user.
- Agent prompt:
`Implement a strict citation gate in /ask: if answer lacks valid citations to retrieved chunks, escalate to ticket and log reason 'missing citations'.`

### User Story 2.2
As an admin, I want confidence thresholds so uncertain answers are escalated safely.
- Subtasks:
1. Compute confidence from retrieval + model signals.
2. Add configurable threshold.
3. Escalate below threshold with reason.
- Agent prompt:
`Add confidence scoring to answer flow (based on retrieval quality and answer checks), configure threshold in settings, and auto-escalate below threshold with explicit reason.`

### User Story 2.3
As an employee, I want warnings when documents may be outdated.
- Subtasks:
1. Add document effective/expiry metadata.
2. Check staleness in retrieved sources.
3. Add warning/escalation policy for stale docs.
- Agent prompt:
`Implement stale-document detection using effective_date/version metadata and append warning or escalation when retrieved sources are outdated.`

## 3) Learning Loop (Unknown Handling + Ticket Clustering)
Priority: `P1`

### User Story 3.1
As an editor, if I don’t know an answer, I want the system to route it correctly without losing context.
- Subtasks:
1. Keep current “Teach Me” escalation to open tickets.
2. Create “missing knowledge topics” table.
3. Link escalated clarifications to topic entries.
- Agent prompt:
`Add a missing_knowledge_topics table and link enrichment escalations to it. Keep existing teach-me escalation behavior and persist structured unknown reasons.`

### User Story 3.2
As an editor, I want repeated similar tickets grouped so one answer solves many.
- Subtasks:
1. Add similarity clustering for open tickets.
2. Show grouped tickets in editor UI.
3. Allow resolving an entire cluster with one response.
- Agent prompt:
`Implement ticket clustering for semantically similar open tickets and add a bulk-resolve action so one editor answer can close and annotate all tickets in the cluster.`

## 4) Ingestion Intelligence (Structure + Semantic Chunking + Dedupe)
Priority: `P1`

### User Story 4.1
As an admin, I want auto-extracted structure so retrieval and filters are better.
- Subtasks:
1. Parse headings/sections/titles from uploaded files.
2. Store structured metadata per chunk.
3. Expose metadata in dashboard.
- Agent prompt:
`Enhance ingestion to extract section/title structure from PDF/MD/TXT and store metadata with each chunk; display extracted metadata status in dashboard.`

### User Story 4.2
As an employee, I want semantically coherent chunks so responses are more readable and precise.
- Subtasks:
1. Replace naive chunking with section-aware chunking.
2. Keep overlap for continuity.
3. Reindex existing corpus with migration script.
- Agent prompt:
`Implement semantic/section-aware chunking with overlap, migrate existing docs, and provide a safe reindex command for current database.`

### User Story 4.3
As an admin, I want duplicate detection to reduce noisy indexing.
- Subtasks:
1. Add near-duplicate hash/similarity detection on ingest.
2. Skip or merge duplicates with logs.
3. Report duplicate stats in dashboard.
- Agent prompt:
`Add duplicate and near-duplicate detection during upload/indexing, skip redundant chunks, and expose duplicate metrics in dashboard.`

## 5) Editor Productivity (Draft Assist + Faster Publish + SLA)
Priority: `P2`

### User Story 5.1
As an editor, I want a draft answer proposal to speed up resolution.
- Subtasks:
1. Add “Suggest Draft” button for ticket.
2. Generate draft strictly from retrieved context.
3. Require editor approval before publish.
- Agent prompt:
`Add an editor-only draft suggestion flow for open tickets using strict retrieved context, with manual approve/edit before any annotation to KB.`

### User Story 5.2
As an editor, I want one-click answer + publish actions.
- Subtasks:
1. Add compact action controls in ticket row.
2. Resolve ticket + annotate in one API call.
3. Show immediate status feedback.
- Agent prompt:
`Create one-click resolve-and-annotate editor action with clear success/error states and no full-page reload.`

### User Story 5.3
As an admin, I want SLA visibility on unresolved items.
- Subtasks:
1. Add oldest-open ticket metric.
2. Add aging buckets (0-1d, 2-3d, 4+d).
3. Surface in dashboard and admin panel.
- Agent prompt:
`Implement SLA metrics for open tickets (oldest age, aging buckets) and display in the admin dashboard.`

## 6) Governance & Safety (Server-Side Roles + Audit)
Priority: `P0`

### User Story 6.1
As an organization, I need enforced server-side access control.
- Subtasks:
1. Add auth (token/session) middleware.
2. Enforce role checks per endpoint.
3. Return proper authorization errors.
- Agent prompt:
`Implement server-side authentication and role-based authorization for all endpoints. Keep current personas but enforce permissions in backend, not only UI.`

### User Story 6.2
As an admin, I need complete audit logs for every critical action.
- Subtasks:
1. Add audit_log table.
2. Log uploads, model switches, ticket actions, enrichment answers.
3. Add audit export endpoint.
- Agent prompt:
`Add immutable audit logging for all critical operations and expose a secure admin endpoint to query/export audit records.`

### User Story 6.3
As a security owner, I want abuse/PII protections.
- Subtasks:
1. Add basic rate limiting.
2. Add PII redaction hooks for logs.
3. Add retention config for tickets/interactions.
- Agent prompt:
`Add rate limiting, configurable PII redaction for logs, and data retention policies for interactions/tickets with scheduled cleanup support.`

## 7) Multilingual Quality
Priority: `P1`

### User Story 7.1
As an employee, I want answers in my language reliably.
- Subtasks:
1. Improve language detection with fallback.
2. Force response language policy.
3. Add language mismatch checks.
- Agent prompt:
`Improve language detection and enforce same-language response policy with fallback handling and mismatch validation.`

### User Story 7.2
As an editor, I want canonical + translated knowledge management.
- Subtasks:
1. Store canonical answer + translated variants.
2. Link variants to same knowledge entity.
3. Prefer canonical for retrieval logic, translate on output.
- Agent prompt:
`Implement canonical knowledge records with linked language variants; prefer canonical retrieval and generate/serve locale-specific responses.`

### User Story 7.3
As an admin, I want per-language quality metrics.
- Subtasks:
1. Track answer/escalation rate by language.
2. Track editor override rate by language.
3. Show in dashboard.
- Agent prompt:
`Add language-segmented quality metrics (answer rate, escalation rate, override rate) and surface them in admin dashboard.`

## 8) Quality Measurement & Continuous Improvement
Priority: `P1`

### User Story 8.1
As an admin, I want quality KPIs to monitor real impact.
- Subtasks:
1. Define KPI schema (`resolution_rate`, `first_response_quality`, `repeat_ticket_rate`).
2. Compute daily aggregates.
3. Show trend panels in dashboard.
- Agent prompt:
`Implement KPI tracking and daily aggregates for support quality metrics, then add trend visualization data endpoints for dashboard charts.`

### User Story 8.2
As a product owner, I want benchmark-based regression checks before changes.
- Subtasks:
1. Create evaluation dataset from real anonymized tickets.
2. Add offline eval runner.
3. Require score thresholds before release.
- Agent prompt:
`Build an evaluation harness with an anonymized benchmark dataset and scoring rules; fail CI/local check when quality drops below thresholds.`

### User Story 8.3
As an admin, I want a weekly failure-intent review.
- Subtasks:
1. Auto-group escalations by intent/topic.
2. Generate weekly summary report.
3. Suggest top 10 missing knowledge actions.
- Agent prompt:
`Create weekly intent-mining for escalations and generate a prioritized missing-knowledge report with recommended ingestion/editor actions.`

## Suggested Delivery Plan
- Sprint 1 (`P0`): 1, 2, 6
- Sprint 2 (`P1`): 3, 4, 7, 8
- Sprint 3 (`P2`): 5

## Execution Rule For Agents
- For every story:
1. Implement backend first.
2. Add tests or verification script.
3. Expose minimal UI.
4. Update docs (`README.md`, `HANDOVER.md`) before closing.
