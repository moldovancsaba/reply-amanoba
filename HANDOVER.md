# Handover - {reply}

## Snapshot
Project: `/Users/moldovancsaba/Projects/reply-amanoba`
State: Working MVP with active UI/UX refinement cycle.
Date context: 2026-02-24 session.

## Delivered So Far
1. End-to-end internal help bot with strict RAG + escalation.
2. Editor ticket resolution and annotation loop.
3. Clarification generation and Improve Me enrichment loop.
4. Persona-based UI (Employee/Editor/Admin) with function separation.
5. Admin dashboard for system/annotation/operations health.
6. Local model switching, comparison, and runtime fallback.
7. Unified launcher with local + public tunnel startup.
8. Global CSS extraction and major dark-theme redesign.
9. Ticket delete/dismiss action in editor workflow.

## Most Recent User Direction
- User is highly focused on visual polish.
- Explicit requirement: one centralized alignment system across all personas and all components.
- User is very sensitive to tiny spacing/indent inconsistencies.

## Current UI Focus Area
- `app/web/styles.css` contains centralized token system.
- Alignment baseline currently controlled through `--align-x` and centralized selectors.
- Ticket rows and top controls have been iterated repeatedly; this remains the highest-friction area with the user.

## Backend/Logic Status
- Core flows work:
  - upload -> reindex -> ask -> escalate -> editor resolve/dismiss -> re-annotate
  - improve-me generation -> answer -> re-annotate
- API routes are complete for current MVP scope.

## Known Defects To Fix Next
1. `app/main.py` duplicate argument:
   - `timeout=120` appears twice in one `requests.post` call.
2. `app/main.py` enrichment counter bug:
   - `created += added` appears duplicated in `/enrichment/generate` ticket-based branch.

## Recommended Immediate Next Steps (for next agent)
1. Fix the two backend defects above.
2. Perform pixel-alignment QA in browser for all personas:
   - compare left x-origin for: `h2`, notes, button rows, status lines, ticket text, table header.
3. Add visual regression check (Playwright screenshot baseline) to prevent alignment regressions.
4. Add server-side role enforcement (currently persona separation is UI-level).

## Test Checklist
- Employee:
  - Ask with Enter
  - Escalation creates ticket
- Editor:
  - Upload docs (`pdf/md/txt`)
  - Improve Me question appears
  - Submit answer annotates
  - Open tickets: Answer + Delete work
- Admin:
  - Model list/select works
  - Dashboard loads
  - Public URL loads and copies
- Global:
  - No full-page reload on actions
  - Alignment baseline visually consistent

## Important Files
- `/Users/moldovancsaba/Projects/reply-amanoba/app/main.py`
- `/Users/moldovancsaba/Projects/reply-amanoba/app/db.py`
- `/Users/moldovancsaba/Projects/reply-amanoba/app/schemas.py`
- `/Users/moldovancsaba/Projects/reply-amanoba/app/web/index.html`
- `/Users/moldovancsaba/Projects/reply-amanoba/app/web/styles.css`
- `/Users/moldovancsaba/Projects/reply-amanoba/start_helpbot.command`

## Handover Notes
- User expectation is not “good enough”; user wants strict visual uniformity.
- Prioritize concrete UI fixes over conceptual explanations.
- Validate with screenshots when possible before claiming done.
