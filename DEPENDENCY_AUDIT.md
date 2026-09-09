# Dependency Audit Policy

Status: Active SSOT
Version: 6.7.0
Last updated: 2026-08-08

This repository treats published runtime package dependencies and local reference/tooling dependencies differently.

## Release Gate

`npm run audit:dependencies` enforces:

- `npm audit --omit=dev` must have zero findings.
- Full `npm audit` may only contain explicitly accepted dev/reference-tooling advisories listed in this document.
- Accepted advisories must have an owner, reason, and review date.

## Accepted Dev / Reference Tooling Advisories

### GHSA-qx2v-qp2m-jg93

Owner: GDS platform
Review date: 2026-07-06
Scope: `apps/reference-next` development/reference fixture via `next@15.5.18` and nested `postcss@8.4.31`
Severity: moderate

Reason:

- The finding is currently reported through Next's nested PostCSS dependency in the private App Router reference fixture.
- `next` is kept in `devDependencies` for the reference fixture so it is not part of the production dependency audit or published GDS package runtime surface.
- The latest stable Next line available during this release still declares the same nested PostCSS version, so forcing a framework major or npm's suggested downgrade is not a safe corrective action.

Operational behavior:

- Do not ship public consumer guidance that requires `apps/reference-next` as a runtime dependency.
- Recheck monthly or when Next publishes a patched stable dependency graph.
- Remove this exception once `npm audit --json` no longer reports the advisory through the reference fixture.

### GHSA-f88m-g3jw-g9cj

Owner: GDS platform
Review date: 2026-08-22
Scope: `apps/reference-next` development/reference fixture via `next@15.5.20` and nested `sharp@0.34.5` (libvips CVE-2026-33327/33328/35590/35591)
Severity: high

Reason:

- The finding is reported through Next's nested `sharp` image-optimization dependency in the private App Router reference fixture, the same non-shipped dependency chain as `GHSA-qx2v-qp2m-jg93`.
- `next` is kept in `devDependencies` for the reference fixture so it is not part of the production dependency audit or published GDS package runtime surface. The reference fixture does not serve or process untrusted images.
- The only automated fix path (`npm audit fix --force`) downgrades `next` to `9.3.3`, a major regression far below the supported `15.x` reference line, so it is not a safe corrective action.

Operational behavior:

- Do not ship public consumer guidance that requires `apps/reference-next` as a runtime dependency.
- Recheck monthly or when Next publishes a patched `sharp`/libvips dependency graph.
- Remove this exception once `npm audit --json` no longer reports the advisory through the reference fixture.

### GHSA-6g55-p6wh-862q

Owner: GDS platform
Review date: 2026-08-23
Scope: nested `postcss@8.4.31` (`<=8.5.11` vulnerable range) reached via two dev-only paths — `apps/reference-next`'s `next` dependency (same non-shipped reference fixture as the other accepted advisories above) and the root `tsup` devDependency used to build the published packages
Severity: high

Reason:

- The advisory ("Arbitrary file read and information disclosure via attacker-controlled sourceMappingURL in CSS comments") requires processing attacker-controlled CSS input containing a malicious `sourceMappingURL` comment. Neither path applies here: `apps/reference-next` doesn't serve or process untrusted CSS, and `tsup` only ever processes GDS's own first-party, repo-controlled source during the build — no untrusted CSS ever reaches PostCSS in either case.
- `postcss` is a nested dev-tooling dependency in both paths, not a runtime dependency of any published `@sovereignsquad/*` package (`npm audit --omit=dev` reports zero findings) — it never ships in built package output.
- The only automated fix (`npm audit fix --force`) downgrades `next` to `9.3.3`, a major regression far below the supported `15.x` reference line, so it is not a safe corrective action.

Operational behavior:

- Do not ship public consumer guidance that requires `apps/reference-next` or `tsup` as a runtime dependency (already true — both are dev-only).
- Recheck monthly or when Next/tsup publish a patched nested PostCSS dependency graph.
- Remove this exception once `npm audit --json` no longer reports the advisory through either path.

### GHSA-r28c-9q8g-f849

Owner: GDS platform
Review date: 2026-08-24
Scope: nested `postcss@8.4.31` reached via the same two dev-only paths as `GHSA-6g55-p6wh-862q` above — `apps/reference-next`'s `next` dependency (non-shipped App Router reference fixture) and the root `tsup` devDependency used to build the published packages
Severity: high

Reason:

- The advisory ("Path Traversal in Previous Source Map Auto-Loading (sourceMappingURL) leads to Arbitrary `.map` File Disclosure") is the same PostCSS source-map class as `GHSA-6g55-p6wh-862q`: it requires PostCSS to process attacker-controlled CSS containing a malicious `sourceMappingURL`. Neither path does that — `apps/reference-next` never serves or processes untrusted CSS, and `tsup` only ever processes GDS's own first-party, repo-controlled source during the build.
- `postcss` is a nested dev-tooling dependency in both paths, not a runtime dependency of any published `@sovereignsquad/*` package (`npm audit --omit=dev` reports zero findings) — it never ships in built package output.
- Newly disclosed after the 3.12.0 line; surfaced here once the react-router production advisory (GHSA-qwww-vcr4-c8h2) was remediated by the React 19 / react-router 8 upgrade and stopped masking the full-audit dev findings. Forcing a nested-`postcss` override across `next`/`tsup` risks build-graph churn in the reference fixture for no shipped benefit, matching the disposition already recorded for `GHSA-6g55-p6wh-862q`.

Operational behavior:

- Do not ship public consumer guidance that requires `apps/reference-next` or `tsup` as a runtime dependency (already true — both are dev-only).
- Recheck monthly or when Next/tsup publish a patched nested PostCSS dependency graph.
- Remove this exception once `npm audit --json` no longer reports the advisory through either path.

### GHSA-fxqj-rqcc-2cmp

Owner: GDS platform
Review date: 2026-09-05
Scope: nested `postcss@8.4.31` reached via the same dev-only path as the other `postcss` advisories above — `apps/reference-next`'s `next` dependency (non-shipped App Router reference fixture)
Severity: moderate

Reason:

- GitHub's own advisory title is "incomplete fix of `GHSA-6g55-p6wh-862q`" — this is a refinement of the sourceMappingURL/arbitrary-file-read class already accepted above, in the identical dependency chain, not a new exploit surface. The same disposition applies verbatim: it requires PostCSS to process attacker-controlled CSS containing a malicious `sourceMappingURL`, and `apps/reference-next` never serves or processes untrusted CSS.
- `postcss` is a nested dev-tooling dependency, not a runtime dependency of any published `@sovereignsquad/*` package (`npm audit --omit=dev` reports zero findings) — it never ships in built package output.
- `next@15.5.21` (the latest stable 15.x line) still hard-pins `postcss@8.4.31` internally (not a range), so no in-place patch or update within the supported 15.x reference line resolves it. The only fix is `next@16.3.0`, a semver-major bump — already tracked as "Accepted, review at next sweep" in the Staleness & Deprecation Sweep table below, and matching the same "no safe automated fix" disposition recorded for this advisory's siblings.

Operational behavior:

- Do not ship public consumer guidance that requires `apps/reference-next` as a runtime dependency (already true — dev-only).
- Recheck monthly or when Next publishes a patched stable dependency graph, alongside the tracked `next@16` sweep item.
- Remove this exception once `npm audit --json` no longer reports the advisory through the reference fixture.

## Staleness & Deprecation Sweep (added 2026-07-24, housekeeping issue #406)

`scripts/generate-dependency-risk-report.mjs` now also records an
`npm outdated`-derived staleness snapshot and a deprecation check
(`npm view <pkg> deprecated`) in `dependency-risk-report.json`'s `staleness`
field, run as part of `audit:dependencies`. This is deliberately
**warn-and-record, not release-blocking** — flipping every outdated or
soft-deprecated dependency into a hard gate overnight would create an
unplannable cascade of forced upgrades. Findings are triaged here, by hand,
after each sweep; only a specific triaged item ever becomes a hard
requirement (tracked as its own dedicated issue, never a blanket policy
change to this file).

First sweep (2026-07-24): 28 outdated packages, 0 deprecated packages found.
Disposition for the 13 packages with a newer major version available:

| Package(s) | Disposition | Reason |
|---|---|---|
| `@mantine/core`, `@mantine/dates`, `@mantine/hooks`, `@mantine/modals`, `@mantine/notifications` (7.17.8 → 9.4.2) | Tracked, no action | Intentional multi-version support lane — `verify:mantine` already smoke-tests Mantine 7/8/9 compatibility on every release; this is the CI matrix working as designed, not staleness. |
| `react`, `react-dom` (18.3.1 → 19.2.7) | Upgraded (workspace runtime) | Bumped the dev/app runtime to React 19 so the playground can adopt `react-router@8` (which peer-requires React ≥19.2.7) and remediate GHSA-qwww-vcr4-c8h2 (issue #430). The published peer range stays `^18.2.0 || ^19.0.0` — React 18 remains a supported consumer lane, still validated via `verify:mantine`'s Mantine 7 + React 18 consumer-install smoke. |
| `next` (15.5.21 → 16.2.11) | Accepted, review at next sweep | Dev-only reference-fixture dependency (`apps/reference-next`), not a published runtime dependency; already carries its own accepted-advisory entries above. |
| `typescript` (packages build 5.9.3 → 7.0.2 available; reference apps pinned 6.0.x) | Accepted, review at next sweep | Intentional cross-major lane: the published packages are typechecked and their `.d.ts` emitted under TS 5.9.x (peer floor `5.4.5`, root workspace line), while the private reference apps run TS 6.0.x. Tooling-only, no runtime exposure and no known compatibility blocker. A package-build TS major bump (to 6.x/7.x) warrants its own dedicated verification pass rather than a routine bump. Authoritative support line: [COMPATIBILITY_AND_RELEASES.md](COMPATIBILITY_AND_RELEASES.md) + `compatibility.matrix.json`. |
| `@babel/core` (7.29.7 → 8.0.1), `@testing-library/jest-dom` (6.9.1 → 7.0.0), `@types/node` (24.13.3 → 26.1.1), `jsdom` (26.1.0 → 29.1.1) | Accepted, review at next sweep | Dev/test tooling only, no runtime exposure; routine upgrades deferred to avoid unrelated churn in this housekeeping batch. |

Owner: GDS platform. Review cadence: re-run `npm run audit:dependencies` and
re-triage this table at least once per minor release, or sooner if a
deprecation notice appears (`staleness.deprecatedPackages` non-empty).

## Resolved During 3.0.7

- `vitest` was upgraded from `3.2.4` to `4.1.8`, resolving GHSA-5xrq-8626-4rwp for the local test runner.

