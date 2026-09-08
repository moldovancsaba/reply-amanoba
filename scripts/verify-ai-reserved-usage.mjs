// Reserved-usage gate for the ai.* sub-brand accent lane (issue 697).
//
// The lane's tokens (--gds-ai-gradient/-panel/-accent) are reserved to a named, closed set of
// Scout AI surfaces (THEME_GOVERNANCE.md): the gradient belongs to Scout AI exclusively and is
// never a general action colour. This scans the consumer surface — gds-core's component
// source and the shipped static stylesheet — for any reference to the reserved token family
// and fails loudly, naming file and line, on anything not explicitly allowlisted.
//
// Widening the allowlist requires a governance-reviewed change to it in the same change set as
// the sanctioned component (THEME_GOVERNANCE.md).
//
// Emission sites inside packages/gds-theme/src are exempt by construction: this scan targets
// consumers, never the emitter.

import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join, relative } from 'node:path';

const ROOT = new URL('..', import.meta.url).pathname;

// Sanctioned consumer entries, relative to the repo root. Two shapes:
//   'path/to/File.tsx'                        — the whole file is exempt (a small,
//                                                single-purpose consumer component).
//   'path/to/file.css::<exact trimmed line>'  — only that one line is exempt, for a large
//                                                shared file (styles.css) where a whole-file
//                                                exemption would blind the gate to every other
//                                                line in it.
//
// First widened by issue 700: SemanticButton's `gradient` brand intent is the one
// SemanticButton treatment sanctioned to consume `ai.gradient`, reserved for AI-identity CTAs
// exclusively (e.g. "Ask Scout AI"). Its resting paint lives in packages/gds-theme/styles.css,
// keyed on `data-gds-brand-button='gradient'` — allowlisted by that single line's exact text,
// not the whole file, so a real unsanctioned reference anywhere else in styles.css still fails
// loudly (verify-ai-reserved-usage.test.mjs / gate-mutants.config.mjs both prove this).
// SemanticButton.tsx itself references no `--gds-ai-` literal and needs no entry. Every other
// sanctioned component (AISearchCard, the chat surfaces, the AI promo panel, BottomTabBar's
// emphasized disc, the focus ring, the featured ring) still lands in a follow-on issue in this
// same delivery. A future PR adding another sanctioned consumer adds its entry here in the
// same change set, per THEME_GOVERNANCE.md's reserved-usage rule.
export const AI_RESERVED_USAGE_ALLOWLIST = new Set([
  "packages/gds-theme/styles.css::background-image: var(--gds-ai-gradient, none);",
]);

const TOKEN_MARKER = '--gds-ai-';

/** Recursively lists files under `dir` whose name ends with one of `exts`. */
export function readAllFiles(dir, exts, acc = []) {
  if (!existsSync(dir)) return acc;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (!/node_modules|dist|__snapshots__/.test(entry.name)) readAllFiles(path, exts, acc);
    } else if (exts.some((ext) => entry.name.endsWith(ext))) {
      acc.push(path);
    }
  }
  return acc;
}

/**
 * Scans `files` (absolute paths) for the reserved `--gds-ai-` token marker, and returns one
 * violation per non-allowlisted line that references it. `root` is used only to compute the
 * relative path checked against `allowlist`. An allowlist entry is either a bare relative path
 * (exempts the whole file) or `path::<exact trimmed line text>` (exempts only that one line, so
 * a large shared file stays scanned everywhere else).
 */
export function scanForUnsanctionedAiReferences(files, allowlist, root) {
  const violations = [];
  for (const file of files) {
    const relPath = relative(root, file);
    const wholeFileAllowed = allowlist.has(relPath);
    const lines = readFileSync(file, 'utf8').split('\n');
    lines.forEach((line, index) => {
      if (!line.includes(TOKEN_MARKER)) return;
      if (wholeFileAllowed) return;
      const trimmed = line.trim();
      if (allowlist.has(`${relPath}::${trimmed}`)) return;
      violations.push({ file: relPath, line: index + 1, text: trimmed });
    });
  }
  return violations;
}

function main() {
  const targets = [
    ...readAllFiles(join(ROOT, 'packages/gds-core/src'), ['.ts', '.tsx']),
    join(ROOT, 'packages/gds-theme/styles.css'),
  ].filter((file) => existsSync(file));

  const violations = scanForUnsanctionedAiReferences(targets, AI_RESERVED_USAGE_ALLOWLIST, ROOT);
  const allowlistEntries = AI_RESERVED_USAGE_ALLOWLIST.size;

  console.log('AI reserved-usage gate (issue 697)');
  console.log(`  files scanned:     ${targets.length}`);
  console.log(`  allowlist entries: ${allowlistEntries}`);
  console.log(`  violations:        ${violations.length}`);

  if (violations.length) {
    console.error('');
    console.error('FAIL --gds-ai-* referenced outside the sanctioned allowlist:');
    for (const v of violations) console.error(`  ${v.file}:${v.line}: ${v.text}`);
    console.error('');
    console.error('The ai.* lane is reserved for Scout AI surfaces plus the focus ring and the featured');
    console.error('ring (THEME_GOVERNANCE.md) — never a general action colour. Widening the allowlist');
    console.error('requires a governance-reviewed change to it in the same change set as the sanctioned');
    console.error('component.');
    process.exit(1);
  }

  console.log('\nNo unsanctioned --gds-ai-* reference found.');
}

// Only run as a CLI; the test suite imports the functions above directly.
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
