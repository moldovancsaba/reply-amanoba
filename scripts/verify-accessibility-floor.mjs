// Accessibility minimums no theme may cross, checked across every preset and scheme.
//
// No warning tier: a floor breach fails the build.
//
// Output: audit/accessibility-floor.json

import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = new URL('..', import.meta.url).pathname;
const { auditGdsAccessibilityFloor, gdsAccessibilityFloorRules, describeGdsAccessibilityFloor } =
  await import(join(ROOT, 'packages/gds-theme/dist/index.js'));

if (!auditGdsAccessibilityFloor) {
  console.error('FAIL could not load the accessibility floor from packages/gds-theme/dist — run `npm run build` first.');
  process.exit(1);
}

// Canary: feeds the floor a context that breaches every rule and asserts it comes back dirty,
// or a clean run below would be meaningless.
const { validateGdsAccessibilityFloor, getGdsVibeThemeCssVariables } = await import(join(ROOT, 'packages/gds-theme/dist/index.js'));
const canaryTokens = {
  ...getGdsVibeThemeCssVariables('default', 'light'),
  '--gds-focus-ring-width': '1px',
  '--gds-focus-ring-style': 'none',
  '--gds-control-height-md': '30px',
  '--gds-line-height-md': '1.2',
  '--gds-motion-duration-slow': '5000ms',
  '--gds-motion-policy': 'never-reduce',
  '--gds-control-disabledText': '#cccccc',
  '--gds-control-disabledBg': '#cccccc',
  '--gds-badge-soft-info': '#ffffff',
  '--gds-badge-soft-info-fg': '#fefefe',
  // White on this yellow is ~1.2:1 — breaches primary-cta-text-contrast, which the canary must
  // prove fires even though it only reports.
  '--gds-vibe-primary': '#ffee00',
  // Same reasoning for ai-accent-text-contrast (issue 697): a near-white fill measures well
  // under 4.5:1 and the report-severity rule must still fire on it.
  '--gds-ai-accent': '#ffee00',
  // Same reasoning for outline-accent-text-contrast (issue 700): a near-white value on the
  // 'default' light preset's own (light) page surface measures well under 4.5:1.
  '--gds-brand-accent-action': '#ffee00',
};
const canary = validateGdsAccessibilityFloor({ presetId: 'canary', scheme: 'light', tokens: canaryTokens });
const canaryRules = new Set(canary.map((v) => v.ruleId));
if (canaryRules.size < gdsAccessibilityFloorRules.length) {
  console.error(`FAIL canary detected only ${canaryRules.size} of ${gdsAccessibilityFloorRules.length} rules.`);
  console.error('     A deliberately non-compliant theme must breach every rule. Rules that stayed silent:');
  for (const r of gdsAccessibilityFloorRules) if (!canaryRules.has(r.id)) console.error(`       ${r.id}`);
  console.error('     Until this passes, a clean run below would mean nothing.');
  process.exit(1);
}

const { presetsChecked, rulesEvaluated, violations, reports } = auditGdsAccessibilityFloor();

mkdirSync(join(ROOT, 'audit'), { recursive: true });
writeFileSync(join(ROOT, 'audit/accessibility-floor.json'), `${JSON.stringify({
  presetsChecked, rulesEvaluated, violationCount: violations.length, reportCount: reports.length,
  rules: gdsAccessibilityFloorRules.map((r) => ({ id: r.id, axis: r.axis, wcag: r.wcag, rationale: r.rationale })),
  violations,
}, null, 2)}\n`);

console.log('Accessibility floor (issue 559)\n');
console.log(`  rules            ${String(rulesEvaluated).padStart(4)}`);
console.log(`  preset x scheme  ${String(presetsChecked).padStart(4)}`);
console.log(`  canary           ${String(canaryRules.size).padStart(4)}/${gdsAccessibilityFloorRules.length} rules proven live`);
console.log(`  violations       ${String(violations.length).padStart(4)}`);
console.log(`  reported         ${String(reports.length).padStart(4)}  measured, not enforced (issue 680)`);
if (reports.length) {
  // Printed in full: a measured-but-unenforced axis that nobody reads is the same as not
  // measuring it. Grouped by rule so the standing of the whole preset set is visible at a glance.
  const byRule = new Map();
  for (const r of reports) byRule.set(r.ruleId, [...(byRule.get(r.ruleId) ?? []), r]);
  for (const [ruleId, rows] of byRule) {
    console.log(`\n  ${ruleId} — ${rows.length} preset/scheme combination(s) below the threshold:`);
    for (const r of rows) console.log(`    ${r.presetId} (${r.scheme}): ${r.actual}, wants ${r.required}`);
  }
}

if (violations.length) {
  console.error('');
  for (const v of violations.slice(0, 40)) {
    console.error(`FAIL ${v.ruleId} — ${v.presetId}/${v.scheme}: ${v.actual}, required ${v.required}`);
    console.error(`     ${v.wcag}\n     Fix: ${v.fix}`);
  }
  if (violations.length > 40) console.error(`  … and ${violations.length - 40} more.`);
  console.error(`\n${violations.length} accessibility floor violation(s). A floor breach is never advisory.`);
  process.exit(1);
}

// Docs must match the rule set the gate enforces.
const doc = join(ROOT, 'docs/ACCESSIBILITY_FLOOR.md');
const expected = describeGdsAccessibilityFloor();
try {
  if (!readFileSync(doc, 'utf8').includes(expected)) {
    console.error(`FAIL ${doc} does not match the rule set. Run \`npm run docs:a11y-floor\` and commit the result.`);
    process.exit(1);
  }
} catch {
  console.error(`FAIL ${doc} is missing. Run \`npm run docs:a11y-floor\`.`);
  process.exit(1);
}

console.log(`\nEvery preset clears the floor on all ${rulesEvaluated} rules.`);
