import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, unlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { readAllFiles, scanForUnsanctionedAiReferences } from './verify-ai-reserved-usage.mjs';

const ROOT = process.cwd();
const SCRIPT = join(ROOT, 'scripts/verify-ai-reserved-usage.mjs');

function runGate() {
  try {
    const output = execFileSync('node', [SCRIPT], { cwd: ROOT, encoding: 'utf8' });
    return { exitCode: 0, output };
  } catch (error) {
    return { exitCode: error.status ?? 1, output: `${error.stdout ?? ''}${error.stderr ?? ''}` };
  }
}

describe('ai reserved-usage gate (issue 697)', () => {
  it('is green on the tree as shipped', () => {
    const { exitCode, output } = runGate();
    expect(exitCode).toBe(0);
    expect(output).toContain('No unsanctioned --gds-ai-* reference found.');
  });

  it('the pure scanner detects an unsanctioned reference, naming file and line', () => {
    const dir = mkdtempSync(join(tmpdir(), 'gds-ai-reserved-usage-'));
    try {
      const file = join(dir, 'UnsanctionedComponent.tsx');
      writeFileSync(file, "const style = { background: 'var(--gds-ai-accent)' };\n// second reference: --gds-ai-panel\n");
      const violations = scanForUnsanctionedAiReferences([file], new Set(), dir);
      expect(violations).toHaveLength(2);
      expect(violations[0]).toMatchObject({ file: 'UnsanctionedComponent.tsx', line: 1 });
      expect(violations[0].text).toContain('--gds-ai-accent');
      expect(violations[1]).toMatchObject({ file: 'UnsanctionedComponent.tsx', line: 2 });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('the pure scanner exempts an allowlisted file', () => {
    const dir = mkdtempSync(join(tmpdir(), 'gds-ai-reserved-usage-'));
    try {
      const file = join(dir, 'AISearchCard.tsx');
      writeFileSync(file, "const style = { background: 'var(--gds-ai-accent)' };\n");
      const violations = scanForUnsanctionedAiReferences([file], new Set(['AISearchCard.tsx']), dir);
      expect(violations).toHaveLength(0);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('exempts only the exact allowlisted line in a large shared file, not the whole file (issue 700)', () => {
    const dir = mkdtempSync(join(tmpdir(), 'gds-ai-reserved-usage-'));
    try {
      const file = join(dir, 'styles.css');
      writeFileSync(file, 'a { color: red; }\nb { background: var(--gds-ai-gradient); }\nc { color: var(--gds-ai-accent); }\n');
      const allowlist = new Set(['styles.css::b { background: var(--gds-ai-gradient); }']);
      const violations = scanForUnsanctionedAiReferences([file], allowlist, dir);
      expect(violations).toHaveLength(1);
      expect(violations[0]).toMatchObject({ file: 'styles.css', line: 3 });
      expect(violations[0].text).toContain('--gds-ai-accent');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('readAllFiles walks recursively and skips node_modules/dist/__snapshots__', () => {
    const dir = mkdtempSync(join(tmpdir(), 'gds-ai-reserved-usage-walk-'));
    try {
      writeFileSync(join(dir, 'Real.tsx'), 'export const x = 1;\n');
      const skipped = join(dir, 'node_modules');
      mkdirSync(skipped);
      writeFileSync(join(skipped, 'Vendor.tsx'), 'export const y = 1;\n');
      const found = readAllFiles(dir, ['.tsx']);
      expect(found).toContain(join(dir, 'Real.tsx'));
      expect(found).not.toContain(join(skipped, 'Vendor.tsx'));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  // Real-gate dry run: proves the CLI itself, not just the pure function, fails loudly on a
  // genuinely unsanctioned reference — following the same "shell out to the real script, mutate
  // a real file, clean up" pattern as verify-budgets-real.test.mjs.
  it('fails loudly, naming file and line, when a real gds-core file references --gds-ai- outside the allowlist', () => {
    const fixture = join(ROOT, 'packages/gds-core/src/__ai-reserved-usage-dry-run.test-fixture.ts');
    writeFileSync(fixture, "export const notSanctioned = 'var(--gds-ai-accent)';\n");
    try {
      const { exitCode, output } = runGate();
      expect(exitCode).toBe(1);
      expect(output).toContain('__ai-reserved-usage-dry-run.test-fixture.ts:1');
      expect(output).toContain('--gds-ai-accent');
      expect(output).toContain('FAIL');
    } finally {
      unlinkSync(fixture);
    }
  });
});
