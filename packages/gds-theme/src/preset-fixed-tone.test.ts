import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// Issue 724. An element carrying data-gds-fixed-tone is excluded from every theme-preset
// repaint rule, in the rule's own selector, at zero specificity. jsdom's selector engine
// cannot evaluate :where(), so this is a static contract over the shipped stylesheet; the
// matching behaviour itself is a browser fact (Baseline 2021).

const CLAUSE = ':where(:not([data-gds-fixed-tone]))';
const EXEMPT_MEDIA = [/^@media \(forced-colors: active\)/, /^@media \(prefers-reduced-motion: reduce\)/];

const stylesCss = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), '..', 'styles.css'), 'utf8');

interface GatedSelector {
  line: number;
  selector: string;
  subject: string;
  exemptMedia: boolean;
}

function subjectOf(selector: string) {
  let depth = 0;
  let split = -1;
  for (let i = 0; i < selector.length; i += 1) {
    const ch = selector[i];
    if (ch === '(' || ch === '[') depth += 1;
    else if (ch === ')' || ch === ']') depth -= 1;
    else if (ch === ' ' && depth === 0) split = i;
  }
  return selector.slice(split + 1);
}

function collectGatedSelectors(css: string): GatedSelector[] {
  const found: GatedSelector[] = [];
  let depth = 0;
  let exemptStartDepth = -1;
  css.split('\n').forEach((raw, index) => {
    const line = raw.trim();
    if (exemptStartDepth < 0 && EXEMPT_MEDIA.some((re) => re.test(line))) exemptStartDepth = depth;
    const opens = (line.match(/{/g) ?? []).length;
    const closes = (line.match(/}/g) ?? []).length;
    depth += opens - closes;
    const exemptMedia = exemptStartDepth >= 0;
    if (exemptMedia && closes && depth <= exemptStartDepth) {
      exemptStartDepth = -1;
      return;
    }
    if (!/^html\[/.test(line) || !line.includes('data-gds-theme-preset')) return;
    const match = line.match(/^(.*?)(?:\s*\{|,)$/);
    if (!match) return;
    found.push({ line: index + 1, selector: match[1], subject: subjectOf(match[1]), exemptMedia });
  });
  return found;
}

const gated = collectGatedSelectors(stylesCss);
const covered = gated.filter((entry) => !entry.exemptMedia && !/^body\b/.test(entry.subject));
const bodyRules = gated.filter((entry) => !entry.exemptMedia && /^body\b/.test(entry.subject));
const accessibilityResets = gated.filter((entry) => entry.exemptMedia);

describe('data-gds-fixed-tone preset opt-out (#724)', () => {
  it('parses a non-trivial gated selector set in every category, so the assertions below cannot pass vacuously', () => {
    expect(covered.length).toBeGreaterThan(100);
    expect(bodyRules.length).toBeGreaterThan(0);
    expect(accessibilityResets.length).toBeGreaterThan(0);
  });

  it('excludes the attribute on the subject of every preset-gated rule outside the accessibility blocks', () => {
    // Strips one trailing pseudo-element (`::placeholder`) or pseudo-class (`:hover`/`:active`,
    // issue 700's outline-accent/gradient state rules) after the fixed-tone clause -- a
    // pseudo-class must follow every other simple selector on a compound selector, including
    // `:where()`, so the clause itself still gates the base element either way.
    const missing = covered.filter(({ subject }) => !subject.replace(/:{1,2}[a-z-]+$/, '').endsWith(CLAUSE));
    expect(missing.map(({ line, selector }) => `${line}: ${selector}`)).toEqual([]);
  });

  it('never opts body out: its rules publish the text/dimmed custom properties the page depends on', () => {
    const wrong = bodyRules.filter(({ selector }) => selector.includes('data-gds-fixed-tone'));
    expect(wrong.map(({ line, selector }) => `${line}: ${selector}`)).toEqual([]);
  });

  it('keeps the forced-colors and reduced-motion resets applying to opted-out elements', () => {
    const wrong = accessibilityResets.filter(({ selector }) => selector.includes('data-gds-fixed-tone'));
    expect(wrong.map(({ line, selector }) => `${line}: ${selector}`)).toEqual([]);
  });

  it('covers every surface the consumer inventory names, and the element rules behind them', () => {
    const hooks = [
      '.gds-paper',
      '.gds-card',
      '.mantine-Button-root',
      '.mantine-Popover-dropdown',
      '.mantine-AppShell-navbar',
      '.mantine-AppShell-header',
      '.mantine-AppShell-footer',
      '.mantine-AppShell-main',
      '.mantine-Checkbox-input',
      '.mantine-Input-input',
      '.mantine-NativeSelect-input',
      '.mantine-Textarea-input',
      '.mantine-Badge-root',
      '.mantine-Anchor-root',
      '.mantine-List-root',
      '.mantine-Table-root',
      '.mantine-Menu-item',
      '.mantine-Radio-label',
      '.mantine-Switch-label',
    ];
    const uncovered = hooks.filter((hook) => !covered.some(({ subject }) => subject.startsWith(hook)));
    expect(uncovered).toEqual([]);
    expect(covered.some(({ subject }) => subject === `a${CLAUSE}`)).toBe(true);
    expect(covered.some(({ subject }) => subject === `h1${CLAUSE}`)).toBe(true);
  });

  it('uses :where() only, so no gated rule gained specificity', () => {
    expect(stylesCss).not.toMatch(/(?<!:where\():not\(\[data-gds-fixed-tone\]\)/);
  });

  it('keeps the component-specific badge attribute working alongside the generic one', () => {
    expect(stylesCss).toContain(`.mantine-Badge-root:not([data-gds-badge-fixed-tone])${CLAUSE}`);
  });
});
