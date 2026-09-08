import { describe, expect, it } from 'vitest';
import {
  auditGdsAccessibilityFloor, describeGdsAccessibilityFloor,
  gdsAccessibilityFloorRules, validateGdsAccessibilityFloor,
} from './accessibility-floor';
import { getGdsVibeThemeCssVariables, getGdsVibeThemes } from './vibe-themes';

const base = getGdsVibeThemeCssVariables('default', 'light');
const probe = (over: Record<string, string>) =>
  validateGdsAccessibilityFloor({ presetId: 'probe', scheme: 'light', tokens: { ...base, ...over } });

describe('accessibility floor (issue 559)', () => {
  it('has no advisory tier — every breach is a violation', () => {
    // A warning tier would make the floor advisory, and an advisory floor is not a floor.
    const found = probe({ '--gds-focus-ring-width': '1px' });
    expect(found).toHaveLength(1);
    expect(found[0].severity).toBe('violation');
  });

  it.each([
    ['focus-ring-min-width', { '--gds-focus-ring-width': '1px' }],
    ['focus-ring-is-not-removed', { '--gds-focus-ring-style': 'none' }],
    ['focus-ring-is-not-removed', { '--gds-focus-ring-color': 'transparent' }],
    ['control-height-min-target', { '--gds-control-height-md': '30px' }],
    ['body-line-height-min', { '--gds-line-height-md': '1.2' }],
    ['motion-duration-bounded', { '--gds-motion-duration-slow': '5000ms' }],
    ['reduced-motion-not-overridden', { '--gds-motion-policy': 'never-reduce' }],
    ['disabled-control-still-distinguishable', { '--gds-control-disabledText': '#cccccc', '--gds-control-disabledBg': '#cccccc' }],
    ['outline-accent-text-contrast', { '--gds-brand-accent-action': '#ffee00' }],
  ])('rule %s fires on its own breach', (ruleId, tokens) => {
    expect(probe(tokens).map((v) => v.ruleId)).toContain(ruleId);
  });

  it('outline-accent-text-contrast is report severity, not a build-failing violation (issue 700)', () => {
    const found = probe({ '--gds-brand-accent-action': '#ffee00' });
    const finding = found.find((v) => v.ruleId === 'outline-accent-text-contrast');
    expect(finding).toBeTruthy();
    expect(finding!.severity).toBe('report');
  });

  it('outline-accent-text-contrast stays silent on the real default/light pairing (issue 700)', () => {
    expect(probe({}).map((v) => v.ruleId)).not.toContain('outline-accent-text-contrast');
  });

  it('every rule names the axis field to change, not just the failure', () => {
    // A violation a reader cannot act on is a nuisance rather than a gate.
    for (const v of probe({ '--gds-focus-ring-width': '1px', '--gds-line-height-md': '1.2' })) {
      expect(v.fix).toMatch(/axes\.|semantic token source/);
      expect(v.wcag).toMatch(/^\d\.\d/);
    }
  });

  it('every rule carries a WCAG reference and a rationale', () => {
    for (const rule of gdsAccessibilityFloorRules) {
      expect(rule.wcag).toMatch(/^\d\.\d/);
      expect(rule.rationale.length).toBeGreaterThan(30);
    }
  });

  it('clears every shipped preset in both schemes', () => {
    const { violations, presetsChecked } = auditGdsAccessibilityFloor();
    expect(violations).toEqual([]);
    expect(presetsChecked).toBe(getGdsVibeThemes().length * 2);
  });

  it('describes itself from the rules, so the docs cannot drift', () => {
    const described = describeGdsAccessibilityFloor();
    for (const rule of gdsAccessibilityFloorRules) expect(described).toContain(rule.id);
  });
});
