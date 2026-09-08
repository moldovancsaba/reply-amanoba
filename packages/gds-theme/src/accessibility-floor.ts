// The accessibility floor: minimums no theme may cross. The axes let a theme control shape,
// density, type, elevation, motion and reaction; this is the contract for which values are
// not available. No warning tier — a floor breach fails the build.
//
// Does not reimplement contrast. `createGdsThemeAccessibilityReport()` already scores colour
// pairs across every preset and scheme; the floor calls it and adopts its findings.

import { createGdsThemeAccessibilityReport } from './accessibility-report';
import { contrastRatio } from './color-math';
import {
  GDS_CONTROL_HEIGHT_EXCEPTIONS,
  GDS_CONTROL_SIZES,
  GDS_MIN_TARGET_PX,
} from './axes';
import { getGdsVibeThemeCssVariables, getGdsVibeThemes } from './vibe-themes';

/**
 * A floor breach is always a build failure.
 *
 * `report` is deliberately NOT a softer breach: it marks an axis the floor has begun *measuring*
 * but has never enforced, where turning the measurement into a gate would retroactively fail
 * presets that predate it. Its findings never fail a build, and a rule may only use it while that
 * remediation is genuinely outstanding — a `report` rule that could be enforced should be
 * enforced. See issue 680 for the first one.
 */
export type GdsFloorSeverity = 'violation' | 'report';

/** Context a rule evaluates against: one preset in one scheme, with its resolved tokens. */
export interface GdsFloorContext {
  presetId: string;
  scheme: 'light' | 'dark';
  /** Every resolved `--gds-*` value for this preset and scheme. */
  tokens: Record<string, string>;
}

/** A single breach, phrased so the fix names the axis field to change. */
export interface GdsFloorViolation {
  ruleId: string;
  presetId: string;
  scheme: 'light' | 'dark';
  actual: string;
  required: string;
  wcag: string;
  fix: string;
  severity: GdsFloorSeverity;
}

/** A machine-checked minimum, tied to the axis that can breach it. */
export interface GdsFloorRule {
  id: string;
  axis: 'shape' | 'density' | 'type' | 'elevation' | 'motion' | 'reaction' | 'color' | 'cross-axis';
  wcag: string;
  rationale: string;
  evaluate(ctx: GdsFloorContext): GdsFloorViolation[];
}

const px = (value: string | undefined): number | null => {
  const m = /^(-?\d*\.?\d+)px$/.exec(String(value ?? '').trim());
  return m ? parseFloat(m[1]) : null;
};

const violation = (
  ctx: GdsFloorContext,
  rule: Pick<GdsFloorRule, 'id' | 'wcag'>,
  actual: string,
  required: string,
  fix: string,
): GdsFloorViolation => ({
  ruleId: rule.id, presetId: ctx.presetId, scheme: ctx.scheme, actual, required, wcag: rule.wcag, fix, severity: 'violation',
});

/** Same shape as {@link violation}, marked `report` so it measures without failing the build. */
const report = (
  ctx: GdsFloorContext,
  rule: Pick<GdsFloorRule, 'id' | 'wcag'>,
  actual: string,
  required: string,
  fix: string,
): GdsFloorViolation => ({
  ruleId: rule.id, presetId: ctx.presetId, scheme: ctx.scheme, actual, required, wcag: rule.wcag, fix, severity: 'report',
});

/**
 * The floor.
 *
 * Every rule is checkable from resolved tokens alone, so the gate runs over all 25 presets in
 * both schemes without a browser. Rules needing real geometry belong to the runtime harness.
 */
export const gdsAccessibilityFloorRules: readonly GdsFloorRule[] = [
  {
    id: 'focus-ring-min-width',
    axis: 'reaction',
    wcag: '2.4.11 Focus Appearance (AA)',
    rationale: 'Below 2px a focus ring is not reliably visible, and a keyboard user has no other indication of where they are.',
    evaluate(ctx) {
      const width = px(ctx.tokens['--gds-focus-ring-width']);
      if (width === null || width >= 2) return [];
      return [violation(ctx, this, `${width}px`, '>= 2px', 'Raise `axes.reaction.focusRing.width` to at least 2px.')];
    },
  },
  {
    id: 'focus-ring-is-not-removed',
    axis: 'reaction',
    wcag: '2.4.7 Focus Visible (AA)',
    rationale: 'A theme must not be able to erase focus indication by setting the ring to none or transparent.',
    evaluate(ctx) {
      const style = String(ctx.tokens['--gds-focus-ring-style'] ?? '').trim();
      const colour = String(ctx.tokens['--gds-focus-ring-color'] ?? '').trim();
      const out: GdsFloorViolation[] = [];
      if (style === 'none' || style === 'hidden') {
        out.push(violation(ctx, this, style, 'solid | dashed | double', 'Set `axes.reaction.focusRing.style` to a visible style.'));
      }
      if (/transparent/i.test(colour)) {
        out.push(violation(ctx, this, colour, 'a visible colour role', 'Point `axes.reaction.focusRing.colorRole` at a visible semantic role.'));
      }
      return out;
    },
  },
  {
    id: 'control-height-min-target',
    axis: 'density',
    wcag: '2.5.5 Target Size (AAA) / 2.5.8 Target Size Minimum (AA)',
    rationale: 'A primary control smaller than 44px is unpleasant to hit on a phone and fails the stricter target-size guidance GDS holds itself to.',
    evaluate(ctx) {
      const out: GdsFloorViolation[] = [];
      for (const size of GDS_CONTROL_SIZES) {
        if (GDS_CONTROL_HEIGHT_EXCEPTIONS[size]) continue;   // recorded exception
        const height = px(ctx.tokens[`--gds-control-height-${size}`]);
        if (height === null || height >= GDS_MIN_TARGET_PX) continue;
        out.push(violation(ctx, this, `${size} = ${height}px`, `>= ${GDS_MIN_TARGET_PX}px`,
          `Raise \`axes.density.controlHeights.${size}\`, or record an exception if this size is genuinely not a primary target.`));
      }
      return out;
    },
  },
  {
    id: 'body-line-height-min',
    axis: 'type',
    wcag: '1.4.12 Text Spacing (AA)',
    rationale: 'Body text below 1.5 line-height is materially harder to read for users with dyslexia or low vision.',
    evaluate(ctx) {
      const lh = Number(ctx.tokens['--gds-line-height-md']);
      if (!Number.isFinite(lh) || lh >= 1.5) return [];
      return [violation(ctx, this, String(lh), '>= 1.5', 'Raise `axes.type.lineHeights.md` to at least 1.5.')];
    },
  },
  {
    id: 'motion-duration-bounded',
    axis: 'motion',
    wcag: '2.2.2 Pause, Stop, Hide (A)',
    rationale: 'A transition longer than two seconds is an animation the user cannot skip.',
    evaluate(ctx) {
      const out: GdsFloorViolation[] = [];
      for (const [name, value] of Object.entries(ctx.tokens)) {
        if (!name.startsWith('--gds-motion-duration-')) continue;
        const ms = /^(\d*\.?\d+)ms$/.exec(String(value).trim());
        if (!ms || parseFloat(ms[1]) <= 2000) continue;
        out.push(violation(ctx, this, `${name} = ${value}`, '<= 2000ms',
          `Lower \`axes.motion.durations.${name.replace('--gds-motion-duration-', '')}\`.`));
      }
      return out;
    },
  },
  {
    id: 'reduced-motion-not-overridden',
    axis: 'motion',
    wcag: '2.3.3 Animation from Interactions (AAA)',
    rationale: 'A theme may make motion calmer than the user asked for; it may never make it louder.',
    evaluate(ctx) {
      const policy = String(ctx.tokens['--gds-motion-policy'] ?? 'system').trim();
      if (['system', 'reduce', 'no-motion'].includes(policy)) return [];
      return [violation(ctx, this, policy, 'system | reduce | no-motion',
        'Remove the policy override; there is no value that ignores a reduced-motion preference.')];
    },
  },
  {
    id: 'badge-tone-pairs-legible',
    axis: 'color',
    wcag: '1.4.3 Contrast (Minimum) (AA)',
    rationale: 'A status badge is often the only signal that something needs attention, so an illegible one is a functional failure rather than a cosmetic one.',
    evaluate(ctx) {
      const out: GdsFloorViolation[] = [];
      for (const lane of ['soft', 'solid'] as const) {
        for (const tone of ['success', 'warning', 'danger', 'info', 'neutral'] as const) {
          const bg = ctx.tokens[`--gds-badge-${lane}-${tone}`];
          const fg = ctx.tokens[`--gds-badge-${lane}-${tone}-fg`];
          if (!bg || !fg) {
            out.push(violation(ctx, this, `${lane}-${tone} missing`, 'both halves emitted',
              'Emit the pair from emitBadgeToneCssVariables; a half-emitted pair renders an unverifiable badge.'));
            continue;
          }
          const ratio = contrastRatio(fg, bg, bg);
          if (ratio === null || ratio < 4.5) {
            out.push(violation(ctx, this, `${lane}-${tone} = ${ratio ?? 'unresolvable'}:1`, '>= 4.5:1',
              'The foreground is derived against the background; if this fails, the derivation or the state colour changed.'));
          }
        }
      }
      return out;
    },
  },
  {
    id: 'primary-cta-text-contrast',
    axis: 'color',
    wcag: '1.4.3 Contrast (Minimum) (AA)',
    rationale:
      'The primary call to action is the one control a page most expects to be pressed, and its label is normal-weight text. Nothing else in GDS measured this pairing: createBrandTheme\'s assertContrast gates text on page/surface/inverse and derives a readable foreground for support, and no floor rule covered the action fill, so a preset could ship a CTA below AA and every gate stayed silent (issue 680).',
    evaluate(ctx) {
      const fill = ctx.tokens['--gds-vibe-primary'];
      if (!fill) return [];
      // The pairing that actually renders: the governed Button rule paints
      // `background: var(--gds-vibe-primary)` with `color: var(--mantine-color-white)`.
      const ratio = contrastRatio('#ffffff', fill, fill);
      if (ratio === null || ratio >= 4.5) return [];
      return [report(ctx, this, `white on ${fill} = ${ratio}:1`, '>= 4.5:1',
        'Darken the preset\'s `primary` vibe colour, or point the CTA at an in-palette tone that clears AA. Reported, not enforced: presets predating this rule have never been measured on this axis.')];
    },
  },
  {
    id: 'ai-accent-text-contrast',
    axis: 'color',
    wcag: '1.4.3 Contrast (Minimum) (AA)',
    rationale:
      'The reserved ai.accent role (issue 697) is a non-text, sub-brand identity colour — the Your Field handoff\'s Scout orange, #ff6b35 — never a general text or action fill. White on it measures 2.84:1, clearing neither AA text threshold. This rule measures and prints that number every run so it stays derived (Rule 14), never retyped as prose, without treating a non-text lane as a text-contrast failure it never claimed to pass.',
    evaluate(ctx) {
      const fill = ctx.tokens['--gds-ai-accent'];
      if (!fill) return [];
      const ratio = contrastRatio('#ffffff', fill, fill);
      if (ratio === null || ratio >= 4.5) return [];
      return [report(ctx, this, `white on ${fill} = ${ratio}:1`, '>= 4.5:1',
        'Reserved non-text sub-brand lane; gradient-filled controls carry >=14px/600 text per the lane rule (THEME_GOVERNANCE.md), not this fill alone.')];
    },
  },
  {
    id: 'outline-accent-text-contrast',
    axis: 'color',
    wcag: '1.4.3 Contrast (Minimum) (AA)',
    rationale:
      'SemanticButton\'s `outline-accent` brand intent (issue 700) is the first component that renders an accent as label text on the page — precisely the condition accent-axis.ts\'s `outline` mode enforcement entry names ("No component renders an accent as text on the page today"). That entry governs a different, unrelated mechanism (the 10-name categorical GdsAccentAxis ramp GdsIconBadge draws from), not this single brand-accent role, so it is deliberately left untouched (THEME_GOVERNANCE.md) — this rule is the purpose-built measurement for the pairing that actually renders: `--gds-brand-accent-action` as both the label and the 1.5px stroke, on the page surface a transparent-fill button sits on. Measured across all 27 presets x 2 schemes: every light-scheme value clears 4.5:1 (the generic derivation path ensureContrasts it against canvasLight in semantic-token-source.ts), but 25 of 27 dark-scheme values do not, because that same derivation reuses the light-derived value unchanged in dark mode rather than re-deriving against canvasDark — a real, pre-existing gap this rule surfaces rather than silently enforcing away (class-usa and gold-athlete\'s bespoke emission paths pass in both schemes). Reported, not enforced, until that gap is closed at the source.',
    evaluate(ctx) {
      const fill = ctx.tokens['--gds-brand-accent-action'];
      const page = ctx.tokens['--gds-bg-page'];
      if (!fill || !page) return [];
      const ratio = contrastRatio(fill, page, page);
      if (ratio === null || ratio >= 4.5) return [];
      return [report(ctx, this, `${fill} on ${page} = ${ratio}:1`, '>= 4.5:1',
        'Re-derive `--gds-brand-accent-action` against this scheme\'s own canvas (semantic-token-source.ts currently ensures it against canvasLight only, then reuses that value unchanged for dark) rather than the light-derived value.')];
    },
  },
  {
    id: 'disabled-control-still-distinguishable',
    axis: 'color',
    wcag: '1.4.1 Use of Color (A)',
    rationale: 'A disabled control whose text and background are the same value is invisible rather than merely muted.',
    evaluate(ctx) {
      const fg = ctx.tokens['--gds-control-disabledText'];
      const bg = ctx.tokens['--gds-control-disabledBg'];
      if (!fg || !bg || fg !== bg) return [];
      return [violation(ctx, this, `${fg} on ${bg}`, 'foreground != background',
        'Give `control.disabledText` and `control.disabledBg` different values in the semantic token source.')];
    },
  },
];

/** Evaluates every floor rule against one preset/scheme. */
export function validateGdsAccessibilityFloor(ctx: GdsFloorContext): GdsFloorViolation[] {
  return gdsAccessibilityFloorRules.flatMap((rule) => rule.evaluate(ctx));
}

/**
 * Evaluates the floor across every preset and scheme, and folds in the colour-contrast
 * findings from the existing accessibility report rather than re-deriving them.
 */
export function auditGdsAccessibilityFloor(): {
  presetsChecked: number;
  rulesEvaluated: number;
  violations: GdsFloorViolation[];
  /** Findings from `report`-severity rules: measured, never build-failing (issue 680). */
  reports: GdsFloorViolation[];
} {
  const presets = getGdsVibeThemes();
  if (!presets.length) throw new Error('No presets found — the accessibility floor cannot pass vacuously.');
  if (!gdsAccessibilityFloorRules.length) throw new Error('Empty floor rule set — refusing to pass.');

  const violations: GdsFloorViolation[] = [];
  for (const { id } of presets) {
    for (const scheme of ['light', 'dark'] as const) {
      violations.push(...validateGdsAccessibilityFloor({ presetId: id, scheme, tokens: getGdsVibeThemeCssVariables(id, scheme) }));
    }
  }

  // Contrast is not recomputed here; the report already scores every colour pair across
  // every preset and scheme.
  for (const finding of createGdsThemeAccessibilityReport().findings) {
    if (finding.severity !== 'blocking') continue;
    violations.push({
      ruleId: 'color-contrast-floor',
      presetId: finding.themeId,
      scheme: finding.mode,
      actual: `${finding.ratio}:1 (${finding.foreground} on ${finding.background})`,
      required: `>= ${finding.minimumRatio}:1`,
      wcag: '1.4.3 Contrast (Minimum) (AA)',
      fix: `Adjust the ${finding.role} pair in the semantic token source.`,
      severity: 'violation',
    });
  }

  return {
    presetsChecked: presets.length * 2,
    rulesEvaluated: gdsAccessibilityFloorRules.length,
    violations: violations.filter((v) => v.severity !== 'report'),
    reports: violations.filter((v) => v.severity === 'report'),
  };
}

/** Human-readable spec of the floor, used to keep the docs in step with the rules. */
export function describeGdsAccessibilityFloor(): string {
  return gdsAccessibilityFloorRules
    .map((r) => `- **${r.id}** (${r.axis} axis, ${r.wcag}) — ${r.rationale}`)
    .join('\n');
}
