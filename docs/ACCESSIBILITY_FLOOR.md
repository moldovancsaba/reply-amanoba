# Accessibility floor

Minimums no GDS theme may cross, enforced by `npm run verify:a11y-floor` across every preset
in both colour schemes.

**There is no warning tier.** A floor breach fails the build. A warning would make the floor
advisory, and an advisory floor is not a floor.

This file is GENERATED from `packages/gds-theme/src/accessibility-floor.ts`. A floor described
differently from how it is checked is a floor nobody can rely on — run `npm run docs:a11y-floor`.

## Rules (11)

- **focus-ring-min-width** (reaction axis, 2.4.11 Focus Appearance (AA)) — Below 2px a focus ring is not reliably visible, and a keyboard user has no other indication of where they are.
- **focus-ring-is-not-removed** (reaction axis, 2.4.7 Focus Visible (AA)) — A theme must not be able to erase focus indication by setting the ring to none or transparent.
- **control-height-min-target** (density axis, 2.5.5 Target Size (AAA) / 2.5.8 Target Size Minimum (AA)) — A primary control smaller than 44px is unpleasant to hit on a phone and fails the stricter target-size guidance GDS holds itself to.
- **body-line-height-min** (type axis, 1.4.12 Text Spacing (AA)) — Body text below 1.5 line-height is materially harder to read for users with dyslexia or low vision.
- **motion-duration-bounded** (motion axis, 2.2.2 Pause, Stop, Hide (A)) — A transition longer than two seconds is an animation the user cannot skip.
- **reduced-motion-not-overridden** (motion axis, 2.3.3 Animation from Interactions (AAA)) — A theme may make motion calmer than the user asked for; it may never make it louder.
- **badge-tone-pairs-legible** (color axis, 1.4.3 Contrast (Minimum) (AA)) — A status badge is often the only signal that something needs attention, so an illegible one is a functional failure rather than a cosmetic one.
- **primary-cta-text-contrast** (color axis, 1.4.3 Contrast (Minimum) (AA)) — The primary call to action is the one control a page most expects to be pressed, and its label is normal-weight text. Nothing else in GDS measured this pairing: createBrandTheme's assertContrast gates text on page/surface/inverse and derives a readable foreground for support, and no floor rule covered the action fill, so a preset could ship a CTA below AA and every gate stayed silent (issue 680).
- **ai-accent-text-contrast** (color axis, 1.4.3 Contrast (Minimum) (AA)) — The reserved ai.accent role (issue 697) is a non-text, sub-brand identity colour — the Your Field handoff's Scout orange, #ff6b35 — never a general text or action fill. White on it measures 2.84:1, clearing neither AA text threshold. This rule measures and prints that number every run so it stays derived (Rule 14), never retyped as prose, without treating a non-text lane as a text-contrast failure it never claimed to pass.
- **outline-accent-text-contrast** (color axis, 1.4.3 Contrast (Minimum) (AA)) — SemanticButton's `outline-accent` brand intent (issue 700) is the first component that renders an accent as label text on the page — precisely the condition accent-axis.ts's `outline` mode enforcement entry names ("No component renders an accent as text on the page today"). That entry governs a different, unrelated mechanism (the 10-name categorical GdsAccentAxis ramp GdsIconBadge draws from), not this single brand-accent role, so it is deliberately left untouched (THEME_GOVERNANCE.md) — this rule is the purpose-built measurement for the pairing that actually renders: `--gds-brand-accent-action` as both the label and the 1.5px stroke, on the page surface a transparent-fill button sits on. Measured across all 27 presets x 2 schemes: every light-scheme value clears 4.5:1 (the generic derivation path ensureContrasts it against canvasLight in semantic-token-source.ts), but 25 of 27 dark-scheme values do not, because that same derivation reuses the light-derived value unchanged in dark mode rather than re-deriving against canvasDark — a real, pre-existing gap this rule surfaces rather than silently enforcing away (class-usa and gold-athlete's bespoke emission paths pass in both schemes). Reported, not enforced, until that gap is closed at the source.
- **disabled-control-still-distinguishable** (color axis, 1.4.1 Use of Color (A)) — A disabled control whose text and background are the same value is invisible rather than merely muted.

## What is not here

Colour contrast is enforced, but not by these rules: `createGdsThemeAccessibilityReport()`
already scores every colour pair across every preset and scheme, and the floor adopts its
blocking findings. A second contrast implementation could disagree with the first, and two
accessibility verdicts on one pair is worse than one.

Rules needing real rendered geometry belong to the runtime harness rather than this token-level
gate. A rule that cannot be evaluated is worse than a missing rule, because it looks like
coverage.
