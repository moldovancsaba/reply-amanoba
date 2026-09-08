# Theme Governance

Status: Active SSOT
Version: 6.7.0
Last updated: 2026-09-07

This document defines the approved adopter-facing theme lanes for products that need branding without creating a second design authority.

## Base rule

- `gdsTheme` is the only shared token authority. Its **default semantic-role token layer** (`--gds-bg-*`, `--gds-text-*`, `--gds-border-card`, `--gds-text-on-inverse`) is defined at `:root` in `styles.css` so every surface resolves one governed value instead of a per-call-site fallback — see [`docs/SEMANTIC_ROLE_TOKENS.md`](docs/SEMANTIC_ROLE_TOKENS.md) for the values, the per-token-pair WCAG AA contrast contract (policed by `verify:token-contrast-scoring`), and the preset/brand override precedence.
- Adopters must use one of the approved theme lanes:
  - `gdsTheme`
  - `gdsDarkPublicTheme`
  - `gdsFlatSurfaceTheme`
  - `gdsEditorialPublicTheme`
  - `createPublicBrandTheme(...)`
- Products may not fork the shared theme into a permanent parallel token system.
- Public and operator accent surfaces must resolve from shared semantic contracts such as `AccentPanel`, not product-local `light-dark(...)` patches or raw `*.0` shade assumptions.
- `extendGdsTheme(...)` is no longer a canonical adopter path. It remains temporarily exported only for bounded internal/runtime composition inside GDS-controlled implementation.

## Allowed extension surfaces

- primary color and semantic brand palette
- typography family where product identity or locale coverage requires it
- shell defaults for dark or light products
- component default props when they remain compatible with shared interaction meaning
- narrow `theme.other` tokens for non-Mantine rendering surfaces such as email, OG images, or certificates

## Not allowed

- changing shared interaction meaning through theme overrides
- declaring a second token layer as the real authority while `gdsTheme` remains nominal
- product-specific page styling that bypasses the theme for repeated surfaces

## White-label and tenant theming

White-label or tenant theming is allowed only when:

- the base product still resolves from `gdsTheme`
- tenant overrides remain scoped to documented brand surfaces
- contrast, readability, and focus states still meet the shared baseline
- switching tenants does not introduce a second runtime provider authority

## Identity provider branding policy

Identity providers are part of the same governance envelope as theme authority. Adopters that render social auth must use `SocialAuthButtons` and pass providers from the approved policy list in `gds-adoption.json`.

- approved providers are declared under `compliance.identityProviderBranding.approvedProviders`
- forbidden customizations are declared under `compliance.identityProviderBranding.forbiddenCustomizations`
- allowed visual variants are declared under `compliance.identityProviderBranding.allowedVariants`
- minimum accessible touch target is declared under `compliance.identityProviderBranding.minTouchTargetPx`
- color authority is declared under `compliance.identityProviderBranding.colorAuthority`

Allowed policy:

- keep visual identity within policy-approved SocialAuth behavior
- never implement local third-party-branded auth controls that bypass GDS action semantics
- do not mutate provider icon/mark, loading, disabled, or label mechanics via per-product wrappers unless approved in policy
- use `getSupportedProviderIdentityIds()` and `getProviderIdentityPolicy(provider)` when a consumer needs runtime audit or logging metadata
- represent tenant-disabled and provider-error states through the shipped provider props, not local disabled button wrappers

Recommended model:

1. start from the closest shipped lane
2. use `createPublicBrandTheme(...)` when a branded public product needs governed overrides
3. apply tenant-level overrides only on documented brand surfaces and only through the approved lane

For public/editorial products that want one sanctioned entrypoint instead of ad hoc merging, use `createPublicBrandTheme({ editorialSerif, flatSurfaces, overrides })` from `@sovereignsquad/gds-theme`.

## Creator-authored experience theming

Some products need a bounded creator-, editor-, or customer-authored visual canvas. GDS allows this only as a narrow experience override lane, never as a second app-wide theme authority.

Ownership boundary:

- GDS owns app chrome, navigation, shells, shared controls, consent surfaces, legal rows, recovery states, and system messaging.
- The adopting product owns storage, moderation, sanitization, and publish flow for creator-authored presentation data.
- Creator-authored overrides may own only the approved experience canvas.

Allowed override modes:

- scoped class hooks on a bounded canvas root
- scoped CSS injected after base experience styles
- creator-authored media, colors, and decorative presentation inside the approved canvas

Not allowed:

- replacing `PublicShell`, `PublicNav`, `DocsPageShell`, or other GDS-owned chrome
- hiding required consent, legal, or recovery controls
- redefining shared action semantics, focus handling, or system state meaning
- unbounded CSS that leaks into the surrounding page
- treating creator CSS as a second theme authority for the full product

Required render order:

1. render GDS shell and system-owned controls
2. render the approved creator canvas
3. apply creator-scoped overrides after base experience styles only inside that canvas
4. fall back safely to the base GDS presentation if overrides are missing or invalid

Required documentation path:

- declare the exception in `gds-adoption.json` with category `product-authored-experience`
- keep `scope` narrow to the actual canvas files
- include `a11yRequirements`, `testingRequirements`, and `observabilityRequirements`
- describe what shared controls must remain governed outside the canvas

Recommended implementation shape:

```ts
type ExperienceThemeOverrideMode = 'none' | 'css-class' | 'scoped-css';

type ExperienceOverrideContract = {
  mode: ExperienceThemeOverrideMode;
  scopeId: string;
  renderOrder: 'after-base-experience-styles';
  mayOverride: string[];
  mustNotOverride: string[];
};
```

This is a governance contract first. Products still own storage and moderation, but they may not use that as justification for replacing GDS-owned application structure.

## Atmosphere has a scale

A preset's atmosphere is published at **two** scales; using the wrong one is a defect.

| Token | Composed for | Use it for |
| --- | --- | --- |
| `--gds-vibe-hero` | a full-width band | page washes, hero sections, anything hundreds of pixels wide |
| `--gds-vibe-swatch` | a small box | swatches, legend dots, chips, preview tiles — any surface previewing a theme at small size |

`--gds-vibe-hero` is a `linear-gradient(135deg, …)` — a barely-perceptible wash across a hero
band, but a hard diagonal when cropped into a small box (a long gradient sampled through a
small window, not a scale-independent ramp). `--gds-vibe-swatch` is a radial from the centre,
which reads identically at any size since there's no axis for a small box to crop differently.
It's derived from the same `primary`/`accent` the vibe is built from, mixed against the
scheme's own surface, so it can't drift from the preset it describes.

A `flatSurfaces` brand lane (Class USA, Gold Athlete) has no atmosphere by definition and gets
a flat tint instead of an invented gradient — same reasoning that neutralises `glow` and
`gradient` for those lanes.

### Consuming it

Apply `[data-gds-theme-swatch]` and the governed value is used for you:

```html
<span data-gds-theme-swatch></span>
```

Or read the token directly. A component previewing a preset **other** than the active one asks
`getGdsVibeThemeCssVariables(preset, scheme)` for that preset's values, which is what
`VibeThemePicker` does for all 25 swatches — the ambient `var()` describes the active theme only.

## Font lanes must cover every supported language

Only a font stack that supports 100% of the languages GDS ships may be a font lane — no
partial lane, no lane covering "most" scripts. A lane that covers some languages silently
determines which languages a product can display, with nothing said at the point of choice —
and pushes every consuming app to rediscover and solve that locally.

### The contract

- Every lane's stack ends with the **universal script fallback** — one Noto family per script
  in the locale catalog (`Noto Sans`, `Noto Sans Hebrew`, `Noto Sans Arabic`, `Noto Sans SC`,
  `Noto Sans JP`, `Noto Sans KR`). The lane's own display face still leads, so Latin text keeps
  the lane's character; the browser only reaches a Noto entry for glyphs the display face
  lacks.
- Every lane declares `localeCoverage` of the **entire** locale catalog. A lane that cannot is
  not a lane.
- The script list is **derived** from `gdsLocaleMetadata` via `getGdsLocaleScripts()`, never
  written out. Adding a locale in a new script makes the font map incomplete and **fails the
  build** rather than shipping a lane that cannot draw it.
- Fonts load from `fonts.googleapis.com` only, always with `display=swap`.

### Enforcement

`npm run verify:font-lane-coverage`, in the release chain. It derives both sides from the same
catalog, so it cannot be satisfied by writing a list.

**What it does not prove, stated plainly:** it verifies structure — that each script has a
declared family, that each lane's stack names it, that coverage is the whole catalog, and that
loading is from the approved host with `swap`. It does not read the font binaries, so it cannot
detect a family mapped to a script whose glyphs it does not contain. That mapping is small,
lives in one place, and every entry is commented; proving real glyph coverage would require
reading each family's `unicode-range` from the font service live, which is the kind of
network-dependent assertion already exempted elsewhere in the chain.

## Dark-mode rule

- a product may default to dark when that is part of its deliberate shell identity
- dark products must still provide readable tokens for text, paper, card, alert, table, and link surfaces
- mixed-mode islands remain exceptions, not the default layout strategy
- preset styles must set `--mantine-color-text` and `--mantine-color-dimmed` from `--gds-vibe-text` and `--gds-vibe-muted` on body, shell, card, and paper surfaces so nested Mantine components cannot keep stale light-mode foregrounds on dark backgrounds
- dark and dark-forward VibeTheme controls must use `--gds-vibe-control` and `--gds-vibe-control-text` for inputs, default buttons, and code-like surfaces rather than assuming the base Mantine default variant remains readable
- mixed-preview surfaces, such as the Theme Lab shipped-lane gallery and VibeTheme contract preview, must use `data-gds-local-contrast` plus local `--gds-vibe-*`, Mantine foreground variables, local control tokens, and a local radius token when they intentionally render a light preview card inside a dark page

## Theme trust hardening

Owned contrast is a first-class contract. A surface that intentionally renders with a different readability envelope than the surrounding page must not rely on ambient page colors or ad hoc route-local `Paper` styling.

Required split of responsibilities:

- `BoundedPreviewSurface` owns preview isolation. It prevents nested shell demos from escaping their frame and painting over the page.
- `getGdsOwnedContrastProps(...)` owns mixed-surface readability. It marks the surface with `data-gds-owned-contrast` and `data-gds-local-contrast`, then applies the package-owned `--gds-local-background`, `--gds-local-radius`, `--gds-vibe-*`, and control-text tokens.
- These are separate contracts. Preview isolation does not replace owned contrast, and owned contrast does not replace preview isolation.

Required runtime behavior:

- any GDS-controlled route that renders a live shell inside documentation must use `BoundedPreviewSurface`
- any GDS-controlled route that renders a light card inside a dark shell, a dark control cluster inside a light shell, or any other mixed readability island must use `getGdsOwnedContrastProps(...)`
- consumers may not declare `data-gds-owned-contrast` or `data-gds-local-contrast` directly in product-local route code
- package-owned controlled surfaces must keep local control tokens for buttons, inputs, selects, code blocks, badges, labels, and dimmed copy

Release blocking policy:

- `npm run verify:theme-trust-runtime` is mandatory for release verification
- `npm run verify:forced-colors-runtime` is mandatory for release verification
- source-level owned-contrast compliance must fail if route code declares owned-contrast markers directly instead of using the package helper
- if a route-level preview or mixed-theme surface fails owned contrast or preview isolation, rollback to the previous stable release line and keep the board item open
- exceptions require a documented package-owned helper or primitive, never a route-local style patch

Forced-colors contract:

- `@media (forced-colors: active)` must replace decorative gradients with system-backed canvas/control colors
- controls must resolve from `ButtonFace` / `ButtonText`
- disabled states must resolve from `GrayText`
- selected/active states must resolve from `Highlight` / `HighlightText`
- focus indicators must stay visibly outlined in forced-colors mode
- this contract binds **every** theme lane, including the expressive vibe/brand presets (`cosmic`, `neon-night`, …) whose `!important` gradients must not out-specify the forced-colors reset; a specificity backstop in `styles.css` guarantees this
- runtime acceptance requires the browser-level `verify:forced-colors-runtime` gate, not only static CSS review — which now sweeps the new-component pattern routes across 8 presets (including the vibe lanes) so a gradient-leak regression in any lane is caught

## Opting one element out of the preset repaint (`data-gds-fixed-tone`, issue #724)

The active theme preset repaints surfaces through `html[data-gds-theme-preset] …` rules in
`styles.css`: `.gds-paper`/`.gds-card`, `.mantine-Button-root`, `.mantine-Popover-dropdown`,
the `AppShell` header/navbar/footer/main, inputs, checkboxes, links, headings, dimmed text, and
the expressive vibe lanes' own variants of each. Several of those rules are `!important`. That
repaint is what makes an unstyled app coherent under any preset, and it stays. An element that
carries its own intentional styling opts out by setting `data-gds-fixed-tone` on that element:

```tsx
<Paper data-gds-fixed-tone bg="navy.2">…</Paper>
<Button data-gds-fixed-tone variant="outline">…</Button>
<Popover.Dropdown data-gds-fixed-tone>…</Popover.Dropdown>
```

Contract:

- Every preset-gated rule outside the `forced-colors` and `prefers-reduced-motion` blocks carries
  `:where(:not([data-gds-fixed-tone]))` on its subject, so an opted-out element is not matched at
  all. No counter-rule, no consumer-side `!important`, no source-order dependency.
- `:where()` has zero specificity. Adding the clause changed no rule's specificity, so every
  existing cascade relationship — between GDS's own rules, and against any consumer counter-rule
  written before the attribute existed — is unchanged.
- Element-level only. Descendants are not opted out; the attribute goes on each element that keeps
  its own styling.
- `body` is never opted out: its rules publish the `--mantine-color-text` and
  `--mantine-color-dimmed` values the page depends on. The forced-colors and reduced-motion resets
  apply to every element regardless of the attribute.
- Contrast on an opted-out element is the consumer's responsibility, as for any consumer styling.

`data-gds-badge-fixed-tone` (`GdsBadge`, `StatusBadge`, `GdsCountBadge`, `GdsRemovableTag`) is
the component-specific precedent and keeps working; the Badge rule honours both attributes.
`data-gds-local-contrast` is a different contract: package-owned, set only by
`getGdsOwnedContrastProps`, and it excludes descendants too.

`packages/gds-theme/src/preset-fixed-tone.test.ts` enforces the contract on every preset-gated
selector in both directions.

## Appendix: Amanoba dark shell + yellow CTA

Amanoba is a dark-default LMS/game product. Recommended recipe:

```ts
import { createPublicBrandTheme } from '@sovereignsquad/gds-theme/client';

export const amanobaMantineTheme = createPublicBrandTheme({
  flatSurfaces: true,
  overrides: {
    primaryColor: 'amanoba',
    colors: {
      amanoba: [/gds-* yellow scale */],
      amanobaYellow: [/gds-* alias scale */],
      ink: [/gds-* dark grey scale */],
    },
    other: {
      brand: { /gds-* email/OG/chart tokens */ },
      email: { /gds-* transactional email palette */ },
    },
    components: {
      Text: { defaultProps: { c: 'gray.2' } },
      Card: { defaultProps: { bg: 'ink.8', withBorder: true } },
      /gds-* form + modal dark surfaces */
    },
  },
});
```

Rules:

- use `@sovereignsquad/gds-theme/client` in client providers; use `@sovereignsquad/gds-theme/server` only for SSR-safe theme data
- do not call `withGdsMotion()` unless product marketing explicitly wants shared hover motion
- keep provider-branded OAuth colors in documented exception surfaces, not in `primaryColor`

## Approved preset modes

- `high-contrast` (`resolveGdsThemePreset('high-contrast')`) is the approved **accessibility** lane: a maximal-contrast, flat, undecorated preset with pure black/white surfaces, WCAG AAA body and meta text in both schemes, solid borders, near-black filled controls, and no decorative gradients. It is a first-class selectable preset (issue #453) — distinct from OS-driven `forced-colors` support, which GDS also honors — for products or users that want a deliberately high-contrast shell. Verified by `verify:token-contrast-scoring` and `verify:theme-accessibility`.
- `colorblind-safe` (`resolveGdsThemePreset('colorblind-safe')`) is the approved **accessibility** lane whose brand palette is drawn from the Okabe-Ito colorblind-safe qualitative set (blue `#0072b2` / vermillion `#d55e00`) so categorical/brand color stays distinguishable across deuteranopia, protanopia, and tritanopia (issue #453). It complements — it does not replace — GDS's standing rule that state is never signalled by hue alone (semantic components carry a label + icon per WCAG 1.4.1), which is what keeps success/danger distinguishable under every preset.
- `gdsDarkPublicTheme` is the approved preset for products that deliberately default to a dark public shell.
- `gdsFlatSurfaceTheme` is the approved preset for products that need flatter operational surfaces without creating a second token authority.
- `gdsEditorialPublicTheme` is the approved preset for public/editorial products that need serif-forward storytelling and flatter public surfaces without creating a private token branch.
- `createPublicBrandTheme()` is the approved composition helper for branded public products that need to layer serif headings, flat surfaces, and product-local token overrides in one governed merge path.
- `extendGdsTheme()` is deprecated for consumer use and should not appear in adopter docs, templates, or theme ownership files.
- the live token/theme lab at `https://sovereignsquad.github.io/general-design-system/themes` is the public reference surface for testing these shipped preset lanes interactively
- `withGdsMotion()` remains opt-in only. Shared motion is not part of the canonical base theme.
- `AccentPanel` is the approved cross-mode accent-surface primitive. If a product needs emphasis or rollout surfaces, start there before inventing page-local color-mode handling.

## Z-index / stacking layers

GDS does not publish a second, competing z-index scale. `@mantine/core/styles.css` (loaded via the mandatory `@sovereignsquad/gds-theme/styles.css` import) already ships a documented CSS variable scale — `--mantine-z-index-app` (100), `--mantine-z-index-modal` (200), `--mantine-z-index-popover` (300), `--mantine-z-index-overlay` (400), `--mantine-z-index-max` (9999) — and GDS defers to it as the single stacking authority rather than inventing a parallel one that could drift out of sync.

`gdsZIndexToken` (`@sovereignsquad/gds-theme`) exposes this scale by documented, typed tier name (`app`, `modal`, `popover`, `overlay`, `max`) so GDS's own components and consumers don't need to know Mantine's internal variable names. Any GDS component that renders fixed/sticky page-level chrome outside a Mantine overlay primitive (e.g. `BottomTabBar`, `FloatingActionPlacement`) must use `gdsZIndexToken.app` rather than an ad hoc number — this was a real, unpublished gap (see `DESIGN_SYSTEM_COMPETITIVE_GAP_ANALYSIS.md` P0 item 3) where two such components independently hardcoded different arbitrary values (200 and 20) with no shared authority. Consumers building custom overlays outside GDS's own component set should align with the same scale instead of guessing a number.

## Elevation

FOUNDATION.md's "no decorative shadow layering" policy governs cards and surfaces, not overlays — FOUNDATION.md explicitly says "Overlays may use elevation." Previously this had no published contract at all (issue #395): `gdsTheme.components.Popover.defaultProps.shadow` is now explicitly set to `'md'` (`shadows.md`'s already-established, deliberate soft-shadow value), giving Popover and everything built on top of it — Menu, HoverCard, and Select/Combobox/MultiSelect/Autocomplete dropdowns, none of which set their own `shadow` prop internally — a documented elevation tier instead of an undocumented Mantine default. `shadows.sm` is deliberately left untouched: `Card`'s own `shadow: 'sm'` default already depends on it, and changing it would be a real visual regression for every card in the system. Mantine's `Modal` does not expose a theme-configurable `shadow` prop at all, so its elevation remains Mantine's own fixed, non-GDS-owned styling.

## Density

`GdsDensityProvider`/`useGdsDensity` (`@sovereignsquad/gds-core`) publish a global density-mode axis (`compact`/`comfortable`/`spacious`) that a product can set once at the app or section level, rather than relying only on each component's own scattered local density prop (`AdvancedDataTable`'s density state, `CardContracts`'s `density` prop). This is new and purely additive: existing components' own defaults and props are unchanged. New density-aware call sites should read `useGdsDensity()` (or use `useGdsCardContract()`, the density-aware wrapper around `resolveGdsCardContract`) as the extension pattern, falling back to the ambient value only when no explicit `density` prop is passed — an explicit prop always wins.

## Raw `--gds-*` custom properties vs. the Mantine-rendered scale (issue #642)

A brand theme built with `createBrandTheme(...)` exposes the same design decision through
**two intentionally different scales**, not one value in two formats:

- **The raw `--gds-*` custom properties** (`cssVariables`, e.g. `--gds-radius-xs/sm/md/lg`)
  are the **public, consumer-facing design-intent scale** — documented in
  `DESIGN_TOKENS_DTCG.md`/`docs/SAFE_STYLING.md`, meant to be read directly in plain CSS or
  via `var()`, and the values this repository treats as governed.
- **The `mantineTheme` object's own scales** (`theme.radius`, `theme.fontSizes`, etc.) are a
  **separate, brand-remapped rendering scale** that Mantine components consume directly
  (`radius="md"`) and that a brand handoff may set independently of the raw axis. This is
  not a bug or a pending reconciliation — a theme is free to render at different steps than
  its own raw scale states, the same way a design system's spacing tokens and its grid
  columns are related but not identical.

A consumer who wires the raw custom properties onto the document for use in plain CSS gets
different values than what GDS's own Mantine components render on the same page **by
design**. `GdsShapeElevationSystemReference` computes and states live how many of the 14
semantic radius roles a given theme actually differentiates, rather than asserting a fixed
number — read that page, not this paragraph, for a theme's current state.

**Typography tracking follows the same split, with one real gap.** `GdsTypographyAxis`
supports a `tracking` map, and `resolveGdsTypographyTokens()` emits `--gds-tracking-*` from
it — but `createBrandTheme(...)`'s own `cssVariables` output does not include this map's
result, unlike the radius/color roles above. A theme that wants tracking applied to the
document must call `resolveGdsTypographyTokens()` itself and merge the result. Whether a
given brand lane (e.g. `class-usa`) *should* populate a tracking scale by default is a
brand-design decision left to that theme, not a system default every lane must set.
`tracking` values ARE validated at resolution time (issue #695) — `normal`, a signed
px/rem/em/ch length, or a `var()` reference; a percentage, a bare unitless number, or an
arbitrary string throws `GdsAxisError` instead of shipping straight through to CSS.

**Elevation roles and italic typography (issue #695).** `GdsElevationRole` carries two more
members, `sidebar` and `pin` (appended after `tooltip`), each defaulting to step 1 in
`GDS_DEFAULT_ELEVATION_AXIS.roles`. `GdsElevationAxis.roles` accepts `GdsElevationStep |
GdsElevationValue` per role — a role may either pin a shared step or declare its own
directional shadow (or `{ kind: 'none' }`) without touching the shared, monotonic step ramp,
mirroring how a shape-axis role can already carry a literal radius instead of a step name.
`GdsTypographyAxis` also gained `fontStyles?: Partial<Record<GdsTextSizeStep, GdsFontStyle>>`
(`GdsFontStyle = 'normal' | 'italic'`), resolving to `--gds-font-style-<step>` and emitted only
for the steps a theme declares. No preset declares any of these inputs yet; this is pure
mechanism work in `packages/gds-theme/src/axes.ts`.

## Layout (shell geometry, issue #698)

`GdsThemeAxes` carries a `layout?: GdsLayoutAxis` key (`packages/gds-theme/src/axes.ts`),
following the file's own recipe for adding an axis: a type, `GDS_DEFAULT_LAYOUT_AXIS`,
`validateGdsLayoutAxis`, and a `resolveGdsLayoutTokens` branch in `resolveGdsAxisTokens`.

Unlike `motion` (which emits only declared overrides), `layout` is emitted unconditionally, like
`shape` and `density`: every preset resolves the full nine-token `--gds-layout-*` namespace
whether or not it declares `layout` at all, so `DiscoveryShell`/`BottomTabBar` reading
`var(--gds-layout-*, ...)` get a real token on every preset, and their literal fallback is a
safety net for rendering with no GDS theme runtime present, not the normal path.

The nine tokens: `sidebar-width` (280px default), `header-height` (60px), `footer-height`
(68px), `nav-item-height` (44px), `content-max-width` (1400px), `list-rail-width` (480px),
`bottom-bar-height` (64px), `content-bottom-padding` (derived, `calc(bottom-bar-height +
space-xl)`), and `sheet-top-radius` (defaults to the shape axis's `sheet` ROLE token, not a
step, so repointing that role repaints bottom sheets without a second declaration).

Governance rules:

- **Scheme-independent.** The resolver takes no `scheme` parameter; shell geometry does not
  fork light/dark the way color and accent do.
- **Density-invariant, deliberately.** `compact`/`spacious` do not scale `--gds-layout-*`
  values — the density resolver's own rationale (scaling would redesign, not adjust: a 240px
  sidebar at 0.75 becomes an unreadable 180px) applies here even more directly, since shell
  regions are structural, not typographic rhythm.
- **44px target floor, same enforcement shape as density.** `headerHeight`/`footerHeight`/
  `bottomBarHeight` throw below 44px with no exception path — these regions host interactive
  44px targets (the burger toggle, footer actions, bottom-tab items) and cannot be shorter than
  the targets they contain. `navItemHeight` enforces the same floor through one recorded
  exception, `GDS_LAYOUT_DIMENSION_EXCEPTIONS.navItemHeight`, mirroring
  `GDS_CONTROL_HEIGHT_EXCEPTIONS`'s precedent: a dense sidebar nav row may render below 44px
  visual height only where the interactive row (full row width plus vertical padding) still
  preserves a 44px effective hit target. A consumer declaring a sub-44px `navItemHeight` owns
  that obligation.
- **Backward-compatible by construction.** `DiscoveryShell`'s `sidebarWidth`/`headerHeight`
  prop defaults and its inline footer height, and `BottomTabBar`'s bar-height `calc()`, changed
  to `var(--gds-layout-*, <same literal as before>)` — with no theme runtime present, rendered
  geometry is pixel-identical to before this axis existed. An explicit prop still wins over the
  token default.
- **Not the `GDS_SHELL_HEIGHTS` constants.** `resolveGdsShellHeightTokens`'s `--gds-shell-height-*`
  set is a separate, fixed, non-themed `PublicShell` header-variant table; the two namespaces
  are not interchangeable and folding one into the other is a distinct, future decision.

## Design rule profiles (milestone: Design Rule Profiles, issues #643-#653)

Full narrative, research grounding, worked adoption example, and FAQ:
[`docs/DESIGN_RULE_PROFILES.md`](docs/DESIGN_RULE_PROFILES.md). This section states only the
governance rules — the constraints future changes to this axis must keep satisfying — not the
full technical explanation, matching how this file treats every other cross-cutting axis.

`GdsThemeAxes` carries a ninth, optional axis: `designRuleProfile?: GdsDesignRuleProfile`
(`packages/gds-theme/src/axes.ts`). Governance rules:

- **Additive and optional, always.** `GDS_DEFAULT_DESIGN_RULE_PROFILE` (no proportion claim,
  `custom` harmony, `1.25` Major Third type scale, `AA` contrast) is a profile every existing
  theme already satisfies with zero behavior change. A future change to this axis must
  preserve that: no theme may be silently opted into a stricter profile.
- **Computed, never hand-typed.** Color-proportion classification (`resolveGdsColorProportionProfile`),
  type-scale naming (`resolveGdsTypeScaleProfile`), and color-harmony classification
  (`resolveGdsColorHarmonyProfile`) all read live source values (role names, the real
  typography axis ratio, real preset hex colors) — a hand-asserted classification for any
  preset is a governance violation of this axis, not an acceptable shortcut.
- **`validateGdsDesignRuleProfile(profile, themeId)`** follows this file's own `axes.ts`
  established pattern: throws a single `GdsAxisError` on the *first* violation found, matching
  `validateGdsShapeAxis`/`validateGdsDensityAxis` — it does not accumulate every violation into
  one report (that is `GdsBrandThemeError`'s distinct, separate pattern in `brand-tokens.ts`).
- **Declared and measured are two different, non-reconcilable metrics — never conflate them.**
  Declared/intended (token-role classification) and measured/rendered (real pixel-area
  sampling on the reference site) can legitimately diverge; no future work may present one as
  a "correction" of the other, or silently drop the distinction in UI or docs.
- **Enforcement stays source-level and advisory-first.** The lint rule, dev-time warning, and
  `gds-compliance check-design-rules` CLI are opt-in and `warn`-severity by default; the
  measurement budget (`designRuleUnclassifiedRate`) is `advisory: true`. Promoting any of
  these to a hard blocking gate is an explicit, separate governance decision — not a change
  bundled silently into an unrelated fix.
- **No simulated or estimated measurement.** Anywhere a measured percentage is shown (the
  Theme Lab panel included), it must be read from the real, committed
  `audit/design-rule-coverage.json` artifact — never a placeholder or hand-typed estimate.

## CSS VibeThemes

GDS must provide expressive color lanes for real products. Light mode and dark mode are scheme choices, not the full theme offering. A VibeTheme is a package-owned visual contract that combines a Mantine theme preset with CSS variables for canvas, shell, surface, border, text, muted text, primary, accent, glow, gradient, and hero treatments.

Approved colorful preset ids:

- `sunset` - warm orange product energy
- `oceanic` - cool cyan-blue product clarity
- `forest` - grounded green product trust
- `ruby` - bold red high-attention product surfaces
- `amber` - golden operational warmth
- `neon-night` - dark-forward lime campaign surfaces
- `skyline` - indigo technology surfaces
- `aurora` - teal-cyan optimistic app surfaces
- `coral` - expressive creator, commerce, and social surfaces
- `mint` - clean growth, health, and learning surfaces
- `orchid` - grape editorial and premium surfaces
- `royal` - confident violet SaaS and professional surfaces
- `cosmic` - highly saturated blue-violet-cyan-magenta launch and showcase surfaces

Usage rule:

```ts
import {
  getGdsVibeThemes,
  resolveGdsThemePreset,
  resolveGdsVibeTheme,
  useGdsThemePresetState,
} from '@sovereignsquad/gds-theme/client';

const theme = resolveGdsThemePreset('coral');
const vibe = resolveGdsVibeTheme('coral');
const availableVibes = getGdsVibeThemes();

const { selection, setPreset, setScheme, setFontLane, reset } = useGdsThemePresetState();
```

Runtime rule:

- `useGdsThemePresetState(...)` must set `data-gds-theme-preset`, `data-gds-theme-runtime`, `data-gds-font-lane`, `data-mantine-color-scheme`, and the `--gds-vibe-*` CSS variables on the document root.
- `useGdsThemePresetState(...)` also loads the active font lane's web font: the default `'inter'` lane is loaded statically by `packages/gds-theme/styles.css`; every other lane is loaded on demand via a single non-blocking `<link id="gds-font-lane-stylesheet">`, added/swapped/removed as the lane changes (issue 529). The package stylesheet must never statically `@import` more than the default lane's font — that duplicates each lane's own governed `cssImportUrl` (font-lanes.ts) by hand and defeats every lane's declared `loadStrategy: 'non-blocking-stylesheet'`.
- The official site must use the selected VibeTheme across the whole shell, not only inside the Theme Lab card.
- VibeTheme visuals must be CSS-only: gradients, color-mix, surface variables, and component tokens are allowed; pixel/image backgrounds are not the default theme mechanism.
- `cosmic` is the sanctioned high-saturation reference lane. If teams need a dramatic multicolour app vibe, start from `cosmic` instead of building route-local image or gradient systems.

Do not create a product-local theme catalog to achieve colorful branding. If a color lane is missing, add it to the GDS preset registry and VibeTheme registry, document the intended product use, add live Theme Lab coverage, and verify the lane through package tests.

Avoid:

- route-local theme state that resets on navigation
- hardcoded app-only gradients outside the GDS VibeTheme registry
- using image backgrounds as the theme identity
- changing only `primaryColor` while leaving shell, controls, cards, nav, focus, and page canvas visually neutral
- consumer-owned `createTheme(...)`, `mergeMantineTheme(...)`, or `extendGdsTheme(...)` theme catalogs

## Reserved sub-brand accent lanes (`ai.*`, issue #697)

A preset may declare a reserved sub-brand accent lane — `GdsVibeTheme.ai` in
`packages/gds-theme/src/vibe-themes.ts` — when its brand carries a distinct identity for one
sub-system that is deliberately never a general action color. The first (and, at the time of
writing, only) lane is Your Field's Scout AI identity: `--gds-ai-gradient` (an orange identity
gradient), `--gds-ai-panel` (a navy promo-panel gradient), and `--gds-ai-accent` (the Scout
orange used as a static color). Every preset without an `ai` field emits none of these
variables; their absence is the default, not an omission.

Reserved-usage contract, exhaustive:

> The `ai.*` token family is a reserved sub-brand lane. Its sanctioned consumers are,
> exhaustively: AI search/entry surfaces (`AISearchCard`), chat surfaces (`ChatThread`,
> `ChatMessage`, `ChatInput`, `StreamingIndicator`), the AI promo panel component, the
> emphasized AI tab disc in `BottomTabBar`, the preset's focus ring (via
> `GdsFocusRingSpec.colorRole` referencing the ai accent role), the featured ring, and —
> widened by issue 700 — `SemanticButton`'s `gradient` brand intent exclusively, for
> AI-identity call-to-action buttons (e.g. "Ask Scout AI"). It is never a general action color
> beyond that one opt-in intent: every other `SemanticButton` intent, link, badge, and non-AI
> control keeps the preset's primary/accent roles. A gradient-filled control carries text at
> 14px minimum and weight 600 minimum, since white on the Scout orange fill measures 2.84:1 —
> below the AA text floor — and is never claimed as a text-contrast pass; `SemanticButton`
> enforces this floor itself via `GDS_BUTTON_GRADIENT_TEXT_FLOOR`. Where a preset declares no
> `ai` lane, the `gradient` intent renders a solid primary-fill fallback rather than a broken
> transparent button.

Governance rules:

- **Encoded twice, checked once each way.** The reservation is declared as a claim
  (`axes.designRuleProfile.reservedAccents`, an array of `{ role, surfaces }`) and enforced as a
  mechanical gate (`scripts/verify-ai-reserved-usage.mjs`, wired as `verify:ai-reserved-usage`
  into `verify:release`), which scans `packages/gds-core/src/**` and
  `packages/gds-theme/styles.css` for the literal string `--gds-ai-` against an explicit
  allowlist and fails loudly, naming file and line, on anything unlisted.
- **Widening the allowlist is a governance decision.** Adding a sanctioned consumer's file to
  the gate's allowlist happens in the same change set as that consumer, never ahead of it and
  never as an unrelated drive-by edit.
- **`reservedAccents` is validated, not just typed.** `validateGdsDesignRuleProfile` throws if a
  role is reserved more than once, or if a reservation names zero surfaces — a reservation
  nothing may consume is a contradiction.
- **Classified as non-text, everywhere.** `ai-gradient`/`ai-panel` are the `effect` token
  category; `ai-accent` is `color` but excluded from the readable-text hard gates
  (`verify:token-contrast-scoring`). The `ai-accent-text-contrast` accessibility-floor rule
  measures white on the accent fill and prints the ratio every run at `report` severity — it
  never fails the build, because the lane never claims a text-contrast pass in the first place.
- **Dark values are authored, never copied.** The handoff a reserved lane's light values come
  from may define no dark scheme at all; the dark sibling for each field is a deliberate
  decision derived with this package's own contrast machinery (`ensureContrast`/
  `mixCssColors`), documented in the vibe entry's own comment — never the light value reused
  silently in dark mode.
- **No consuming component may construct an `ai.*` variable name dynamically** (string
  concatenation, template interpolation of the role name, etc.) — the gate matches the literal
  `--gds-ai-` substring, and a dynamically-built reference would evade it while still violating
  the reservation.

## Two unrelated "accent" concepts, and why `outline` mode was not enforced (issue #700)

GDS carries two structurally unrelated mechanisms that both use the word "accent":

- `GdsAccentAxis`/`GDS_ACCENT_NAMES` (`packages/gds-theme/src/accent-axis.ts`) — ten
  categorical accent names (plum, indigo, ocean, teal, forest, bronze, terracotta, magenta,
  slate, grape), each with a shade ramp, consumed by `GdsIconBadge` for categorical/tag UI. Its
  `GDS_ACCENT_MODE_ENFORCEMENT.outline` entry is measured but deliberately left
  `enforced: false` — no component renders one of these ten named accents as label text.
- The single brand accent role — `vibe.accent` / `--gds-brand-accent-action` — consumed by
  `SemanticButton`'s `accent` brand intent (fill) and, since issue 700, its `outline-accent`
  intent (stroke + label on a transparent ground).

`outline-accent` is the first component to render the *second* mechanism's accent role as label
text, but it does **not** flip `accent-axis.ts`'s `outline` enforcement: that entry governs the
unrelated ten-name categorical ramp, and flipping it was measured (issue 700) to produce 1080
real, pre-existing contrast failures across the preset catalog that have nothing to do with
`SemanticButton`. Instead, `accessibility-floor.ts` carries a purpose-built rule,
`outline-accent-text-contrast`, that measures `--gds-brand-accent-action` against
`--gds-bg-page` (the surface a transparent-fill button actually sits on) across every preset
and scheme. Measured result: every light-scheme pairing clears 4.5:1 (the generic derivation
path in `semantic-token-source.ts` `ensureContrast`s this role against `canvasLight`), but 25 of
27 dark-scheme pairings do not — that derivation reuses the light-derived value unchanged in
dark mode rather than re-deriving against `canvasDark` (class-usa and gold-athlete's bespoke
emission paths are the two that pass in both schemes). The rule is therefore `report`, not
enforced, exactly like `primary-cta-text-contrast` and `ai-accent-text-contrast` before it: a
real, honestly-measured gap, printed every run, closed at the source (re-deriving the dark
value) rather than papered over by enforcing a mode that would fail 1080 unrelated cases.

## Importing an externally-produced design (issue #535)

A theme lane's source material — a Figma file, a screenshot, an AI design
tool's output (Claude Design or otherwise), a brand guideline PDF — is
allowed to originate outside this repository. What that source material is
allowed to become is not: it must be re-derived into the same governed
`GdsVibeTheme`/brand-token contract every other lane uses, never consumed
directly as CSS, an image, or a copy-pasted color value.

The concrete shape an incoming handoff should take is
[`TEMPLATES/DESIGN_HANDOFF_TEMPLATE.md`](TEMPLATES/DESIGN_HANDOFF_TEMPLATE.md) (issue 539) —
the fill-in structure codified from the ClassScout v2 handoff, whose defining property is that
an implementer never has to guess: every rule carries its rationale, ambiguities are named in
an "open items" list rather than left to be discovered, and the states/content/count contracts
arrive as requirements rather than afterthoughts.

This is the same one-directional principle [`docs/FIGMA_UI_KIT.md`](docs/FIGMA_UI_KIT.md)
already states for the opposite direction — "the code tokens and component
contracts are authoritative... never the reverse" — and the same
"borrowing" discipline [`PATTERN_SERVICE_MODEL.md`](PATTERN_SERVICE_MODEL.md)
already requires when studying an external reference: "study the shape,
rebuild as a governed contract" — not "copy the external artifact as
product styling authority."

Required process for any externally-sourced design:

1. **Extract intent, not values.** Identify the palette (primary, accent,
   any secondary hues), the light/dark surface treatment, and the overall
   feel (flat/brand-serious vs. expressive/gradient-forward). Do not lift a
   hex value straight from a screenshot or a Figma variable and drop it
   into a token file unverified — every value that ships must be
   deliberately chosen for both schemes, not "whatever the source file
   happened to show in whichever mode it was captured in."
2. **Map into the full `GdsVibeTheme` field list** (`packages/gds-theme/src/vibe-themes.ts`)
   — every light/dark pair, not just the ones the source material made
   obvious. A source design that only shows one color scheme does not
   excuse skipping the other; the missing scheme must be deliberately
   designed and contrast-checked, not derived by an unverified brightness
   flip. This is not a hypothetical risk — issues #533/#534 were exactly
   this failure mode (a dark-mode value silently inherited from a
   light-mode source instead of being independently designed), shipped to
   production before being caught.
3. **Verify every pairing the new lane produces** against WCAG AA (4.5:1
   normal text, 3:1 large text/UI components) in both schemes, from real
   computed styles on the live pattern catalog — not visual impression of
   the source material.
4. **Register and verify exactly like any other lane** — add it to the
   preset registry, add live Theme Lab coverage, add package tests, pass
   `npm run verify:release`. An externally-sourced theme gets no exemption
   from any rule in this document.
5. **Trace it to a GitHub issue** and document the source (which design,
   whose brand, what tool produced it) in the commit/PR — a future
   maintainer needs to know a lane's provenance without guessing.

The operational tool for this — a copy-pasteable prompt that walks a fresh
Claude Code session through this exact process, including how to handle a
source design as input — is [`TEMPLATES/GDS_THEME_CREATION_PROMPT.md`](TEMPLATES/GDS_THEME_CREATION_PROMPT.md).
See `CONTRIBUTING.md`'s "Importing an externally-designed theme" section
for the maintainer-facing walkthrough.

## 3.0.0 theme explorer proof contract

The GitHub Pages theme route must prove all approved lanes before the 3.0.0 release:

- preset selection for `gdsTheme`, `gdsDarkPublicTheme`, `gdsFlatSurfaceTheme`, `gdsEditorialPublicTheme`, and `createPublicBrandTheme(...)`
- light, dark, and auto color-scheme proof copy
- comparison mode between two shipped lanes
- reset behavior that returns to the baseline `gdsTheme` lane
- explicit unsupported-lane guidance that explains why `extendGdsTheme(...)`, `createTheme(...)`, and `mergeMantineTheme(...)` are prohibited in consumer-owned theme files

If any lane regresses contrast or keyboard/focus visibility, block the release and keep consumers on the previous stable package line until the lane is fixed.

## Runtime persistence contract

Theme selection is part of the public reference-site runtime contract, not temporary page state.

The official website and any governed adopter that offers theme or typography switching must preserve the selected runtime across:

- internal navigation
- direct links to nested routes
- GitHub Pages or static-host SPA fallback reloads
- full browser refreshes
- remounting the theme explorer after visiting another route

Required implementation:

1. Store only serializable theme intent:
   - preset id
   - effective color scheme
   - font lane id
   - brand primary id
   - governed brand flags
   - runtime key
2. Reconstruct the Mantine theme from GDS helpers on startup:
   - `resolveGdsThemePreset(...)`
   - `applyGdsFontLane(...)`
   - or use the canonical `useGdsThemePresetState(...)` hook, which performs the same validation, reconstruction, persistence, and root attribute application
3. Apply the reconstructed runtime to the root provider before route content depends on it.
4. Set root runtime attributes for inspection and regression checks:
   - `data-mantine-color-scheme`
   - `data-gds-theme-runtime`
   - `data-gds-font-lane`
5. Pass the active runtime selection back into `ReferenceThemeExplorer` so controls reflect the whole-site runtime instead of resetting to local defaults.
6. Treat storage as best-effort. If `localStorage` is blocked, theme application must still work for the current session.
7. Add regression coverage that selects a non-default preset and non-default font lane, remounts the app on a nested route, and verifies the selected runtime survives.

What ruins the system:

- keeping theme selection only in component-local `useState`
- storing a full Mantine theme object instead of serializable intent
- rebuilding the root provider from `gdsTheme` defaults on every direct route load
- letting `/themes` controls own a private runtime that differs from the site shell
- using route-local CSS, `light-dark(...)` patches, or page-specific providers to fake runtime changes
- persisting only color scheme while dropping preset, font lane, or brand-generator options
- allowing static-host 404 fallback loads to reset the runtime
- using `extendGdsTheme(...)`, `createTheme(...)`, or `mergeMantineTheme(...)` in consumer-owned theme files as a shortcut around the approved lane contract

Preferred reference-site shape:

```ts
const {
  selection,
  setSelection,
  setPreset,
  setScheme,
  setFontLane,
  reset,
} = useGdsThemePresetState({ storageKey: 'gds-reference-theme-selection' });

<GdsProvider
  theme={selection.theme}
  defaultColorScheme={selection.colorScheme}
  forceColorScheme={selection.colorScheme === 'auto' ? undefined : selection.colorScheme}
>
  <ReferenceThemeExplorer
    initialSelection={selection}
    onSelectionChange={setSelection}
  />
</GdsProvider>;
```

Review checklist for runtime-theme work:

- Can a visitor choose `Oceanic wave`, switch to dark mode, choose `Space Grotesk`, then open `/live-proofs/surfaces` directly without losing the runtime?
- Do direct links and static-host fallback pages serve the same persisted runtime as normal internal navigation?
- Are font files or imports available for every advertised font lane?
- Does `createGdsTokenGraph()` still expose a complete light/dark token pair for every shipped lane?
- Does `gds-theme-tokens validate` pass before the release branch is promoted?
- Has `gds-theme-tokens diff --compare <previous-graph.json>` been reviewed for intentional token changes and rollback safety?
- Does the test harness use a real in-memory storage implementation instead of a no-op storage mock?
- Does CI fail if the persistence contract is removed from the official reference app?
