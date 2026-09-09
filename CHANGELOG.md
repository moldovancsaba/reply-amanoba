# Changelog

All notable policy changes to the General Design System are recorded here.

## Unreleased — A governed activity pictogram family, a generated brand badge, an element-level opt-out from the theme-preset repaint, a layout axis, a logo lockup / notification bell / compare button, detail-page facts / provider-claim surfaces, the trust-layer component family, sidebar/pin elevation roles with validated tracking and italic typography inputs, a reserved Scout AI sub-brand accent lane, two new SemanticButton brand intents for it, and a ListingCard featured/selected ring with a pick badge, a row tile, and a browse-selection hook (#708, #699, #724, #698, #710, #713, #711, #709, #695, #697, #700, #701)

### `ListingCard` featured/selected ring, a pick badge, a media-left row tile, and `useGdsBrowseSelection` (#701)

`ListingCard` gains a `selected` prop rendering the identical ring surface treatment as
`featured` — a 1px accent border, a featured ring, and the elevated shadow — with no hover-lift
in any state: selection reads as a persistent ring, never a transient transform. Both props set
`data-gds-listing-featured`/`data-gds-listing-selected`; `selected` also sets
`aria-current="true"` when `interactiveMode` is `'surface-link'`.

A governed `pickBadge` prop (`true`, or `{ label, ariaLabel }`) renders a "Pick" overlay badge —
a star glyph and localized label on a dark scrim pill — inside the media tile's inline-start/top
corner, over both a supplied `image` and the generated fallback thumbnail, stacking
deterministically with the existing `mediaOverlay` slot (issue 679) and leaving
`mediaAffordance` unaffected. New exported geometry constants (`GDS_LISTING_TILE_SIZE_PX`,
`GDS_LISTING_PICK_BADGE_INSET_PX`, and the badge's icon/type/padding constants) are
reference-derived from the Your Field v3 `ProgramListCard` and follow the `GDS_PIN_*` precedent
in `GdsMapPinBadge`.

The card contract's `mediaPlacement` (previously resolved but unconsumed) now drives a fixed
96×96 row-tile media form for the `media-left` variant, using the existing `thumbnail` radius
role; the top-media form is unchanged.

A new `useGdsBrowseSelection` hook (`packages/gds-core/src/BrowseSelection.client.ts`) provides
single-selection state, controlled or uncontrolled, shareable between a `ListingCard` list and a
set of `GdsMapPinBadge` pins in a browse split view — one id drives both the selected card's
ring and the matching pin's `state="selected"`, so a "selected card = selected pin" browse
experience no longer means every consumer reinventing the sync.

Two pre-existing defects in the `featured` badge are also fixed, visibly: the label was a
hardcoded English `"Featured"` string (now localized via `gds.listingCard.featuredLabel`), and
its color came from Mantine's raw `color="violet"` (now the same accent tint token pair the
card's active affordance state already uses). Cards using none of the new props render DOM
identical to 6.7.0 aside from these two fixes.

### Two new SemanticButton brand intents: `outline-accent` and `gradient` (#700)

`SemanticButton`'s `brandVariant` gains two values carrying the full micro-action state axis
(hover, pressed, focus, disabled, loading, transient feedback): `outline-accent` (transparent
fill, 1.5px accent stroke and label) and `gradient` (the reserved Scout AI gradient fill from
#697, layered over a solid primary-chain fallback where a preset declares no `ai` lane). The
four pre-existing `brandVariant` values, and the `primary`/`secondary`/`subtle`/`danger`
vocabulary, are unchanged.

Both intents' resting paint carries no inline style — a stylesheet `:hover`/`:active` rule
cannot override an inline style without `!important` — so their entire state axis lives in
`packages/gds-theme/styles.css`, keyed on the existing `data-gds-brand-button` attribute. The
governed button repaint rule (and its class-usa/gold-athlete/cosmic/preview-surface variants)
now explicitly excludes both new intents rather than relying on selector-specificity accidents.
A transient success/error feedback treatment on either intent works by `SemanticButton`
withholding that attribute for the duration, handing the cascade to the existing governed
success/danger button rules — no second timing constant, no parallel mechanism;
`GDS_BUTTON_FEEDBACK_DURATION_MS` and its timer are reused unchanged. New exports:
`GDS_BUTTON_GRADIENT_TEXT_FLOOR` (`{ fontSizePx: 14, fontWeight: 600 }`, since white on the
Scout orange fill clears only the 3:1 non-text threshold — applied directly in the `gradient`
intent's stylesheet rule, not left to whatever `size` a caller passes) and
`GDS_BUTTON_OUTLINE_ACCENT_STROKE_PX` (`1.5`). `SemanticButton` now also sets `aria-busy` while
loading and announces a transient feedback label change via a polite live region, for every
brand intent.

`outline-accent` reuses the same `--gds-brand-accent-action` chain the existing `accent` fill
already uses for its rest-state stroke/label — not a new "accent pressed" token, which does not
exist and which this issue's Non-Goals forbid inventing — deriving pressed as a `color-mix`
darken of that same value, the same technique the hover wash already uses.

`gradient` widens the `ai.*` reserved-usage contract (THEME_GOVERNANCE.md, issue #697):
`SemanticButton`'s `gradient` intent is now the one brand intent sanctioned to consume
`ai.gradient`, exclusively for AI-identity CTAs (e.g. "Ask Scout AI") — every other intent,
link, badge, and non-AI control still keeps the preset's primary/accent roles.
`scripts/verify-ai-reserved-usage.mjs`'s allowlist gained a per-line-content entry (not a
whole-file exemption) for the one new rule in `packages/gds-theme/styles.css` that consumes it,
so an unrelated `--gds-ai-*` reference anywhere else in that file still fails the gate.

`outline-accent`'s accent-as-label-text pairing is measured by a new report-severity
accessibility-floor rule, `outline-accent-text-contrast` (`--gds-brand-accent-action` against
`--gds-bg-page`): every light-scheme value clears 4.5:1, but 25 of 27 dark-scheme values do
not, because the generic derivation path ensures that role's contrast against `canvasLight`
only and reuses the same value unchanged in dark mode — a real, pre-existing gap, printed every
run, not silently enforced away. This is deliberately *not* the same thing as flipping
`accent-axis.ts`'s unrelated `GDS_ACCENT_MODE_ENFORCEMENT.outline` entry, which governs a
different, categorical ten-name accent ramp (`GdsIconBadge`'s) and which measured to 1080 real,
pre-existing failures unrelated to this issue's scope — see THEME_GOVERNANCE.md.

### A reserved Scout AI sub-brand accent lane (#697)

The Your Field handoff defines a distinct Scout AI sub-brand — an orange identity gradient
(`linear-gradient(135deg, #ff6b35 0%, #ff9055 100%)`), a navy promo-panel gradient
(`linear-gradient(124deg, #0d2340 0%, #1a3a6a 100%)`), and the Scout orange `#ff6b35` doubling
as the system-wide focus ring and featured-card ring — with an explicit rule that the gradient
belongs to Scout AI exclusively and is never a general action color. Nothing in GDS could
express that: the token graph's color validation rejects a gradient-valued role outright (no
gradient-carrying `BrandSemanticRole` exists), and no gate could say "this token may be
consumed only by these surfaces."

`GdsAiAccentLane` (`packages/gds-theme/src/vibe-themes.ts`) is a new optional field,
`GdsVibeTheme.ai`, carrying `gradient`/`panel`/`accent` as light/dark pairs. Only the
`your-field` preset declares it; every other preset's emitted token set is byte-identical
before and after this change. `emitAiAccentCssVariables` merges the lane into the semantic
variable set inside `getGdsVibeThemeCssVariables`, before the existing dark-collapse loop, so
`--gds-ai-gradient`, `--gds-ai-panel`, and `--gds-ai-accent` (plus their `-dark` siblings)
collapse onto their base names in dark mode exactly like every other `--gds-*` role.
`getGdsAiAccentLane(id)` (re-exported from `index.ts`/`client.ts`/`server.ts`) reads the same
lane without parsing CSS, returning `undefined` for every preset but `your-field`.

The handoff's token source ships no dark scheme at all. Each dark value is a deliberate
decision, not the light value reused: `ai.accent` and both `ai.gradient` stops are nudged
toward white with the package's own `ensureContrast`/`mixCssColors` machinery until they clear
3:1 (WCAG 1.4.11) against `your-field`'s dark canvas (`#0a1626`); both `ai.panel` stops clear
3:1 against the dark surface (`rgba(17, 36, 58, 0.9)`) it actually sits on top of, so the panel
keeps reading as an elevated surface instead of disappearing into an equally dark shell.

`inferNodeCategory` (`token-operations.ts`) classifies `ai-gradient`/`ai-panel` as `effect` and
`ai-accent` as `color`, so the token graph and `verify:token-contrast-scoring` stay clean and
the ai roles are excluded from the readable-text hard gates they never claim to pass. A new
report-severity accessibility-floor rule, `ai-accent-text-contrast`, measures white on
`ai.accent` per preset × scheme and prints the ratio (2.84:1 for `your-field`, below AA) every
run without failing the build — the number is derived, never retyped as prose.

The reservation is encoded twice. `GdsDesignRuleProfile.reservedAccents`
(`packages/gds-theme/src/axes.ts`) is a new optional array of `{ role, surfaces }` claims;
`validateGdsDesignRuleProfile` now throws on a duplicate role or an entry with zero surfaces.
`your-field`'s vibe entry declares its own reservation. Mechanically,
`scripts/verify-ai-reserved-usage.mjs` (`npm run verify:ai-reserved-usage`, wired into
`verify:release`) scans `packages/gds-core/src/**` and `packages/gds-theme/styles.css` for the
literal `--gds-ai-` against an explicit allowlist — empty today, since every sanctioned
consumer (`AISearchCard`, the chat surfaces, the AI promo panel, `BottomTabBar`'s emphasized
disc, the focus ring, the featured ring) lands in a follow-on issue in this delivery — and
fails loudly, naming file and line, on anything unlisted. The reserved-usage contract itself is
documented in `THEME_GOVERNANCE.md`.

This issue ships the token mechanism, contract, and gates only — no consuming component, no
`SemanticButton` gradient variant, and no change to any other preset's visual output.

### A governed activity pictogram family (#708)

`packages/gds-core` had no home for a drawn domain-object visual identity: `GdsIcons` is a Tabler-backed UI-chrome dictionary with no activity/sport marks and no contextual-treatment or scale-tuning contract, so a consumer needing one drew its own ungoverned SVGs. `pictograms.tsx` adds the mechanism the Your Field v3 Design Guidelines' "Visual Language & Pictogram Guidelines" page specifies: a validated `GdsPictogramFamily` registry (`createGdsPictogramFamily`, mirroring `GdsIcons`' closed-dictionary validation style), the `GdsPictogram` component resolving a key to one of four contextual treatments (`list`/`detail`/`hero`/`pin`), three scales (16/32/72px, each with its own grid-space stroke weight tuned for legibility), and four interaction states expressed exclusively through existing color tokens — `selected` reads the semantic `--gds-accent` role, `disabled` applies the same reduced-opacity convention already used by `ChoiceChip`/`GdsRemovableTag`, and `hover` changes no drawing color at all (surface emphasis is the host's job).

The mechanism ships with the governed 11-entry activity family (`gdsActivityPictograms`): soccer, baseball, basketball, swimming, tennis, flag football, martial arts, camps, lacrosse, athletics, and hockey. Every drawing is the real, unmodified path data of a published icon-library glyph on a 24×24 optical grid — never hand-drawn, never redrawn or traced from a reference. Six reuse `@tabler/icons-react` verbatim (soccer, baseball, basketball, tennis, flag football, camps) — the same MIT-licensed dependency `GdsIcons` already ships. Three more come from the wider Iconify catalogue (200+ open, permissively-licensed collections), sourced when Tabler had no matching glyph: swimming (icon-park-outline's swim-ring, path coordinates halved from its native 48×48 grid to this family's fixed 24×24 — the only rescaling any entry needed), athletics (streamline-ultimate), hockey (lucide-lab) — all still real, unmodified path data, rendered `stroke` like the Tabler entries. The remaining two (lacrosse, martial-arts) exist as real icons only in filled form in every set checked (mdi, Apache-2.0) — including martial-arts depicting a person, the one shipped drawing that does — so they render `fill` (`GdsPictogramDefinition.fillMode`) rather than being forced into a stroke they were never drawn as. Any future pictogram gap follows the same order: Tabler first, then the wider Iconify catalogue, real source data only — see `docs/ICON_REGISTRY.md`. One drawing per pictogram: treatment and scale never swap geometry, only size, stroke (where applicable), and (for `hero`) a fixed three-layer scale/opacity echo recipe for depth without a second color. An unresolvable key renders the family's `fallbackKey` if set, otherwise a layout-stable empty slot, with a development-only warning — never a throw. Default labels for the shipped family are localized to all 12 supported locales through the existing `useGdsTranslation` mechanism.

`GdsPictogramSystemReference`, wired into the reference site's pattern registry, proves the whole system live: all 11 pictograms, every treatment crossed with every pictogram, every state crossed with every pictogram, the hero-layer recipe, the full numeric contract, and the source guidelines' usage rules (`gdsPictogramUsageRules`) — every count and value read from the exports, never retyped. A pictogram composes into `GdsMapPinBadge`/`GdsBadge`'s existing `icon: GdsIconKey | ReactNode` slot unchanged, since every mark is monochrome `currentColor` and adopts the host's state colors automatically.

### A generated brand badge fills the favicon/app-icon gap (#699)

`getGdsWebAppManifest` has shipped a manifest generator since issue 458, but its `icons` array
starts empty — GDS had no artwork of its own to put there, so every consumer had to hand-author
a favicon before their product looked finished in a browser tab. `GdsGeneratedMark` already
rendered the right square-mark artwork, but only as a live-DOM React component whose colors are
CSS `var(...)` references — useless as a standalone favicon file, which has no cascade to
resolve against.

`buildGdsBrandBadgeSvg` (`packages/gds-core/src/generated-brand-badge.ts`, exported from
`@sovereignsquad/gds-core/server` only) is `GdsGeneratedMark`'s headless twin, following the
`generated-art-svg.ts` precedent: literal hex colors via `resolveGdsGeneratedPaletteHex`,
`react-dom/server` for the motif, and geometry (48-unit canvas, gradient direction, motif
proportion, ±20° tilt) held identical to the React component by twin tests rather than shared
code. Unlike the thumbnail/hero builders, its motif sits directly on the raw gradient with
nothing behind it, so it additionally guarantees the white motif clears WCAG 1.4.11's 3:1
non-text contrast floor against the gradient's own midpoint — darkening both stops together, by
just enough, for every built-in preset and both color schemes. A `maskable` option produces a
full-bleed square with the motif scaled into the W3C maskable safe zone; the default is a
rounded square at a `0.25` corner-radius ratio (`GdsGeneratedMark`'s 12px default over its
48-unit canvas). Every badge embeds the resolved `computeGdsThemeIdentity` hash as a
`data-gds-theme-identity` attribute, giving a consumer a ready-made cache key.

Colors resolve exclusively through the governed palette resolver or an explicit `colors`
override — never an invented literal — and every throw case (missing palette source, empty
label, `maskable`/`cornerRadiusRatio` combined, an out-of-range ratio, a non-finite size) is an
actionable error, not a silently wrong badge. The reference site's Theme Lab shows the mechanism
live: every card in the shipped-lanes vibe gallery now carries a `GdsGeneratedMark` badge
specimen seeded and colored from that preset, so a future preset shows its badge automatically.

Validated against the Your Field bundle's three real app-icon variants (navy `#0B223E`, sage
`#90A287`, terracotta `#CA8570`, sourced from `packages/gds-theme/src/your-field.ts`'s exported
constants) at the bundle's own 96×96/22px presentation, and documented end-to-end in
`docs/GENERATED_IMAGERY.md` — the generated default, the favicon and manifest-icon recipes, the
maskable variant, and all three replacement levels, worked through with the real Your Field PNGs
swapped in for the generated default with no GDS code changed.

### `data-gds-fixed-tone` excludes one element from every preset rule

A consumer (classscout, gap-request item 12) rediscovered the same defect six times in six weeks:
the active preset repaints `.gds-paper`/`.gds-card`, `.mantine-Button-root`,
`.mantine-Popover-dropdown`, and the `AppShell` navbar unconditionally, several rules with
`!important`, and an instance styled on purpose had no way out except a counter-rule at matched or
higher specificity. One of those counter-rules tied on specificity and won only by import order; one
shipped at `(0,4,2)` and was silently beaten by a GDS rule at `(0,5,1)`. `Badge` already had the
answer — `:not([data-gds-badge-fixed-tone])` in the preset rule's own selector — for one component.

Every rule gated on `html[data-gds-theme-preset]` now carries `:where(:not([data-gds-fixed-tone]))`
on its subject, except the `body` rules and the forced-colors and reduced-motion blocks. An element
carrying `data-gds-fixed-tone` is not matched by the repaint at all; its descendants are unaffected.
`:where()` contributes no specificity, so no rule moved in the cascade and every existing consumer
counter-rule keeps behaving as before until it is deleted. The Badge rule honours both attributes.

Documented in `THEME_GOVERNANCE.md`, `docs/THEME_STYLING_HOOKS.md`, and
`docs/CLASSSCOUT_INTEGRATION.md` B19. `preset-fixed-tone.test.ts` asserts the contract on every
gated selector in both directions and that no gated rule uses a bare `:not()`; the release-doc
completeness gate requires `llms.txt` and `README.md` to keep naming the attribute.

### A `layout` axis governs shell geometry (#698)

Sidebar width, header/footer heights, nav-item height, content max-width, browse list-rail
width, mobile bottom-tab-bar height, content bottom padding, and bottom-sheet top radius were
the one dimensional concern the theme-axis mechanism did not govern: `DiscoveryShell` hardcoded
`sidebarWidth = 280`/`headerHeight = 60` as prop defaults and a footer height of `68` inline;
`BottomTabBar` hardcoded `BOTTOM_TAB_HEIGHT = 64`. A brand lane had no lever for any of it short
of restating the same prop at every call site, or forking the component.

`GdsLayoutAxis` (`packages/gds-theme/src/axes.ts`) is the ninth key on `GdsThemeAxes`, following
the file's own four-part recipe for adding an axis: a type, `GDS_DEFAULT_LAYOUT_AXIS`,
`validateGdsLayoutAxis`, and a `resolveGdsLayoutTokens` branch in `resolveGdsAxisTokens`. It
resolves nine `--gds-layout-*` custom properties, emitted unconditionally like shape and
density so every preset carries the full namespace whether or not it declares an override:

- `--gds-layout-sidebar-width` (280px), `--gds-layout-header-height` (60px),
  `--gds-layout-footer-height` (68px), `--gds-layout-nav-item-height` (44px)
- `--gds-layout-content-max-width` (1400px), `--gds-layout-list-rail-width` (480px)
- `--gds-layout-bottom-bar-height` (64px)
- `--gds-layout-content-bottom-padding` — derived, `calc(bottom-bar-height + space-xl)`,
  ~96px at defaults, so a lane that raises its bar height gets correct padding for free
- `--gds-layout-sheet-top-radius` — defaults to the shape axis's `sheet` ROLE token
  (`var(--gds-radius-sheet)`), not a step, so repointing `sheet` repaints bottom sheets too

`headerHeight`/`footerHeight`/`bottomBarHeight` enforce the 44px target floor with no exception
path — these regions host interactive 44px targets and cannot be shorter than the targets they
contain. `navItemHeight` enforces the same floor through one recorded exception,
`GDS_LAYOUT_DIMENSION_EXCEPTIONS.navItemHeight`: a dense sidebar nav row may render below 44px
visual height only where the interactive row (full row width plus vertical padding) still
preserves a 44px effective hit target — a consumer adopting a sub-44px value owns that
obligation. `calc()`/`var()` declared values pass through unvalidated, matching the density
resolver. Density mode does not scale layout tokens; shell geometry does not compress at
`compact`.

`DiscoveryShell`'s `sidebarWidth`/`headerHeight` prop defaults and its inline footer height, and
`BottomTabBar`'s bar-height `calc()`, now read the tokens with the pre-token literals as
`var(..., <literal>)` fallbacks — pixel-identical rendering with no GDS theme runtime present,
and an explicit prop still wins. `BOTTOM_TAB_HEIGHT` remains exported as the fallback constant.

A consumer stacking content above a fixed bottom tab bar should read
`--gds-layout-content-bottom-padding` rather than inventing a bottom-padding literal, so nothing
hides behind the bar.

### `HeroSearchPanel` and `QuickStartCard`: home discovery intent surfaces (#710)

Two new public `@sovereignsquad/gds-core` exports, composed entirely from existing primitives and
tokens — neither introduces a new token, preset, or dependency.

`HeroSearchPanel` (`.client.tsx`, holds field-value state) is a card-shaped `<form role="search">`:
a consumer-defined, flex-wrapping row of `FormField`-wrapped text fields
(`GDS_HERO_SEARCH_FIELD_FLEX_BASIS_PX`/`GDS_HERO_SEARCH_FIELD_MIN_WIDTH_PX` govern the wrap), a
primary CTA (`SemanticButton` on the governed `search` vocabulary action, overridable via
`primaryActionLabel`), an optional secondary CTA that renders only when both
`secondaryActionLabel` and `onSecondaryAction` are supplied, and an optional `trustLine` slot.
Controlled (`values`/`onChange`) and uncontrolled (`defaultValues`) modes both hand `onSubmit` the
current record verbatim — no trimming, coercion, or validation — on the primary CTA or Enter in
any field. Duplicate `fields` keys resolve rather than throw: the last definition wins for
rendering, one value slot backs both. Card surface/radius/elevation bind to `--gds-bg-card`,
`--gds-border-card`, `--gds-radius-card`, and `--gds-elevation-card`; fields bind to
`--gds-radius-input` and the existing `GDS_MIN_TARGET_PX` 44px control-height floor.

`QuickStartCard` (server-safe, hook-free) is a single native `<button>` scenario card — an icon
square (`GDS_QUICK_START_ICON_BOX_PX`, `--gds-radius-md`, `--gds-bg-page` tint, icon in
`--gds-brand-primary`), a bold label, and an optional description — so click, Enter, and Space all
activate it through native semantics with no custom key handling. Its hover lift lives in
`packages/gds-theme/styles.css` as a new `.gds-quick-start-card` class, following the
`.gds-tour-btn` precedent for package-level component classes: resting `--gds-elevation-card` to
hovered `--gds-elevation-panel` plus `translateY(-2px)`, transitioned exclusively on
`--gds-motion-duration-base`/`--gds-motion-ease-standard` — use this pair for the shared
base-speed/standard-ease transition rather than a component-local duration, so the lift
disappears under reduced motion with no separate override.

Neither component renders fetched data, so the loading/empty/error/success states contract does
not apply internally — documented as such in `COMPONENTS_AND_PATTERNS.md`. Both are registered in
the pattern catalog (`family: 'public'`, `section: 'Public Product Surfaces'`) with live demos
covering the zero/one/five/ten field counts and the ten-scenario quick-start set, and ship a new
`gds.heroSearchPanel.ariaLabel` message key translated across all 12 locale packs. Component
census 309 -> 311; `docs/AI_AGENT_GUIDE.md` and `llms.txt` both restated the count and are
updated.

### `GdsLogoLockup`, `GdsNotificationBell`, `GdsCompareButton` (#713)

Three small, preset-agnostic `@sovereignsquad/gds-core` exports the Your Field delivery needed
and no governed counterpart existed for.

`GdsLogoLockup` is the real-asset counterpart to the generated-imagery system: GDS shipped only
generated identity art (`GdsGeneratedMark`/`GdsGeneratedAvatar`/`GdsGeneratedThumbnail`/
`GdsGeneratedHero`) with no component for a consumer's actual logo file, so every brand consumer
hand-composed its lockup. It renders a consumer-supplied mark (`src` + required `alt`, or an
arbitrary `mark` node) plus an optional wordmark and badge pill, an `onInverse` variant for dark
grounds, and a `framed` presentation — card surface, border, `gdsRadius('card')`,
`gdsElevation('card')` — that honors the brand-guidelines rule that a real mark always sits on a
light contrasting badge. `src` without `alt` throws in development (mirroring
`GdsSavedIndicator`'s required-label stance); `src` and `mark` together throw as a contract
violation. A broken/slow `src` keeps the lockup's layout — the plain `img` element's native
behavior — and the wordmark never depends on the mark having loaded.

`GdsNotificationBell` is the trigger affordance the Notifications family
(`GdsNotificationProvider`/`NotificationCenter`/`InlineAlert`/`BannerNotice`) had no governed
entry point for: a circular bell button sized to `--gds-control-height-md` (the 44px floor) with
an optional unread dot. The dot is an inline `GdsBadgeShapeCircle` colored via
`var(--gds-badge-attention)` — a filled `currentColor` SVG survives forced-colors mode, unlike a
`background-color`-painted span — with a decorative `box-shadow` separation ring in
`var(--gds-bg-card)`; state is carried by the `aria-label` swap ("Notifications" /
"Notifications (unread)"), never by the dot alone. It composes with
`GdsNotificationProvider`/`useGdsNotifications` by prop (`unread={notifications.length > 0}`) —
it is the trigger only, no dropdown panel of its own.

`GdsCompareButton` is a fully controlled `aria-pressed` compare-before-decide toggle — off/added
states with an icon+label swap (`Compare`/`Added to compare`, the new `Compare` icon key mapping
to `IconArrowsLeftRight`) — a different semantic from `GdsSavedIndicator` (favoriting, icon-only):
the label states the action's current outcome, not just its availability. It deliberately holds
no internal state mirroring the `added` prop, unlike a known bundle defect this component exists
to not repeat, so an externally driven state change always re-renders correctly.

All three bind only existing `--gds-*` semantic-role tokens (zero raw hex/rgb, zero bare themed
pixel literals) and ship colocated tests, pattern-registry demos (`logo-lockup`, extending
`notifications` and `badges`), export-coverage entries, and new `gds.notificationBell.*`/
`gds.compareButton.*` message keys translated across all 12 locale packs. Component census
313 -> 316.

### `DetailFactsTable` and `ProviderCTA`: detail-page facts and provider-claim surfaces (#711)

Listing detail pages had two surfaces GDS could not express from package exports alone: a
key-facts block whose rows are guaranteed present even when data is missing, and a provider-claim
prompt. Every prior option was page-local composition — exactly what Rules 10/15/16 forbid on the
reference site.

`DetailFactsTable` renders real `<dl>`/`<dt>`/`<dd>` semantics (never `div`s styled to look like a
table) for a fixed nine-fact schema — age range, activity type, format, location, indoor/outdoor,
price, booking, source, and last checked — via the exported `GDS_DETAIL_FACT_IDS` /
`GdsDetailFactId` contract. All nine rows always render: a missing, `null`, or whitespace-only
value renders the localized unknown phrase in place of the value, never a blank cell or a dropped
row, so a reader can always see a listing's data provenance and freshness. A `facts` prop replaces
the schema entirely for a custom fact set, with the same never-omit-a-row substitution applied per
row. The label column's width ships as the exported, documented `GDS_DETAIL_FACTS_LABEL_COLUMN_PX
= 130` constant rather than a literal repeated in prose (Rule 14). Server-safe: no client-only
behavior.

`ProviderCTA` is a calm, factual "are you the provider?" panel — headline, body copy, and a
primary + optional ghost action pair — composed from the governed button lane (`SemanticButton`,
`CtaButtonGroup`, `createGdsVocabularyPack`) rather than reimplemented locally; the ghost action is
Mantine's own `variant="subtle"` lane on the same governed button. It fires only the `onAction` /
`onSecondary` callbacks it is given — no claim workflow, navigation, or data mutation of its own.

Both compose as `DetailProfileShell` sections in `page` and `drawer` modes, take every visual
value from governed tokens (`--gds-bg-card`/`--gds-bg-info-tag`, `--gds-border-card`,
`--gds-radius-card`, `--gds-text-primary`/`--gds-text-secondary`), and ship colocated tests, a
pattern-registry demo (`detail-facts-provider-cta`), export-coverage entries, and new
`gds.detailFacts.*` / `gds.providerCta.*` message keys translated across all 12 locale packs.
Component census 316 -> 318.

### A governed trust-layer component family (#709)

The Your Field product (successor to ClassScout) aggregates children's-activity listings from
external sources — provider websites, public feeds, provider claims — data that is inherently
uncertain: prices change, schedules move, ages go unconfirmed. The product's trust strategy is
honesty about that uncertainty, and until now that honesty existed only as a reference
implementation outside this repository (inline styles, hardcoded `$` currency, English-only
copy) — exactly the kind of local composition Rules 10/15/16 forbid on the reference site and
that would otherwise leave every consumer to hand-roll its own provenance badges.

`TrustBadge` renders a closed, eight-label operational vocabulary — `official_source`,
`public_source`, `provider_claimed`, `recently_checked`, `price_estimate`, `schedule_estimate`,
`age_not_confirmed`, `reported_outdated` — through the exported `TRUST_BADGE_DEFINITIONS`
tone/icon/message-id mapping (Rule 14: documentation derives the label table from this export,
never retypes it); an unrecognized runtime value falls back to `public_source` rather than
rendering unstyled. `PriceEstimateLabel` states price certainty instead of implying a fixed
number — free, an unknown price that asks the reader to confirm, a provider-confirmed amount, or
an estimate — formatted through `formatGdsCurrency` with no hardcoded currency symbol; an
explicit `0` always wins as "Free" even when `status` is `'unknown'` (branch order is contract).
`LastCheckedLabel` states when data was last checked, or a stale caution the consumer sets
explicitly — GDS never computes a freshness window itself. `ReportOutdatedLink` is a real
`<button>` with a one-way idle-to-sent transition: activating it swaps in a persistent thank-you
confirmation, announced through an always-present polite live region, and fires the
consumer-owned `onReport` exactly once even under two activations in the same tick (a ref guard,
not just state). `SourceBlock` is a detail-page information-source card: a title, four rows that
are never omitted — an absent value states the unknown wording explicitly rather than dropping
the row — a standing "confirm with the provider" line, and the embedded `ReportOutdatedLink`
when `onReport` is set; its source-type link goes through the existing `GdsInlineLink` primitive,
which already applies the safe `rel` for an external target. `ConfirmChecklist` is an amber
check-before-booking card — the warning-tint tone pair (`--gds-badge-soft-warning` background,
`--gds-state-warning` border, `--gds-badge-soft-warning-fg` text) so it stays legible by
construction across every preset and both color schemes — with six default items; checking one
applies a line-through and reduced emphasis while it remains a real, labelled, checked checkbox,
and an explicit empty `items` array renders nothing. Every interactive target (the report button,
each checklist row) meets the governed 44px minimum.

The vocabulary is deliberately humble: a test asserts no rendered default string in the family
contains "verified", "safe", "guaranteed", "best", or "perfect". Seven new `GdsIcons` registry
entries back the family (`Confirmed`, `Freshness`, `Stale`, `Price`, `Schedule`, `Checklist`,
`SourceInfo`), and the full `gds.trust.*` copy contract ships translated across all 12 locale
packs. Stateless (`TrustBadge`, `PriceEstimateLabel`, `LastCheckedLabel`) and stateful
(`ReportOutdatedLink`, `SourceBlock`, `ConfirmChecklist`) halves split across `TrustLayer.tsx` /
`TrustLayer.client.tsx` per this repo's `.client.tsx` convention. Reference-site live proof
(`trust-layer`) composes all eight badge labels, all four price states, fresh and stale
freshness, the report action pre- and post-activation, `SourceBlock` with real data and with
every value absent, and the checklist at zero/one/six items. Component census 318 -> 324.

### Sidebar/pin elevation roles, role-level shadow values, validated tracking, and italic typography inputs (#695)

An externally designed brand system due later in this release declares visual decisions the
theme-axis mechanism could not yet represent: a 240px sidebar shell carrying a directional
shadow (`2px 0 16px 0 rgba(13,35,64,0.04)`, cast sideways onto the page canvas) and map pins
carrying a dedicated shadow (`0 2px 8px 0 rgba(11,34,62,0.1)`) — neither `sidebar` nor `pin` was
a `GdsElevationRole`, and a directional shadow cannot honestly occupy a slot on the shared
5-step elevation ramp every other surface resolves through. This is pure mechanism work in
`packages/gds-theme/src/axes.ts`: no preset declares any of these inputs yet.

`GdsElevationRole`/`GDS_ELEVATION_ROLES` gain `sidebar` and `pin`, appended after `tooltip` so
existing enumeration order is preserved; `GDS_DEFAULT_ELEVATION_AXIS.roles` maps both to step 1,
the resting-surface step `card`/`panel` already use, so a preset declaring nothing renders
exactly what it renders today once a consumer adopts the role token. `GdsElevationAxis.roles`
widens to accept `GdsElevationStep | GdsElevationValue` per role — mirroring how the shape axis
already lets a radius role carry a literal value — so a theme can either pin a role to a shared
step or declare its own directional shadow (or `{ kind: 'none' }`) without touching the
monotonic step ramp at all; `resolveGdsElevationTokens` runtime-guards the role key set (a JSON-
derived theme bypasses TypeScript, so a typo cannot become a silently unstyled surface) and
still emits a dense token for every role, undeclared roles falling back to `defaultStep`.

`GdsTypographyAxis`'s `tracking` map (`--gds-tracking-<step>`) previously emitted any string
unchecked — a malformed value shipped as a silent visual defect rather than failing the build,
the exact failure mode the axis validators otherwise prevent. It is now validated against
`normal`, a signed px/rem/em/ch length, or a `var()` reference (percentages are deliberately
excluded — `letter-spacing` does not accept one). A new `fontStyles?:
Partial<Record<GdsTextSizeStep, GdsFontStyle>>` input (`GdsFontStyle = 'normal' | 'italic'`)
resolves to `--gds-font-style-<step>`, emitted only for the steps a theme declares — a consumer
reads `var(--gds-font-style-<step>, normal)`.

No component adopts any of these tokens yet — `GdsMap.client.tsx`, shell/sidebar components, and
card/thumbnail components are unchanged; that wiring is follow-on work later in this delivery.
`GdsShapeElevationSystemReference` iterates `GDS_ELEVATION_ROLES`/`GDS_RADIUS_ROLES` directly, so
the new `sidebar`/`pin` elevation rows (and the pre-existing `thumbnail` radius role, now
test-proven to resolve distinctly from `image`/`card`) appear on the reference site from the
shipped constants alone, with no playground-local code.

## 6.7.0 - 2026-08-27 — Padel Africa preset, ListingCard media overlays, and a primary-CTA contrast rule (#678, #679, #680)

### `padel-africa` joins the preset catalog (#678)

A 26th selectable preset carrying colour and the brand's layout defaults, transcribed from the
brand's own Website Colour Theme Guide. `getGdsThemePresets()` returns it,
`resolveGdsThemePreset('padel-africa')` resolves it, and it has a full vibe entry, so it takes part
in every preset-combinatorial gate.

Ramps are explicit rather than interpolated. `createBrandTheme`'s five-ramp form blends between the
values it is given and, for this palette, produced `#b56e57` — a colour in no brand asset — which
`primaryShade` then painted on every link and button. Each colour is registered as a ten-step ramp
with the brand value written in verbatim at the painted index, asserted per colour.

One contradiction inside the guide is resolved explicitly rather than silently: its "Best
Variations" table assigns Lime to the active tab/pill, but every rendered example in the same guide
shows an Emerald-filled active pill, and both the Application Guidelines and the DO list assign Lime
to focus and motion. The rendered UI and the prose agree against that one cell, so `activePill` is
Emerald and `focusRing` is Lime.

All 16 ratcheted budgets held, including the three sitting at zero slack — the preset introduced no
untokenized values. `vibe-themes.test.ts` asserted a literal `25` presets and went stale the moment
one was added; it now derives the count.

### ListingCard gains media overlays (#679)

A theme's `components:` block carries colour, radius and size defaults; it cannot make a card grow a
category pill over its photo. So the two pieces the brand needs live in the component (Rule 16):
`mediaOverlay` for content on the media's top-left, and
`ListingCardAffordance.presentation: 'outline-on-media'`, which drops the disc entirely and carries
its own drop-shadow so a save control stays legible directly over a photograph. Both token-driven.
Additive: the positioned wrapper renders only when something is actually overlaid.

### A floor rule for primary-CTA contrast, reporting rather than blocking (#680)

Nothing in GDS measured the contrast of a primary CTA's label against its fill. `assertContrast`
gates text on page/surface/inverse and *derives* a passing foreground for support; no floor rule
covered the action fill; and `verify:component-color-pairs` only sees pairs a component declares in
source. A preset could ship a CTA below AA in silence.

`primary-cta-text-contrast` closes the measurement gap, using a new `report` severity that never
fails a build. `auditGdsAccessibilityFloor()` returns `reports` alongside `violations`, so existing
zero-violation checks keep their meaning.

Reporting is not a courtesy to the new preset. The rule immediately found **20 preset/scheme
combinations below 4.5:1 in the existing set** — `neon-night` at 1.98:1, `athlete-gold` 2.15:1,
`sunset` 2.8:1, `forest` 3.3:1 among them. Enforcing would have retroactively failed roughly eight
shipped presets, which is its own remediation project. That standing is now recorded so a future
decision to enforce has a baseline. `padel-africa`'s Emerald sits at 3.62:1, with the in-palette AA
alternative the guide already names as CTA hover (Forest Green, 9.55:1) exported as
`PADEL_AFRICA_ACCESSIBLE_CTA`.

The canary gained a breaching primary so the ninth rule is proven live like the other eight, and
`docs/ACCESSIBILITY_FLOOR.md` is regenerated from the rule set.

### Known issue in this release

The preset's catalog label and description, and two phrases from the previous release, ship as
English fallbacks in the eleven non-English locale packs. `translate.googleapis.com` returned `429`
to every attempt across two days; the #660 safety mechanism correctly preserved committed
translations rather than overwriting them, and `verify:i18n-leakage` passes. Per #668 a plain re-run
will not repair them once the limit clears — the keys must be deleted first so the generator treats
them as missing.

## 6.6.0 - 2026-08-25 — Consumer-adoption gap analysis, compliance false-positive fixes, and the last two /foundations axes (#661, #666, #670-#677)

### Button micro-feedback gets its own pattern

`SemanticButton`'s transient success/error treatment had no page of its own — it was one sentence
inside another pattern's summary. `GdsButtonFeedbackReference` is a live, clickable proof plus the
per-action feedback table, derived from `GdsVocabulary` at render time. Component census 308 -> 309;
`docs/AI_AGENT_GUIDE.md` and `llms.txt` both restated the count and are updated.

`GDS_BUTTON_FEEDBACK_DURATION_MS` is extracted from the literal inside `SemanticButton`'s effect
and exported, so the page reads the real revert duration instead of restating it (Rule 14).

The page reports what each action declares for its feedback colour and defers to the live proof
for what paints, because the rendered colour legitimately depends on the active preset's governed
Button rules. It does not assert a colour per preset.

An earlier draft of this entry claimed the declared colour never renders, citing measurements
that turned out to be invalid: they came from a browser pane whose page was never painted
(`visibilityState: "hidden"`), where Chrome throttles style recalculation and `getComputedStyle`
serves the value from first paint. An inline `background` override read back correctly while
computing to the stale colour, and a scan of all 1572 matching rules found no applicable
`!important` that could have produced the reported result. The claim, and a speculative
`styles.css` change written against it, were both withdrawn before landing; #677 is closed with
the full account.

### `gds-compliance` raw-color scan: `var()` fallback exclusion, and `themeOwnershipPaths` now honored outside strict mode (#670)

Two related false positives in `strict.raw-color`/the always-on `forbidden-color` check, found
auditing how messmass, management, classscout, camera, savetheworld, salesleadgenerator, and sso
actually use GDS. Both are fixed at the shared `RAW_COLOR_PATTERN` matching layer, not per rule.

`color: "var(--mantine-color-dimmed, #868e96)"` (management's real violating line) used to trip
`forbidden-color`/`strict.raw-color` even though the actual styling authority is the CSS
variable and the hex is only a defensive pre-hydration fallback. `strict.inline-color` already
excluded `var(...)` values for JSX `style={{}}`; the plain-text scan now does the same via a new
`stripVarFallbacksForColorScan()` helper, for both the basic and strict rule paths. A nested-paren
fallback (`var(--x, rgba(0,0,0,0.5))`) is a known, tested v1 limitation and still flags.

Separately: `COMPLIANCE_TOOLKIT.md` has always documented that `compliance.themeOwnershipPaths`
exempts a raw-color literal from the scan -- true for `strict.raw-color` (`isStrictThemeOwnedPath`,
now renamed `isThemeOwnedPath` since it's shared), but never actually implemented for the
always-on `forbidden-color` check that every non-strict consumer runs (`strictMode` is `false` or
unset for all seven audited repos). `scanSourceFile()` now receives the manifest and honors
`themeOwnershipPaths` there too, so a categorical/data-visualization color palette (messmass's
`lib/chartTheme.ts`) can be declared out of scope the same way strict-mode consumers already
could, rather than the documented behavior silently not applying to anyone actually using it.

### Pattern matrix links to the actual pattern, not just plain text (#666)

The "Pattern matrix" table on `/coverage` showed each pattern's name and route as inert text
-- `entry.route` names only the family page, with no way to reach the specific pattern from the
matrix. Each pattern name is now a real link to its family page's `#entry-<anchor>` deep link
(the same id `PatternEntryCard` already sets), using the registry's own canonical `anchor`
field rather than reaching for `.id` directly. Verified live: 119 links generated, each
resolving to the correct family route.

The evidence table on the same page states `route` too but has no `family` field to build the
same link from without changing its own registry shape -- left as plain text rather than
shipping a broken link; not in this pass.

Confirmed while fixing this: internal `#entry-*` hash-anchor links (this one and the existing
`components-index-page.tsx` links) don't scroll to their target after a client-side navigation
-- a pre-existing router-level gap, not introduced here. Tracked separately (#667).

### Site UX defects: zero-gap cards, slider mark overflow, redundant Theme Toggle box (#666)

Reported live against the deployed site with annotated screenshots. Four defects, all traced
to shared components:

Every stacked pattern-demo card on `/components` rendered flush against the next with 0px
gap, site-wide -- `SurfacePresentation.ts`'s `inline` mode (the default) returned only
`{minHeight}`, no flex or gap at all. Every presentation mode is now a flex column with a
governed `var(--mantine-spacing-lg)` gap; a lone child is unaffected since gap has no effect
with nothing to space.

`GdsRatingScale`'s boundary marks overflowed the Slider track by 4px on each side -- Mantine
centers its own dot marker on each mark's position, and the label inherited that shift with no
compensation. Fixed with two governed CSS rules in `packages/gds-theme/styles.css`; verified
0px overflow both directions.

The `theme-toggle` pattern demo wrapped one `<ThemeToggle />` button in a second, near-duplicate
card -- its own outer card already carried the title and description. Now renders the button
directly, matching every other single-component demo entry.

`FormControlFamilyDemo` (Segmented control / Slider and rating / Wizard stepper / Date and time
inputs) used a bare unstyled `<div>` and `<br />` tags for layout -- FOUNDATION.md's own
prohibited pattern. Replaced with `GdsStack`, the governed primitive already used throughout
the file. This was also the reason the site-wide gap fix alone didn't reach these four cards --
they were nested two levels inside the fixed container via the unstyled divs.

18 more `<br />` layout instances remain elsewhere in `pattern-pages.tsx`, not swept in this
pass -- tracked, not silently left.


### GdsColorSystemReference and GdsAccessibilitySystemReference close /foundations to all 7 axes (#661)

`/foundations` claimed seven axes in its own description; two -- colour/theming and
accessibility -- had no reference component at all, confirmed live and traced to source.

`GdsColorSystemReference` groups every `BrandSemanticRole` by its 60-30-10 proportion
classification with live-resolved swatches, states the governed accent axis (names, shades,
per-mode enforcement), and embeds the existing `GdsAccentContrastMatrix` rather than
re-deriving contrast, so this page can never disagree with `verify:accent-contrast`.

`GdsAccessibilitySystemReference` lists every governed accessibility-floor rule and calls
`auditGdsAccessibilityFloor()` directly for a live "Holds"/violation-count verdict across all
25 presets x 2 schemes -- the same function `verify:a11y-floor` runs.

Both caught and fixed their own layout defect before shipping: a bare Mantine `Table` cell
holding a long sentence gets squeezed to whatever width the other columns leave it -- measured
live at 52px wide, wrapping one rationale into a 452px-tall row. Fixed with `SimpleDataTable`
plus an explicit minimum width, and a 2-row key/value table replaced with a plain `Stack` where
a table wasn't the right structure to begin with.

Component census 306 -> 308; `docs/AI_AGENT_GUIDE.md` and `llms.txt` both restated the count
(one gated since #665, so it failed loudly on the stale number rather than being missed).

## 6.5.0 - 2026-08-24 — Accessibility floor sweep, an opt-in lazy locale registry, and hardened release tooling (#629, #632, #641, #656-#662)

A cross-cutting maintenance release: the touch-target floor sweep closed its two largest
contributors (NavLink and ReferenceLinkGrid, 791 -> 268 violations), `gds-core` gained an
opt-in lazy locale registry so a consumer can trim its bundle instead of shipping all twelve
locales, and three release-chain defects were fixed at the source — a suite that read as
flaky was actually worker oversubscription, a documentation version gate governed 14 of 43
documents that claim to be current, and the translation-pack generator could silently
overwrite committed translations with English on a network failure. Three closed p0 tracking
issues (#573, #576, #577) were reconciled against what had actually shipped.

### Obligation coverage, first tranche (#656)

`PartnerDiscovery.tsx` carried 51 of the 404 unmet obligations, the largest of 72 contributing
files. Its 48 prop atoms now document what the name cannot: which props are controlled and must
be supplied in pairs, that the map adapter renders but never filters, that `backHref` is a href
rather than a callback so the back control stays a real link, and that every `onEvent` is a
privacy-safe hook with no built-in destination. The 3 remaining atoms are `shareState` variants
needing a demo rather than a comment.

`obligationGaps` 404 -> 356 (prop 726/1054, variant 69/107, accent 10/10).

### Opt-in lazy locale registry (#662, Fix 2 of #532)

`@sovereignsquad/gds-core/locales/lazy` and its eleven per-locale subpaths let a consumer
register only the locale dictionaries it ships, instead of `getGdsMessages`'s default eager
import of all twelve. `getGdsMessagesLazy` keeps the same synchronous signature and
fallback-to-English behavior — no async, no Suspense — the difference is what makes a locale
available: importing its subpath, not shipping automatically. `locales/lazy/all` restores
today's coverage in one import for a consumer that wants the registry without trimming yet.

Measured directly rather than assumed from tree-shaking theory (esbuild, minified, isolated
from the rest of the client bundle): the eager path is 202.7KB; one locale through the lazy
registry is 11.3KB.

Purely additive — `getGdsMessages`, `gdsLocales`, and the default `GdsProvider`/`GdsI18nRuntime`
behavior are unchanged. Migrating the default itself is a separate decision, deliberately left
open (`docs/I18N_RUNTIME.md`), because it would silently change every existing consumer's
locale fallback behavior across GDS's roughly ten downstream products.

### ReferenceLinkGrid's two links meet the 44px touch-target floor (#659)

Traced 286 of the 564 remaining touch-target violations to one shared component,
`ReferenceLinkGrid.tsx` (used across seven playground surfaces): its card-title link and
"Open section" footer link are both 20-25px tall text. Fixed with the same invisible-padding
technique issue 628 established for the checkbox and swatch grid — `.gds-touch-target-pad-link`,
`inset: -12px`, no visible layout change. Safe here in a way the swatch grid was not
automatically safe: each anchor sits inside `Paper p="lg"` in a `SimpleGrid spacing="lg"`, so
the padded hit area has no neighbor within 12px in any direction.

`touchTargetFloorViolations` 564 -> 268.

### Touch-target findings carry their classification inputs (#659)

The runtime sweep records each finding's `parent` and computed `display` — the two inputs the
WCAG 2.5.8 inline-link exception reads — so a violation can be classified from the committed
artifact without re-running the browser sweep.

### The documentation version gate is derived, not listed (#658)

`verify-docs-governance-consistency.mjs` governed a hand-written array of 14 documents. 43
declare `Status: Active SSOT`, so 29 were ungoverned and 30 headers had drifted — most sitting
at `6.0.0` across four releases.

The governed set is now derived from that declared status, plus two documents governed by
explicit exception with their reasons in source. Documents declaring `Planned`, `Proposed`,
`Decision record` or `Active reference` stay ungoverned by their own statement, and the gate
prints them on every run so the exclusion set cannot grow silently. A floor check refuses to
pass if the derivation collapses.

48 documents are now governed and stamped `6.4.0`. `docs/DEEP_AUDIT_PLAN.md` and
`docs/HEALTH_RETENTION_PLAN.md` had statuses that were no longer true (`Approved plan, not yet
executed`, `Proposed`) and now state their delivered state.

Carries the first mutant on a `verify:references` sub-gate, planting a stale version on a
document the old array never covered.

### Translation generators no longer degrade committed packs silently (#660)

`scripts/lib/translate.mjs` calls a network endpoint and returned the English source on any
failure, indistinguishable from a real translation. `npm run artifacts:refresh` on a
rate-limited or offline machine therefore rewrote shipped translations in English with no
warning and exit code 0 — observed replacing `Layoutschema JSON` and `Overlay-Stack-Governance`
with their English sources.

`translate()` now resolves `{ text, ok }`. `chooseTranslationValue()` holds the invariant that a
failed call never replaces a stored translation. `generate-site-phrase-translations.mjs`
preserves committed values, reports attempted/succeeded/failed counts, and exits non-zero above
a 10% failure rate. `generate-component-message-packs.mjs` defers a failed id instead of
writing English into a new entry, which the next run would have skipped as already-present —
making "gets fixed on the next run" true rather than assumed.

### NavLink meets the 44px touch-target floor (#629)

`gdsTheme.components.NavLink` resolves `min-height` from `--gds-control-height-md` rather than
inheriting Mantine's 41px default. No number is restated: the density axis already declares
`controlHeights.md` as 44px and clamps it to `GDS_MIN_TARGET_PX` under every density mode, so
NavLink was the only piece missing — it had no theme entry consuming the token. Propagates to
every lane through `extendGdsTheme`, asserted per lane in `theme.test.ts`.

Measured across the same 26-route sweep: `touchTargetFloorViolations` 791 -> 564, with
NavLink-root findings at zero. Visible change is 41px -> 44px per navigation row.

The `221x25` instances #629 also attributed to NavLink are Mantine `Text` rendered as a link —
a different component, carried separately.

### Version-tracking gaps found while cutting the release, fixed at the source

`apps/playground/src/site-copy.ts`'s `stableGdsVersion`/`targetGdsVersion` are hand-restamped
literals, checked by `verify-install-bootstrap-docs.mjs` (matching this repository's existing
convention for the version-pinned install commands in `RELEASE_PUBLISH.md` and
`INSTALLATION_GUIDE.md`); missed on the first pass and caught by this release's own preflight.

`page-copy.ts`'s per-locale install `eyebrow` field was a hardcoded `'6.4.0 public install
path'` literal, inconsistent with its own sibling `lead`/`upgradeSectionTitle` fields in the
same object, which already interpolate `targetGdsVersion`. Converted to match — one release-time
literal fewer to forget. `info-pages.test.tsx`'s assertion against the rendered install page was
the same class of defect as the docs-governance mutant above (a version frozen in a test that
breaks at the next bump): now asserts against the live `targetGdsVersion` constant instead.

`installCopy`'s per-locale "What changed in 6.4.0" narrative section is real authored content
describing that specific release, not a version-tracking literal — left as-is rather than
mechanically renamed to a release it doesn't describe; tracked as #664.

Also fixed: eleven `package.json` files carried internal cross-package pins
(`@sovereignsquad/gds`'s own `dependencies` on `gds-admin`/`gds-core`/`gds-theme`, and similar
pins in `gds-admin`, `gds-core`, and all three apps) that `check-release-alignment.mjs` does not
check — caught only by `npm install`'s own workspace resolution failing on a version no sibling
package declares anymore. Left unpublished stale, this would have shipped `@sovereignsquad/gds`
depending on the *previous* release's sibling packages under a new version number. Tracked as
#663 to close the gate gap for the next release.

### The docs-governance mutant's anchor now tracks VERSION (#663, self-inflicted, caught by this release's own preflight)

The mutant added for #658 hardcoded `Version: 6.4.0` as its plant target on `docs/BADGE_SYSTEM.md`.
Cutting 6.5.0 restamped that file to `6.5.0`, so the anchor no longer existed and the mutant
went `INVALID` rather than `KILLED` — this release's own preflight caught it before it reached
`main`. The anchor and its replacement now read the live `VERSION` file, so the mutant survives
every future version bump instead of expiring at the next one.

### Test suite determinism (#641)

`vitest.config.ts` caps `maxWorkers` at 4. Every test file boots a full jsdom + React +
Mantine tree, so a worker costs memory rather than CPU, and one worker per core oversubscribes:
on a 10-core/16GB machine the default measured 121s wall / 551s test time with 8 timeout
failures, against 39s / 58s and no failures at 4. Five consecutive full-suite runs at load
average 9.91 pass 856/856. No retry mechanism was introduced.

## 6.4.0 - 2026-08-21 — Design Rule Profiles: computed 60-30-10, type-scale, and color-harmony as a gated theme axis (#643-#653)

A new, optional eighth theme axis lets a theme declare which established design-quality
conventions it follows, computed from real values rather than asserted, with enforcement and
measurement at every layer GDS can actually see. Full narrative and worked adoption example:
[`docs/DESIGN_RULE_PROFILES.md`](docs/DESIGN_RULE_PROFILES.md).

### The `GdsDesignRuleProfile` contract (#643)

`GdsThemeAxes.designRuleProfile?: GdsDesignRuleProfile` (`packages/gds-theme/src/axes.ts`)
carries a color-proportion rule (`'60-30-10' | 'none'`), a color-harmony classification, a
named modular type-scale ratio, and a WCAG contrast target. `GDS_DEFAULT_DESIGN_RULE_PROFILE`
is additive and optional — every existing theme already satisfies it with zero behavior
change. `validateGdsDesignRuleProfile` throws a single `GdsAxisError` on the first violation,
matching this file's existing shape/density axis validators.

### Classification, computed not hand-typed (#644, #645, #646)

- `resolveGdsColorProportionProfile(presetId)` classifies every `BrandSemanticRole` into
  `dominant`/`secondary`/`accent`, shared identically across all 25 presets.
- `resolveGdsTypeScaleProfile(presetId)` names the live modular type-scale ratio
  (`GDS_DEFAULT_TYPOGRAPHY_AXIS.scale.ratio`) against the six historically named ratios.
- `resolveGdsColorHarmonyProfile(presetId)` computes each preset's primary/accent hue-angle
  relationship (monochromatic/analogous/triadic/split-complementary/complementary/custom)
  from real hex values — never a hand-assigned label.

### Enforcement (#647, #648, #652)

- `@sovereignsquad/gds-eslint-config`'s new `no-accent-as-background` rule (opt-in via
  `accentBackgroundVariables`) flags accent-classed `--gds-*` tokens used as a background
  fill.
- `createBrandTheme`'s three overloads accept an optional `designRuleProfile`, default to the
  computed profile for a named preset, and return it on the result; a dev-only console warning
  fires once when `overrides` sets a background to an accent-classed color.
- `gds-compliance check-design-rules` scans committed source for the same misuse plus any
  `createBrandTheme(...)` call missing a `designRuleProfile`, both `warn`-severity by default;
  `compliance.designRuleProfile.enforced: true` in `gds-adoption.json` promotes
  `accent-as-background` to a blocking `error`.

### Measurement (#649, #650)

`npm run audit:design-rule-coverage` captures every visible element's rendered
background-color area across 25 presets × 2 schemes × 4 reference-site routes via headless
Chrome, classifies each against the #644 role split, and area-weights the result into
`audit/design-rule-coverage.json` — a real, honestly-scoped measurement (methodology and
known limitations recorded in the artifact itself), not an estimate. `audit/budgets.json`'s
`designRuleUnclassifiedRate` tracks the worst-case unclassified percentage across all 25
presets, `advisory: true` pending a full release cycle of stability.

### Seeing it live (#651)

`GdsDesignRuleProfilePanel`, wired into `ReferenceThemeExplorer` on
[`/themes`](https://sovereignsquad.github.io/general-design-system/themes), renders declared
role classification alongside measured rendered reality as two donut charts, plus type-scale
and color-harmony badges, updating live on preset switch — every number read at render time
from the live resolvers and a generated copy of the coverage artifact.

### Documentation (#653)

New [`docs/DESIGN_RULE_PROFILES.md`](docs/DESIGN_RULE_PROFILES.md): research grounding (the
60-30-10 rule, named type-scale ratios, color-harmony classification, cited), the full
contract, how classification/enforcement/measurement work, the Theme Lab, a worked adoption
example, and an FAQ. `THEME_GOVERNANCE.md`'s design-rule section is now a pointer plus the
standing governance rules. `README.md` and `llms.txt` each gained one new entry.

## 6.3.0 - 2026-08-17 — ClassScout remaining gap-request items (#642)

### Read-only rating display: `GdsRatingDisplay` (item 11, #642)

For a display-only rating (a star glyph plus a value and review count) — `GdsRatingScale`
is a `GdsSlider` preset for *choosing* a rating, the wrong tool for showing a fixed one.
`GdsRatingDisplay` takes `value`/`max`/`count`, renders filled/half/empty glyphs colored
from the existing `--gds-star` token, and exposes one accessible name for the whole rating
rather than one per star. New `StarFilled` icon key added to the governed icon dictionary
to support the filled glyph.

### Compact `BannerNotice` variant and a narrow aside layout token (#642)

- `BannerNoticeProps.variant?: 'panel' | 'compact'` — `compact` is a one-line centered
  status strip with no title/eyebrow/badge, for a page-level status line that doesn't
  warrant a heading. `title` is now optional (required only for the default `panel`
  variant).
- `GdsLayoutSize` gained an `'aside'` step (`18rem`), between the spacing-scale tokens and
  `'content'` (42rem), for a filter rail or detail-page aside narrower than any existing
  named content width. Available on `GdsSidebar.sidebarWidth` and everywhere else
  `GdsLayoutSize` is accepted (`maxWidth`, `minColumnWidth`, `size`).

### Raw token scale vs. Mantine-rendered scale, documented (item 8, #642)

`THEME_GOVERNANCE.md` now states plainly that a brand theme's raw `--gds-*` custom
properties (the public design-intent scale) and its `mantineTheme` object's own scales
(`theme.radius`, `theme.fontSizes`) are two intentionally different, independently-set
scales — not a bug, not pending reconciliation. Also documents that
`resolveGdsTypographyTokens()`'s `tracking` output is not included in `createBrandTheme`'s
`cssVariables`, so a consumer wanting it applied to the document wires it themselves.
Doc-only; no code change.

### Stale consumer-facing docs corrected

- `README.md`, `INSTALLATION_GUIDE.md`, `CLIENT_UPGRADE_PROMPT.md`: corrected the claim
  that GitHub Packages installs work "even though the packages are public" — all seven
  `@sovereignsquad/*` packages are currently private to the `sovereignsquad` org, so a
  consumer token needs org read access in addition to `read:packages` scope.
- `INSTALLATION_GUIDE.md`, `VERIFIED_CONSUMER_INSTALL_PROOF.md`: corrected a stale "current
  major line: 3.0.x" to `6.x`, matching `COMPATIBILITY_AND_RELEASES.md`'s SSOT value.
- `docs/CLASSSCOUT_INTEGRATION.md`: bumped the intro/install pin from 4.0.0/6.0.0 to 6.2.0;
  added B12 (`GdsIconBadge`), B13 (`GdsRemovableTag.disabled`), and B14 (generated-imagery
  tint/`onSelect`) — three 6.2.0 additions (#638) that had no integration-guide section.

## 6.2.0 - 2026-08-17 — ClassScout upstream asks, and Foundations rebuilt to the 7 axes (#632, #633-#638)

### ClassScout upstream asks: BottomTabBar renderItem, tag disabled, icon badge, Meter, imagery tint (#638)

Five small package changes plus a documentation fix, from ClassScout's own gap-request audit:

- `BottomTabBarProps.renderItem?: (item, active, emphasized) => ReactNode` — per-item render
  override, mirroring `PublicNav`'s existing `renderLink`.
- `GdsRemovableTagProps.disabled?: boolean` — native `disabled` + `aria-disabled`, matching the
  existing convention on other native-button GDS components.
- `GdsIconBadge` (new): icon-only categorical-accent disc, decorative by default. A separate
  narrow component, not an extension of `GdsBadge`, which deliberately requires `label`.
- `GdsMeter` (new): a static score/measurement with real `role="meter"` semantics, built on
  Mantine's low-level `Progress.Root`/`Progress.Section` compound API — the top-level `Progress`
  component silently drops a passed `role`.
- `GdsGeneratedThumbnail`/`GdsGeneratedHero`: `tintWithBackground`/`mixRatio` mix the resolved
  palette toward another color (`color-mix()` live-DOM, `mixCssColors` for the literal-hex
  path); category badges gain `onSelect`, rendering as a real `<button>` when given.
- `GdsShapeElevationSystemReference` now computes and states how many of the 14 radius roles
  currently share a value, and documents that a theme's Mantine `theme.radius` is a separate
  mechanism from GDS's own shape axis — the two scales are not required to align.

### Rebuild Foundations to the 7 axes (#632, #633-#637)

`/foundations` previously held 30 entries unrelated to `docs/SITE_ARCHITECTURE.md`'s own
"Target shape" table (7 axes: colour/theming, typography, density & spacing, shape &
elevation, motion, icons, accessibility). Fixed:

- `PatternFamily` gains `components` and `systems`, each hosted at its existing top-level
  route (same `legacyPaths` redirect mechanism `/foundations` already used). 25 entries moved
  to where they fit: 4 to `systems` (badges, fit-score-chip, meaning-badge, generated-imagery),
  19 to `components` (Controls & Inputs, Shell & Navigation), 5 to `operations`'s Workflow
  Guidance, 1 to `public`'s Discovery & Listing.
- `/components` and `/systems` became real hosting pages via a shared `FamilyEntryBrowser`
  (extracted from `PatternFamilyPage`).
- Four new `foundations` registry entries, each backed by a new `packages/gds-core` reference
  component reading live from its axis's resolver rather than a hand-typed value:
  `GdsShapeElevationSystemReference`, `GdsDensitySpacingSystemReference`,
  `GdsIconSystemReference`, `GdsTypographySystemReference`.

Remaining: a compact colour/theming entry linking to `/themes`, and Phase 6 (accessibility).

### Fix a decoy section name and give every family page a filter and jump-to-section grid (#631)

A reader looking for the `SemanticButton`/`ActionBar` click-feedback micro-interaction couldn't
find it: `/foundations` has a section titled "Motion & Micro-interactions" that turned out to
hold only `GdsMotionSystemReference`, a token reference table (durations/easings/presets) with
zero buttons — while the actual button system lived under a section called "Workflow Guidance,"
giving no hint it existed. A direct violation of `docs/SITE_ARCHITECTURE.md`'s own "section
names mean what a reader expects" rule, reinforced a second time by matching copy on `/systems`.

Fixed as data, not a page patch. `motion-system`'s section is renamed **Motion Tokens**, with
its summary now pointing to Semantic Actions, and `semantic-actions`'s summary now states it
carries the micro-interaction feedback — both entries are reachable by the same search term now.
`systems-page.tsx`'s matching card is corrected to match.

The deeper problem — `/foundations` alone renders 30 entries across 8 sections as one
undifferentiated scroll, and `PatternFamilyPage` is shared by six routes holding 114 registry
entries total — is fixed once, at the shared component, the way `/components` already solved it
in issue 626 Phase 2: a live text filter (title/section/summary/sourceComponent) plus a "Jump to
section" anchor grid, ported onto `PatternFamilyPage` rather than reinvented. `ReferenceSection`
gained an optional `id` prop, forwarded to `SectionPanel`'s existing one, so section headings
became anchor targets — the same mechanism `PatternEntryCard` already used per-entry, extended
one level up rather than duplicated.

**Deliberately not fixed here, filed as issue 632 instead:** `/foundations`'s actual 8 sections
don't match `docs/SITE_ARCHITECTURE.md`'s own "Target shape" table (which defines Foundations as
color/typography/spacing/shape/motion/icons/a11y axes), nor the live page's own `familyMeta`
description ("Shells, controls, cards, and baseline workflow rules") — three different claims
about the same page. That's a real re-categorization decision, not a navigation-polish fix, and
forcing it through here would repeat the exact mistake this repository already declined to make
once this cycle in the #628 typography-outlier round.

### Gating proportions, not just token provenance: a live UX audit closes real gaps and adds two measurement gates (#628)

A live-rendered audit of a single ListingCard example ("Danube Sunset Run") turned into a
site-wide sweep across all 26 reference-site routes, because every gate this repo runs proves
a value traces to a token — none of them ever checked whether a rendered target meets a size
floor, whether an element has any governed styling at all, or how many typographic treatments
one component uses. `CardContracts.ts` computed a resolved `minTouchTarget` per card and
nothing read it.

**Ungoverned anchors — closed completely.** All 17 bare `<a>` tags site-wide, live-traced to
source (not guessed): `SiteFooter`'s "Install GDS" (8 routes), `DemoFooter`'s "Request a
feature" (7 routes), and scattered `primaryAction`/`exitAction`/breadcrumb props, now render
through `GdsInlineLink` — which gained an optional `ariaLabel` prop for the one call site that
needed a dynamic accessible name. `verify-playground-gds-only.mjs` gained a 7th check
detecting any future bare anchor; verified live (planted one, confirmed the gate caught it at
the exact file:line, reverted). `bare<a>` now measures 0 across every route.

**Touch targets — the real ones fixed at the root, a real gate built to hold the line.** Six
genuinely unjustified controls (no recorded rationale, not in a dense context) now meet GDS's
44px floor directly: `DocsCodeBlock`'s copy button, `NumberStepper`'s +/- controls,
`ProductCard`'s overflow menu, `gds-admin`'s `PageHeader` overflow menu, `ListingCard`'s
save/share affordances (the exact card this investigation started from), and `ThemeToggle`'s
default size (was Mantine's unreviewed 28px, rendered on every route via the site header).
`KanbanBoard`'s and `GdsRichTextEditor`'s dense-toolbar controls got a
`data-gds-target-exception` marker instead of resizing — their rows are packed tightly enough
that padding out the hit area risked adjacent controls overlapping, which
`GDS_CONTROL_HEIGHT_EXCEPTIONS`' own "where the control stands alone" caveat anticipates.
`AdvancedDataTable`'s isolated checkbox and `VibeThemePicker`'s 32px swatch grid (the single
highest-leverage fix — one component behind 50 of the original 146 findings) both got a real
44px hit area via a new invisible-padding CSS technique that doesn't grow the visible control.

New `verify-touch-target-floor-runtime.mjs` sweeps all 26 routes and measures every
interactive element against `GDS_MIN_TARGET_PX`, recognizing three real exemptions (WCAG
2.5.8 inline links, documented Mantine xs/sm control heights, the new marker attribute).
Deliberately *not* wired as a hard-fail chain member: it found 791 real remaining violations,
dominated by the site's own `NavLink` component (~250 instances, filed as #629 — fixing it
means visibly growing the whole navigation's row height, a decision this pass shouldn't make
unilaterally). It measures and writes `audit/touch-target-floor.json`; a new
`touchTargetFloorViolations` budget entry is what enforces the ratchet, same pattern as
`themeMatrixUntraceableRate`.

**Typography stratification — investigated, not gated, and said so.** Traced the four "worst
offender" containers a preliminary audit flagged and found three of four aren't sprawl
defects on inspection: a deliberate bespoke brand-mockup preview, a real shell component with
six earned typographic roles, and a documentation section correctly showing four different
chip-group variants together (which a Paper/Card-based container heuristic can't distinguish
from one overloaded card). Mechanically collapsing these to hit a target number would have
been exactly the kind of metric-gaming this repository's own audit work has spent this cycle
fixing. Filed as #630 instead: a typography-stratification gate needs a real single-card
container boundary before it can measure anything honestly, and today's heuristic (any
Paper/Card ancestor) isn't one. No gate shipped on a boundary already known to be wrong.

## 6.1.0 - 2026-08-16 — One semantic token source, obligation coverage, and gate mutation testing

### A new, small shell-height token set closes PublicShell's last two header literals (#625)

`PublicShell`'s three header heights (64/72/88px) were bare numbers; `compact` (64) already
matched `--gds-space-3xl` by coincidence, but `default` (72) and `branded-quiet` (88) never
landed close enough to any existing spacing step to round onto without a visible resize —
that would have been a new design decision disguised as a token swap, not a fix. Recommended
against inventing a value; on approval, added a small, fixed (not per-theme — nothing
currently varies these per brand) `--gds-shell-height-*` token set instead
(`resolveGdsShellHeightTokens`, `gds-theme/axes.ts`), formalizing the shipped 64/72/88 values
exactly as they render today, wired in at `resolveGdsAxisTokens` — the file's own documented
single place a new axis is added. Surfaced a real coupling in the process: the token-graph
validator classifies every token by its role prefix (`inferNodeCategory`), and an
unrecognized prefix silently defaults to `'color'` — which then fails the color-pattern check
against a plain `'72px'` value. The graph-validation test caught it immediately; the prefix
is now registered alongside `radius`/`space`/`control-height`/`font-size`. Verified live:
each token resolves to its exact intended pixel value, and the audit's `padding-top: 88px`
finding is gone. `untraceableRenderRate`: 7.428% → 7.418%.

### #627 closed to 91%: a parenthesis-blind comma split, the last two known contributors fixed, two real GDS literals tokenized (#625, #627)

The single largest remaining fix: a naive `value.split(',')` shredded any multi-value
`transition-timing-function` containing 2+ comma-separated `cubic-bezier()` functions
mid-parenthesis — `'cubic-bezier(0.2, 0, 0, 1), cubic-bezier(0.2, 0, 0, 1)'` (two transitioned
properties sharing one easing, completely ordinary) became eight fragments like
`'cubic-bezier(0.2'`, none of which could ever match a token. Replaced with a
paren-depth-aware split in both the monolithic and chunked render classifiers.

Both classifiers also now resolve Mantine's `border-top-width: calc(0.0625rem *
var(--mantine-scale))` formula directly and index the result, rather than leaving it
permanently untraceable because it combines a literal with a variable instead of referencing
one custom property — recognizing the same resolvable Mantine constant everywhere it appears
(`Input`, active buttons, `ThemeIcon`, ...) instead of enumerating selectors. Required setting
`border-top-style` on the probe first: a UA reports width as 0 with no style set, the same
outline-width lesson applied a second time.

Two real, GDS-owned literals fixed at the source, not the classifier: `.gds-tour-launch`,
`.gds-tour-btn`, and `.gds-tour-card__actions`/`__nav` (`packages/gds-theme/styles.css`) now
reference `--gds-space-*`/`--gds-font-size-*` tokens instead of bare rem values — rounded to
the nearest existing step where no exact match existed, a small deliberate visual change,
verified live rather than assumed. `PublicShell`'s `headerVariant='compact'` (64px) now
references `--gds-space-3xl`, the only one of its three header heights landing exactly on an
existing step with zero visual change; the other two (72px, 88px) don't round without a
visible resize and stay bare numbers pending an owner decision on new token values.

Combined result on the same 40-cell Phase 1 slice #627 flagged: **9.857% → 7.428%**, closing
91% of the regression. `untraceableRenderRate` ratcheted to the honest current number.
Understood, not closed: `.gds-tour-card__progress`'s `letter-spacing: 0.02em` has no GDS
tracking-token axis to reference at all, and several Mantine components (`SegmentedControl`,
`Button`) declare component-scoped custom properties or plain literal transitions invisible
to a document-body probe — the same limitation already diagnosed for Badge's font-size.

### The color-mix() serialization gap fixed; the two remaining known contributors traced precisely (#625)

Chrome serializes a resolved CSS `color-mix()` as `color(srgb r g b / a)` — 0-1 float
components — never as `rgb()`/`rgba()`, even when the token it resolved through declares a
plain hex value. String-equality comparison never matched the two forms of the same color.
GDS uses literal `color-mix(in srgb, ...)` directly across `AccentPanel`, `ChoiceChip`,
`EditorialCard`, `EditorialHero`, and `GdsGeneratedHero` — not just its own JS math helper —
so this was real, and the identical fix had already shipped once before for the same reason
in `verify-badge-contrast-runtime.mjs`. Applied to both classifiers.

The two remaining known contributors from #625's original correction are now precisely
diagnosed rather than vague: Mantine's `border-top-width: 1px` on `Input`/active-button
elements is `calc(0.0625rem * var(--mantine-scale))` in Mantine's own CSS — a real,
scale-aware expression no probe can trace because it combines a literal with a variable
rather than referencing one custom property; GDS has no override for it and this reads as
accepted Mantine baseline geometry, not a gap. `.gds-tour-launch` (`packages/gds-theme/styles.css`)
declares `gap: 0.4rem` and `padding: 0.45rem 0.9rem` as bare literals — genuinely
untokenized, confirmed live — but no existing `--gds-space-*` step lands close enough to
swap in without visibly resizing the control, so picking one is an owner decision, not a
same-day classifier fix. Both recorded on #625 with exact file/line detail.

### Three more oracle gaps closed; the badge cluster traced to a probing-methodology limit, not a design gap (#625, #627)

Investigating #627's stale render-rate finding surfaced that the shared render classifier —
already treated as the complete reference the theme-matrix oracle was brought up to parity
with — had the same gap class itself: `TRACKED` checks font-size and all four border-radius
corners and both gap axes, but `PROBES` had no matching entry for any of them. Added
`fontSize`/`borderTopLeftRadius`/`rowGap` (one length probe per value-space category, mirrored
into `verify-theme-coverage-matrix.mjs`'s own oracle for `border-radius`). Real but small effect
on Phase 1 (28972 → 28959 literal observations) — most of it doesn't touch this gap.

That small effect matters more than its size: it confirms the DOMINANT untraceable cluster
(Badge `font-size`/`letter-spacing`, ~90 theme-hits) is NOT this bug. Traced to source: Mantine
declares `--badge-fz-{xs..xl}` inside the `.mantine-Badge-root` rule itself, not at `:root` —
a real, well-organized token, just one a generic `document.body`-appended probe element can
never see, because the custom property was never set in any of the probe's actual ancestors.
The classifier's probing methodology — resolve every token from a context-free div — is
structurally blind to any Mantine custom property scoped to a component class rather than the
root. Recorded on #627 rather than worked around: fixing it means probing from within a real
rendered instance of each context, which is a different, larger change than a probe-list
addition.

The whole-space and Phase 1 numbers were corrected in the prior entry; the theme-coverage-matrix
gate itself still wasn't measuring cleanly. It tracks ten properties — color, background-color,
border-color, border-radius, padding, font-size, font-weight, box-shadow, transition-duration,
outline-width — but its token oracle only ever resolved a value through four of them: color,
padding-top, border-top-width, font-size. A token expressed only as a font-weight, a box-shadow,
or a transition-duration shared no value space with any of those four and could never appear in
the oracle, so every element rendering one — even correctly, from a real token — was misclassified
untraceable. Brought to parity with the shared render classifier's already-complete probe set.
Separately, Mantine's `ScrollArea` ships its own static scrollbar padding and transition in
`ScrollArea.css`, untouched by any GDS theme override; both classifiers now exempt it by selector
(`mantine-ScrollArea-scrollbar`/`-thumb`) rather than inventing a token for browser-chrome geometry
that was never a design decision. Measured on the same 52-cell sweep, same tree: **26.83% →
15.67%**. `themeMatrixUntraceableRate` ratcheted with the full count and reasoning in
`audit/budgets.json`'s `justifiedBy`. A pre-existing, unrelated test-isolation leak was found and
fixed in the same pass: issue 602's gate-suite mutation-floor test restored the artifact it
deliberately mutates but not the report file `verify-budgets.mjs` writes as a side effect, leaving
a full `npm run preflight` run reporting a dirty tree after a clean test pass. Fixed by backing up
and restoring both. A newly-found, separately-scoped signal filed as #627: the Phase 1
`untraceableRenderRate` budget (7.2%) reads stale against a fresh 40-cell re-run (~9.86%,
observation count and literal count both up since the last correction, literals disproportionately
so) — not touched here, pending its own investigation into what's newly untokenized.

### The four-area navigation lands, and every component is one click away (#626 Phases 3–4)

The primary navigation is now the reading-order story: What Is GDS · Install · **Foundations ·
Components · Patterns · Systems** · API · Themes · Governance. `/foundations` is a top-level
area with its old catalog URL redirecting forever; **`/systems`** umbrellas the deep dives
(theming, badges, imagery, motion, maps, i18n) plus a Resources grid carrying every page that
left the primary nav — one canonical home, not fewer doors. Every registry entry is now
deep-linkable (`#entry-<id>`), and `/components` links each of the 165 registered components
straight to its exact home through the new governed `GdsInlineLink` — which exists because the
GDS-only site had no inline-anchor primitive at all (Rule 15, again). Contract tests updated
deliberately (nav order, redirects incl. the elevated foundations URL); everything verified
live: the redirect lands, the anchors resolve, the 165 deep links render.

### The complete element list exists, and it cannot omit (#626 Phase 2)

`/components`, in the primary navigation: every public component the packages export — all
297 — derived from the same census the release gates read (Rule 14). Registered components
(165) link to their canonical catalog home; reviewed helpers (132) state their exemption
reason in full view, so even the decision to not give something a page is visible. Filterable
by name, section, or reason. The completeness claim is held by a gate:
`verify:component-catalog-parity` now also fails on index drift, and the generator itself
fails on any component that is neither registered nor exempted. Closing the loop surfaced one
stale record — `GdsSchemaForm` was simultaneously registered and exempted; the dead exemption
is removed and the gate and index now agree exactly (165/132). New route wired through the
full obligations: locale coverage declarations, static routes, nav entry.

### The site's structure starts its rebuild: one home per capability, and motion is no longer hidden (#626 Phase 1)

Owner directive: the structure grew by accretion — the badge system filed under "Messaging
Primitives" with its own vocabulary split across three families, generated imagery buried in
"Editorial & Brand Storytelling", the map system under "Public Shells & Docs", and a shipped
motion vocabulary (six durations, five easings, seven presets, the reaction axis) appearing on
**no page at all**. `docs/SITE_ARCHITECTURE.md` is now the IA's SSOT — every placement carries
its why, and the standing rules are: one canonical home per capability, section names mean
what a reader expects, nothing shipped is invisible, structure is site data while capability
is GDS. Phase 1 delivers: Badges & Indicators unified under foundations, Generated Imagery
and the Map System re-homed as coherent sections, feedback meaning feedback again — and
**`GdsMotionSystemReference`** (gds-core, Rule 16), surfacing the entire motion system live
from the exports consumers import, hover-driven preset specimens included, with reduced
motion honoured by the tokens and the visit's actual mode stated. Phases 2–4 (the census-held
complete element list, nav regrouping with legacy redirects, per-component canonical pages)
are tracked on #626.

### The untraceability instrument accused inert values — corrected, every number re-measured (#625)

The covering array's first element-level look found the top "literal" at the worst cell was
`outline-width: 3px` on 1,838 elements — the UA default (`medium`) on elements whose
`outline-style` is `none`, a value that paints nothing. Both classifier copies (the shared
render capture and the theme-coverage-matrix's own sweep) counted it as an untraceable design
value. Corrected in both — the two must classify identically — and every affected number
re-measured in full, never adjusted: whole-space mean 18.6% → **7.8%** (591 cells re-run),
Phase 1 18.4% → **7.2%** (more than half the historic F5 headline was the defect), theme
matrix 33.02% → **26.7%**. Budgets ratcheted to the corrected baselines with the reasoning
recorded. This is the §3.1.1 verification-layer defect class — 18% of this repository's
historic bugs — caught by the instrument's own first honest cross-examination. #625 now
carries the true remaining coordinates.

### Render coverage is a measured number: the covering array runs the whole space (#583)

Phase 3 of the deep audit exists: `audit:render-coverage`. An 8-factor model derived from the
system (25 themes × 24 routes × 12 locales × 3 theme-defined viewports × scheme, reduced-motion,
forced-colors, interaction state — viewport first-class per §3.1.1's 18% defect share), a
deterministic IPOG array (603 rows; same model, byte-identical array), WGA selection that
measures the existing runtime gates FIRST (202 tuples; 12 rows dropped as already covered —
augmentation, not replacement), §3.1.3 transition-cost ordering, and the Phase-1 classifier
reused verbatim. **First full run: 591/591 cells executed, 0 skipped, 460 seconds — 100%
pairwise coverage on every factor group and 100% a11y-critical 3-way triples**, written to
`audit/coverage-array.json` with per-cell results. Mean untraceable across the whole space:
18.6% — Phase 1's hand-picked 18.4% is now a validated estimate rather than a hope, and
"clean" is finally distinguishable from "unvisited".

Building it surfaced and fixed three real tooling blind spots: CDP commands stranded forever
when their execution context died (the shared client now rejects all pending calls on socket
close), `client.close()` hung on an already-closed socket (now resolves immediately), and the
monolithic capture reproducibly crashed the renderer on `/api` — a page Phase 1 never visited —
now executed in chunked transport with identical classifier semantics. Its first measured
finding — brand lanes on `/` and `/themes` at phone width ~9 points above the untraceable
mean — is filed as #625 with exact coordinates. The §3.1.4 mutation-score stopping rule is not
implemented (the full array runs to completion, so no budget decision arises) — recorded in
the plan, not omitted.

### Rule 16: everything visible on the site is a GDS capability — the overlay engine moves in (#624)

Owner directive, recorded as CLAUDE.md Rule 16: no visible behavior on the reference site may
be produced by page-local implementation — the page composes what the packages export and
supplies only data. First application: the DOM phrase-translation engine (visible on every
localized page) moves from the playground into gds-core as `translateGdsDom` +
`useGdsDomPhraseTranslation`, carrying everything the site's defects taught it — per-node
original-text memory, app-update discrimination, verbatim protection with translating nav
labels — parameterised by a phrase-index loader, so the DICTIONARY stays consumer data. The
site keeps its generated packs and a loader; a consumer can now obtain the same behaviour by
installing the package and supplying theirs. The sweep for other page-implemented visible
mechanisms found none: `SiteTourLauncher`, `ThemeBuilder`, and the route fallback are pure
compositions of GDS exports over site data, which is the model.

### The phrase overlay no longer freezes dynamic text, and the map preview adapts to phones

Two defects from an owner phone screenshot, both fixed at the source:

**The site's phrase overlay was clobbering every dynamically-updated text.** Its mutation
observer re-ran the pass on any DOM change, and the pass wrote each node's REMEMBERED
first-seen value back — so any text React updated later (the map's "is loading" → "4 markers"
status, any live count, any swapped aria-label) froze at its first value, in every locale
including English. The overlay now tracks what IT last wrote per node: a current value that is
neither the remembered English nor the overlay's own write is app-authored, becomes the new
remembered English, and is translated from there — never reverted. Three regression tests,
including the aria-label swap case (a save toggle naming its next action).

**The on-map preview clipped to a sliver on phones.** Measured: the card is 597px tall against
a 320px map in a 208px column — a pin-anchored balloon can never fit, and Leaflet clips what
does not. Two fixes: the popup host now declares its real footprint before Leaflet's
measurement (an empty host measured 51px and locked it), and **the preview adapts** — below
480px of container width it docks to the map surface (full-width, height-capped, scrollable);
above, the balloon renders height-capped, opened only after the container re-measure so
auto-pan aims at true geometry. Verified live at both widths: docked-in-bounds-and-scrollable
on mobile, balloon-with-card on desktop.

### The generated-imagery system covers identity: avatars, marks, and the site's own assets (#565)

Two new shapes on the shared engine, closing the gaps that would have forced allowlist growth
the moment an imagery-exclusivity gate lands:

- **`GdsGeneratedAvatar`** — a deterministic identity mark from a person's name: initials on
  the house gradient (legible at 24px, no wrong-gender or wrong-skin-tone guesses, no photo
  pipeline, no third-party avatar service seeing your users), with a seeded gradient ANGLE so
  two people with the same initials still differ — geometry varies, hue never does, because
  hue belongs to the theme. `role="img"` named by the person; the initials are aria-hidden.
- **`GdsGeneratedMark`** — the compact logo-shaped composition for app tiles, brand squares
  and workspace switchers: the gradient with one motif rendered prominently at a bounded
  seeded tilt. Decorative by default; a named image only when it stands alone.

And the reference site now practices the rule on itself: the favicon was a hand-authored SVG
with a hardcoded `#863bff` — the exact exception the rule cannot afford — and is now GENERATED
from the default theme's resolved palette, alongside a 1200×630 og:image card composed from
the same palette and the system's own pin silhouette (rasterised via the wasm renderer;
deliberately geometry-only, since rasterised text needs a bundled font while every crawler
renders og:title as text beside the image anyway). Both regenerate deterministically in
`artifacts:refresh`, so drift is a clean-tree failure. Five tests; proofs on the
generated-imagery entry. Rasterised social output beyond the OG card (full favicon .ico
matrix, per-page cards) remains open scope on #565's epic if ever needed.

### The map's markers are the governed pin, themed live, with the preview on the map (#620, #621, #622, #623)

Owner report from the live site, all four halves verified in source before fixing:

**The map drew plain dots** — the pin vocabulary and its #545 states never reached the one
surface they were designed for. The marker is now an engine-injected SVG built from
`GDS_PIN_SILHOUETTE_PATH` (the same path `GdsBadgeShapePin` renders — exported so the two
cannot drift), with live `var()` colour references, the selected state's tail-tip scale and
emphasis stroke on-map, the approximate state's dashed neutral, density-scaled size, and the
tail tip as the geographic anchor. Selecting a pin now opens the preview ON the map: a Leaflet
popup hosting `GdsMapPinPreviewCard` through a React portal, re-opened deterministically across
the selection-triggered re-init (which had been destroying it — caught live). Verified in a
real browser: governed path, live-var fill, popup with the card inside, 2.25 selected stroke.

**Baked-value components ignored the active theme** — `GdsMap` and `GdsPinSystemReference`
defaulted `preset`/`colorScheme` to `'default'/'light'`, and the reference site passed neither,
so marker colours never followed a theme switch. New `useGdsAmbientTheme` reads and observes
the attributes the runtime already writes to `<html>`; both components default to it. The
in-DOM generated imagery needed nothing — verified it already reads live `var()` refs; only
the data-URI export path bakes hex, correctly, and must name its theme.

**Demos read as a sports product** — the emoji-mode categories, thumbnail badges, and the map's
own data now span four domains (swimming, dance, music, cooking; painting, choir, gardening),
because a proof where every pin is a sport reads as a sports-app component, not a general one.

**Aspect ratios were a claim without a proof** — all four ratios in the thumbnail vocabulary
now render on the generated-imagery entry; the hero proofs already existed on that entry.

### The design-intake contract is codified (#538, #539, #540)

`TEMPLATES/DESIGN_HANDOFF_TEMPLATE.md` codifies the structure of the handoff that set the bar
(ClassScout v2): fidelity stated per category, token tables carrying role AND rationale with
computed contrast inline, exclusivity annotations, accessibility in the same tables as the
spec, the states contract with content guidance, content rules with the degrade-to-zero
requirement, the no-fixed-counts confirmation, the literal-values allowlist, and a closing
"open items, don't guess" list — every micro-example quoted from the real handoff, none
invented. The defining property, stated as the template's own test: an implementer never has
to guess. Referenced from CONTRIBUTING's importing walkthrough and THEME_GOVERNANCE's
importing section, which previously described the shape only in the abstract.

`GDS_THEME_CREATION_PROMPT.md` §7 grows from four questions to the full structured brief —
fidelity line, states contract, content rules, count-driven compositions, provenance — because
the prompt is what a fresh agent actually runs, and putting the questions there is what makes
them get asked. This closes the #538 tracker: its other three children (#541, #542, #543)
landed earlier this cycle.

### The map's geometry is theme-scaled; the themed-basemap issue completes (#569)

The wash (#549) and the area-fill recipe (#550) were the colour halves; this closes the
geometry half. Pin dots now derive from the density axis — 1.5 × the `sm` space step, 18px at
default density and measured live shrinking to 12px under a compact step — instead of a fixed
18px, and the default map/panel heights are rem-based through the same `--mantine-scale`
factor the axes use. No map-specific colour, radius, spacing, or duration constant exists;
the one deliberate literal is the pin dot's 2px halo, which is map paint per the documented
literal-values allowlist. The map participates in the theme coverage matrix by construction —
the matrix sweeps every declared route and the map's proof route is one of them.

### The map degrades honestly when tiles cannot load (#570)

`GdsMap` now detects total tile failure — `GDS_TILE_FAILURE_THRESHOLD` consecutive errors with
zero successful loads; partial flake keeps its mostly-working imagery — and renders a
`StateBlock` banner **beside** the map, never in place of it: markers, the text-equivalent
list and the ODbL credit need no tiles and keep working, and "no tiles" is never allowed to
read as "no places". Classification is honest by construction (`classifyGdsTileFailure`): a
cross-origin `<img>` error exposes no cause except being offline, so the copy says "offline"
only when the browser says so and otherwise states the cause is indeterminate, naming the
candidates (network, CSP, the host) rather than guessing one. Retry is bounded and jittered
(two auto-attempts, then a labelled manual control) so GDS never becomes load generation for a
host already in trouble. The `offline` prop declares tile-less environments as the *intended*
state — the layer is never requested, and the notice speaks in empty-state voice. Six new
catalogue keys in all 12 packs. Verified live with the tile host blocked at the network layer
in a real browser: banner and retry render, the list still carries every place, the credit
still shows.

### The area-fill recipe, and two handoff rules codified as standing requirements (#550, #541, #542)

`getGdsMapAreaFill(accentColor, canvasColor)` ships from `gds-theme` beside the color-math
utilities: the governed neighborhood-fill recipe — accent mixed into the **active theme's**
canvas at an exported weight, painted at an exported opacity, with a canvas-coloured hairline
so boundaries read as paper. Categorical always, never a measurement scale. Painting polygons
stays the consumer's map library; the colour is governed. The adjacency constraint ("adjacent
areas never share a family") is documented as the consumer's — it needs the adjacency graph
only the product holds. The no-clustering/no-DOM-markers rule and the synced-list requirement
are now stated as REQUIRED architecture in `docs/MAP_SYSTEM.md` §6 and cross-referenced from
`MapPanel.renderMap`'s own JSDoc; mechanical enforcement was evaluated and rejected as
fragile-by-construction, which the issue named as an acceptable outcome.

Two ClassScout handoff rules become standing GDS requirements: **no composition may depend on
a fixed count** (#541 — zero/one/full demonstrated and tested, with `GdsMapFilterRail`'s tests
as the model) and **the states contract** (#542 — loading/empty/error/success defined through
the governed `AsyncSurface`/`StateBlock` vocabulary, unavailable explicitly waived when it
does not apply). Both live in `COMPONENTS_AND_PATTERNS.md` §5 and in `CONTRIBUTING.md`'s
"Adding a Component or Pattern" step 7, with the mechanical-check evaluations recorded: the
fixed-count checker would misfire constantly, and a registry state-coverage field would be a
hand-maintained claim of exactly the kind Rule 14 forbids — deferred until derivable.

### The map filter rail and the pin preview card ship (#547, #548)

`GdsMapFilterRail`: composed on `PillBar` (which already owns the scrollable roving-tabindex
radiogroup and contrast-correct selected treatment) plus the rail contract — "All" always
first with consumers speaking `null`, counts rendered into labels, the "All" total only when
every option carries a count (a partial sum reads as a total and is not one), a check glyph on
the selected pill, and ResizeObserver-backed `onHeightChange` so the map insets its viewport by
the rail's real height. The "All" default routes through the message catalogue (all 12 packs).
Shipping its key exposed a real generator bug: the pack generator computed "missing" against
`en` only, so a partially-applied earlier run had left `zh` and `ko` short forever — now
per-locale, and the parity gate is what caught it.

`GdsMapPinPreviewCard`: composed from the generated-imagery system (`badges="none"` — at this
tile size badge pills become clutter, the activity is named in text), `GdsSavedIndicator`,
`MeaningBadge`, and the elevation axis's `sheet` role. Every field has a defined absent
treatment (no categories → no media region; no trust badge → row omitted — the absence of a
claim is not a claim; `loading` → a same-shape skeleton), and control labels are
consumer-supplied and required, per the no-shipped-English rule.

Proven as ONE composition on the gds-map demo: the rail filters the same markers the map
renders, its counts computed from that data, and selecting a real pin opens that pin's card
with save/close wired live. `docs/MAP_SYSTEM.md` §9 now documents both, per its own rule that
a "not built yet" section outliving its issue is a defect in the document.

### The basemap wash ships, and the map system has its SSOT (#549, #572)

`GdsMapBasemapWash` makes consumer-rendered tiles read as part of the themed page: one layer
combining `backdrop-filter: saturate(…)` with a tint of the **active theme's** canvas colour —
because, per the recorded rationale, desaturation alone yields a grey map. No literal colour
exists in the component; a cream theme washes cream, a dark theme washes dark, and the var
stays live so a theme switch re-resolves it (the #598 detector's territory). It ships as a
wrapping composer (the #546 anchor lesson — no positioned wrapper pushed onto consumers), is
`pointer-events: none` and `aria-hidden`, sits below Leaflet's overlay panes so markers render
un-washed, and renders nothing without children — so a `MapPanel` state block structurally
cannot be washed. Four tests; proven live on the gds-map demo over real OSM tiles.

`docs/MAP_SYSTEM.md` now exists as the map programme's SSOT: the GDS-vs-product ownership
boundary, the layer model, the Leaflet-not-MapLibre record, the ODbL/tile-host obligations a
consumer inherits, the accessibility architecture with the no-clustering decision preserved,
and the degradation model — with the not-yet-built parts (#547, #548, #550, #569, #570) stated
as absences that become defects in the document itself once their issues close.

### The stale-theme-value detector exists, and it caught its own planted defect (#598)

`verify:stale-theme-values-runtime` closes what #561 deliberately left open: the theme identity
proves the remount *key* is sound, not that the remount *empties* every themed value. The
definition that makes the detector honest: after switching in place (through the Theme Lab's own
native selects, the path a user takes), every watched element-property must equal what a **fresh
load of the target theme** renders — ground truth with no hand-written expectations, and anything
that legitimately doesn't vary by theme is identical in both snapshots, so it cannot
false-positive. ~16,400 element-properties compared, including SVG paint-server references
(resolved to their content, because React `useId` names differ per load) and image sources.
Switch latency measured (84ms) against a stated 3000ms budget; keyboard focus and scroll position
verified to survive the remount. Proven by a planted module-scope cached `getComputedStyle`
value — the exact memoised-without-theme-deps failure the issue names: the detector reported it
as a single precise `StaleValueReport` and went green on revert. Two of its own defects were
found and fixed en route: a 600-element snapshot cap that silently excluded the plant, and a
regex whose escapes a template literal consumed before the page parsed them.

### GdsMapPinBadge carries the full state contract (#545)

`state?: 'idle' | 'hovered' | 'selected' | 'approximate'`, governed by the source spec's own
principle, now enforced by test: **the fill belongs to the activity — state is carried by
silhouette and scale**, so no state ever repaints the category's hue. Hover widens the stroke
to 2.25 and darkens it one step down the *same accent's* shade ladder (the spec's fixed navy
was rejected as brand-hardcoded; the ladder's steps are already WCAG-verified and need no new
token). Selected scales the marker by 1.15 **around the tail tip**, so the anchored geographic
point holds still (a center-origin scale would drift the point the pin exists to mark). 
Approximate swaps the solid accent stroke for a dashed neutral while the icon keeps the accent.
Saved is deliberately NOT a state — it composes `GdsSavedIndicator mode="corner"` (issue 546),
because a pin can be saved while hovered. `GdsBadgeShapePin` needed no change: Tabler's
component forwards `strokeDasharray`, verified by test rather than assumed. Documented in
`docs/BADGE_SYSTEM.md` and proven live in `GdsPinSystemReference`'s new states row, with the
selected-scale constant surfaced from the export rather than retyped.

### The focus ring no longer waits for JavaScript (#552)

`styles.css` gated its governed focus rules on `html[data-gds-theme-preset]` — an attribute set
client-side, post-mount. No server-rendered or pre-hydration paint can carry it, so a keyboard
user tabbing before JS mounted got no governed focus indicator at all. Verified structurally and
live. Every `--gds-vibe-*` token the rule reads has always had a `:root` default in the same
file, so the gate was an authoring artifact, not a dependency: the focus rules (main and
forced-colors variants) are now **unconditional** — an accessibility floor applies from first
paint; cosmetic rules stay gated. `.mantine-NavLink-root` and `.mantine-Tabs-tab` are named in
the selector list (verified: they render as `<a>`/`<button>` and were element-covered, but the
list is read as the coverage contract and Mantine may change the rendered element). New gate:
`verify:focus-ring-runtime` renders a no-JS fixture linking only the published stylesheet, tabs
through native controls plus both Mantine classes via real CDP key events, and asserts the 2px
solid ring computes — mutation-tested by re-gating one selector group. One shared stylesheet
serves every brand lane, so the fix covers `class-usa`, `gold-athlete`, and `brand` at once.

### class-usa owns its radius scale instead of borrowing Mantine's (#551)

`createBrandTheme('class-usa')` shipped `defaultRadius: 'lg'` against Mantine's *stock* scale,
so Card's 16px was a coincidence (stock `lg` happens to be 1rem), Badge's "pill" was a 32px
accident, and the handoff's 8/12/16/24/pill scale existed nowhere. The theme now owns
`radius: { xs 8, sm 12, md 16, lg 24, xl pill }`, so the named steps *mean* the handoff tiers:
Button moves from a hardcoded `0.75rem` to `sm` (same pixels, now scale-derived), Card/Paper to
`md`, Badge's `xl` becomes genuinely pill. A consumer with no overrides now gets the handoff
geometry. Tier expectations pinned in `brand-tokens.test.ts`; `gold-athlete`/`brand` untouched.

### The literal-values allowlist is documented positively (#543)

`COMPLIANCE_TOOLKIT.md` now names the five places a literal value is *allowed* — theme/token
sources, the GDS packages themselves (governed by their own gates, not consumer rules),
generated SVG output, map paint, and the PWA manifest — each with the reason its output has no
stylesheet or theme in scope, cross-checked against what `gds-compliance` actually enforces.

### The seven Hebrew phrases are translated (#611)

Owner-approved. Register matched to the peer locales' descriptive renderings; like all locale
copy they await the later human review pass. Hebrew leakage drops to zero and the
`englishLeakageWorstLocale` budget ratchets 0.36% → 0.15% (now German's three verbatim
loanword phrases).

### Compliance rules no longer read comments as code (#615)

Two releases went red on rule matches inside comments: `#600` — a GitHub issue reference — read
as a three-digit hex by the raw-colour rule, and "`<select>`'s options" in a comment read as a
native control. `gds-compliance` now lexes comments out (a real string-aware state machine, not
a regex — `https://` inside a string must survive) before any of its six source scanners run;
the markdown documentation scanner is untouched. Regression test added and mutation-checked:
neutering the stripper fails the suite. Decision recorded: `packages/**` stays outside consumer
compliance scanning because the packages are the colour/control *authority* the rules exist to
route consumers toward — scanning the authority with consumer rules is a category error.

### The Mantine compat gate now separates install failures from compatibility results (#604)

`verify-mantine8-compat` re-resolves a full dependency tree from the registry by design ("a
fresh consumer install works"), which let a registry hiccup (`ETARGET` on a version that
existed) read as a red compatibility result on `main`. Installs now retry twice with backoff,
and a failure that survives is reported as `DEPENDENCY INSTALL FAILED … NOT a Mantine
compatibility result — the compatibility check never ran`. Demonstrated with an unresolvable
pin: three attempts, then the classified message. The build step — the actual compatibility
check — is unchanged and still fails on real incompatibility.

### Three mutation-score artifacts, three different instruments — now named so (#603)

`mutation-score.json` measured mutants against the audit's *static analyses* while its name
claimed the whole subject; renamed to `static-analysis-mutation-score.json`, with the
three-instrument map recorded in `scripts/audit/mutate.mjs`. Regenerating it surfaced that
mutant M7's anchor — the phrase "Live Demos" — had been silently invalidated by #606's rename,
and the first re-anchor ("Live proofs", two words) *ran but survived* because the leakage
measure's own `isProse` rule excludes sub-three-word phrases by design. Re-anchored on counted
prose; KILLED again, score back at 85.7%, both lessons written at the mutant.

### The nine forward-trace gaps are classified, and the /live-demos URLs redirect (#612, #606)

Each of #612's nine token gaps now carries a verified reason and a review date in
`scripts/audit/forward-trace.config.mjs` — classified means *explained, not excused*: they still
count in `tokensWithGaps`, the ratchet still only moves by closing one, and an expired or stale
classification is dropped at trace time. And closing #606 surfaced that the `/live-demos` →
`/live-proofs` rename had shipped **without redirects** — exactly what the issue said must not
happen. The whole legacy family now redirects (verified live: `/live-demos`,
`/live-demos/analytics`, `/live-demos/food` all land on their renamed counterparts).

### The viewport-reachability gate ships, and it found real defects on its first run (#619)

The follow-up #619 owed: a gate that fails when a governed surface makes content unreachable at
phone width. `verify:viewport-reachability-runtime` sweeps all 24 declared routes in headless
Chrome at a true 390px (CDP device emulation — headless Chrome silently refuses window widths
under ~500px, so `--window-size` alone audits a page no phone shows). The rule that makes it
honest, per element beyond the viewport edge: inside a working `overflow-x` rail → reachable;
inside an ancestor a transform parks entirely outside the current clip region (AppShell's
collapsed mobile navbar, including one nested in a `BoundedPreviewSurface`) → off-canvas,
reachable by interaction; clipped by `overflow: hidden` or inflating the document scroll width →
**broken**. `aria-hidden` subtrees and `alt=""` images are never content. The discarded first
detector's 34 false positives all fall into the categories above.

**First honest run found two real system defects**, both the same root cause: a single
unbreakable token (an email address in an `h4` pushed a 272px card to 524px on
`/request-feature`; one long compound kept a pattern title's column from shrinking on
`/patterns/public`). Fixed at token level in `gds-theme`'s stylesheet — `overflow-wrap: anywhere`
on governed text and anchors — so min-content widths can shrink and no consumer solves this
per surface. Gate mutation-tested end to end: removing the stylesheet rule and rebuilding makes
the gate fail on exactly those routes; restoring it makes all 24 pass.

### Components themselves are now translatable, not just the site documenting them (#617)

The reference site's phrase overlay rewrites the rendered DOM, so the site looked fully localized
while a consumer who installs `@sovereignsquad/gds-core` and sets a locale still got English empty
states, retry buttons and error titles. **69 default-prop literals across 30 components** now
resolve through `getGdsMessages` — the parameter keeps its English fallback at the call site, so
behaviour for `en` and for hosts passing explicit props is unchanged — and all 12 locale packs
grew from 188 to 258 keys, machine-translated pending the later human review pass.

**Four ids were already silently English in every locale.** `gds.navigation.openMobile` and three
`gds.featureBand.*` ids existed at `t()` call sites but in no pack, so every locale rendered their
English fallback, permanently. The parity gate could never catch this: it compared the packs to
each other, and a key missing from all twelve is invisible to that comparison.
`verify:i18n-message-parity` now also checks **source → pack** — every `t('id', 'English')` in
`gds-core` must have its id in the packs with matching English text. Both new checks are
mutation-tested: the missing-everywhere case and the drifted-English case each fail the gate.

The packs are now **derived, not hand-maintained** (Rule 14):
`scripts/generate-component-message-packs.mjs` reads the call sites and appends what is missing —
byte-for-byte append, never a rewrite, so a corrected translation and the exact text of existing
lines both survive every run. Wired into `artifacts:refresh` before the phrase generator. The
translate helper moved to `scripts/lib/translate.mjs`, shared with the site-phrase generator so
the two cannot drift apart on endpoint or locale list.

### The shape allowlist no longer breaks when unrelated lines move (#614)

`verify:shape-token-adoption`'s allowlist was keyed by `file:line`, chosen so an entry would stop
matching when its line moved and force a re-examination. Observed across its life: it fired seven
times, every one an edit inserting lines above an untouched declaration, every one resolved by
retyping the number — it never once caught a changed declaration. Re-keyed by the file plus the
declaration's own source text: the identical declaration moving stays covered, a **different**
declaration in the same place is refused (verified by substituting `'7px'` for an allowlisted
`'50%'` — the gate reports both the new violation and the now-stale entry), and an entry matching
no declaration at all now fails the gate, which the line key could not check.

### Translated object keys, and prose dropped for being long (#617)

Two defects found by asking why specific sentences were still English on the Korean site.

**The page-copy generator translated object KEYS.** In `ReferenceThemeExplorer.copy.ts` the key
`'dark-public'` had become `'어둠의 대중'` — "darkness of the masses" — so every
`presetSummaries['dark-public']` lookup missed and `ko`/`ja`/`zh` fell back to English for those
entries. **24 keys were corrupted this way.** The page still rendered, because the per-field
fallback added in #587 caught it, and that is exactly why it went unnoticed: the data was wrong
and nothing crashed. A quoted key is an identifier the code looks itself up by, never copy — the
generator no longer descends into an `ObjectProperty`'s key, and the three affected blocks were
removed and regenerated. Keys intact, values translated: `dark-public` now reads
`어두운 공개 테마` / `ダークなパブリックテーマ` / `深色公共主题`.

**Long prose was silently dropped.** The extractor rejected any string over **240 characters**,
so the descriptive paragraphs that explain what a proof demonstrates never entered the corpus —
34 strings excluded on length alone. That limit was conservative rather than required: the
endpoint is a GET, so a ceiling exists, but it was *measured* rather than guessed — 286, 573 and
1,144-character strings all round-trip correctly. Raised to 900, which leaves real headroom while
covering every paragraph the site writes.

English on the Korean site is now **188** distinct strings, from 395 when this started. What
remains is dominated by API value names rendered as data — a contrast matrix listing `plum`,
`outline`, `deepest` names values a consumer types in code, and translating them would make the
page disagree with the API — and by strings composed at render time, which no static corpus can
match.

### Atmosphere has a scale, and the page can no longer style around GDS (#618, #619)

**#618 — the swatch gradient, fixed at token level** as ruled. `--gds-vibe-hero` is composed for
a full-width band; painted into a 40px circle its 135° ramp stops being a wash and becomes a hard
diagonal. That is why 23 presets looked like soft tints and the two gold lanes read as a metallic
split — they simply have the highest-contrast heroes.

New `--gds-vibe-swatch`. Use it for any small surface previewing a theme — a swatch, a legend dot, a chip, a preview tile. It is a radial from the centre, so it reads identically at 40px and 400px
because no axis lets a small box crop it differently. **Derived** from the same `primary`/`accent`
the rest of the vibe is built from and mixed against the scheme's own surface, so all 25 presets
carry it in both schemes with no per-preset authoring and no way to drift. A `flatSurfaces` brand
lane gets an honest flat tint rather than an atmosphere GDS invented for it.

`[data-gds-theme-swatch]` is a **complete** surface — size, radius and border from the governed
scales, not just a background. A consumer needing one attribute rather than an attribute plus
their own dimensions is the difference between a governed swatch and a governed colour.

That exposed a blind spot in `verify:token-reachability`: it knew `var(--gds-*)` and
`getPropertyValue()`, but not a **record lookup**. A component previewing a preset other than the
active one indexes `getGdsVibeThemeCssVariables()`'s result — the ambient `var()` describes only
the active theme. `--gds-vibe-hero` had passed solely because `styles.css` references it too, so
the blindness stayed hidden until a token was consumed *only* through the record. **The mutation
suite then caught my fix twice** for being too broad: matching every file made the token emitters'
own indexing count as consumption, and matching every `--gds-*` in components still let an expired
extension point look referenced. Scoped to `--gds-vibe-*` in components.

**#619 — the badge introduction.** It was a raw `<div>` with `<br />` separators: neither wrapping
nor scrollable, so badges were cut off with no way to reach them. Rebuilt as documentation — every
row is a `GdsInline`, and each section states what it demonstrates.

Established rather than assumed, as the decision required: **GDS already had both answers** — 
`GdsInline`, which wraps by default, and an `overflow-x` rail with a `nowrap` row, which the chip
groups already use. The introduction used neither. Documented in LAYOUT_PRIMITIVES.md, including
that any check for this must ask whether the overflowing box *scrolls*, or it reports every
deliberate rail as a defect. Measured at 390px: of 28 badges, the one outside the viewport is
inside a scrollable rail — reachable, not broken.

**The page can no longer style around GDS.** `verify:playground-gds-only` read **4 files** and
checked **2 things**; the playground has 17 source files, so 13 were ungoverned and the property
held by luck. It now reads every hand-written source file and checks six leak forms — inline
styles, raw Mantine imports, unsanctioned stylesheets, CSS modules, `<style>` elements and
CSS-in-JS — with **0 violations**, and comments excluded so it cannot repeat the #615 false
positive. Verified by planting an inline style in a previously ungoverned file.

### The map never loaded, and five cards showed a broken-image glyph

Reported from a phone with screenshots.

**The map had never worked.** `leaflet.css` was imported nowhere — not in the packages, not in
the playground. Leaflet's tile layout *is* CSS: `.leaflet-tile` is `position: absolute`, placed
by transform. Without it the tiles load, are the right images, and fall into normal document
flow with blank gaps between them. Measured: tiles computing `position: static`, **47% of the
container covered**. It reads as "never loaded", and waiting does not help because nothing is
still loading.

GDS owns the map contract, so GDS ships the stylesheet behind its own specifier —
`@sovereignsquad/gds-core/map.css`, copied from Leaflet at build with a guard that refuses to
write a file missing the tile rules. **100% coverage, tiles absolutely positioned.** `GdsMap`
also calls `invalidateSize()` on init and on resize, which is independently right for containers
that settle late.

**Generated thumbnails everywhere** (owner directive). Five card families — `ListingCard`,
`EditorialCard`, `EditorialHero`, `PublicProductCard`, `PublicFoodCard` — rendered a grey box
with a generic photo glyph when no image was supplied. That glyph is the universal *broken
image* picture: it tells a reader something **failed**, when in fact nothing was ever supplied.
They now paint `GdsGeneratedThumbnail` — deterministic branded art from the card's own identity,
no network, no asset pipeline, coloured from `var(--gds-brand-*)` so it follows the theme in
both schemes.

That surfaced a real duplication: the thumbnail's lead badge repeated the card title, which the
card prints directly beneath it — twice on screen and twice in the accessibility tree, caught by
`getByText` matching two nodes. `GdsGeneratedThumbnail` gains `badges="none"` for the
fallback-image case.

**Pin emoji overflowed the head**, leaving almost none of the dark disc visible, so the
composition the docs describe was not what the page showed. It was a bare `0.5`, set
independently of the icon bound beside it. Now derived from `GDS_PIN_ICON_SCALE` — a Tabler glyph
paints ~0.83 of its viewBox, an emoji nearly its whole em, so the same number renders the emoji
visibly larger.

**The skeleton band read as broken** because a permanently-loading band with nothing beside it
is indistinguishable from a surface that failed. It now sits next to the resolved band it stands
in for, labelled.

### Copy GDS owns is now localised too, and the language files are documented (#617)

With the selector fixed (#616), what remained was copy that **never entered the corpus at all**.
Measured on the Korean site across five routes: **395 distinct English strings**, from three
separate causes rather than the one the issue assumed.

- **The extractor read only the playground's own pages.** Copy GDS itself owns — a component's
  default prop, a theme preset's label and description, the Theme Lab's mock — was never offered
  for translation. Those files are now sources.
- **Single-token strings were dropped as identifiers**, so `Choir` and `Saved` rendered in
  English while the two-word `Verified host` beside them translated. Ordinary capitalised words
  are now included; identifier *shapes* (`GdsBadge`, `partner-discovery`), bare lowercase keys
  and acronyms stay out.
- **The overlay skipped `a`, `button` and `label`.** The principle was sound — interactive
  labels belong to the copy layer — but it left **25 link texts and 8 button texts in English**,
  because the copy layer did not in fact cover them. The verbatim list is now narrow and
  concrete: code and form-control values.

**395 → 201.** The remainder is dominated by things that *should* stay English: API value names
rendered as data (`plum`, `outline`, `deepest` — a contrast matrix naming values a consumer types
in code) and strings computed at render time, which no static corpus can match.

**Two traps caught by measuring rather than assuming.** Adding the theme files pulled in their
**CSS values** — `rgba(...)`, `0 6px 16px …`, `1 1 320px` — as if they were copy: 800-odd junk
entries, a wasted translation request each, and a leakage metric that counted them as
untranslated English (`ko` jumped to 112). And `rel="noreferrer noopener"` came in as a phrase.
Both are now filtered by shape, which brought leakage back to **7 in `he` and 3 in `de`** — the
known tracked cases.

**README now documents where every language file lives** — package messages
(`packages/gds-core/src/locales/`), generated site phrase packs, and structured page copy —
which are hand-written, which are regenerated, and how to correct wording. It states plainly
that the wording is machine-generated and unreviewed, with the two measured limits: single words
are unreliable without context, and API value names stay English on purpose.

### The language selector only ever worked once (#616)

Reported from a phone: the selector read **Français** over a **fully Korean** page. Switching to
Hebrew or Hungarian changed the selector and one button, and left the rest Korean.

`translateSiteDom` rewrote each text node **in place**, destroying the English the phrase index
is keyed by. So the pass worked exactly once. On the second switch it looked up the *previous
language's* text in the new locale's English-keyed map, matched nothing, and left everything
alone. The only strings that changed were the handful React re-renders from `page-copy` — which
is why a single button read "Enregistrer" on an otherwise Korean page.

Switching **back to English** was broken in the same way and for the same reason: the pass
returned early for `en`, so it never put back what an earlier locale had overwritten.

Every node's English source is now remembered in a `WeakMap`, and each pass translates from that
rather than from whatever the last pass left behind. English restores instead of returning early.

Verified in Chrome by switching **ko → fr → ko → ru → en → he**, counting script characters at
each step: 3,864 hangul → 3 → 3,864 → 11,939 cyrillic → English restored → 7,099 Hebrew. Every
switch fully replaces the previous language, including switching back to a locale already used.

No reload is needed — the fix removes the reason one would have been.

### `coverageStatus` is derived, not written (#608)

All 113 registry entries carried the same hand-typed `'live-proof'`. **A field with one observed
value cannot be observed to be wrong** — which is how seven entries claimed a live proof while
their cards rendered "No interactive demo renders here" (#600).

`verify:pattern-live-proof` closed the false-positive direction: nothing could *claim* a proof
without one. It did not make the claim derived, and a correct claim is still an unverified one.

Whether a pattern has a live proof is something the system can be asked — does `renderEntryDemo()`
have a case for this id, or does `PatternEntryCard` special-case it. That is now computed into
`generated-pattern-coverage.ts` and read by the registry, so **removing a demo demotes its entry
automatically** and `/coverage` tells the truth without anyone remembering to edit it (Rule 14).
A generated file rather than a runtime computation because `pattern-pages.tsx` imports the
registry, so the registry cannot import the pages back — the same reason
`generated-component-census.ts` exists, with the same `--check` drift guard.

**`verify:pattern-live-proof` was deleted, not kept alongside.** With the status derived,
`live-proof` *means* "has a case", so the gate could never fail — and a gate that cannot fail is
the exact thing this codebase treats as a defect. `verify:pattern-coverage` replaces it: the
remaining risk is a stale generated artifact, which is what the drift check catches. Its mutant
moved across unchanged — removing a demo without regenerating still fails.

**`pending-primitive` and `blocked` were removed from the union** rather than left declared. They
had never been used once across 113 entries, and an enum member nothing can produce is not a
state the system has. Keeping them left `/coverage` rendering "0 patterns are blocked" forever,
which reads as a measurement and was not one.

Two consumers broke and both failed loudly rather than silently: the export-coverage gate parsed
`coverageStatus` out of the registry source and reported all 113 as `unknown`, and the
accessibility-evidence script could not resolve the new import under Node's type-stripping. The
first now derives the status the same way; the second needed an explicit `.ts` specifier, which
the playground tsconfig already permits.

### `GdsViewportFrame`: the capability that was missing, built (#609)

#600 could not live-prove `bottom-tab-navigation` and **said so** rather than staging it —
`BottomTabBar` is `position: fixed` and `hiddenFrom="sm"`, so on a documentation page it either
rendered nothing or pinned itself over the site's own navigation. Rule 15: *the missing
capability IS the finding.* The finding is now a shipped primitive, and the entry is a real live
proof again.

The frame solves two different problems with two different mechanisms, because no single trick
covers both:

- **A fixed child escaping to the window** — `contain: layout paint` makes the frame a
  *containing block* for `position: fixed` descendants. `overflow: hidden` alone does not.
- **A breakpoint gate reading the viewport** — a media query cannot be made to resolve against
  an element, so the frame publishes its width class through context and the gated component
  reads it. `useGdsViewportFrame()` returns `null` outside a frame, so **wrapping something in a
  frame is opt-in and changes nothing for existing consumers.**

Container queries were the alternative for the second problem. They solve it cleanly but would
require rewriting every gated component against `@container`, changing behaviour for consumers
who are not inside a frame — a much larger blast radius for the same result.

**Both mechanisms are CSS**, and that was forced rather than chosen. The first design published
the frame width through React context. `check-export-contract` refused it: reading a context
would make every gated component a client component, and `BottomTabBar` is exported from the
**server** entrypoint. A capability for embedding surfaces should not push its subjects out of
the server lane, so the design changed rather than the boundary.

**A regression this nearly shipped, caught by measuring the real component.** The first
verification probed a synthetic `<div>` and reported the gate working. `BottomTabBar` set
`display: 'flex'` as an *inline* style, which beats any stylesheet rule — so the media query
hiding it above `sm` could never win, and **the bar would have appeared on every desktop page
for every existing consumer.** `display` now belongs to the stylesheet; the inline style keeps
only appearance.

Verified in real Chrome, not jsdom, across the full matrix:

| | outside a frame | inside a compact frame |
| --- | --- | --- |
| **1280×900** | `display: none` — unchanged | `display: flex`, pinned to the frame |
| **390×800** | `display: flex` — unchanged | `display: flex`, pinned to the frame |

The bar measures 358px inside a 360px frame with `escapedToWindowBottom: false`.

This generalises beyond documentation: embedded previews, kiosk panes and split views all need a
bounded viewport, and every consumer was previously left to solve it locally.

### The theme coverage matrix is reproducible (#599)

#562 §5 required "the same commit produces the same result set". It did not: three runs
measured 33.94 / 33.95 / 33.96 percent with element counts of 28,242 / 28,128 / 28,238. The rate
band was small, but the **~114-element spread** meant the sweep sampled a different amount of
each route depending on render timing, so a genuine 0.05pp regression was indistinguishable
from noise — and the budget sat at 34 to clear a band nobody could tighten.

The cause was a fixed `wait(350)`: a guess about how long React, lazy routes and web fonts take,
and a different guess on a loaded runner. It now **waits for a settled DOM** — node count
unchanged across two consecutive polls — which replaces the guess with the condition it was
standing in for. A cell that never settles is reported rather than silently sampled mid-render.

**Three consecutive runs on one commit now produce identical results: 28,203 properties checked,
9,313 untraceable, 0 unsettled cells.** Budget tightened **34 → 33.02**, the measured value, with
the headroom removed.

The artifact records exact counts again. It had been deliberately bucketing to the nearest
thousand and rounding to whole percent — the honest response to variance at the time, and no
longer needed now that the precision is real.

**Stated rather than implied:** 36 cells hit the 400-element sample cap, so those routes are
sampled rather than swept exhaustively. `cellsTruncatedAtSampleCap` reports it on every run,
because a coverage number that hides truncation reads as more coverage than it has.

### The site still called its proof surfaces "Live Demos" — in English (#613, #610)

#606 renamed the route to `/live-proofs` and **never renamed the label**. `site-routes.ts`
shipped `label: 'Live Demos'`, `showcase-pages.tsx` titled the page "Live Demos", and all
eleven locales faithfully translated that — `ライブデモ`, `라이브 데모`, `现场演示`. The primary
navigation, the most-read copy on the site, contradicted Rule 15 in every language for months.

The filed issue blamed machine translation. **It was wrong**: the translations were correct
renderings of an English source nobody had fixed. Investigating which half was wrong before
editing either (Rule 14) is what turned a terminology-glossary project into a two-line rename.

**Two reasons `verify:site-claims` missed it**, and both are now closed:

- `site-routes.ts` and `site-copy.ts` **were not scanned at all** — the files holding every
  navigation label.
- `'Live Demos'` is **ten characters**, and the gate skipped anything under twelve as "an
  internal identifier". The capture regex required twelve too, so the string was never even
  collected. The exemption was inverted for exactly the case that matters: **a nav label is
  short because it is visible.** Length never separated identifiers from copy — shape does, and
  slugs like `demo-surfaces` were already excluded by shape.

Widening the sweep immediately found a second instance the same day, in `showcase-pages.tsx`.

**#610** adds the other half: a substring rename is now caught by its *signature*. `RETIRED_VOCABULARY`
declares `demo → proof` with the legitimate derived forms, and any word beginning with `proof`
that is not one of them is flagged — which is precisely what `proofnstrations` was. Both defects
are now mutants, so the gate is verified against the exact artifacts that shipped.

### A font lane must render every language GDS ships (owner directive, 2026-08-13)

**Only a font stack that supports 100% of the supported languages may be a font lane.** Eleven
of twelve lanes declared coverage of latin, or latin plus cyrillic. That meant **choosing a
font silently chose which languages the product could display**, and nothing said so at the
point of choice. `ja`, `ko` and `zh` had no lane at all: the packages shipped locale packs for
all three while no lane's fonts contained one of their glyphs.

A partial lane is a **detour** in the Rule 10 sense — it does not remove the problem, it
relocates it onto every consuming app, which then rediscovers it one product at a time.

Every lane now ends in a universal script fallback naming one Noto family per script in the
catalog, so the lane's own display face still leads for Latin text and the browser only reaches
a Noto entry for glyphs that face lacks. **Nothing is copied**: the script list comes from
`getGdsLocaleScripts()`, newly exported from the same `gdsLocaleMetadata` that defines the
locales, so adding a locale in a new script makes the font map incomplete and fails the build
rather than shipping a lane that cannot draw it.

New gate `verify:font-lane-coverage`, in the release chain, with a mutant that strips the
fallback. Documented in THEME_GOVERNANCE.md — including **what the gate cannot prove**: it
verifies structure, not font binaries, so a family mapped to a script whose glyphs it lacks
would still pass. Proving real glyph coverage means reading `unicode-range` from the font
service live, the same network dependency already exempted for `audit:dependencies`.

### ReferenceThemeExplorer took the home route down in any locale it had no copy for (#587)

Adding the three locales exposed a latent crash in `gds-core`, not in the app. `resolveExplorerCopy`
layered the locale over `{}` and returned early on `Object.keys(mergedCopy).length > 0` — but the
object literal **unconditionally** sets `schemes`, `schemeDescriptions`, `presetLabels` and
`presetSummaries`, and spreading `undefined` yields `{}` rather than absence. That condition was
therefore true for **every non-English locale**, the completeness check beneath it was unreachable,
and `as ExplorerCopy` asserted a shape the object did not have.

Nothing surfaced while every supported locale happened to have full explorer copy. `ja`/`ko`/`zh`
produced an object with four empty maps and no `tokenLabels`, and the first `copy.tokenLabels[0]`
**blanked the reference site's home route** — white page, in three languages.

Copy now falls back **per field** over English, so a locale that translates most of the explorer
keeps its translations and shows English only where it has none, instead of reverting the whole
component over one missing string. The 27 lines of unreachable completeness checking are deleted
rather than left looking like a safeguard. `ja`, `ko` and `zh` explorer copy is generated, so the
fallback is a safety net rather than the shipped state.

Verified in a real browser against the built bundle, not jsdom — **the jsdom suite passed the whole
time.** It renders the route without the code-split explorer chunk that actually crashed.

### The site can now render in Japanese, Korean and Chinese (#587)

`gds-core` shipped locale packs for `ja`, `ko` and `zh` while the reference site had a phrase
pack for none of them. A prospective adopter in those markets evaluated an English-only site
while the packages they would install supported their language — **the capability existed and
was invisible.**

Locale coverage was verified *within* the package corpus and *within* the site corpus, never
*between* them, so the gap passed every gate. `verify:locale-coverage` now asserts parity
across the two, and `localesWithoutSitePack` is ratcheted 3 → 0.

Both halves are generated, not hand-authored. The 1,302 site phrases came from the existing
generator. The per-locale blocks in `page-copy.ts` and `site-copy.ts` needed a new one
(`generate-page-copy-locales.mjs`) because the runtime phrase overlay deliberately skips text
inside `a`, `button` and `label` — which is most of what those maps hold — so a locale without
its own blocks would have rendered Japanese body copy under English navigation.

That generator finds locale maps **structurally** (an object whose keys are all locale ids and
which has an `en`). Two earlier attempts are worth recording because both failed quietly: one
tested for `ObjectExpression` when every map is written `} as const` and found **one map out of
nine**; the next only looked at exported declarations and missed the maps nested inside
`headerContextCopy`. Both now fail loudly instead — the guard compares what was found against
the blocks actually present in the file.

### Untranslated English: fixed at the source, then gated (#517, #588, #611)

Four i18n gates were green while full English paragraphs sat mid-page in Hebrew and Arabic.
They all check **key parity** — that a pack carries the right keys — and none of them ever
looked at a **value**. A pack could carry every key and translate none of them and the build
stayed green.

**The root cause was in the generator, not the packs.** It kept any stored value that was
non-empty, and a phrase left in English is non-empty — so a missed translation was *never
retried* and survived every regeneration for as long as the file existed. Leakage could only
accumulate. It now retries any value its own leak measurement flags, so a failed translation
request repairs itself on the next run instead of freezing into the artifact. That alone fixed
every English sentence #517 reported, including the `.npmrc` install paragraph in `he` and the
badge-shape paragraph in `ar`.

`zh`'s `gds.action.trendingUp` sat in English beside a correctly translated
`gds.action.trendingDown`; it is now `趋势上升`, taken from the same endpoint the generator uses.

**Detecting leakage without a hand-maintained allowlist.** A value identical to English is not
automatically wrong — `"GdsAccessGate / resolveGdsAccessState"` must stay identical everywhere.
Two derived signals separate the cases, and both were chosen from measured data rather than
imagined:

- **Peer evidence** — if another locale rendered this phrase differently, the phrase is
  translatable, so a pack that left it in English has missed it. This is the reasoning #517
  used by hand ("all 7 other non-English locales translated these correctly"), computed.
  Compared on **letters only**: Arabic renders a component list with Arabic commas, which
  differs byte-wise while translating nothing.
- **Script**, read from `gdsLocaleMetadata` rather than retyped — a Latin-script language
  legitimately shares words with English, so `Pause`, `Filter` and `Message` are real German
  and French words, not defects. A non-Latin pack shares none.

That matters concretely: the naive "identical to English" count flagged **15** package values,
of which exactly **one** was a real defect. The measurement change is stated in
`audit/budgets.json`, and `englishLeakageWorstLocale` moved 3.8 → 0.54 — **not an improvement
of that size**, because the two numbers answer different questions.

New gate `verify:i18n-leakage`, in the release chain, with a mutant that restores the `zh`
defect. What remains is recorded with reasons rather than hidden: **7 Hebrew phrases the
translate endpoint cannot resolve** (verified — all seven return byte-identical English;
#611 carries them, since writing them by hand would be inventing translations under Rule 11)
and 2 German loanword phrases where the machine output differs only in capitalisation.

### "What changed in 6.0.0" described the 3.14 line, in all 9 locales (#518)

`changedTitle` was find-and-replaced on every version bump while `changedDescription` was not
touched since 2026-07-23, so every visitor read a heading claiming the current version above a
list of months-old features.

Rewriting the summary would have bought one release. The description now states that the
changelog is the authoritative record and that this page deliberately does not restate it —
**a claim that cannot go stale, because it makes no per-version assertion.** Rule 14: prose
that cannot be derived should not pretend to be current.

### A pattern may not claim a live proof it does not render (#600, #607, #608, #609)

The pattern catalog told the public, on `/coverage`, that **113 patterns are rendered in
interactive routes**. Seven were not. Those seven reached the `default:` branch of
`renderEntryDemo()`, which prints "No interactive demo renders here" — so the page contradicted
its own registry, per entry, in public.

The reason it survived is the part worth keeping. `pattern-registry.test.tsx` asserted:

```ts
expect(patternRegistry.every((entry) => entry.coverageStatus === 'live-proof')).toBe(true);
```

That is not a check. It is the claim restated as a test, and it *required* the very uniformity
that made the field meaningless — all 113 entries carried the same value, so no entry could be
observed to be wrong. A test that mandates a claim will pass for exactly as long as the claim is
written down, which is the failure shape the gate-mutation work exists to find.

**Six of the seven are now proven for real** — the accent contrast matrix (whose figures come
from `evaluateGdsAccentContrast()`, the same function `verify:accent-contrast` runs in CI),
the searchable select, the conversation surface, media-with-fallback, the number stepper, and
the AI search card.

**The seventh was not staged.** `BottomTabBar` renders `position: fixed` and `hiddenFrom="sm"`,
so on a documentation page it shows nothing at desktop widths and pins itself over the page at
mobile widths, reading as the site's own navigation. Proving it honestly needs a bounded
viewport frame GDS does not have. Per Rule 15 the absence is stated rather than worked around:
`bottom-tab-navigation` is now `static-reference`, says on the page why, and the missing
capability is #609. Its three `BOTTOM_TAB_*` export-coverage entries moved to `support-api` to
match. This is also the first entry to use a second `coverageStatus` value at all.

New gate `verify:pattern-live-proof`: a `live-proof` entry must be rendered by a `case` in
`renderEntryDemo()` or an explicit `PatternEntryCard` branch. It refuses to pass vacuously if it
parses zero entries or zero cases. Its mutant promotes the one honest `static-reference` back to
`live-proof`; the gate must refuse it.

**The first version of that gate was too loose, and the deployed site is what caught it.** It
also accepted "some playground page references one of the entry's `sourceComponent`
identifiers", on the reasoning that such an entry must be proven inside a larger surface. That
is a different question, and three more entries were passing on it:

- `GdsPinSystemReference` and `GdsMap` were rendered inside `BadgeMapDemo`, which serves the
  `badges` entry — so the components appeared on `/patterns/public` while the **`pin-system`
  and `gds-map` cards showed the fallback**;
- `maturity-capabilities` referenced its functions from `info-pages.tsx`, a different route
  entirely.

Checking the deployed page found three "No interactive demo renders here" panels where the gate
reported clean. Proximity is not proof — the rule now asks where a proof is *attributed*, not
whether an identifier occurs somewhere. The two map demos were extracted so each entry renders
its own, `maturity-capabilities` gained a demo derived from `getGdsMaturitySummary()`, and the
loose disjunct is gone. **All 112 live-proof entries now render on their own card.**

Filed #608 for what this does **not** fix: `coverageStatus` is still written rather than derived
(Rule 14). The gate closes the false-positive direction only — a correct claim is still an
unverified one, and three of the four enum values remain unreachable. Deliberately not bundled,
per the #590 lesson.

Also fixed (#607): the #606 rename was applied as a substring replacement and turned
"demonstrations" into **"proofnstrations"**, live in a heading on `/patterns`, in
`apps/playground/README.md`, and translated into all 8 site locale packs.

### The reference site is documentation with proofs, not demos (CLAUDE.md Rule 15)

Owner correction, and the language was the smaller half of it. Describing the reference site as
"my demo" is wrong twice: nothing on it belongs to whoever built it, and calling it a demo
invites treating it as a sandbox where a shortcut is acceptable.

That is not a style point — it drives behaviour, and it did in this session. Proving the
saved-indicator's corner form needed a positioned wrapper, the reflex was an inline `style` on
the page. `verify-playground-gds-only` refused it. **The wrapper was not missing from the page;
it was missing from the primitive**, which had been pushing layout onto every consumer. The
gate caught a design flaw, not a lint violation.

Rule 15 states the approach for when a proof needs a capability GDS does not have: build it in
the system and document through it; if it cannot be built now, state the absence plainly and
carry it in an issue; never stage it. **If documenting something honestly requires a
workaround, the system is incomplete** — the workaround is only where the incompleteness
surfaced.

Filed #606: the site currently calls its own proof surfaces "demos" on reader-visible
surfaces — a nav card and badge in 9 locales, and the public `/live-demos` route — plus 114
internal `coverageStatus: 'live-demo'` uses. The term, the URL (outward-facing, needs
redirects) and the enum are decisions for the owner rather than a unilateral rename.

### GdsSavedIndicator: one save toggle, not two (#546)

The map spec puts a saved heart in two places — a pin's upper-right corner and the preview
card's action row. Building it twice is how the two drift apart, so it ships once, in two
geometries.

- A `button` with `aria-pressed`, **never a decorative heart**. The accessible name states the
  ACTION available while `aria-pressed` carries the state, so neither is announced twice and
  they cannot contradict each other.
- `saveLabel`/`unsaveLabel` are **consumer-supplied and required**, the rule `GdsMapPinBadge`'s
  `label` already follows: GDS ships no English default, because a default is the string that
  survives into a localized product. They name the item as well as the action — a page of pins
  otherwise announces "Save" a dozen times with nothing to tell them apart.
- Size comes from `--gds-control-height-*`, not the spec's literal 48px. `md` is the 44px step
  and it moves with the density axis; `corner` takes the `sm` step — a smaller STEP, not a
  magic number, and still a real tap target rather than an icon-sized hit area.
- The corner form takes its `anchor` as a prop and composes through the governed badge stack,
  following `GdsCountBadge`'s existing convention. The first version made the caller supply a
  positioned wrapper — and the playground's GDS-only gate refused the inline style that
  required, which was the correct answer: **a primitive that needs styling wrapped around it to
  work is an unfinished primitive.**

Shipped with the full checklist rather than a component alone: export-coverage entry,
pattern-registry `sourceComponent`, `COMPONENTS_AND_PATTERNS.md` contract, 6 tests, and a real
live demo — because `coverageStatus: 'live-demo'` is a claim, which is the defect #600 records.

### No hardcoded values, gated (#605)

Owner correction: I described "250+" as hardcoded and then **removed** it. The standing
requirement is no hardcoded values — "the page cannot compute it in this slot" is a constraint
to solve, not a licence to delete information from the site.

**The count is now derived.** `collectPublicComponents()` moved into
`scripts/lib/component-census.mjs`, imported by **both** `verify-component-catalog-parity` and
a new generator. Copying it would have made one fact with two implementations, free to drift.
The page quotes **289 because that is what the gate enforces**; a mutant publishing a stale
count (289 → 250) fails the build.

The translation blocker was shallow once examined: the phrase extractor sees only string
literals and rejects braces, so `{count}` was out — but `%count%` survives extraction, is
translated with the placeholder intact (German keeps `%count%`), and the number lands
afterwards. All eight locales keep the sentence.

**Then the same sweep across every visible surface.** Two more instances, both of them one
fact stored twice:

- The analytics chart's `summary` — the **text equivalent a screen-reader user is given
  instead of the picture** — stated 62/38 in prose while the table fallback stated 62/38 in
  its own literals. A text equivalent that disagrees with its data is worse than none, because
  it is trusted. Both now build from one `CHANNEL_SHARES`, which also asserts the shares total
  100.
- `value="18 / 20"` sat beside `progress={90}` — the bar a sighted reader believes and the
  text a screen reader announces, free to disagree. The percentage is computed from the
  fraction.

`verify:site-claims` now fails on **any number typed into visible prose**. A sentence that
interpolates through a `%placeholder%` is derived by construction and needs no registration; a
written one must be registered with what makes it true, or must not exist. Currently **zero**
registered exceptions — and the negative control was run both ways, because a gate reporting
zero is what a broken gate reports.

### Approved or non-existent: every claim on the reference site now carries its evidence (#605)

Owner directive, 2026-08-13: *"We can have either Approved or Non-existing cases, especially on
visible surfaces."* A guarantee on the reference site is a promise GDS makes to a client, and
there is no third state where a promise is merely plausible.

New gate `npm run verify:site-claims`, chained into `verify:release`. It scans the six
visible-surface sources for absolutes — "every preset", "always", "never", "identical",
"guaranteed", "100%" — and fails on any that is not registered with the evidence that supports
it. Four evidence kinds: **derived** (the page computes it), **gate**, **test**, or
**contract** (a convention with no mechanical check, which must carry a `reviewBy` so it cannot
become permanent by neglect). It also fails on a registered claim the site no longer states, so
the registry describes the site as it is.

**14 absolute claims found: 2 derived, 7 gated, 2 tested, 3 conventions.**

**Three of the claims had no evidence, and I had assumed they did.** Checking each reference
rather than trusting it is what caught them — the same defect one level up:

- *"One GdsProvider at the app root — never nest a second one."* Nothing checked that the site
  obeys its own rule. Added to `verify-playground-shell-contract`, negative control run both
  ways. A nested provider re-declares the theme, so identity, scheme and the governed variant
  lane stop agreeing between subtrees — the class of bug #597 traced.
- *"the pin fills with a fixed dark-neutral disc in emoji mode (never the accent)."* The
  existing emoji test asserted the glyph renders and shape is ignored — not that the fill is
  neutral. Now asserted, including that the ring keeps the accent.
- *"a failed image never collapses a card or shifts layout."* The tests covered which content
  renders, not that the box keeps its height when that content is the fallback. Now asserted
  across the image, missing-src and errored states.

One claim was **removed rather than evidenced**: the home tour opened with "250+ governed,
accessible React components". True today (289) but written rather than computed, and that slot
cannot derive one — the tour body must stay a string literal or the phrase extractor cannot
translate it for the eight locales. Under Rule 14 an unverifiable claim does not get a softer
version; it stops existing. The derived counts live on /api and /coverage.

### Documentation is now derived, not described (CLAUDE.md Rule 14)

Owner directive after finding the badge panel wrong on the live site: **documentation must be
a single source of truth with the system, and the reference site is the product.** GDS cannot
ask a client to trust a design system whose own site misstates its behaviour.

Correcting the sentence was not a fix — a sentence drifts from the tokens the moment either
changes, and nothing notices, which is exactly how the wrong one shipped. So the panel now
**counts distinct token values at render time** and picks its wording from that measurement.
If `warning` is ever pinned, or `info` stops following the preset's text colour, the page says
so on the next render with nobody editing it.

The phrases stay string literals so the site's phrase extractor still translates them; only
the *choice* between them is computed. A test recomputes the counts independently and asserts
the page agrees, so page and tokens cannot disagree without failing the build.

**CLAUDE.md Rule 14** codifies it: a checkable claim must be computed rather than written;
prose that cannot be derived must be gated or must say plainly that it is convention rather
than guarantee; a defect in what the page *claims* carries the same severity as a defect in
what the system *does*; and when docs and behaviour disagree, establish which half is wrong
before editing either — rewriting docs to match behaviour blesses a drift, pinning behaviour
to match stale docs breaks working code.

Issue #605 tracks the sweep of the remaining 9 checkable absolutes found across the reference
site, including a gate so the sweep cannot decay.

### The accent contract, and a panel that taught a rule its own examples broke

Raised from the live site: category badges, pins and generated thumbnails look identical in
every theme. **Measured: all 50 preset x scheme combinations resolve byte-identical accent
palettes.** That half is deliberate and now stated as a contract — accents are a fixed category
vocabulary, so a category means the same thing in every theme.

**A preset may override them, and that override is now verified.** `verify-accent-contrast`
passed `undefined` for the axis: it checked the DEFAULT palette 25 times over. No preset
overrides today, so the numbers were right and the check was not — the first preset to declare
its own accents would have had its real colours go unverified while the gate reported a clean
sweep of colours nothing renders. It now resolves each preset's own axis and reports how many
declare one. Demonstrated: a plausible brand override (`#f5e663` for `forest`) raises **6
enforced violations at 1.28:1**, where the shared vocabulary raises none.

**The same panel was teaching something false.** It labelled three badges FIXED. Measured
across the presets, only danger is:

| Label shown | Reality |
| --- | --- |
| `WARNING — FIXED` | 23 distinct values in light |
| `INFO — FIXED` | 8 distinct values in each scheme |
| `DANGER — FIXED` | genuinely fixed |
| `SUCCESS — SHIFTS` | genuinely shifts |

Investigated before rewriting either half, rather than blessing a drift by editing the docs
around it. The implementation is deliberate and reasoned in source: `danger` and
`warning-dark` are fixed "alarm anchors" — an alarm colour that moves is not an alarm —
verified byte-identical across the two hand-authored presets. `warning` in light is the anchor
mixed with the preset's own hue and pushed to 3:1. `info` in light **is the preset's own text
colour**, which is why it moves the most. So the copy was the wrong half, and it now states
the actual rule.

### The gate suite could shrink to nothing with every budget green (#601, #602)

Split out of #590, which had stalled repeatedly. Not because it was large — because it bundled
a 15-minute rename and a 15-minute ratchet with an open-ended design question and a
**self-referential** mutant (the suite would have had to mutate its own mutant list mid-run).
The two cheap fixes that close the real risk were hostage to the two hard ones. Split into
#601/#602/#603; the first two are here.

**#601 — a budget key named something it did not measure.** `gateMutationScore` read
`mutation-score.json#/mutationScore`: the Phase 5 *render* mutants. It read like the gate
suite's score and pointed elsewhere — F24's trap, one word with three referents. Renamed to
`phase5MutationScore`.

**#602 — the gate suite had no floor on its own coverage.** Only `gateSuiteUnexplainedSurvivors`
was ratcheted, and that is not a floor: **deleting a mutant lowers coverage without raising
survivors**, because a mutant that no longer exists survives nothing. The suite that verifies
every other gate was the one gate whose coverage could silently shrink.

New `gateSuiteMutationScore` budget, seeded at the measured 100% (29 of 29). Demonstrated both
ways rather than asserted: with three mutants simulated as deleted,
`gateSuiteUnexplainedSurvivors` still reports **OK** while the new budget blocks at 89.7%.
That contrast is the issue, reproduced.

The proof is a test that runs `verify-budgets` for real against a temporarily-lowered artifact
— not a re-implementation of its arithmetic, which would pass in exactly the cases where the
arithmetic is wrong. #590 asked for a suite mutant instead; that recurses, and a direct
negative control proves the same property without it.

### The one animation GDS shipped animated nothing (#592)

`ChatSurface`'s typing indicator declared `animation: gds-chat-typing 1s infinite`. **The repo
contained zero `@keyframes`.** Three static dots, shipping as a "streaming indicator" — a
reference to nothing, which looks exactly like a static-by-design decision until you check.

- `@keyframes gds-chat-typing` is now defined in `styles.css`, opacity-only so a loop that runs
  for the length of every response stays off the paint path.
- The dots carry `data-gds-motion`, so the existing governed rule neutralises them — no new
  media query. **Verified live: 1 running animation normally, 0 under
  `prefers-reduced-motion`,** with `animation-name` computing to `none`. Without that
  attribute a working infinite animation would have kept running for a user who asked for
  less — WCAG 2.2.2.
- New `ambient: 1000ms` step on the motion scale, added deliberately rather than mapped: every
  other step is a *transition* duration and the longest is 360ms, which loops frantically. The
  dot stagger is a fraction of the same token instead of a second hardcoded number.

**`createGdsMotionCssVariables` now derives its output from `gdsMotionDurations`/`gdsMotionEasings`
instead of hand-listing them.** That duplication is why adding `ambient` to the record emitted
no CSS and nothing failed — the scale was called the source of truth and was in fact a copy.

`motion-keyframes` is now in the registry's `EXPECTED_KINDS`, so a kind sitting at zero fails
extraction instead of passing unremarked, which is how this survived.

### A governed token the component kept a private copy of (#591)

`--gds-tour-spotlight-padding` was declared and read by nothing. The report inferred from the
CSS that the spotlight hole took the target rect with no inflation — **that was wrong about the
mechanism, and the difference matters.** `GdsTour.client.tsx` already inflated the rect, by a
hardcoded `8`: the same number the token declares. So the token was not an unimplemented
feature, it was a governed value with a private copy beside it, and a theme retuning the token
changed nothing.

Reading the token changes **no pixels** — which is why this needed no design review under #586
§13, unlike the visible change the report anticipated. Verified live: horizontal inflation is
exactly the token's 8px on both sides, with the 12px radius applied.

`verify:token-reachability` now counts `getPropertyValue('--gds-*')` as a reference. Reading a
custom property through the CSSOM is a real consumption — the tour must turn it into a number
before it can inflate a measured rect — and a gate blind to that reports a live token as
orphaned, which pressures toward a false allowlist entry or a contorted CSS-only component.

### Badge contrast is now measurable, and it is measured (#597)

**The defect was not low contrast. It was unmeasurable contrast.** Mantine's
`variant="light"` paints a low-alpha tint, and GDS's own badge rule mixed its tint against
`transparent` — a translucent background has no contrast ratio of its own. Every contrast
sweep in this system therefore walked past **344 badges** without measuring them. Zero
findings from a check that cannot see its subject reads exactly like a clean bill of health,
which is how #534 shipped as "two badges in dark mode" when it was every badge the rule
touched, at **2.25:1** worst case across the 25 presets.

New gate `npm run verify:badge-contrast`, chained into `verify:release`. It sweeps every
pattern route in both schemes and fails on **two** conditions, the first mattering more:

1. **Uncomputable** — a partially transparent badge background, whose legibility depends on
   whatever surface it is dropped onto, so no token-level guarantee can exist.
2. **Below 4.5:1** — the pair resolves opaquely but fails WCAG 1.4.3.

It also fails if it finds no badges at all: zero findings from a broken selector is
indistinguishable from a clean system, which is the failure this gate exists to end.

**Measured on the rendered page: 544 badges across 48 route x scheme combinations — 0
uncomputable, 0 below 4.5:1.** The gate's negative control is not hypothetical; during
development it reported 344 uncomputable and then 26 low-contrast pairs, each of which was a
real shipped defect traced to source.

Six root causes, all fixed in the shared system rather than at any call site:

- `packages/gds-theme/styles.css` — the preset badge rule mixed against `transparent` and
  used a 76% hue-mixed foreground. Now mixes against `--gds-bg-card` with `--gds-text-body`:
  **8.63:1** worst case across 25 presets x 2 schemes.
- `packages/gds-theme/styles.css` — the owned-contrast badge rule had the same shape, mixing
  against `--gds-vibe-surface`, which is translucent in most presets.
- The Class USA / Gold Athlete dark-mode badge override (#533) is **deleted**, not moved: it
  existed to correct the general rule's transparent mix, which no longer happens. Two
  `!important` declarations went with it.
- `GdsProvider` now applies the governed `light` variant to whatever theme it is handed. A
  guarantee a caller can drop by passing a different theme object is not a guarantee — and
  `resolveGdsThemePreset` does not descend from `gdsTheme`, so it was being dropped.
- `GdsCountBadge`, `FitScoreChip`, `MeaningBadge`, `LabelTag` — each paired a themeable fill
  with a FIXED foreground. `GdsCountBadge` used `--gds-state-info-dark` assuming the suffix
  named a dark colour; it names the dark-SCHEME value and resolves to `rgb(239, 242, 246)`.
  That badge rendered white on white at **1.07:1**. All four now use foregrounds derived
  against the fill they actually land on.
- New derived tokens `--gds-brand-accent-fg`, `--gds-brand-accent-action-fg`,
  `--gds-bg-info-tag-fg`, `--gds-brand-accent-tint-fg` (+ `-dark`), and
  `--gds-badge-solid-neutral` redefined from the card surface to a real neutral fill.

### The same defect outside badges, found by generalising the check (#597)

The badge work found one mistake four times: a themeable fill paired with a FIXED foreground.
`--gds-text-on-inverse` is derived to sit on `--gds-bg-inverse` and on nothing else. The
runtime badge gate cannot see these outside badges; the token gates cannot see them either,
because the pair is never *declared* — a component composes it at the point of use out of two
tokens that were never meant to meet.

New gate `npm run verify:component-color-pairs`, chained into `verify:release`: it reads every
style object in `packages/` that sets both a background and a text colour, resolves both
through `var()` fallbacks, and measures the pair across all 25 presets x 2 schemes. Static —
no browser, no build — so it carries a real mutant rather than a dated exemption.

It found five latent defects on first run, all now fixed:

| Component | Pairing | Worst |
| --- | --- | --- |
| `SemanticButton` (primary) | `text-on-inverse` on `brand-primary` | **1.00:1** — the same colour, default/dark |
| `ChoiceChip` (active) | `text-on-inverse` on `brand-primary` | **1.00:1** |
| `BottomTabBar` (emphasized) | `text-on-inverse` on `brand-accent` | 1.22:1, athlete-gold/dark |
| `ListingCard` (tint pill) | `brand-accent-action` on `brand-accent-tint` | 1.60:1, high-contrast/dark |
| `SemanticButton` (accent) | `text-on-inverse` on `brand-accent-action` | 1.66:1, gold-athlete/dark |

**Scope, stated precisely:** these are latent, not currently visible. A live sweep of all 24
pattern routes in both schemes found **zero** rendered elements whose text matched their
background — the playground does not exercise these combinations today. They are public
component API, so a consumer reaches them before we would.

`--gds-brand-primary-fg` joins the derived foregrounds. One pair is **reported, not enforced**:
disabled control text measures 3.50:1 in class-usa light, and WCAG 1.4.3 exempts text in an
inactive component. The gate prints it with its ratio every run rather than dropping it —
an exemption nobody can see is indistinguishable from a gap nobody noticed.

**Visible change, stated rather than slipped in:** badge text is neutral instead of
hue-tinted, and `LabelTag` renders on a soft tint instead of a transparent background. The
tint still carries the preset's identity. Token baseline delta: 400 additions, 200 changes,
0 removals — the new foregrounds and the redefined neutral, nothing else.


No package or component source changed; no version bump. Tooling, governance,
and documentation only.

- **The verification gates are now mutation-tested themselves (#580)**: added
  `npm run verify:gates`, chained into `verify:release`. The verdict is **inverted** —
  under a planted defect a correct gate must FAIL; a gate that still exits 0 has proven
  it does not detect that defect, and each mutant's `claim` string names precisely which
  assertion is thereby unsupported.

  **8 of 10 mutants killed (80%)**, including one reproducing #516's exact false pass —
  which is why #516 had to land first.

  **Two genuine survivors, and they are a real finding (F22).** `verify:theme-tokens`
  and `verify:theme-accessibility` both pass with a semantic token renamed, and with
  `--gds-text-body` set to near-white on a light canvas. Verified as gate weaknesses
  rather than bad mutants: `validateGdsTokenGraph()` validates 425 tokens = 17 roles ×
  25 themes (the vibe atmosphere palette), and the accessibility report scores
  `vibe.textLight` directly, not the derived `--gds-*` roles. **The 73 semantic tokens
  that determine what a component looks like are outside both gates' scope** — so a
  semantic token can be renamed or dropped below its contrast floor and `verify:release`
  stays green. This is F12's root cause reaching further than F12 reported, and it is the
  mechanism behind #537. Recorded as `KNOWN_SURVIVORS` against #585, dated, and reported
  every run.

- **Semantic token values now have exactly one definition (#554)**: added
  `packages/gds-theme/src/semantic-token-source.ts` and
  `npm run verify:token-single-source`, chained into `verify:release` after `build`.

  GDS's Core Principle 3 is "One Token Source". That promise was false inside the theme
  package itself: `brand-tokens.ts` derived the Class USA and Gold Athlete semantic roles
  as `role -> { light, dark }` pairs while `vibe-themes.ts` carried the same values a
  second time as flat `--gds-*` records — with a comment asking a human to keep them
  byte-identical. The cost had already been paid once, in the `GdsProvider` inline-style
  token bug fixed in 5.0.1/5.0.2.

  **There was a third copy, not two** (F26). `createBrandTheme` applied per-lane overrides
  *after* calling the emitter, so `--gds-brand-accent-action` was unreachable through the
  emitter. The first consolidation attempt therefore introduced a divergence in exactly
  that role. Folded into lane emitters (`emitClassUsaCssVariables`,
  `emitGoldAthleteCssVariables`) which are now the complete definition of a lane.

  **Migration proof**: all 25 presets x 2 schemes plus both brand themes were snapshotted
  from the pre-refactor build and compared after — **0 values changed, 0 tokens removed,
  16 added**. The 16 are the camelCase aliases (`--gds-brand-primaryPressed`,
  `--gds-text-onInverse`) that the provider path already emitted and the document path did
  not; both paths now emit identical key sets. The snapshot is committed as a fixture and
  a test fails on any future drift.

  The gate asserts **both** structure (no parallel table outside the owning module) and
  behaviour (both consumption paths resolve every shared role to the same value). Structure
  alone would have missed the third copy, which was two assignment statements rather than
  a table.

  `vibe-themes.ts` cannot simply import `brand-tokens.ts` — that path closes a cycle
  through `token-operations.ts`. The dependency-free module is what makes single-sourcing
  reachable at all, not merely tidier.

- **The two export-coverage gates now share one internal-exports list**: `verify:api-docs-coverage`
  and `verify:pattern-export-coverage` each carried a hand-maintained copy of the same 23
  names — the same dual-source pattern #554 exists to remove, sitting in the tooling that
  polices it. Extracted to `scripts/config/internal-exports.config.mjs`.

- **Registry-derived obligation coverage (#581)**: added
  `npm run verify:obligation-coverage`, chained into `verify:release`. Obligations are
  derived from `audit/registry.json` rather than a hand-maintained checklist, so a new
  prop, variant or accent acquires its obligations **the moment it is added** — there is
  no list to forget because there is no list. Props owe a JSDoc line, variants owe a
  playground demonstration, accents owe contrast evidence; the obligations are
  deliberately *not* uniform across kinds, and each kind's rationale is stored in
  `scripts/audit/obligation-model.config.mjs` so the choice is reviewable rather than
  folklore.

  **Measured: 410 gaps** — 373 props with no JSDoc, 37 variants rendered by no demo.
  Ratcheted via the new `obligationGaps` budget: existing debt never blocks work, adding
  to it does.

  **A measurement change is stated, not hidden.** `registryAtomsWithoutCoverage` drops
  1,699 → 0 because it now measures a different thing (atoms with neither an obligation
  model nor a recorded owner). The 1,699 did not evaporate — 410 carry real unmet
  obligations under the new budget. `audit/budgets.json` records this in the entry
  itself, because a budget that silently changes meaning is indistinguishable from one
  that was gamed.

- **Map accessibility surface (#568)**: `GdsMap` renders a text-equivalent marker list, a
  throttled live region, and selection state exposed to assistive technology.

  **The list is not optional and not opt-out.** `listPlacement` moves it; nothing removes it. A
  prop that hid it would make conformance a consumer choice — and for a raster map the list *is*
  the conformance path, because tile imagery is decorative by nature and cannot be described.

  **One ordering, shared by map and list**, sorted by label rather than left in insertion order:
  a list sequenced by "whatever the API returned" is not navigable, and a keyboard user
  traversing it cannot tell where they are.

  **Announcements are throttled and coalesced.** Continuous panning fires `moveend` repeatedly,
  and an unthrottled live region turns that into a screen reader reading coordinates over and
  over — worse than silence, because the user cannot escape it. What it says is informative
  rather than a coordinate readout: *"12 places in view"*.

  Selection is carried by `aria-pressed` and a **border**, not only a background tint — under
  forced colors a background is replaced by the system palette and selection would vanish,
  which is the state a keyboard user most needs to see. An approximate position says so rather
  than implying precision.

  **Conformance claim, scoped per Rule 12**: keyboard operability of the list, selection
  exposure, ordering stability, live-region politeness, and the empty state are covered by
  automated tests. **Not covered**: real screen-reader output, forced-colors rendering in a
  browser, and Leaflet's own keyboard panning — those need the runtime harness and are stated
  as gaps rather than implied by the tests that do pass.

- **Governed map surface on OpenStreetMap (#566, #567)**: `@sovereignsquad/gds-core/map`
  exports `GdsMap` — Leaflet-backed, real OSM raster tiles, on a dedicated subpath so consumers
  who never render a map do not pay for a browser-only 40KB engine.

  **Leaflet, not MapLibre**, and the reasoning is recorded in `vendor-governance.json` so a
  future reversal is a decision rather than a rediscovery: no WebGL requirement (which matters
  for forced-colors and low-end devices), and raster tiles suffice because vector-tile styling
  is out of scope. It is pinned **exactly**, not by range — a map engine renders third-party
  tiles and holds imperative DOM, so a silent minor bump is not something to learn about from a
  broken map.

  **The tile source and its attribution are one object.** OSM data is ODbL-licensed and the
  credit is a licence condition, not a styling preference — so `GdsMapTileSource` cannot be
  constructed without one, `assertGdsTileSource` throws rather than falling back to a default
  (silently substituting a different map would be worse than failing), and the shared source is
  frozen so nobody can strip the credit off it for every map at once. Leaflet's own attribution
  control is disabled and the credit renders as GDS UI, so it does not live inside third-party
  chrome a consumer might restyle away.

  **The map re-initialises on theme identity**, which is exactly the case #561 was built for:
  Leaflet reads resolved colours when it constructs its panes, so no CSS variable change ever
  reaches them. Without an explicit destroy and re-init, a theme switch leaves a map painted in
  the previous theme.

  Marker labels are **required and consumer-supplied** — never derived from an icon's import
  name, because "IconMapPin2" is not what a screen-reader user needs to hear, and a marker whose
  only identity is its colour does not exist for them. State is announced rather than only
  styled, and a failure to load says so instead of leaving an empty box that reads as "nothing
  here".

  The imagery gate now excludes slippy-map **tile templates** by their structural `{z}/{x}/{y}`
  signature: map tiles are the map itself, and no thumbnail generator can produce the surface of
  the earth. A stock-photo URL cannot match that signature, so the exclusion cannot widen.

- **Generated imagery only (#563, #564)**: the reference site no longer renders a single
  third-party image, and `npm run verify:generated-imagery-only` fails the build if one
  returns.

  **Measured first**: the surface was two `picsum.photos` photos in the media-field demo — the
  page that demonstrates the design system was illustrating it with hosted stock photography.
  Both now render through `GdsGeneratedThumbnail`. A hosted photo breaks three properties the
  rest of the system guarantees at once: it needs the network, it does not follow the theme,
  and it is not deterministic.

  **Two false-positive classes were removed rather than allowlisted.** RFC 2606 reserves
  `example.com` for documentation and it resolves to nothing, so a demo showing "here is a URL
  a user pasted" is text in a form field, not an image — four permanent allowlist entries for a
  case that cannot load a pixel would be an allowlist people stop reading. And the CSS `url()`
  rule was narrowed to image resources after it caught the Google Fonts `@import`: a font, not
  an image, governed by the font-lane system instead.

  **That second false positive was masking a negative control** — the gate was failing for the
  wrong reason, so its real coverage went unmeasured. A gate that fails is not the same as a
  gate that works.

- **Theme coverage matrix (#562)**: `npm run verify:theme-coverage-matrix` renders the site
  under a covering design and checks that each tracked property **resolves from a governed
  token**, per preset and per scheme.

  **The coverage problem it fixes.** The audit's Phase 1 executed **4 of 24 routes at 5 of 25
  presets**. Every finding it produced is real; every *non*-finding was worthless, because "no
  untraceable value on route X under preset Y" says nothing when X and Y were never visited.
  This sweep visits **all 24 routes and all 25 presets** in both schemes — 50 cells rather than
  the 1,200 of an exhaustive matrix, chosen so neither factor is left unvisited.

  Ten tracked properties across all six axes, so the check is provenance rather than
  appearance: a value that merely looks right under the one preset somebody opened is exactly
  the defect.

  **The first result was wrong and is worth recording.** It reported **52% untraceable** —
  almost all of it an artifact of comparing computed styles against *declared* token values.
  `#ffffff` never equals `rgb(255, 255, 255)`; `calc(1rem * var(--mantine-scale))` never equals
  `16px`. That measures formats, not provenance. The oracle is now resolved **in the browser**
  through a probe element, and the real figure is **34%**.

  **It is deliberately NOT in the per-push chain.** Adding it there was tried and measured:
  a CI run passed 39 minutes against a ~7 minute baseline, because the sweep runs on both
  Mantine legs. It runs on demand via `npm run verify:theme-coverage-matrix`, and
  `verify:budgets` still guards the committed measurement — so a regression fails the build
  without every push paying for a browser sweep. A gate slow enough to make people avoid
  pushing is a gate that gets removed.

  **The sweep is not yet reproducible**, which §5 requires: three consecutive runs on one
  commit measured 33.94/33.95/33.96% with element counts of 28,128–28,242, because render
  timing changes how much of a route is present when it samples.

  So the artifact records **whole percent and counts to the nearest thousand** — the precision
  the sweep can actually hold. One decimal was tried and still moved, because the band straddles
  33.9 and 34.0. This is not rounding to make a number look stable: a committed artifact that
  changes every run makes the clean-tree rule unsatisfiable, and a real leaked mutation would
  then stop being visible among the noise. Recording two decimals would assert precision the
  measurement does not have. Tracked as #599.

  **That number is not comparable to `untraceableRenderRate`'s 18.4%**, and it has its own
  budget key so nobody compares them. F5's figure came from a colour-focused sweep over 4
  routes; this one tracks padding, font-size, font-weight, transition-duration and
  outline-width across every route and preset. **A higher number here is a wider lens, not a
  regression.**

- **Total theme re-application (#561)**: `GdsProvider` gains `themeApplicationMode`
  (`remount` | `reload` | `cascade-only`), `reloadOnThemeChange`, and
  `onBeforeThemeApply`/`onAfterThemeApply`. The themed subtree is keyed on a **theme
  identity**, so a switch re-creates the components holding values the cascade cannot reach.

  **Why the cascade is not enough.** `var()` values update on a switch by themselves. What
  does not: a value read with `getComputedStyle` at mount and put in state, a `useMemo` whose
  deps omit the theme, SVG or canvas painted once from resolved colours, and third-party
  surfaces initialised with a theme snapshot — which will include the map engine.

  **The identity hashes the RESOLVED tokens, not the declaration.** Two declarations that
  render identically produce the same identity and therefore no remount — repainting the world
  to arrive at the same pixels is cost without benefit. Conversely a theme changing only its
  radius scale *does* change identity, because every axis is in the resolved set. Keying on
  `preset + scheme` would miss exactly that.

  **The provider does NOT remount its whole subtree, and that was measured rather than
  assumed.** Keying everything under `GdsProvider` was tried first: it destroys the state of
  any theme control living inside the provider — the normal arrangement — and broke three of
  the playground's own runtime tests by resetting the very picker used to change the theme. A
  default that resets the control you just used is a defect, not a guarantee. Total
  re-application is opt-in per subtree via the new **`GdsThemeBoundary`**, placed around the
  surfaces that actually hold theme-derived state outside the cascade.

  **The reload escape hatch is deliberate**, and it is the owner's own requirement: for a
  surface that cannot be made to re-read a theme, a reload is the only honest way to guarantee
  full application. Better an explicit reload than a page silently rendering half the previous
  theme.

  `verify:theme-identity` proves all 50 preset/scheme identities are distinct and stable — a
  collision would mean a switch that does not remount. The **runtime stale-value detector
  needs a browser and is filed separately** rather than stubbed: a rule that cannot be
  evaluated looks like coverage.

- **Theme Lab live accent contrast matrix (#596)**: `GdsAccentContrastMatrix` renders every
  accent × shade × mode for the selected preset and scheme, with its **measured ratio and
  required threshold** — never a bare "fails". A theme author sees which combinations pass
  before shipping rather than after a failed build.

  **It reads the same evaluator the gate does.** A second computation could disagree with
  `verify:accent-contrast`, and a UI that contradicts the build is worse than no UI.

  **The matrix clears the floor it reports on.** State is carried by icon and numeric ratio,
  not colour — a component reporting on accessibility that failed 1.4.1 itself would be worse
  than not shipping it. The table is captioned, so a screen-reader user knows what 120 numbers
  mean before reading them, and accents are row headers rather than plain cells.

  Under `auto` colour scheme it renders **both** tables: `auto` means "follow the OS", so a
  single table would show one scheme's numbers while the reader might be looking at the other.

  Modes the gate does not enforce are shown and marked, with the reason — hiding them would
  make the matrix look like it covers less than it does. Copy added in all 8 locales.

- **Scheme-aware badge tones, resolving #534 (#595)**: badge tone colours are now **computed
  pairs** — background and foreground derived against each other, per preset, per scheme.

  **#534's actual mechanism was not what the follow-up issues assumed**, and the record is
  corrected. It is not the `-dark` pinning in `toneColors`; it is `StatusBadge` rendering
  Mantine's `variant="light"` — pastel text on a low-alpha tint of the same hue — measuring
  **1.81:1** and **2.55:1** in dark mode. GDS never controlled that pair because it came from
  a Mantine variant rather than a GDS token, and **an rgba tint's contrast cannot be computed
  at all**, which is precisely how it shipped unnoticed.

  **A second, separate defect was found while verifying the first.** `toneColors.success` read
  `--gds-state-success` while the other three tones read their `-dark` variants. Paired with a
  fixed white foreground it fails 4.5:1 in **9 of 25 presets in light mode** (class-usa 4.10,
  sunset 4.40) — shipping today. The inconsistency is why it survived: three tones were pinned
  dark and one was not.

  Both have the same root cause and the same fix, the one #537 established for `support`:
  **never pair a fixed foreground with a variable background.** The soft lane mixes the state
  colour against a real surface — producing an opaque colour whose contrast *can* be measured,
  unlike an alpha tint — and derives the foreground against the result.

  **500 pairs verified** (25 presets × 2 schemes × 2 lanes × 5 tones), 0 failures, and a new
  accessibility-floor rule keeps it that way. A status badge is often the only signal that
  something needs attention, so an illegible one is a functional failure, not a cosmetic one.

- **Accent axis — components migrated (#594)**: `GdsBadge`, `GdsMapPinBadge` and the
  generated-art engine now read `--gds-accent-*` instead of a module constant, so a category
  colour follows the active theme.

  **The two hand-authored tables in `gds-core` are gone.** `gdsBadgeAccentColors` (10 values)
  and `gdsBadgeAccentShades` (40) are now *derived* from the axis and marked `@deprecated`;
  they survive only as `var()` fallbacks and for non-DOM rendering. Nothing is typed by hand,
  including the fallbacks — a hand-written fallback is a second definition that drifts the
  moment the axis changes.

  **The resolved-value split that made this its own slice.** `generated-art-engine` composes
  OG images, share cards and email *outside a document*, where a CSS custom property resolves
  to nothing — a `var()` there produces an unpainted shape rather than an error. It now has
  two resolvers: `resolveCategoryColorToken` for the live DOM and `resolveCategoryColorHex`
  for everything else, the latter resolving against the requested preset so a themed share
  image carries that theme's accents. Every other consumer wants the reference; this one
  needs both, which is why it could not be folded into #593.

  The file's own comment justified the old behaviour — *"deliberately theme-independent fixed
  sRGB, there is no variable to reference"*. That premise was true when written and the accent
  axis makes it false; the comment is replaced rather than left to mislead.

  Nine test assertions pinned the literal values. They were updated to assert the **new
  contract** — token in the DOM, literal outside it — not loosened.

- **Accent axis — token layer (#593, split from #560)**: the last frozen surface in GDS is
  now theme-controlled. `--gds-accent-<name>-<shade>` for 10 accents × 4 shades × 25 presets
  × 2 schemes, with `npm run verify:accent-contrast` chained into `verify:release`.

  **No hand-authored values.** A theme declares ten **base** colours; every shade is derived.
  An earlier cut of this change carried all 40 shade values explicitly to reproduce the
  hand-tuned table byte-for-byte — that was the wrong trade and it was removed. What must be
  preserved is the **guarantee** (white text stays legible on every accent and shade), not the
  specific hexes somebody once picked.

  **The cost, stated:** 28 of 40 shade values changed, by at most 14/255 per channel. **The
  guarantee did not** — all 2,000 enforced combinations pass. It is now *computed* rather
  than *asserted*, which is the entire reason the palette was frozen in the first place.

  **Enforcement is scoped to what actually renders**, established by reading the components.
  A first pass enforced all three modes and reported 3,000 failures of 6,000 — every one an
  artifact of the model. `GdsBadge` renders accents **filled only**; no component draws an
  accent as text on a page, and the emoji disc is `aria-hidden` decoration whose meaning is
  carried by the required label. Those two modes are measured and reported but not enforced,
  each with its reason, because gating a mode that does not render is how a gate teaches
  people to ignore it.

  Derivation is darker-only (lightening breaks the filled-mode guarantee — the source records
  `teal` failing at +4 lightness), factors must strictly decrease, neighbouring shades must be
  perceptibly apart, and a declared ramp **replaces** rather than merges over the default —
  merging would let a new base silently keep the old base's shades, which is defect #537 in
  another namespace.

- **Accessibility floor: minimums no theme may cross (#559)**: added
  `npm run verify:a11y-floor`, chained into `verify:release`, plus a generated
  `docs/ACCESSIBILITY_FLOOR.md`.

  The six axes gave themes real control over shape, density, type, elevation, motion and
  reaction. Every one of those is a lever that can be pulled into an accessibility regression
  while looking like a design decision. This is the contract naming which pulls are not
  available: 7 rules × 25 presets × 2 schemes, each rule citing its WCAG criterion and naming
  the axis field to change.

  **There is no warning tier.** A floor breach fails the build. A warning would make the floor
  advisory, and an advisory floor is not a floor.

  **Contrast is enforced but not reimplemented.** `createGdsThemeAccessibilityReport()` already
  scores every colour pair across every preset and scheme; the floor adopts its blocking
  findings. A second contrast implementation could disagree with the first, and two
  accessibility verdicts on one pair is worse than one.

  **The documentation is generated from the rules.** A floor described differently from how it
  is checked is a floor nobody can rely on, so the gate fails if the doc and the rule set
  disagree.

  **The gate proves its own rules are live before trusting a clean run.** Zero violations is
  both the result we want and what a floor checking nothing reports — the count alone cannot
  tell them apart, which is finding F19's failure mode. So it first feeds the floor a
  deliberately non-compliant theme and asserts every rule fires; if that canary comes back
  clean, the gate fails rather than reporting a green it cannot justify.

  Rules needing real rendered geometry are deliberately absent — they belong to the runtime
  harness. A rule that cannot be evaluated is worse than a missing rule, because it looks like
  coverage.

- **Motion and reaction axes (#558)**: axes five and six — the theme now controls timing,
  interaction feedback and focus-ring geometry. `--gds-reaction-{hover,active,pressed}` plus
  resolved `-lift`/`-scale` values, `--gds-focus-ring-{width,offset,style,color}`,
  `--gds-transition-scope`, and per-preset `--gds-motion-*` overrides.

  **Motion emits nothing unless a preset overrides it.** The global scale in `styles.css` is
  generated from `motion.ts` (#584) and *is* the default; restating it per preset would
  duplicate it 25 times and stop the generated stylesheet being the source.

  **Shipped names are honoured over the issue's.** The spec says `normal`/`enter`/`emphasized`;
  the shipped tokens are `base`/`entrance`/`emphasis`. Renaming would break every consumer of
  `--gds-motion-duration-base` and re-fork the scale #584 had just unified — the axis exists
  to let a theme change these *values*, not their names.

  **There is deliberately no way to ignore a user's reduced-motion preference.** A theme may
  make motion calmer than the user asked for; it may never make it louder, so no such value
  exists in the type.

  The focus ring is validated hardest, because it is the one piece of feedback a keyboard user
  cannot do without: at least 2px, a style from a closed set, and a colour **role** rather than
  a literal — a literal cannot follow the theme, and cannot be contrast-checked against the
  surface it lands on. Intensity resolves to concrete values so components never branch on the
  keyword, and `none` means none rather than a small nudge.

- **Typography and elevation axes (#557)**: axes three and four. Typography emits
  `--gds-font-size-*`, `--gds-weight-*`, `--gds-line-height-*`, `--gds-tracking-*` and
  `--gds-font-lane-*`; elevation emits `--gds-elevation-0…4` plus seven surface roles.

  **The type scale carries Mantine's sizes as overrides rather than deriving them.** Mantine's
  ramp is not a clean modular scale — `0.875 → 1` is ×1.1429 while `1 → 1.125` is ×1.1250 — so
  any single ratio would round its way to different numbers and change every piece of text on
  the site. The ratio governs only the steps Mantine has no equivalent for (`2xs`, `2xl`,
  `3xl`, `4xl`).

  **A namespace collision was caught before it shipped.** Size steps are `--gds-font-size-*`,
  **not** `--gds-text-*` — the latter is already the semantic *colour* namespace
  (`--gds-text-body`, `--gds-text-primary`). One prefix meaning both a colour and a size would
  leave a reader unable to tell which `--gds-text-lg` is, and the token graph's category
  inference keys off exactly that prefix.

  Invariants are the ones a theme can plausibly get wrong: weights must **ascend** (a
  `semibold` lighter than `medium` renders as broken text, not as a style), the ratio must be
  a real ratio, every lane must name a **registered** font lane (an unregistered one falls
  back to the browser default, which reads as the theme failing to load), and elevation must
  **not decrease** (a modal flatter than the card behind it reads as a rendering bug).

  The DTCG generator gained `number`, `cssShadow` and `cssKeyword` types, and now treats any
  `calc()` as computed — a derived step like `calc(1rem * 1.4238)` is no more a static
  `dimension` than one referencing `var(--mantine-scale)`.

- **Density axis: spacing, control heights, and a hit-target floor (#556)**: the second axis
  through the #555 mechanism. 10 spacing steps + 5 control heights + a density mode, emitted
  as `--gds-space-*`, `--gds-control-height-*` and `--gds-density`.

  Mantine's `theme.spacing` is fed from the axis, as `theme.radius` is from the shape axis,
  so every `p="md"` and `var(--mantine-spacing-*)` already in the codebase resolves through
  one declaration. `xs`–`xl` are Mantine's values verbatim; `none`, `3xs`, `2xs`, `2xl`, `3xl`
  and all control heights are additions nothing consumed before.

  **The 44px target floor is enforced, and it found a design contradiction.** A control
  *declared* below 44px is a build error. But *scaling* is clamped, not rejected — 44px under
  `compact` ×0.75 is 33px, so throwing would have made the floor quietly ban an entire density
  mode instead of protecting it. Spacing tightens; hit targets hold their line. `xs`/`sm` scale
  freely because their recorded exception is precisely the statement that they are not primary
  hit targets.

  `verify:density-token-adoption` treats `0` as a reset rather than a spacing decision, and
  `em` values as font-relative by intent. Of 47 spacing declarations: 10 token-governed, 2
  migrated on exact matches (`0.625rem` = `xs`, `1rem` = `md`), 5 allowlisted with reasons —
  `0.5rem` genuinely falls between `2xs` and `xs`, so migrating it is a visible change and a
  decision rather than a sweep.

- **Theme axes, and the shape axis as the first one (#555)**: a theme can now declare
  non-colour design decisions. `packages/gds-theme/src/axes.ts` adds a typed axis mechanism;
  the shape axis delivers a 7-step radius scale plus 14 semantic roles (`card`, `button`,
  `pin`, `thumbnail`, …), emitted as `--gds-radius-*` per preset and published in the DTCG
  graph. `verify:shape-token-adoption` is chained into `verify:release`.

  **The scale feeds Mantine's `theme.radius`**, so the 130 `radius="md"`-style props and 16
  `var(--mantine-radius-*)` references already in the codebase became axis-governed with
  **zero component edits**. Migrating 146 call sites by hand would have been the obvious
  reading of "components read radius from tokens" and the wrong one — it would have left the
  axis as a parallel scale that only new code consults, which is the dual-source shape #554
  spent a change set removing.

  **Zero visual regression, verified**: 0 token values changed, 0 removed, 1,050 added
  (21 tokens × 25 presets × 2 schemes). The defaults are Mantine's own values **captured
  verbatim, including `calc(… * var(--mantine-scale))`** — writing a tidier `0.5rem` would
  have silently dropped the scale factor, a real rendering change disguised as cleanup.

  15 hardcoded `borderRadius` literals were found. 7 migrated to tokens; 8 allowlisted in two
  clearly separated categories — **6 `circle`** (`50%` is a shape, not a radius step; feeding
  it through the scale would turn avatars and status dots into rounded squares under any
  small-radius theme) and **2 `debt`** (Theme Lab chrome at 18px and 12px, off-scale, whose
  migration would visibly reshape the page that demonstrates theming — a design review, not a
  sweep, and near-dated accordingly).

  Axes validate at **theme-construction time**, not render time: a bad radius found while
  rendering is a visual defect someone has to notice; found while building the theme it is an
  error naming the offending key.

- **Quality budgets are reported on every pull request (#582)**: added
  `npm run budgets:report` and a `budget-report` job in `.github/workflows/quality.yml`.
  `verify:budgets` now writes `audit/budget-results.json`, and the report renders from that
  artifact rather than re-measuring — a second resolver could disagree with the gate, and a
  report that contradicts the gate is worse than no report.

  Each budget shows measured, base, delta and direction. A `max` budget rising is a
  regression; a `min` budget rising is an improvement — inverting that would tell a reviewer
  the opposite of the truth about their own change, so it is unit-tested. The comment is
  updated in place via a marker, so a 30-push PR carries one comment rather than thirty.

  The job **cannot fail the build**. #578's gate enforces correctness; a broken comment must
  not block a correct PR, and a comment that can fail a build is one people will want
  removed.

  **Scope of verification, stated:** the report script, the delta rules and the workflow YAML
  are verified locally (5 unit tests; a synthetic base exercising regression, improvement and
  missing-artifact paths). The **PR-comment path itself is unverified** — it runs only on
  `pull_request`, and this repository pushes directly to `main`, so it will first execute on
  whichever PR is opened next.

  `scripts/**/*.test.mjs` is now in the vitest include. Nothing under `scripts/` could be
  tested before, which is why 45 verification gates had no unit coverage.

- **Motion is single-sourced and every shipped transition is on the scale (#584)**: added
  `npm run tokens:motion-css` (generator), `verify:motion-css` (drift) and
  `verify:motion-scale`, all chained into `verify:release`.

  The reported symptom was that button micro-animations looked absent. The measured reality
  was that they existed and bypassed the governed curve — 34 interactive elements computing
  to `0.14s`/`ease` while `--gds-motion-duration-fast` resolved to `.12s` on the same page.

  **The F1 decision, and why both of the issue's options were wrong.**
  `createGdsMotionCssVariables` is a documented consumer-facing support API, so deleting it
  breaks a published contract; and the stylesheet must work for consumers who never call JS,
  so deleting the static block is worse. What was duplicated is the nine **values**, typed a
  second time into `styles.css`. Those blocks are now generated from
  `gdsMotionDurations`/`gdsMotionEasings` and drift-checked — including the reduced-motion
  block, generated from the same emitter under its `no-motion` policy so the two cannot
  disagree about what "reduced" means.

  **Perceived timing changes, stated for review rather than slipped in**: links and buttons
  move `140ms → 120ms`, the tour spotlight `220ms → 240ms`, and everything gains the governed
  `cubic-bezier(0.2, 0, 0, 1)` in place of browser `ease`. Reduced-motion behaviour is
  provably unchanged — both guard rules were read before substituting.

  **New finding, #592**: `ChatSurface` declares `animation: gds-chat-typing 1s infinite` and
  **`@keyframes gds-chat-typing` is defined nowhere — GDS ships zero `@keyframes`**. The
  typing indicator is three static dots. Were it working it would also ignore
  `prefers-reduced-motion`, since the dots carry none of the guarded selectors.

- **The published token graph now describes the system, and the contrast gate can see it
  (#585)**: `tokens/gds.tokens.json` carried 17 vibe atmosphere roles while the tokens that
  paint components number 34, and the overlap was exactly one (`accent`). A design tool
  importing it received background colours and none of the roles that determine what a
  component looks like.

  `createGdsTokenGraph()` now emits 425 atmosphere nodes (unchanged) plus 850 semantic nodes
  — 34 roles x 25 presets — read from the same resolver the runtime applies to the document.
  Scheme is a first-class dimension: each semantic token carries `{ light, dark }` under
  `$extensions`, with `$value` holding the light value so a tool that ignores extensions
  still gets something correct. 35 theme-invariant tokens are published once in a `global`
  group. Overlap ratchets **1 → 34 (100% coverage)**, measured by the generator rather than
  transcribed.

  **F22 is closed, and not by the graph alone.** Publishing the roles makes a *rename* fail
  `verify:tokens-dtcg` — but regenerating the artifact makes it pass again, so drift-checking
  is not sufficient for a *value* regression. That was measured, not assumed. The real fix is
  that `createGdsThemeAccessibilityReport()` now scores the semantic roles it never scored:
  450 new checks across 25 presets × 2 schemes (300 → 750). All pass today, so enforcement
  was added without changing any value; with `--gds-text-body` planted at `#f5f5f5` the gate
  blocks at 1.04:1 instead of passing silently.

  `KNOWN_SURVIVORS` is now **empty** and the gate mutation score is **17/17**.

  **No token value changed.** DTCG types are inferred, and an unclassifiable value throws
  rather than shipping a guessed `$type` — which fired immediately on `--gds-vibe-control`, a
  `color-mix()` over `var()` references, now typed `cssComputed` rather than being called a
  colour it cannot be parsed as. Global tokens are read from plain `:root` only, never the
  reduced-motion or forced-colors overrides. Artifact grows 152 KB → 620 KB.

- **All 15 unreachable tokens classified, and a gate against new ones (#586)**: added
  `npm run verify:token-reachability` and `scripts/token-reachability.config.mjs`.

  14 are documented extension points — published roles a consumer application reads while
  no GDS component does — each with evidence and an expiry date. 1 is a **pending wire-up**:
  `--gds-tour-spotlight-padding` is declared beside `--gds-tour-spotlight-radius`, which is
  consumed, while `GdsTour` sets the spotlight hole straight from the measured rect. Filed
  as #591, because wiring it up widens the cut-out by 8px and a token that starts rendering
  is a visible change to review, not cleanup to slip in.

  **The badge question is answered from the code, not the coincidence.** `GdsBadge`'s tone
  table reads `--gds-state-*` by design, as its own doc comment says, so the four dead
  `--gds-badge-*` tokens are unrelated to #534. Two of them are not dead *values* though —
  `badge.info` and `badge.urgencyBg` reach components under the alias names
  `--gds-bg-info-tag` and `--gds-brand-accent-tint`. Only `--gds-badge-attention` and
  `--gds-badge-validation` reach nothing at all.

  **The budget's 0 means "unclassified", not "clean"** — fifteen tokens are still referenced
  by nothing; none is now undocumented. Stated in the budget entry itself.

- **Two redundant token spellings removed**: `--gds-brand-primaryPressed` and
  `--gds-text-onInverse` were camelCase twins of aliases consumers actually read, emitted
  because `cssVarName` leaves the camelCase segment intact. The #554 entry above recorded
  them as "additive and harmless"; the reachability census found them as dead tokens, so
  they are deleted at the emitter. `--gds-control-disabledBg`/`-disabledText` are also
  camelCase but ARE consumed, so the naming is not uniformly dead and they stay.

- **Mantine dependency-boundary census (#589)**: added
  `npm run verify:mantine-governance`, chained into `verify:release` after `build`, plus
  `scripts/mantine-governance.config.mjs` as the delegation SSOT.

  Of 92 consumed `--mantine-*` custom properties: **4 governed** by `gdsTheme`, **6
  delegated** with a written reason and an expiry date, **1 dynamic** (a name built at
  runtime, so not statically governable), **8 lane-only**, **73 ungoverned**.

  **Governance is measured, not declared.** The gate compares GDS's theme against
  Mantine's `DEFAULT_THEME` and counts a variable as governed only when GDS actually
  changes the value. A hand-maintained roster of governed variables could claim authority
  GDS does not have — the same dual-source pattern #554 removed from the theme package.

  **New finding F28**: eight variables are governed by exactly one preset. `ChoiceChip`
  consumes `--mantine-color-teal-6`, but a GDS teal ramp exists only in
  `partnerDiscoveryThemePreset` — under the other seven lanes it renders Mantine's stock
  teal. These change owner depending on the active theme, so reviewing one preset shows
  governance and reviewing another shows a framework default.

  **The budget's definition changed and says so.** `undeclaredMantineDependencies` moved
  from the Phase 4c CSS-declaration count to the census; **87 -> 81 is not six variables
  remediated** — nothing was declared and no rendered value changed. The old number asked
  "is it written down?", the new one asks "does GDS dictate it?". Recorded in the budget
  entry itself.

- **`verify:gates` now runs after `build` (F27)**: it ran at step 2 of `verify:release`
  while `build` ran at step 5, so `verify:theme-tokens` and `verify:smoke-import-surface`
  — both of which read from `dist/` — were mutated on a tree that had no `dist/` on a
  clean CI checkout. They exited non-zero before any mutation was applied, which the
  inverted verdict scored as `KILLED`. **CI had been counting two mutants it never
  tested.** Invisible locally, where a leftover `dist/` always exists.

  Found by the F25 baseline assertion on its first CI run, the same day it landed.
  Verified by deleting every `dist/` in the workspace and running the full chain from
  scratch — the genuine CI condition rather than an approximation.

- **Two defects found in the audit's own tooling while building the above** (F24, F25 in
  `audit/FINDINGS.md`):

  - **F24** — a first cut modelled `jsdoc` for the `export` kind and reported 3/497.
    All **494** would have been false accusations: the registry records an export at its
    *barrel* line, while the JSDoc lives on the declaration in the component's own file.
    Caught only because 0.6% was implausible beside `verify:api-jsdoc-coverage` reporting
    99.8% for the same surface. `export` is now recorded in `COVERED_ELSEWHERE` naming
    the gate that actually owns it.

  - **F25** — `verify:obligation-coverage` read the budget key that had just been
    ratcheted to 0, so it failed on **every clean run**. It was invisible because the
    mutation suite reported its mutant `KILLED`: the verdict is inverted, so a gate that
    always fails "detects" everything and scores a perfect kill *precisely because it is
    broken*. Fixed twice over — the gate reads the correct key and treats a *missing*
    budget as a hard failure rather than `Infinity` (the #516 vacuous-pass shape), and
    `verify:gates` now runs every gate **clean before mutating it**, marking its mutants
    `INVALID / BASELINE BROKEN` if that run does not exit 0.

    This is the exact mirror of the false-`SURVIVED` class `requiresBuild` fixed: both
    come from interpreting a gate's exit code without first establishing what a clean
    exit code is.

  The gate suite also now snapshots and restores every `audit/*.json` artifact its child
  gates write — F21 recurring in a harness written before that lesson landed.

  Coverage is enforced: a release-chain gate with neither mutants nor a **dated,
  reasoned** exemption fails the suite. 8 gates carry mutants, 15 are exempted with
  written reasons and review dates — including `verify:gates` itself, whose exemption
  states plainly that the gate verifying gates is currently unverified.

  Three defects in the suite's own first run were self-corrected (F23): a mutant that
  falsely accused a working gate by not rebuilding first, a clean-tree check that flagged
  its own untracked files, and a summary that printed `0 survived` when two had.
- **Phase 1 of the audit is now validated by measurement (#579)**: added
  `scripts/audit/render-mutants.mjs`, which rebuilds the workspace between baseline and
  mutant so a render-time analysis can be mutation-tested at all. **Both mutants
  killed, 2/2 = 100%.**

  M1 is the result that matters. Planting the *default* theme's `--gds-support` value
  as a literal produced `default/light +0` while `class-usa +51`, `gold-athlete +45`
  and `high-contrast +90` — the same literal classified `token` under the theme whose
  map contains it and `literal` under the three that do not. A classifier that
  string-matched globally would have raised `default` too; it did not, so per-theme
  provenance resolution is confirmed by measurement rather than by reading the code.

  **#579's own premise had to be corrected.** It specified planting a *radius* equal to
  the default theme's value — but GDS has no `--gds-radius-*` token at all (that is
  #555), so a hardcoded radius is `literal` under every theme, the count rises
  everywhere including `default`, and the discriminating test collapses into M2.
  Retargeted to `--gds-support`, which genuinely differs per preset. The injection site
  also moved: `MapPanel`'s `borderRadius: 12` sits on an iframe that only renders when
  an `iframeSrc` is supplied, so the anchor existed but never rendered.

  Consequently `untraceableRenderRate` is **promoted from advisory to blocking**, and a
  new `renderMutationScore` budget pins M1/M2 at 100% — Phase 1 stays trustworthy only
  while they stay killed. The "Phase 1 is entirely unvalidated" caveats are withdrawn
  from `docs/HEALTH_RETENTION_PLAN.md` and `audit/completeness-critique.md`.

  The overall audit verdict is **unchanged**: 8 of 12 mutants now run, 3 remain
  (M8, M9, M10), and the gate requires 100% on all twelve. Two more passing is progress,
  not a pass.
- **`gds-a11y` JSDoc coverage gate was a false pass (#516)**: the gate reported
  `gds-a11y: 0/0 public exports documented (100.0%)` while the package carried **17
  undocumented exports**. Four parts:
  - **Barrel detection is now by content, not filename.** The walker skipped anything
    named `index.ts`/`client.ts`/`server.ts` as a presumed re-export barrel.
    `packages/gds-a11y/src/index.ts` is that package's entire 368-line implementation
    and was skipped purely because of its name.
  - **All 17 exports documented**, written from the implementation. Notably
    `applyGdsA11ySuppressions`: an **expired suppression does not suppress** — the
    finding stays active and its message is annotated with the expiry date, so a lapsed
    waiver surfaces rather than silently continuing to hide a defect. Previously
    undocumented behaviour.
  - **`0/0 = 100%` can no longer pass.** Every package in the list has public exports,
    so measuring zero means extraction broke, not that the package is empty. It now
    fails loudly. This is the generalisable half: the same vacuous pass would have
    hidden the next package too.
  - **`CONTRIBUTING.md` corrected.** It claimed enforcement by `eslint-plugin-jsdoc`'s
    `require-jsdoc` in `@sovereignsquad/gds-eslint-config`. False on both counts — the
    rule sits behind an `enforceExportedJsdoc` flag only ever set `true` in that
    package's own unit test, and `npm run lint` targets `apps/playground` alone, which
    would not cover the packages the exports live in even if it were wired. The doc now
    names the real mechanism (`verify:api-jsdoc-coverage`) and records the correction.

  Coverage after: gds-core 981/984, gds-admin 85/85, gds-theme 156/156,
  gds-a11y 17/17 — overall 1239/1242 (99.8%).
- **`ChoiceChip` selected state failed WCAG AA (#537)**: the selected chip paired
  `--gds-text-on-inverse` with `--gds-support` — two semantic roles never designed to
  meet. `text.onInverse` is built to sit on `bg.inverse`; it measured **1.89:1 in
  class-usa dark** and 3.837:1 in class-usa light, against a 4.5:1 requirement. The 23
  generic vibe lanes passed only by coincidence, because they run through
  `ensureContrast`; the two hand-authored brand lanes bypass that path entirely.
  Four changes, in the shared packages:
  - `ensureContrast` promoted from private in `vibe-themes.ts` into shared
    `color-math.ts`, plus a new `readableForeground`. Its being private is *why* the
    brand lanes hand-authored instead of deriving.
  - New `--gds-text-on-support`, derived per preset per scheme against `support` itself.
  - `resolveVibeSemanticCssVariables` now treats a hand-authored table as an **override
    layer, not a replacement**. It previously returned the table wholesale, so any role
    it omitted simply vanished for the brand lanes — the structural root cause, and the
    reason a role added in future can no longer silently disappear from those lanes.
  - `assertContrast` in `brand-tokens.ts` now gates the pairing that actually renders,
    in both schemes. Every prior contrast gate checked *designed* pairings, and this
    pairing was never designed at all — which is how a 1.89:1 chip shipped green.

  Verified across **all 25 presets × both schemes: 50 combinations, 0 below 4.5:1**.
  class-usa light 3.837 → 5.12:1, class-usa dark 1.89 → 10.39:1, gold-athlete dark
  2.73 → 7.2:1. Tightest in the system is now `skyline/light` at 4.51:1.
  The fix *introduced* a 4.1:1 failure mid-way (the hand-authored `support` override
  winning over a foreground derived against the derived `support`); roles derived from
  another role are now recomputed after the override layer applies. Only exhaustive
  checking surfaced it.
- **CI action runtime (#575)**: all six workflows pinned `actions/checkout@v4`,
  which declares `using: node20`, so every run emitted a Node 20 deprecation
  annotation — a standing violation of Rule 1, which forbids any deprecation
  warning reaching `main`. Bumped to `@v5` in all seven usages
  (`quality.yml`, `deploy-pages.yml` ×2, `publish-github-packages.yml`,
  `auto-tag-release.yml`, `release-bundles.yml`, `board-sync.yml`).
  `v5` chosen deliberately over `v6`/`v7`: `v5.0.0`'s only change is the
  node20 → node24 runtime move, whereas `v6` changes credential persistence — on
  a path where `auto-tag-release.yml` pushes a tag — and `v7` changes fork-PR
  checkout semantics. A deprecation fix should carry no behavioural delta. Every
  other action was already Node 24-based (`setup-node@v6`, `configure-pages@v5`,
  `upload-pages-artifact@v3`, `deploy-pages@v4`, `upload-artifact@v4`).
- **`scripts/verify-kanban-drag-accessibility-runtime.mjs` (#574)**: the gate's
  synthetic Space keypress carried Windows-only virtual key codes and no
  character payload, so Chrome synthesized no keypress event and never activated
  the `<button>` on macOS. The gate reported `keyboard operability regression`
  against a component that was working correctly, and `npm run verify:release`
  could not complete on a macOS host. Added `text`/`unmodifiedText` while keeping
  the existing key codes, so the event Linux/CI sees is a superset of what it saw
  before. Verified: fails identically on Chrome 151 and Chrome for Testing 149
  before the fix; passes on both after; a negative control dispatching `a`
  instead of Space still fails the gate, so the assertion is not vacuous.
  `KanbanBoard` was never at fault and is untouched.
- **Board taxonomy (#573)**: added `area: map`, `area: imagery`, `area: motion`,
  `area: governance`, `area: i18n`, `area: playground`, and `area: build` to
  `scripts/board-labels.config.mjs`. The last four canonicalize labels that
  already existed in malformed, unaudited form. All 46 open issues now carry
  exactly one status label; `npm run audit:board:strict` reports 0 violations,
  down from 25.
- **`PROJECT_BOARD.md` and `CLAUDE.md` Rule 7**: both stated that no org-level
  Projects v2 board exists. One does, and it is writable from environments whose
  token carries the `project` scope. Both now record the dual-board model with an
  explicit order of authority — the label board is portable and authoritative,
  project 11 is a richer layer used when reachable — plus the
  `Execution Sequence (HVB)` banding and how `status: blocked` maps onto the v2
  columns.
- **`HANDOVER.md` section 5**: runtime-gate triage now names a third outcome
  besides flake and regression — a host mismatch — with the ordered checks that
  distinguish it.

## 6.0.0 - 2026-08-09 — BREAKING: Re-base the class-usa brand lane onto the ClassScout v2 palette (#536)

Owner supplied a full v2 design-system handoff for ClassScout NYC (README,
design reference, and `class-usa-v2-token-spec.md`) and asked GDS to adopt
it. Archived under `brand-requests/class-usa/`. Re-based `class-usa` in
place — no deprecated names, values, or aliases retained.

**Breaking change**: `ClassUsaColorRampName` renames from `'navy' |
'terracotta' | 'sage' | 'cream' | 'slate'` to `'navy' | 'brand' | 'action' |
'trust' | 'cream' | 'slate'` (five ramps → six); the Mantine `colors` keys
rename to match (`classUsaBrand`/`classUsaAction`/`classUsaTrust` replace
`classUsaTerracotta`/`classUsaSage`). Full old-to-new mapping in
`DEPRECATIONS_AND_MIGRATIONS.md`'s new "Brand-lane token renames" section.

- `packages/gds-theme/src/brand-tokens.ts`: six default ramps pasted from
  the spec (the spec's own `action[2]` placeholder replaced with its
  suggested `#e8a87c`); `deriveClassUsaSemanticTokens` rewritten to the
  spec's 30-role semantic table, deriving from ramp anchors
  (`navy[6]`/`navy[9]`/`action[6]`/`brand[5]`/`trust[6]`/`cream[0]`/`slate[6]`)
  where the spec's value matches a ramp step; `Button.defaultProps.color` is
  now `classUsaAction` (was implicit navy via `primaryColor`, which stays
  `classUsaNavy` for chrome only); Button radius 12px; fonts Playfair
  Display / Inter (Bogart/Garet, neither loadable from any font lane,
  removed). The primary-button WCAG gate now checks white against
  `brand.accent` (the color the button actually renders), not
  `brand.primary` — it was silently checking the wrong color after the
  color/font change.
- `packages/gds-theme/src/vibe-themes.ts`: `class-usa`'s `GdsVibeTheme`
  entry and `classUsaSemanticCssVariables` re-pasted from the same spec
  table. `vibe.accent` (the single scheme-invariant field every other
  vibe-level consumer reads) is deliberately anchored to the action ramp
  (`#c24a0a`), not the brand ramp (`#f5793b`): computed white-label
  contrast is 4.91:1 on `#c24a0a` and only 2.73:1 on `#f5793b` — the spec's
  own text says action orange is "the only colour that carries a label" and
  its own contrast table only certifies that one for text use. The full
  light/dark split the spec actually wants lives in the scheme-aware
  `--gds-*` semantic-role tokens instead. Dark canvas moved from a
  navy-dark tint to neutral charcoal (`#14171c`) per the spec's explicit
  ruling — navy is now reserved for accents and the inverse shell only.
- `packages/gds-theme/styles.css`: the generic cross-theme rule that fills
  every flatSurfaces button from `--gds-vibe-primary` (navy) was silently
  overriding `Button.defaultProps.color` for every button on the site —
  found live via CDP: CTAs rendered navy in both schemes despite the
  Mantine-level color change. Added a class-usa-scoped override (Gold
  Athlete untouched) painting the primary/filled button from
  `--gds-vibe-accent` in both light and dark, so CTAs actually read as
  action orange as the spec requires.
- `packages/gds-theme/src/font-lanes.ts`: new `playfair-display` generic
  font lane (display Playfair Display, body Inter).
- Tests: 3 previously-pinned assertions across `brand-tokens.test.ts`,
  `vibe-themes.test.ts`, `GdsProvider.test.tsx`, plus
  `generated-art-engine.test.ts` (a cross-package consumer of the old
  primary hex), updated to the new palette; added coverage for the new
  ramp shape, the accent light/dark split, and the CTA color/radius.

Verified live via CDP (local build, not yet deployed): primary button
background/border/radius on `/patterns/operations` in both class-usa light
(`#c24a0a`, 12px radius) and dark (`#c24a0a`, 12px radius) — before the
`styles.css` fix both rendered navy `#0f2c4a`. Ran a programmatic WCAG
contrast audit across 8 routes × 2 schemes; every finding was either (a) a
false positive from the audit script's own inability to parse `color()`
background values on the Theme Lab's isolated `[data-gds-owned-contrast]`
preview cards, (b) a pre-existing, theme-agnostic Mantine-baseline color
(red/blue/teal/gray) unrelated to class-usa, or (c) one real, pre-existing
`ChoiceChip` selected-state defect (pairs `text.onInverse` with `support`,
two roles never designed to pair) affecting both class-usa and gold-athlete
— confirmed NOT a regression (old class-usa values computed 2.549:1
light / 1.959:1 dark for this same pairing; the new palette computes
3.837:1 / 1.89:1 — light improved, dark essentially unchanged) — filed
separately as issue 537 rather than folded into this change, since the real
fix is a component-level token-pairing decision, not a palette re-base.

Issue: #536 (closed by this commit). Related: #533, #534 (the exact
independently-authored-dark-mode-value discipline this re-base follows),
#537 (filed, not fixed, confirmed not a regression).

Owner asked for GDS to be able to import externally-produced designs (Figma
files, screenshots, AI design-tool output including Claude Design) under
GDS's own limitations, discoverable from the repo. This ships the governed
pathway as documentation and process, not new runtime code — no source or
component behavior changed.

- `THEME_GOVERNANCE.md`: new "Importing an externally-produced design"
  section stating the one-directional rule — a source design may inform a
  new theme lane's intent, but must never be consumed directly as CSS, an
  image, or a copy-pasted color value; every value ships as a re-derived,
  independently WCAG-AA-verified `GdsVibeTheme`/brand-token entry, same as
  every other lane. Cites #533/#534 as the exact failure mode (an unverified
  or reused dark-mode value) this process exists to prevent.
- `CONTRIBUTING.md`: new "Importing an externally-designed theme" section —
  the maintainer-facing, numbered mechanics (file the issue, extract intent
  not values, map into the full `GdsVibeTheme` contract with independently
  designed dark-mode values, verify WCAG AA from real computed styles,
  register and run `verify:release` like any other lane, document and close
  the issue) — matching the style of the existing "Adding a Component or
  Pattern" section.
- `TEMPLATES/GDS_THEME_CREATION_PROMPT.md`: added a new "If you're starting
  from a source design" section covering how to responsibly handle a source
  design as input (extract intent not values, design a genuine dark-mode
  counterpart rather than deriving one mechanically, let accessibility win
  over source fidelity, record provenance) — and updated the report-back
  section to ask for the source's origin and any accessibility deviations.
- `README.md`: added a discoverability link to the theme-creation prompt
  and the new governance sections from the "Use with AI coding agents"
  section, so the import pathway is findable from the repo's front page.

Issue: #535 (closed by this commit).

## 5.0.2 - 2026-08-09 — Fix: Class USA/Gold Athlete status badges illegible in dark mode; 8-route audit; CLAUDE.md Rule 12

Owner reported the 5.0.1 fix as still broken after checking the live site —
correctly, because it had not been pushed yet, and because a second,
separate legibility bug remained: generic (non-fixed-tone) badges — e.g.
"UNSAVED CHANGES"/"SAVED" status pills — used `color-mix(76% primary, 24%
text)` for text on a `color-mix(15% primary, transparent)` background.
That formula works in light mode (primary-heavy text on a light canvas) but
collapses to near-unreadable dark-on-dark for Class USA/Gold Athlete in
dark mode, since `--gds-vibe-primary` is a single fixed hex that doesn't
adapt between schemes, unlike `--gds-vibe-text`/`--gds-vibe-surface`.

`packages/gds-theme/styles.css`: added a dark-mode-scoped override for
`class-usa`/`gold-athlete` using `--gds-vibe-text` (already proven legible)
for badge text and an accent-tinted background/border, matching the same
governed-second-color pattern already used for buttons (#531).

Also ran a systematic, programmatic (WCAG contrast-ratio, not visual
impression) audit across 8 routes (`/`, `/patterns`, `/patterns/operations`,
`/patterns/data`, `/patterns/foundations`, `/themes`, `/live-demos`, `/api`)
in Class USA dark mode per the owner's explicit demand for a real audit,
not a spot check. Found and fixed the badge issue above; found and filed
(did not fix here) a separate, pre-existing, universal issue affecting
`[data-gds-badge-fixed-tone]` badges in dark mode across every theme,
including default — see #534.

Added CLAUDE.md Rule 12: verification claims ("confirmed"/"fixed"/
"legible"/"done") must state exactly what was checked (route, element,
environment — local build vs. deployed) and must not imply broader
coverage than was actually verified; "audit" means an exhaustive sweep,
not a spot check.

## 5.0.1 - 2026-08-09 — Fix: Class USA/Gold Athlete dark mode had illegible navy-on-navy text (#533)

Owner reported the live site: selecting Class USA + dark mode via Theme Lab
rendered "Open section" links, badge text, and some headings in the same
near-black navy as the card background. Traced through the live CSS
cascade: `GdsProvider`'s own wrapper `Box` applied the theme object's
`other.gdsCssVariables` (built by `createBrandTheme`) as an inline style
containing both a token's light value AND its `-dark` variant as separate,
unrelated custom properties (e.g. `--gds-text-body: '#0b223e'` alongside
`--gds-text-body-dark: '#faf7f1'`) — nothing ever picked the dark one, and
since inline styles beat any external stylesheet rule, every element inside
that Box always got the frozen light-mode value regardless of the active
color scheme. Invisible for the default theme (its `other.gdsCssVariables`
doesn't define these semantic-role tokens at all, so CSS's own
`light-dark()` default in styles.css correctly took over); broke every
brand theme with a hand-authored dark variant — Class USA and Gold Athlete.

`packages/gds-theme/src/GdsProvider.tsx`: extracted the wrapper into a new
`GdsThemeVariablesScope` component that resolves each `{base, base-dark}`
pair against the live, reactive scheme via Mantine's own
`useComputedColorScheme()` (the same hook `ThemeToggle` already uses)
before applying it as the inline style, instead of dumping the raw
theme-object values unconditionally. Light mode is unaffected. Verified
live (computed styles + screenshots, before/after, both presets, both
schemes) and with two new unit tests in `GdsProvider.test.tsx`.

## 5.0.0 - 2026-08-09 — BREAKING: `ReferenceThemeExplorer` moved behind a dedicated subpath (#532)

Owner asked to actually fix the `vendor-gds` bundle-size overage from #532
rather than just re-baseline the ceiling — this is the real fix for the
larger of the two root causes.

`ReferenceThemeExplorer` was gds-core's single largest client-bundle module
(`ReferenceThemeExplorer.tsx` + `.copy.ts`, ~112.7KB, bigger than the
`GdsRichTextEditor` subtree that established this pattern), and every real
consumer renders it on one or two specific routes, not universally. Moved
it out of the main `.`/`./client` barrels into a dedicated
`@sovereignsquad/gds-core/reference-theme-explorer` subpath, mirroring
`rich-text-editor`'s proven split (561KB→217KB for `reference-vite`, which
never imports it). **This is a breaking change**: `import {
ReferenceThemeExplorer } from '@sovereignsquad/gds-core'` (or `./client`)
no longer resolves it — see `DEPRECATIONS_AND_MIGRATIONS.md`'s new
"Component-export relocations" section for the one-line migration.

The playground renders `ReferenceThemeExplorer` on first paint of both `/`
and `/themes`, so it was kept as a static, eager import (not lazy — that
would add a loading-flash waterfall to the site's own landing page), just
from the new subpath, with a dedicated higher-priority chunking rule in
`vite.config.ts` so it lands in its own `vendor-gds-theme-explorer` chunk
instead of fused into `vendor-gds`. Net effect on the playground's own
build: `vendor-gds` dropped from 954KB to 665KB — comfortably under both
the original 940KB ceiling and the 960KB one from the prior release,
without raising anything further. Verified live: `ReferenceThemeExplorer`
still renders correctly on both routes after the split.

The other root cause identified in #532 (gds-core's 12 locale dictionaries,
122.8KB, always bundled eagerly) is unresolved — it needs an async/Suspense
redesign of `GdsI18nRuntime`'s currently-synchronous message-lookup API,
which is a larger, separate piece of work, not folded into this release.

## 4.1.15 - 2026-08-09 — Audited the vendor-gds bundle-size warning; re-baselined ceiling, filed real fixes (#532)

Owner asked for the `vendor-gds` chunk-size warning to be genuinely fixed —
"clean out the trash, the dead ends" — not silenced. Investigated before
touching anything: built the last committed commit (`d627f83`) in an
isolated worktree and confirmed the 954KB chunk was already 953.55KB there,
over the `chunkSizeWarningLimit: 940` ceiling, *before* this session's work
(this session's own ~85-file diff added 545 bytes total). Scanned every
non-test source file in `gds-core`, `gds-theme`, `gds-admin` (200 files) for
dead code: found none — every export is either imported in-repo or
re-exported from a public barrel. The size is legitimately-earned showcase
surface, not neglect.

Re-baselined `apps/playground/vite.config.ts`'s `chunkSizeWarningLimit`
940→960 (same documented pattern as its two prior re-baselines) to reflect
that audited-clean reality. Filed #532 for the two real fixes the
investigation surfaced, both genuine architecture changes rather than
same-day patches: (1) subpath-extracting `ReferenceThemeExplorer` (112.7KB),
`GdsSchemaForm`'s demo subtree (32.1KB), and `KanbanBoard`+`AdvancedDataTable`'s
demo subtree (25.8KB) behind dedicated entry points, mirroring
`rich-text-editor`'s proven ~184KB combined savings — but all three are
currently exported from the main barrel, so this is a breaking change
needing a major version; (2) lazy-loading gds-core's 12 locale message
dictionaries (122.8KB, only 1 ever active per visitor) instead of eager
bundling, which requires reworking `GdsI18nRuntime`'s currently-synchronous
message-lookup API — also not a one-line fix.

## 4.1.14 - 2026-08-09 — Bolder navy/orange brand presence on flatSurfaces buttons in light mode (#531)

Owner directive: after confirming Class USA's navy-filled buttons are
intentional, governed brand behavior (`createBrandTheme('class-usa')` sets
`primaryColor: 'classUsaNavy'`), the owner asked for both of a flatSurfaces
brand's colors (primary + accent) to read as more dominant on buttons in
light mode — the prior treatment was "too faded... that was never the
intention." `packages/gds-theme/styles.css`, scoped to
`html[data-mantine-color-scheme='light']` + `[data-gds-theme-preset='class-usa']`/
`='gold-athlete']`: filled/primary buttons gain a solid 2px accent-colored
border (Class USA's terracotta, Gold Athlete's metallic gold) alongside
their existing primary fill, and default/secondary buttons switch from the
faint neutral-bordered/7%-tinted treatment to a full-strength primary
border and text color. A first pass of this fix wasn't scoped to light
mode and was caught, before shipping, regressing dark-mode default-button
contrast to ~1.12:1 (`--gds-vibe-canvas`/`--gds-vibe-primary` are both
near-black in dark mode) — rescoped to light mode only; dark mode keeps its
pre-existing, already-verified contrast pairing.

## 4.1.13 - 2026-08-09 — Full-site "not canonical" audit batch (#530)

Owner demanded an exhaustive whole-site audit — every bug, not just colors,
and every component the reference site is supposed to represent. Ran 6
parallel deep-read audits (site shell/nav, `/patterns` content components,
the entire `styles.css`, a hardcoded-style sweep across `gds-core`,
`/live-demos/*`, and a fresh look at an earlier unresolved report) plus
live browser verification of the strongest theoretical lead, which was
ruled out (an unblurred `text-shadow` on MetricCard numerals — no visible
doubling in either color scheme).

**High** — `high-contrast`'s own doc comment says "No gradients or glows,"
but it never set `flatSurfaces: true` like `class-usa`/`gold-athlete`, so
the accessibility preset was hit by the same fabricated-gradient bug #527
fixed for those two. Fixed at the source (`vibe-themes.ts`) and folded into
the shared override blocks; also fixed a specificity/source-order bug where
`high-contrast`'s own old `.mantine-AppShell-main` override was dead code.

**Medium** — `ReferenceSection`/`SectionPanel` nested eyebrow/description/
link *inside* the `<h3>` (invalid HTML, live on `/patterns`) — gave
`SectionPanel` a real `eyebrow` slot and routed through its existing
`description`/`action` slots instead. `ReferenceLinkGrid` used a different
breakpoint (`xl`) than every sibling grid (`lg`), causing a visible
column-count mismatch on `/patterns` itself. A shared `feedback` state in
`showcase-pages.tsx` let clicking "Delete" flash an error state on the
separate "Submit" button; split into independent state. Replaced raw
`<br/>` spacing hacks with governed `GdsStack`/`GdsCluster`, and two empty
`<div/>` media placeholders (collapsed to 0×0) with real
`GdsGeneratedThumbnail`s. A locale validity check in `App.tsx` was
always-truthy, so a malformed `?locale=` param corrupted state instead of
falling back cleanly. The mobile-nav toggle's `aria-label` was hardcoded
English despite 9 supported locales; localized, and fixed a latent
duplicate-burger bug via a new `hideMobileNavigationToggle` prop on
`DiscoveryShell`. Two bare `white`/`#ffffff` literals in `styles.css`
replaced with `var(--mantine-color-white)`. Two hand-restated `rgba()`
literals in `ReferenceThemeExplorer.tsx` (that would silently drift from
the real theme values) replaced with `color-mix()`.

**Low/latent** — `GdsBreadcrumbs` key-collision risk fixed; a dead
`data-sticky-sidebar` attribute with zero consumers removed; `FeatureBand`
`aria-hidden` inconsistency and untranslated default copy fixed.

**Explicitly investigated and not changed**: `FeatureBand`'s `999px` pill
radius is an established codebase convention, not an inconsistency;
`ReferenceLinkGrid`'s Anchor color renders with full contrast despite an
open question about which CSS rule wins; the text-shadow "doubling" theory
did not reproduce live.

A handful of lower-severity hardcoded-value findings (`MapPanel.tsx`,
`MediaCard.tsx`, `GdsPageTemplates.tsx`, `GdsFormControls.tsx`,
`BottomTabBar.tsx`, `GdsGeneratedThumbnail.tsx`, `MediaWithFallback.tsx`,
`EditorialHero.tsx`, and a repo-wide eyebrow letter-spacing inconsistency)
were deliberately deferred to a follow-up rather than expanding this batch
further — none are visible defects that were reported.

## 4.1.12 - 2026-08-08 — Font-lane loading now uses the governed non-blocking system (#529)

Deep audit prompted by a user report of dim/hard-to-read text on `/patterns`
(that specific symptom could not be reproduced against the shipped build —
byte-for-byte identical to the live deployment, clean contrast in a real
headless-browser repro in both color schemes). The audit surfaced a real,
unrelated "not canonical" defect instead: `packages/gds-theme/styles.css` had
a single blocking `@import` pulling **all 10 built-in font lanes**
(`font-lanes.ts`) — ~46 font-file variants across Barlow, DM Sans, Instrument
Serif, Inter, Manrope, Nunito, Plus Jakarta Sans, Source Serif 4, Space
Grotesk, and Work Sans — on every page load, even though only **one** lane is
ever active at a time (`useGdsThemePresetState`'s `fontLane` selection).

`font-lanes.ts` already had a fully-designed governed system for this — each
lane declares its own `cssImportUrl` and `loadStrategy:
'non-blocking-stylesheet'`, and `getGdsFontLaneStylesheetUrls()` exists
specifically for "governed non-blocking loading" — none of it was wired up.
`styles.css` instead hand-duplicated the font URLs and loaded every lane
unconditionally and render-blockingly, dead code sitting next to the real
mechanism. On a slow/cellular connection this produces a pronounced flash of
fallback-styled text while dozens of unused font files are still in flight
ahead of the one actually needed.

Fixed: `styles.css` now statically loads only the default `'inter'` lane.
`theme-runtime.ts`'s `applyDocumentRuntime` (called by
`useGdsThemePresetState`) manages a single non-blocking
`<link id="gds-font-lane-stylesheet">` for the active lane, added/swapped/
removed as the lane changes, using each lane's own governed `cssImportUrl`.
Test coverage added to `GdsProvider.test.tsx`; `THEME_GOVERNANCE.md` updated
to document the loading contract explicitly.

## 4.1.11 - 2026-08-08 — #523 was incomplete (#527); EditorialCard dark-mode fallback (#526)

User report, direct inspection of the live site: gradient backgrounds and a
mismatched header/"menu" bar under a flat-surface theme, plus a stark white
box inside dark-mode cards, plus general readability complaints.

**#527 — #523 only fixed part of the fabricated-gradient problem.** It
patched `body::before` and the site-wide Button/Checkbox rules, but missed
six other rules in `packages/gds-theme/styles.css` that paint the same
kind of decorative gradient/glow unconditionally on every preset, including
Class USA and Gold Athlete (`flatSurfaces: true`, no gradient/glow/colored
shadow anywhere in their real brand definition):
`body` itself (the base rule layers a second, un-gated diagonal wash under
the already-neutralized `--gds-vibe-gradient`; invisible in normal layout
since `.mantine-AppShell-main` sits on top, but exposed by iOS Safari's
elastic overscroll bounce, which reveals whatever is painted behind the
viewport — the likely source of the "gradient background" seen live on a
mobile device), `.mantine-AppShell-main` (+dark variant),
`.mantine-AppShell-header`/`.navbar`/`.footer` (+the navbar-specific extra
layer), `.gds-paper`/`.gds-card` (+dark variant), and the dark-mode
`.mantine-Popover-dropdown`. Added explicit `class-usa`/`gold-athlete`
override rules for each, mirroring the exact pattern already used elsewhere
in this file for the 12 "enhanced atmosphere" vibe presets — solid
backgrounds, no box-shadow, `!important` to win regardless of specificity
(the `body` override also pins `background-attachment: fixed !important`
explicitly, since the `background` shorthand otherwise silently resets it
to `scroll`). Verified live: computed styles on the header/main/card/body
under Class USA report `backgroundImage: none` in both light and dark mode,
not just visually inspected. Audited every remaining `gradient(...)` rule
in the file referencing `--gds-vibe-primary/accent/shell/canvas/surface`
against the full `flatSurfaces` preset list (`class-usa`, `gold-athlete` —
confirmed via `theme-presets.ts`/`vibe-themes.ts`, no others exist) to
confirm no further instances remain.

**#526 — `EditorialMediaFallback`** (`packages/gds-core/src/EditorialCard.tsx`)
hardcoded `background: 'var(--mantine-color-gray-0)'` — a fixed shade that
doesn't invert with color scheme, unlike `tonePalette.muted.background` a
few lines above in the same file, which already used `light-dark(...)`
correctly. Fixed to `light-dark(var(--mantine-color-gray-0),
var(--mantine-color-dark-6))`, matching that existing idiom, plus an
explicit `color: var(--mantine-color-dimmed)` for the fallback icon.

Both verified against the live deployment cross-checked byte-for-byte
against the local build under test (content-hashed CSS/JS filenames matched
exactly) before concluding the bugs were genuinely shipped, not stale-cache
artifacts — headless Chrome cannot reach the public internet from this
sandbox (confirmed via `ERR_CONNECTION_RESET`), so verification used `curl`
plus direct computed-style inspection against a locally-served identical
build instead of a live screenshot.

**Note on `verify:release`:** the full chain (build/lint/541 tests/boundary/
forced-colors/theme-trust/kanban-a11y/mantine-compat/component-catalog) is
clean for this change. One unrelated Vite chunk-size warning
(`vendor-gds-*.js` ~953 KB, over the 940 KB threshold) is present in the
`apps/playground` build output; confirmed via a clean worktree build of the
already-shipped `main` commit (`432293e`) that it predates this change and
isn't caused by it. Tracked separately as #528 rather than bundled into this
unrelated fix.

## 4.1.10 - 2026-08-08 — Badge glyph mode: emoji as an alternative to Tabler icons (#525)

Client feedback: the badge system should support emoji as an alternative to
Tabler icons, for different purposes (a more playful surface, say, versus a
formal one). Two hard requirements from the client, refined across several
rounds of clarification with a real reference screenshot (a sports-activity
map using `GdsMapPinBadge`): switching to emoji is a whole-badge-system
mode, not a one-badge-at-a-time opt-in, and a category with no emoji falls
back to its Tabler icon automatically — never a visible gap. The mode must
never reach `GdsGeneratedThumbnail`/`GdsGeneratedHero`, which always keep
composing from Tabler icons.

- Added `GdsIconStyleContext`/`useGdsBadgeIconStyle` (`packages/gds-theme`):
  the ambient badge glyph mode (`'tabler'` default, `'emoji'` opt-in),
  mirroring the existing `GdsI18nContext` precedent. `GdsProvider` gained
  `defaultBadgeIconStyle` to set it app-wide.
- `GdsBadge` and `GdsMapPinBadge` (`packages/gds-core`) both gained
  `emoji`/`iconStyle` props. The failsafe is a plain data-presence check —
  no `emoji` on a badge/pin means it keeps its Tabler `icon` even when the
  mode is `'emoji'` — deliberately not a runtime "does this device render
  this glyph" probe, which would be unreliable across browsers, untestable
  in CI without flaking, and SSR-hostile.
- Emoji renders on a fixed dark-neutral surface, never the badge's own
  accent/tone color — emoji are OS-rendered color glyphs whose color can't
  be forced via CSS the way a Tabler `currentColor` stroke icon can, so
  contrast against an arbitrary accent can't be guaranteed the same way.
  `GdsBadge` renders a small `var(--mantine-color-dark-7)` coin behind the
  glyph; `GdsMapPinBadge` fills the whole pin with that color while the
  ring/silhouette keeps `accent` — modeled directly on the client's
  reference screenshot. `filled`/`shape` are ignored (with a dev-mode
  warning) while emoji is active, since neither composes with it in v1.
- New `packages/gds-core/src/category-registry.ts`: `GdsCategoryDefinition`
  (`key`/`label`/`accent`/optional `shade`/required `icon`/optional
  `emoji`) and a `resolveGdsCategoryBadgeIcon` resolver. No business
  taxonomy shipped — categories are a consumer's own domain vocabulary, per
  the same reasoning `GdsMapPinBadge`'s existing icon docs already give.
  `icon` required + `emoji` optional is the structural guarantee behind
  "emoji affects only the badge": the generated-imagery components have no
  code path that reads `emoji` at all.
- Playground: `/patterns/feedback` gained a live Tabler/emoji toggle
  (`SportsEmojiModeDemo`) — a Soccer/Basketball/Baseball category set as
  badges and map pins in both modes, beside `GdsGeneratedThumbnail`s for
  the same categories that never change. `verify-forced-colors-runtime.mjs`
  gained a required-component case for the emoji glyph disc.
- Docs: `BADGE_SYSTEM.md` new "Badge glyph mode" section;
  `GENERATED_IMAGERY.md` cross-reference explaining why that system never
  reads emoji.

## 4.1.9 - 2026-08-08 — Fabricated brand gradients/glows, site-wide, for governed flat-surface themes (#523)

User feedback after 4.1.8/#522: the vibe-gallery card's identity dot and
swatch box were still showing a blended gradient, and investigating why led
to a much bigger finding than #522 actually fixed.

#522 only fixed the fabricated gradient/glow where it was reported — the
Theme Lab's owned-contrast preview surfaces. The same fabrication also
exists, unfixed, in places that affect the **entire live site**, not a demo
card, whenever Class USA or Gold Athlete is the actually-selected theme:
`styles.css`'s site-wide (not owned-contrast-scoped) `.mantine-Button-root`
and `.mantine-Checkbox-input:checked` rules painted a
`primary->accent` gradient on every real button/checkbox on every page; the
real page `body` background and its `::before` atmospheric wash apply a
brand-colored radial-gradient unconditionally for any active theme; several
`box-shadow` rules use a brand-colored glow. None of this is what the real,
governed brand specifies — `createBrandTheme('class-usa')` and
`createBrandTheme('gold-athlete')` (`packages/gds-theme/src/brand-tokens.ts`)
both default `flatSurfaces: true` and never configure a gradient, glow, or
colored shadow anywhere.

- Added `flatSurfaces?: boolean` to `GdsVibeTheme` (`vibe-themes.ts`), set
  `true` on the `class-usa`/`gold-athlete` entries — a real, data-level fact
  about the system, not a component special-casing an id string.
- `getGdsVibeThemeCssVariables` now neutralizes `--gds-vibe-glow` (→
  `transparent`), `--gds-vibe-gradient` (→ `none`), and publishes a new
  `--gds-vibe-atmosphere` (→ `0`) for `flatSurfaces` lanes — one source of
  truth that every existing CSS consumer already reads, so the fix
  propagates everywhere automatically instead of needing a rule-by-rule
  patch across the stylesheet.
- `body::before`'s atmospheric wash now respects `--gds-vibe-atmosphere`
  (defaults to `1` for lanes that don't set it, so the other 23 vibe lanes
  are unaffected).
- The site-wide Button/Checkbox rules now use a solid `var(--gds-vibe-primary)`
  fill for all 25 presets, not just the two flat lanes — none of the 25
  shipped theme presets ever actually configure a gradient-variant control,
  so a gradient there was never correct for any of them.
- `ReferenceThemeExplorer.tsx`'s vibe-gallery card, vibe-contract panel, and
  Athlete Gold reference mockup: swatch/identity-dot for `flatSurfaces`
  lanes now render as a hard-edge two-color split (both real solid colors,
  zero invented blend color) instead of a smooth gradient; backgrounds and
  glow shadows are flat for those two lanes.

## 4.1.8 - 2026-08-08 — Theme Lab vibe-gallery preview: real colors, no fabricated gradients (#522)

Two rounds of user-reported live bugs on `/themes`, both about the vibe-
gallery card previewing colors/styling that don't match what a theme
actually produces:

1. The Class USA vibe-gallery card's color-preview swatch didn't visually
   read as connected to the lane's real colors, despite the underlying
   values being correct (verified by pulling the deployed JS bundle:
   `#ff6b35` present, the old `#ca8570` fully gone). Root cause:
   `packages/gds-core/src/ReferenceThemeExplorer.tsx`'s swatch box rendered
   `vibe.hero` — a `linear-gradient` tuned to ~12-16% opacity for use as a
   background wash behind the vibe-contract panel's own grid of solid
   swatches further down the page. Reused as the *only* color preview on
   the compact gallery card, that same low-opacity gradient reads as an
   indistinct pale blob rather than the lane's real colors, for every one
   of the 25 vibe presets, not just Class USA. Fixed by rendering the swatch
   at full strength — `linear-gradient(135deg, ${vibe.primary},
   ${vibe.accent})`, an honest two-color brand-identity preview, matching
   the same recipe already used by the small identity dot beside the lane
   name. `vibe.hero`'s other use (the vibe-contract panel's own background,
   sitting *behind* other solid swatches rather than acting as one) was
   left untouched — appropriately subtle there.
2. Investigating (1) surfaced a second, more serious problem: the "Preview
   this vibe" button itself (and every other filled-style button rendered
   inside a vibe-gallery/vibe-contract/athlete-gold-reference preview, via
   a shared `styles.css` owned-contrast rule) was painted with a
   `linear-gradient(135deg, var(--gds-vibe-primary), var(--gds-vibe-accent))`
   — a gradient the underlying governed theme never actually specifies. A
   real `createBrandTheme('class-usa')` Button, for example, is a single
   solid navy fill; nothing in its definition produces a gradient. The
   Theme Lab was misrepresenting what a button looks like under a theme
   that already has its own solid-color rule, across all 25 lanes. Fixed by
   changing that `styles.css` rule to a solid `var(--gds-vibe-primary)`
   fill — matching what a real filled button on that theme actually
   renders, everywhere the rule applies, not just Class USA.

## 4.1.7 - 2026-08-08 — Class USA accent/slate/info color refinement from Figma prototype (#521)

Refined three of the Class USA brand theme's tokens against a ClassScout
mobile-app Figma prototype (fileKey `DHb3LghHT02dtcKlTsL3cX`), extracted
empirically via the Figma MCP (`get_design_context`/`get_screenshot` on the
Home/Search/Bookings/Saved/Profile screens — the file defines no Figma
variables, so values were read from raw fills/text colors).

- **Accent (terracotta ramp)**: re-anchored from the muted `#ca8570` to the
  prototype's vivid coral `#ff6b35` (repeated on every CTA, badge,
  notification dot, and the FAB). The interactive/`--gds-brand-accent-action`
  step moved from `#a85a44` to `#d63900` — both ramp endpoints were rebuilt
  in HSL space around the new hue so the AA-safe interactive step (4.72:1
  white-on-fill, was 4.99:1) sits where the theme's own `assertContrast`
  gate already expects it.
- **Slate (secondary/meta text)**: re-anchored from `#434c59` (8.13:1 on
  cream) toward the prototype's visibly lighter, cooler blue-grey. The
  prototype's literal pixel value (`#6b7897`, 4.13:1 on cream) fails the
  theme's own 4.5:1 AA gate, so the new anchor (`#5e6a86`, 5.06:1 on cream /
  5.41:1 on white) keeps the same hue/saturation direction while staying
  the lightest tone that still clears AA.
- **New `state.info` role**: `#1d6fa5` light / `#51a8e1` dark, sourced from
  the prototype's "Verified" badge — this role previously just reused navy.
- Navy, sage, and cream were re-checked against the same prototype and left
  unchanged (navy: `#0d2340` measured vs. `#0b223e` shipped, a sub-1%
  difference within rendering noise; sage: no instance observed in the
  sampled screens; cream: no direct pixel sample, visually consistent).
- Updated in lockstep: `packages/gds-theme/src/brand-tokens.ts` (the public
  `createBrandTheme('class-usa')` ramps/derivation) and
  `packages/gds-theme/src/vibe-themes.ts` (the hand-authored `class-usa`
  vibe entry + `classUsaSemanticCssVariables`, which mirror
  `brand-tokens.ts`'s output by existing test contract — both were updated
  together so they don't drift). `tokens/gds.tokens.json` regenerated.
- Explicitly **not** changed, and documented as out of scope in issue #521:
  typography (the prototype uses Plus Jakarta Sans/DM Serif Display, but
  `brand-requests/class-usa/ClassScout-Design-Tokens-and-Components.md` §2
  locks Bogart/Garet as "settled... this doc is the authority" — a font
  swap needs an explicit brand decision, not a code-side guess), the
  prototype's ~7-hue pastel category-icon tint system (a new token-role
  surface, not a refinement of existing roles), and its elevation/shadow
  treatment (the theme still defaults to `flatSurfaces: true`).
- Added `brand-requests/class-usa/ClassScout-Design-Tokens-Refinement-2026-08-08.md`
  recording this as a dated refinement layered on top of the locked v1.0
  token doc, which stays as the historical record of what was originally
  reconciled from the brand PDFs.

## 4.1.6 - 2026-08-08 — 3.9.0→current migration note; fix self-contradictory major-version index (#514)

Consumer-facing docs fix, prompted by real client feedback from four
sibling apps (camera, messmass, fanmass, launchmass) still pinned to
`3.9.0` and planning their upgrade. Investigating that feedback confirmed
the underlying "nothing's been published since 3.9.0" claim was stale (the
registry publish history was verified clean — see #515's own investigation
for the evidence), but surfaced a real, separate gap: no single place told
a 3.9.0 consumer what would actually look or behave differently after
upgrading.

- **`INSTALLATION_GUIDE.md`**: added a "Behavioral changes to budget for
  between 3.9.0 and the current line" subsection under the existing
  npmjs-migration section, consolidating the four changes with real
  user-visible or action-required impact that were previously only
  documented individually, scattered across their own `CHANGELOG.md`
  entries: the mobile input-focus auto-zoom guard (3.11.0, `xs`/`sm`/
  default-size inputs render measurably larger text with no code change),
  `GdsPageTemplateAction.pending`→`loading` (3.13.0, backward-compatible
  alias), `gds-theme`'s date-component stylesheet becoming opt-in (3.14.0,
  requires adding `import '@sovereignsquad/gds-theme/dates.css'` for
  consumers rendering date components), and a clarification that the
  `4.0.0` major bump was a release-process artifact (a published `3.15.0`
  being immutable forced the jump), not an intentional breaking API change.
- **`docs/DOCUMENTATION_VERSIONING.md`**: fixed a self-contradictory version
  index — it said "Current major — 3.x" while citing tag `gds-v4.1.5` (major
  4) in the same line, and listed "Previous major — 2.x" as the archived
  lane even though a real major bump to `4.0.0` shipped 2026-08-07. Now
  correctly reads "Current major — 4.x" / "Previous major — 3.x (last
  release `gds-v3.14.17`, confirmed against the actual tag list)".

## 4.1.5 - 2026-08-08 — Fix clipped select/input text, mobile side-margin waste in the shared section/card primitives, and verify:release preview-server flakiness (#513, #515)

User-reported live bug on `/themes` at a narrow (390px) mobile viewport:
the "Preset" dropdown's selected text rendered with its lower half clipped,
and large side margins around the Theme Lab cards left little room for the
actual controls. Root-caused and fixed at the shared-component level, not
with a page-local CSS patch, so both fixes apply everywhere the underlying
primitives are used.

- **Clipped select/input text.** `packages/gds-theme/styles.css`'s #510
  dark-mode-gap fix added `padding: 0.5rem ...` directly to
  `.mantine-Input-input` / `.mantine-NativeSelect-input` /
  `.mantine-Textarea-input` — but Mantine's own single-line inputs
  vertically center text purely via a fixed `--input-height` and
  `--input-line-height: var(--input-height) - 2px`, with **zero** vertical
  padding of its own (all Mantine input padding is horizontal-only).
  Stacking an extra 8px top/bottom padding on top of that fixed-height box
  shrank the usable content area below the 34px line-height, clipping text.
  Split the rule: Mantine-native classes now get color/background/border
  only; the explicit padding/radius stays on the raw/bare-native-element
  fallback selectors, which have none of Mantine's own sizing and genuinely
  need them. Confirmed live: computed `padding` went from `8px 12px` (text
  clipped) to `0px 34px 0px 12px` (horizontal-only, text fully legible),
  with the line-height unchanged.
- **Wasted mobile side margins.** `SectionPanel` (`packages/gds-core/src/SectionPanel.tsx`)
  — the shared, governed wrapper behind `ReferenceSection` and used across
  dashboards, detail pages, and settings surfaces system-wide — wrapped its
  content in a `Paper` with a flat `p="lg"`. `ReferenceThemeExplorer`
  (`packages/gds-core/src/ReferenceThemeExplorer.tsx`) then nested further
  `Paper` cards with their own flat `p="lg"`/`p="md"` directly inside it,
  stacking two (sometimes three) levels of full desktop-sized card padding
  on a 390px viewport. Measured: the "Theme preset" card's content box was
  252px wide out of 390px (64.6%) before the fix. `SectionPanel`'s padding
  is now responsive (`p={{ base: 'xs', sm: 'sm', md: 'lg' }}`); all 12
  `Paper` instances in `ReferenceThemeExplorer.tsx` (the Theme Lab grid,
  live-preview surfaces, vibe gallery, vibe-contract swatches, proof cards,
  unsupported-pattern cards) got the same treatment, so the whole page is
  fixed consistently, not just the one card in the bug report. `radius`
  drops one token (`xl`→`lg`, `lg`→`md`) rather than going responsive —
  Mantine's `radius` prop is a single `MantineRadius`, not the
  breakpoint-object `StyleProp` spacing props like `p` accept. Measured
  after: 272px of 390px (69.7%). No hardcoded pixel values: every value is
  a Mantine spacing/radius token, using the same responsive style-prop
  pattern already established by `GdsContainer`'s own default padding.
  Verified no desktop (1440px) regression via screenshot.
- **`verify:release` runtime-gate flakiness fixed at its source (#515).**
  `scripts/lib/browser-runtime.mjs`'s `startPreviewServer()` spawned the
  playground preview server via `npm run preview`, which runs `vite` as a
  grandchild process; each `verify-*-runtime.mjs` script cleaned it up with
  an un-awaited `previewServer?.kill('SIGTERM')` on only the `npm` PID,
  which does not reliably reach that grandchild. Confirmed live: a `vite
  preview` process from a completed script was still alive and bound to
  port 4173 several minutes later, letting a *later* script's own preview
  server race against the stale one — producing intermittent failures on a
  shifting set of unrelated routes/themes/viewports across consecutive
  runs (never the same case twice, unlike a real content bug). Fixed by
  spawning detached and killing the whole process group
  (`process.kill(-pid, signal)`, escalating to `SIGKILL` after a 3s grace
  period), awaiting the actual `exit` event before returning — mirroring
  `disposeBrowser`'s existing wait-for-real-exit approach for the Chrome
  side of the same file. All 5 call sites now `await` the kill.
  `scripts/verify-theme-trust-runtime.mjs` also gets a scoped timeout
  increase (`waitForReady` 12s→25s, inter-retry wait 600ms→2000ms): it's
  the last of five Chrome-launching runtime gates in the full `verify:release`
  chain, run only after two full workspace builds, lint, and the test suite,
  and two independent single-browser-per-run architectures for it were
  tried and reverted first — recycling the Chrome session every 4 cases,
  then every case — both made failures *worse* (up to 13 of 22 cases,
  spread across unrelated routes), confirming rapid browser process churn
  is itself disruptive in this environment rather than a fix. The original
  architecture passed reliably in isolation (22/22, twice); more time
  margin at the point in the chain where cumulative load is highest was the
  change that actually held.

## 4.1.4 - 2026-08-08 — Badge icon composition docs: scope the closed-vocabulary rule to `icon`, document `GdsBadgeStack`'s open escape hatch (#497)

Docs-only fix. `BADGE_SYSTEM.md`'s "Canonical icons in badges" bullet and
`COMPONENTS_AND_PATTERNS.md`'s badge rule row both stated, without
qualification, that badge icons always render through the governed
`GdsIcons` dictionary — true for `StatusBadge`/`MeaningBadge`/`GdsBadge`'s
closed `icon` prop, but silent about `GdsBadgeStackLayer`
(`packages/gds-core/src/GdsBadgeStack.tsx`), whose `children: ReactNode` is
untyped by design (`GdsIcon` withholds the `className`/`style`/`ref`
composition surface a layering primitive needs), leaving no documented path
for icons `GdsIcons` has no entry for (sports/hobbies/interest categories,
the same gap `GdsMapPinBadge`'s own docs already cover for its `icon` prop).

- **`docs/BADGE_SYSTEM.md`**: rescoped the "Canonical icons in badges"
  bullet to `GdsBadge`/`StatusBadge`/`MeaningBadge`'s `icon` prop
  specifically, and added a new "Composing icons `GdsIcons` doesn't have"
  section documenting `GdsBadgeStack`/`GdsBadgeStackLayer` as the sanctioned
  composition path — including `GdsBadgeStack`'s whole-stack `label` →
  `role="img"` accessibility contract, and the real `gds-compliance`
  `package-coverage-gap` exception schema
  (`packages/gds-compliance/index.js`'s `EXCEPTION_REQUIRED_FIELDS` plus the
  base `surface`/`reason`/`owner`/`reviewDate` fields every approved
  exception requires) a consumer declares in `gds-adoption.json` to compose
  an external `@tabler/icons-react` icon into a layer.
- **`COMPONENTS_AND_PATTERNS.md`**: mirrored the same scoping and pointer in
  the badges row of the core component contract table.

## 4.1.3 - 2026-08-07 — PageHeader title/action overlap and ActionBar wrapped-row alignment (#511)

Found via user-reported screenshots on a narrow mobile viewport (title text
overlapping its own action button; two related buttons landing on visibly
misaligned rows). Confirmed via live rect measurement and screenshots
before and after the fix, same as #509/#510.

- **`PageHeader` title overlapped its action button.**
  `packages/gds-admin/src/PageHeader.tsx`'s title `Box` had `minWidth: 0`
  inside a `flex: 1` row beside the actions `Group` — a zero-basis flex
  item can always "fit" by shrinking, so `wrap="wrap"` never triggered;
  measured the title box shrunk to 86px while its text needed 217px, and
  with no `overflow-wrap` set, the overflow rendered straight through the
  Save button beside it. Fixed by removing `minWidth: 0` entirely rather
  than replacing it with an invented pixel/rem threshold: the platform's
  own flex default (`min-width: auto`, an item won't shrink past its
  content's natural minimum) is exactly what forces the row to wrap once
  both can't fit — confirmed via measurement that the title now renders at
  its full natural width with zero overflow and no mid-word breaking, no
  magic number required. `overflowWrap: 'break-word'` stays on the title
  as a second line of defense for a single word wider than the row itself.
- **`ActionBar` rows misaligned once wrapped.**
  `packages/gds-core/src/ActionBar.tsx`'s primary/icon-only group carried
  `marginInlineStart: 'auto'` — redundant on one row (the outer
  `justify="space-between"` already handles that split) but forcing a
  right-hugging second row under a left-aligned first row once the outer
  group wrapped on mobile. Removed; confirmed `Cancel`/`Save` now share the
  same `left` position when stacked.
- **Token hygiene pass on #510's own CSS**, prompted by the same review:
  `packages/gds-theme/styles.css`'s native-input rule referenced a literal
  `0.75rem` for horizontal padding where a real Mantine spacing token
  (`--mantine-spacing-sm`, confirmed equal to `0.75rem` by live
  measurement) applies directly — swapped to the token reference. Its
  `border-radius: var(--mantine-radius-sm, 4px)` fallback was already
  correctly token-first and matches this file's existing convention for
  defensive fallbacks, so left as-is.
- Investigated, not changed: `BoundedPreviewSurface`'s `26rem`–`32rem`
  `minHeight` (flagged as "wasted space" on the `DiscoveryShell` demo) is a
  deliberate convention used identically across 8 different playground
  demos, not an isolated mistake — left for a product decision rather than
  an isolated edit.
- All 522 existing tests pass unchanged.

## 4.1.2 - 2026-08-07 — Dark-mode theming gaps: date-picker popover, segmented control, checkboxes, native inputs (#510)

Found via user-reported screenshots of the live playground in dark mode and
confirmed via live `getComputedStyle` inspection against the built site
before any fix landed — not assumed from the screenshots alone.

- **`GdsSegmentedControl` active-indicator contrast failure (WCAG).**
  Measured: white text (`rgb(255,255,255)`) on a near-white indicator
  (`rgb(248,250,252)`) in dark mode — the *selected* value was the least
  legible text on the page. `packages/gds-core/src/GdsFormControls.tsx`'s
  indicator fill reached for `--gds-brand-primary` first, which for the
  `default` preset is a neutral ink token (same value as
  `--gds-text-body-dark`), not an accent color. Swapped the fallback order
  to try `--gds-vibe-primary` (the preset's guaranteed accent hue) first —
  confirmed via the same measurement: indicator now resolves to `#7c3aed`
  (violet), legible white-on-accent.
- **Date/time-picker popover completely unthemed.** The open calendar
  (`.mantine-Popover-dropdown`, portalled to `document.body`) rendered
  Mantine's flat dark-mode default (`rgb(46,46,46)` background, `rgb(66,66,66)`
  border) instead of the theme's card gradient every other surface uses.
  `packages/gds-theme/styles.css` now themes `.mantine-Popover-dropdown`
  (light-mode `--gds-vibe-control` background, dark-mode gradient matching
  `.gds-paper`/`.gds-card`) — covers Select/Combobox/DatePicker dropdowns
  generally, not just the date picker.
- **Unchecked `Checkbox` inputs unthemed.** `styles.css` only themed the
  `:checked` state; the resting state fell through to the same flat Mantine
  default as the popover. Added a base `.mantine-Checkbox-input` rule.
- **`GdsSchemaForm` (and any "bring your own input" `FormField` usage)
  native inputs unthemed.** `renderDefaultField` mounts plain native
  `<input>`/`<select>`/`<textarea>` with no Mantine class to hang the
  existing theming rule off of — in dark mode these fell through to the
  *browser's own* native dark-mode form-control chrome (measured:
  `rgb(59,59,59)` background, `2px inset` border), unrelated to GDS's
  palette entirely. `styles.css` now themes bare native
  `input`/`select`/`textarea` elements (excluding checkbox/radio/range/
  file/color/button-like input types) within any GDS-themed page.
- **`GdsSchemaForm` title now a real heading.** Was a plain `<Text>` (a
  `<p>`), unreachable via screen-reader heading navigation; now
  `<Title order={4}>`.
- Not fixed here (design call, not a rendering defect): `ActionBar`'s
  icon-only actions use Mantine's `variant="subtle"` (no visible chrome by
  design) — flagged in #510 for a product decision, not changed.
- All 522 existing tests pass unchanged; every fix re-verified live via
  headless Chrome (`getComputedStyle` + screenshots) against the built
  playground before merging.

## 4.1.1 - 2026-08-07 — Generated Imagery: card-image placeholder use case + docs (#509)

Epic #503 shipped `GdsGeneratedThumbnail`/`GdsGeneratedHero` with full API
docs, but never demonstrated the use case that motivated the original ask:
using generated art as the placeholder image on GDS's own public cards.

- **Playground**: the `generated-imagery` pattern now shows `ListingCard`,
  `PublicProductCard`, and `PublicFoodCard` each using `GdsGeneratedThumbnail`
  as their `image` prop — no source changes to those cards were needed,
  since `image` already accepted `ReactNode`; this was a discoverability gap,
  not a missing capability.
- **`docs/GENERATED_IMAGERY.md`**: added a "Use cases" section (no listing
  photo yet, unphotographable category-only content, art-budget-free hero
  banners, share images) and a "Using it as the card-image placeholder"
  how-to section with a real `ListingCard` composition example.

## 4.1.0 - 2026-08-07 — Generated Imagery: theme-managed thumbnails & heroes (epic #503)

A turnkey, theme-managed generated-imagery system: deterministic, zero-network
SVG+HTML card thumbnails and hero backdrops composed from a consumer's own
category data — no image hosting, no AI/generative-model image calls, no
per-consumer design work. Generalizes a ClassScout engineering proposal
(verified against source before this was built: every one of its 26 activity
colors was an exact match to `gdsBadgeAccentShades`) into a first-class GDS
primitive, theme-managed by default rather than tied to one consumer's brand.

### New `GdsGeneratedThumbnail` and `GdsGeneratedHero` (#504, #505, #506)

- **`gdsSeededRandom`, `gdsGeneratedPaletteCssRefs`, `resolveGdsGeneratedPaletteHex`**
  (`generated-art-engine.ts`, #504): the shared foundations. A dependency-free
  deterministic PRNG (no `Math.random()` anywhere in this system, so SSR and
  client renders never mismatch), and a palette resolver with two sources —
  `paletteSource: 'theme'` (default, reads `--gds-brand-primary`/
  `--gds-brand-accent` as CSS var references, works for any of the 25 presets
  or a custom brand with zero config) and `paletteSource: 'category'` (opts
  into the existing fixed `gdsBadgeAccentColors`/`gdsBadgeAccentShades`, for
  consumers who want category color stable across theme changes).
- **`GdsGeneratedThumbnail`** (#505): card-scale — an accent wash, an
  oversized low-opacity icon motif (seeded rotation/scale/position from the
  entity id), and up to `maxBadges` ranked category badges. Two rendering
  layers on purpose: the SVG background is decorative and `aria-hidden`
  unconditionally; the badges are real HTML, individually accessible, never
  collapsed into one `role="img"` name.
- **`GdsGeneratedHero`** (#506): banner-scale — an accent wash, one of four
  pluggable background strategies (`'wash'`, `'mosaic-abstract'`,
  `'icon-field'`, or a consumer-supplied `{ type: 'region-mosaic', regions }`
  for geo-mosaic products — GDS ships the rendering adapter only, no city or
  brand data), and a seeded scatter of up to 6 ranked badges at a fixed size
  ladder (one large, two medium, three small — fixed slots, not free
  placement, so it never reads as clutter).
- **Contrast is guaranteed, not assumed, for `paletteSource: 'theme'`.**
  `'theme'` colors arrive as CSS `var(...)` references with no resolved hex
  to check in JS, so every surface carrying fixed white text is pushed
  through `color-mix(in srgb, <color> 30%, black)` first — provably clears
  ≥7:1 against white even for the lightest possible input (pure white), so
  the guarantee holds without knowing the theme's actual values.
- Fixed `getGdsVibeThemeCssVariables`'s (`gds-theme`) return type to
  `Record<string, string>`: it already merges in the full `--gds-*` semantic
  role set at runtime, but the narrower object-literal type TypeScript
  inferred across the compiled `.d.ts` boundary didn't reflect that. No
  behavior change — verified against all three existing call sites.
- Registered as the `generated-imagery` playground pattern with a live demo:
  `GdsGeneratedThumbnail` with both palette sources side by side, and
  `GdsGeneratedHero` across all four background strategies.
- See [`GENERATED_IMAGERY.md`](docs/GENERATED_IMAGERY.md) for the full
  palette/background/accessibility reference, and
  [`BADGE_SYSTEM.md`](docs/BADGE_SYSTEM.md)'s new "The same activity identity,
  worn a different way" section for how this extends the map-pin accent+shade
  table rather than inventing a parallel one.

### Headless SVG generation for `og:image` and email (#508)

- **`buildGdsThumbnailSvg()`/`buildGdsHeroSvg()`** (`generated-art-svg.ts`):
  framework-agnostic, non-React twins of the two components above, returning
  a complete, self-contained `<svg>` string — for `og:image` routes, email,
  or any SSR/rasterization context with no live browser CSS cascade.
  Exported from **`@sovereignsquad/gds-core/server` only**, never
  `/index`/`/client`: they use `react-dom/server`'s `renderToStaticMarkup`
  for icon rendering (Node/edge-SSR-safe, the same primitive Next.js's own
  SSR pipeline is built on), which has no browser bundle to leak into by
  mistake if it stayed out of the default barrel.
- `'theme'`-mode colors need a literal resolved value here (no live DOM to
  read a CSS variable from), so `resolveGdsGeneratedPaletteHex` requires
  either `themePresetId` (one of the 25 built-in presets) or an explicit
  `colors` override — see the module docs for why there's no silent
  fallback guess. The `color-mix(...30%, black)` contrast guarantee is
  reproduced as real RGB arithmetic on the literal hex (same ratio, same
  provable floor), and badge labels are hand-laid-out SVG `<text>`, since
  there's no HTML/CSS cascade to fall back on outside a browser.
- Placement math (motif transform, hero badge-slot table, mosaic tiling,
  icon-field scatter) is intentionally re-derived here rather than imported
  from the React components: small, pure arithmetic with its own tests
  holding both sides to the same documented geometry, not business logic
  worth risking a cross-module refactor of two already-shipped components
  over.
- `docs/GENERATED_IMAGERY.md` gains the Next.js recipe (serving the SVG
  string directly as `Content-Type: image/svg+xml` — the exact gap the
  originating ClassScout proposal called out: "no listing has a share
  image... per-listing `og:image` was never built") and a worked
  `region-mosaic` example, using a fabricated region set, not real
  consumer geo data.
- Every generated sample was rendered to PNG via headless Chromium and
  visually inspected, not just asserted structurally correct: this caught
  a real bug in an early draft (icon `<svg>` elements ended up with
  duplicate, conflicting `width`/`height`/`viewBox` attributes from a
  string-splice approach, leaving nested-SVG percentage sizing unresolved)
  before it shipped.

## 4.0.0 - 2026-08-07 — Unified, always-theme-aware badge system (epic #484)

Everything below ships together as the badge-system release: foundations (#485, #486), shape
vocabulary (#487), canonical badge icons (#494), the component layer (#488–#491), cleanup
(#493), docs (#492), the guided-tour mobile fix (#495), a live-site modal fix (#496), the badge
composition gallery (#499), a docs fix for hand-built pin compositions (#500), the new
`GdsMapPinBadge` marker component (#501), and a contrast-safe within-accent shade axis for it
(#502).

**Published as 4.0.0, not 3.15.0.** `3.15.0` (the subset through #496) was already published to
the registry earlier the same day this line's remaining work (#499–#502) landed; a published
package version is immutable, so the only way to ship the rest of this epic is under a new
version number. This release is a major bump for that reason alone — every change in it is
additive and backward compatible, none of it breaking.

### `GdsMapPinBadge` gains a contrast-safe `shade` axis for within-accent differentiation (#502)

`accent` is a closed union of 10 colors — the right granularity for top-level
categories, but too coarse when several related sub-categories (e.g.
different sports) need to read as one accent family while staying
individually distinguishable. `fillOpacity` doesn't solve this: it's
transparency, not color differentiation, and does nothing in outline mode.

- **New `GdsBadgeAccentShade` type** (`'base' | 'deep' | 'deeper' |
  'deepest'`) and **`gdsBadgeAccentShades` precomputed palette**
  (`GdsBadge.tsx`, alongside `gdsBadgeAccentColors`): 10 accents × 4 levels,
  40 fixed hex values.
- **New `GdsMapPinBadge` prop `shade?: GdsBadgeAccentShade`**, default
  `'base'` (backward compatible — omitting it changes nothing).
- **Darker-only, not an arbitrary limit.** Sweeping lightness deltas across
  all 10 accents against the white icon color `GdsMapPinBadge` uses in
  filled mode shows that lightening any accent — even slightly — drops some
  of them below the 4.5:1 WCAG AA bar the base palette already guarantees
  (`teal` fails first, at only +4 lightness; `ocean`/`bronze`/`forest`/
  `terracotta` follow shortly after). Darkening has generous headroom for
  all 10, so that's the only direction offered. Every one of the 40 shade
  values is verified ≥ 4.5:1 against white in tests, the same bar
  `gdsBadgeAccentColors` itself is held to.
- **Proportional spacing, not a fixed lightness delta.** Each accent's four
  levels interpolate from that accent's own base lightness down to a shared
  lightness floor in three equal steps, rather than subtracting the same
  fixed amount from every accent — a fixed delta reaches the floor at
  different points for different accents (`teal` starts darker than most),
  producing near-duplicate `deeper`/`deepest` colors for exactly those
  accents. Proportional spacing keeps all four steps visually distinct for
  every accent, verified in tests (no two of one accent's four shades may
  be equal).
- The interactive spec artifact's "Sports family" demo — which had
  previously used a live-computed, unverified lightness shift including
  *lighter* steps — is corrected to use the real, contrast-verified
  `gdsBadgeAccentShades` values.
- `docs/BADGE_SYSTEM.md` gains a "Within-accent differentiation: `shade`,
  not transparency" section.
- Scoped to `GdsMapPinBadge`; `GdsBadge`'s own `accent` prop is unaffected.

### New `GdsMapPinBadge`: governed map-pin marker, exactly two layers, no ring (#501)

A consumer team building activity-category map pins (Football, Trophy, ChefHat, TheaterMasks,
Atom, Palette, Music — none of which exist in `GdsIcons`, so external icon sourcing is the
correct path) went through three iterations, each with a different real defect: icon not
centered, icon stroke heavier than the pin outline, and a visible label reading "BallFootball"
(traced to deriving the label from Tabler's `IconBallFootball` component's own `displayName`
rather than supplying a real category label). Each defect was individually documentable, but the
actual gap was that hand-composing a category-colored pin marker had a set of constants to match
by hand and no component to just use.

Building the component surfaced two more defects of its own, caught before release: a first
draft added a ring capsule behind the icon to guarantee contrast in filled mode, but the opaque
disc shrank the icon to a sliver; dropping the ring without also changing the icon's own color
just moved the bug — the icon kept the accent color and disappeared once the pin behind it
filled with that same color. The shipped design fixes the actual problem instead of routing
around it with a third layer.

- **New `@sovereignsquad/gds-core` export `GdsMapPinBadge`**: `accent` (required, one of the
  curated 10 — never a free color), `icon` (a `GdsIconKey` **or** any externally-sourced icon
  element), `label` (required, consumer-supplied — the component never derives it from an icon's
  own display name), `filled` (solid pin for real basemap imagery vs. the default outline mode
  for schematic contexts), `fillOpacity` (0–1, filled mode only — touches only the pin's own
  fill, never the icon), `size`. **No `ring` prop.**
- **Exactly two layers: the pin, and the icon.** The icon's color always contrasts the pin's own
  fill instead of ever reusing it — outline mode shares one `accent` color between pin and icon
  (no fill to collide with); filled mode switches the icon to an inverse (white-on-dark) color and
  keeps it fully opaque regardless of `fillOpacity`. With no ring to share space with, the icon is
  sized to `0.46` of the marker (up from the `0.42` used inside `GdsBadge`'s small inline
  `shape="pin"` badge, but not the removed ring's old `0.62` footprint — the pin head is a circle,
  and wide-content icons like `IconMasksTheater`/`IconBike` render past its boundary above ~0.48,
  verified against the widest icons actually shipped rather than only centered ones).
- Built on `GdsBadgeStack`/`GdsBadgeStackLayer`, locking in `translateY(-4.1667%)` for the icon —
  the pin head's own circle center solved from its path's arc geometry (radius 8, center
  `(12, 11)` in the 24-unit path — one unit above the path box's own center, not the same point)
  — and `stroke={1.75}` matching the pin. An externally-sourced icon element has its `stroke`
  forced to `1.75` via `cloneElement` regardless of what the consumer's element passed — the exact
  mismatch that kept recurring by hand is no longer possible to reproduce through this component.
- Fixes a real, previously-shipped rendering bug found while building this: `GdsBadgeStackLayer`
  applies its `scale` prop via a CSS class reading a custom property, but supplying a `style`
  prop with its own `transform` (needed for the pin's vertical offset) takes cascade priority
  over that class rule and silently drops the scale — `GdsBadge`'s own `shape="pin"` composition
  had exactly this bug since it shipped: the icon was rendering completely unscaled. Both are now
  fixed by including the scale directly in the same transform string as the offset.
- The shipped "Badges on a map" section of the composition gallery (`/patterns/feedback`) now
  uses `GdsMapPinBadge` in place of its previous hand-composed pin markers.
- Docs: [`docs/BADGE_SYSTEM.md`](docs/BADGE_SYSTEM.md) gains a "Map markers: use
  `GdsMapPinBadge`, don't hand-compose one" section, including the real
  `IconBallFootball`/`"BallFootball"` example as the concrete reason the accessible label must
  never come from an icon library's own display name, and the ring-capsule/icon-contrast
  rationale above.

### Docs: hand-built shape+icon compositions must match GdsBadge's own centering/stroke contract (#500)

A consumer team built their own map-pin icons (composing `GdsBadgeShapePin` + an icon by hand
instead of `GdsBadge`'s `shape="pin"` prop) and shipped icons sitting visibly low in the pin
head, at a mismatched stroke weight to the pin outline, and in one flat color with no
per-category differentiation. The centering math (`scale: 0.42` + `translateY(-4.1667%)` for
pins — the pin head circle's own solved center, not an eyeballed value — `scale: 0.55` with no
offset for the other five shapes) existed only as a source comment in `GdsBadge.tsx`, never as an
explicit rule in the SSOT doc a consumer building their own composition would actually read.

- [`docs/BADGE_SYSTEM.md`](docs/BADGE_SYSTEM.md) gains a "Hand-built shape+icon compositions
  must match `GdsBadge`'s own contract" section stating the exact scale/offset/stroke values and
  pointing to `GdsBadge.tsx`'s own composition and the live "Badges on a map" gallery section as
  reference implementations.
- Documentation-only fix; not lint-enforceable, since it governs compositions built outside
  GDS's own components.

### Badge composition gallery: cards, buttons, maps, profiles, overlays, live cross-theme (#499, part of epic #498)

The badges pattern demonstrated vocabulary swatches but no real usage guidance — how a badge
actually looks composed onto the surfaces it ships on. Adds that composition directly to the
governed pattern demo, built entirely from shipped components (no design-preview artifact
stands in for this).

- **New `GdsVibeThemeScope`** (`@sovereignsquad/gds-theme`): scopes a subtree to one theme
  preset/scheme's `--gds-vibe-*`/`--gds-*` semantic CSS variables via `getGdsVibeThemeCssVariables`,
  so a page can preview several presets side by side without switching the app-wide theme.
  Server-safe (no hooks/browser APIs).
- **Badges on cards**: `ProductCard`'s `status`/`footer` and `ListingCard`'s `score`/`reason`
  slots — already `ReactNode` — now render real `GdsBadge`s in the badges gallery demo, no
  component changes required.
- **Badges beside buttons**: `GdsCountBadge`'s `anchor` corner-anchors to an icon inside a real
  `<button>`, matching the existing anchor contract; the button carries the accessible name,
  the pill stays decorative.
- **Badges on a map**: `MapPanel`'s `renderMap` composes `GdsBadgeShapePin` with a **`fill`
  override** (it defaults to a Tabler *outline* icon) into a filled marker, positioned with
  `GdsBox`'s `pos`/`top`/`left` style props — no ad hoc inline CSS.
- **Badge clusters on a profile**: multiple badges in a wrapping row beside identity, distinct
  from `GdsBadgeStack`'s corner model, which stays reserved for one verification mark.
- **Badges in overlays**: a `GdsDialog` confirming a badge was earned, and an `InlineAlert` with
  a badge in its `action` slot.
- **Live cross-theme matrix**: `VibeThemePicker` + `GdsVibeThemeScope` prove, live, that
  `GdsBadge`'s `tone="success"` genuinely shifts per preset while `warning`/`danger`/`info`
  (and every `accent`) render the same fixed value on every preset — three distinct, verified
  behaviors the gallery shows side by side instead of one blanket "theme-aware" claim.
- Docs: [`docs/BADGE_SYSTEM.md`](docs/BADGE_SYSTEM.md) gains a "Composition gallery" section
  documenting each surface and `GdsVibeThemeScope`.

### Fix permanently-open, un-closeable modal on `/patterns/feedback` (#496)

The destructive-actions demo hardcoded `ConfirmDialog`'s `opened` to a literal `true` with
no-op `onClose`/`onConfirm`, so the dialog opened on first paint and could never be dismissed —
Escape, backdrop click, and the close button all called the no-op, and Mantine's `Modal` locks
body scroll and traps focus while open, so the entire page was inert behind it.

- Replaced with a real `useState`-driven demo (trigger button, working close), matching the
  existing `OverlayAliasDemo` convention already used elsewhere in the file.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) gains a standing rule: overlay demos must never hardcode
  `opened` to a literal `true`.

### Badge system components: GdsBadge, GdsCountBadge, GdsRemovableTag, GdsBadgeStack (#488–#491), cleanup (#493), and docs (#492)

Completes epic #484's component layer on the shipped foundations (#485–#487, #494). New
front-door doc: [`docs/BADGE_SYSTEM.md`](docs/BADGE_SYSTEM.md).

- **`GdsBadgeStack` + `GdsBadgeStackLayer`** (#488): Font Awesome-model layering box (square
  `1em`-default, centered/corner layers, corner scale via custom properties). Corner dots
  separate from the base mark with a CSS **mask cutout**, never a ring painted in the page
  background (which breaks over vibe-theme gradients); all layers are `currentColor` DOM/SVG,
  so forced-colors keeps them visible.
- **`GdsBadge`** (#489): static status/meaning label with the closed two-axis color union —
  semantic `tone` (`success|warning|danger|info|neutral` → `--gds-state-*` tokens) XOR curated
  `accent` (10 fixed values, each test-verified ≥ 4.5:1 against white via
  `pickGdsAutoForeground`); governed `icon` (GdsIcons) and `shape` (GdsBadgeShapes) props;
  `label` required — meaning never in color alone; never interactive.
- **`GdsCountBadge`** (#490): numeric/dot count, `value` XOR `dot` at the type level,
  corner-anchorable to any element via the stack. Its `role="status"` live region is
  **always mounted** (a region mounted later never announces its first appearance) and
  announces "{count} {label}" — "99+ notifications", never the reverse.
- **`GdsRemovableTag`** (#491): the removable filter token as one shared component — whole
  chip is a `<button>` with a required, consumer-localized `removeLabel` (no baked-in
  strings). Adopted by all four former inline copies: `ActiveFilterChips`
  (ListingPrimitives), `DataToolbar`, `BrowseSurface`, and gds-admin's `ResponsiveDataView`.
- **Cleanup (#493)**: `MeaningBadge` and `FitScoreChip` gained their missing live demos;
  both components' `{...props}`-after-`style` spread no longer lets a caller's `style` wipe
  token colors (merge order fixed, regression-tested); the `PillBar`/`SoftChipGroup`/
  `FilterChipGroup` radiogroups now implement the roving tabindex + arrow-key contract their
  ARIA roles promise (one tab stop per group; arrows/Home/End move selection and focus).
  The stale DTCG token count item was already fixed in #485's commit.
- All new components registered (registry + live demo + export coverage + catalog parity)
  and asserted mounted-and-painted under forced-colors on `/patterns/feedback`.

### Guided tour: step card no longer buried under the spotlighted target (#495)

On small viewports, a tall spotlighted section (e.g. the home page Theme Lab) overlapped the
bottom-anchored step card and painted **over** it — Skip/Next unreachable, page inert behind the
scrim: the tour was unusable on mobile. Root cause: the card rendered *inside*
`.gds-tour-spotlight`, whose own overlay-level stacking context capped the card below the
elevated `[data-gds-tour-active-target]` (overlay + 1).

- The card is now a portal **sibling** of the spotlight at `overlay + 2`, giving the intended
  order scrim < spotlighted target < step card; regression-tested structurally (dialog must not
  be a descendant of the spotlight) and re-verified on a 390×844 viewport with hit-testing.
- [`docs/GUIDED_TOUR.md`](docs/GUIDED_TOUR.md) now documents the layering guarantee.

### Badge shape vocabulary: six Tabler-geometry silhouettes (#487, part of epic #484)

Approved via the visual proposal review: six badge silhouettes, all authored from Tabler's own
`iconNode` path data through Tabler's public `createReactComponent` — imported geometry, never
hand-drawn, so the 24×24 space, corner language, and `currentColor` stroke behavior match the
`GdsIcons` registry by construction.

- **New `@sovereignsquad/gds-core` exports**: `GdsBadgeShapeCircle` (← `IconCircle`),
  `GdsBadgeShapeSquircle` (← `IconSquareRounded`), `GdsBadgeShapeHexagon` (← `IconHexagon`),
  `GdsBadgeShapeShield` (← `IconShield`), `GdsBadgeShapeRosette` (← `IconRosette`), and
  `GdsBadgeShapePin` (← `IconMapPin`'s balloon silhouette, decorative inner dot dropped so the
  head can host a composed icon — for badges placed on maps), plus the closed `GdsBadgeShapes`
  dictionary and `GdsBadgeShapeName` union that #489's `GdsBadge` will typecheck its `shape`
  prop against.
- Deliberately siblings of `<GdsIcon />`, not registry keys: they expose the full Tabler prop
  surface (`className`/`style`/`ref`/rest-spread) that badge composition (#488) needs and
  `<GdsIcon />` intentionally withholds.
- Suggested shape-to-meaning pairing (documented default, not enforced): circle=interest/count,
  squircle=persona, hexagon=activity, shield=verification, rosette=certification, pin=location.
- Demoed on the badges pattern; asserted mounted-and-painted under forced-colors on
  `/patterns/feedback`; documented in [`docs/ICON_REGISTRY.md`](docs/ICON_REGISTRY.md) and
  [`COMPONENTS_AND_PATTERNS.md`](COMPONENTS_AND_PATTERNS.md).

### Badges render the canonical GdsIcons set (#494, part of epic #484)

The governed icon dictionary's `status` category maps one-to-one onto badge semantics, yet no
badge component used it — `StatusBadge` was text-only and `MeaningBadge`'s `icon` was an
ungoverned `ReactNode` pass-through.

- **`StatusBadge` gains opt-in `withIcon`**: renders the canonical status icon
  (`Success`/`Warning`/`Danger`/`Info`) through `GdsIcon` ahead of the label, decorative
  (`aria-hidden`) since the label carries the meaning; `neutral` has no canonical status icon and
  renders none.
- **`MeaningBadge`'s `icon` prop now routes canonical `GdsIcons` keys through `GdsIcon`**
  (e.g. `icon="Warning"`, `icon="Star"`); any other `ReactNode` renders exactly as before, so
  existing custom-markup callers are unaffected.
- Playground demos show both; [`COMPONENTS_AND_PATTERNS.md`](COMPONENTS_AND_PATTERNS.md)'s badge
  rules now state that badge icons come from the governed dictionary, never ad hoc SVG.

### Badge-system foundation: auto-foreground contrast helper (#486, part of epic #484)

Badges render on 25 theme presets × custom brand colors; a static foreground color can't stay
readable across all of them. Adds the pick-a-safe-foreground primitive the upcoming `GdsBadge`
family needs, on top of the existing consumer contrast checker (#453).

- **New `pickGdsAutoForeground(background, options?)`** (`@sovereignsquad/gds-theme`): tries each
  of `options.candidates` (default `['#ffffff', '#000000']`) against `background` in order and
  returns the first that clears the requested WCAG threshold (`AA`/`AAA` × `normal`/`large`,
  same as `checkGdsContrast`); if none clear the bar, returns whichever scored highest so the
  result is always the best available choice. **Never throws** — an unparseable `background` or
  candidate falls back to the first candidate, since this is meant to be safe to call directly in
  a render path over a caller-supplied color GDS doesn't control.
- Exists because neither existing option works here: Mantine's `autoContrast` is a structural dead
  end for `var(--gds-*, fallback)` values, and `getGdsContrastRatio` correctly throws on
  unparseable input, which would crash a render.
- Docs: [`docs/CONTRAST_CHECKER.md`](docs/CONTRAST_CHECKER.md) gains a `pickGdsAutoForeground`
  section alongside `getGdsContrastRatio`/`checkGdsContrast`.

### Badge-system accessibility audit: 6 confirmed bug fixes (#478–#483)

Fixes found while researching the upcoming customizable badge system, landed ahead of that work per the agreed "fix bugs now, then badge system" sequencing. No public API removals; `brandContrastRatio` now throws on unparseable input instead of silently scoring it as black (see #483).

- **Removable filter chips are now keyboard-operable** (#478): `ActiveFilterChips`, `DataToolbar`, `BrowseSurface`, and `ResponsiveDataView` rendered their remove affordance as a `<div onClick>`, unreachable by keyboard. They now render via Mantine `Badge`'s polymorphic `component="button"` with a real `aria-label` and `type="button"`.
- **`StatusBadge`/`LabelTag` semantic color no longer gets overridden by theme presets** (#479): the preset decorative Badge tint (specificity `(0,2,1)`) beat Mantine's own color-prop-driven Badge styling (specificity `(0,1,0)`), silently repainting every semantic status badge to the same brand tint under any of the 25 presets. Both components now mark themselves `data-gds-badge-fixed-tone`, which the preset rule excludes.
- **`FitScoreChip` tooltip is now keyboard-reachable, and its "good"/"partial" bands can no longer render identically** (#480): the chip gained `tabIndex`/focus-triggered tooltip events, and the `partial` band now reads `--gds-brand-accent-action` instead of duplicating `--gds-brand-accent`.
- **`GdsIcon`/`resolveGdsIconKey` now resolves the lowercase form of every multi-word icon key** (#481): the previous single-character-capitalize fallback broke 14 keys (`TrendingUp`, `EyeOff`, `ChevronDown`, etc.), silently falling back to the generic `Help` icon. Replaced with a full case-insensitive lookup table.
- **`createBrandTheme()` now emits `--gds-text-on-inverse`** (#482), the fully-kebab-cased name every consumer and preset actually reads, alongside the previously-emitted (but unused) `--gds-text-onInverse`.
- **`brandContrastRatio()` now throws on unparseable hex input** (#483) instead of silently coercing it to black via `NaN`-bitwise-coercion, which previously produced a plausible-but-wrong contrast ratio for any caller passing e.g. a CSS variable reference.

### Badge-system foundation: semantic role tokens for all 25 presets (#485, part of epic #484)

Only `class-usa`/`gold-athlete` (2 of 25 presets) defined the `--gds-state-*`/`--gds-badge-*`/`--gds-brand-*`/etc. semantic role variables a theme-aware badge needs. Rather than give the new badge system a fallback-chain-only escape hatch, the gap is closed at the token layer: every preset now defines the full role set.

- **New `packages/gds-theme/src/color-math.ts`**: sRGB color parsing/mixing/contrast utilities, extracted from `accessibility-report.ts` (no behavior change there — same math, now shared) so they can back the new derivation below too.
- **New `deriveVibeSemanticCssVariables()`** (`vibe-themes.ts`): for the 23 presets without a hand-authored semantic set, mixes each role from that preset's own `GdsVibeTheme` palette in sRGB (matching the runtime `color-mix(in srgb, ...)`) and pushes it toward black/white until it clears WCAG AA (text) or non-text AA (3:1) against its background — verified for all 25 presets, both modes, in `vibe-themes.test.ts`. `state-danger`/`state-danger-dark`/`state-warning-dark` are fixed, non-preset-tinted anchors rather than derived, matching the values already shared identically between the two hand-authored presets. `class-usa`/`gold-athlete`'s existing values are untouched.
- Docs: [`docs/SEMANTIC_ROLE_TOKENS.md`](docs/SEMANTIC_ROLE_TOKENS.md) updated to reflect that all 25 presets — not just 2 — now define the full role set; [`docs/DESIGN_TOKENS_DTCG.md`](docs/DESIGN_TOKENS_DTCG.md)'s stale "391 tokens across 23 presets" corrected to the actual 425/25.

## 3.14.17 - 2026-08-02 — Guided tour: consistent rollout across every primary destination (#475)

Extends the onboarding tour from two surfaces to **every primary site destination** through one shared launcher, so the "Take the guided tour" experience is identical everywhere and the gate-safe auto-start rule lives in exactly one place.

- **New governed module export — `GdsTourButton` (`@sovereignsquad/gds-core`):** a themeable `.gds-tour-launch` launcher whose label reads the new localized `gds.tour.launch` key (added to all 12 locale packs) and starts a tour via `useGdsTour`. Customers get a drop-in launcher instead of hand-rolling a raw control.
- **New shared site control — [`SiteTourLauncher`](apps/playground/src/SiteTourLauncher.tsx):** composes `GdsTourButton` + a gate-safe first-run auto-start (bare URL + real browser). The consistent launcher appears on every page and the auto-start decision is centralized here instead of copy-pasted per page.
- **Auto-start pages:** Home, Use with AI, Pattern Catalog, Live Demos, API Reference, Coverage, Maturity, Use Cases, Governance, and Request a Feature each ship a page-specific spotlight tour that runs once for first-time visitors and replays on demand.
- **Manual-only on gate routes:** Install (`/install`) and Themes (`/themes`) expose the launcher button but **omit** auto-start, because the accessibility / theme-trust / forced-colors runtime gates visit those bare routes — an auto-opened overlay would surface mid-verification. This corrects the Install page, which briefly carried auto-start.
- **Gate safety, centralized:** auto-start fires only when `window.location.search === ''` (real visitors arrive on clean paths; gates visit `/?locale=xx` or deep sub-routes) **and** `Element.prototype.scrollIntoView` exists (`true` in Chrome, `false` under jsdom, so page unit tests never auto-fire). No automation sniffing.
- Docs ([`docs/GUIDED_TOUR.md`](docs/GUIDED_TOUR.md)) updated with the full destination list, the manual-only rule, and the new `GdsTourButton` surface; `verify:release` (including the accessibility, theme-trust, and forced-colors runtime gates) stays green.

## 3.14.16 - 2026-08-02 — Guided tour: first-run onboarding on the home page (#474)

Brings the auto-running onboarding tour to the home/overview page, where every visitor lands.

- The home page now auto-runs a short tour once for first-time visitors — spotlighting the live Theme Lab, the "what GDS gives you" band, and the get-started links — with a "Take the guided tour" launcher that replays it on demand.
- **Gate-safe by a no-query signal:** the home route *is* loaded by the `theme-trust` gate, but only ever as `/?locale=xx`, so auto-start is gated on `window.location.search === ''` — a real visitor on bare `/` sees it once; the gate's `/?locale=…` visits (query present) never trigger the overlay. Documented in [`docs/GUIDED_TOUR.md`](docs/GUIDED_TOUR.md). No module API change; `theme-trust` and the full `verify:release` stay green.

## 3.14.15 - 2026-08-02 — Guided tour: auto-run once for first-time visitors (#473)

Completes the onboarding half of the guided-tour module (#466): the GDS site now **auto-runs** the tour once for first-time visitors, not just on manual click.

- The [Use with AI](https://sovereignsquad.github.io/general-design-system/ai) page mounts `GdsGuidedTour` with `open persist="localStorage"`, so a fresh visitor sees the tour once and never again; the "Take the guided tour" launcher still replays it on demand.
- **Gate-safe by construction:** auto-start is scoped to the `/ai` route, which no headless runtime gate (`theme-trust`, `forced-colors`, `input-zoom`, `kanban-drag`) loads — so the overlay can never surface during a verification run. The rationale (and guidance for consumers doing their own auto-start) is documented in [`docs/GUIDED_TOUR.md`](docs/GUIDED_TOUR.md). No module API change.

## 3.14.14 - 2026-08-02 — Guided Onboarding Tour module (spotlight coach-marks) (#466)

Ships a first-class, reusable **guided onboarding tour** so every product on GDS gets one governed, accessible, i18n'd first-run flow with no app-level forks — and dogfoods it on the GDS site.

- **New module** (`@sovereignsquad/gds-core`): `GdsTourProvider`, `useGdsTour()`, the declarative `GdsGuidedTour`, the `GdsTourStep` type, and `useHasSeenTour()`. A tour dims the viewport, cuts a spotlight hole over the current target, and anchors a step card with Back / Next / Skip / Done and a "Step _n_ of _m_" indicator; advancing moves the spotlight and scrolls the next target into view. Targets are referenced by a stable `data-gds-tour-target` id or a React ref (#467–#470).
- **Governed scrim token** (`gds-theme`): `--gds-overlay-scrim` (light/dark; `transparent` under forced-colors) plus `--gds-tour-spotlight-radius`/`-padding`, so no raw `rgba()` dim lands in product code.
- **Accessibility**: focus-trapped `role="dialog"` step card with `aria-labelledby`/`describedby`, focus-return-to-invoker, `Esc`/arrows/`Enter`/`Tab` handling, a polite "Step _n_ of _m_" live region, forced-colors outline degrade, and `prefers-reduced-motion` support. Controls read new `gds.tour.*` keys shipped across all 12 locale packs.
- **Docs**: new [`docs/GUIDED_TOUR.md`](docs/GUIDED_TOUR.md) (contract + consumer drop-in), a Guided Onboarding Tour rules section in `COMPONENTS_AND_PATTERNS.md`, and `llms.txt` coverage.
- **Dogfood**: the [Use with AI](https://sovereignsquad.github.io/general-design-system/ai) page gains a "Take the guided tour" control that spotlights the llms.txt entry point, the install/bootstrap step, and the non-negotiable agent rules.
- Board taxonomy extended with an `area: onboarding` label.

## 3.14.13 - 2026-07-31 — Vendor-neutral AI-tool naming across docs, templates, and site copy (#465)

Removes tool-specific branding from the AI-integration surface so GDS reads as consumable by any LLM-powered coding tool rather than naming particular products. No runtime, token, or component-API change.

- **Removed the design-tool sync feature** (`.design-sync/` inputs, previews, and conventions, plus `docs/CLAUDE_DESIGN.md`): the sync path was documentation and committed preview inputs for a specific external design tool, not part of the shipped packages. The authoritative design-to-code path remains the DTCG token export (`tokens/gds.tokens.json`) and the handoff mapping documented in [`docs/FIGMA_UI_KIT.md`](docs/FIGMA_UI_KIT.md) and [`DESIGN_HANDOFF.md`](DESIGN_HANDOFF.md).
- **Removed `TEMPLATES/CLAUDE.md.template`**: the cross-tool `TEMPLATES/AGENTS.md.template` (the `AGENTS.md` standard read by any agentic coding tool) is the single drop-in repo-rules template going forward.
- **Neutralized AI-tool naming** in `README.md`, `llms.txt`, `docs/AI_AGENT_GUIDE.md`, `docs/CLASSSCOUT_INTEGRATION.md`, `TEMPLATES/AGENTS.md.template`, the `/ai` playground page, and all 8 non-English site-phrase packs — product names replaced with neutral descriptors ("AI coding agents", "any LLM-powered coding tool"). The `/ai` page drops the design-tool sync section and the tool-specific drop-in row; its link grid and repo-rules table are updated accordingly.

## 3.14.12 - 2026-07-27 — KanbanBoard zone-based wheel-scroll routing (#464)

Fixes a real desktop-trackpad complaint from a consumer (`salesleadgenerator`): in multi-column layout the columns live in a horizontal `ScrollArea`, and a two-finger "natural scroll" gesture over a card could be captured by that horizontal region instead of scrolling the page.

- **`data-gds-kanban-column-header="<columnId>"`** (always on, additive): each column header now exposes a stable hit-region attribute so consumers can target it — previously cards/columns/bodies/footers had `data-gds-kanban-*` but the header did not.
- **`columnPanZone?: 'header' | 'none'`** (default `'none'`, fully backward compatible): opts into Linear-style zone routing. With `'header'`, a wheel gesture over a column header pans the columns horizontally regardless of gesture shape (a single non-passive listener on the `ScrollArea` viewport, routing via `closest('[data-gds-kanban-column-header]')`), while a gesture over a card or empty space is never captured and scrolls the page normally. Fine-pointer (desktop) only, inert in stacked orientation, RTL-aware. Existing consumers who don't opt in see zero behavior change.
- Unit-tested for the routing decision (which zone captures the gesture, asserted via `preventDefault`); the physical trackpad scroll is a real-browser verification, since headless synthetic wheel events don't reproduce trackpad-driver behavior. Documented in `COMPONENTS_AND_PATTERNS.md`.

## 3.14.11 - 2026-07-26 — Complete JSDoc coverage on the public API + coverage gate

- **Full JSDoc backfill** (#414): every public export across the consumer-facing packages now carries a JSDoc block — **1,136/1,136 public exports documented (100%)**, up from ~6%: `gds-core` 914, `gds-admin` 85, `gds-theme` 137, `gds-a11y` already complete. Component functions get a summary of what they are and their governed behavior; props interfaces get an interface-level block plus **per-property docs on the fields consumers hover** (defaults, throw conditions, accessibility roles), all written from the actual implementation. Comment-only — no runtime or type changes beyond the emitted `.d.ts` now carrying the docs, so editors surface field-level hover documentation for the entire shipped API.
- **New coverage gate** (#414): `verify:api-jsdoc-coverage` (wired into `verify:references` / `verify:release`) asserts public exports stay documented at a ≥95% floor per package and overall, mechanically enforcing the ship-with-docs Definition of Done so the surface can't silently regress. Closes #414.

## 3.14.10 - 2026-07-26 — First-class accessibility theme presets: high-contrast + colorblind-safe

Delivers the theme-preset half of #453 — the two accessibility presets peers ship (Primer's high-contrast/colorblind lanes) that GDS lacked as *selectable presets* despite having forced-colors support and contrast CI.

- **`high-contrast` preset** (#453): a maximal-contrast, flat, undecorated accessibility lane. Pure black/white canvases and surfaces, black/white body text (21:1), near-pure dark-gray/light-gray meta text (AAA, ≥11:1 both schemes), solid black/white borders, near-black filled controls (`primaryColor: 'dark'` + `autoContrast`), no shadows, and no decorative gradients (a scoped `styles.css` rule flattens the body and suppresses the vibe overlay). Distinct from OS-driven `forced-colors` support (which GDS also honors) — this is a preset a product or user can choose deliberately.
- **`colorblind-safe` preset** (#453): a brand palette drawn from the **Okabe-Ito** colorblind-safe qualitative set (Okabe & Ito, 2008) — `primary` = blue `#0072b2`, `accent` = vermillion `#d55e00`, the classic pairing that stays distinguishable across deuteranopia/protanopia/tritanopia — with `autoContrast` filled controls and dark-on-light AA/AAA text. It targets the categorical/brand palette; GDS's standing "never signal state by hue alone" rule (semantic components carry a label + icon per WCAG 1.4.1) already keeps success/danger distinguishable under every preset.
- Both are first-class entries in `getGdsThemePresets()` (so they appear in the Theme Lab automatically) and ship full `GdsVibeTheme` token sets. The token graph, `verify:token-contrast-scoring` (now **200 readable-text pairs across 25 themes**, all ≥ AA — the two new lanes clear AAA), `verify:theme-accessibility` (300 checks), the DTCG export (`tokens/gds.tokens.json`, 25 themes), and `THEME_GOVERNANCE.md` are all updated. Additive; no change to existing presets.

## 3.14.9 - 2026-07-26 — Default semantic-role token layer for the base gdsTheme

- **Default semantic-role token layer** (#451): the base `gdsTheme` now defines its **structural** semantic roles at `:root` in `@sovereignsquad/gds-theme/styles.css` — `--gds-bg-canvas`/`--gds-bg-page`/`--gds-bg-surface`/`--gds-bg-card`/`--gds-bg-inverse`, `--gds-border-card`, `--gds-text-body`/`--gds-text-primary`/`--gds-text-meta`/`--gds-text-secondary`, and `--gds-text-on-inverse` (light/dark via CSS `light-dark()`). Previously these role variables were defined only by `createBrandTheme(...)`, so the base theme left them undefined and every component fell back to a divergent per-call-site guess (`--gds-bg-surface` resolved to `#eee`/`gray-1`/`white` in different places). Now a component reads **one governed default** regardless of where it's used.
  - **Values match the contrast-gated `default` theme**, so the documented per-token-pair **WCAG AA contrast contract** (`text-body`/`text-meta` on `bg-surface`/`bg-canvas`, `text-on-inverse` on `bg-inverse`) is guaranteed and **already policed by `verify:token-contrast-scoring`** — no new gate needed. Full table in the new [`docs/SEMANTIC_ROLE_TOKENS.md`](docs/SEMANTIC_ROLE_TOKENS.md).
  - **Additive and no-regression:** brand/vibe-preset application injects role variables as inline `:root` styles that win over the stylesheet layer, so **presets are unaffected by construction**. The decorative/state/accent roles (`--gds-brand-accent`, `--gds-state-*`, `--gds-focus-ring`, `--gds-badge-*`, …) are deliberately **left undefined** at the default layer so their hue stays a brand/preset decision rather than a fixed default.
  - **Role-misuse fix:** `BottomTabBar`'s top border read `--gds-text-secondary` (a text role) for a border; it now reads `--gds-border-card`, so the new governed `text-secondary` value cannot darken that hairline. `THEME_GOVERNANCE.md` and `README.md` updated.

## 3.14.8 - 2026-07-26 — Consumer WCAG contrast checker

- **Consumer contrast checker** (#453): two additive, pure, **server-safe** exports in `@sovereignsquad/gds-theme` (root, `/server`, and `/client` entries) surface the same WCAG 2.x contrast math GDS hard-gates its own tokens with (`verify:token-contrast-scoring`), so consumers can score **their own** brand/custom color pairs before shipping instead of re-implementing the formula:
  - **`getGdsContrastRatio(foreground, background)`** — returns the WCAG contrast ratio (1–21, 2-dp), accepting `#hex` (3-/6-digit), `rgb()`, and `rgba()`; a translucent foreground is composited over the background first so the scored color is the one a user sees. Throws on an unparseable color.
  - **`checkGdsContrast(foreground, background, options?)`** — checks a pair against a chosen WCAG threshold and reports `{ ratio, required, passes, level, size }`. Defaults to the GDS baseline **AA / normal (4.5:1)**; `level` (`'AA'`|`'AAA'`) and `size` (`'normal'`|`'large'`) select the 4.5 / 3 / 7 / 4.5 thresholds.

  No React, no DOM — safe in a Server Component, route handler, or build script. Documented in [`docs/CONTRAST_CHECKER.md`](docs/CONTRAST_CHECKER.md). This is the consumer-facing checker slice of #453; the broader theme-builder/preset work on that issue remains open.

## 3.14.7 - 2026-07-26 — Screen-organized case studies + recorded foldable decision; JSDoc backfill batch

- **Screen-organized case studies + recorded foldable decision** (#459): new [`docs/CASE_STUDIES_BY_SCREEN.md`](docs/CASE_STUDIES_BY_SCREEN.md) walks three screen types — a list-detail admin (operational shell + `GdsSplit`/`DetailProfileShell`), a public discovery surface (`PublicShell` + `BottomTabBar` + a supporting pane), and a kiosk/large-screen lane — each composed only from the canonical layout templates and walked across the named size classes (`compact`→`xlarge`), complementing the existing migration/adoption case study. The **foldable / dual-screen build-or-not decision** is recorded: **defer** — a single-window fold reads as a normal responsive size-class boundary today, and CSS `viewport-segments` is too narrowly supported to gate on — with a re-evaluation trigger when `viewport-segments` reaches baseline browser support; multi-window/Window Management orchestration stays a permanent non-goal. `RESPONSIVE_AND_PLATFORM_GUIDANCE.md` and `README.md` updated. Docs-only.
- **JSDoc backfill — batch** (#414): added interface- and per-prop JSDoc to `FormFieldProps`, `ActionBarProps`/`ActionBarAction`/`ActionBarIconAction`, and `ReferenceSectionProps`, so consumer editors surface field-level docs on hover for these primitives. Docs-only (JSDoc → `.d.ts`), no behavior change. #414 remains open as the phased backfill tracker.

## 3.14.6 - 2026-07-26 — PWA thin build: web-app-manifest generator, standalone detection, safe-area tokens

Delivers the standards-based PWA pieces scoped in #455 (PWA = *partial build*), keeping GDS a component/theme library rather than an app framework.

- **PWA thin build** (#458): three additive, tree-shakeable helpers in `@sovereignsquad/gds-theme` (no breaking changes):
  - **`getGdsWebAppManifest(options)`** — server-safe generator returning a valid, spec-shaped W3C web-app-manifest object from GDS theme/brand inputs, so the manifest's `theme_color`/`background_color` stay aligned to the active theme instead of a hand-maintained duplicate. Required `name`/`themeColor`/`backgroundColor` (throws otherwise); defaults `display: 'standalone'`, `start_url`/`scope`/`id: '/'`, `short_name` falls back to `name`. GDS does not serve the manifest — consumers serialize the result to their `manifest.webmanifest` (e.g. a Next.js `app/manifest.ts`). Exported from the root, `/server`, and `/client` entries.
  - **`useGdsStandaloneDisplayMode()`** — SSR-safe client hook reporting whether the app runs as an installed PWA and its current `display-mode` (`standalone`/`fullscreen`/`minimal-ui`/`browser`), updating live on mode change; detects the `display-mode` media features plus iOS Safari's legacy `navigator.standalone`. Exported from the root and `/client` entries.
  - **`gdsSafeAreaInset`** + **`--gds-safe-area-inset-{top,right,bottom,left}`** — governed safe-area inset custom properties in `styles.css` (each `env(safe-area-inset-*, 0px)`, resolving to `0px` on non-notched displays) exposed as ready-to-use `var(...)` strings, so shells/consumers read one inset source instead of hard-coding `env(safe-area-inset-*)`.

  **Explicit non-goals (unchanged):** service-worker/offline caching and an install-prompt UX framework — application-architecture concerns owned by the consuming app; GDS documents the integration point only. Documented in `docs/PWA_VIEWPORT_POLICY.md` and `docs/RESPONSIVE_AND_PLATFORM_GUIDANCE.md`.

## 3.14.5 - 2026-07-26 — Meta-text-on-page contrast hard-gated (every lane clears WCAG AA)

- **Meta-on-page contrast nudged to WCAG AA and promoted to a hard gate** (#460): eight expressive light lanes (`dark-public`, `editorial`, `sunset`, `ruby`, `skyline`, `coral`, `orchid`, `royal`) shared the default `mutedLight` (`#64748b`), which on their tinted light canvases produced meta/muted-text-on-page contrast of **4.26–4.48** — just under 4.5:1. `verify:token-contrast-scoring` previously reported these as non-blocking *advisories*. Those lanes now carry a slightly darker `mutedLight` (`#5f6d82`) — a minimal nudge that clears 4.5:1 on every one of those canvases (worst case 4.70) while keeping meta-on-card comfortably above AA (≥5.2) — and the `muted`-on-`canvas` pair is **promoted from advisory to a hard release gate**. The gate now hard-asserts **184 readable-text fg/bg pairs across 23 themes** at WCAG AA 4.5:1 with an empty advisory tier. Only `mutedLight` in light mode changed (dark mode, body text, and the lanes that were already ≥4.5 are untouched); the DTCG token export (`tokens/gds.tokens.json`) is regenerated to match, and `VPAT_CONFORMANCE.md` is updated.

## 3.14.4 - 2026-07-26 — Theme Lab result cards re-theme like a built-in theme + active-preset indicator

Bug fix for the Theme Lab (`/themes`, `ReferenceThemeExplorer`), reported as "Theme Lab ruins the page."

- **Theme Lab control/result cards now re-theme their own backgrounds like any built-in theme** (#461): the three primary control cards ("Theme preset", "Brand builder options", "Current selection summary") were wrapped in a bespoke *owned-contrast* surface (`role: 'theme-lab-controls'`) built from an internally contradictory token set — a **dark** `surfaceDark` gradient `background`/`backgroundColor` combined with a **light** `surface`. Under the global `html[data-gds-theme-preset] [data-gds-owned-contrast]` `!important` rule this painted the cards as **dark boxes on an otherwise light page** for every light preset (amber, cosmic, editorial, …), inconsistent with the rest of the page and, per the report, unreadable. The override is removed: those cards are now plain `.gds-paper` surfaces that re-theme **both background and text** through the same `html[data-gds-theme-preset] .gds-paper` rule every other card uses — readable in light and dark across all presets, exactly "as any built-in theme." Owned contrast stays reserved for the intentional vibe *swatch* surfaces (the shipped-lane gallery, the VibeTheme contract, and the Athlete Gold reference), whose job is to preview a specific vibe atmosphere rather than match the page. The now-unused `theme-lab-controls` value is retired from the `GdsOwnedContrastRole` public union.
- **Active-preset "Selected" indicator** (#461): the Theme Lab control panel now shows a clear `Selected: <preset>` badge on both the preset picker and the current-selection summary, so an active Theme Lab preset is visibly labelled (previously there was no active-state affordance in the control area).
- **Gates updated at the source** (#461): `verify-owned-contrast-compliance.mjs` and `verify-theme-trust-runtime.mjs` previously *required* the `theme-lab-controls` owned surface (they had codified the buggy behavior). They now assert the opposite — the control cards must **not** carry a bespoke owned-contrast surface, must share the page's global `.gds-paper` background, and must render exactly two visible "Selected" indicators — and the retired role is guarded against reintroduction. The `core.test.tsx` explorer test was updated to match. No API change beyond the retired role literal; only the Theme Lab's own rendering is affected.

## 3.14.3 - 2026-07-25 — Shared `gdsBreakpointByAlias` size-class helper

- **`gdsBreakpointByAlias` public helper** (#457): the breakpoint alias→width map (`{ xs: 36em, sm: 48em, md: 62em, lg: 75em, xl: 88em }`) was duplicated inside `KanbanBoard` (`useGdsKanbanOrientation`) and `DiscoveryShell`. It is now a single exported `gdsBreakpointByAlias` from `@sovereignsquad/gds-core`, consumed by both — one source of truth aligned with the named size-class vocabulary in `docs/RESPONSIVE_AND_PLATFORM_GUIDANCE.md`. Additive and backward compatible: existing behavior is unchanged; consumers building custom responsive chrome can now resolve the same governed widths instead of hard-coding pixel breakpoints. `verify:token-contrast-scoring` (#456) also joined the release chain in this line — real per-token-pair WCAG scoring of the readable-text fg/bg pairs across all shipped themes — though it added no consumer-facing behavior.

## 3.14.2 - 2026-07-25 — Forced-colors hardening for themed surfaces + widened forced-colors gate coverage (3.14.0 quality follow-up)

Quality follow-up to the 3.14.0 primitives (epic #440, group A). This closes the theme half of #445 by fixing a real forced-colors accessibility bug that the widened gate coverage surfaced.

- **Forced-colors: themed surfaces/controls no longer leak decorative gradients** (#445): the vibe/brand theme presets (`cosmic`, `neon-night`, `sunset`, …) paint `!important` gradient `background`/`background-image` on Papers, cards, buttons, badges, and the app shell using preset-scoped selectors carrying **two** attributes (`[data-mantine-color-scheme][data-gds-theme-preset='…']`, plus `:not()` clauses for buttons). `@sovereignsquad/gds-theme/styles.css` already had a `@media (forced-colors: active)` reset, but its `html[data-gds-theme-preset] …` selectors carry only **one** attribute, so those preset rules out-specified it and their gradients survived into forced-colors mode — a genuine contrast bug (a forced-colors user could get unreadable text over a gradient). Added a forced-colors **specificity backstop** that re-applies the neutralization (surfaces → `Canvas`, controls → `ButtonFace`/`Highlight`, badges → `Mark`, `background-image: none`, decorative `body::before`/`::after` hidden) with a never-matching `:not(#gds-never)` id guard, which contributes an ID's specificity weight so it outranks every preset rule regardless of attribute/`:not()` count — in any theme lane, present or future. No API change; only forced-colors rendering of the expressive theme lanes is affected (the default/dark lanes were already correct).
- **Widened forced-colors gate coverage** (#445): `scripts/verify-forced-colors-runtime.mjs` now drives route coverage off the pattern-catalog families that mount the 3.14.0 components — it visits `/patterns/operations` (Kanban) and `/patterns/foundations` (Forms), which the old fixed 4-route list never did — and exercises the two new-component routes across **8** theme presets (well beyond the previous 3), spanning neutral, dark, flat-surface, editorial, brand-discovery, high-saturation vibe (`cosmic`/`neon-night`), and warm lanes. It adds targeted per-component checks for the Kanban collapse toggle + column footer and the schema form's checkbox-group + repeatable rows, so a vanished-in-forced-colors or decorative-background regression on those specific controls fails `verify:release`. This widened sweep is what caught the vibe-theme gradient leak above; CI-only, no package impact.

## 3.14.1 - 2026-07-25 — GdsSchemaForm i18n + themed checkbox-group (3.14.0 quality follow-up)

Quality follow-up to the 3.14.0 primitives (epic #440, group A):

- **Themed checkbox-group** (#444): `GdsSchemaForm`'s `checkbox-group` now renders governed Mantine `Checkbox` controls instead of a raw native `<input type="checkbox">`, so the checkboxes inherit the GDS theme and the governed forced-colors remap like every other control (closing a real light/dark and forced-colors gap for that field). No API change.
- **GdsSchemaForm i18n** (#443): the previously hardcoded-English user-facing strings — the `repeatable` row-count live announcements (`Row added/removed, N rows.`), the checkbox-group/repeatable validation messages (`requires at least one selection.`, `has a row with a missing required field.`, min/max rows), the shared `is required.` message, the default add/remove-row button labels, and the `(required)` marker — now route through `useGdsTranslation()` with new `gds.schemaForm.*` keys added across all **12** locale packs (English output unchanged). Added a test for the `repeatable` aria-live announcement. Consumer `addRowLabel`/`removeRowLabel` overrides still win. (Number-embedded pre-existing validation messages for `minLength`/`maxLength`/`pattern`/`email`/`number` remain a tracked follow-up under #443.)
- **Docs** (#446): `llms.txt` updated to describe the 3.14.0 Kanban props, the `GdsSchemaForm` `checkbox-group`/`repeatable` field types, and the opt-in `dates.css`; `docs/SCHEMA_FORMS.md` notes the themed checkbox + localized messages.

## 3.14.0 - 2026-07-25 — Kanban server-pagination/footer/collapsible, opt-in date CSS, GdsSchemaForm primitives, label-based issue board

- **KanbanColumn server-paginated count** (#432): the header count badge now renders `column.totalCount` when set, falling back to `column.items.length`. Server-paginated columns (where `items` hold only the loaded page) can show their real total instead of the loaded-page count. Additive and backward compatible — omitting `totalCount` is unchanged. `COMPONENTS_AND_PATTERNS.md` updated.
- **KanbanColumnData.title accepts ReactNode** (#434): column headings can now be a `ReactNode` (icon + label, colored dot, custom count pill), matching `KanbanItem.title`. Set the new `KanbanColumnData.ariaLabel` when `title` is not a plain string so move-menu targets and drag announcements keep a meaningful accessible name (string titles need nothing). Backward compatible.
- **KanbanColumn footer slot** (#435): new `footer` / `renderFooter(column)` on `KanbanColumn`, and `renderColumnFooter(column)` on `KanbanBoard`, render an element below the card list (pagination / "load more" / per-column actions) outside the drag `SortableContext`. Additive.
- **Collapsible KanbanColumn** (#436): opt-in `collapsible` (off by default) renders a header disclosure toggle (`button` with `aria-expanded`/`aria-controls`) that folds a column body to its title + count; a collapsed column is not a drop target. Controllable board-wide via `collapsedColumnIds` + `onCollapsedChange(columnId, collapsed)` or per-column via `collapsed` + `onCollapsedChange(collapsed)`. New localized `gds.kanban.collapseColumn`/`expandColumn` strings across all 12 locales. Mirrors the `enableDrag` opt-in pattern — zero behavior change on upgrade.
- **gds-theme date-component CSS is now opt-in** (#433): `@sovereignsquad/gds-theme/styles.css` no longer unconditionally `@import`s `@mantine/dates/styles.css`. The Mantine dates stylesheet moved to a separate `@sovereignsquad/gds-theme/dates.css` export, imported only by consumers who render a GDS date component (`GdsDateInput`/`GdsDateTimeInput`/`GdsDateRangeInput`, or a `GdsSchemaForm` `date` field). Consumers who use no date component now need neither `@mantine/dates` nor `dayjs` and no longer hit a build-time "Module not found: @mantine/dates/styles.css". `@mantine/dates` and `dayjs` remain required peers of `gds-core` (where the date components live and are imported in JS); `gds-theme` deliberately does **not** declare them — it ships `dates.css` as an opt-in CSS `@import` only, not a JS dependency, so a consumer that loads `dates.css` already has them via `gds-core`. **Migration for date-component users:** add `import '@sovereignsquad/gds-theme/dates.css';` alongside your existing `styles.css` import. The playground adds it; `INSTALLATION_GUIDE.md` and `COMPONENTS_AND_PATTERNS.md` document it.
- **GdsSchemaForm checkbox-group + repeatable field primitives** (#437): two new `GdsSchemaFieldType`s. `checkbox-group` renders a grouped multi-select as a `fieldset` of checkboxes (value `string[]`; `required` = at least one), distinct from the single `boolean` checkbox and the `select` dropdown; a JSON Schema `enum` opts into it with `x-gds-fieldType: 'checkbox-group'`. `repeatable` is an "add another row of N fields" primitive — the descriptor carries nested `fields`, the value is an array of row objects, with governed add/remove controls, `minRows`/`maxRows` bounds, per-row required-sub-field validation, row-context button labels, focus management on add/remove, and an `aria-live` row-count announcement. Both flow through the `renderers` override map and `onEvent`. Additive (existing forms unaffected); no new i18n keys (labels are descriptor-driven, matching the component's existing string handling). Documented in `docs/SCHEMA_FORMS.md`; admin CRUD tutorial updated.
- **JSDoc backfill — batch 1** (#414): added export-level JSDoc to a coherent batch of compact gds-core primitives — the selection-chip family (`PillBar`, `SoftChipGroup`, `FilterChipGroup`, `GdsSelectionOption`, `GdsSelectionGroupProps`), `FitScoreChip`, `NumberStepper`, and `ListingCard` (with `ListingCardProps`, `ListingMetadataRow`, `ListingCardAffordance`, `ListingCardMediaRatio`, `MAX_LISTING_CARD_ACTIONS`). Docs-only, no behavior change. #414 remains open as the phased tracker for the remaining public-export backfill.
- **Project board moved from Projects v2 to a label-based issue board** (#438, supersedes #431): the board is now **GitHub Issues grouped by `status:` labels** (`PROJECT_BOARD.md`), not an org-level Projects v2 board. Every board operation is a label change the ambient `GITHUB_TOKEN` can perform, so the `GDS_PROJECT_TOKEN` PAT requirement is gone — the fragile part that could not be managed from the maintainer's agent/mobile workflow and drifted after each release. New taxonomy SSOT (`scripts/board-labels.config.mjs`: `status:`/`priority:`/`area:` labels), a provisioner (`npm run board:labels`, `gh label create --force`), a rewritten label audit (`audit:board` non-strict inside `verify:release` so a missing `gh` never blocks a release; `audit:board:strict` fails when an open issue isn't in exactly one status column), and a label-based `board:sync-release` (closing a delivered issue is its "move to Done"). The retired Projects v2 scripts (`audit-project-board.mjs`, `complete-3-4-board.mjs`, `sync-hvb-board.mjs`) and their npm entries are removed; `.github/workflows/board-sync.yml` now provisions labels and runs the strict audit with the default token. Docs updated: `PROJECT_BOARD.md` (new), `RELEASE_PUBLISH.md`, `docs/BOARD_SYNC_CHECKLIST.md`, `README.md`.
- **Dependency & CI hardening** (#439): patched the newly-disclosed high-severity `brace-expansion` DoS advisory (**GHSA-mh99-v99m-4gvg**, `<= 5.0.7`) by moving the single tree entry to `5.0.8` within `minimatch`'s existing `^5.0.0` range (`npm update brace-expansion`, no override) — production audit is clean again. Also fixed the `quality.yml` "Override to Mantine 9" CI step, which the #433 work had tipped into an ERESOLVE: removing the incorrect `@mantine/dates`/`dayjs` peer declaration from `gds-theme` (its JS never imports them — they belong on `gds-core`) restored the Mantine-9 resolution while keeping the opt-in `dates.css`.

## 3.13.0 - 2026-07-24 — React 19 runtime, react-router 8, Kanban generics/affordance, dev diagnostics, DX docs

- **Workspace runtime upgraded to React 19; react-router advisory remediated** (#430): bumped the dev/app React runtime from 18.3.1 to 19.2.7 across the reference apps and package dev/test tooling, and migrated the playground from `react-router-dom@7` to `react-router@8` (the DOM bindings merged into `react-router` in v7+). This clears the high-severity **GHSA-qwww-vcr4-c8h2** production advisory (`react-router` 7.12.0–8.2.0; react-router 8 peer-requires React ≥19.2.7, which is why the React bump was needed). **The published peer contract is unchanged** — `react`/`react-dom` peers stay `^18.2.0 || ^19.0.0`, so React 18 remains a fully supported consumer lane (still validated by `verify:mantine`'s Mantine 7 + React 18 consumer-install smoke); only the workspace's own dev/CI runtime moved to 19. No GDS component code changed (the codebase was already React-19-clean: `createRoot`, no removed APIs, `@types/react` already 19; Mantine 7.17.8 already declares a React 19 peer). `compatibility.matrix.json` now labels React 19 + Mantine 7 as the primary CI/workspace line. A newly-disclosed dev-only PostCSS source-map advisory (**GHSA-r28c-9q8g-f849**, nested in `next`/`tsup`, non-shipped) — previously masked by the react-router production failure — is documented as an accepted dev advisory in `DEPENDENCY_AUDIT.md`, matching the identical `GHSA-6g55-p6wh-862q` disposition.
- **KanbanCard move-menu icon no longer implies drag** (#429): the "move to column" menu trigger defaulted to `IconArrowsMove` (a 4-way-arrows glyph) whenever `onMoveItem` was set — independent of `enableDrag` — which universally reads as "grab and drag me." But that control is a tap-to-open destination menu, and real pointer/touch drag is gated behind `enableDrag` (a separate grip handle), so on touch the icon promised a free-drag it never performed. The default is now a new governed `More` (vertical-dots) glyph (`IconDotsVertical`) — the standard "tap to open a menu" affordance with zero drag implication — used whether `enableDrag` is on or off. New optional `moveMenuIcon?: ReactNode` and `moveMenuLabel?: string` props on `KanbanBoard`/`KanbanColumn`/`KanbanCard` let consumers override the trigger's icon/verb without losing the governed menu. Backward compatible (icon swap + additive props); the accessible `"Move: {name}"` label and menu behavior are unchanged. Also adds the reusable `GdsIcons.More` kebab icon. See the "Move-menu icon vs. drag-handle icon" note in [`COMPONENTS_AND_PATTERNS.md`](COMPONENTS_AND_PATTERNS.md).
- **`GdsPageTemplateAction.loading` (deprecates `pending`)** (#405): the page-template action busy-state prop is now named `loading`, matching every other GDS action/button API (`SemanticButton`, `ConfirmDialog`, `GdsAccessGateAction`, table row actions) — the previous outlier `pending` name forced consumers to remember which API used which. Backward compatible: `pending` is still honored as an alias (mapped to `loading`) with a one-time dev-only deprecation warning, and will be removed in a future major version; if both are set, `loading` wins. No repo consumer used `pending`, so blast radius is limited to the deprecation path. `docs/PAGE_TEMPLATES.md` updated.
- **Dev-mode misuse diagnostics** (#404): new dev-only `gdsDevWarnOnce` helper (`@sovereignsquad/gds-theme`, with `resetGdsDevWarnings` for test isolation) surfaces three previously-silent misuse gaps as one-time `console.warn` messages, stripped entirely from production builds (`process.env.NODE_ENV === 'production'` guard). Wired into: (1) `useGdsTranslation` — warns when called without a `GdsProvider` ancestor (detected by context-identity), which otherwise silently pins every string to built-in English; (2) `GdsDateInput`/`GdsDateTimeInput`/`GdsDateRangeInput` — warn on transposed `minDate`/`maxDate` or an out-of-range `value`; (3) `GdsAccessGate` — now routes contract-validation findings through the warning in addition to the optional `onEvent`, so an invalid state/reason/action combination is no longer completely silent when `onEvent` is omitted. Additive and non-breaking: correct usage sees zero behavior change and zero production cost; only incorrect usage gains a dev-time signal. This complements — never replaces — GDS's existing fail-loud `throw` for hard contract breaks.
- **Generic `KanbanBoard` item/column typing** (#399): `KanbanBoard`, `KanbanColumn`, `KanbanCard`, and their prop interfaces are now generic over the item and column shape — `KanbanBoard<TItem extends KanbanItem, TColumn extends KanbanColumnData<TItem>>`, with `KanbanColumnData<TItem extends KanbanItem = KanbanItem>` carrying `items: TItem[]`. Both parameters default to the base `KanbanItem` / `KanbanColumnData`, so this is a **backward-compatible typing enhancement with no runtime behavior change** — existing non-generic call sites (including the playground's `KanbanBoardDemo` and all prior tests) compile unchanged. Consumers who extend `KanbanItem` / `KanbanColumnData` with app-specific required fields now receive them fully typed inside `renderItem(item, column)` **without a cast** (previously the fixed `(KanbanItem, KanbanColumnData)` callback signature made an extended-shape `renderItem` a compile error at the call site — surfaced by a real downstream consumer build). `onMoveItem` keeps its string-id signature and no `@dnd-kit` types leak onto the public surface (`verify:boundary` unaffected). See the "Typed item/column extension" note in [`COMPONENTS_AND_PATTERNS.md`](COMPONENTS_AND_PATTERNS.md).

## 3.12.0 - 2026-07-23 — competitive gap-closing batch (#387-#398)

Following `DESIGN_SYSTEM_COMPETITIVE_GAP_ANALYSIS.md`'s benchmark against Material Design 3, Fluent UI 2, IBM Carbon, Ant Design 5, Shopify Polaris, Adobe Spectrum, Atlassian Design System, and Chakra UI, this release closes every P0/P1/P2 gap identified plus three incidental tech-debt items, in one consolidated batch.

**P0 — highest-impact gaps:**
- **Date/time picker family** (#389): `GdsDateInput`, `GdsDateTimeInput`, `GdsDateRangeInput` wrap `@mantine/dates` (new peer dependency, matching the `@mantine/core` engine class). `GdsSchemaForm`'s `date` field type now renders `GdsDateInput` instead of a bare native `<input type="date">`, keeping its stored value as an ISO (`yyyy-mm-dd`) string for backward compatibility.
- **`GdsBreadcrumbs`** (#390): standalone, independently reusable breadcrumb trail (labeled `nav` landmark; last item always renders as the non-link current page even if it carries an `href`). `DocsPageShell` now uses it internally instead of an inline, duplicated implementation.
- **z-index token scale** (#391): `gdsZIndexToken` (`@sovereignsquad/gds-theme`) documents and defers to Mantine's own `--mantine-z-index-*` CSS variable scale as the single stacking authority. Fixed two real ad hoc violations found during implementation — `BottomTabBar` and `FloatingActionPlacement` each independently hardcoded different arbitrary z-index values (200 and 20) with no shared authority; both now use `gdsZIndexToken.app`.

**P1 — real gaps, narrower blast radius:**
- **`GdsRichTextEditor`** (#392): Tiptap-backed rich text editor (user-confirmed dependency choice, matching the `@dnd-kit` precedent's reasoning), composed into `ContentOpsEditor`'s demo as the description field. Fully encapsulated behind a dedicated `@sovereignsquad/gds-core/rich-text-editor` subpath export — kept out of the main package entry so its larger "Content engine" dependency stays genuinely opt-in (confirmed: `apps/reference-vite`'s own vendor chunk stayed at its pre-change size, since it never imports the subpath).
- **Global density-mode primitive** (#393): `GdsDensityProvider`/`useGdsDensity` publish a `compact`/`comfortable`/`spacious` axis products can set once, plus `useGdsCardContract()` documenting the fall-back-to-ambient-density extension pattern. Purely additive — no existing component's default changed.
- **`GdsColumnGrid`/`GdsColumnGridItem`** (#394): named 12-column (configurable) track-span grid, matching Carbon's 2x Grid / Ant Design's 24-col `Grid`, complementing `GdsGrid`'s equal-width auto-column layouts.
- **Overlay elevation scale** (#395): `Popover.defaultProps.shadow` is now explicitly `'md'`, cascading to Menu/HoverCard/Select-family dropdowns. `Card`'s existing `shadow: 'sm'` default is untouched. `Modal` has no theme-configurable shadow prop in this Mantine version, documented as a real constraint rather than left silently unaddressed.
- **CJK locale coverage** (#396): `zh` (Simplified Chinese), `ja`, `ko` message locales ship with full parity (168 keys each) and correct `direction`/`script` metadata. Machine-translated via the same disclosed Google Translate approach already used for playground site-phrases — **not reviewed by a native speaker**; flagged for review before treating as production-quality.

**P2 — lower urgency:**
- **Icon catalog expansion** (#397): ~40 new semantic icon keys (navigation, commerce, security, rich-text-editor toolbar, plus location/building/folder/archive/connectivity/flag/tool/phone/drag-handle).
- **Financial/network chart types** (#398): `candlestick` (OHLC) and `sankey` (flow) ship as a new governed "Set C" (`gdsChartSetCTypeRegistry`), with their own validation rules (OHLC high/low range containment; sankey source/target/non-negative-flow).

**Incidental tech debt fixed:**
- **Locale-metadata drift** (#387): `es` (Spanish) shipped full messages in `gds-core/locales` but was missing from `gds-theme`'s `gdsLocaleMetadata` (RTL/script-detection registry), silently mis-defaulting to English direction/script rules. Fixed, plus a new parity test guards against recurrence. `GDS_GAP_INVENTORY.md`'s stale "not covered" claims (charts, uploads, command palette, evidence panels) were also corrected.
- **Vibe-theme/preset drift guard** (#388): the 23 `GdsThemePresetId` entries each need a Mantine theme override (`theme-presets.ts`) and an independently hand-authored CSS "vibe theme" object (`vibe-themes.ts`). Confirmed the two intentionally draw from different color sources (Mantine's functional ramp vs. a bespoke, more saturated palette), so merging them isn't a safe mechanical refactor — added `vibe-themes.test.ts` instead, which fails CI if a preset id is ever added to one file without the other.

See `DESIGN_SYSTEM_COMPETITIVE_GAP_ANALYSIS.md` for the full comparison this batch was scoped against.

## 3.11.1 - 2026-07-23 — release-cutover patch (no functional change)

- **Patch-only version bump** (#385): `gds-v3.11.0`'s git tag was created by `auto-tag-release.yml` before that same day's fix to the workflow (explicit `gh workflow run --ref` dispatch, added to work around GitHub Actions' anti-recursion rule for `GITHUB_TOKEN`-authored pushes) landed, so the tag was stuck pointing at the pre-fix commit and `release-bundles.yml` never created a GitHub Release/tarball for it. Moving the existing tag directly was attempted and blocked (git proxy `403` on both tag deletion and force-push, no GitHub-API tag-ref tool available). This patch bump carries no functional/code change beyond the version-bump surfaces themselves — it exists solely so the now-fixed automated pipeline produces a correctly-tagged `gds-v3.11.1` release and GitHub Packages publish with zero manual tag surgery.

## 3.11.0 - 2026-07-23 — mobile input-zoom guard + accessible Kanban drag-and-drop

- **Mobile input-focus auto-zoom guard** (#379, #380): `gdsTheme` now floors the effective font-size of every Mantine `Input`-based control (`TextInput`, `Textarea`, `NativeSelect`, `Select`, `PasswordInput`, `NumberInput`, `MultiSelect`, `Autocomplete`, `TagsInput` — including `gds-admin`'s `AdminTextInput`/`AdminTextarea`/`AdminSelect`, which are thin pass-throughs) to at least 16px at `xs`/`sm`/the implicit default size, via a new `components.Input.vars` entry in `packages/gds-theme/src/theme.ts` using `max(1rem, var(--mantine-font-size-sm))`. This prevents iOS Safari's/Chrome's forced page-zoom on input focus — a distinct mechanism from pinch-zoom, unaffected by `getGdsPwaViewportMetaContent`/`zoomPolicy` — without touching pinch-zoom or OS-level text scaling, so it ships as a silent default rather than an opt-in flag. `md`/`lg`/`xl` sizes (already ≥16px) are unchanged. `GdsSchemaForm`'s `renderDefaultField` raw-native-element fallback (no Mantine class, so the theme fix can't reach it) gets its own matching inline font-size floor. **Visual diff for existing consumers on upgrade:** any `xs`/`sm`/default-size input renders measurably larger text (12–14px → 16px) with no code change required. See the new "Input-focus auto-zoom" section in [`docs/PWA_VIEWPORT_POLICY.md`](docs/PWA_VIEWPORT_POLICY.md).
- **Accessible Kanban pointer/touch/keyboard drag-and-drop** (#381–#384): `KanbanBoard` gains an opt-in `enableDrag?: boolean` prop (default `false` — zero behavior change for existing consumers). When enabled, each `KanbanCard` gets an additional drag handle built on `@dnd-kit/core` + `@dnd-kit/sortable` (new dependencies of `@sovereignsquad/gds-core`, fully encapsulated — never a public export or consumer import, see the new "Interaction engine" class in [`DEPENDENCY_GOVERNANCE.md`](DEPENDENCY_GOVERNANCE.md)), supporting pointer, touch, and keyboard (Space to pick up, arrows to move, Space to drop, Escape to cancel) dragging with localized live-region announcements (new `gds.kanban.drag*`/`announce*` message keys across all 9 locale packs). Native HTML5 `draggable`/`dragstart` is still never used (it remains inoperable by keyboard/screen-reader users and is prohibited by the accessibility release gate) — the existing keyboard-accessible "Move to column" menu stays unconditionally rendered and fully functional in both modes, and is the guaranteed accessible-equivalent fallback. `onMoveItem`'s signature widens with an optional trailing `toIndex` parameter (backward compatible — existing 3-argument call sites are unaffected). New `scripts/verify-kanban-drag-accessibility-runtime.mjs` (wired into `verify:release`) asserts the Move menu and drag handle both keep working with `enableDrag` on.

## 3.10.0 - 2026-07-22 — kanban + media + PWA viewport lane

- **Kanban Board pattern**: new governed responsive `KanbanBoard` (`KanbanColumn`/`KanbanCard` sub-parts, `useGdsKanbanOrientation` hook) in `@sovereignsquad/gds-core`. Portrait-mobile viewports render one stacked column per row; landscape phones, tablets, and desktop render multi-column with horizontal scroll — resolved automatically via a `useMediaQuery`-backed hook, no consumer CSS or breakpoint logic required. Cards move between columns through a keyboard-accessible "move to column" menu (`onMoveItem(itemId, fromColumnId, toColumnId)`) instead of native HTML5 drag-and-drop, which cannot be operated by keyboard or screen-reader users. Registered in the pattern catalog (`kanban-board`, operations family) with a live demo, i18n keys (`gds.kanban.*`) added to all 9 locale packs, and a new `docs`/`COMPONENTS_AND_PATTERNS.md` "Kanban Board Rules" section.
- **`MediaPreviewCard` / `AdminResourceCard` missing-media handling**: added `hideWhenNoMedia?: boolean` to `MediaPreviewCard` (and threaded through `AdminResourceCard`/`AdminResourceGrid`/`AdminResourceManager`) so records with no `src`/`thumbnailSrc` can omit the media area entirely instead of showing the default "No media" placeholder. Missing media already never crashed the card; this adds explicit opt-in control for record types that structurally never have media (e.g. lead/contact rows).
- **PWA viewport & zoom policy**: new `getGdsPwaViewportMetaContent(...)` export from `@sovereignsquad/gds-theme` generates the canonical `<meta name="viewport">` content. Default `zoomPolicy: 'browser-default'` leaves pinch-zoom untouched (required for WCAG 1.4.4/1.4.10); the new `'app-shell-fixed'` lane adds `maximum-scale=1, user-scalable=no` as a reviewed, documented, opt-in exception for installed PWA app shells. See [`docs/PWA_VIEWPORT_POLICY.md`](docs/PWA_VIEWPORT_POLICY.md) for the required accessibility mitigations, scope limits, and exit condition.
- **Fully automatic release cutover**: new `.github/workflows/auto-tag-release.yml` triggers on any push to `main` that changes the root `VERSION` file, and creates/pushes the matching `gds-v<VERSION>` tag using the workflow's own `GITHUB_TOKEN` — no maintainer needs to run `git tag`/`git push` or draft a release in the GitHub web UI for a routine version bump. That tag push fans out into the existing `release-bundles.yml` (build, `verify:release`, pack, create the GitHub Release, attach tarballs) and `publish-npm.yml`, which now also triggers on the tag push (previously `workflow_dispatch`-only) instead of requiring a manual run. See the updated "GitHub Actions publish path" section in [`RELEASE_PUBLISH.md`](RELEASE_PUBLISH.md).
- **GitHub Packages distribution channel**: new `.github/workflows/publish-github-packages.yml` publishes all seven packages to GitHub Packages' npm-compatible registry (`https://npm.pkg.github.com`) on the same `gds-v*` tag trigger, authenticated with the workflow run's own ambient `GITHUB_TOKEN` — no `NPM_TOKEN`/npm.com account dependency, so it keeps working even when the npmjs.com publish is blocked. Unlike the release-bundle tarball fallback, it's a real resolving registry, so the `@sovereignsquad/gds` umbrella package installs correctly there too. `scripts/publish-packages.mjs` and `scripts/check-registry-publication.mjs` already supported a configurable `GDS_NPM_REGISTRY`, so no script changes were needed — only the `.npmrc`/registry-URL wiring in the new workflow and consumer install docs (README.md, `INSTALLATION_GUIDE.md` §9, `RELEASE_PUBLISH.md`).

## 3.9.0 - 2026-07-01 — brand-completion-lane (#362–#368)

- **Choice-chip family catalog coverage** (#362): the `choice-chips` catalog demo now mounts the full exported family — `ChoiceChip` (static, link/button, multi-select toggles) plus the stateful `PillBar`, `SoftChipGroup`, and `FilterChipGroup` selection groups (including a disabled option) — so every-theme render, forced-colors, and a11y-evidence gates exercise the whole family, not just the base chip.
- **MissingDataPrompt catalog coverage + i18n catalog keys** (#363): the `state-blocks` catalog demo renders `MissingDataPrompt` with a required-fields list, a call-to-action slot, and a `role="status"` recovery variant. Localized message keys `gds.state.missingData.title` / `gds.state.missingData.description` are added to all 9 locale packs (public catalog). `MissingDataPrompt` stays server-safe (no `'use client'`, no context hook) with English semantic defaults, so consumers localize by passing `title`/`description` resolved from the catalog — the component remains usable from `@sovereignsquad/gds-core/server`.
- **Form control family catalog coverage + i18n catalog keys** (#364): the `forms` and `form-field` catalog demos now mount `GdsSegmentedControl` (default, disabled, and many-item overflow), `GdsSlider` (1–10 plus a min-equals-max boundary), `GdsRatingScale` (1–5), and `GdsWizardStepper` (first/mid/last steps). Localized wizard/slider/rating message keys (`gds.form.wizard.back|next|finish|progress|optional`, `gds.form.slider.label`, `gds.form.rating.aria`) are added to all 9 packs (public catalog). The controls stay server-safe with English prop defaults; no public API changes.
- **Overlay Dialog/SidePanel alias coverage + i18n** (#365): the `modals` and `drawers` catalog demos now open `GdsDialog` and `GdsSidePanel` (open/close, focus trap, focus return, opaque surface). The overlay close-button `aria-label` now routes through the new `gds.overlay.close` key (added to all 9 packs) via the client-only `OverlayManager`, instead of Mantine's built-in English default.
- **Chart-wrapper family catalog coverage + i18n catalog keys** (#366): the `reporting-contracts` catalog demo now renders all seven wrappers (`GdsAreaChart`, `GdsSparkline`, `GdsLongitudinalChart`, `GdsBenchmarkBarChart`, `GdsRadarChart`, `GdsMaturityRadarChart`, `GdsGaugeChart`) in populated, empty, and loading states so the accessible table fallback is exercised per theme. Localized chart table-header keys (`gds.chart.table.label|value|secondaryValue|group`, all 9 packs) are added to the public catalog; `GdsChart` stays server-safe (English header defaults, consumer-overridable via `tableValueHeader`/`groupLabel`). The reporting legend swatches use `getGdsSeriesColor` instead of raw Mantine tokens, and a focused test asserts the benchmark wrapper mirrors its input into the table fallback.
- **Gold Athlete (Habigoal) brand preset** (#367): new first-class governed brand lane `gold-athlete`, mirroring the `class-usa` apparatus — `brand-tokens.ts` (five 10-step ramps `gold`/`charcoal`/`crimson`/`ivory`/`slate`, `CreateGoldAthleteBrandThemeOptions`, `createBrandTheme('gold-athlete')`), `vibe-themes.ts` (`gold-athlete` VibeTheme + semantic CSS variables, with the single-id brand-semantic guard widened to a preset map so the new lane's variables emit), `goldAthleteThemePreset` in `theme-presets.ts` (catalog + registry), and `index`/`client`/`server` exports (`goldAthleteThemePreset`, `GoldAthleteColorRampName`, `GoldAthleteColorRamps`, `CreateGoldAthleteBrandThemeOptions`). The palette passes WCAG AA in both schemes: charcoal body/heading text on ivory surfaces in light mode, ivory text with metallic-gold accents on near-black surfaces in dark mode, and a separate accessible `--gds-brand-accent-action` (darkened gold `#8a5a00`, 5.9:1 with white) for filled accent controls so the gold accent is never used as on-white body text. Additive — existing presets and the decorative `athlete-gold` vibe are unchanged.
- **Component→pattern-registry parity gate** (#368): new blocking `verify:component-catalog-parity` gate (wired into `verify:release`) fails CI if any public PascalCase UI component exported from `@sovereignsquad/gds-core` or `@sovereignsquad/gds-admin` (via each package's `index`/`client`/`server` entrypoints) is neither registered in the pattern registry (`apps/playground/src/pattern-registry.ts`, as a `sourceComponent`) nor listed in the reviewed exemption allowlist `boundary/component-catalog-exemptions.json` with a reason. Hooks (`use*`) and type-only exports are dropped by classification. Registry membership is what drives catalog render (every-theme + forced-colors), i18n routing, and a11y evidence, so this closes the exact gap that let the 17 lane components ship export-covered but unrendered/unevidenced. Sequencing this gate last surfaced that #362–#366 rendered the lane components in the catalog demos (`pattern-pages.tsx`) but never recorded them as `sourceComponent`, so the gate now completes that link: `PillBar`/`SoftChipGroup`/`FilterChipGroup` on the choice-chips row, `MissingDataPrompt` on state-blocks, `GdsSegmentedControl`/`GdsSlider`/`GdsRatingScale`/`GdsWizardStepper` on forms, the seven chart wrappers (`GdsAreaChart`/`GdsSparkline`/`GdsLongitudinalChart`/`GdsBenchmarkBarChart`/`GdsRadarChart`/`GdsMaturityRadarChart`/`GdsGaugeChart`) on reporting-contracts, `GdsDialog` on modals, and `GdsSidePanel` on drawers. The 124 genuinely non-catalog exports (layout/style primitives, typography atoms, providers/context, i18n formatters, page templates/catalog helpers, chart sub-parts and variant wrappers, `Admin*`/`Partner*` sub-parts of registered composites, icon surfaces, and client-runtime composites whose canonical pattern is registered elsewhere) are exempted with grouped reasons; stale exemptions warn non-fatally. Deterministic sub-second static scan, no build/network. See [`docs/COMPONENT_CATALOG_PARITY.md`](docs/COMPONENT_CATALOG_PARITY.md).

## 3.8.0 - 2026-07-01

- **Opaque overlay surfaces** (#342): GDS now owns the painted background of every overlay/dropdown surface (`Popover`, `Menu`, `Select`/`Combobox`, `MultiSelect`, `Autocomplete`, `HoverCard`). `styles.css` sets an opaque, GDS-owned `--gds-overlay-surface` token (white / `--mantine-color-dark-6` / system `Canvas` under forced-colors) and applies it to all dropdown containers with hard fallbacks, so overlays stay solid even when the vendor base stylesheet is absent or an unlayered consumer reset competes. Resolves the cross-client transparent-dropdown failures.
- **Mandatory stylesheet import documented + guarded** (#344): every consumer integration path — `INSTALLATION_GUIDE.md`, `docs/CLASSSCOUT_INTEGRATION.md`, `docs/AI_AGENT_GUIDE.md`, `README.md`, the Vite/Next `TEMPLATES`, and the playground install code — now shows `import '@sovereignsquad/gds-theme/styles.css'` as the first bootstrap step. `verify-install-bootstrap-docs.mjs` fails CI if any tracked integration doc or template omits it, preventing the documentation gap that caused unstyled/transparent surfaces in consumer apps.
- **Public type-boundary gate** (#343): `verify:public-types` (in `verify:release`) scans the built consumer-facing `.d.ts` for `@mantine/*` references and fails on any not recorded in `boundary/public-type-allowlist.json`. This seals GDS's public type surface — a vendor major or accidental pass-through now fails in GDS's CI, not in a consumer's compile step — and freezes the documented intentional exposures (the `GdsPrimitives` passthrough; theme-override types) so the surface can only shrink deliberately, never grow by accident. See [`docs/PUBLIC_TYPE_BOUNDARY.md`](docs/PUBLIC_TYPE_BOUNDARY.md).
- **Single install surface** (#346): consumers install `@sovereignsquad/gds` (+ React) only — the engine (`@mantine/*`, `@tabler/icons-react`) is pulled in automatically as auto-installed peers and never listed by the consumer. The engine stays a peer on purpose (single resolved instance, no dual-instance/skew failures). New `verify:install-surface` gate enforces that all GDS packages pin the **same** engine range, React/react-dom remain consumer-owned peers, the umbrella declares the full engine, and the GDS-owned `GdsIcons` surface is reachable (so consumers never import `@tabler/icons-react` directly). Install docs now lead with the single command.
- **Boundary contract suite** (#347): one named `verify:boundary` gate (in `verify:release`) composes the boundary gates into a single verdict — public type surface (#343), single install surface (#346), and a new **public CSS-selector gate** (`verify:css-boundary`) that freezes the `.mantine-*` selectors in the published stylesheet to `boundary/public-css-allowlist.json` and fails on new ones. Together with the runtime opaque-overlay checks (#342) and the export contract, this makes the vendor boundary a one-way ratchet: the vendor surface can only shrink deliberately, never grow by accident. See [`docs/BOUNDARY_CONTRACT.md`](docs/BOUNDARY_CONTRACT.md).
- **Vendor version governance** (#348): `vendor-governance.json` is the single source of truth for the GDS-owned engine version; `verify:vendor-pin` (in `verify:release`) fails if any package's engine/platform peer ranges drift from the manifest. Engine upgrades become deliberate, reversible internal migrations behind the sealed public contract via [`docs/VENDOR_UPGRADE_RUNBOOK.md`](docs/VENDOR_UPGRADE_RUNBOOK.md) — the CI matrix smoke-tests the candidate major, breakage is absorbed in adapters only, and rollback is a single revert.
- **Styling-contract migration to GDS hooks** (#345): GDS themes its core surface group (`Paper`, `Card`, `Alert`, `Code`) through GDS-owned classes (`.gds-paper`, `.gds-card`, `.gds-alert`, `.gds-code`) attached globally via theme `classNames`, instead of vendor-internal `.mantine-*` selectors. The published stylesheet's vendor-selector surface shrank 38 → 34 (enforced down by `verify:css-boundary`); migration is visually identical (the GDS class lands on the same element) and verified by the theme-trust + forced-colors runtime checks across all presets. Remaining selectors migrate incrementally under the gate. See [`docs/THEME_STYLING_HOOKS.md`](docs/THEME_STYLING_HOOKS.md).
- **Overlay adapter seam** (#349): a GDS-owned `OverlayAdapter` interface (with default `mantineOverlayAdapter`) makes the overlay engine swappable without changing any consumer call site or the public component API. `GdsProvider` accepts an `overlayAdapter` prop (defaults to current behavior); `useOverlayAdapter()` exposes it; `surfaceProps(role)` carries the `data-gds-overlay-surface` hook (#342). All adapter types are GDS-owned, so the seam does not widen the vendor type boundary. Swappability is proven by a test that reads a different engine purely by swapping the injected adapter. Overlay components adopt the seam incrementally. See [`docs/OVERLAY_ADAPTER.md`](docs/OVERLAY_ADAPTER.md).
- **Class USA first-class theme and primitive completion** (#359): `createBrandTheme('class-usa')` and `classUsaThemePreset` now ship the locked Class USA ramps, fonts, semantic tokens, token graph, and CSS variables, including accessible `--gds-brand-accent-action` for filled accent controls. `GdsProvider` applies theme-owned variables to the document root so portalled overlays inherit the active brand. Core now exposes brand-button variants, pill/soft/filter chip groups, dialog/side-panel aliases, missing-data prompts, saved/rating listing-card anatomy, and a larger accessible chart wrapper set (`GdsAreaChart`, `GdsSparkline`, `GdsLongitudinalChart`, `GdsBenchmarkBarChart`, `GdsRadarChart`, `GdsMaturityRadarChart`, `GdsGaugeChart`, `GdsCalendarHeatmapChart`, `GdsHistogramChart`, `GdsDivergingBarChart`, `GdsSlopeChart`, `GdsSymmetryChart`) with table fallback and opt-in decimation.
- **Gold-Athlete handoff closure** (Habigoal #502): the Athlete Gold lane now benefits from global portal variable propagation and the expanded package-native controls requested by the Habigoal audit: `GdsSegmentedControl`, `GdsSlider`, `GdsRatingScale`, `GdsWizardStepper`, chart-kit wrappers, `MissingDataPrompt`, and strict compliance rules for app-local raw hex/rgb colors, inline color literals, and non-token numeric radii. `docs/CLASSSCOUT_INTEGRATION.md` was updated with the 3.8.0 single-package install and migration surface list.

## 3.7.0 - 2026-06-30

- **Published consumer smoke**: `verify:published` now runs registry polling and a clean npm consumer fixture that installs the published package line outside the monorepo, type-checks imports, and verifies runtime exports across all seven public packages.
- **Athlete Gold reference surface**: `/themes` now includes a package-owned black-and-metallic-gold navigation reference panel backed by the shipped Athlete Gold VibeTheme tokens.
- **Schema upload adapter**: `GdsSchemaForm` accepts `uploadAdapter` for file-upload fields, with progress, retry, cancel, remove, upload-result payloads, and metadata-only upload events.
- **Actionable table cells**: `GdsDataTable` columns can mark `interactive: true`; grid-cell focus remains roving while `Enter`/`F2` enters nested controls and `Escape` returns to the cell.
- **Release board sync**: `board:sync-release` closes explicitly delivered issues and normalizes closed GitHub project-board cards to `Done`, with dry-run and idempotent behavior.

## 3.6.0 - 2026-06-26 (sprint 2)

Two batches shipped under the 3.6.0 version number before per-batch versioning
discipline began in 3.7.0 — kept as two dated entries, sprint 1 first, for
historical accuracy rather than merged or renumbered.

- **GdsDataTable keyboard navigation** (#333): roving grid-cell focus with Up/Down/Left/Right/Home/End traversal, `role="grid"` semantics, `aria-selected`/`aria-rowindex` per row, and `aria-live` row/column announcements.
- **GdsSchemaForm `FileUploadField`** (#334): `'file-upload'` schema field type now maps JSON Schema `format: "binary"` / `data-url` into the governed `UploadDropzone`, supports accept/multiple/max-size/progress metadata, validates required and oversized files, and submits `File[]` payloads.
- **VibeTheme expansion + `VibeThemePicker`** (#335): honey-amber `warm` lane plus the new `Athlete Gold` black-and-gold performance lane are registered in `theme-presets` and `vibe-themes`; `VibeThemePicker` renders swatch buttons for all 20 vibe presets with keyboard-accessible `role="radiogroup"` and live glow/border selection states.
- **CI Mantine 9 matrix** (#336): `quality.yml` now runs `validate` across `mantine-7` and `mantine-9` with `fail-fast: false`, overriding Mantine packages to `^9` in the second leg via `npm install --no-save`.
- **AccessGate `render-degraded-while-locked` policy** (#337): new `protectedContentPolicy` value renders the protected subtree with `aria-hidden` + `inert` while the gate is locked, enabling SEO-crawlable and hydration-ready degraded content surfaces.

## 3.6.0 - 2026-06-26 (sprint 1)

- **`/ai` route** (#327): live playground page at `/ai` surfacing `llms.txt`, install steps, a drop-in `AGENTS.md` agent-rule template, non-negotiable agent rules, and a design-tool sync entry point. Registered in locale-coverage and gds-adoption governance contracts; all 9 locale packs covered.
- **10 design-tool sync previews** (#328): hand-authored preview components for the 10 ClassScout components shipped in 3.5.0 — `BottomTabBar`, `SearchableSelect`, `FitScoreChip`, `ChatThread`, `ChatMessage`, `ChatInput`, `StreamingIndicator`, `MeaningBadge`, `MediaWithFallback`, `NumberStepper`, `AISearchCard`. For upload to the canonical design-tool sync project.
- **ClassScout integration guide** (#330): `docs/CLASSSCOUT_INTEGRATION.md` with install, GdsProvider bootstrap with `createBrandTheme`, and per-contract usage examples for all 10 B1–B10 gaps; drop-in `AGENTS.md` agent rules for the ClassScout repo.
- **Mantine 9 migration audit** (#329): `docs/MANTINE9_MIGRATION.md`; `verify:mantine` already passes Mantine 9 with no GDS code changes required.
- Opened milestone #26 (GDS 3.6.0) and 5 backlog issues (#327-331).
- Closed all 10 ClassScout issues (#316-325) with 3.5.0 delivery notes; closed milestone #25.

## 3.5.0 - 2026-06-21

- ClassScout pure-GDS unblock (issues #316–#325): closed the 10 gaps required for consumers to ship on pure GDS with no app-level forks.
  - `gds-theme`: `createBrandTheme({ brandColors, fonts })` plus a brand-named semantic token layer (`brand.primary`, `bg.page`, `text.*`, `price`, `state.*`) emitted as `--gds-*` variables on top of the governed token graph, with WCAG-AA contrast enforcement (#316).
  - `gds-core`: `'bottom-tab'` mobile navigation mode + `BottomTabBar` (safe-area aware, raised center action) for `PublicShell`/`DiscoveryShell` (#317); `SearchableSelect` combobox with async/grouped options and full keyboard a11y (#318); `FitScoreChip` (#319); `ListingCard` `reason`/`score`/`actions` composition slots (#320); conversation surface `ChatThread`/`ChatMessage`/`ChatInput`/`StreamingIndicator` (#321); `MeaningBadge` distinct from `StatusBadge` (#322); `MediaWithFallback` resilient media (#323); `NumberStepper` (#324); `AISearchCard` governed assistant-entry pattern (#325).
  - All new UI consumes GDS tokens, is keyboard- and screen-reader-accessible, and is registered in the pattern export/API-docs coverage registries.
- Added an AI-agent integration layer so GDS is consumable by any LLM-powered coding tool: `llms.txt` (universal machine-readable entry point), `docs/AI_AGENT_GUIDE.md`, and a "Use with AI" quick-start in the README.
- Added a drop-in repo rule template `TEMPLATES/AGENTS.md.template` (cross-tool `AGENTS.md` standard) so consuming repos make every agent session build with GDS automatically.
- Added a design-tool sync integration: syncing GDS into a visual design tool so the design agent builds screens with the real GDS components, with committed sync inputs that make a re-sync one command.
- Synced all 252 components (249 hand-authored, render-verified previews + 3 floor-carded body-portal overlays) into the canonical GDS design-tool sync project, with a conventions header teaching the GdsProvider/prop-token/semantic-action build idiom.

## 3.4.14 - 2026-06-13

- Added dependency-governance policy for React, Mantine, Tabler, dependency classes, replacement triggers, and exception lifecycle.
- Classified API reference exports by stability and implementation boundary so consumers can distinguish canonical GDS contracts from Mantine/Tabler-backed surfaces.
- Added `GdsIcon name="..."` semantic icon support while keeping the existing `icon` prop compatible.
- Added dependency-boundary exception validation, dependency-risk reporting, and expanded compatibility smoke coverage for Mantine 7/React 18 plus Mantine 8/9 React 19.

## 3.4.13 - 2026-06-13

- Fixed `DocsShell` mobile headers so translated brand labels truncate safely instead of wrapping into the action controls.
- Added `DocsHeaderActionSelect` as the package-owned bounded header select for language and compact docs-shell actions.
- Added browser runtime verification for Russian, German, Hebrew, and Arabic mobile header layouts to catch clipped controls, horizontal overflow, and brand/action overlap before release.
- Updated governance and component rules so responsive localization failures block release instead of being treated as cosmetic defects.

## 3.4.12 - 2026-06-12

- Fixed cosmic and dark-forward preset overrides so mixed light preview cards no longer inherit forced white text, forced dimmed text, or fixed `28px` Paper/Card radius values from the surrounding page.
- Added local contrast CSS ownership for preview-surface buttons, inputs, badges, code, nested cards, foreground text, muted text, backgrounds, borders, and radius tokens.
- Stopped generated phrase translation from mutating interactive controls after React render so buttons, links, labels, selects, and input attributes do not become scrambled or stale.
- Added regression coverage for local contrast surfaces, radius governance, and safe phrase translation boundaries.

## 3.4.9 - 2026-06-12

- Fixed the Theme Lab shipped-lane gallery so light VibeTheme preview cards keep their own dark foreground tokens when the surrounding page is in dark mode.
- Scoped the preset contrast guard away from `[data-gds-local-contrast]` surfaces so intentional mixed-preview cards can own local readable text, controls, and code colors.
- Added release verification coverage for the local contrast marker used by the gallery cards.

## 3.4.8 - 2026-06-12

- Hardened the shared preset stylesheet so VibeTheme surfaces push readable foreground tokens through Mantine text, dimmed text, shell, card, paper, input, table, alert, code, link, and default-button surfaces.
- Fixed dark and dark-forward colorful lanes where nested Mantine components could keep light-mode text colors on dark or saturated backgrounds.
- Added theme-governance verification for preset contrast token coverage so future theme changes cannot silently drop the foreground contract.

## 3.4.7 - 2026-06-07

- Removed English-only route coverage from the official playground so every public route keeps all supported locales available instead of falling back to English.
- Added checked-in generated phrase resources and release verification for route, demo, pattern, and use-case copy that still comes from registry/demo data.
- Localized shared shell/reference labels including navigation section headings and reference link actions, and fixed direct localized URL initialization via `?locale=...`.

## 3.4.6 - 2026-06-07

- Fixed the package-owned theme explorer i18n resolver so incomplete locale resources no longer render mixed-language surfaces by merging partial translations with English fallback copy.
- Added regression coverage proving partial theme-explorer locale resources fall back to one complete language until the locale has full nested copy coverage.
- Aligned package, reference-app, and install metadata to the `3.4.6` patch release line.

## 3.4.5 - 2026-06-07

- Added strict consumer-admin migration enforcement with `approvedAdminPrimitives` and `strict.admin.local-wrapper` detection for product-local admin layout, form, action, card, breadcrumb, media, and field shims.
- Added package-native core exports for primitive/layout, typography role, sanctioned style utility, semantic chart, and icon-tone contracts so strict consumers have installable replacements for common local wrappers.
- Expanded the playground/reference site with Spanish full-copy route coverage and updated release/install guidance for the `3.4.5` package line.

## 3.4.4 - 2026-06-07

- Added the package-native operational telemetry contract with `GdsOperationalEvent`, `GdsEventPayloadPolicy`, `GdsTelemetryAdapter`, `emitGdsEvent`, `createGdsTelemetryAdapter`, event taxonomy, and UX failure reason registry.
- Added non-blocking adapter dispatch with emitted, adapter-unavailable, payload-rejected, sampled-out, sampling-disabled, and dropped states plus bounded retry and timeout behavior for analytics adapters.
- Expanded telemetry tests, API coverage, install guidance, API reference, user guide, and LLD documentation for privacy-safe payload rules, accessibility boundaries, rollback, and operational behavior.

## 3.4.3 - 2026-06-07

- Fixed mobile shell navigation so `DiscoveryShell` opens reliably from the hamburger and closes the mobile menu when a navigation item is selected, with `closeMobileNavigationOnItemSelect` available for rare controlled-menu opt-outs.
- Fixed inline mobile navigation in `DocsShell` and `PublicShell` so documentation and public flow menus collapse back to the hamburger/menu state after link or action activation.
- Added regression coverage for mobile navigation close-on-selection behavior across discovery and public shell contracts.

## 3.4.2 - 2026-06-06

- Moved reference-site localized route labels, app-shell copy, page copy, and theme-explorer copy out of React runtime components into dedicated i18n resource contracts.
- Added public locale metadata helpers in `@sovereignsquad/gds-theme`: `gdsLocaleMetadata`, `getGdsLocaleMetadata(...)`, `isGdsRtlLocale(...)`, and `getGdsLocaleIdsByScript(...)`.
- Updated `GdsProvider` and font-lane coverage to resolve RTL and script support from locale metadata instead of hardcoded language arrays.
- Replaced the locale coverage verifier so CI fails on component-local language dictionaries, `locale === ...` branches, localized route labels, and locale arrays outside approved i18n resource files.

## 3.4.1 - 2026-06-06

- Fixed the public reference site locale experience so the overview route no longer mixes English cards and links into Russian, Italian, Hebrew, Arabic, Hungarian, German, or French full-copy locales.
- Localized the shared site footer and primary navigation labels used by the full-copy routes.
- Added regression coverage to fail when Russian overview renders the English strings `Operational clarity`, `Public trust`, or `Browse patterns`.

## 3.4.0 - 2026-06-06

- Added issue-backed maturity capability contracts for the seven recommended high-value GDS delivery areas: admin delivery, runtime feedback, foundation surfaces, global readiness, adoption governance, theme operations, and product-system delivery.
- Added `getGdsMaturityCapabilities()`, `getGdsRecommendedMaturityCapabilities()`, `getGdsMaturityCapability(...)`, and `getGdsMaturitySummary()` to `@sovereignsquad/gds-core`.
- Added the localized `/maturity` GitHub Pages route so developers and product owners can inspect benefits, package lanes, primary contracts, UX states, accessibility, observability, rollback, testing, and operational behavior in every supported site language.
- Created the 3.4.0 GitHub project-board issue set using the issue #81 production-grade structure.
- Updated API, user-guide, CLI/LLD, install, compatibility, release, and README guidance to the `3.4.0` npm release line.

## 3.3.0 - 2026-06-06

- Added registry-backed `/api` documentation for published GDS package exports, including import paths, runtime lanes, state contracts, accessibility notes, and verification metadata.
- Added `/use-cases` as the product-owner adoption guide for matching product needs to package lanes, primary contracts, risk, accessibility obligations, and operational checks.
- Added `API_REFERENCE.md`, `USER_GUIDE.md`, and `CLI_AND_LLD.md` so GitHub readers can discover the same API, product, CLI, and low-level design contracts without relying only on the GitHub Pages UI.
- Added release gates for API documentation coverage, route localization coverage, package message parity, and native-dialog i18n copy enforcement.
- Fixed project-board audit issue-state pagination so the strict board audit reports all current open project items.

## 3.0.7 - 2026-06-06

- Added package-owned block layout cookbook APIs: `getGdsLayoutTemplates`, `getGdsLayoutTemplate`, and `GdsLayoutTemplatePreview`.
- Replaced the reference-site layout cookbook's app-local raw form controls with the GDS-owned preview component while preserving template selection, JSON editing, diagnostics, copy behavior, and rendered preview states.
- Added `npm run audit:dependencies` to the release gate, upgraded Vitest to `4.1.8`, moved the private Next reference fixture to dev-only scope, and documented the remaining upstream Next/PostCSS dev advisory in `DEPENDENCY_AUDIT.md`.

## 3.0.6 - 2026-06-06

- Added package-native admin CRUD primitives in `@sovereignsquad/gds-admin`: `AdminTextInput`, `AdminTextarea`, `AdminCheckbox`, `AdminSelect`, `AdminFileUpload`, `AdminFormSection`, `AdminFormStatus`, `AdminFormActions`, and `AdminCrudForm`.
- Added hardened admin data/resource surfaces: `AdminDataTable`, `AdminAnalyticsTable`, `AdminModal`, `AdminDetailDrawer`, `AdminReviewLayout`, `AdminResourceManager`, `AdminResourceGrid`, `AdminResourceCard`, `AdminResourceToolbar`, and `AdminResourceEmptyState`.
- Added core interaction/runtime contracts: `GdsConfirmProvider`, `useGdsConfirm`, `GdsToastProvider`, `useGdsToasts`, typed `GdsIcon`, `MediaPreviewCard`, `PublicCaptureFlow`, `PlaybackControls`, and creator theme validation/boundary utilities.
- Expanded `gds-compliance` strict mode to detect direct Mantine imports, direct Tabler imports, raw form controls/buttons, browser dialogs, raw table markup, inline styles, and undeclared local GDS adapters with exception-aware suppression.

## 3.0.5 - 2026-06-02

- Fixed the `cosmic` VibeTheme to behave as a dark-forward runtime lane so the Theme Lab and live preview no longer render washed-out light panels or low-contrast muted text.
- Tightened `cosmic` glass panels, inputs, badges, code blocks, and dimmed text treatment so the high-saturation multicolour background remains vivid while content stays readable.

## 3.0.4 - 2026-06-02

- Added `cosmic` as the first intentionally high-saturation CSS VibeTheme, with a multicolour blue-violet-cyan-magenta background, star-field atmosphere, glass panels, and vivid gradient primary controls.
- Documented `cosmic` as the sanctioned dramatic showcase lane so consumers do not need route-local image backgrounds or private gradient systems for bold launch/public surfaces.

## 3.0.3 - 2026-06-02

- Added package-owned CSS VibeThemes for the colorful preset line so `sunset`, `oceanic`, `forest`, `ruby`, `amber`, `neon-night`, `skyline`, `aurora`, `coral`, `mint`, `orchid`, and `royal` now expose full canvas, shell, surface, border, text, accent, glow, gradient, and hero tokens instead of only changing `primaryColor`.
- Extended `useGdsThemePresetState(...)` to apply `data-gds-theme-preset` plus `--gds-vibe-*` CSS variables on the document root, making whole-site theme switching persistent across direct links and route changes.
- Updated the GitHub Pages Theme Lab with a visual VibeTheme gallery and current-token proof section, and documented the no-image-background/no-local-theme-catalog governance rule.

## 3.0.0 - 2026-05-31

- Delivered the adoption-platform release with hardened install/bootstrap docs, reference-site governance, public feature-request intake, media/upload contracts, reporting/evidence/chart contracts, auth/access identity hardening, strict compliance expansion, and verified reference codemods.
- Added 3.0.0 release-readiness checks covering board scope, implementation evidence, release safety, client communication, and registry verification gates.
- Updated the publish runbook so major releases require strict board audit before and after version bump and may not be announced until npm publication is verified.

## 2.6.7 - 2026-05-31

- Cut and published the `2.6.7` npm release line so the widened docs/reference-shell layout is available through the public package line and not only on repository `main`.
- Updated `DocsPageShell` in `@sovereignsquad/gds-core` to use the full available page width for the official site and other reference/docs surfaces, removing the narrow article cap that was squeezing wide content such as the theme-governance explorer.
- Aligned the public site copy, install guidance, and versioned docs/routes to the `2.6.7` line.

## 2.6.5 - 2026-05-29

- Cut and published the `2.6.5` npm release line so consumers can update to the canonical theme-governance hardening through the public registry instead of relying on unpublished mainline changes.
- Deprecated consumer-facing `extendGdsTheme(...)` as a canonical adopter path and formalized the approved theme lanes around `gdsTheme`, `gdsDarkPublicTheme`, `gdsFlatSurfaceTheme`, `gdsEditorialPublicTheme`, and `createPublicBrandTheme(...)`.
- Added manifest-scoped theme-governance enforcement fields plus `gds-compliance` detection for direct `extendGdsTheme(...)` usage and parallel local branding-layer theme construction in declared theme-ownership files.
- Aligned templates, reference consumers, install/governance docs, and theme guidance to the canonical adopter theme path so teams can copy a governed theme setup directly from GDS.

## 2.6.4 - 2026-05-28

- Deprecated consumer-facing `extendGdsTheme(...)` as a canonical adopter path and formalized the approved theme lanes around `gdsTheme`, the shipped public presets, and `createPublicBrandTheme(...)`.
- Added manifest-scoped theme-governance enforcement fields plus `gds-compliance` detection for direct `extendGdsTheme(...)` usage and parallel local branding-layer theme construction in declared theme-ownership files.
- Aligned templates, reference consumers, install/governance docs, and theme guidance to the canonical adopter theme path so teams can copy a governed theme setup directly from GDS.
- Added `ReferenceSection`, `ReferenceLinkGrid`, `ReferenceLocaleNotice`, and `ReferenceThemeExplorer` to `@sovereignsquad/gds-core` plus `ReferenceSiteShell` to `@sovereignsquad/gds-admin` so the official website can consume GDS-owned docs/reference primitives instead of site-local Mantine composition.
- Converted `apps/playground` onto the new reference-site primitives, replaced the remaining direct page-level Mantine composition in the public site source, and added a strict `gds-adoption.json` baseline for the website.
- Updated the public route structure, docs copy, and rulebooks so the GitHub Pages site is described as both the official GDS website and a strict live reference consumer rather than a separate playground exception.
- Added the registry-backed GitHub Pages pattern catalog under `/patterns` with dedicated family routes for foundations, public, operations, data, access, and feedback coverage.
- Expanded the public docs site to show live demos for the remaining workflow and responsive-guidance contracts that had previously been represented only as reference notes.
- Hardened the GitHub Pages playground with route-level lazy loading, deterministic vendor chunking, and contained shell previews so the public site behaves like a documentation site instead of a nested application demo.
- Added an interactive `/themes` explorer so adopters can switch among shipped theme presets, test light/dark behavior, inspect the bounded creator-authored theming lane, and compare theme lanes directly on the public site.
- Added a dedicated `/live-demos` hub so the public website separates official docs from runtime showcase sections more clearly.
- Strengthened `@sovereignsquad/gds-compliance` exception enforcement to fail stale exception scopes, uncovered local exception adapters, and incomplete creator-authored experience exception metadata.
- Updated theme governance, exception-surface, compliance, and adoption rulebooks to formalize the creator-authored experience contract and the new repo-to-manifest exception checks.
- Added `SocialAuthButtons` to `@sovereignsquad/gds-core` as the canonical provider-entry cluster for Google, Apple, GitHub, Microsoft, LinkedIn, Discord, X, Facebook, and email-shaped auth lanes.
- Added `ShareButtonGroup` to `@sovereignsquad/gds-core` as the canonical public sharing surface for native share, copy-link, email, message, and social-channel actions.
- Enhanced `AuthShell` to support governed social-auth placement and divider rhythm instead of consumer-local auth-provider layouts.
- Published the umbrella install path `@sovereignsquad/gds` as the preferred public npm entrypoint while keeping the granular runtime packages available for stricter dependency boundaries.
- Aligned release automation and public documentation so npm publication, GitHub Pages guidance, and release-bundle fallback distribution target the same live release line.

## 2.6.3 - 2026-05-27

- Added `showGdsNotification(...)` to `@sovereignsquad/gds-theme/client` as the canonical semantic notification helper for consumers already governed by the shared provider stack.
- Enhanced `AuthShell` with `headerActions` so products can place theme toggles or other small auth-entry controls without rebuilding the shell locally.
- Enhanced `PageHeader` to accept `subtitle` as an alias for `description`, reducing consumer-only adapter code.
- Hardened `SemanticButton` to use the label-first prerender path by default, removing the need for client repos to carry their own hydration-safe semantic-button wrappers.

## 2.6.2 - 2026-05-27

- Expanded shared Mantine peer support to include `^9.0.0` across the runtime packages and verified fresh packed-consumer installs against Mantine `9.2.1`, React `19.2.0`, and Next `15.5.18`.
- Added `npm run verify:mantine` as the canonical compatibility command and broadened the smoke harness to cover both Mantine `8.3.6` and `9.2.1` in isolated clean installs.
- Added root optional native bindings for supported macOS and Linux x64 environments so fresh local `npm install` runs provision the Vite/tsup native layer more reliably.
- Updated README, compatibility guidance, migration guidance, release runbook, and consumer-install proof docs to reflect the Mantine 9 support line and the current temporary release-asset install path.
- Added `ChoiceChip` to `@sovereignsquad/gds-core` as the canonical neutral chip for lightweight filter, scope, taxonomy, and mode selection without page-local badge wrappers.
- Added `getSemanticActionLabel(...)` to `@sovereignsquad/gds-core` as a server-safe semantic-label helper for SSR/static fallback rendering without exposing raw vocabulary access as the only consumer path.
- Recorded the current Narimato reference-consumer audit and updated the Narimato project note to reflect direct npm package consumption plus its intentionally local exceptions.

## 2.6.1 - 2026-05-26

- Renamed the public package line from the unpublished placeholder `@gds/*` scope to the real npm organization scope `@sovereignsquad/gds-*`.
- Updated package metadata, workspace wiring, reference consumers, compliance manifests, docs, and release scripts to consume the `@sovereignsquad/gds-*` package family consistently.
- Fixed release-environment dependency gaps (`@floating-ui/core`, `@humanfs/core`, and `@babel/core`) so local and CI release verification run cleanly on the current toolchain.
- Updated the packed Mantine 8 compatibility harness to install the renamed tarballs correctly and verified the `@sovereignsquad/gds-*` line against Mantine `8.3.6`, React `19.2.0`, and Next `15.5.18`.
- Clarified npm as the canonical future registry source and documented public GitHub release assets as the approved temporary install path while npm publication remains unavailable.
- Added `npm run pack:release`, release-bundle checksums/manifests, and the `GDS Release Bundles` GitHub Actions workflow.
- Added `VERIFIED_CONSUMER_INSTALL_PROOF.md` to make the current Next 15 / React 19 / Mantine 8 install evidence explicit for adopter teams.

## 2.6.0 - 2026-05-26

- Added `SectionPanel` and `ConsumerDashboardGrid` to `@sovereignsquad/gds-core` as the canonical operational panel rhythm and consumer dashboard layout scaffolds.
- Hardened shared operational contracts in `@sovereignsquad/gds-admin`: `AppShell` now supports primary/secondary/account navigation regions and header context, `PageHeader` now supports subtitle/status/overflow actions, `ResponsiveDataView` now supports active filter chips plus mobile filter surfaces, and `EditorScaffold` / `ContentOpsEditor` now support context and sticky footer action regions.
- Enhanced `@sovereignsquad/gds-core` `BrowseSurface`, `EditorialCard`, `FilterDrawer`, and `MediaField` to reduce remaining local public/gds-admin overrides.
- Added `createPublicBrandTheme()` to `@sovereignsquad/gds-theme` and formalized the branded public theme merge path.
- Widened shared Mantine peer ranges to include `8.3.x` and added `npm run verify:mantine8` as a packed-consumer compatibility smoke for Mantine 8.3.6 + React 19.2.0 + Next 15.5.18.
- Documented the canonical searchable-selection decision: use governed Mantine recipe composition rather than a new shared wrapper until a stronger repeated contract emerges.

## 2.5.1 - 2026-05-25

- Expanded `@sovereignsquad/gds-compliance` with configurable banned imports plus default stale-SSOT reference detection so consumer repos can catch lingering legacy UI dependencies and outdated documentation paths through shared tooling.
- Added [COMPLIANCE_TOOLKIT.md](COMPLIANCE_TOOLKIT.md) as the canonical CI and local enforcement contract for `@sovereignsquad/gds-eslint-config` and `@sovereignsquad/gds-compliance`.
- Updated template and adoption artifacts to use the canonical repository path and the current machine-readable manifest contract.

## 2.5.0 - 2026-05-25

- Added new cross-project public and consumer contracts in `@sovereignsquad/gds-core`: `BrowseSurface`, `EditorialCard`, `ConsumerSection`, and `MediaField`.
- Added new content-operations contracts in `@sovereignsquad/gds-admin`: `ContentOpsEditor`, `ContentOpsSection`, and `ContentOpsActionBar`.
- Added `gdsEditorialPublicTheme` to `@sovereignsquad/gds-theme` as the approved serif-forward, flatter editorial preset.
- Added `ADOPTION_AND_MIGRATION_PLAYBOOK.md` plus manifest-driven compliance settings for documentation paths, stale-reference detection, and protected surface declarations.
- Updated the Next.js and Vite reference consumers plus shared component tests to exercise the new browse, consumer, media, and content-operations contracts.

## 2.4.4 - 2026-05-25

- Enhanced `PublicShell` with canonical header variants, class-name hooks, and server-safe mobile navigation modes so public consumers can stop shipping repo-local spacing and nav overrides.
- Enhanced `PublicBrandFooter` with documented layout variants and slot-level class hooks for narrative, media, quote, and legal regions.
- Enhanced `PublicProductCard` with localized state-label overrides plus pickup and inventory helper-note support for menu, discovery, and retail-like public surfaces.
- Updated the Vite and Next.js reference consumers plus shared component tests to exercise the new public-surface contracts end to end.

## 2.4.3 - 2026-05-25

- Added `AccentPanel` as the canonical light/dark-safe accent surface contract for public and operator-facing emphasis panels.
- Added `EditorialHero`, `FeatureBand`, and `PublicBrandFooter` to `@sovereignsquad/gds-core` for shared public/editorial composition without repo-local layout authority.
- Hardened release verification with export-contract checks that fail on missing published export targets or server entrypoints that drift into client-only modules.
- Updated the Next.js and Vite reference consumers to exercise the new public/editorial primitives and the server-safe import path.
- Added an authenticated publish runbook and shared `publish:dry-run` / `publish:npm` scripts for the five public GDS packages.
- Added `verify:published` plus a manual GitHub Actions publish workflow so authenticated CI can publish and verify registry availability with bounded retry behavior.

## 2.4.2 - 2026-05-25

- Added `@sovereignsquad/gds-core` `PublicProductCard` for media-first public menu, catalog, and offer surfaces with price/state/action hierarchy.
- Added `es` locale support plus canonical `GdsLocale` and `getGdsMessages(locale)` exports for host-i18n bridges.
- Extended shared lint/gds-compliance tooling to support manifest-driven approved dependency/import exceptions such as `lucide-react`.
- Updated compatibility, governance, and Pesti Est adoption docs for registry-first CI usage and locale/exception guidance.

## 2.4.1 - 2026-05-25

- Added `@sovereignsquad/gds-core` `AccessRecoveryPanel` as the canonical protected-content, expired-session, and recoverable failure surface.
- Updated component contracts to treat access recovery as a first-class shared pattern family.
- Resolved the learner-shell evaluation by documenting that LMS learner shells remain local adapters until broader portfolio reuse is proven.
- Updated Amanoba guidance to consume shared access recovery now while keeping learner shell, course cards, and gamification list cards local for now.

## 2.3.2 - 2026-05-25

- Added `@sovereignsquad/gds-core` `GameBoardTile` for memory-match and flip/select game boards (reduced-motion aware).
- Added `docs/AMANOBA_BLOCKING_CONTRACTS.md` scaffolds for remaining Amanoba-only surfaces (LearnerAppShell, course cards, recovery panel).
- Refreshed `GDS_GAP_INVENTORY.md` §2B to reflect 2.3.0–2.3.1 shipped package surfaces.
- Added Amanoba dark-shell + yellow CTA `extendGdsTheme` recipe appendix to `THEME_GOVERNANCE.md`.

## 2.4.0 - 2026-05-25

- Added `compatibility.matrix.json`, `schemas/gds-adoption.schema.json`, and `TEMPLATES/gds-adoption.json.template` as machine-readable compatibility and adoption contracts.
- Added `@sovereignsquad/gds-eslint-config` and `@sovereignsquad/gds-compliance` to provide shared lint and compliance enforcement for adopting repositories.
- Added new public composition primitives in `@sovereignsquad/gds-core`: `PublicNav`, `PublicSiteFooter`, `DocsPageShell`, `DocsCodeBlock`, `CtaButtonGroup`, `PlaceholderPanel`, `SimpleDataTable`, and `StatsSection`.
- Expanded `@sovereignsquad/gds-theme` with `gdsDarkPublicTheme`, `gdsFlatSurfaceTheme`, and root-provider theme/default color-scheme overrides.
- Added reference consumer fixtures under `apps/reference-vite` and `apps/reference-next`, plus `npm run verify:references` for fixture and manifest validation.
- Added `DEPRECATIONS_AND_MIGRATIONS.md` to formalize contract retirement, migration guidance, and release handover expectations.

## 2.3.1 - 2026-05-25

- Changed `@sovereignsquad/gds-core` `PageHeader` eyebrow styling to a neutral default, removing forced uppercase and decorative tracking from the canonical contract.
- Added opt-in `eyebrowVariant="ornamental"` for products that explicitly want decorative eyebrow styling.
- Removed forced hover motion and transform transitions from the canonical `@sovereignsquad/gds-theme` base theme.
- Added `withGdsMotion()` as an explicit opt-in theme helper for products that want shared motion styling.
- Expanded `COMPATIBILITY_AND_RELEASES.md` with an explicit Next.js App Router consumer path for `server` and `client` package entrypoints.

## 2.3.0 - 2026-05-24

- Added publish-ready package metadata and explicit `client` / `server` subpath exports for `@sovereignsquad/gds-theme`, `@sovereignsquad/gds-core`, and `@sovereignsquad/gds-admin`.
- Added `COMPATIBILITY_AND_RELEASES.md` to define the active Mantine/React/Next consumption contract, install guidance, and version-alignment rules.
- Added new shared package primitives and scaffolds for `MetricCard`, `ProgressCard`, `ProductCard`, `StateBlock`, `DataToolbar`, `PublicShell`, `AuthShell`, `ArticleShell`, `UploadDropzone`, `MediaCard`, `AccessSummary`, `ResponsiveDataView`, `WorkspaceHeader`, and `EditorScaffold`.
- Expanded admin primitives to support mobile footer navigation, richer page-header action slots, and shared empty-state handling in tables.
- Added release-alignment verification via `npm run verify:release` and a shared pull-request checklist template.
- Added `THEME_GOVERNANCE.md` and `EXCEPTION_SURFACES.md` to cover provider-brand, white-label, tenant-theme, chart, map, embed, and other approved exception surfaces.
- Added portfolio onboarding plans for Impact, Camera, and Pesti Est plus matrix rows reflecting their current GDS adoption pressure.

## 2.2.2 - 2026-05-24

- Updated `@sovereignsquad/gds-theme` `GdsProvider` to include Mantine modals and notifications so the shared provider matches the documented root composition contract.
- Added shared package i18n coverage for theme-toggle labels, empty-data messaging, and semantic error feedback.
- Added a shared Vitest + jsdom test harness plus behavior coverage for `@sovereignsquad/gds-theme`, `@sovereignsquad/gds-core`, and `@sovereignsquad/gds-admin`.
- Added root test commands and pull-request quality gates for build, lint, and tests.

## 2.2.1 - 2026-05-23

- Added `PROJECTS/NARIMATO.md` for Narimato (Mantine-rooted, enforcement phase).
- Updated `PROJECTS/PORTFOLIO_ADOPTION_MATRIX.md` Narimato row from discovery to enforcement.
- Fixed `@sovereignsquad/gds-core` `ConfirmDialog` confirm button color: `brand` → `violet` (valid Mantine palette).

## 2.2.0 - 2026-05-23

- Added `SERVICE_BACKBONE_IMPLEMENTATION_PLAN.md` to define the GDS as a reliable, cross-project service with authority, pattern, adoption, validation, portfolio, and lifecycle layers.
- Added `PROJECTS/PORTFOLIO_ADOPTION_MATRIX.md` to classify projects by foundation signals, migration archetypes, risk, and recommended next actions.
- Added `PROJECTS/MESSMASS_MANTINE_REFACTOR.md` to address the highest known authority-conflict project in the portfolio.
- Expanded `PATTERN_SERVICE_MODEL.md` with required service outputs, contract maturity states, compatibility promises, and portfolio usage guidance.
- Expanded `GOVERNANCE_AND_ADOPTION.md` with authority-conflict handling, adapter-inventory checks, portfolio operations, and normalized remediation categories.

## 2.1.0 - 2026-05-23

- Added `PATTERN_SERVICE_MODEL.md` as the cross-project operating model for borrowing Mantine UI ideas and promoting them into governed GDS contracts.
- Defined mandatory reusable pattern families for shells, page headers, product cards, metrics, data toolbars, responsive data views, auth shells, article/docs shells, and state blocks.
- Added cross-project recommendations for Amanoba, KIDEX, ClassScout, and SSO so each project has a concrete Mantine-only pattern-service implementation path.
- Updated foundation, component, governance, and README guidance to prohibit page-local reinvention of reusable patterns and to require local adapter paths for shared contracts.

## 2.0.0 - 2026-05-22

- **Massive Consolidation & Hardening:** Refactored the sprawling 17-file structure into 3 hardened, professional rulebooks (`FOUNDATION.md`, `COMPONENTS_AND_PATTERNS.md`, and `GOVERNANCE_AND_ADOPTION.md`).
- Eliminated all outdated, duplicated, and inconsistent language.
- Enforced a strictly professional, prescriptive tone for all Mantine, UX, and Token boundaries.

## 1.3.3 - 2026-05-22

- Added the normative color-mode and readability contract for dark mode, light mode, contrast, mixed-mode exceptions, and Mantine theme responsibilities.
- Updated foundation and runtime guidance to make human readability and active-mode consistency release gates.

## 1.3.2 - 2026-05-21

- Updated the Amanoba migration plan from planned to in-progress.
- Recorded completed Amanoba Phase 0/1 runtime work, active Mantine guardrails, and the current course-surface migration snapshot.
- Added remaining Amanoba high-priority gaps for lesson runtime, quiz runtime, final exam, auth, dashboard, saved lessons, practice hub, admin/editor forms, and deletion-phase legacy dependencies.

## 1.3.1 - 2026-05-21

- Updated the SSO migration plan to mark admin-shell migration and legacy theme-stack removal as completed.
- Clarified the SSO path toward the remaining docs/editorial migration and final deletion pass.

## 1.3.0 - 2026-05-21

- Added a primitive policy matrix for direct-versus-wrapper decisions.
- Added implementation tables for variants, sizes, breakpoints, shell switches, and responsive behavior.
- Added enforcement guidance for lint rules, import boundaries, and drift checks.
- Added reusable starter templates for providers, theme, shell, page header, and button wrappers.
- Updated runtime, readiness, and README guidance to point directly to the new implementation assets.

## 1.2.3 - 2026-05-21

- Promoted implementation-readiness checks into the required reading order.

## 1.2.2 - 2026-05-21

- Added root provider/gds-theme implementation notes to the required project-adoption contract.

## 1.2.1 - 2026-05-21

- Added component contracts for date/time inputs, file uploads, loaders/skeletons, tooltips, breadcrumbs, and pagination.
- Expanded migration deliverables with provider/gds-theme implementation notes plus validation and deletion checklist expectations.

## 1.2.0 - 2026-05-21

- Tightened the SSOT from “Mantine preferred/first” language to an explicit Mantine-only product primitive policy.
- Clarified that no new product UI may bypass Mantine with ad hoc primitives or alternate component frameworks.
- Updated migration, governance, adoption, and KIDEX adapter language to reflect the stricter policy.
- Added `MANTINE_RUNTIME.md` to define provider, theme, notifications, modals, wrapper, CSS, form, overlay, and data-display runtime requirements.

## 1.1.0 - 2026-05-21

- Expanded the SSOT into a stricter multi-project policy repository.
- Added Mantine platform policy and navigation/responsive rules to the required reading order.
- Tightened component, foundation, UX, governance, and project-adoption contracts for cross-project enforcement.
- Clarified that product repositories may document only adapters, exceptions, migration state, and validation commands.
- Added repository hygiene rules for shared Git usage across consuming projects.
- Added implementation-readiness requirements so projects document root provider, theme ownership, primitive policy, legacy boundaries, responsive strategy, and drift controls before the first Mantine PR.
- Expanded the SSO project plan with local-adapter requirements, phase exit criteria, validation commands, and initial implementation sequence.

## 1.0.0 - 2026-05-21

- Established `/Users/Shared/Projects/GENERAL_DESIGN_SYSTEM` as the cross-project design, UI, and UX SSOT.
- Added normative foundation rules for theme ownership, tokens, layout, accessibility, responsiveness, and internationalization.
- Added strict component contracts for buttons, icon buttons, inputs, forms, cards, modals, drawers, tables, navigation, alerts, notifications, empty states, loaders, errors, and pagination.
- Added UX rules for app shells, dashboards, learner flows, admin/editor flows, destructive actions, authentication, search, filters, responsive behavior, and content tone.
- Added governance rules for adoption, project adapters, exceptions, review, migration order, and definition of done.
- Added the project adoption contract that each project must reference in its local documentation.
- Added a true-refactor Mantine migration playbook for legacy projects.
- Added project-specific migration planning under `PROJECTS/`, including the initial SSO refactor plan.
- Added contributing guidance for operating this directory as a shared standalone git repository.
