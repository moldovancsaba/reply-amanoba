import { useState, type KeyboardEvent, type MouseEvent, type ReactNode } from 'react';
import { useGdsTranslation } from '@sovereignsquad/gds-theme';
import { ActionIcon, Badge, Box, Card, Group, Stack, Text, Title } from '@mantine/core';
import { GdsIcons } from './icons';
import { GdsGeneratedThumbnail } from './GdsGeneratedThumbnail';
import { GdsVocabulary, getSemanticActionLabel, type SemanticAction } from './vocabulary';
import { resolveGdsCardContract, type GdsCardDensity, type GdsCardInteractiveMode, type GdsCardSize, type GdsCardVariant } from './CardContracts';

/** Aspect ratio for a `ListingCard`'s media slot. */
export type ListingCardMediaRatio = '1:1' | '4:3' | '16:9';

/** One key/value metadata row shown in a `ListingCard` (with an optional icon and tone). */
export interface ListingMetadataRow {
  id: string;
  label: ReactNode;
  value?: ReactNode;
  icon?: ReactNode;
  tone?: 'default' | 'positive' | 'warning' | 'muted';
}

/** A save/share-style affordance on a `ListingCard`, described by a semantic `action` and its handler/href. */
export interface ListingCardAffordance {
  action: SemanticAction;
  onClick?: () => void;
  href?: string;
  ariaLabel?: string;
  disabled?: boolean;
  active?: boolean;
  /**
   * `contained` (default) keeps the usual subtle disc. `outline-on-media` drops the disc entirely
   * and carries a shadow instead, so the glyph stays legible directly over a photograph — the
   * presentation a save control needs when it sits on the media rather than in the footer
   * (issue 679).
   */
  presentation?: 'contained' | 'outline-on-media';
}

/** Governed `pickBadge` config: `true` for the localized default, or an override. */
export interface ListingCardPickBadge {
  /** Overrides the visible label. Defaults to the localized `gds.listingCard.pickLabel`. */
  label?: ReactNode;
  /** Overrides the accessible name. Defaults to the visible label when it is a plain string. */
  ariaLabel?: string;
}

/** Props for {@link ListingCard}. */
export interface ListingCardProps {
  title: ReactNode;
  href?: string;
  description?: ReactNode;
  image?: ReactNode;
  imageAlt?: string;
  /**
   * Stable identity seeding the generated thumbnail shown when no `image` is supplied — same
   * seed, same composition, every render. Defaults to `href`, then `imageAlt`, then the title
   * when it is a plain string. Supply it explicitly when a listing's title or URL can change
   * while the listing itself does not, so its art stays put.
   */
  mediaSeed?: string;
  mediaRatio?: ListingCardMediaRatio;
  metadata?: ListingMetadataRow[];
  featured?: boolean;
  /**
   * Marks this card as the current selection in a browse surface (e.g. the card whose map pin
   * is selected, driven by {@link useGdsBrowseSelection}). Renders the identical ring surface
   * treatment as `featured` — one visual language, two semantic triggers. Sets
   * `aria-current="true"` when `interactiveMode` is `'surface-link'`; in every mode the
   * `data-gds-listing-selected` attribute is present, and the composing browse surface owns any
   * list-level ARIA semantics (e.g. `listbox`/`option`) beyond that (issue 701).
   */
  selected?: boolean;
  sponsoredDisclosure?: ReactNode;
  price?: ReactNode;
  rating?: ReactNode;
  ratingLabel?: string;
  primaryAction?: ReactNode;
  /** "Why this fits" reason content (e.g. 2–4 reasons). Rendered as a labeled region. */
  reason?: ReactNode;
  /** Accessible label for the reason region. */
  reasonLabel?: string;
  /** Match-quality element, typically a `<FitScoreChip />`. */
  score?: ReactNode;
  /** Footer affordances (2–4). When present, replaces the default primaryAction footer slot. */
  actions?: ReactNode[];
  /**
   * Content overlaid on the media's top-left corner — typically a category pill. A theme cannot
   * express this (it is composition, not a default), and absolute positioning it consumer-side
   * would be a page-local reimplementation, so the card owns it (issue 679).
   */
  mediaOverlay?: ReactNode;
  /**
   * Affordance overlaid on the media's top-right corner, e.g. a save control that sits on the
   * photo rather than in the footer. Pass `presentation: 'outline-on-media'` for the no-disc form.
   */
  mediaAffordance?: ListingCardAffordance;
  /**
   * Governed editorial "Pick" badge rendered inside the media tile's inline-start/top corner: a
   * pill on a dark scrim with a star glyph, sitting first in the same top-left overlay stack as
   * `mediaOverlay` when both are given. `true` renders the localized default label
   * (`gds.listingCard.pickLabel`); an object overrides the label and/or accessible name. Renders
   * over both a supplied `image` and the generated fallback (issue 701).
   */
  pickBadge?: boolean | ListingCardPickBadge;
  saveAction?: ListingCardAffordance;
  saved?: boolean;
  shareAction?: ListingCardAffordance;
  compact?: boolean;
  size?: GdsCardSize;
  density?: GdsCardDensity;
  variant?: GdsCardVariant;
  interactiveMode?: GdsCardInteractiveMode;
  revealContent?: ReactNode;
  onSurfaceActivate?: () => void;
  defaultFlipped?: boolean;
}


const toneColorMap: Record<NonNullable<ListingMetadataRow['tone']>, string | undefined> = {
  default: undefined,
  positive: 'teal',
  warning: 'orange',
  muted: 'gray',
};

/** Maximum footer affordances a `ListingCard` renders; extra `actions` are dropped past this cap. */
export const MAX_LISTING_CARD_ACTIONS = 4;

/**
 * Row-tile (`media-left`/`compact` card contract) media size, in pixels. Reference-derived from
 * Your Field v3's `ProgramListCard`, which renders its generated media tile at a fixed
 * `flex: 0 0 96px` square (issue 701).
 */
export const GDS_LISTING_TILE_SIZE_PX = 96;

/**
 * Pick badge inset from the media tile's block-start/inline-start corner, in pixels.
 * Reference-derived from `ProgramListCard`'s pick pill, positioned at `left: 6, top: 6` inside
 * the tile (issue 701).
 */
export const GDS_LISTING_PICK_BADGE_INSET_PX = 6;

/** Pick badge star glyph size, in pixels. Reference-derived from `ProgramListCard`'s `9px` icon (issue 701). */
export const GDS_LISTING_PICK_BADGE_ICON_PX = 9;

/** Pick badge label font size, in pixels. Reference-derived from `ProgramListCard`'s `9px` type (issue 701). */
export const GDS_LISTING_PICK_BADGE_FONT_SIZE_PX = 9;

/** Pick badge label line height, in pixels. Reference-derived from `ProgramListCard`'s `13.5px` line height (issue 701). */
export const GDS_LISTING_PICK_BADGE_LINE_HEIGHT_PX = 13.5;

/** Pick badge block-direction (top/bottom) padding, in pixels. Reference-derived from `ProgramListCard`'s `2px 6px` padding (issue 701). */
export const GDS_LISTING_PICK_BADGE_PADDING_BLOCK_PX = 2;

/** Pick badge inline-direction (left/right) padding, in pixels. Reference-derived from `ProgramListCard`'s `2px 6px` padding (issue 701). */
export const GDS_LISTING_PICK_BADGE_PADDING_INLINE_PX = 6;

function resolveCardActions(actions?: ReactNode[]): ReactNode[] | null {
  if (!actions || actions.length === 0) {
    return null;
  }
  if (actions.length > MAX_LISTING_CARD_ACTIONS) {
    throw new Error(`ListingCard supports at most ${MAX_LISTING_CARD_ACTIONS} actions, received ${actions.length}.`);
  }
  return actions;
}

function isNestedInteractiveTarget(eventTarget: EventTarget | null, currentTarget: EventTarget | null) {
  if (!(eventTarget instanceof Element) || !(currentTarget instanceof Element)) {
    return false;
  }

  const nestedInteractive = eventTarget.closest('a, button, input, select, textarea, [role="button"], [role="link"]');
  return Boolean(nestedInteractive && nestedInteractive !== currentTarget);
}

/**
 * Fallback when no image is supplied: deterministic branded art from the listing's identity
 * (same seed, same composition every render), themed by the active preset.
 */
function ListingImageFallback({
  mediaRatio,
  seed,
  title,
}: {
  mediaRatio: ListingCardMediaRatio;
  seed: string;
  title: string;
}) {
  return (
    <GdsGeneratedThumbnail
      seed={seed}
      // Title used as the category label; no invented taxonomy.
      categories={[{ key: 'listing', label: title, icon: 'Gallery' }]}
      // Title already renders below; avoids duplicate text.
      badges="none"
      aspectRatio={mediaRatio === '1:1' ? '1:1' : mediaRatio === '16:9' ? '16:9' : '4:3'}
    />
  );
}

function ListingAffordance({ affordance }: { affordance: ListingCardAffordance }) {
  const config = GdsVocabulary[affordance.action];
  const Icon = config.icon;
  const label = affordance.ariaLabel ?? getSemanticActionLabel(affordance.action);
  const onMedia = affordance.presentation === 'outline-on-media';
  // No disc: the glyph reads directly against the photograph, so it carries its own shadow
  // rather than relying on a container for separation. Tokens, not literals, so any preset
  // can recolour it.
  const mediaStyle = onMedia
    ? {
        background: 'transparent',
        color: 'var(--gds-text-on-inverse, var(--mantine-color-white))',
        filter: 'drop-shadow(0 1px 2px rgba(0, 0, 0, 0.55))',
      }
    : undefined;
  const activeStyle = affordance.active
    ? {
        // Accent on its own tint is 1.60:1 in high-contrast dark; use the tint's derived foreground.
        color: 'var(--gds-brand-accent-tint-fg, var(--gds-brand-accent-action, var(--gds-vibe-accent, var(--mantine-primary-color-filled))))',
        background: 'var(--gds-brand-accent-tint, var(--mantine-color-default-hover))',
      }
    : undefined;

  if (affordance.href) {
    return (
      <ActionIcon
        component="a"
        href={affordance.href}
        variant={onMedia ? 'transparent' : 'subtle'}
        size="xl"
        aria-label={label}
        data-gds-active={affordance.active ? 'true' : undefined}
        style={{ ...mediaStyle, ...activeStyle }}
        disabled={affordance.disabled}
      >
        <Icon size="1rem" stroke={1.75} />
      </ActionIcon>
    );
  }

  return (
    <ActionIcon
      variant={onMedia ? 'transparent' : 'subtle'}
      size="xl"
      aria-label={label}
      onClick={affordance.onClick}
      data-gds-active={affordance.active ? 'true' : undefined}
      style={{ ...mediaStyle, ...activeStyle }}
      disabled={affordance.disabled}
    >
      <Icon size="1rem" stroke={1.75} />
    </ActionIcon>
  );
}

/**
 * Governed "Pick" overlay badge: a pill on a dark scrim, star glyph plus label, rendered inside
 * the media tile's top-left corner. The scrim guarantees legibility over arbitrary media —
 * consumer-supplied photo or the generated fallback alike (issue 701).
 */
function ListingPickBadge({
  config,
  defaultLabel,
}: {
  config: boolean | ListingCardPickBadge;
  defaultLabel: string;
}) {
  const override = typeof config === 'object' ? config : undefined;
  const label = override?.label ?? defaultLabel;
  const ariaLabel = override?.ariaLabel ?? (typeof label === 'string' ? label : defaultLabel);

  return (
    <Group
      gap={4}
      wrap="nowrap"
      align="center"
      role="group"
      aria-label={ariaLabel}
      data-gds-listing-pick-badge
      style={{
        width: 'fit-content',
        borderRadius: 'var(--gds-radius-pill)',
        backgroundColor: 'var(--gds-badge-pick-bg, color-mix(in srgb, var(--gds-text-primary) 60%, transparent))',
        color: 'var(--gds-text-on-inverse, var(--mantine-color-white))',
        fontSize: `${GDS_LISTING_PICK_BADGE_FONT_SIZE_PX}px`,
        lineHeight: `${GDS_LISTING_PICK_BADGE_LINE_HEIGHT_PX}px`,
        fontWeight: 700,
        paddingBlock: `${GDS_LISTING_PICK_BADGE_PADDING_BLOCK_PX}px`,
        paddingInline: `${GDS_LISTING_PICK_BADGE_PADDING_INLINE_PX}px`,
      }}
    >
      <GdsIcons.Star size={GDS_LISTING_PICK_BADGE_ICON_PX} stroke={1.75} fill="currentColor" aria-hidden="true" />
      <span>{label}</span>
    </Group>
  );
}

/**
 * Governed listing/result card for search, catalog, and recommendation surfaces:
 * media, title, description, metadata rows, price/rating, an optional match
 * `score` and "why this fits" `reason` region, and up to
 * {@link MAX_LISTING_CARD_ACTIONS} footer affordances plus save/share actions.
 * Honors the shared card contract (`size`/`density`/`variant`/`interactiveMode`)
 * and supports an optional flip-to-reveal back face via `revealContent`.
 *
 * `featured`/`selected` render an identical ring surface treatment (1px accent border, featured
 * ring, elevated shadow) — never a hover-lift, so selection always reads as a persistent ring,
 * not a transient transform (issue 701).
 */
export function ListingCard({
  title,
  href,
  description,
  image,
  imageAlt,
  mediaSeed,
  mediaRatio = '4:3',
  metadata = [],
  featured = false,
  selected = false,
  sponsoredDisclosure,
  price,
  rating,
  ratingLabel: ratingLabelProp,
  primaryAction,
  reason,
  reasonLabel: reasonLabelProp,
  score,
  actions,
  mediaOverlay,
  mediaAffordance,
  pickBadge,
  saveAction,
  saved = false,
  shareAction,
  compact = false,
  size = 'md',
  density = 'comfortable',
  variant = 'default',
  interactiveMode = 'none',
  revealContent,
  onSurfaceActivate,
  defaultFlipped = false,
}: ListingCardProps) {
  const { t } = useGdsTranslation();
  const ratingLabel = ratingLabelProp ?? t('gds.listingCard.ratingLabel', "Rating");
  const reasonLabel = reasonLabelProp ?? t('gds.listingCard.reasonLabel', "Why this fits");
  const featuredLabel = t('gds.listingCard.featuredLabel', "Featured");
  const pickLabel = t('gds.listingCard.pickLabel', "Pick");

  const [flipped, setFlipped] = useState(defaultFlipped);
  const resolvedActions = resolveCardActions(actions);
  const contract = resolveGdsCardContract({ compact, size, density, variant });
  const resolvedSaveAction = saveAction ? { ...saveAction, active: saveAction.active ?? saved } : undefined;
  const cardPadding = contract.padding;
  const isInteractive = interactiveMode !== 'none';
  const isFlipMode = interactiveMode === 'flip' && Boolean(revealContent);
  const isRowTile = contract.mediaPlacement === 'left';
  const emphasized = featured || selected;
  const titleContent =
    href && typeof title === 'string' && interactiveMode === 'none' ? (
      <Text component="a" href={href} inherit td="none">
        {title}
      </Text>
    ) : (
      title
    );

  const activateSurface = (event: MouseEvent<HTMLElement> | KeyboardEvent<HTMLElement>) => {
    if (isNestedInteractiveTarget(event.target, event.currentTarget)) {
      return;
    }

    if (isFlipMode) {
      setFlipped((current) => !current);
      onSurfaceActivate?.();
      return;
    }

    if (interactiveMode === 'surface-button') {
      onSurfaceActivate?.();
      return;
    }

    if (interactiveMode === 'surface-link' && href) {
      onSurfaceActivate?.();
      if (typeof window !== 'undefined') {
        window.location.assign(href);
      }
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (!isInteractive || isNestedInteractiveTarget(event.target, event.currentTarget)) {
      return;
    }

    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      activateSurface(event);
    }
  };

  const surfaceLabel = typeof title === 'string' ? title : 'listing';
  const interactiveProps = isInteractive
    ? {
        role: interactiveMode === 'surface-link' ? 'link' : 'button',
        tabIndex: 0,
        onClick: activateSurface,
        onKeyDown: handleKeyDown,
        'aria-expanded': isFlipMode ? flipped : undefined,
        'aria-current': selected && interactiveMode === 'surface-link' ? ('true' as const) : undefined,
        'aria-label': isFlipMode ? `Toggle details for ${surfaceLabel}` : surfaceLabel,
      }
    : {};

  // Same seed/title resolution the fallback thumbnail has always used (regression-pinned).
  const mediaSeedValue = mediaSeed ?? href ?? imageAlt ?? (typeof title === 'string' ? title : 'gds-listing');
  const fallbackTitle = typeof title === 'string' ? title : (imageAlt ?? 'Listing');
  const mediaContent =
    image ?? (
      <ListingImageFallback
        // The row tile is a fixed square frame; the fallback fills it edge to edge rather than
        // rendering at the card's own (possibly non-square) mediaRatio and letterboxing.
        mediaRatio={isRowTile ? '1:1' : mediaRatio}
        seed={mediaSeedValue}
        title={fallbackTitle}
      />
    );

  const pickBadgeElement = pickBadge ? <ListingPickBadge config={pickBadge} defaultLabel={pickLabel} /> : null;

  // Pick badge and mediaOverlay share one top-left stack (badge first) when both are present;
  // an overlay-only card keeps its original standalone wrapper exactly (issue 679 invariant).
  const topLeftOverlay = pickBadgeElement ? (
    <Box
      style={{
        position: 'absolute',
        insetBlockStart: `${GDS_LISTING_PICK_BADGE_INSET_PX}px`,
        insetInlineStart: `${GDS_LISTING_PICK_BADGE_INSET_PX}px`,
        zIndex: 1,
      }}
    >
      {mediaOverlay ? (
        <Stack gap="xs">
          {pickBadgeElement}
          {mediaOverlay}
        </Stack>
      ) : (
        pickBadgeElement
      )}
    </Box>
  ) : mediaOverlay ? (
    <Box style={{ position: 'absolute', top: 'var(--mantine-spacing-xs)', left: 'var(--mantine-spacing-xs)', zIndex: 1 }}>
      {mediaOverlay}
    </Box>
  ) : null;

  const affordanceOverlay = mediaAffordance ? (
    <Box style={{ position: 'absolute', top: 0, right: 0, zIndex: 1 }}>
      <ListingAffordance affordance={mediaAffordance} />
    </Box>
  ) : null;

  const hasMediaOverlayContent = Boolean(mediaOverlay || mediaAffordance || pickBadge);

  const mediaBlock = isRowTile ? (
    <Box
      data-gds-listing-row-tile
      style={{
        position: 'relative',
        flex: `0 0 ${GDS_LISTING_TILE_SIZE_PX}px`,
        width: GDS_LISTING_TILE_SIZE_PX,
        height: GDS_LISTING_TILE_SIZE_PX,
        overflow: 'hidden',
        borderRadius: 'var(--gds-radius-thumbnail)',
      }}
    >
      {mediaContent}
      {topLeftOverlay}
      {affordanceOverlay}
    </Box>
  ) : hasMediaOverlayContent ? (
    // Positioned wrapper only when something is actually overlaid, so a card without
    // overlays keeps its previous DOM exactly (issue 679).
    <Box style={{ position: 'relative' }}>
      {mediaContent}
      {topLeftOverlay}
      {affordanceOverlay}
    </Box>
  ) : (
    mediaContent
  );

  const contentItems = (
    <>
      {(featured || sponsoredDisclosure) ? (
        <Group justify="space-between" gap="sm" wrap="wrap">
          {featured ? (
            <Badge
              variant="light"
              style={{
                color: 'var(--gds-brand-accent-tint-fg, var(--gds-brand-accent-action, var(--gds-vibe-accent, var(--mantine-primary-color-filled))))',
                backgroundColor: 'var(--gds-brand-accent-tint, var(--mantine-color-default-hover))',
              }}
            >
              {featuredLabel}
            </Badge>
          ) : (
            <span />
          )}
          {sponsoredDisclosure ? (
            <Text size="xs" c="dimmed">
              {sponsoredDisclosure}
            </Text>
          ) : null}
        </Group>
      ) : null}

      <Stack gap={4}>
        <Title order={contract.titleOrder} lineClamp={2}>
          {titleContent}
        </Title>
        {description ? (
          <Text size="sm" c="dimmed" lineClamp={contract.descriptionClamp}>
            {description}
          </Text>
        ) : null}
      </Stack>

      {metadata.length ? (
        <Stack gap="xs">
          {metadata.map((item) => (
            <Group key={item.id} justify="space-between" align="flex-start" gap="sm" wrap="nowrap">
              <Group gap="xs" wrap="nowrap" style={{ minWidth: 0, flex: 1 }}>
                {item.icon}
                <Text size="sm" c={item.tone ? toneColorMap[item.tone] : 'dimmed'} lineClamp={1}>
                  {item.label}
                </Text>
              </Group>
              {item.value ? (
                <Text size="sm" fw={500} ta="right">
                  {item.value}
                </Text>
              ) : null}
            </Group>
          ))}
        </Stack>
      ) : null}

      {reason ? (
        <Stack gap={4} role="group" aria-label={reasonLabel}>
          <Text size="xs" fw={700} tt="uppercase" c="dimmed">
            {reasonLabel}
          </Text>
          {reason}
        </Stack>
      ) : null}

      <Group justify="space-between" align="center" gap="sm" wrap="wrap">
        <Stack gap={2} style={{ minWidth: 0, flex: 1 }}>
          {price ? (
            <Text
              fw={700}
              size={contract.size === 'xs' || contract.size === 'sm' ? 'md' : 'lg'}
              style={{ color: 'var(--gds-price, var(--gds-brand-accent-action, var(--gds-vibe-accent, var(--mantine-primary-color-filled))))' }}
            >
              {price}
            </Text>
          ) : null}
          {rating ? (
            <Group gap={4} wrap="nowrap" aria-label={ratingLabel}>
              <GdsIcons.Star
                size="1rem"
                stroke={1.75}
                fill="currentColor"
                style={{ color: 'var(--gds-star, var(--gds-brand-accent-action, var(--gds-vibe-accent, var(--mantine-primary-color-filled))))' }}
              />
              <Text size="sm" fw={600} style={{ color: 'var(--gds-star, var(--gds-brand-accent-action, var(--gds-vibe-accent, var(--mantine-primary-color-filled))))' }}>
                {rating}
              </Text>
            </Group>
          ) : null}
        </Stack>

        {score ? <Group gap="xs" wrap="nowrap">{score}</Group> : null}

        <Group gap="xs" wrap="nowrap" justify="flex-end" style={{ marginInlineStart: 'auto' }}>
          {resolvedSaveAction ? <ListingAffordance affordance={resolvedSaveAction} /> : null}
          {shareAction ? <ListingAffordance affordance={shareAction} /> : null}
          {resolvedActions ? null : primaryAction}
          {isFlipMode ? <Text size="xs" c="dimmed">Press Enter or Space to reveal details.</Text> : null}
        </Group>
      </Group>

      {resolvedActions ? (
        <Group gap="sm" wrap="wrap" role="group" aria-label="Listing actions">
          {resolvedActions.map((action, index) => (
            // eslint-disable-next-line react/no-array-index-key
            <span key={index}>{action}</span>
          ))}
        </Group>
      ) : null}
    </>
  );

  return (
    <Card
      withBorder
      radius="lg"
      padding={cardPadding}
      {...contract.dataAttributes}
      data-gds-listing-card
      data-gds-card-interactive-mode={interactiveMode}
      data-gds-card-flipped={isFlipMode ? String(flipped) : undefined}
      data-gds-listing-featured={featured ? 'true' : undefined}
      data-gds-listing-selected={selected ? 'true' : undefined}
      style={{
        background: 'var(--gds-bg-card, var(--gds-vibe-surface, var(--mantine-color-body)))',
        borderColor: 'var(--gds-border-card, var(--gds-vibe-border, var(--mantine-color-default-border)))',
        ...(isInteractive ? { cursor: 'pointer', transition: 'transform var(--gds-motion-duration-fast) var(--gds-motion-ease-standard), box-shadow var(--gds-motion-duration-fast) var(--gds-motion-ease-standard)' } : {}),
        // Emphasis is a ring, never a lift: no transform is set here, in any state.
        ...(emphasized
          ? {
              borderColor: 'var(--gds-listing-featured-border, var(--gds-brand-accent-action, var(--gds-vibe-accent, var(--mantine-primary-color-filled))))',
              boxShadow: 'var(--gds-ring-featured, 0 0 0 3px color-mix(in srgb, var(--gds-vibe-accent) 14%, transparent)), var(--gds-elevation-panel)',
            }
          : {}),
      }}
      {...interactiveProps}
    >
      <Stack gap={contract.gap}>
        {isFlipMode && flipped ? (
          revealContent
        ) : isRowTile ? (
          <Group gap={contract.gap} wrap="nowrap" align="flex-start" data-gds-listing-row>
            {mediaBlock}
            <Stack gap={contract.gap} style={{ minWidth: 0, flex: 1 }}>
              {contentItems}
            </Stack>
          </Group>
        ) : (
          <>
            {mediaBlock}
            {contentItems}
          </>
        )}
      </Stack>
    </Card>
  );
}
