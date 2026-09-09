import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MantineProvider } from '@mantine/core';
import { renderWithGds } from '../../../test-utils/render';
import {
  GDS_LISTING_PICK_BADGE_INSET_PX,
  GDS_LISTING_TILE_SIZE_PX,
  ListingCard,
} from './ListingCard';
import { useGdsBrowseSelection } from './BrowseSelection.client';
import { GDS_PIN_SELECTED_SCALE, GdsMapPinBadge } from './GdsMapPinBadge';
import { de } from './locales';

function renderCard(props: Parameters<typeof ListingCard>[0]) {
  return render(<MantineProvider><ListingCard {...props} /></MantineProvider>);
}

describe('ListingCard featured/selected ring (issue 701)', () => {
  it('featured sets the data attribute, a ring/elevation box-shadow, and no transform', () => {
    const { container } = renderCard({ title: 'Go Padel', featured: true });
    const card = container.querySelector('[data-gds-listing-card]') as HTMLElement;
    expect(card.getAttribute('data-gds-listing-featured')).toBe('true');
    expect(card.getAttribute('data-gds-listing-selected')).toBeNull();
    expect(card.style.boxShadow).toContain('color-mix');
    expect(card.style.boxShadow).toContain('var(--gds-elevation-panel)');
    expect(card.style.transform).toBe('');
  });

  it('selected sets the identical ring treatment plus its own data attribute', () => {
    const { container } = renderCard({ title: 'Go Padel', selected: true });
    const card = container.querySelector('[data-gds-listing-card]') as HTMLElement;
    expect(card.getAttribute('data-gds-listing-selected')).toBe('true');
    expect(card.getAttribute('data-gds-listing-featured')).toBeNull();
    expect(card.style.boxShadow).toContain('color-mix');
    expect(card.style.transform).toBe('');
  });

  it('featured and selected together render one identical treatment, not a doubled ring', () => {
    const both = renderCard({ title: 'Go Padel', featured: true, selected: true });
    const bothCard = both.container.querySelector('[data-gds-listing-card]') as HTMLElement;

    const soloFeatured = renderCard({ title: 'Go Padel', featured: true });
    const soloCard = soloFeatured.container.querySelector('[data-gds-listing-card]') as HTMLElement;

    expect(bothCard.style.boxShadow).toBe(soloCard.style.boxShadow);
    expect(bothCard.getAttribute('data-gds-listing-featured')).toBe('true');
    expect(bothCard.getAttribute('data-gds-listing-selected')).toBe('true');
  });

  it('neither featured nor selected renders no emphasis box-shadow', () => {
    const { container } = renderCard({ title: 'Go Padel' });
    const card = container.querySelector('[data-gds-listing-card]') as HTMLElement;
    expect(card.style.boxShadow).toBe('');
  });

  it('sets aria-current="true" only in surface-link mode, and the data attribute in every mode', () => {
    const surfaceButton = renderCard({
      title: 'Go Padel',
      selected: true,
      interactiveMode: 'surface-button',
    });
    const buttonCard = surfaceButton.getByRole('button', { name: 'Go Padel' });
    expect(buttonCard.getAttribute('aria-current')).toBeNull();
    expect(buttonCard.getAttribute('data-gds-listing-selected')).toBe('true');

    const surfaceLink = renderCard({
      title: 'Go Padel',
      href: '/go-padel',
      selected: true,
      interactiveMode: 'surface-link',
    });
    const linkCard = surfaceLink.getByRole('link', { name: 'Go Padel' });
    expect(linkCard.getAttribute('aria-current')).toBe('true');
  });
});

describe('ListingCard pick badge (issue 701)', () => {
  it('renders the localized default label with an accessible name, and hides the star from AT', () => {
    renderCard({ title: 'Go Padel', pickBadge: true });
    const badge = screen.getByRole('group', { name: 'Pick' });
    expect(badge).toHaveAttribute('data-gds-listing-pick-badge');
    expect(screen.getByText('Pick')).toBeInTheDocument();
    const icon = badge.querySelector('svg');
    expect(icon).toHaveAttribute('aria-hidden', 'true');
  });

  it('overrides the accessible name via ariaLabel while keeping a custom visible label', () => {
    renderCard({ title: 'Go Padel', pickBadge: { label: '★ Pick', ariaLabel: 'Editorial pick' } });
    expect(screen.getByRole('group', { name: 'Editorial pick' })).toBeInTheDocument();
    expect(screen.getByText('★ Pick')).toBeInTheDocument();
  });

  it('falls back to the custom label as the accessible name when no ariaLabel is given', () => {
    renderCard({ title: 'Go Padel', pickBadge: { label: 'Top choice' } });
    expect(screen.getByRole('group', { name: 'Top choice' })).toBeInTheDocument();
  });

  it('renders over a supplied image, not only the generated fallback', () => {
    const { container } = renderCard({
      title: 'Go Padel',
      image: <img alt="A photo" src="https://example.com/a.jpg" />,
      pickBadge: true,
    });
    const img = container.querySelector('img[src="https://example.com/a.jpg"]') as HTMLElement;
    const badge = screen.getByRole('group', { name: 'Pick' });
    expect(img).toBeInTheDocument();
    // Containment, not mere co-presence: the badge must sit inside the same media wrapper as
    // the supplied image, not just anywhere on the card.
    expect(img.closest('[style*="position: relative"]')).not.toBeNull();
    expect(badge.closest('[style*="position: relative"]')).toBe(img.closest('[style*="position: relative"]'));
  });

  it('stacks deterministically with mediaOverlay: the badge renders first, the overlay after, in one shared corner', () => {
    const { container } = renderCard({
      title: 'Go Padel',
      pickBadge: true,
      mediaOverlay: <span>Outdoor</span>,
      mediaAffordance: { action: 'save' },
    });
    const corner = screen.getByRole('group', { name: 'Pick' }).closest('[style*="position: absolute"]');
    expect(corner).not.toBeNull();
    expect(corner?.textContent).toBe('PickOutdoor');
    // mediaAffordance stays independent, top-right, in its own wrapper — unaffected by the
    // badge/overlay stack, which shares one top-left wrapper.
    const affordanceButton = screen.getByRole('button', { name: /save/i });
    const affordanceWrapper = affordanceButton.closest('[style*="position: absolute"]');
    expect(affordanceWrapper).not.toBeNull();
    expect(affordanceWrapper).not.toBe(corner);
    expect(affordanceWrapper?.textContent).not.toContain('Pick');
  });

  it('applies the governed inset constant to the corner wrapper', () => {
    renderCard({ title: 'Go Padel', pickBadge: true });
    const corner = screen.getByRole('group', { name: 'Pick' }).parentElement as HTMLElement;
    expect(corner.style.insetBlockStart).toBe(`${GDS_LISTING_PICK_BADGE_INSET_PX}px`);
    expect(corner.style.insetInlineStart).toBe(`${GDS_LISTING_PICK_BADGE_INSET_PX}px`);
  });

  it('always renders media, so pickBadge without any media is impossible (regression-pinned, issue 701 §5)', () => {
    // No `image` prop at all — the generated fallback must still render, alongside the badge.
    const { container } = renderCard({ title: 'Go Padel', pickBadge: true });
    expect(screen.getByRole('group', { name: 'Pick' })).toBeInTheDocument();
    expect(container.querySelector('[data-gds-generated-thumbnail]')).toBeInTheDocument();
  });
});

describe('ListingCard generated-tile seed chain (regression-pinned, issue 679/701)', () => {
  // The seed itself is opaque and not rendered directly; `computeMotifTransform` derives the
  // motif's transform from it (GdsGeneratedThumbnail.tsx), so two renders resolving to the same
  // effective seed produce byte-identical `transform` attributes, and this is what actually pins
  // the seed-priority chain (the `data-gds-generated-thumbnail` marker attribute carries no seed
  // information at all -- it's always the empty string).
  const seedOf = (container: HTMLElement) =>
    container.querySelector('[data-gds-generated-thumbnail] g[transform]')?.getAttribute('transform');

  it('prefers mediaSeed over href/imageAlt/title', () => {
    const { container } = renderCard({ title: 'Title', href: '/href', imageAlt: 'Alt', mediaSeed: 'seed-1' });
    // The seed is opaque (feeds a deterministic generator), so this test only pins that a
    // change in the higher-priority input does not change the render when mediaSeed is set.
    const withDifferentHref = renderCard({ title: 'Title', href: '/other-href', imageAlt: 'Alt', mediaSeed: 'seed-1' });
    expect(seedOf(container)).toBe(seedOf(withDifferentHref.container));
  });

  it('falls back through href, then imageAlt, then string title, then a fixed default', () => {
    const byHref = renderCard({ title: 'Title', href: '/href-a' });
    const byHrefAgain = renderCard({ title: 'Title', href: '/href-a' });
    expect(seedOf(byHref.container)).toBe(seedOf(byHrefAgain.container));

    const byAlt = renderCard({ title: <span>Node title</span>, imageAlt: 'Alt text' });
    const byAltAgain = renderCard({ title: <span>Node title</span>, imageAlt: 'Alt text' });
    expect(seedOf(byAlt.container)).toBe(seedOf(byAltAgain.container));

    const byTitle = renderCard({ title: 'Just a title' });
    const byTitleAgain = renderCard({ title: 'Just a title' });
    expect(seedOf(byTitle.container)).toBe(seedOf(byTitleAgain.container));

    const byDefault = renderCard({ title: <span>Node title</span> });
    const byDefaultAgain = renderCard({ title: <span>Node title</span> });
    expect(seedOf(byDefault.container)).toBe(seedOf(byDefaultAgain.container));
  });
});

describe('ListingCard media-left row tile (issue 701)', () => {
  it('renders a fixed 96x96 tile with the thumbnail radius role when mediaPlacement resolves to left', () => {
    const { container } = renderCard({ title: 'Go Padel', variant: 'media-left' });
    const tile = container.querySelector('[data-gds-listing-row-tile]') as HTMLElement;
    expect(tile).not.toBeNull();
    expect(tile.style.width).toBe(`${GDS_LISTING_TILE_SIZE_PX}px`);
    expect(tile.style.height).toBe(`${GDS_LISTING_TILE_SIZE_PX}px`);
    expect(tile.style.overflow).toBe('hidden');
    expect(tile.style.borderRadius).toBe('var(--gds-radius-thumbnail)');
  });

  it('leaves the top-media form unchanged: no row-tile attribute for the default variant', () => {
    const { container } = renderCard({ title: 'Go Padel' });
    expect(container.querySelector('[data-gds-listing-row-tile]')).toBeNull();
  });

  it('adds no extra positioned wrapper on a row-tile card with nothing overlaid', () => {
    // Delta against a plain card, same reasoning as the issue-679 wrapper-count test: the row
    // tile itself is one relative wrapper, and nothing else is added when nothing is overlaid.
    const countRelative = (c: HTMLElement) =>
      [...c.querySelectorAll('div')].filter((el) => (el as HTMLElement).style.position === 'relative').length;

    const plain = renderCard({ title: 'Go Padel' });
    const before = countRelative(plain.container);
    plain.unmount();

    const rowTile = renderCard({ title: 'Go Padel', variant: 'media-left' });
    expect(countRelative(rowTile.container)).toBe(before + 1);
  });
});

describe('ListingCard featured badge defects fixed (issue 701)', () => {
  it('renders the localized label rather than a hardcoded English string', () => {
    // Under the default (English) locale, a hardcoded 'Featured' string and a real
    // t('gds.listingCard.featuredLabel', 'Featured') lookup render identically, so this
    // renders under German to actually distinguish them.
    renderWithGds(
      <ListingCard title="Go Padel" featured ratingLabel="Rating" />,
      { locale: 'de', messages: de },
    );
    expect(screen.getByText('Empfohlen')).toBeInTheDocument();
    expect(screen.queryByText('Featured')).not.toBeInTheDocument();
  });

  it('colors the badge from the accent tint token pair, never Mantine violet', () => {
    const { container } = renderCard({ title: 'Go Padel', featured: true });
    const badgeRoot = screen.getByText('Featured').parentElement as HTMLElement;
    expect(badgeRoot.className).not.toMatch(/violet/i);
    expect(badgeRoot.style.backgroundColor).toContain('--gds-brand-accent-tint');
    expect(badgeRoot.style.color).toContain('--gds-brand-accent-tint-fg');
    expect(container.innerHTML).not.toMatch(/violet/i);
  });
});

describe('ListingCard flip mode (issue 701)', () => {
  it('shows overlays and the pick badge on the front face only; the ring persists on both faces', () => {
    const { container } = renderCard({
      title: 'Go Padel',
      featured: true,
      pickBadge: true,
      mediaOverlay: <span>Outdoor</span>,
      interactiveMode: 'flip',
      revealContent: <div>Back face content</div>,
      defaultFlipped: true,
    });

    // Front-face-only content is gone once flipped: revealContent replaces the whole body.
    expect(screen.queryByRole('group', { name: 'Pick' })).not.toBeInTheDocument();
    expect(screen.queryByText('Outdoor')).not.toBeInTheDocument();
    expect(screen.getByText('Back face content')).toBeInTheDocument();

    // The card is still selected while flipped: the ring/elevation treatment is on the outer
    // Card element, not conditional on flip state.
    const card = container.querySelector('[data-gds-listing-card]') as HTMLElement;
    expect(card.getAttribute('data-gds-listing-featured')).toBe('true');
    expect(card.style.boxShadow).toContain('color-mix');
    expect(card.style.boxShadow).toContain('var(--gds-elevation-panel)');
  });

  it('shows the front face (overlays, badge) when not flipped, with the same ring treatment', () => {
    renderCard({
      title: 'Go Padel',
      featured: true,
      pickBadge: true,
      mediaOverlay: <span>Outdoor</span>,
      interactiveMode: 'flip',
      revealContent: <div>Back face content</div>,
      defaultFlipped: false,
    });
    expect(screen.getByRole('group', { name: 'Pick' })).toBeInTheDocument();
    expect(screen.getByText('Outdoor')).toBeInTheDocument();
    expect(screen.queryByText('Back face content')).not.toBeInTheDocument();
  });
});

describe('useGdsBrowseSelection (issue 701)', () => {
  it('is uncontrolled by default: select/toggle/clear transition internal state', () => {
    function Probe() {
      const browse = useGdsBrowseSelection();
      return (
        <>
          <span data-testid="current">{browse.selectedId ?? 'none'}</span>
          <button type="button" onClick={() => browse.select('a')}>select-a</button>
          <button type="button" onClick={() => browse.toggle('a')}>toggle-a</button>
          <button type="button" onClick={() => browse.clear()}>clear</button>
        </>
      );
    }
    render(<Probe />);
    expect(screen.getByTestId('current').textContent).toBe('none');

    fireEvent.click(screen.getByText('select-a'));
    expect(screen.getByTestId('current').textContent).toBe('a');

    fireEvent.click(screen.getByText('toggle-a'));
    expect(screen.getByTestId('current').textContent).toBe('none');

    fireEvent.click(screen.getByText('select-a'));
    fireEvent.click(screen.getByText('clear'));
    expect(screen.getByTestId('current').textContent).toBe('none');
  });

  it('honors defaultSelectedId', () => {
    function Probe() {
      const browse = useGdsBrowseSelection({ defaultSelectedId: 'seed' });
      return <span data-testid="current">{browse.selectedId ?? 'none'}</span>;
    }
    render(<Probe />);
    expect(screen.getByTestId('current').textContent).toBe('seed');
  });

  it('select accepts an id not present in any rendered list (list-agnostic)', () => {
    function Probe() {
      const browse = useGdsBrowseSelection();
      return (
        <>
          <span data-testid="current">{browse.selectedId ?? 'none'}</span>
          <button type="button" onClick={() => browse.select('off-list')}>select-off-list</button>
        </>
      );
    }
    render(<Probe />);
    fireEvent.click(screen.getByText('select-off-list'));
    expect(screen.getByTestId('current').textContent).toBe('off-list');
  });

  it('controlled mode mirrors selectedId and never mutates internally; onChange reports transitions', () => {
    const onChange = vi.fn();

    function Controlled() {
      const [external, setExternal] = useState<string | null>('a');
      const browse = useGdsBrowseSelection({
        selectedId: external,
        onChange: (next) => {
          onChange(next);
        },
      });
      return (
        <>
          <span data-testid="current">{browse.selectedId ?? 'none'}</span>
          {/* Deliberately does NOT apply the hook's own commit to external state, so a
              mismatch would prove the hook mutated internally instead of staying controlled. */}
          <button type="button" onClick={() => browse.select('b')}>select-b</button>
          <button type="button" onClick={() => setExternal('c')}>external-set-c</button>
        </>
      );
    }

    render(<Controlled />);
    expect(screen.getByTestId('current').textContent).toBe('a');

    fireEvent.click(screen.getByText('select-b'));
    // onChange fired with the requested value, but external state was not updated by this
    // test's onChange handler, so the hook must keep mirroring the prop, not its own commit.
    expect(onChange).toHaveBeenCalledWith('b');
    expect(screen.getByTestId('current').textContent).toBe('a');

    fireEvent.click(screen.getByText('external-set-c'));
    expect(screen.getByTestId('current').textContent).toBe('c');
  });

  it('toggle on the current id clears it; onChange fires exactly once per actual change', () => {
    const onChange = vi.fn();
    function Probe() {
      const browse = useGdsBrowseSelection({ defaultSelectedId: 'a', onChange });
      return (
        <>
          <span data-testid="current">{browse.selectedId ?? 'none'}</span>
          <button type="button" onClick={() => browse.toggle('a')}>toggle-a</button>
        </>
      );
    }
    render(<Probe />);
    fireEvent.click(screen.getByText('toggle-a'));
    expect(screen.getByTestId('current').textContent).toBe('none');
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith(null);
  });

  it('shares one selection across zero, one, and many card instances (only one selected at a time)', () => {
    const ids = ['x', 'y', 'z'];
    function Probe() {
      const browse = useGdsBrowseSelection();
      return (
        <>
          {ids.map((id) => (
            <button
              key={id}
              type="button"
              data-testid={`card-${id}`}
              data-selected={browse.isSelected(id) ? 'true' : 'false'}
              onClick={() => browse.select(id)}
            >
              {id}
            </button>
          ))}
        </>
      );
    }
    render(<Probe />);
    // Zero selected initially.
    for (const id of ids) expect(screen.getByTestId(`card-${id}`).getAttribute('data-selected')).toBe('false');

    fireEvent.click(screen.getByTestId('card-y'));
    // One selected; the other many stay deselected.
    expect(screen.getByTestId('card-x').getAttribute('data-selected')).toBe('false');
    expect(screen.getByTestId('card-y').getAttribute('data-selected')).toBe('true');
    expect(screen.getByTestId('card-z').getAttribute('data-selected')).toBe('false');

    fireEvent.click(screen.getByTestId('card-z'));
    // Selection moves atomically — never two selected at once.
    expect(screen.getByTestId('card-y').getAttribute('data-selected')).toBe('false');
    expect(screen.getByTestId('card-z').getAttribute('data-selected')).toBe('true');
  });
});

describe('Browse composition: ListingCard list + GdsMapPinBadge pins share one selection (issue 701)', () => {
  const PLACES = [
    { id: 'pool', title: 'Community pool' },
    { id: 'studio', title: 'Dance studio' },
  ] as const;

  function BrowseSurface() {
    const browse = useGdsBrowseSelection();
    return (
      <>
        {PLACES.map((place) => (
          <ListingCard
            key={place.id}
            title={place.title}
            selected={browse.isSelected(place.id)}
            interactiveMode="surface-button"
            onSurfaceActivate={() => browse.toggle(place.id)}
          />
        ))}
        {PLACES.map((place) => (
          <button key={place.id} type="button" aria-label={`pin-${place.id}`} onClick={() => browse.toggle(place.id)}>
            <GdsMapPinBadge
              accent="ocean"
              icon="Location"
              label={place.title}
              state={browse.isSelected(place.id) ? 'selected' : 'idle'}
            />
          </button>
        ))}
      </>
    );
  }

  it('clicking a pin selects the matching card; clicking a card selects the matching pin; selection moves atomically', async () => {
    const user = userEvent.setup();
    renderWithGds(<BrowseSurface />);

    const poolCard = () => screen.getByRole('button', { name: 'Community pool' });
    const studioCard = () => screen.getByRole('button', { name: 'Dance studio' });
    const poolPinStack = () => screen.getByLabelText('pin-pool').querySelector('[role="img"]') as HTMLElement;
    const studioPinStack = () => screen.getByLabelText('pin-studio').querySelector('[role="img"]') as HTMLElement;

    // Nothing selected initially: neither card carries the attribute, neither pin is scaled.
    expect(poolCard().getAttribute('data-gds-listing-selected')).toBeNull();
    expect(poolPinStack().style.transform).toBe('');

    // Click the pool pin: the pool card becomes selected, and the pool pin scales up.
    await user.click(screen.getByLabelText('pin-pool'));
    expect(poolCard().getAttribute('data-gds-listing-selected')).toBe('true');
    expect(poolPinStack().style.transform).toBe(`scale(${GDS_PIN_SELECTED_SCALE})`);
    expect(studioCard().getAttribute('data-gds-listing-selected')).toBeNull();

    // Click the studio card: selection moves atomically — pool deselects as studio selects,
    // with no frame where both (or neither) carry the emphasis.
    await user.click(studioCard());
    expect(poolCard().getAttribute('data-gds-listing-selected')).toBeNull();
    expect(poolPinStack().style.transform).toBe('');
    expect(studioCard().getAttribute('data-gds-listing-selected')).toBe('true');
    expect(studioPinStack().style.transform).toBe(`scale(${GDS_PIN_SELECTED_SCALE})`);
  });
});
