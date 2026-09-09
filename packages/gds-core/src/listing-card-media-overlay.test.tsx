import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MantineProvider } from '@mantine/core';
import { ListingCard } from './ListingCard';

function renderCard(props: Parameters<typeof ListingCard>[0]) {
  return render(<MantineProvider><ListingCard {...props} /></MantineProvider>);
}

describe('ListingCard media overlays (issue 679)', () => {
  it('renders overlay content over the media', () => {
    renderCard({ title: 'Go Padel', mediaOverlay: <span>Outdoor</span> });
    expect(screen.getByText('Outdoor')).toBeInTheDocument();
  });

  it('renders a media affordance with an accessible name', () => {
    renderCard({
      title: 'Go Padel',
      mediaAffordance: { action: 'save', ariaLabel: 'Save Go Padel', presentation: 'outline-on-media' },
    });
    expect(screen.getByRole('button', { name: 'Save Go Padel' })).toBeInTheDocument();
  });

  // The point of the presentation: legible directly on a photograph, which means no disc and a
  // shadow of its own rather than a container.
  it('drops the disc and carries a shadow in outline-on-media presentation', () => {
    renderCard({
      title: 'Go Padel',
      mediaAffordance: { action: 'save', ariaLabel: 'Save', presentation: 'outline-on-media' },
    });
    const button = screen.getByRole('button', { name: 'Save' });
    expect(button.style.background).toBe('transparent');
    expect(button.style.filter).toContain('drop-shadow');
  });

  it('leaves the contained presentation unchanged', () => {
    renderCard({ title: 'Go Padel', mediaAffordance: { action: 'save', ariaLabel: 'Save' } });
    const button = screen.getByRole('button', { name: 'Save' });
    expect(button.style.background).not.toBe('transparent');
    expect(button.style.filter).toBe('');
  });

  // Additive only: the positioned wrapper appears exactly when something is overlaid. Counted as a
  // delta rather than an absolute, because ListingCard already contains a relative element of its
  // own -- asserting zero measured that pre-existing node instead of this feature.
  it('adds the positioned wrapper only when something is overlaid', () => {
    const countRelative = (c: HTMLElement) =>
      [...c.querySelectorAll('div')].filter((el) => (el as HTMLElement).style.position === 'relative').length;

    const plain = renderCard({ title: 'Go Padel' });
    const before = countRelative(plain.container);
    plain.unmount();

    const overlaid = renderCard({ title: 'Go Padel', mediaOverlay: <span>Outdoor</span> });
    expect(countRelative(overlaid.container)).toBe(before + 1);
  });

  // Extended for issue 701: the pick badge is a third overlay/badge trigger for the same
  // positioned wrapper, on its own, with no mediaOverlay/mediaAffordance present.
  it('adds the positioned wrapper for pickBadge alone too', () => {
    const countRelative = (c: HTMLElement) =>
      [...c.querySelectorAll('div')].filter((el) => (el as HTMLElement).style.position === 'relative').length;

    const plain = renderCard({ title: 'Go Padel' });
    const before = countRelative(plain.container);
    plain.unmount();

    const picked = renderCard({ title: 'Go Padel', pickBadge: true });
    expect(countRelative(picked.container)).toBe(before + 1);
  });
});
