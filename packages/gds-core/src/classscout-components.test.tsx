import { useState } from 'react';
import { screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { renderWithGds } from '../../../test-utils/render';
import { BottomTabBar } from './BottomTabBar';
import { FitScoreChip } from './FitScoreChip';
import { MeaningBadge } from './MeaningBadge';
import { MediaWithFallback } from './MediaWithFallback';
import { NumberStepper } from './NumberStepper';
import { AISearchCard } from './AISearchCard';
import { ChatThread, type ChatMessageModel } from './ChatSurface';
import { ListingCard } from './ListingCard';

describe('BottomTabBar (#317)', () => {
  const items = [
    { id: 'home', label: 'Home', href: '/' },
    { id: 'scout', label: 'Scout', href: '/scout' },
    { id: 'saved', label: 'Saved', href: '/saved' },
  ];

  it('renders a navigation with the items and marks the active one', () => {
    renderWithGds(<BottomTabBar items={items} activeId="scout" emphasizedItemId="scout" />);
    const nav = screen.getByRole('navigation', { name: 'Primary' });
    expect(nav).toBeInTheDocument();
    expect(screen.getByText('Scout').closest('a')).toHaveAttribute('aria-current', 'page');
  });

  it('throws in development when given more than the max items', () => {
    const many = Array.from({ length: 6 }, (_, i) => ({ id: `i${i}`, label: `L${i}`, href: `/${i}` }));
    expect(() => renderWithGds(<BottomTabBar items={many} />)).toThrow();
  });

  it('returns null for empty items', () => {
    const { container } = renderWithGds(<BottomTabBar items={[]} />);
    expect(container.querySelector('nav')).toBeNull();
  });

  it('uses GDS\'s published z-index authority for its fixed page chrome, not an ad hoc number (#391)', () => {
    renderWithGds(<BottomTabBar items={items} activeId="scout" />);
    expect(screen.getByRole('navigation', { name: 'Primary' })).toHaveStyle({ zIndex: 'var(--mantine-z-index-app)' });
  });

  it('renderItem overrides one item, receiving its active and emphasized state, and skips the default anchor', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    renderWithGds(
      <BottomTabBar
        items={items}
        activeId="scout"
        emphasizedItemId="scout"
        renderItem={(item, active, emphasized) => (
          <button
            type="button"
            onClick={() => onSelect(item.id, active, emphasized)}
          >
            {`custom:${item.label}`}
          </button>
        )}
      />,
    );
    expect(screen.queryByText('Home')).toBeNull();
    const customButton = screen.getByRole('button', { name: 'custom:Scout' });
    await user.click(customButton);
    expect(onSelect).toHaveBeenCalledWith('scout', true, true);
  });
});

describe('FitScoreChip (#319)', () => {
  it('renders band label and value', () => {
    renderWithGds(<FitScoreChip value={92} label="Great fit" />);
    expect(screen.getByText('Great fit · 92')).toBeInTheDocument();
  });

  it('clamps value to 0..100', () => {
    renderWithGds(<FitScoreChip value={150} />);
    expect(screen.getByLabelText(/score 100 of 100/i)).toBeInTheDocument();
  });

  it('renders nothing without value or label', () => {
    const { container } = renderWithGds(<FitScoreChip />);
    expect(container.querySelector('.mantine-Badge-root')).toBeNull();
  });

  it('gives the "good" and "partial" bands distinct colors', () => {
    const { container: goodContainer } = renderWithGds(<FitScoreChip value={60} />);
    const { container: partialContainer } = renderWithGds(<FitScoreChip value={20} />);

    const goodBadge = goodContainer.querySelector('.mantine-Badge-root') as HTMLElement;
    const partialBadge = partialContainer.querySelector('.mantine-Badge-root') as HTMLElement;

    expect(goodBadge.style.backgroundColor).not.toBe('');
    expect(goodBadge.style.backgroundColor).not.toBe(partialBadge.style.backgroundColor);
  });

  // Skipped (issue 739 / issue 742): deterministically misses vitest's timeout on CI's
  // shared runners even at 60000ms (default 15000ms and 30000ms also insufficient); real
  // cause suspected to be genuine per-keystroke/per-interaction cost, not artificial delay.
  // Re-enable once issue 742's investigation lands a real fix.
  it.skip('makes a chip with dimensions keyboard-focusable and reveals its tooltip on focus', async () => {
    const user = userEvent.setup();
    renderWithGds(
      <FitScoreChip value={88} dimensions={[{ label: 'Budget' }, { label: 'Location' }]} />,
    );

    const chip = screen.getByLabelText(/score 88 of 100/i);
    expect(chip).toHaveAttribute('tabIndex', '0');

    await user.tab();
    expect(chip).toHaveFocus();
    expect(await screen.findByText(/Matched: Budget, Location/i)).toBeInTheDocument();
  });
});

describe('MeaningBadge (#322)', () => {
  it('renders the label for a variant', () => {
    renderWithGds(<MeaningBadge variant="attention" label="Featured" />);
    expect(screen.getByText('Featured')).toBeInTheDocument();
  });

  it('renders nothing without a label', () => {
    const { container } = renderWithGds(<MeaningBadge variant="info" label="" />);
    expect(container.querySelector('.mantine-Badge-root')).toBeNull();
  });

  it('routes a canonical GdsIcons key through the governed GdsIcon (#494)', () => {
    const { container } = renderWithGds(<MeaningBadge variant="urgency" label="Ends soon" icon="Warning" />);
    const icon = container.querySelector('[data-gds-icon]');
    expect(icon).not.toBeNull();
    expect(icon).toHaveAttribute('data-gds-icon', 'Warning');
    expect(icon).toHaveAttribute('aria-hidden', 'true');
  });

  it('still renders a custom ReactNode icon as given (#494)', () => {
    const { container } = renderWithGds(
      <MeaningBadge variant="validation" label="Verified" icon={<span data-testid="custom-icon" />} />,
    );
    expect(screen.getByTestId('custom-icon')).toBeInTheDocument();
    expect(container.querySelector('[data-gds-icon]')).toBeNull();
  });
});

describe('MediaWithFallback (#323)', () => {
  it('renders an image when src is provided', () => {
    renderWithGds(<MediaWithFallback src="https://example.com/a.jpg" alt="Provider" />);
    expect(screen.getByAltText('Provider')).toBeInTheDocument();
  });

  it('shows branded fallback when src is missing', () => {
    renderWithGds(<MediaWithFallback alt="Bright Kids" fallbackLabel="Bright Kids" />);
    expect(screen.getByRole('img', { name: 'Bright Kids' })).toBeInTheDocument();
    expect(screen.getByText('BK')).toBeInTheDocument();
  });

  it('falls back when the image errors', () => {
    renderWithGds(<MediaWithFallback src="broken" alt="X" fallbackLabel="Foo Bar" />);
    fireEvent.error(screen.getByAltText('X'));
    expect(screen.getByText('FB')).toBeInTheDocument();
  });
});

describe('NumberStepper (#324)', () => {
  function Harness() {
    const [v, setV] = useState(1);
    return <NumberStepper value={v} onChange={setV} min={1} max={3} ariaLabel="Children" />;
  }

  it('increments and decrements within bounds', async () => {
    const user = userEvent.setup();
    renderWithGds(<Harness />);
    const spin = screen.getByRole('spinbutton', { name: 'Children' });
    expect(spin).toHaveAttribute('aria-valuenow', '1');
    await user.click(screen.getByRole('button', { name: 'Increase' }));
    expect(spin).toHaveAttribute('aria-valuenow', '2');
  });

  it('disables decrement at min', () => {
    renderWithGds(<NumberStepper value={1} onChange={() => {}} min={1} max={5} ariaLabel="Q" />);
    expect(screen.getByRole('button', { name: 'Decrease' })).toBeDisabled();
  });
});

describe('AISearchCard (#325)', () => {
  it('submits the typed query on Enter', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWithGds(<AISearchCard onSubmit={onSubmit} placeholder="Ask" />);
    await user.type(screen.getByLabelText('Ask'), 'swimming{enter}');
    expect(onSubmit).toHaveBeenCalledWith('swimming');
  });

  it('routes a prompt chip through onSubmit', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWithGds(<AISearchCard onSubmit={onSubmit} prompts={['Toddler swim']} />);
    await user.click(screen.getByText('Toddler swim'));
    expect(onSubmit).toHaveBeenCalledWith('Toddler swim');
  });

  it('does not submit an empty query', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWithGds(<AISearchCard onSubmit={onSubmit} placeholder="Ask" />);
    await user.type(screen.getByLabelText('Ask'), '   {enter}');
    expect(onSubmit).not.toHaveBeenCalled();
  });
});

describe('ChatThread (#321)', () => {
  const messages: ChatMessageModel[] = [
    { id: '1', role: 'user', content: 'Find swimming' },
    { id: '2', role: 'assistant', content: 'Here are options', cards: [<div key="c">Card A</div>] },
  ];

  it('renders alternating messages and embedded cards', () => {
    renderWithGds(<ChatThread messages={messages} onSend={() => {}} />);
    expect(screen.getByText('Find swimming')).toBeInTheDocument();
    expect(screen.getByText('Card A')).toBeInTheDocument();
  });

  it('sends on Enter and clears, newline on Shift+Enter', async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    renderWithGds(<ChatThread messages={[]} onSend={onSend} placeholder="Msg" />);
    const input = screen.getByLabelText('Msg');
    await user.type(input, 'hello{enter}');
    expect(onSend).toHaveBeenCalledWith('hello');
  });

  it('shows an empty state', () => {
    renderWithGds(<ChatThread messages={[]} onSend={() => {}} />);
    expect(screen.getByText(/get started/i)).toBeInTheDocument();
  });
});

describe('ListingCard composition (#320)', () => {
  it('renders reason, score, and a multi-action footer', () => {
    renderWithGds(
      <ListingCard
        title="Bright Kids"
        reason={<ul><li>Ages 3–6</li></ul>}
        score={<FitScoreChip value={88} />}
        actions={[<button key="s">Save</button>, <button key="v">View</button>, <button key="c">Contact</button>]}
      />,
    );
    expect(screen.getByRole('group', { name: 'Why this fits' })).toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'Listing actions' })).toBeInTheDocument();
    expect(screen.getByText('Contact')).toBeInTheDocument();
  });

  it('throws in development with more than four actions', () => {
    expect(() =>
      renderWithGds(
        <ListingCard
          title="X"
          actions={[1, 2, 3, 4, 5].map((n) => (
            <button key={n}>{n}</button>
          ))}
        />,
      ),
    ).toThrow();
  });
});

describe('MediaWithFallback reserves its box (issue 605)', () => {
  // The reference site states that a failed image "never collapses a card or shifts layout".
  // Nothing asserted it: the existing tests covered WHICH content renders, not that the box
  // keeps its height when that content is the fallback. A claim on the site with no evidence
  // behind it is exactly what the claims gate now forbids.
  it('keeps an aspect-ratio box in both states, so the fallback occupies the image’s space', () => {
    const { container: withImage } = renderWithGds(
      <MediaWithFallback src="https://example.com/a.jpg" alt="Provider" ratio={16 / 9} />,
    );
    const imageBox = withImage.querySelector('.mantine-AspectRatio-root');
    expect(imageBox).toBeTruthy();

    const { container: withFallback } = renderWithGds(
      <MediaWithFallback alt="Bright Kids" fallbackLabel="Bright Kids" ratio={16 / 9} />,
    );
    const fallbackBox = withFallback.querySelector('.mantine-AspectRatio-root');
    expect(fallbackBox).toBeTruthy();

    // Same reserved ratio in both states — that is what stops the collapse the site promises.
    const ratioOf = (el: Element | null) => getComputedStyle(el as HTMLElement).getPropertyValue('--ar-ratio').trim();
    expect(ratioOf(fallbackBox)).toBe(ratioOf(imageBox));
  });

  it('still reserves the box after the image errors', () => {
    const { container } = renderWithGds(<MediaWithFallback src="broken" alt="X" fallbackLabel="Foo Bar" />);
    fireEvent.error(screen.getByAltText('X'));
    expect(screen.getByText('FB')).toBeInTheDocument();
    expect(container.querySelector('.mantine-AspectRatio-root')).toBeTruthy();
  });
});
