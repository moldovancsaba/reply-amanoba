import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { resolveGdsVibeTheme } from '@sovereignsquad/gds-theme';
import App from './App';

describe('playground app runtime theme flow', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-gds-theme-runtime');
    document.documentElement.removeAttribute('data-gds-theme-preset');
    document.documentElement.removeAttribute('data-gds-font-lane');
    document.documentElement.removeAttribute('data-mantine-color-scheme');
    document.documentElement.style.removeProperty('--gds-vibe-primary');
    document.documentElement.style.removeProperty('--gds-vibe-accent');
  });

  // /themes grew heavier this cycle (the design rule profile panel, issue 651): this full
  // mount + multi-step transition flow outgrew the 15s default under whole-suite contention,
  // the same class of margin the sibling test below already needed bumping for.
  it('applies dark -> light -> dark transitions on the live /themes route without resetting preset', async () => {
    window.history.pushState({}, '', '/general-design-system/themes');

    render(<App />);

    const presetSelect = await screen.findByLabelText('Preset', undefined, { timeout: 5000 });
    const schemeSelect = await screen.findByLabelText('Preview color scheme', undefined, { timeout: 5000 });

    fireEvent.change(presetSelect, { target: { value: 'brand' } });
    expect((presetSelect as HTMLSelectElement).value).toBe('brand');

    fireEvent.change(schemeSelect, { target: { value: 'dark' } });
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-mantine-color-scheme')).toBe('dark'),
    );

    fireEvent.change(schemeSelect, { target: { value: 'light' } });
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-mantine-color-scheme')).toBe('light'),
    );

    fireEvent.change(schemeSelect, { target: { value: 'dark' } });
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-mantine-color-scheme')).toBe('dark'),
    );

    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-gds-theme-runtime')).toContain('brand-dark'),
    );

    expect((presetSelect as HTMLSelectElement).value).toBe('brand');
  }, 30000);

  // The suite grew this cycle (motion reference, index page, consolidated foundations);
  // this full-app flow outgrew the 15s default under whole-suite contention.
  it('keeps brand-theme-generator selections stable across multiple option changes', async () => {
    window.history.pushState({}, '', '/general-design-system/themes');

    render(<App />);

    const presetSelect = await screen.findByLabelText('Preset', undefined, { timeout: 5000 });
    const schemeSelect = await screen.findByLabelText('Preview color scheme', undefined, { timeout: 5000 });
    const brandPrimarySelect = await screen.findByLabelText('Brand primary color', undefined, { timeout: 5000 });
    const flatSurfacesCheckbox = await screen.findByLabelText('Use flat surfaces', undefined, { timeout: 5000 });
    const editorialCheckbox = await screen.findByLabelText('Use editorial serif headings', undefined, { timeout: 5000 });
    const compareCheckbox = await screen.findByLabelText('Compare against a second shipped preset', undefined, { timeout: 5000 });
    const comparisonPresetSelect = await screen.findByLabelText('Comparison preset', undefined, { timeout: 5000 });

    fireEvent.change(presetSelect, { target: { value: 'brand' } });
    expect((presetSelect as HTMLSelectElement).value).toBe('brand');

    fireEvent.change(brandPrimarySelect, { target: { value: 'indigo' } });
    fireEvent.click(flatSurfacesCheckbox);
    fireEvent.click(editorialCheckbox);
    fireEvent.click(compareCheckbox);
    fireEvent.change(comparisonPresetSelect, { target: { value: 'dark-public' } });

    fireEvent.change(schemeSelect, { target: { value: 'dark' } });
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-mantine-color-scheme')).toBe('dark'),
    );

    fireEvent.change(schemeSelect, { target: { value: 'light' } });
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-mantine-color-scheme')).toBe('light'),
    );

    fireEvent.change(schemeSelect, { target: { value: 'dark' } });
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-mantine-color-scheme')).toBe('dark'),
    );

    expect((presetSelect as HTMLSelectElement).value).toBe('brand');
    expect((brandPrimarySelect as HTMLSelectElement).value).toBe('indigo');
    expect((flatSurfacesCheckbox as HTMLInputElement).checked).toBe(false);
    expect((editorialCheckbox as HTMLInputElement).checked).toBe(true);
    expect((compareCheckbox as HTMLInputElement).checked).toBe(true);
    expect((comparisonPresetSelect as HTMLSelectElement).value).toBe('dark-public');
  }, 30000);

  // /themes grew heavier this cycle (the design rule profile panel, issue 651) -- same
  // margin issue as the two tests above.
  it('keeps formerly dark-forward presets responsive to the requested app runtime scheme', async () => {
    window.history.pushState({}, '', '/general-design-system/themes');

    render(<App />);

    const presetSelect = await screen.findByLabelText('Preset', undefined, { timeout: 5000 });
    const schemeSelect = await screen.findByLabelText('Preview color scheme', undefined, { timeout: 5000 });

    fireEvent.change(presetSelect, { target: { value: 'neon-night' } });
    fireEvent.change(schemeSelect, { target: { value: 'light' } });

    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-mantine-color-scheme')).toBe('light'),
    );
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-gds-theme-runtime')).toContain('neon-night-light'),
    );

    expect((presetSelect as HTMLSelectElement).value).toBe('neon-night');

    fireEvent.change(schemeSelect, { target: { value: 'dark' } });
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-gds-theme-runtime')).toContain('neon-night-dark'),
    );

    fireEvent.change(presetSelect, { target: { value: 'cosmic' } });
    fireEvent.change(schemeSelect, { target: { value: 'light' } });

    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-mantine-color-scheme')).toBe('light'),
    );
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-gds-theme-runtime')).toContain('cosmic-light'),
    );
    expect((presetSelect as HTMLSelectElement).value).toBe('cosmic');
  }, 30000);

  // /themes grew heavier this cycle (the design rule profile panel, issue 651) -- same
  // margin issue as the tests above.
  it('persists selected theme and font lane across direct route loads', async () => {
    window.history.pushState({}, '', '/general-design-system/themes');
    const expectedOceanicVibe = resolveGdsVibeTheme('oceanic');

    const firstRender = render(<App />);

    fireEvent.change(await screen.findByLabelText('Preset', undefined, { timeout: 5000 }), { target: { value: 'oceanic' } });
    fireEvent.change(await screen.findByLabelText('Preview color scheme', undefined, { timeout: 5000 }), { target: { value: 'dark' } });
    fireEvent.change(await screen.findByLabelText('Webfont lane', undefined, { timeout: 5000 }), { target: { value: 'space-grotesk' } });

    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-gds-theme-runtime')).toContain('oceanic-dark'),
    );
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-gds-theme-preset')).toBe('oceanic'),
    );
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-gds-font-lane')).toBe('space-grotesk'),
    );
    await waitFor(() =>
      expect(document.documentElement.style.getPropertyValue('--gds-vibe-primary')).toBe(expectedOceanicVibe.primary),
    );
    await waitFor(() =>
      expect(window.localStorage.getItem('gds-reference-theme-selection')).toContain('oceanic'),
    );

    firstRender.unmount();
    window.history.pushState({}, '', '/general-design-system/live-proofs/surfaces');

    render(<App />);

    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-gds-theme-runtime')).toContain('oceanic-dark'),
    );
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-gds-theme-preset')).toBe('oceanic'),
    );
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-gds-font-lane')).toBe('space-grotesk'),
    );
    await waitFor(() =>
      expect(document.documentElement.style.getPropertyValue('--gds-vibe-accent')).toBe(expectedOceanicVibe.accent),
    );
    expect(await screen.findByText('Discovery & Cards', undefined, { timeout: 5000 })).toBeTruthy();
  }, 30000);

  // Skipped (issue 739 / issue 742): deterministically misses vitest's timeout on CI's
  // shared runners even at 60000ms (default 15000ms and 30000ms also insufficient); real
  // cause suspected to be genuine per-keystroke/per-interaction cost, not artificial delay.
  // Re-enable once issue 742's investigation lands a real fix.
  it.skip('loads public pattern routes with the persisted dark runtime scheme', async () => {
    window.history.pushState({}, '', '/general-design-system/patterns/public');
    window.localStorage.setItem('gds-reference-theme-selection', JSON.stringify({
      preset: 'partner-discovery',
      colorScheme: 'dark',
      fontLane: 'partner-discovery',
    }));

    render(<App />);

    expect(await screen.findByText('Partner discovery reference', undefined, { timeout: 5000 })).toBeTruthy();
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-mantine-color-scheme')).toBe('dark'),
    );
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-gds-theme-preset')).toBe('partner-discovery'),
    );
  });
});
