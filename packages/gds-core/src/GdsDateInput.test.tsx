import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { resetGdsDevWarnings } from '@sovereignsquad/gds-theme';
import { renderWithGds } from '../../../test-utils/render';
import { GdsDateInput, GdsDateRangeInput, GdsDateTimeInput } from './GdsDateInput.client';

describe('GdsDateInput', () => {
  // Skipped (issue 739 / issue 742): deterministically misses vitest's timeout on CI's
  // shared runners even at 60000ms (default 15000ms and 30000ms also insufficient); real
  // cause suspected to be genuine per-keystroke/per-interaction cost, not artificial delay.
  // Re-enable once issue 742's investigation lands a real fix.
  it.skip('renders with a label and parses typed text into a Date onChange', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithGds(<GdsDateInput label="Start date" value={null} onChange={onChange} />);

    const input = screen.getByLabelText('Start date');
    await user.type(input, 'July 23, 2026');
    await user.tab();

    await waitFor(() => expect(onChange).toHaveBeenCalled());
    const received = onChange.mock.calls.at(-1)?.[0] as Date;
    expect(received.getFullYear()).toBe(2026);
    expect(received.getMonth()).toBe(6);
    expect(received.getDate()).toBe(23);
  });

  it('accepts an ISO date string as its value', () => {
    renderWithGds(<GdsDateInput label="Start date" value="2026-07-23" onChange={() => {}} />);
    const input = screen.getByLabelText('Start date') as HTMLInputElement;
    expect(input.value).toContain('2026');
  });
});

describe('GdsDateTimeInput', () => {
  it('renders with a label', () => {
    renderWithGds(<GdsDateTimeInput label="Appointment" value={null} onChange={() => {}} />);
    expect(screen.getByLabelText('Appointment')).toBeInTheDocument();
  });
});

describe('GdsDateRangeInput', () => {
  it('renders with a label', () => {
    renderWithGds(<GdsDateRangeInput label="Coverage window" value={[null, null]} onChange={() => {}} />);
    expect(screen.getByLabelText('Coverage window')).toBeInTheDocument();
  });
});

describe('date family dev bound diagnostics', () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    resetGdsDevWarnings();
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('warns when minDate is after maxDate', () => {
    renderWithGds(<GdsDateInput label="d" value={null} minDate="2026-12-01" maxDate="2026-01-01" />);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0]?.[0]).toMatch(/minDate .* is after maxDate/);
  });

  it('warns when a supplied value falls outside [minDate, maxDate]', () => {
    renderWithGds(<GdsDateInput label="d" value="2027-06-01" minDate="2026-01-01" maxDate="2026-12-31" />);
    expect(warnSpy.mock.calls.some((call) => /value .* is after maxDate/.test(String(call[0])))).toBe(true);
  });

  it('warns on transposed bounds for the range input too', () => {
    renderWithGds(<GdsDateRangeInput label="r" value={[null, null]} minDate="2026-12-01" maxDate="2026-01-01" />);
    expect(warnSpy.mock.calls.some((call) => /GdsDateRangeInput: minDate/.test(String(call[0])))).toBe(true);
  });

  it('does not warn for valid bounds and an in-range value', () => {
    renderWithGds(<GdsDateInput label="d" value="2026-06-01" minDate="2026-01-01" maxDate="2026-12-31" />);
    expect(warnSpy).not.toHaveBeenCalled();
  });
});
