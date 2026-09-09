'use client';

import { useState } from 'react';

/** Options for {@link useGdsBrowseSelection}. */
export interface GdsBrowseSelectionOptions {
  /** Uncontrolled initial selection. Ignored once `selectedId` is supplied (controlled mode). */
  defaultSelectedId?: string | null;
  /**
   * Controlled selection. Once supplied (including `null`), the hook mirrors this value instead
   * of managing its own state, and every transition reports through `onChange` rather than
   * mutating internally.
   */
  selectedId?: string | null;
  /** Fires whenever the selection changes, in both controlled and uncontrolled mode. */
  onChange?: (selectedId: string | null) => void;
}

/** Return value of {@link useGdsBrowseSelection}: the current selection plus its transitions. */
export interface GdsBrowseSelection {
  /** The currently selected id, or `null` when nothing is selected. */
  selectedId: string | null;
  /** True when `id` is the current selection. */
  isSelected: (id: string) => boolean;
  /** Selects `id` outright, replacing any prior selection. */
  select: (id: string) => void;
  /** Selects `id`, or clears the selection when `id` is already selected. */
  toggle: (id: string) => void;
  /** Clears the selection. */
  clear: () => void;
}

/**
 * Single-selection state shareable between a `ListingCard` list and a set of `GdsMapPinBadge`
 * pins in a browse surface: one id drives both the selected card's ring treatment (`selected`
 * prop) and the matching pin's `state="selected"` — the "selected card = selected pin" behavior
 * a browse split view needs, without every consumer reinventing the sync (issue 701).
 *
 * Works uncontrolled (internal state, the default) or controlled (mirrors `selectedId`, calling
 * `onChange` instead of self-managing — controlled mode never mutates internal state). `select`
 * accepts any id, including one not present in the rendered list; `toggle` selects, or clears
 * when the id is already current; `clear` always clears. A pure client-side state mechanism — no
 * timers, no async, no persistence — SSR-safe by the `.client.ts` convention (excluded from the
 * package's server barrel).
 *
 * @example
 * ```tsx
 * const browse = useGdsBrowseSelection();
 * <ListingCard title="Riverside Field" selected={browse.isSelected('riverside')} onSurfaceActivate={() => browse.toggle('riverside')} />
 * <GdsMapPinBadge accent="forest" icon="Location" label="Riverside Field" state={browse.isSelected('riverside') ? 'selected' : 'idle'} />
 * ```
 */
export function useGdsBrowseSelection(options: GdsBrowseSelectionOptions = {}): GdsBrowseSelection {
  const { defaultSelectedId = null, selectedId, onChange } = options;
  const [internalSelectedId, setInternalSelectedId] = useState<string | null>(defaultSelectedId);
  const isControlled = selectedId !== undefined;
  const current = isControlled ? selectedId : internalSelectedId;

  const commit = (next: string | null) => {
    if (!isControlled) {
      setInternalSelectedId(next);
    }
    if (next !== current) {
      onChange?.(next);
    }
  };

  return {
    selectedId: current,
    isSelected: (id: string) => id === current,
    select: (id: string) => commit(id),
    toggle: (id: string) => commit(id === current ? null : id),
    clear: () => commit(null),
  };
}
