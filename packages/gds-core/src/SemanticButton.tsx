'use client';

import React, { useEffect, useState } from 'react';
import { Button, VisuallyHidden } from '@mantine/core';
import type { ButtonProps } from '@mantine/core';
import { useGdsTranslation } from '@sovereignsquad/gds-theme';
import { IconCheck, IconX } from '@tabler/icons-react';
import { getSemanticActionLabel, resolveSemanticActionConfig } from './vocabulary';
import type { GdsVocabularyPack, SemanticActionId } from './vocabulary';

/**
 * How long a transient success/error feedback treatment stays on the button before it reverts
 * to its resting label. Exported so documentation surfaces read the real value rather than
 * restating it (Rule 14) -- a change here changes both the behaviour and the page.
 */
export const GDS_BUTTON_FEEDBACK_DURATION_MS = 2000;

/**
 * Minimum label typography for the `gradient` brand intent (issue 700). White label text on
 * the reserved Scout AI gradient fill (issue 697) clears only the WCAG non-text threshold
 * (3:1) -- the `ai-accent-text-contrast` accessibility-floor rule measures and reports this
 * every run -- never the 4.5:1 normal-text floor, so the gradient intent's label is held at or
 * above this size and weight as a hard component invariant rather than relying on a preset to
 * compensate.
 */
export const GDS_BUTTON_GRADIENT_TEXT_FLOOR = { fontSizePx: 14, fontWeight: 600 } as const;

/** Outline stroke width for the `outline-accent` brand intent, in px (issue 700). */
export const GDS_BUTTON_OUTLINE_ACCENT_STROKE_PX = 1.5;

/** Props for `SemanticButton`; extends Mantine `ButtonProps` and the native button attributes. */
export interface SemanticButtonProps extends ButtonProps, Omit<React.ComponentPropsWithoutRef<'button'>, keyof ButtonProps | 'leftSection' | 'children'> {
  /** Governed semantic action id whose label and icon are resolved from the vocabulary. */
  action: SemanticActionId;
  /**
   * Applies a brand color treatment; the `disabled` variant also disables the button.
   * `outline-accent` (transparent fill, accent stroke and label) and `gradient` (the reserved
   * Scout AI gradient fill, issue 697, over a solid primary-chain fallback where a preset
   * declares no `ai` lane) are issue 700: their full hover/pressed/disabled state axis lives in
   * `packages/gds-theme/styles.css`, keyed on the `data-gds-brand-button` attribute this
   * component emits, because a stylesheet `:hover`/`:active` rule cannot override an inline
   * style. Never combine either with a destructive action -- style those through the existing
   * `danger` vocabulary intent instead.
   */
  brandVariant?: 'primary' | 'secondary' | 'accent' | 'disabled' | 'outline-accent' | 'gradient';
  loading?: boolean;
  /** Triggers a transient success/error feedback treatment (icon, color, label) for ~2s. */
  feedbackState?: 'success' | 'error' | null;
  /** Overrides the label shown during a feedback state. */
  feedbackText?: string;
  /** Renders the untranslated label on first paint to avoid a hydration mismatch, then upgrades; defaults to true. */
  prerenderLabelOnly?: boolean;
  /** Additional vocabulary packs consulted when resolving the action's label and icon. */
  vocabularyPacks?: GdsVocabularyPack[];
  /**
   * Overrides the rendered label with an arbitrary node instead of the vocabulary-resolved
   * string -- the vocabulary system is string-only by design (it exists to keep a label
   * translated and consistent everywhere the same `action` appears), so a caller that genuinely
   * needs non-string content (e.g. a caller-supplied `ReactNode` label prop of its own) sets this
   * rather than reimplementing button chrome locally. The icon, `feedbackState`, and
   * `prerenderLabelOnly` behavior are unaffected -- only the text content changes.
   */
  label?: React.ReactNode;
}

const brandButtonStyles: Record<NonNullable<SemanticButtonProps['brandVariant']>, React.CSSProperties> = {
  primary: {
    background: 'var(--gds-brand-primary, var(--gds-vibe-primary, var(--mantine-primary-color-filled)))',
    borderColor: 'var(--gds-brand-primary, var(--gds-vibe-primary, var(--mantine-primary-color-filled)))',
    // --gds-text-on-inverse is derived against --gds-bg-inverse only; against
    // --gds-brand-primary it is 1.00:1 in the default dark preset (identical color).
    color: 'var(--gds-brand-primary-fg, var(--gds-text-on-inverse, var(--mantine-color-white)))',
  },
  secondary: {
    background: 'var(--gds-bg-card, var(--gds-vibe-surface, var(--mantine-color-white)))',
    borderColor: 'var(--gds-border-card, var(--gds-vibe-border, var(--mantine-color-gray-3)))',
    color: 'var(--gds-brand-primary, var(--gds-vibe-text, var(--mantine-color-dark-7)))',
  },
  accent: {
    background: 'var(--gds-brand-accent-action, var(--gds-brand-accent, var(--gds-vibe-accent, var(--mantine-primary-color-filled))))',
    borderColor: 'var(--gds-brand-accent-action, var(--gds-brand-accent, var(--gds-vibe-accent, var(--mantine-primary-color-filled))))',
    // 1.66:1 against the gold-athlete dark accent.
    color: 'var(--gds-brand-accent-action-fg, var(--gds-text-on-inverse, var(--mantine-color-white)))',
  },
  disabled: {
    background: 'var(--gds-control-disabledBg, var(--mantine-color-gray-2))',
    borderColor: 'var(--gds-control-disabledBg, var(--mantine-color-gray-2))',
    color: 'var(--gds-control-disabledText, var(--mantine-color-gray-6))',
  },
  // `outline-accent` and `gradient` (issue 700) carry no inline paint. A stylesheet
  // `:hover`/`:active` rule cannot override an inline style without `!important`, so their
  // entire resting paint -- and every state layered on top of it -- lives in
  // packages/gds-theme/styles.css, keyed on `data-gds-brand-button`; the component contributes
  // only the attribute.
  'outline-accent': {},
  gradient: {},
};

/**
 * Button whose label is resolved from a governed semantic `action` id (via the GDS
 * vocabulary) rather than a hardcoded string, so the same action reads and
 * localizes consistently everywhere it appears. Supports brand variants, a
 * `loading` busy state, and transient success/error feedback. Use it for any
 * action tied to a known semantic verb; fall back to Mantine's raw `Button` only
 * for one-off actions with no vocabulary entry.
 */
export function SemanticButton({
  action,
  brandVariant,
  loading,
  feedbackState,
  feedbackText,
  prerenderLabelOnly = true,
  vocabularyPacks = [],
  label: labelOverride,
  ...props
}: SemanticButtonProps) {
  const { t } = useGdsTranslation();
  const config = resolveSemanticActionConfig(action, vocabularyPacks);

  const [mounted, setMounted] = useState(!prerenderLabelOnly);
  const [internalFeedback, setInternalFeedback] = useState<'success' | 'error' | null>(null);

  useEffect(() => {
    if (prerenderLabelOnly) {
      setMounted(true);
    }
  }, [prerenderLabelOnly]);

  useEffect(() => {
    if (feedbackState) {
      setInternalFeedback(feedbackState);
      const timer = setTimeout(() => setInternalFeedback(null), GDS_BUTTON_FEEDBACK_DURATION_MS);
      return () => clearTimeout(timer);
    }
  }, [feedbackState]);

  let Icon = config.icon;
  let label: React.ReactNode = labelOverride ?? getSemanticActionLabel(action, t, vocabularyPacks);
  let color = props.color;
  const brandStyle = brandVariant ? brandButtonStyles[brandVariant] : undefined;
  const disabled = props.disabled || brandVariant === 'disabled';

  // `outline-accent`/`gradient` paint entirely from the stylesheet (see brandButtonStyles
  // above), so while a transient feedback treatment is showing, the governed success/danger
  // button rules in styles.css must be the ones that paint instead -- withholding the attribute
  // for exactly that window is what hands the cascade to those existing rules, so feedback
  // fully overrides intent paint with no new stylesheet rule and no residual stroke/gradient.
  // The four pre-existing variants are untouched: their resting paint is inline and already
  // wins the cascade regardless of feedback, exactly as before this change (issue 700).
  const isStylesheetPaintedBrand = brandVariant === 'outline-accent' || brandVariant === 'gradient';
  const brandButtonAttr = isStylesheetPaintedBrand && internalFeedback ? undefined : brandVariant;

  if (!mounted) {
    const { leftSection, style, ...buttonProps } = props;
    return (
      <Button {...buttonProps} loading={loading} color={color} data-gds-brand-button={brandButtonAttr} disabled={disabled} aria-busy={loading ? true : undefined} style={{ ...brandStyle, ...style }}>
        {labelOverride ?? getSemanticActionLabel(action, undefined, vocabularyPacks)}
      </Button>
    );
  }

  if (internalFeedback === 'success') {
    const defaultFeedback = config.feedback ?? { icon: IconCheck, color: 'teal', messageId: 'gds.feedback.saved' };
    Icon = defaultFeedback.icon;
    label = feedbackText || t(defaultFeedback.messageId, 'Success');
    color = defaultFeedback.color;
  } else if (internalFeedback === 'error') {
    Icon = IconX;
    label = feedbackText || t('gds.feedback.error', 'Something went wrong');
    color = 'red';
  }

  return (
    <>
      <Button
        {...props}
        leftSection={<Icon size="1rem" />}
        loading={loading}
        color={color}
        data-gds-brand-button={brandButtonAttr}
        disabled={disabled}
        aria-busy={loading ? true : undefined}
        style={{ ...brandStyle, ...props.style }}
      >
        {label}
      </Button>
      {/* Announces the transient feedback label swap; the current source set no aria-live
          before this change, so the state change was silent to screen readers. */}
      {internalFeedback ? (
        <VisuallyHidden role="status" aria-live="polite" aria-atomic="true">
          {label}
        </VisuallyHidden>
      ) : null}
    </>
  );
}
