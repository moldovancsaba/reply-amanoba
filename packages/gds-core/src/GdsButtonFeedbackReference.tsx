'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Badge, Box, Group, Stack, Text } from '@mantine/core';
import { GdsVocabulary } from './vocabulary';
import {
  GDS_BUTTON_FEEDBACK_DURATION_MS,
  GDS_BUTTON_GRADIENT_TEXT_FLOOR,
  GDS_BUTTON_OUTLINE_ACCENT_STROKE_PX,
  SemanticButton,
} from './SemanticButton';
import { SimpleDataTable } from './SimpleDataTable';

type FeedbackRow = {
  action: string;
  color: string;
  messageId: string;
  Icon: React.ComponentType<{ size?: string | number }>;
} & Record<string, unknown>;

// Every vocabulary action that declares a success-feedback config, read from GdsVocabulary at
// render time. An action added or re-coloured in the vocabulary changes this table with no edit
// here (Rule 14).
const feedbackRows: FeedbackRow[] = Object.entries(GdsVocabulary)
  .filter(([, config]) => Boolean(config.feedback))
  .map(([action, config]) => ({
    action,
    color: config.feedback!.color,
    messageId: config.feedback!.messageId,
    Icon: config.feedback!.icon,
  }));

const totalActions = Object.keys(GdsVocabulary).length;
// One representative action per distinct feedback colour, so the live proof below covers every
// colour the system actually uses rather than a hand-picked three.
const representativeByColor = Array.from(
  feedbackRows.reduce((acc, row) => (acc.has(row.color) ? acc : acc.set(row.color, row)), new Map<string, FeedbackRow>()).values(),
);

function LiveFeedbackButton({ action }: { action: string }) {
  const [state, setState] = useState<'success' | 'error' | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // The prop must return to null or a second click cannot re-trigger the effect that drives the
  // treatment -- the same thing a real consumer does after an async action settles.
  const fire = useCallback((next: 'success' | 'error') => {
    setState(next);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setState(null), GDS_BUTTON_FEEDBACK_DURATION_MS);
  }, []);

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  return (
    <Group gap="xs" wrap="nowrap">
      <SemanticButton action={action as never} feedbackState={state} onClick={() => fire('success')} />
      <SemanticButton action={action as never} variant="light" feedbackState={state} onClick={() => fire('error')} />
    </Group>
  );
}

const brandIntentDemos = [
  { variant: 'outline-accent' as const, label: 'outline-accent' },
  { variant: 'gradient' as const, label: 'gradient' },
];

/**
 * Live proof for the two brand intents added by issue 700 -- rest, hover/pressed (stylesheet-
 * driven; hover/press the rest button to see them), disabled, loading, and the same transient
 * feedback treatment as every other brand intent, reusing `LiveFeedbackButton`'s fire/timer
 * pattern so both intents prove they hand off to the governed success/danger paint during
 * feedback rather than keeping their own stroke/gradient.
 */
function BrandIntentDemo({ variant }: { variant: 'outline-accent' | 'gradient' }) {
  const [state, setState] = useState<'success' | 'error' | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fire = useCallback((next: 'success' | 'error') => {
    setState(next);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setState(null), GDS_BUTTON_FEEDBACK_DURATION_MS);
  }, []);

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  return (
    <Group gap="xs" wrap="wrap" data-gds-brand-intent-demo={variant} align="center">
      <SemanticButton action="save" brandVariant={variant} feedbackState={state} onClick={() => fire('success')} />
      <SemanticButton action="delete" brandVariant={variant} feedbackState={state} onClick={() => fire('error')} />
      <SemanticButton action="preview" brandVariant={variant} loading />
      <SemanticButton action="preview" brandVariant={variant} disabled />
    </Group>
  );
}

/**
 * Reference for the button micro-feedback axis: the transient success/error treatment
 * `SemanticButton` applies after an action settles. The treatment is per action -- each governed
 * action declares its own feedback icon and message key, so a `delete` confirms with "Deleted"
 * and its own glyph rather than a generic tick. Every value below is read from `GdsVocabulary`
 * and `GDS_BUTTON_FEEDBACK_DURATION_MS` at render time, so the page cannot drift from the
 * component.
 *
 * The colour column reports what the vocabulary declares. What paints also depends on the active
 * preset's governed Button rules, so the page shows the declaration and lets the live proof above
 * show the rendering rather than asserting a colour it cannot verify per preset.
 */
export function GdsButtonFeedbackReference() {
  return (
    <Stack gap="md" data-gds-button-feedback-reference="">
      <Stack gap="2xs">
        <Group gap="xs" align="center">
          <Text fw={700}>Live — click either button</Text>
          <Badge variant="light">{GDS_BUTTON_FEEDBACK_DURATION_MS}ms</Badge>
        </Group>
        <Text size="sm">
          The left button fires the success treatment, the right one fires the error treatment.
          Watch the label and the icon change, then revert on their own after{' '}
          {GDS_BUTTON_FEEDBACK_DURATION_MS}ms — the duration is owned by the component
          (`GDS_BUTTON_FEEDBACK_DURATION_MS`), not by the caller. One row per distinct feedback
          colour declared in the vocabulary.
        </Text>
        <Stack gap="xs">
          {representativeByColor.map((row) => (
            <Group key={row.action} gap="sm" align="center" wrap="wrap">
              <Box miw={110}>
                <Text size="xs" c="dimmed" ff="monospace">{row.color}</Text>
              </Box>
              <LiveFeedbackButton action={row.action} />
            </Group>
          ))}
        </Stack>
      </Stack>

      <Stack gap="2xs">
        <Text fw={700}>Success feedback is per action</Text>
        <Text size="sm">
          {feedbackRows.length} of {totalActions} governed actions declare their own success
          feedback — icon, colour, and message key. The icon and message are the action&apos;s
          own, not a generic tick: a `delete` confirms with &ldquo;Deleted&rdquo; and its own
          glyph.
        </Text>
        <Text size="sm" c="dimmed">
          The colour column is what each action declares in the vocabulary. What reaches the
          screen also depends on the active preset&apos;s governed Button rules, which repaint
          some button surfaces by design — click the buttons above in the preset you care about
          rather than reading a colour off this table.
        </Text>
        <SimpleDataTable<FeedbackRow>
          columns={[
            { key: 'action', header: 'Action' },
            {
              key: 'Icon',
              header: 'Icon',
              render: (row) => <row.Icon size="1rem" />,
            },
            { key: 'color', header: 'Colour (declared)' },
            {
              key: 'messageId',
              header: 'Message key',
              // A bare key in a table cell gets squeezed by the other columns; a minimum width
              // keeps the dotted id on one line instead of wrapping per segment.
              render: (row) => <Box miw={200}><Text size="xs" c="dimmed" ff="monospace">{row.messageId}</Text></Box>,
            },
          ]}
          rows={feedbackRows}
          getRowKey={(row) => row.action}
        />
      </Stack>

      <Stack gap="2xs">
        <Text fw={700}>Error feedback is uniform</Text>
        <Text size="sm">
          Where success is per action, the error treatment is fixed: the same cross icon and the
          `gds.feedback.error` message key for every action, so a failure reads the same way
          wherever it appears. Pass `feedbackText` to replace the label in either state — the icon
          stays governed.
        </Text>
      </Stack>

      <Stack gap="2xs" data-gds-brand-intent-reference="">
        <Group gap="xs" align="center">
          <Text fw={700}>Brand intents — outline-accent &amp; gradient</Text>
          <Badge variant="light">{GDS_BUTTON_OUTLINE_ACCENT_STROKE_PX}px stroke</Badge>
          <Badge variant="light">{GDS_BUTTON_GRADIENT_TEXT_FLOOR.fontSizePx}px / {GDS_BUTTON_GRADIENT_TEXT_FLOOR.fontWeight} label floor</Badge>
        </Group>
        <Text size="sm">
          Two brand treatments the inline brand-variant map cannot express — an accent-as-outline
          action (transparent fill, accent stroke and label) and the reserved Scout AI gradient
          action (a solid fill where the active preset declares no <code>ai</code> lane). Hover
          and press either &quot;Save&quot; button to see the stylesheet-driven wash/deepen and
          brightness states; the &quot;Delete&quot; button fires the same transient error
          treatment as above, which fully replaces the intent&apos;s own paint for its duration.
          The third and fourth buttons in each row are the loading and disabled states.
        </Text>
        <Stack gap="xs">
          {brandIntentDemos.map((demo) => (
            <Group key={demo.variant} gap="sm" align="center" wrap="wrap">
              <Box miw={110}>
                <Text size="xs" c="dimmed" ff="monospace">{demo.label}</Text>
              </Box>
              <BrandIntentDemo variant={demo.variant} />
            </Group>
          ))}
        </Stack>
      </Stack>
    </Stack>
  );
}
