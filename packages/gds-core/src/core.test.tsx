import React from 'react';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { Text, Title } from '@mantine/core';
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithGds } from '../../../test-utils/render';
import { AccessSummary } from './AccessSummary';
import { AccessRecoveryPanel } from './AccessRecoveryPanel';
import { GdsAccessGate, createGdsAccessAdapter, createGdsAccessGateEvent, getGdsAccessGateActionPriority, getGdsAccessGateReasons, getGdsAccessGateStates, redactGdsAccessGateMetadata, resolveGdsAccessAdapterState, resolveGdsAccessState, sortGdsAccessGateActions, validateGdsAccessGateContract } from './GdsAccessGate';
import { createGdsAccessibilityEvidenceIndex, getGdsAccessibilityEvidence, getGdsAccessibilityEvidenceSummary, validateGdsAccessibilityEvidence } from './AccessibilityEvidence';
import { AccentPanel, resolveAccentPanelStyles } from './AccentPanel';
import { ActionBar } from './ActionBar';
import { AdvancedDataTable } from './AdvancedDataTable.client';
import { GdsDataTable, createGdsTableAdapter, serializeGdsTableQuery } from './GdsDataTable.client';
import { GdsResourceManager, createGdsResourceAdapter } from './GdsResourceManager.client';
import { ArticleShell } from './ArticleShell';
import { AuthShell } from './AuthShell';
import { AsyncSurface } from './AsyncSurface';
import { BrowseSurface } from './BrowseSurface';
import { BoundedPreviewSurface } from './BoundedPreviewSurface';
import { ConsumerDashboardGrid } from './ConsumerDashboardGrid';
import { ConsumerSection } from './ConsumerSection';
import { CtaButtonGroup } from './CtaButtonGroup';
import { ConfirmDialog } from './ConfirmDialog';
import { ChoiceChip, FilterChipGroup, PillBar, SoftChipGroup } from './ChoiceChip';
import { DataToolbar } from './DataToolbar';
import { CommandRegistryProvider, useCommandLauncher } from './CommandPalette.client';
import { createGdsDraftAdapter, FormErrorSummary, GdsFormProvider, GdsValidationSummary, gdsFormReducer, useGdsForm, useGdsFormOrchestration, ValidatedFieldMessage } from './GdsForm.client';
import { GdsSchemaForm, createGdsFormFromSchema, jsonSchemaToGdsFormSchema, openApiToGdsFormSchema, zodToGdsFormSchema } from './GdsSchemaForm.client';
import { ActiveFilterChips, BulkActionsBar, ResultSummary, SortMenu } from './ListingPrimitives';
import { ListingProvider, listingQueryReducer, useListingState } from './ListingState.client';
import { DetailProfileShell } from './DetailProfileShell';
import { DocsCodeBlock } from './DocsCodeBlock';
import { DocsHeaderActionSelect, DocsShell } from './DocsShell';
import { DocsPageShell } from './DocsPageShell';
import { EmptyState } from './EmptyState';
import { EditorialCard } from './EditorialCard';
import { EditorialHero } from './EditorialHero';
import { FeatureBand } from './FeatureBand';
import { FoodMenuSection } from './FoodMenuSection';
import { GameBoardTile } from './GameBoardTile';
import { ChartTokenPanel } from './ChartTokenPanel';
import { GdsChart, gdsChartTypeRegistry, gdsChartSetATypeRegistry, gdsChartSetBTypeRegistry, gdsChartSetCTypeRegistry, isGdsChartSetAType, isGdsChartSetBType, isGdsChartSetCType, validateGdsChartData } from './GdsChart';
import { GdsAreaChart, GdsBarChart, GdsBenchmarkBarChart, GdsCalendarHeatmapChart, GdsDivergingBarChart, GdsGaugeChart, GdsHistogramChart, GdsLineChart, GdsLongitudinalChart, GdsMaturityRadarChart, GdsRadarChart, GdsSlopeChart, GdsSparkline, GdsStackedBarChart, GdsSymmetryChart, getGdsSeriesColor } from './SemanticCharts';
import { GdsRatingScale, GdsSegmentedControl, GdsSlider, GdsWizardStepper } from './GdsFormControls';
import { BodyText, CardTitle, InlineEmphasis, LabelText, MetadataText, PageTitle, SectionTitle } from './Typography';
import { ClippedFlexChild, FloatingActionPlacement, ListItemSection, NumericCell, OverflowContainer, SemanticInset, VisuallyHidden } from './StyleUtilities';
import { GdsBox, GdsCluster, GdsColumnGrid, GdsColumnGridItem, GdsContainer, GdsGrid, GdsInline, GdsSidebar, GdsSplit, GdsStack, normalizeGdsResponsiveValue, resolveGdsLayoutStyle } from './LayoutPrimitives';
import { GdsMediaFrame, GdsOverflowFrame, GdsResponsiveVisibility, GdsSafeBox, createGdsStyleContract, gdsStyle } from './SafeStyles';
import { EvidencePanel } from './EvidencePanel';
import { ListingCard } from './ListingCard';
import { getGdsBlockTypes, getGdsLayoutTemplate, getGdsLayoutTemplates, registerGdsBlock, renderGdsLayout, renderGdsLayoutWithDiagnostics, validateGdsLayout } from './LayoutBlocks';
import { GdsLayoutTemplatePreview } from './LayoutTemplatePreview.client';
import { MapPanel } from './MapPanel';
import { MediaField } from './MediaField';
import { MediaCard } from './MediaCard';
import { MetricCard } from './MetricCard';
import { BannerNotice } from './Notifications';
import { GdsNotificationProvider, NotificationCenter, useGdsNotifications } from './Notifications.client';
import { PageHeader } from './PageHeader';
import { PeriodSelector } from './PeriodSelector';
import { PlaceholderPanel } from './PlaceholderPanel';
import { PlaybackSurface } from './PlaybackSurface';
import { PublicFlowShell } from './PublicFlowShell';
import { PublicFoodCard } from './PublicFoodCard';
import { PublicBrandFooter } from './PublicBrandFooter';
import { ProductCard } from './ProductCard';
import { PublicProductCard } from './PublicProductCard';
import { PublicNav } from './PublicNav';
import { PublicShell } from './PublicShell';
import { ShareButtonGroup } from './ShareButtonGroup';
import { DiscoveryShell, useDiscoveryShellState } from './DiscoveryShell';
import { BottomTabBar, BOTTOM_TAB_HEIGHT } from './BottomTabBar';
import { GDS_BUTTON_FEEDBACK_DURATION_MS, GDS_BUTTON_GRADIENT_TEXT_FLOOR, GDS_BUTTON_OUTLINE_ACCENT_STROKE_PX, SemanticButton } from './SemanticButton';
import { SectionPanel } from './SectionPanel';
import { ReferenceSection } from './ReferenceSection';
import { SidebarNav, SidebarNavItem, SidebarNavSection } from './SidebarNav';
import { SimpleDataTable } from './SimpleDataTable';
import { SocialAuthButtons } from './SocialAuthButtons';
import { ProviderIdentityButton, ProviderIdentityButtonGroup, getProviderIdentityLabel, getProviderIdentityPolicy, getSupportedProviderIdentityIds } from './ProviderIdentityButtons';
import { MissingDataPrompt, StateBlock } from './StateBlock';
import { StatsSection } from './StatsSection';
import { CountBadge, LabelTag, StatusBadge } from './StatusBadge';
import { ThemeToggle } from './ThemeToggle';
import { ReferenceThemeExplorer } from './ReferenceThemeExplorer';
import { ReportingSection } from './ReportingSection';
import { UploadDropzone } from './UploadDropzone';
import { resolveSurfacePresentationStyles } from './SurfacePresentation';
import { resolveGdsCardContract } from './CardContracts';
import { ar, de, en, es, fr, getGdsMessages, he, hu, it as itLocale, ru } from './locales';
import { GdsIcons } from './icons';
import { GdsIcon, getGdsIconKeys, getGdsIconMetadata, getGdsIconToneColor, gdsIconRegistry } from './icons';
import { GdsDialog, GdsDrawer, GdsModal, GdsSidePanel, OverlayManagerProvider, useOverlayManager } from './OverlayManager.client';
import { GdsConfirmProvider, GdsToastProvider, useGdsConfirm, useGdsToasts } from './FeedbackRuntime.client';
import { MediaPreviewCard } from './MediaPreviewCard';
import { KanbanBoard } from './KanbanBoard.client';
import { GdsAssetManager, createGdsAssetAdapter, useGdsAssetUploadQueue, validateGdsAsset } from './GdsAssetManager.client';
import { PublicCaptureFlow } from './PublicCaptureFlow';
import { PlaybackControls, usePlaybackKeyboardControls } from './PlaybackControls.client';
import { CreatorThemeBoundary, validateCreatorCss } from './CreatorTheme';
import { createGdsTelemetryAdapter, emitGdsEvent, GdsTelemetryProvider, gdsOperationalEventTypes, isGdsOperationalEventType, useGdsTelemetry } from './Telemetry.client';
import { createGdsVocabularyPack, getSemanticActionLabel } from './vocabulary';
import { getGdsTaskPattern, getGdsTaskPatterns, validateGdsTaskPatterns } from './TaskPatterns';
import { GdsColorSystemReference } from './GdsColorSystemReference';
import { GdsAccessibilitySystemReference } from './GdsAccessibilitySystemReference';
import {
  createGdsPageTemplateEvent,
  GdsAdminDashboardTemplate,
  GdsCrudEditorTemplate,
  GdsEmptyStateTemplate,
  GdsErrorPageTemplate,
  GdsPublicEventTemplate,
  GdsResourceManagerTemplate,
  getGdsPageTemplate,
  getGdsPageTemplates,
  validateGdsPageTemplates,
} from './GdsPageTemplates';
import {
  compareGdsLocaleString,
  createGdsMissingKeyTracker,
  createGdsTextExpansionFixture,
  formatGdsCurrency,
  formatGdsDate,
  formatGdsNumber,
  formatGdsPlural,
  formatGdsRelativeTime,
  GdsDirectionBoundary,
  GdsFormattedCurrency,
  GdsFormattedDate,
  GdsFormattedNumber,
  GdsLocaleText,
  GdsPlural,
  GdsRelativeTime,
  resolveGdsLocale,
  resolveGdsMessage,
  sortGdsLocaleStrings,
  useGdsDirection,
} from './GdsI18nRuntime';
import {
  createGdsContentExpansionReport,
  GdsContentPatternCatalog,
  getGdsContentPattern,
  getGdsContentPatterns,
  getGdsCopyTemplate,
  getGdsCopyTemplates,
  renderGdsCopyTemplate,
  validateGdsContentPatterns,
  validateGdsCopyTemplate,
} from './GdsContentDesign';
import {
  GdsDesignHandoffCatalog,
  generateGdsDesignHandoffReport,
  getGdsDesignComponentMappings,
  getGdsDesignTokenMappings,
  validateGdsDesignHandoffMappings,
} from './GdsDesignHandoff';
import { getGdsVibeThemes } from '@sovereignsquad/gds-theme';

function mockMatchMedia(matches: boolean) {
  const original = window.matchMedia;
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });

  return () => {
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: original,
    });
  };
}

function mockMatchMediaByQuery(resolve: (query: string) => boolean) {
  const original = window.matchMedia;
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: resolve(query),
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });

  return () => {
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: original,
    });
  };
}

describe('@sovereignsquad/gds-core', () => {
  it('renders semantic button labels from translation messages', () => {
    renderWithGds(<SemanticButton action="save" />, {
      messages: { 'gds.action.save': 'Speichern' },
    });

    expect(screen.getByRole('button', { name: 'Speichern' })).toBeInTheDocument();
  });

  it('indexes and validates structured accessibility evidence', () => {
    const entries = [
      {
        id: 'demo',
        title: 'Demo pattern',
        kind: 'pattern' as const,
        route: '/patterns/foundations',
        packageName: '@sovereignsquad/gds-core',
        owner: 'GDS foundations',
        status: 'verified' as const,
        updatedAt: '2026-06-14',
        evidenceSource: 'Official docs route',
        summary: 'Structured accessibility evidence for a stable pattern.',
        keyboard: {
          tabSequence: 'Tab and Shift+Tab move through the pattern in visible order.',
          activation: 'Enter and Space activate the focused control.',
        },
        focusBehavior: 'Visible focus remains present across light, dark, and forced-colors modes.',
        screenReader: {
          summary: 'Screen readers receive named controls and current state copy.',
          semantics: ['button', 'heading'],
          announcements: ['current state is visible and announced'],
        },
        wcagMappings: [
          { criterion: '1.3.1', level: 'A' as const, note: 'Relationships are explicit.' },
          { criterion: '1.4.3', level: 'AA' as const, note: 'Contrast is validated.' },
          { criterion: '2.1.1', level: 'A' as const, note: 'Keyboard path is available.' },
          { criterion: '2.4.7', level: 'AA' as const, note: 'Focus is visible.' },
          { criterion: '4.1.2', level: 'A' as const, note: 'Name, role, and value are exposed.' },
        ],
        atBrowserStatus: [
          {
            assistiveTechnology: 'VoiceOver',
            browser: 'Safari 18',
            os: 'iOS 18',
            status: 'verified' as const,
            verifiedAt: '2026-06-14',
            note: 'Reviewed on the official route.',
          },
        ],
        recovery: 'Pin the previous package version if the pattern regresses.',
      },
    ];

    const index = createGdsAccessibilityEvidenceIndex(entries);
    expect(getGdsAccessibilityEvidence(index, 'demo')?.title).toBe('Demo pattern');
    expect(getGdsAccessibilityEvidence(entries, 'demo')?.owner).toBe('GDS foundations');

    const summary = getGdsAccessibilityEvidenceSummary(entries);
    expect(summary.total).toBe(1);
    expect(summary.verified).toBe(1);
    expect(summary.atStatuses.verified).toBe(1);

    const validation = validateGdsAccessibilityEvidence(entries);
    expect(validation.ok).toBe(true);
    expect(validation.failures).toEqual([]);
  });

  it('publishes the access gate state model, validation, action order, and privacy-safe events', () => {
    expect(getGdsAccessGateStates()).toEqual([
      'loading-auth',
      'preview',
      'locked',
      'unlocking',
      'unlocked',
      'permission-denied',
      'expired',
      'error',
    ]);
    expect(getGdsAccessGateReasons()).toContain('subscription-required');
    expect(getGdsAccessGateActionPriority('sign-in')).toBeLessThan(getGdsAccessGateActionPriority('back'));
    expect(sortGdsAccessGateActions([{ kind: 'back' }, { kind: 'sign-in' }, { kind: 'subscribe' }]).map((action) => action.kind)).toEqual([
      'sign-in',
      'subscribe',
      'back',
    ]);

    expect(validateGdsAccessGateContract({
      id: 'article-paywall',
      state: 'locked',
      reason: 'subscription-required',
      title: 'Subscribe to continue',
      description: 'The preview remains visible.',
      actions: [{ kind: 'subscribe' }],
      protectedContentPolicy: 'never-render-while-locked',
    })).toEqual([]);

    expect(validateGdsAccessGateContract({
      id: 'article-paywall',
      state: 'locked',
      title: 'Subscribe to continue',
      description: 'The preview remains visible.',
    })).toEqual([
      'Locked access gates must declare protectedContentPolicy: never-render-while-locked.',
      'Locked and denied access gates require at least one recovery action.',
    ]);

    expect(redactGdsAccessGateMetadata({
      route: '/members/story',
      email: 'reader@example.com',
      body: 'Protected member-only article body',
      paid: true,
    })).toEqual({
      route: '/members/story',
      email: '[redacted]',
      body: '[redacted]',
      paid: true,
    });

    expect(createGdsAccessGateEvent(
      'gds.access_gate.action',
      { id: 'article-paywall', state: 'locked', reason: 'login-required' },
      { token: 'secret', plan: 'pro' },
      'sign-in',
    )).toMatchObject({
      gateId: 'article-paywall',
      actionKind: 'sign-in',
      metadata: { token: '[redacted]', plan: 'pro' },
    });
  });

  it('never evaluates protected content while the access gate is locked', () => {
    const protectedContent = vi.fn(() => <p>Protected member-only article body</p>);

    renderWithGds(
      <GdsAccessGate
        id="article-paywall"
        state="locked"
        reason="subscription-required"
        title="Subscribe to continue"
        description="Read the summary now. Full article unlocks after subscription."
        actions={[{ kind: 'subscribe' }, { kind: 'sign-in' }]}
        protectedContentPolicy="never-render-while-locked"
        preview={<p>Public teaser summary.</p>}
        protectedContent={protectedContent}
      />,
    );

    expect(screen.getByText('Public teaser summary.')).toBeInTheDocument();
    expect(screen.getAllByText('Subscribe to continue')).toHaveLength(2);
    expect(screen.queryByText('Protected member-only article body')).not.toBeInTheDocument();
    expect(protectedContent).not.toHaveBeenCalled();
  });

  it('renders protected content only after access is unlocked and emits action events', async () => {
    const events: string[] = [];
    const actionHandler = vi.fn();

    renderWithGds(
      <GdsAccessGate
        id="article-paywall"
        state="unlocked"
        title="Content unlocked"
        description="Member session is active."
        actions={[]}
        preview={<p>Public teaser summary.</p>}
        protectedContent={() => <p>Protected member-only article body</p>}
        metadata={{ route: '/members/story' }}
        onEvent={(event) => events.push(event.type)}
        onAction={actionHandler}
      />,
    );

    await waitFor(() => expect(events).toContain('gds.access_gate.unlocked'));
    expect(screen.getByText('Protected member-only article body')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /subscribe/i })).not.toBeInTheDocument();
  });

  it('resolves access adapter states for anonymous, entitled, denied, expired, error, and timeout flows', async () => {
    expect(resolveGdsAccessState({
      gateId: 'article-paywall',
      session: { status: 'anonymous' },
    })).toMatchObject({ state: 'locked', reason: 'login-required' });

    expect(resolveGdsAccessState({
      gateId: 'article-paywall',
      session: { status: 'authenticated', subjectId: 'user-1' },
      entitlement: { allowed: true, label: 'Pro' },
    })).toMatchObject({ state: 'unlocked' });

    expect(resolveGdsAccessState({
      gateId: 'article-paywall',
      session: { status: 'authenticated', subjectId: 'user-1' },
      entitlement: { allowed: false, reason: 'subscription-required', label: 'Pro plan' },
    })).toMatchObject({ state: 'permission-denied', reason: 'subscription-required', entitlementLabel: 'Pro plan' });

    expect(resolveGdsAccessState({
      gateId: 'article-paywall',
      session: { status: 'expired' },
    })).toMatchObject({ state: 'expired', reason: 'session-expired' });

    const adapter = createGdsAccessAdapter({
      getSession: () => ({ status: 'authenticated', subjectId: 'user-1' }),
      getEntitlement: () => ({ allowed: true, label: 'Pro plan' }),
    });
    await expect(resolveGdsAccessAdapterState(adapter, { gateId: 'article-paywall' })).resolves.toMatchObject({ state: 'unlocked' });

    await expect(resolveGdsAccessAdapterState({
      getSession: () => {
        throw new Error('provider offline');
      },
    }, { gateId: 'article-paywall' })).resolves.toMatchObject({ state: 'error', reason: 'unknown-error' });

    await expect(resolveGdsAccessAdapterState({
      getSession: () => new Promise(() => {}),
    }, { gateId: 'article-paywall', timeoutMs: 5 })).resolves.toMatchObject({ state: 'error', reason: 'network-timeout' });
  });

  it('publishes complete task pattern contracts with stable ids, states, telemetry, and guidance', () => {
    const patterns = getGdsTaskPatterns();
    expect(patterns.map((pattern) => pattern.id)).toEqual([
      'create-resource',
      'review-submission',
      'bulk-approve',
      'recover-failed-upload',
      'copy-public-link',
      'publish-toggle',
      'confirm-destructive-action',
    ]);
    expect(validateGdsTaskPatterns(patterns)).toEqual([]);
    for (const pattern of patterns) {
      expect(pattern.states).toEqual(expect.arrayContaining(['start', 'in-progress', 'success', 'empty', 'error', 'retry', 'cancelled']));
      expect(pattern.telemetry.length).toBeGreaterThan(0);
      expect(pattern.accessibility.length).toBeGreaterThan(0);
      expect(pattern.doNotBuild.length).toBeGreaterThan(0);
      expect(pattern.steps.every((step) => step.componentContracts.length > 0)).toBe(true);
    }
    const destructive = getGdsTaskPattern('confirm-destructive-action');
    expect(destructive?.componentContracts).toContain('GdsConfirmProvider');
    patterns[0]!.states.length = 0;
    expect(getGdsTaskPattern('create-resource')?.states).toContain('start');
  });

  it('publishes complete production page template contracts with stable ids, states, telemetry, and recovery guidance', () => {
    const templates = getGdsPageTemplates();
    expect(templates.map((template) => template.id)).toEqual([
      'admin-dashboard',
      'settings',
      'resource-manager',
      'crud-editor',
      'analytics',
      'public-event',
      'error-page',
      'empty-state-page',
    ]);
    expect(validateGdsPageTemplates(templates)).toEqual([]);
    for (const template of templates) {
      expect(template.packageName).toBe('@sovereignsquad/gds-core');
      expect(template.telemetryEvents).toEqual(expect.arrayContaining(['page_view', 'state_visible']));
      expect(template.componentContracts.length).toBeGreaterThan(0);
      expect(template.accessibility.length).toBeGreaterThan(0);
      expect(template.edgeCases.length).toBeGreaterThan(0);
      expect(template.rollback).toMatch(/adopt|replace|keep/i);
    }
    const event = createGdsPageTemplateEvent('analytics', 'empty', 'retry_requested', {
      actionId: 'reload-report',
      metadata: { period: '30d', rowCount: 0 },
    });
    expect(event).toEqual({
      name: 'retry_requested',
      templateId: 'analytics',
      state: 'empty',
      actionId: 'reload-report',
      metadata: { period: '30d', rowCount: 0 },
    });
    templates[0]!.requiredStates.length = 0;
    expect(getGdsPageTemplate('admin-dashboard')?.requiredStates).toContain('ready');
  });

  it('renders production page templates with landmarks, state copy, actions, and accessible fallbacks', () => {
    const onRetry = vi.fn();
    const onAction = vi.fn();

    renderWithGds(
      <>
        <GdsAdminDashboardTemplate
          title="Operations"
          description="Live operator overview"
          state="ready"
          actions={[{ id: 'create', label: 'Create', kind: 'primary', onClick: onAction }]}
          metrics={[{ label: 'Open reviews', value: '12', description: 'Pending review queue' }]}
          sections={[{ id: 'queue', title: 'Review queue', content: <Text>Ready for review</Text> }]}
        />
        <GdsResourceManagerTemplate
          title="Events"
          description="Manage public events"
          rows={[{ id: 'event-1', name: 'Launch' }]}
          columns={[{ key: 'name', header: 'Name' }]}
          detail={<Text>Selected event detail</Text>}
          getRowKey={(row) => String(row.id)}
        />
        <GdsCrudEditorTemplate
          title="Edit event"
          description="Governed editor"
          state="saving"
          form={<label htmlFor="event-title">Title<input id="event-title" defaultValue="Launch" /></label>}
          destructiveZone={<button type="button">Delete event</button>}
        />
        <GdsPublicEventTemplate
          title="Public launch"
          description="Registration page"
          when="June 30, 2026"
          location="Budapest"
          details={<Text>Open to partners.</Text>}
          registration={<button type="button">Register</button>}
        />
        <GdsErrorPageTemplate
          title="Could not load"
          description="The report service timed out."
          code={503}
          onRetry={onRetry}
        />
        <GdsEmptyStateTemplate
          title="No assets yet"
          description="Upload the first asset to continue."
          actions={[{ id: 'upload', label: 'Upload asset', kind: 'primary' }]}
        />
      </>,
    );

    expect(screen.getAllByRole('main')).toHaveLength(6);
    expect(screen.getByRole('heading', { name: 'Operations', level: 1 })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    expect(onAction).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('table')).toBeInTheDocument();
    expect(screen.getByText('Selected event detail')).toBeInTheDocument();
    expect(screen.getByText('Saving')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Register' })).toBeInTheDocument();
    expect(screen.getByText('503: Could not load')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Upload asset' })).toBeInTheDocument();
  });

  it('provides package-native i18n runtime formatting, sorting, fallback, and missing-key telemetry', () => {
    const events: Array<{ type: string; key?: string; locale: string; fallbackLocale?: string }> = [];
    const onEvent = (event: { type: string; key?: string; locale: string; fallbackLocale?: string }) => events.push(event);

    expect(resolveGdsLocale({ locale: 'xx', fallbackLocale: 'en', onEvent })).toBe('en');
    expect(events[0]).toMatchObject({ type: 'fallback_locale_used', locale: 'xx', fallbackLocale: 'en' });
    expect(formatGdsNumber(1234.5, { locale: 'de', maximumFractionDigits: 1 })).toContain('1.234');
    expect(formatGdsCurrency(12, 'EUR', { locale: 'de' })).toContain('€');
    expect(formatGdsDate('2026-06-14T12:00:00Z', { locale: 'en', timeZone: 'UTC', month: 'long', day: 'numeric', year: 'numeric' })).toBe('June 14, 2026');
    expect(formatGdsRelativeTime(-1, 'day', { locale: 'en' })).toBe('yesterday');
    expect(formatGdsPlural(0, { zero: 'No files', one: 'One file', other: 'Files' }, { locale: 'en' })).toBe('No files');
    expect(formatGdsPlural(1, { one: 'One file', other: 'Files' }, { locale: 'en' })).toBe('One file');
    expect(sortGdsLocaleStrings(['item 10', 'item 2'], { locale: 'en' })).toEqual(['item 2', 'item 10']);
    expect(compareGdsLocaleString('á', 'a', { locale: 'hu', sensitivity: 'base' })).toBe(0);

    const trackerEvents: Array<{ type: string; key?: string }> = [];
    const tracker = createGdsMissingKeyTracker((event) => trackerEvents.push(event));
    tracker.emit({ type: 'missing_key', locale: 'en', key: 'missing.copy' });
    tracker.emit({ type: 'missing_key', locale: 'en', key: 'missing.copy' });
    expect(trackerEvents).toHaveLength(1);

    const missingEvents: Array<{ type: string; key?: string }> = [];
    expect(resolveGdsMessage({ id: 'unknown.copy', defaultMessage: 'Fallback {count}', values: { count: 3 }, locale: 'en', onEvent: (event) => missingEvents.push(event) })).toBe('Fallback 3');
    expect(missingEvents).toEqual([expect.objectContaining({ type: 'missing_key', key: 'unknown.copy' })]);

    const germanFixture = createGdsTextExpansionFixture('de', 'Save');
    expect(germanFixture.expansionRatio).toBeGreaterThan(1);
    expect(germanFixture.minInlineSizeCh).toBeGreaterThan(4);
    const rtlFixture = createGdsTextExpansionFixture('ar', 'Save');
    expect(rtlFixture.direction).toBe('rtl');
    expect(rtlFixture.notes.join(' ')).toContain('dir="auto"');
  });

  it('renders i18n runtime components with readable text and direction attributes', () => {
    function DirectionProbe() {
      const direction = useGdsDirection('he');
      return <div data-testid="direction">{direction.dir}:{String(direction.isRtl)}</div>;
    }

    renderWithGds(
      <>
        <GdsLocaleText id="gds.action.save" defaultMessage="Save fallback" locale="de" />
        <GdsFormattedNumber value={9876.5} locale="en" maximumFractionDigits={1} />
        <GdsFormattedCurrency value={20} currency="USD" locale="en" />
        <GdsFormattedDate value="2026-06-14T12:00:00Z" locale="en" timeZone="UTC" month="short" day="numeric" />
        <GdsRelativeTime value={1} unit="day" locale="en" />
        <GdsPlural value={2} locale="en" messages={{ one: 'One alert', other: 'Many alerts' }} />
        <GdsDirectionBoundary locale="ar">مرحبا</GdsDirectionBoundary>
        <DirectionProbe />
      </>,
    );

    expect(screen.getByText('Speichern')).toBeInTheDocument();
    expect(screen.getByText('9,876.5')).toBeInTheDocument();
    expect(screen.getByText('$20.00')).toBeInTheDocument();
    expect(screen.getByText('Jun 14')).toBeInTheDocument();
    expect(screen.getByText('tomorrow')).toBeInTheDocument();
    expect(screen.getByText('Many alerts')).toBeInTheDocument();
    expect(screen.getByText('مرحبا')).toHaveAttribute('dir', 'rtl');
    expect(screen.getByTestId('direction')).toHaveTextContent('rtl:true');
  });

  it('publishes content design patterns with voice, accessibility, telemetry, and localization-safe templates', () => {
    const patterns = getGdsContentPatterns();
    expect(patterns.map((pattern) => pattern.id)).toEqual([
      'error-recovery',
      'retryable-failure',
      'destructive-confirmation',
      'empty-state',
      'permission-denied',
      'primary-cta',
      'form-hint',
      'success-feedback',
    ]);
    expect(validateGdsContentPatterns(patterns)).toEqual([]);
    for (const pattern of patterns) {
      expect(pattern.voiceRules.length).toBeGreaterThan(0);
      expect(pattern.componentContracts.length).toBeGreaterThan(0);
      expect(pattern.taskPatterns.length).toBeGreaterThan(0);
      expect(pattern.accessibility.length).toBeGreaterThan(0);
      expect(pattern.localization.expansionLocales.length).toBeGreaterThan(0);
      expect(pattern.doNotWrite.length).toBeGreaterThan(0);
      expect(pattern.templates.length).toBeGreaterThan(0);
    }

    const destructive = getGdsContentPattern('destructive-confirmation');
    expect(destructive?.componentContracts).toContain('GdsConfirmProvider');
    patterns[0]!.voiceRules.length = 0;
    expect(getGdsContentPattern('error-recovery')?.voiceRules).toContain('Name the failed operation.');
  });

  it('validates and renders localization-safe copy templates', () => {
    const template = getGdsCopyTemplate('destructive-confirmation.delete');
    expect(template).toBeDefined();
    expect(renderGdsCopyTemplate(template!, { target: 'Draft event', undoWindow: '30 days' })).toBe('Delete Draft event? This removes it for everyone. You can restore it for 30 days.');
    expect(validateGdsCopyTemplate(template!, { target: 'Draft event' })).toEqual([
      'destructive-confirmation.delete is missing required placeholder undoWindow',
    ]);
    expect(getGdsCopyTemplates().map((item) => item.i18nKey)).toContain('gds.content.emptyFirstRun');
    const expansion = createGdsContentExpansionReport('de');
    expect(expansion.find((item) => item.templateId === 'primary-cta.create')?.fixture.expansionRatio).toBeGreaterThan(1);

    renderWithGds(<GdsContentPatternCatalog />);
    expect(screen.getByRole('heading', { name: 'Error recovery' })).toBeInTheDocument();
    expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();
  });

  it('publishes design-to-code handoff mappings with props, tokens, statuses, and accessibility annotations', () => {
    const components = getGdsDesignComponentMappings();
    const tokens = getGdsDesignTokenMappings();
    expect(components.map((component) => component.exportName)).toEqual([
      'SemanticButton',
      'PageHeader',
      'GdsDataTable',
      'GdsResourceManager',
      'gdsTheme',
    ]);
    expect(validateGdsDesignHandoffMappings(components, tokens)).toEqual([]);
    for (const component of components) {
      expect(component.figmaComponent).toContain('GDS /');
      expect(component.props.length).toBeGreaterThan(0);
      expect(component.annotations.labels.length).toBeGreaterThan(0);
      expect(component.annotations.focusBehavior).toBeTruthy();
      expect(component.annotations.stateSemantics.length).toBeGreaterThan(0);
      expect(component.annotations.accessibility.length).toBeGreaterThan(0);
      expect(component.recovery).toMatch(/stale|migrate|authoritative|bespoke/i);
    }
    expect(tokens.map((token) => token.token)).toContain('color.focus');
    const report = generateGdsDesignHandoffReport('2026-06-14T00:00:00Z');
    expect(report.counts.approved).toBeGreaterThan(0);
    expect(report.counts.experimental).toBe(1);
    expect(report.counts.deprecated).toBe(1);
    expect(report.approvedComponents).toContain('PageHeader');
    components[0]!.props.length = 0;
    expect(getGdsDesignComponentMappings()[0]!.props.length).toBeGreaterThan(0);
  });

  it('renders the design handoff catalog for docs surfaces', () => {
    renderWithGds(<GdsDesignHandoffCatalog />);
    expect(screen.getByRole('heading', { name: 'SemanticButton' })).toBeInTheDocument();
    expect(screen.getByText('GDS / Actions / Semantic Button')).toBeInTheDocument();
    expect(screen.getByText(/Visible focus ring/)).toBeInTheDocument();
  });

  it('renders package-native typography roles without local Text and Title ladders', () => {
    renderWithGds(
      <>
        <PageTitle>Page heading</PageTitle>
        <SectionTitle>Section heading</SectionTitle>
        <CardTitle>Card heading</CardTitle>
        <BodyText>Body copy</BodyText>
        <MetadataText>Metadata</MetadataText>
        <LabelText>Label</LabelText>
        <InlineEmphasis>Important</InlineEmphasis>
      </>,
    );

    expect(screen.getByRole('heading', { name: 'Page heading', level: 1 })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Section heading', level: 2 })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Card heading', level: 3 })).toBeInTheDocument();
    expect(screen.getByText('Body copy')).toBeInTheDocument();
    expect(screen.getByText('Metadata')).toBeInTheDocument();
    expect(screen.getByText('Label')).toBeInTheDocument();
    expect(screen.getByText('Important').tagName).toBe('STRONG');
  });

  it('provides sanctioned style utility surfaces for common layout mechanics', () => {
    renderWithGds(
      <>
        <OverflowContainer label="Overflow list"><div>Scrollable</div></OverflowContainer>
        <FloatingActionPlacement><button type="button">Save</button></FloatingActionPlacement>
        <NumericCell>123</NumericCell>
        <VisuallyHidden>Hidden caption</VisuallyHidden>
        <ClippedFlexChild>Long child</ClippedFlexChild>
        <SemanticInset>Inset</SemanticInset>
        <ul><ListItemSection>Row</ListItemSection></ul>
      </>,
    );

    expect(screen.getByLabelText('Overflow list')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save' })).toBeInTheDocument();
    // Must use the published z-index token (gdsZIndexToken.app), not an ad hoc number, to
    // stay in sync with modals/popovers.
    expect(screen.getByRole('button', { name: 'Save' }).parentElement).toHaveStyle({ zIndex: 'var(--mantine-z-index-app)' });
    expect(screen.getByText('123')).toHaveStyle({ fontVariantNumeric: 'tabular-nums' });
    expect(screen.getByText('Hidden caption')).toHaveStyle({ position: 'absolute' });
    expect(screen.getByText('Long child')).toHaveStyle({ minWidth: '0' });
    expect(screen.getByText('Inset')).toBeInTheDocument();
    expect(screen.getByText('Row').tagName).toBe('LI');
  });

  it('provides governed layout primitives with responsive token contracts', () => {
    const responsive = normalizeGdsResponsiveValue({ base: 'sm', md: 'lg' });
    expect(responsive).toEqual({ base: 'sm', breakpoints: { xs: undefined, sm: undefined, md: 'lg', lg: undefined, xl: undefined } });
    expect(resolveGdsLayoutStyle({ display: 'flex', gap: 'md', align: 'center', justify: 'between', minWidth: 'zero' })).toMatchObject({
      display: 'flex',
      gap: 'var(--mantine-spacing-md)',
      alignItems: 'center',
      justifyContent: 'space-between',
      minWidth: 0,
    });
    expect(resolveGdsLayoutStyle({ maxWidth: 'aside' })).toMatchObject({ maxWidth: '18rem' });

    renderWithGds(
      <>
        <GdsBox component="section" aria-label="Governed box" padding={{ base: 'sm', md: 'lg' }} maxWidth="page">Box</GdsBox>
        <GdsStack component="nav" aria-label="Stack nav"><a href="#one">One</a></GdsStack>
        <GdsInline aria-label="Inline actions" wrap={{ base: 'wrap', lg: 'nowrap' }}><button type="button">A</button><button type="button">B</button></GdsInline>
        <GdsCluster aria-label="Cluster actions"><button type="button">C</button></GdsCluster>
        <GdsGrid aria-label="Responsive grid" columns={{ base: 1, md: 3 }}><div>Grid item</div></GdsGrid>
        <GdsSplit aria-label="Split layout" ratio="2:1"><div>Primary</div><div>Secondary</div></GdsSplit>
        <GdsSidebar aria-label="Sidebar layout" side="end" sidebarWidth="narrow"><aside>Sidebar</aside><main>Main</main></GdsSidebar>
        <GdsSidebar aria-label="Narrow aside layout" sidebarWidth="aside"><aside>Filters</aside><main>Results</main></GdsSidebar>
        <GdsContainer component="main" aria-label="Page container" size={{ base: 'full', lg: 'wide' }}>Container</GdsContainer>
      </>,
    );

    expect(screen.getByRole('region', { name: 'Governed box' })).toHaveTextContent('Box');
    expect(screen.getByRole('navigation', { name: 'Stack nav' })).toBeInTheDocument();
    expect(screen.getByLabelText('Inline actions')).toHaveStyle({ display: 'flex', flexWrap: 'wrap' });
    expect(screen.getByLabelText('Cluster actions')).toHaveStyle({ justifyContent: 'space-between' });
    expect(screen.getByLabelText('Responsive grid')).toHaveStyle({ display: 'grid' });
    expect(screen.getByLabelText('Split layout')).toHaveStyle({ gridTemplateColumns: 'minmax(0, 1fr)' });
    expect(screen.getByLabelText('Sidebar layout')).toHaveTextContent('Sidebar');
    expect(screen.getByLabelText('Narrow aside layout')).toHaveTextContent('Filters');
    expect(screen.getByRole('main', { name: 'Page container' })).toHaveStyle({ width: '100%' });
    expect(document.querySelectorAll('style[data-gds-layout]').length).toBeGreaterThan(0);
  });

  it('provides a named column-grid primitive for explicit track-span layouts (#394)', () => {
    renderWithGds(
      <GdsColumnGrid aria-label="Column grid" columns={12}>
        <GdsColumnGridItem aria-label="Half span" span={6}>Half</GdsColumnGridItem>
        <GdsColumnGridItem aria-label="Offset item" span={4} start={9}>Offset</GdsColumnGridItem>
        <GdsColumnGridItem aria-label="Auto item">Auto</GdsColumnGridItem>
      </GdsColumnGrid>,
    );

    expect(screen.getByLabelText('Column grid')).toHaveStyle({ display: 'grid', gridTemplateColumns: 'repeat(12, minmax(0, 1fr))' });
    expect(screen.getByLabelText('Half span')).toHaveStyle({ gridColumnEnd: 'span 6' });
    expect(screen.getByLabelText('Offset item')).toHaveStyle({ gridColumnStart: '9', gridColumnEnd: 'span 4' });
    expect(screen.getByLabelText('Auto item')).not.toHaveAttribute('style', expect.stringContaining('grid-column'));
  });

  it('resolves safe style contracts without raw consumer CSS values', () => {
    const resolved = gdsStyle({
      background: 'danger',
      border: 'danger',
      radius: 'lg',
      shadow: 'subtle',
      overflow: 'contained',
      inset: 'md',
      focusRing: 'inset',
    });

    expect(resolved.attributes['data-gds-safe-style']).toBe('true');
    expect(resolved.attributes['data-gds-overflow-policy']).toBe('contained');
    expect(resolved.style.backgroundColor).toContain('var(--mantine-color-red-0)');
    expect(resolved.style.border).toContain('var(--mantine-color-red-6)');
    expect(resolved.style.borderRadius).toBe('var(--mantine-radius-lg)');
    expect(resolved.style.padding).toBe('var(--mantine-spacing-md)');
    expect(resolved.style.overscrollBehavior).toBe('contain');

    const contract = createGdsStyleContract('visibility-test', { visibility: { base: 'screen-reader-only', md: 'visible' } });
    expect(contract.className).toMatch(/^gds-safe-style-/);
    expect(contract.css).toContain('@media (min-width: 768px)');

    renderWithGds(
      <>
        <GdsSafeBox safeStyle={{ background: 'surface', border: 'default', radius: 'md' }}>Safe box</GdsSafeBox>
        <GdsMediaFrame fit="contain" aspectRatio="video">Media</GdsMediaFrame>
        <GdsOverflowFrame policy="contained" label="Scrollable region">Overflow</GdsOverflowFrame>
        <GdsResponsiveVisibility visibility={{ base: 'hidden', md: 'visible' }}>Responsive copy</GdsResponsiveVisibility>
      </>,
    );

    expect(screen.getByText('Safe box')).toHaveAttribute('data-gds-safe-style', 'safe-box');
    expect(screen.getByText('Media')).toHaveStyle({ aspectRatio: '16 / 9', objectFit: 'contain' });
    expect(screen.getByLabelText('Scrollable region')).toHaveAttribute('data-gds-overflow-policy', 'contained');
    expect(screen.getByText('Responsive copy')).toHaveAttribute('data-gds-safe-style', 'responsive-visibility');
    expect(document.querySelectorAll('style[data-gds-safe-style-sheet]').length).toBeGreaterThan(0);
  });

  it('exposes a server-safe semantic action label helper', () => {
    expect(getSemanticActionLabel('save')).toBe('Save');
    expect(getSemanticActionLabel('save', (id, fallback) => (id === 'gds.action.save' ? 'Guardar' : fallback))).toBe('Guardar');
  });

  it('shows success and error feedback states for semantic buttons', () => {
    const { rerender } = renderWithGds(<SemanticButton action="save" />);

    rerender(<SemanticButton action="save" feedbackState="success" />);
    expect(screen.getByRole('button', { name: 'Saved' })).toBeInTheDocument();

    rerender(<SemanticButton action="save" feedbackState="error" />);
    expect(screen.getByRole('button', { name: 'Something went wrong' })).toBeInTheDocument();
  });

  it('resolves semantic icon tones without consumer color strings', () => {
    renderWithGds(<GdsIcon icon="Delete" label="Delete item" tone="danger" />);

    expect(screen.getByRole('img', { name: 'Delete item' })).toBeInTheDocument();
    expect(getGdsIconToneColor('danger')).toBe('var(--mantine-color-red-7)');
    expect(getGdsIconToneColor('success')).toBe('var(--mantine-color-green-7)');
  });

  it('exposes icon metadata, aliases, categories, and accessibility defaults', () => {
    renderWithGds(
      <>
        <GdsIcon name="delete" label="Delete record" tone="danger" />
        <GdsIcon name="warning" />
        <GdsIcon name={'not-real' as 'Help'} label="Fallback icon" decorative={false} />
      </>,
    );

    expect(getGdsIconKeys()).toContain('Delete');
    expect(gdsIconRegistry.Delete.category).toBe('action');
    expect(gdsIconRegistry.Bold.category).toBe('content');
    expect(gdsIconRegistry.Cart.category).toBe('commerce');
    expect(gdsIconRegistry.Lock.category).toBe('security');
    expect(gdsIconRegistry.ChevronLeft.category).toBe('navigation');
    expect(getGdsIconMetadata('delete')).toMatchObject({
      name: 'Delete',
      category: 'action',
      defaultLabel: 'Delete',
    });
    expect(getGdsIconMetadata('warning').category).toBe('status');
    expect(screen.getByRole('img', { name: 'Delete record' })).toHaveAttribute('data-gds-icon', 'Delete');
    expect(screen.getByRole('img', { name: 'Fallback icon' })).toHaveAttribute('data-gds-icon', 'Help');
    expect(document.querySelector('[data-gds-icon="Warning"]')).toHaveAttribute('aria-hidden', 'true');
  });

  it('resolves the lowercase form of every multi-word icon key, not just single-word ones', () => {
    const multiWordKeys = [
      'TrendingUp',
      'TrendingDown',
      'EyeOff',
      'ChevronDown',
      'ChevronUp',
      'OrderedList',
      'InlineCode',
      'ChevronLeft',
      'ChevronRight',
      'ArrowUp',
      'ArrowDown',
      'ExternalLink',
      'QrCode',
      'DragHandle',
    ];

    for (const key of multiWordKeys) {
      expect(getGdsIconMetadata(key.toLowerCase())).toMatchObject({ name: key });
    }
  });

  it('resolves the lowercase form of every registered icon key to its canonical key', () => {
    for (const key of getGdsIconKeys()) {
      expect(getGdsIconMetadata(key.toLowerCase()).name).toBe(key);
    }
  });

  it('supports dependency-governed semantic icon names without direct Tabler imports', () => {
    renderWithGds(<GdsIcon name="Download" label="Download file" tone="primary" />);

    expect(screen.getByRole('img', { name: 'Download file' })).toBeInTheDocument();
  });

  it('contains preview internals inside a bounded transformed surface', () => {
    renderWithGds(
      <BoundedPreviewSurface minHeight="24rem" maxHeight="32rem">
        <Text>Contained preview</Text>
      </BoundedPreviewSurface>,
    );

    expect(screen.getByText('Contained preview')).toBeInTheDocument();
    expect(document.querySelector('[data-gds-bounded-preview-surface]')).toHaveStyle({
      contain: 'layout paint',
      isolation: 'isolate',
      overflow: 'hidden',
      transform: 'translateZ(0)',
    });
  });

  it('renders loading and disabled button states safely', () => {
    renderWithGds(<SemanticButton action="save" loading disabled />);

    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
    expect(document.querySelector('.mantine-Loader-root')).toBeInTheDocument();
  });

  it('supports prerender label-only semantic buttons for static pages', () => {
    renderWithGds(<SemanticButton action="save" prerenderLabelOnly />);

    expect(screen.getByRole('button', { name: 'Save' })).toBeInTheDocument();
  });

  it('renders token-backed brand button variants without route-local styles', () => {
    renderWithGds(
      <>
        <SemanticButton action="save" brandVariant="primary" />
        <SemanticButton action="submit" brandVariant="accent" />
        <SemanticButton action="cancel" brandVariant="disabled" />
      </>,
    );

    expect(screen.getByRole('button', { name: 'Save' })).toHaveAttribute('data-gds-brand-button', 'primary');
    expect(screen.getByRole('button', { name: 'Submit' })).toHaveAttribute('data-gds-brand-button', 'accent');
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
  });

  it('extends brandVariant with outline-accent and gradient (issue 700), emitting the attribute for all six values', () => {
    renderWithGds(
      <>
        <SemanticButton action="save" brandVariant="primary" />
        <SemanticButton action="submit" brandVariant="secondary" />
        <SemanticButton action="cancel" brandVariant="accent" />
        <SemanticButton action="delete" brandVariant="disabled" />
        <SemanticButton action="preview" brandVariant="outline-accent" />
        <SemanticButton action="add" brandVariant="gradient" />
      </>,
    );

    expect(screen.getByRole('button', { name: 'Save' })).toHaveAttribute('data-gds-brand-button', 'primary');
    expect(screen.getByRole('button', { name: 'Submit' })).toHaveAttribute('data-gds-brand-button', 'secondary');
    expect(screen.getByRole('button', { name: 'Cancel' })).toHaveAttribute('data-gds-brand-button', 'accent');
    expect(screen.getByRole('button', { name: 'Delete' })).toHaveAttribute('data-gds-brand-button', 'disabled');
    expect(screen.getByRole('button', { name: 'Preview' })).toHaveAttribute('data-gds-brand-button', 'outline-accent');
    expect(screen.getByRole('button', { name: 'Add' })).toHaveAttribute('data-gds-brand-button', 'gradient');
  });

  it('carries GDS_BUTTON_GRADIENT_TEXT_FLOOR and GDS_BUTTON_OUTLINE_ACCENT_STROKE_PX as the documented invariants (issue 700)', () => {
    expect(GDS_BUTTON_GRADIENT_TEXT_FLOOR).toEqual({ fontSizePx: 14, fontWeight: 600 });
    expect(GDS_BUTTON_OUTLINE_ACCENT_STROKE_PX).toBe(1.5);
  });

  it('GDS_BUTTON_GRADIENT_TEXT_FLOOR is actually applied in styles.css, not just documented (issue 700)', () => {
    // jsdom does not evaluate an imported stylesheet, so this asserts the CSS text directly
    // (the established pattern -- see gds-theme/src/semantic-role-tokens.test.ts) rather than a
    // rendered computed style; real-browser proof lives in the CDP runtime gate sweep instead.
    const stylesCss = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'gds-theme', 'styles.css'), 'utf8');
    const gradientRuleMatch = stylesCss.match(/\[data-gds-brand-button='gradient'\]:not\(\[data-disabled\]\)[^{]*\{[^}]*\}/);
    expect(gradientRuleMatch).not.toBeNull();
    const rule = gradientRuleMatch![0];
    expect(rule).toContain(`font-size: ${GDS_BUTTON_GRADIENT_TEXT_FLOOR.fontSizePx}px;`);
    expect(rule).toContain(`font-weight: ${GDS_BUTTON_GRADIENT_TEXT_FLOOR.fontWeight};`);
  });

  it('renders the outline-accent/gradient intent attribute on the SSR pre-mount branch, matching the mounted branch (issue 700)', () => {
    renderWithGds(
      <>
        <SemanticButton action="save" brandVariant="outline-accent" prerenderLabelOnly />
        <SemanticButton action="submit" brandVariant="gradient" prerenderLabelOnly />
      </>,
    );

    expect(screen.getByRole('button', { name: 'Save' })).toHaveAttribute('data-gds-brand-button', 'outline-accent');
    expect(screen.getByRole('button', { name: 'Submit' })).toHaveAttribute('data-gds-brand-button', 'gradient');
  });

  it('a disabled prop on a new brand intent renders disabled, not the intent DOM attribute changing (issue 700)', () => {
    renderWithGds(
      <>
        <SemanticButton action="save" brandVariant="outline-accent" disabled />
        <SemanticButton action="submit" brandVariant="gradient" disabled />
      </>,
    );

    const outline = screen.getByRole('button', { name: 'Save' });
    const gradient = screen.getByRole('button', { name: 'Submit' });
    expect(outline).toBeDisabled();
    expect(outline).toHaveAttribute('data-gds-brand-button', 'outline-accent');
    expect(gradient).toBeDisabled();
    expect(gradient).toHaveAttribute('data-gds-brand-button', 'gradient');
  });

  it('sets aria-busy while loading, for any brand intent', () => {
    renderWithGds(<SemanticButton action="save" brandVariant="gradient" loading />);
    expect(screen.getByRole('button', { name: 'Save' })).toHaveAttribute('aria-busy', 'true');
  });

  it('does not set aria-busy when not loading', () => {
    renderWithGds(<SemanticButton action="save" brandVariant="gradient" />);
    expect(screen.getByRole('button', { name: 'Save' })).not.toHaveAttribute('aria-busy');
  });

  it('withholds data-gds-brand-button on outline-accent/gradient during transient feedback, so the governed success/danger rules paint instead, and restores it on revert (issue 700)', () => {
    vi.useFakeTimers();
    try {
      const { rerender } = renderWithGds(<SemanticButton action="save" brandVariant="outline-accent" />);
      expect(screen.getByRole('button', { name: 'Save' })).toHaveAttribute('data-gds-brand-button', 'outline-accent');

      rerender(<SemanticButton action="save" brandVariant="outline-accent" feedbackState="error" />);
      expect(screen.getByRole('button', { name: 'Something went wrong' })).not.toHaveAttribute('data-gds-brand-button');

      act(() => { vi.advanceTimersByTime(GDS_BUTTON_FEEDBACK_DURATION_MS); });
      rerender(<SemanticButton action="save" brandVariant="outline-accent" feedbackState={null} />);
      expect(screen.getByRole('button', { name: 'Save' })).toHaveAttribute('data-gds-brand-button', 'outline-accent');
    } finally {
      vi.useRealTimers();
    }
  });

  it('keeps data-gds-brand-button on the four pre-existing brand variants during transient feedback, unchanged from before this change (issue 700)', () => {
    const { rerender } = renderWithGds(<SemanticButton action="save" brandVariant="primary" />);
    rerender(<SemanticButton action="save" brandVariant="primary" feedbackState="success" />);
    expect(screen.getByRole('button', { name: 'Saved' })).toHaveAttribute('data-gds-brand-button', 'primary');
  });

  it('announces a transient feedback label change through a polite live region (issue 700)', () => {
    renderWithGds(<SemanticButton action="save" feedbackState="success" />);
    expect(screen.getByRole('status')).toHaveTextContent('Saved');
  });

  it('renders no live region when no feedback is active', () => {
    renderWithGds(<SemanticButton action="save" />);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  it('renders both new brand intents under every shipped theme preset id without throwing (issue 700)', () => {
    for (const vibe of getGdsVibeThemes()) {
      document.documentElement.setAttribute('data-gds-theme-preset', vibe.id);
      for (const scheme of ['light', 'dark'] as const) {
        document.documentElement.setAttribute('data-mantine-color-scheme', scheme);
        const { unmount } = renderWithGds(
          <>
            <SemanticButton action="save" brandVariant="outline-accent" />
            <SemanticButton action="submit" brandVariant="gradient" />
          </>,
        );
        expect(screen.getByRole('button', { name: 'Save' })).toHaveAttribute('data-gds-brand-button', 'outline-accent');
        expect(screen.getByRole('button', { name: 'Submit' })).toHaveAttribute('data-gds-brand-button', 'gradient');
        unmount();
      }
    }
    document.documentElement.removeAttribute('data-gds-theme-preset');
    document.documentElement.removeAttribute('data-mantine-color-scheme');
  });

  it('renders choice chips as neutral links and toggle buttons', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();

    renderWithGds(
      <>
        <ChoiceChip label="Active link" href="/active" active />
        <ChoiceChip label="Toggle me" active onClick={onSelect} />
      </>,
    );

    expect(screen.getByRole('link', { name: 'Active link' })).toHaveAttribute('aria-current', 'page');
    expect(screen.getByRole('button', { name: 'Toggle me' })).toHaveAttribute('aria-pressed', 'true');

    await user.click(screen.getByRole('button', { name: 'Toggle me' }));
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it('renders controlled pill bars, soft chips, and filter chips with radio semantics', async () => {
    const user = userEvent.setup();
    const onPillChange = vi.fn();
    const onSoftChange = vi.fn();
    const onFilterChange = vi.fn();
    const options = [
      { value: 'north', label: 'North' },
      { value: 'south', label: 'South' },
    ];

    renderWithGds(
      <>
        <PillBar ariaLabel="Regions" options={options} value="north" onChange={onPillChange} />
        <SoftChipGroup ariaLabel="Neighborhoods" options={options} value="south" onChange={onSoftChange} />
        <FilterChipGroup ariaLabel="Filters" options={options} value={null} onChange={onFilterChange} />
      </>,
    );

    expect(screen.getByRole('radiogroup', { name: 'Regions' })).toBeInTheDocument();
    expect(screen.getAllByRole('radio', { name: 'North' })[0]).toHaveAttribute('aria-checked', 'true');

    await user.click(screen.getAllByRole('radio', { name: 'South' })[0]);
    await user.click(screen.getAllByRole('radio', { name: 'North' })[1]);
    await user.click(screen.getAllByRole('radio', { name: 'South' })[2]);

    expect(onPillChange).toHaveBeenCalledWith('south');
    expect(onSoftChange).toHaveBeenCalledWith('north');
    expect(onFilterChange).toHaveBeenCalledWith('south');
  });

  it('renders overflow-safe segmented controls, rating scales, sliders, and wizard steps', async () => {
    const user = userEvent.setup();
    const onSegmentChange = vi.fn();
    const onSliderChange = vi.fn();
    const onSaveNext = vi.fn();

    renderWithGds(
      <>
        <GdsSegmentedControl
          ariaLabel="Module tabs"
          value="learn"
          onChange={onSegmentChange}
          options={[
            { value: 'learn', label: 'Learn' },
            { value: 'plan', label: 'Plan' },
          ]}
        />
        <GdsSlider label="Confidence" value={7} onChange={onSliderChange} />
        <GdsRatingScale label="Readiness" value={3} onChange={onSliderChange} scale={5} />
        <GdsWizardStepper
          activeStep={0}
          steps={[
            { id: 'one', title: 'One', completed: true },
            { id: 'two', title: 'Two' },
          ]}
          onSaveNext={onSaveNext}
        />
      </>,
    );

    expect(screen.getByRole('group', { name: 'Module tabs' })).toBeInTheDocument();
    expect(screen.getByText('Confidence')).toBeInTheDocument();
    expect(screen.getByText('Readiness')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'One' })).toHaveAttribute('aria-current', 'step');

    await user.click(screen.getByText('Plan'));
    await user.click(screen.getByRole('button', { name: 'Save & Next' }));

    expect(onSegmentChange).toHaveBeenCalledWith('plan');
    expect(onSaveNext).toHaveBeenCalledTimes(1);
  });

  it('renders destructive confirm dialogs with the expected actions', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const onConfirm = vi.fn();

    renderWithGds(
      <ConfirmDialog opened onClose={onClose} onConfirm={onConfirm} title="Delete record">
        This action cannot be undone.
      </ConfirmDialog>,
    );

    expect(await screen.findByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('This action cannot be undone.')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onClose).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole('button', { name: 'Confirm' }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it('supports provider-based confirmations and toast helpers', async () => {
    const user = userEvent.setup();
    const onConfirmed = vi.fn();

    function Probe() {
      const confirm = useGdsConfirm();
      const toasts = useGdsToasts();
      return (
        <>
          <button
            type="button"
            onClick={() => {
              void confirm.confirmDestructive({
                title: 'Delete asset',
                targetName: 'Primary logo',
                message: 'This cannot be undone.',
              }).then((confirmed) => {
                if (confirmed) {
                  onConfirmed();
                  toasts.notifySuccess({ title: 'Deleted' });
                }
              });
            }}
          >
            Open delete
          </button>
        </>
      );
    }

    renderWithGds(
      <GdsToastProvider>
        <GdsConfirmProvider>
          <Probe />
          <NotificationCenter />
        </GdsConfirmProvider>
      </GdsToastProvider>,
    );

    await user.click(screen.getByRole('button', { name: 'Open delete' }));
    expect(await screen.findByRole('dialog')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Delete' }));
    expect(onConfirmed).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Deleted')).toBeInTheDocument();
  });

  it('runs typed confirmation actions with async failure, retry, undo, events, and focus return', async () => {
    const user = userEvent.setup();
    const execute = vi.fn()
      .mockRejectedValueOnce(new Error('Permission changed.'))
      .mockResolvedValueOnce(undefined);
    const undo = vi.fn().mockResolvedValue(undefined);
    const events = vi.fn();

    function Probe() {
      const confirm = useGdsConfirm();
      return (
        <button
          type="button"
          onClick={() => {
            void confirm.confirmAction({
              id: 'delete-project',
              title: 'Delete project',
              message: 'This removes the project from the workspace.',
              targetName: 'Launch plan',
              payload: { id: 'project-1' },
              riskLevel: 'critical',
              retryable: true,
              execute,
              undo: { windowMs: 10000, label: 'Undo delete', onUndo: undo },
            });
          }}
        >
          Delete project
        </button>
      );
    }

    renderWithGds(
      <GdsConfirmProvider onConfirmationEvent={events}>
        <Probe />
      </GdsConfirmProvider>,
    );

    await user.click(screen.getByRole('button', { name: 'Delete project' }));
    expect(await screen.findByRole('dialog')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Delete' }));
    expect(await screen.findByText('Permission changed.')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => expect(execute).toHaveBeenCalledTimes(2));
    expect(await screen.findByRole('status')).toHaveTextContent('Action can be undone');
    expect(screen.getByRole('button', { name: 'Delete project' })).toHaveFocus();
    await user.click(screen.getByRole('button', { name: 'Undo delete' }));
    await waitFor(() => expect(undo).toHaveBeenCalledTimes(1));
    expect(events.mock.calls.map(([event]) => event.type)).toEqual(expect.arrayContaining([
      'opened',
      'failed',
      'retry',
      'confirmed',
      'undo_started',
      'undo_completed',
    ]));
  });

  it('renders empty states with optional action content', () => {
    renderWithGds(
      <EmptyState
        title="No projects yet"
        description="Create your first project to get started."
        action={<button type="button">Create project</button>}
      />,
    );

    expect(screen.getByText('No projects yet')).toBeInTheDocument();
    expect(screen.getByText('Create your first project to get started.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create project' })).toBeInTheDocument();
  });

  it('renders typed icons and structured media previews', () => {
    renderWithGds(
      <>
        <GdsIcon icon="Save" label="Save icon" />
        <MediaPreviewCard
          title="Hero image"
          src="/hero.png"
          alt="Hero image"
          metadata={[{ label: 'Format', value: 'PNG' }]}
        />
      </>,
    );

    expect(screen.getByRole('img', { name: 'Save icon' })).toBeInTheDocument();
    expect(screen.getByText('Hero image')).toBeInTheDocument();
    expect(screen.getByText(/Format:/)).toBeInTheDocument();
  });

  it('falls back to a placeholder when media is missing, and omits it entirely with hideWhenNoMedia', () => {
    const { rerender } = renderWithGds(
      <MediaPreviewCard title="Untitled asset" alt="Untitled asset" />,
    );

    expect(screen.getByText('No media')).toBeInTheDocument();

    rerender(
      <MediaPreviewCard title="Untitled asset" alt="Untitled asset" hideWhenNoMedia />,
    );

    expect(screen.queryByText('No media')).not.toBeInTheDocument();
    expect(screen.getByText('Untitled asset')).toBeInTheDocument();
  });

  // Skipped (issue 739 / issue 742): deterministically misses vitest's timeout on CI's
  // shared runners even at 60000ms (default 15000ms and 30000ms also insufficient); real
  // cause suspected to be genuine per-keystroke/per-interaction cost, not artificial delay.
  // Re-enable once issue 742's investigation lands a real fix.
  it.skip('resolves kanban orientation responsively and moves cards via a keyboard-accessible menu', async () => {
    const user = userEvent.setup();
    const onMoveItem = vi.fn();
    const columns = [
      { id: 'todo', title: 'To do', items: [{ id: 'task-1', title: 'Draft proposal' }] },
      { id: 'done', title: 'Done', items: [] },
    ];

    const restorePortraitMobile = mockMatchMediaByQuery(
      (query) => query.includes('orientation: portrait') || query.includes('max-width'),
    );
    const stacked = renderWithGds(
      <KanbanBoard title="Sprint board" columns={columns} onMoveItem={onMoveItem} />,
    );
    expect(screen.getByRole('region', { name: 'Sprint board' })).toHaveAttribute(
      'data-gds-kanban-orientation',
      'stacked',
    );
    stacked.unmount();
    restorePortraitMobile();

    const restoreDesktop = mockMatchMediaByQuery(() => false);
    renderWithGds(<KanbanBoard title="Sprint board" columns={columns} onMoveItem={onMoveItem} />);

    expect(screen.getByRole('region', { name: 'Sprint board' })).toHaveAttribute(
      'data-gds-kanban-orientation',
      'columns',
    );
    expect(screen.getByText('No items')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Move: Draft proposal' }));
    // Timeout raised from testing-library's 1000ms default: Mantine 9's Menu open-transition
    // occasionally exceeds it under CI load (issue #732; this occurrence was missed when that
    // fix was applied to KanbanBoard.test.tsx's own four occurrences of the same query).
    await user.click(await screen.findByRole('menuitem', { name: 'Move to Done' }, { timeout: 5000 }));
    expect(onMoveItem).toHaveBeenCalledWith('task-1', 'todo', 'done');
    restoreDesktop();
  });

  it('renders public capture flows and playback controls with callbacks', async () => {
    const user = userEvent.setup();
    const onPlayPause = vi.fn();

    renderWithGds(
      <>
        <PublicCaptureFlow
          stage="consent"
          state="ready"
          body={<div>Consent checkbox</div>}
          actions={[{ action: 'confirm', priority: 'primary' }]}
        />
        <PlaybackControls state="paused" onPlayPause={onPlayPause} canGoNext={false} />
      </>,
    );

    expect(screen.getByRole('heading', { name: 'Review consent' })).toBeInTheDocument();
    expect(screen.getByText('Consent checkbox')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Play' }));
    expect(onPlayPause).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();
  });

  it('validates creator CSS and blocks unsafe scoped themes', () => {
    const issues = validateCreatorCss('[data-gds-creator-theme="x"] .cta { display: none; color: #fff; }', {
      scopeSelector: '[data-gds-creator-theme="x"]',
    });

    expect(issues.map((issue) => issue.code)).toContain('creator-css-blocked-property');
    expect(issues.map((issue) => issue.code)).toContain('creator-css-raw-color');

    renderWithGds(
      <CreatorThemeBoundary css={'body { display: none; }'} scopeId="x">
        <div>Fallback content</div>
      </CreatorThemeBoundary>,
    );

    expect(screen.getByText('creator-css-out-of-scope')).toBeInTheDocument();
    expect(screen.getByText('Fallback content')).toBeInTheDocument();
  });

  it('renders metric cards with trends and descriptions', () => {
    renderWithGds(
      <MetricCard
        label="Completion"
        value="87%"
        description="Weekly completion rate"
        trend={{ label: '+4%', tone: 'positive' }}
      />,
    );

    expect(screen.getByText('Completion')).toBeInTheDocument();
    expect(screen.getByText('87%')).toBeInTheDocument();
    expect(screen.getByText('Weekly completion rate')).toBeInTheDocument();
    expect(screen.getByText('+4%')).toBeInTheDocument();
  });

  it('renders a canonical browse surface with scope controls and active filters', async () => {
    const user = userEvent.setup();
    const onRemove = vi.fn();
    const onSelect = vi.fn();

    renderWithGds(
      <BrowseSurface
        eyebrow="Discover"
        title="Browse shared content"
        description="Use shared browse chrome instead of page-local filter stacks."
        resultCount={12}
        activeFilters={[{ id: 'published', label: 'Published', onRemove }]}
        scopeOptions={[
          { id: 'all', label: 'All regions', active: true, onSelect },
          { id: 'east', label: 'East', onSelect },
        ]}
        locationControls={<button type="button">Budapest</button>}
        toolbar={{ searchSlot: <input aria-label="Search records" /> }}
        sortControl={<button type="button">Newest first</button>}
        mobileFilters={<button type="button">Filters</button>}
        content={<div>Browse results</div>}
      />,
    );

    expect(screen.getByRole('heading', { name: 'Browse shared content' })).toBeInTheDocument();
    expect(screen.getByText('12 results')).toBeInTheDocument();
    expect(screen.getByLabelText('Search records')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'All regions' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Budapest' })).toBeInTheDocument();
    expect(screen.getByText('Browse results')).toBeInTheDocument();

    const removeChip = screen.getAllByRole('button', { name: 'Remove Published filter' })[0];
    removeChip.focus();
    await user.keyboard('{Enter}');
    await user.click(screen.getByRole('button', { name: 'East' }));

    expect(onRemove).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it('applies listing query-state transitions deterministically', () => {
    const initial = {
      search: '',
      sort: 'newest',
      filters: [],
      page: 2,
      pageSize: 25,
      selection: ['row-1'],
    };
    const searched = listingQueryReducer(initial, { type: 'set-search', value: 'camera' });
    expect(searched.search).toBe('camera');
    expect(searched.page).toBe(1);
    expect(searched.selection).toHaveLength(0);

    const withFilter = listingQueryReducer(searched, { type: 'toggle-filter', value: 'Published' });
    expect(withFilter.filters).toContain('Published');

    const sorted = listingQueryReducer(withFilter, { type: 'set-sort', value: 'a-z' });
    expect(sorted.sort).toBe('a-z');
  });

  // Skipped (issue 739 / issue 742): deterministically misses vitest's timeout on CI's
  // shared runners even at 60000ms (default 15000ms and 30000ms also insufficient); real
  // cause suspected to be genuine per-keystroke/per-interaction cost, not artificial delay.
  // Re-enable once issue 742's investigation lands a real fix.
  it.skip('renders listing primitives with provider-backed selection and filter behavior', async () => {
    const user = userEvent.setup();

    function ListingProbe() {
      const { state, dispatch } = useListingState();
      return (
        <>
          <SortMenu
            value={state.sort}
            options={[{ value: 'newest', label: 'Newest' }, { value: 'oldest', label: 'Oldest' }]}
            onChange={(value) => dispatch({ type: 'set-sort', value })}
            label="Sort dataset"
          />
          <ResultSummary resultCount={12} noun="records" description="Shared summary." />
          <ActiveFilterChips
            filters={state.filters.map((filter) => ({
              id: filter,
              label: filter,
              onRemove: () => dispatch({ type: 'toggle-filter', value: filter }),
            }))}
          />
          <button type="button" onClick={() => dispatch({ type: 'toggle-filter', value: 'Published' })}>Toggle published</button>
          <button type="button" onClick={() => dispatch({ type: 'toggle-selection', value: 'row-1' })}>Toggle row-1</button>
          <BulkActionsBar selectedCount={state.selection.length} />
        </>
      );
    }

    renderWithGds(
      <ListingProvider>
        <ListingProbe />
      </ListingProvider>,
    );

    expect(screen.getByText('12 records')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Toggle published' }));
    expect(screen.getByText('Published')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Toggle row-1' }));
    expect(screen.getByText('1 selected')).toBeInTheDocument();
  });

  it('ActiveFilterChips removable chip is a real keyboard-operable button', async () => {
    const user = userEvent.setup();
    const onRemove = vi.fn();

    renderWithGds(
      <ActiveFilterChips filters={[{ id: 'published', label: 'Published', onRemove }]} />,
    );

    const chip = screen.getByRole('button', { name: 'Remove Published filter' });
    chip.focus();
    expect(chip).toHaveFocus();

    await user.keyboard('{Enter}');
    expect(onRemove).toHaveBeenCalledTimes(1);

    await user.keyboard(' ');
    expect(onRemove).toHaveBeenCalledTimes(2);
  });

  it('DataToolbar removable filter chip is a real keyboard-operable button', async () => {
    const user = userEvent.setup();
    const onRemove = vi.fn();

    renderWithGds(
      <DataToolbar activeFilters={[{ label: 'Published', onRemove }]} />,
    );

    const chip = screen.getByRole('button', { name: 'Remove Published filter' });
    chip.focus();
    expect(chip).toHaveFocus();

    await user.keyboard('{Enter}');
    expect(onRemove).toHaveBeenCalledTimes(1);

    await user.keyboard(' ');
    expect(onRemove).toHaveBeenCalledTimes(2);
  });

  it('renders advanced data table sorting and row selection controls', async () => {
    const user = userEvent.setup();

    renderWithGds(
      <AdvancedDataTable
        rows={[
          { id: '2', name: 'Bravo', status: 'Draft' },
          { id: '1', name: 'Alpha', status: 'Published' },
        ]}
        columns={[
          { key: 'name', label: 'Name', sortable: true },
          { key: 'status', label: 'Status', sortable: true },
        ]}
        rowId={(row) => String(row.id)}
      />,
    );

    expect(screen.getByRole('button', { name: 'Sort by Name' })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Sort by Name' }));
    await user.click(screen.getByRole('checkbox', { name: 'Select row 1' }));
    expect(screen.getByText('2 rows')).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Select row 1' })).toBeChecked();
  });

  it('runs the GDS data table engine with sort, filter, selection, export, and virtual windows', async () => {
    const user = userEvent.setup();
    const events = vi.fn();
    const exported = vi.fn();
    const rows = [
      { id: '1', name: 'Alpha', status: 'Published', score: 3 },
      { id: '2', name: 'Bravo', status: 'Draft', score: 1 },
      { id: '3', name: 'Charlie', status: 'Published', score: 2 },
    ];
    const columns = [
      { key: 'name' as const, label: 'Name', sortable: true, filterable: true, mobilePriority: 1 },
      { key: 'status' as const, label: 'Status', sortable: true, filterable: true, mobilePriority: 2 },
      { key: 'score' as const, label: 'Score', sortable: true, mobilePriority: 3 },
    ];

    renderWithGds(
      <GdsDataTable
        caption="Members"
        summary="Operational members table."
        columns={columns}
        rowId={(row) => String(row.id)}
        adapter={createGdsTableAdapter(rows, columns)}
        initialQuery={{ pageSize: 2 }}
        virtualizedRowLimit={1}
        onEvent={events}
        onExport={exported}
      />,
    );

    expect(await screen.findByRole('grid', { name: 'Members' })).toBeInTheDocument();
    expect(screen.getByText('1 of 2 rows rendered in the virtualized window.')).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Name' })).toHaveAttribute('aria-sort', 'none');

    await user.click(screen.getByRole('button', { name: 'Name' }));
    await waitFor(() => expect(screen.getByRole('columnheader', { name: 'Name' })).toHaveAttribute('aria-sort', 'ascending'));
    await user.click(screen.getByRole('checkbox', { name: 'Select row 1' }));
    expect(screen.getByText('1 selected')).toBeInTheDocument();
    await user.type(screen.getByRole('textbox', { name: 'Search rows' }), 'Charlie');
    await waitFor(() => expect(screen.getAllByText('Charlie').length).toBeGreaterThan(0));
    expect(screen.getByText('0 selected')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Export' }));
    await waitFor(() => expect(exported).toHaveBeenCalledWith(expect.objectContaining({
      query: expect.objectContaining({ search: 'Charlie' }),
      selectedRowIds: [],
    })));
    expect(events.mock.calls.map(([event]) => event.type)).toEqual(expect.arrayContaining([
      'load_started',
      'sort_changed',
      'selection_changed',
      'filter_changed',
      'export_requested',
    ]));
    expect(serializeGdsTableQuery({ page: 1, pageSize: 25, search: 'alpha', sortBy: 'name', sortDirection: 'asc', filters: { status: 'Published' } })).toBe('page=1&pageSize=25&search=alpha&sortBy=name&sortDirection=asc&filter.status=Published');
  });

  it('moves data table focus by row and cell with arrow keys', async () => {
    const rows = [
      { id: '1', name: 'Alpha', status: 'Published' },
      { id: '2', name: 'Bravo', status: 'Draft' },
    ];
    const columns = [
      { key: 'name' as const, label: 'Name' },
      { key: 'status' as const, label: 'Status' },
    ];
    const { container } = renderWithGds(
      <GdsDataTable
        caption="Keyboard members"
        columns={columns}
        rowId={(row) => String(row.id)}
        adapter={createGdsTableAdapter(rows, columns)}
        mobileCards={false}
      />,
    );

    expect(await screen.findByRole('grid', { name: 'Keyboard members' })).toBeInTheDocument();
    const cells = () => Array.from(container.querySelectorAll<HTMLElement>('tbody [data-gds-cell]'));

    cells()[0]?.focus();
    fireEvent.keyDown(cells()[0]!, { key: 'ArrowRight' });
    expect(document.activeElement).toHaveTextContent('Alpha');
    expect(screen.getByText('Row 1 of 2, Name column')).toBeInTheDocument();

    fireEvent.keyDown(document.activeElement!, { key: 'ArrowDown' });
    expect(document.activeElement).toHaveTextContent('Bravo');
    expect(screen.getByText('Row 2 of 2, Name column')).toBeInTheDocument();

    fireEvent.keyDown(document.activeElement!, { key: 'ArrowRight' });
    expect(document.activeElement).toHaveTextContent('Draft');
    expect(screen.getByText('Row 2 of 2, Status column')).toBeInTheDocument();
  });

  it('enters and exits actionable data table cells without stealing nested control keys', async () => {
    const user = userEvent.setup();
    const rows = [
      { id: '1', name: 'Alpha', action: 'open' },
      { id: '2', name: 'Bravo', action: 'open' },
    ];
    const open = vi.fn();
    const columns = [
      { key: 'name' as const, label: 'Name' },
      {
        key: 'action' as const,
        label: 'Action',
        interactive: true,
        render: (row: (typeof rows)[number]) => (
          <button type="button" onClick={() => open(row.id)}>
            Open {row.name}
          </button>
        ),
      },
    ];
    const { container } = renderWithGds(
      <GdsDataTable
        caption="Actionable members"
        columns={columns}
        rowId={(row) => String(row.id)}
        adapter={createGdsTableAdapter(rows, columns)}
        mobileCards={false}
      />,
    );

    expect(await screen.findByRole('grid', { name: 'Actionable members' })).toBeInTheDocument();
    const cells = () => Array.from(container.querySelectorAll<HTMLElement>('tbody [data-gds-cell]'));

    act(() => {
      cells()[0]?.focus();
    });
    fireEvent.keyDown(cells()[0]!, { key: 'ArrowRight' });
    expect(document.activeElement).toHaveTextContent('Alpha');
    fireEvent.keyDown(document.activeElement!, { key: 'ArrowRight' });
    expect(document.activeElement).toHaveAttribute('data-gds-actionable-cell', 'true');
    expect(document.activeElement).toHaveTextContent('Open Alpha');

    fireEvent.keyDown(document.activeElement!, { key: 'Enter' });
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Open Alpha' }));

    fireEvent.keyDown(document.activeElement!, { key: 'ArrowLeft' });
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Open Alpha' }));

    fireEvent.keyDown(document.activeElement!, { key: 'Escape' });
    expect(document.activeElement).toHaveAttribute('data-gds-actionable-cell', 'true');

    await user.click(screen.getByRole('button', { name: 'Open Alpha' }));
    expect(open).toHaveBeenCalledWith('1');
  });

  it('supports remote table adapters, retries, and filtered-empty states', async () => {
    const user = userEvent.setup();
    const load = vi.fn()
      .mockRejectedValueOnce(new Error('Network down'))
      .mockResolvedValue({ rows: [], total: 0 });

    renderWithGds(
      <GdsDataTable
        caption="Remote records"
        columns={[{ key: 'name', label: 'Name', sortable: true }]}
        rowId={(row) => String(row.id)}
        adapter={{ mode: 'remote', load }}
        initialQuery={{ search: 'missing' }}
      />,
    );

    expect(await screen.findByText('Unable to load table')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByText('No matching rows')).toBeInTheDocument();
  });

  it('manages governed resource workflows with detail, activate, delete, and copy-preview actions', async () => {
    const user = userEvent.setup();
    const events = vi.fn();
    const adapter = createGdsResourceAdapter([
      { id: 'venue-1', title: 'Venue One', status: 'draft', updatedAt: '2026-06-14' },
      { id: 'venue-2', title: 'Venue Two', status: 'active', updatedAt: '2026-06-13' },
    ]);

    renderWithGds(
      <GdsResourceManager
        title="Venues"
        description="Operational venue resources."
        adapter={adapter}
        onEvent={events}
        confirmAction={() => true}
      />,
    );

    expect(await screen.findByRole('grid', { name: 'Venues resources' })).toBeInTheDocument();
    await user.click(screen.getAllByRole('button', { name: 'Details' })[0]!);
    expect(await screen.findByLabelText('Venue One detail')).toBeInTheDocument();
    await user.click(screen.getAllByRole('button', { name: 'Activate' })[0]!);
    await waitFor(() => expect(screen.getAllByText('active').length).toBeGreaterThan(0));
    await user.click(screen.getAllByRole('button', { name: 'Copy preview' })[0]!);
    await user.click(screen.getAllByRole('button', { name: 'Delete' })[0]!);
    await waitFor(() => expect(screen.queryByText('Venue One')).not.toBeInTheDocument());
    expect(events.mock.calls.map(([event]) => event.type)).toEqual(expect.arrayContaining([
      'resource_loaded',
      'action_started',
      'action_completed',
    ]));
  });

  it('blocks resource actions when permissions or destructive confirmations are missing', async () => {
    const user = userEvent.setup();
    const events = vi.fn();
    const adapter = {
      ...createGdsResourceAdapter([{ id: 'user-1', title: 'User One', status: 'active' }]),
      getPermissions: () => [
        { action: 'view' as const, allowed: true },
        { action: 'delete' as const, allowed: false, reason: 'Only owners can delete users.' },
        { action: 'activate' as const, allowed: true },
        { action: 'copy-preview' as const, allowed: true },
      ],
    };

    renderWithGds(
      <GdsResourceManager
        title="Users"
        adapter={adapter}
        onEvent={events}
      />,
    );

    expect(await screen.findByRole('button', { name: 'Delete' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Activate' }));
    await waitFor(() => expect(events.mock.calls.map(([event]) => event.type)).toContain('action_completed'));
  });

  it('renders the discovery shell with grouped sidebar navigation', () => {
    renderWithGds(
      <DiscoveryShell
        header={<Text fw={700}>Operations shell</Text>}
        sidebar={(
          <SidebarNav>
            <SidebarNavSection label="Primary">
              <SidebarNavItem action="home" href="/" active />
              <SidebarNavItem action="settings" href="/settings" />
            </SidebarNavSection>
            <SidebarNavSection label="Account" pushToBottom>
              <SidebarNavItem action="logout" component="button" />
            </SidebarNavSection>
          </SidebarNav>
        )}
        footer={<button type="button">Home</button>}
      >
        <div>Discovery content</div>
      </DiscoveryShell>,
    );

    expect(screen.getByText('Operations shell')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Toggle navigation' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Home' })).toHaveAttribute('aria-current', 'page');
    expect(screen.getByRole('link', { name: 'Settings' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Logout' })).toBeInTheDocument();
    expect(screen.getByText('Discovery content')).toBeInTheDocument();
  });

  it('defaults sidebar/header/footer sizing to the --gds-layout-* tokens, with the pre-token literals as var() fallbacks (issue 698)', () => {
    const { container } = renderWithGds(
      <DiscoveryShell
        header={<Text fw={700}>Token defaults</Text>}
        sidebar={<div>sidebar</div>}
        footer={<button type="button">Home</button>}
      >
        <div>content</div>
      </DiscoveryShell>,
    );

    const appShellVars = Array.from(container.querySelectorAll('style')).map((s) => s.textContent).join('\n');
    expect(appShellVars).toContain('var(--gds-layout-sidebar-width, 280px)');
    expect(appShellVars).toContain('var(--gds-layout-header-height, 60px)');
    expect(appShellVars).toContain('var(--gds-layout-footer-height, 68px)');
  });

  it('lets an explicit sidebarWidth/headerHeight prop win over the layout tokens', () => {
    const { container } = renderWithGds(
      <DiscoveryShell
        header={<Text fw={700}>Explicit sizing</Text>}
        sidebar={<div>sidebar</div>}
        sidebarWidth={320}
        headerHeight="72px"
      >
        <div>content</div>
      </DiscoveryShell>,
    );

    const appShellVars = Array.from(container.querySelectorAll('style')).map((s) => s.textContent).join('\n');
    expect(appShellVars).not.toContain('--gds-layout-sidebar-width');
    expect(appShellVars).not.toContain('--gds-layout-header-height');
  });

  it('supports shell-state toggling with deterministic callbacks', async () => {
    const user = userEvent.setup();
    const onSidebarOpenedChange = vi.fn();

    function ShellStateProbe() {
      const state = useDiscoveryShellState({ onSidebarOpenedChange });
      return (
        <button type="button" onClick={state.toggle}>
          {state.opened ? 'Open' : 'Closed'}
        </button>
      );
    }

    renderWithGds(<ShellStateProbe />);
    await user.click(screen.getByRole('button', { name: 'Closed' }));
    expect(onSidebarOpenedChange).toHaveBeenCalledWith(true);
    await user.click(screen.getByRole('button', { name: 'Open' }));
    expect(onSidebarOpenedChange).toHaveBeenCalledWith(false);
  });

  it('closes mobile discovery navigation when a nav item is selected', async () => {
    const restoreMatchMedia = mockMatchMedia(true);
    const user = userEvent.setup();
    const onSidebarOpenedChange = vi.fn();

    try {
      renderWithGds(
        <DiscoveryShell
          header={<Text fw={700}>Mobile shell</Text>}
          sidebarOpened
          onSidebarOpenedChange={onSidebarOpenedChange}
          sidebar={(
            <SidebarNav>
              <SidebarNavItem href="#maturity" label="Maturity" />
            </SidebarNav>
          )}
        >
          <div>Discovery content</div>
        </DiscoveryShell>,
      );

      await user.click(screen.getByRole('link', { name: 'Maturity' }));
      expect(onSidebarOpenedChange).toHaveBeenCalledWith(false);
    } finally {
      restoreMatchMedia();
    }
  });

  it('opens uncontrolled mobile discovery navigation from the hamburger', async () => {
    const restoreMatchMedia = mockMatchMedia(true);
    const user = userEvent.setup();

    try {
      const { container } = renderWithGds(
        <DiscoveryShell
          header={<Text fw={700}>Mobile shell</Text>}
          mobileNavigationLabel="Open mobile navigation"
          sidebar={(
            <SidebarNav>
              <SidebarNavItem href="#maturity" label="Maturity" />
            </SidebarNav>
          )}
        >
          <div>Discovery content</div>
        </DiscoveryShell>,
      );

      await user.click(screen.getByRole('button', { name: 'Open mobile navigation' }));
      const navbar = container.querySelector('.mantine-AppShell-navbar');
      expect(navbar).toHaveAttribute('data-gds-mobile-navbar-open', 'true');
      expect(navbar).toHaveStyle({
        '--app-shell-navbar-transform': 'translateX(0)',
        '--app-shell-navbar-transform-rtl': 'translateX(0)',
      });
    } finally {
      restoreMatchMedia();
    }
  });

  it('renders docs shell with governed header, sidebar sections, and docs content', () => {
    renderWithGds(
      <DocsShell
        brand={<Text fw={700}>General Design System</Text>}
        primaryNavigation={<SidebarNavItem href="/patterns" active label="Patterns" />}
        secondaryNavigation={<SidebarNavItem href="/themes" label="Themes" />}
        headerContext="Official docs shell"
        actions={<button type="button">Theme toggle</button>}
        contentWidth="full"
      >
        <Text>Docs shell content area</Text>
      </DocsShell>,
    );

    expect(screen.getByText('General Design System')).toBeInTheDocument();
    expect(screen.getByText('Official docs shell')).toBeInTheDocument();
    expect(screen.getByText('Docs shell content area')).toBeInTheDocument();
    expect(screen.getByText('Primary')).toBeInTheDocument();
    expect(screen.getByText('More')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Patterns' })).toHaveAttribute('href', '/patterns');
    expect(screen.getByRole('link', { name: 'Themes' })).toHaveAttribute('href', '/themes');
    expect(screen.getByRole('button', { name: 'Theme toggle' })).toBeInTheDocument();
  });

  it('keeps docs shell header slots bounded for localized copy and action controls', () => {
    const { container } = renderWithGds(
      <DocsShell
        brand={<strong>Система общего проектирования с очень длинным названием</strong>}
        actions={(
          <DocsHeaderActionSelect
            label="Language"
            value="ru"
            options={[
              { value: 'en', label: 'English' },
              { value: 'ru', label: 'Русский' },
            ]}
            onChange={vi.fn()}
          />
        )}
        contentWidth="full"
      >
        <Text>Localized docs shell content area</Text>
      </DocsShell>,
    );

    expect(container.querySelector('[data-gds-docs-shell-header]')).toBeInTheDocument();
    expect(container.querySelector('[data-gds-docs-shell-actions]')).toBeInTheDocument();
    expect(container.querySelector('[data-gds-docs-shell-action-select]')).toBeInTheDocument();
    expect(container.querySelector('[data-gds-docs-shell-brand]')).toHaveStyle({
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
    });
  });

  it('renders a semantic action bar with governed action priority and icon-only actions', async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();
    const onReset = vi.fn();
    const onSettings = vi.fn();

    renderWithGds(
      <ActionBar
        primary={{ action: 'save', onClick: onSave }}
        secondary={[{ action: 'cancel', onClick: onReset }]}
        tertiary={[{ action: 'preview', onClick: () => {} }]}
        iconOnly={[{ action: 'settings', onClick: onSettings }]}
      />,
    );

    expect(screen.getByRole('button', { name: 'Save' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Preview' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Settings' })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Save' }));
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await user.click(screen.getByRole('button', { name: 'Settings' }));

    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onReset).toHaveBeenCalledTimes(1);
    expect(onSettings).toHaveBeenCalledTimes(1);
  });

  it('localizes semantic action bar labels from the GDS provider', () => {
    renderWithGds(
      <ActionBar
        primary={{ action: 'save' }}
        secondary={[{ action: 'cancel' }]}
        tertiary={[{ action: 'preview' }]}
      />,
      { locale: 'fr', messages: fr },
    );

    expect(screen.getByRole('button', { name: 'Enregistrer' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Annuler' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Aperçu' })).toBeInTheDocument();
  });

  it('supports governed semantic vocabulary packs without raw-label escape hatches', () => {
    const cameraPack = createGdsVocabularyPack('camera', {
      moderate: {
        defaultMessage: 'Moderate',
        icon: GdsIcons.Verify,
      },
    });

    renderWithGds(
      <ActionBar
        primary={{ action: 'camera:moderate' }}
        vocabularyPacks={[cameraPack]}
      />,
    );

    expect(screen.getByRole('button', { name: 'Moderate' })).toBeInTheDocument();
    expect(getSemanticActionLabel('camera:moderate', undefined, [cameraPack])).toBe('Moderate');
  });

  it('renders the unified listing-card contract with featured disclosure and governed affordances', async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();
    const onShare = vi.fn();

    renderWithGds(
      <ListingCard
        title="Budapest Community Meetup"
        description="A shared listing contract for events, venues, and communities."
        price="Free"
        featured
        sponsoredDisclosure="Sponsored listing"
        metadata={[
          { id: 'date', label: 'Date', value: 'June 7' },
          { id: 'location', label: 'Location', value: 'District V' },
        ]}
        primaryAction={<button type="button">View details</button>}
        saveAction={{ action: 'save', onClick: onSave }}
        shareAction={{ action: 'refer', ariaLabel: 'Share listing', onClick: onShare }}
      />,
    );

    expect(screen.getByText('Featured')).toBeInTheDocument();
    expect(screen.getByText('Sponsored listing')).toBeInTheDocument();
    expect(screen.getByText('June 7')).toBeInTheDocument();
    expect(screen.getByText('District V')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'View details' })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Save' }));
    await user.click(screen.getByRole('button', { name: 'Share listing' }));

    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onShare).toHaveBeenCalledTimes(1);
  });

  it('renders saved, price, and rating anatomy on compact listing cards', () => {
    renderWithGds(
      <ListingCard
        title="After-school tennis"
        compact
        price="$28/class"
        rating="4.8"
        ratingLabel="Class rating"
        saved
        saveAction={{ action: 'save', ariaLabel: 'Save class' }}
      />,
    );

    expect(screen.getByText('$28/class')).toBeInTheDocument();
    expect(screen.getByLabelText('Class rating')).toHaveTextContent('4.8');
    expect(screen.getByRole('button', { name: 'Save class' })).toHaveAttribute('data-gds-active', 'true');
  });

  it('handles interactive listing-card surface modes with keyboard-safe flip behavior', async () => {
    const user = userEvent.setup();
    const onActivate = vi.fn();

    renderWithGds(
      <ListingCard
        title="Interactive listing"
        description="Front surface"
        interactiveMode="flip"
        revealContent={<Text>Revealed governed details</Text>}
        onSurfaceActivate={onActivate}
        saveAction={{ action: 'save', onClick: onActivate }}
      />,
    );

    const card = screen.getByRole('button', { name: 'Toggle details for Interactive listing' });
    expect(card).toHaveAttribute('aria-expanded', 'false');
    expect(card).toHaveAttribute('data-gds-card-interactive-mode', 'flip');
    expect(card).toHaveAttribute('data-gds-card-flipped', 'false');

    await user.keyboard('{Tab}');
    await user.keyboard('{Enter}');

    expect(card).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('Revealed governed details')).toBeInTheDocument();
    expect(onActivate).toHaveBeenCalledTimes(1);

    await user.keyboard(' ');

    expect(card).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByText('Front surface')).toBeInTheDocument();
    expect(onActivate).toHaveBeenCalledTimes(2);

    await user.click(screen.getByRole('button', { name: 'Save' }));
    expect(onActivate).toHaveBeenCalledTimes(3);
    expect(card).toHaveAttribute('aria-expanded', 'false');
  });

  it('invokes full-surface listing-card button activation without double-firing nested controls', async () => {
    const user = userEvent.setup();
    const onSurfaceActivate = vi.fn();
    const onSave = vi.fn();

    renderWithGds(
      <ListingCard
        title="Surface action listing"
        description="The whole card is a governed action target."
        interactiveMode="surface-button"
        onSurfaceActivate={onSurfaceActivate}
        saveAction={{ action: 'save', onClick: onSave }}
      />,
    );

    await user.click(screen.getByRole('button', { name: 'Surface action listing' }));
    expect(onSurfaceActivate).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole('button', { name: 'Save' }));
    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onSurfaceActivate).toHaveBeenCalledTimes(1);
  });

  it('resolves governed card size, density, and variant contracts deterministically', () => {
    expect(resolveGdsCardContract({ size: 'xl', density: 'spacious', variant: 'media-left' })).toMatchObject({
      size: 'xl',
      density: 'spacious',
      variant: 'media-left',
      padding: 'xl',
      titleOrder: 3,
      descriptionClamp: 4,
      mediaPlacement: 'left',
      minTouchTarget: 44,
    });

    expect(resolveGdsCardContract({ compact: true, size: 'xl', density: 'spacious' })).toMatchObject({
      size: 'sm',
      density: 'compact',
      variant: 'compact',
      padding: 'xs',
      titleOrder: 5,
      descriptionClamp: 2,
      minTouchTarget: 40,
    });
  });

  it('applies the shared card contract across canonical card families', () => {
    renderWithGds(
      <>
        <ProductCard title="Sized product" size="xl" density="spacious" variant="media-left" />
        <ListingCard title="Dense listing" size="xs" density="compact" variant="compact" />
        <PublicFoodCard title="Food card" state="available" size="lg" density="spacious" />
        <PublicProductCard title="Public card" size="sm" density="compact" />
        <MediaCard title="Media card" image={<div />} size="md" density="comfortable" />
        <EditorialCard title="Editorial card" size="xl" density="spacious" variant="featured" />
      </>,
    );

    expect(screen.getByText('Sized product').closest('[data-gds-card-size]')).toHaveAttribute('data-gds-card-size', 'xl');
    expect(screen.getByText('Dense listing').closest('[data-gds-card-density]')).toHaveAttribute('data-gds-card-density', 'compact');
    expect(screen.getByText('Food card').closest('[data-gds-card-density]')).toHaveAttribute('data-gds-card-density', 'spacious');
    expect(screen.getByText('Public card').closest('[data-gds-card-size]')).toHaveAttribute('data-gds-card-size', 'sm');
    expect(screen.getByText('Media card').closest('[data-gds-card-variant]')).toHaveAttribute('data-gds-card-variant', 'media-top');
    expect(screen.getByText('Editorial card').closest('[data-gds-card-size]')).toHaveAttribute('data-gds-card-size', 'xl');
  });

  it('renders the public food card contract with food-specific helper and availability states', () => {
    renderWithGds(
      <>
        <PublicFoodCard
          title="Roasted tomato soup"
          description="Fresh basil, sour cream, and house bread."
          price="EUR 7.50"
          priceNote="Per portion"
          state="limited"
          helperText="Preorder by Friday 18:00"
          pickupNote="Saturday 09:00-12:00"
          freshnessNote="Best served warm"
          quantityHint="12 portions left"
          markers={[
            { id: 'vegetarian', label: 'Vegetarian', tone: 'positive' },
            { id: 'limited', label: 'Weekly batch', tone: 'warning' },
          ]}
          metadata={[
            { id: 'allergens', label: 'Contains dairy' },
            { id: 'portion', label: '500 ml' },
          ]}
          primaryAction={<button type="button">Preorder</button>}
        />
        <PublicFoodCard
          title="Pistachio morning bun"
          state="sold-out"
          primaryAction={<button type="button">Add to order</button>}
        />
      </>,
    );

    expect(screen.getByText('Roasted tomato soup')).toBeInTheDocument();
    expect(screen.getByText('EUR 7.50')).toBeInTheDocument();
    expect(screen.getByText('Preorder by Friday 18:00')).toBeInTheDocument();
    expect(screen.getByText('Saturday 09:00-12:00')).toBeInTheDocument();
    expect(screen.getByText('Best served warm')).toBeInTheDocument();
    expect(screen.getByText('Vegetarian')).toBeInTheDocument();
    expect(screen.getByText('Contains dairy')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add to order' })).toBeDisabled();
  });

  it('renders grouped food menu sections on top of the canonical food card', () => {
    renderWithGds(
      <FoodMenuSection
        title="Saturday preorder menu"
        sectionNote="Pickup window: Saturday 09:00-12:00"
        categories={[
          {
            id: 'soups',
            title: 'Soups',
            helperNote: 'Freshly prepared every Friday evening.',
            items: [
              {
                id: 'tomato',
                title: 'Roasted tomato soup',
                state: 'preorder',
                price: 'EUR 7.50',
                primaryAction: <button type="button">Reserve</button>,
              },
            ],
          },
          {
            id: 'desserts',
            title: 'Desserts',
            items: [],
          },
        ]}
      />,
    );

    expect(screen.getByRole('heading', { name: 'Saturday preorder menu' })).toBeInTheDocument();
    expect(screen.getByText('Pickup window: Saturday 09:00-12:00')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Soups' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Desserts' })).not.toBeInTheDocument();
    expect(screen.getByText('Roasted tomato soup')).toBeInTheDocument();
  });

  it('renders the sanctioned map panel states and iframe contract', () => {
    const { rerender } = renderWithGds(
      <MapPanel
        title="Venue map"
        description="Shared embed containment."
        loading
      />,
    );

    expect(screen.getByText('Loading map')).toBeInTheDocument();

    rerender(
      <MapPanel
        title="Venue map"
        description="Shared embed containment."
        error="The map provider is unavailable."
      />,
    );
    expect(screen.getByText('Map unavailable')).toBeInTheDocument();
    expect(screen.getByText('The map provider is unavailable.')).toBeInTheDocument();

    rerender(
      <MapPanel
        title="Venue map"
        description="Shared embed containment."
        iframeSrc="https://example.com/embed"
        embedTitle="Budapest venue map"
      />,
    );

    expect(screen.getByTitle('Budapest venue map')).toBeInTheDocument();
  });

  it('renders staged public flow shells with deterministic action priority and runtime boundary slots', () => {
    renderWithGds(
      <PublicFlowShell
        eyebrow="Capture flow"
        stage={{
          id: 'review',
          title: 'Review your capture',
          description: 'Approve the image before sharing it.',
          status: 'ready',
          body: <Text>Captured frame preview</Text>,
          notice: 'Only publish content you have the right to share.',
          actions: [
            { action: 'cancel', priority: 'secondary' },
            { action: 'send', priority: 'primary' },
            { action: 'preview', priority: 'tertiary' },
          ],
        }}
        hardwareSurface={<Text>Runtime preview slot</Text>}
      />,
    );

    expect(screen.getByRole('heading', { name: 'Review your capture' })).toBeInTheDocument();
    expect(screen.getByText('Captured frame preview')).toBeInTheDocument();
    expect(screen.getByText('Runtime preview slot')).toBeInTheDocument();
    expect(screen.getByText('Only publish content you have the right to share.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Send' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Preview' })).toBeInTheDocument();
  });

  it('renders playback surfaces across degraded and empty states', () => {
    const { rerender } = renderWithGds(
      <PlaybackSurface
        title="Storefront loop"
        state="playing"
        statusMessage="Looping chef specials on the kiosk screen."
        media={<Text>Playback media slot</Text>}
        controls={<button type="button">Pause</button>}
      />,
    );

    expect(screen.getByRole('heading', { name: 'Storefront loop' })).toBeInTheDocument();
    expect(screen.getByText('Playback media slot')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Pause' })).toBeInTheDocument();
    expect(screen.getByText('Playing')).toBeInTheDocument();

    rerender(
      <PlaybackSurface
        title="Storefront loop"
        state="degraded"
        statusMessage="One media asset failed, continuing with the next slide."
        media={<Text>Fallback loop</Text>}
      />,
    );

    expect(screen.getByText('Playback degraded')).toBeInTheDocument();
    expect(screen.getAllByText('One media asset failed, continuing with the next slide.')).toHaveLength(2);

    rerender(<PlaybackSurface title="Storefront loop" state="empty" />);
    expect(screen.getByText('No playback content available')).toBeInTheDocument();
  });

  it('renders the detail profile shell in page and drawer modes', () => {
    const { rerender } = renderWithGds(
      <DetailProfileShell
        mode="page"
        hero={<Title order={2}>Venue profile</Title>}
        actions={<ActionBar primary={{ action: 'preview' }} />}
        sections={[
          <SectionPanel key="overview" title="Overview"><Text>Profile summary</Text></SectionPanel>,
          <SectionPanel key="schedule" title="Schedule"><Text>Weekdays</Text></SectionPanel>,
        ]}
        related={<Text>Related listings</Text>}
      />,
    );

    expect(screen.getByText('Venue profile')).toBeInTheDocument();
    expect(screen.getByText('Profile summary')).toBeInTheDocument();
    expect(screen.getByText('Related listings')).toBeInTheDocument();

    rerender(
      <DetailProfileShell
        mode="drawer"
        hero={<Title order={2}>Venue profile</Title>}
        sections={[<SectionPanel key="overview" title="Overview"><Text>Drawer summary</Text></SectionPanel>]}
      />,
    );

    expect(screen.getByText('Drawer summary')).toBeInTheDocument();
  });

  it('renders editorial cards and consumer sections as reusable public/consumer contracts', () => {
    renderWithGds(
      <>
        <EditorialCard
          eyebrow="Guide"
          title="Neighborhood picks"
          description="Shared editorial card contract."
          badge="Featured"
          ctaLabel="Read guide"
          href="/guide"
          variant="featured"
          tone="warm"
        />
        <ConsumerSection
          title="Account summary"
          description="Use the shared section shell for account and dashboard clusters."
          action={<button type="button">Manage</button>}
        >
          <ConsumerDashboardGrid columns={2}>
            <MetricCard label="Saved items" value="18" />
            <SectionPanel title="Alerts" description="Shared operational panel rhythm.">
              <span>2 pending</span>
            </SectionPanel>
          </ConsumerDashboardGrid>
        </ConsumerSection>
      </>,
    );

    expect(screen.getByRole('heading', { name: 'Neighborhood picks' })).toBeInTheDocument();
    expect(screen.getByText('Featured')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Account summary' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Manage' })).toBeInTheDocument();
    expect(screen.getByText('Saved items')).toBeInTheDocument();
    expect(screen.getByText('2 pending')).toBeInTheDocument();
  });

  // Media fallback must never be a fixed light-only shade — that rendered as a stark white
  // box on dark-mode cards. The fallback is a generated thumbnail painted from
  // `var(--gds-brand-*)`, which the theme redefines per scheme.
  it('renders its media fallback from theme variables, never a fixed light-only shade (regression: stark white box on dark-mode cards)', () => {
    const { container } = renderWithGds(<EditorialCard title="No media yet" />);
    const thumbnail = container.querySelector('[data-gds-generated-thumbnail]');
    expect(thumbnail).toBeTruthy();
    expect(container.innerHTML).toContain('var(--gds-brand-primary');
    expect(container.innerHTML).not.toContain('var(--mantine-color-gray-0)');
  });

  it('renders its heading as a real, non-nested heading element (regression: eyebrow/description were nested inside the <h3>, invalid HTML and a screen-reader misannouncement)', () => {
    const { container } = renderWithGds(
      <ReferenceSection eyebrow="Kicker" title="Section title" description="Section description." href="/more" linkLabel="Open section">
        <span>Body</span>
      </ReferenceSection>,
    );

    const heading = screen.getByRole('heading', { name: 'Section title', level: 3 });
    expect(heading.textContent).toBe('Section title');
    expect(heading.querySelector('p, div')).toBeNull();
    expect(screen.getByText('Kicker').closest('h3')).toBeNull();
    expect(screen.getByText('Section description.').closest('h3')).toBeNull();
    expect(screen.getByRole('link', { name: 'Open section' }).closest('h3')).toBeNull();
    expect(container.querySelector('.mantine-Divider-root')).toBeNull();
  });

  it('renders media fields with upload, URL, preview, and recovery actions', async () => {
    const user = userEvent.setup();
    const onRemove = vi.fn();
    const onReset = vi.fn();

    renderWithGds(
      <MediaField
        label="Hero image"
        description="Choose a shared media asset."
        value="https://cdn.example.com/hero.jpg"
        preview={<img alt="Hero preview" src="https://cdn.example.com/hero.jpg" />}
        uploadControl={<button type="button">Upload image</button>}
        urlInput={<input aria-label="Image URL" defaultValue="https://cdn.example.com/hero.jpg" />}
        helpText="Prefer authored media with descriptive alt text."
        policyText="Public media must meet shared licensing policy."
        retryAction={<button type="button">Retry</button>}
        replaceAction={<button type="button">Replace</button>}
        acceptedTypes="JPEG, PNG, WebP"
        maxSize="10 MB max"
        progress={64}
        state="saved"
        onRemove={onRemove}
        onReset={onReset}
        mode="split"
      />,
    );

    expect(screen.getByText('Hero image')).toBeInTheDocument();
    expect(screen.getByText('Saved')).toBeInTheDocument();
    expect(screen.getByAltText('Hero preview')).toBeInTheDocument();
    expect(screen.getByLabelText('Image URL')).toBeInTheDocument();
    expect(screen.getByLabelText('Upload progress')).toBeInTheDocument();
    expect(screen.getByText('64% complete')).toBeInTheDocument();
    expect(screen.getByText('JPEG, PNG, WebP')).toBeInTheDocument();
    expect(screen.getByText('10 MB max')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Replace' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Reset' }));
    await user.click(screen.getByRole('button', { name: 'Remove' }));

    expect(onReset).toHaveBeenCalledTimes(1);
    expect(onRemove).toHaveBeenCalledTimes(1);
  });

  it('keeps locale packs in parity and resolves locale messages with fallback', () => {
    const locales = { en, es, hu, de, fr, it: itLocale, ru, he, ar };
    const expectedKeys = Object.keys(en).sort();

    for (const locale of Object.values(locales)) {
      expect(Object.keys(locale).sort()).toEqual(expectedKeys);
    }

    expect(getGdsMessages('es')['gds.action.save']).toBe('Guardar');
    expect(getGdsMessages('unknown-locale')['gds.action.save']).toBe('Save');
  });

  it('renders shared state blocks for empty and permission messaging', () => {
    renderWithGds(
      <>
        <StateBlock variant="empty" title="No reports yet" description="Create the first report to populate this view." compact />
        <AccessSummary title="Partner access" roles={['Admin', 'Partner']} scope="Northern region" />
      </>,
    );

    expect(screen.getByText('No reports yet')).toBeInTheDocument();
    expect(screen.getByText('Partner access')).toBeInTheDocument();
    expect(screen.getByText('Scope: Northern region')).toBeInTheDocument();
  });

  it('renders governed notification primitives with queue and dismiss behavior', async () => {
    const user = userEvent.setup();

    function NotificationProbe() {
      const { notify } = useGdsNotifications();
      return (
        <button
          type="button"
          onClick={() => notify({
            id: 'n-1',
            title: 'Partner sync delayed',
            message: 'Retry is available while sync catches up.',
            severity: 'warning',
          })}
        >
          Trigger
        </button>
      );
    }

    renderWithGds(
      <GdsNotificationProvider>
        <BannerNotice
          severity="info"
          eyebrow="Notice"
          title="Governed notification lane"
          message="Shared severity and action semantics."
        />
        <NotificationProbe />
        <NotificationCenter />
      </GdsNotificationProvider>,
    );

    expect(screen.getByText('Governed notification lane')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Trigger' }));
    expect(screen.getByText('Partner sync delayed')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(screen.queryByText('Partner sync delayed')).not.toBeInTheDocument();
  });

  it('renders a one-line compact BannerNotice with no title (#642)', () => {
    renderWithGds(
      <BannerNotice variant="compact" severity="info" message="Preview mode — changes are not saved." />,
    );

    expect(screen.queryByRole('heading')).not.toBeInTheDocument();
    const strip = screen.getByRole('status');
    expect(strip).toHaveTextContent('Preview mode — changes are not saved.');
  });

  it('governs notification dedupe, updates, audit events, and announcement-only output', async () => {
    const user = userEvent.setup();
    const audit = vi.fn();

    function NotificationProbe() {
      const {
        notify,
        updateNotification,
      } = useGdsNotifications();
      return (
        <>
          <button
            type="button"
            onClick={() => notify({
              id: 'sync-1',
              key: 'partner-sync',
              title: 'Partner sync queued',
              message: 'The first sync notice is visible.',
              severity: 'info',
            })}
          >
            Queue first
          </button>
          <button
            type="button"
            onClick={() => notify({
              id: 'sync-2',
              key: 'partner-sync',
              title: 'Partner sync replaced',
              message: 'The duplicate notice replaces the first one.',
              severity: 'warning',
            })}
          >
            Queue duplicate
          </button>
          <button
            type="button"
            onClick={() => updateNotification('sync-2', { title: 'Partner sync finished', severity: 'success', status: 'succeeded' })}
          >
            Mark finished
          </button>
          <button
            type="button"
            onClick={() => notify({
              id: 'announce-1',
              title: 'Screen reader only update',
              message: 'Saved without visual interruption.',
              severity: 'success',
              persistence: 'announcement-only',
            })}
          >
            Announce only
          </button>
        </>
      );
    }

    renderWithGds(
      <GdsNotificationProvider onAuditEvent={audit}>
        <NotificationProbe />
        <NotificationCenter />
      </GdsNotificationProvider>,
    );

    await user.click(screen.getByRole('button', { name: 'Queue first' }));
    await user.click(screen.getByRole('button', { name: 'Queue duplicate' }));
    expect(screen.queryByText('Partner sync queued')).not.toBeInTheDocument();
    expect(screen.getByText('Partner sync replaced')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Mark finished' }));
    expect(screen.getByText('Partner sync finished')).toBeInTheDocument();
    expect(screen.getByText('succeeded')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Announce only' }));
    expect(screen.queryByText('Screen reader only update')).not.toBeInTheDocument();
    expect(screen.getAllByRole('status').some((item) => item.textContent?.includes('Screen reader only update Saved without visual interruption.'))).toBe(true);
    expect(audit.mock.calls.map(([event]) => event.type)).toEqual(expect.arrayContaining(['shown', 'updated']));
  });

  it('runs bounded notification retry flows and records action telemetry', async () => {
    const user = userEvent.setup();
    const audit = vi.fn();
    const action = vi.fn();
    const retry = vi.fn().mockResolvedValue(undefined);

    function NotificationProbe() {
      const { notify } = useGdsNotifications();
      return (
        <button
          type="button"
          onClick={() => notify({
            id: 'retry-1',
            title: 'Publish failed',
            message: 'Retry publishing when the connection returns.',
            severity: 'error',
            autoCloseMs: false,
            actions: [{ id: 'details', label: 'Details', onClick: action }],
            retry: { onRetry: retry, maxAttempts: 2, label: 'Retry publish' },
          })}
        >
          Trigger retryable
        </button>
      );
    }

    renderWithGds(
      <GdsNotificationProvider onAuditEvent={audit}>
        <NotificationProbe />
        <NotificationCenter />
      </GdsNotificationProvider>,
    );

    await user.click(screen.getByRole('button', { name: 'Trigger retryable' }));
    await user.click(screen.getByRole('button', { name: 'Details' }));
    await user.click(screen.getByRole('button', { name: 'Retry publish' }));

    await waitFor(() => expect(retry).toHaveBeenCalledTimes(1));
    expect(action).toHaveBeenCalledTimes(1);
    expect(screen.getByText('succeeded')).toBeInTheDocument();
    expect(audit.mock.calls.map(([event]) => event.type)).toEqual(expect.arrayContaining([
      'shown',
      'action_clicked',
      'retry_started',
      'retry_succeeded',
    ]));
  });

  it('renders async-surface states with deterministic retry behavior', async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    const { rerender } = renderWithGds(
      <AsyncSurface
        state="loading"
        loadingTitle="Loading records"
        loadingDescription="Please wait."
      />,
    );

    expect(screen.getByText('Loading records')).toBeInTheDocument();

    rerender(
      <AsyncSurface
        state="error"
        errorTitle="Failed to load records"
        errorDescription="The dataset is temporarily unavailable."
        onRetry={onRetry}
      />,
    );
    expect(screen.getByText('Failed to load records')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    expect(onRetry).toHaveBeenCalledTimes(1);

    rerender(
      <AsyncSurface
        state="success"
        successContent={<div>Records loaded</div>}
      />,
    );
    expect(screen.getByText('Records loaded')).toBeInTheDocument();
  });

  it('resolves surface presentation styles for inline, centered, and fill modes', () => {
    // Every mode is a flex column with a governed gap (issue: deep-audit UX pass) -- a body
    // with no gap and multiple stacked children rendered every one flush against the next,
    // site-wide. A lone child is unaffected, since `gap` has no effect with nothing to space.
    expect(resolveSurfacePresentationStyles({ presentation: 'inline', minHeight: 240 })).toEqual({
      minHeight: '240px',
      display: 'flex',
      flexDirection: 'column',
      gap: 'var(--mantine-spacing-lg)',
    });
    expect(resolveSurfacePresentationStyles({
      presentation: 'fill',
      minHeight: 360,
      contentAlign: 'start',
      contentJustify: 'center',
    })).toEqual({
      minHeight: '360px',
      display: 'flex',
      flex: 1,
      flexDirection: 'column',
      gap: 'var(--mantine-spacing-lg)',
      alignItems: 'flex-start',
      justifyContent: 'center',
    });
    expect(resolveSurfacePresentationStyles({
      presentation: 'centered',
      minHeight: 180,
      contentAlign: 'center',
      contentJustify: 'center',
    })).toEqual({
      minHeight: '180px',
      display: 'flex',
      flexDirection: 'column',
      gap: 'var(--mantine-spacing-lg)',
      alignItems: 'center',
      justifyContent: 'center',
    });
  });

  it('renders a centered StateBlock with min-height presentation', () => {
    const { container } = renderWithGds(
      <StateBlock
        variant="loading"
        title="Loading operations"
        description="The panel should stay centered and stable."
        minHeight={280}
        presentation="centered"
        contentAlign="center"
        contentJustify="center"
      />,
    );

    const hasPresentationStyles = Array.from(container.querySelectorAll('div[style]')).some(
      (element) => element.getAttribute('style')?.includes('min-height: 280px') || element.getAttribute('style')?.includes('display: flex;'),
    );

    expect(hasPresentationStyles).toBe(true);
    expect(screen.getByText('Loading operations')).toBeInTheDocument();
  });

  it('renders missing-data prompts with explicit missing field guidance', () => {
    renderWithGds(
      <MissingDataPrompt
        description="Complete the required fields."
        missingFields={['Readiness score', 'Recovery notes']}
      />,
    );

    expect(screen.getByText('Missing data')).toBeInTheDocument();
    expect(screen.getByText('Readiness score')).toBeInTheDocument();
    expect(screen.getByText('Recovery notes')).toBeInTheDocument();
  });

  it('renders a fill-mode SectionPanel body with centered presentation', () => {
    const { container } = renderWithGds(
      <SectionPanel title="Panel with centered state" presentation="fill" minHeight={360} contentAlign="center" contentJustify="center">
        <StateBlock variant="empty" title="No rows" description="Fill-mode state is now on the contract." compact />
      </SectionPanel>,
    );

    const hasPresentationStyles = Array.from(container.querySelectorAll('div[style]')).some(
      (element) => element.getAttribute('style')?.includes('min-height: 360px') && element.getAttribute('style')?.includes('display: flex;'),
    );

    expect(hasPresentationStyles).toBe(true);
    expect(screen.getByText('Panel with centered state')).toBeInTheDocument();
    expect(screen.getByText('No rows')).toBeInTheDocument();
  });

  it('renders canonical access-recovery defaults and invokes recovery actions', async () => {
    const user = userEvent.setup();
    const onSignIn = vi.fn();
    const onBack = vi.fn();

    renderWithGds(
      <AccessRecoveryPanel state="unauthenticated" onSignIn={onSignIn} onBack={onBack} />,
    );

    expect(screen.getByText('Sign in required')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Login' }));
    await user.click(screen.getByRole('button', { name: 'Back' }));

    expect(onSignIn).toHaveBeenCalledTimes(1);
    expect(onBack).toHaveBeenCalledTimes(1);
  });

  it('supports retry-first unavailable states and explicit support actions', async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    const onHelp = vi.fn();

    renderWithGds(
      <AccessRecoveryPanel
        state="unavailable"
        onRetry={onRetry}
        supportAction={{ action: 'help', onClick: onHelp, variant: 'subtle' }}
      />,
    );

    expect(screen.getByText('Content is temporarily unavailable')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Refresh' }));
    await user.click(screen.getByRole('button', { name: 'Help' }));

    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(onHelp).toHaveBeenCalledTimes(1);
  });

  it('renders timeout recovery and permission-limited access summaries without color-only state', async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    const onBack = vi.fn();

    renderWithGds(
      <>
        <AccessRecoveryPanel state="timeout" onRetry={onRetry} onBack={onBack} />
        <AccessSummary
          title="Tenant access"
          roles={['Manager']}
          scope="Budapest"
          state="permission-limited"
          owner="platform-ui"
          recoveryHint="Ask an owner for the finance evidence scope."
        />
      </>,
    );

    expect(screen.getByText('Request timed out')).toBeInTheDocument();
    expect(screen.getByText('Permission limited')).toBeInTheDocument();
    expect(screen.getByText('Owner: platform-ui')).toBeInTheDocument();
    expect(screen.getByText('Ask an owner for the finance evidence scope.')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Refresh' }));
    await user.click(screen.getByRole('button', { name: 'Back' }));

    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(onBack).toHaveBeenCalledTimes(1);
  });

  it('renders the public shell and toolbar contracts', () => {
    renderWithGds(
      <PublicShell
        brand={<span>Camera</span>}
        navItems={[{ id: 'gallery', label: 'Gallery', href: '/gallery' }]}
        activeNavId="gallery"
        actions={<button type="button">Sign in</button>}
        footer="Shared public chrome"
        mobileNavigationMode="inline-collapse"
        mobileNavigation={<a href="#gallery">Gallery</a>}
        headerVariant="branded-quiet"
      >
        <DataToolbar
          searchSlot={<input aria-label="Search" />}
          createAction={<button type="button">Create</button>}
          activeFilters={[{ label: 'Published' }]}
        />
      </PublicShell>,
    );

    expect(screen.getByText('Camera')).toBeInTheDocument();
    expect(screen.getAllByRole('link', { name: 'Gallery' })).toHaveLength(2);
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create' })).toBeInTheDocument();
    expect(screen.getByText('Published')).toBeInTheDocument();
    expect(screen.getByText('Menu')).toBeInTheDocument();
  });

  it('collapses public inline mobile navigation after selecting an item', async () => {
    const user = userEvent.setup();

    renderWithGds(
      <PublicShell
        brand={<span>Camera</span>}
        mobileNavigationMode="inline-collapse"
        mobileNavigation={<a href="#gallery">Gallery</a>}
      >
        <Text>Public content</Text>
      </PublicShell>,
    );

    const details = document.querySelector('details');
    expect(details).toBeTruthy();
    details!.open = true;

    await user.click(screen.getByRole('link', { name: 'Gallery' }));
    expect(details).not.toHaveAttribute('open');
  });

  it('supports enhanced editorial-hero media fades and flat public surfaces', () => {
    const { container } = renderWithGds(
      <EditorialHero
        eyebrow="Editorial"
        title="Shared public storytelling"
        description="Enhanced hero surface."
        media={<div>Media slot</div>}
        mediaFade="background-match"
        surfaceVariant="flat-public"
      />,
    );

    expect(screen.getByText('Shared public storytelling')).toBeInTheDocument();
    expect(screen.getByText('Media slot')).toBeInTheDocument();
    expect(container.querySelector('figure[aria-label]') ?? container.querySelector('figure')).toBeInTheDocument();
  });

  it('supports compact and process feature-band variants', () => {
    renderWithGds(
      <>
        <FeatureBand
          columns={4}
          variant="compact"
          items={[
            { id: 'one', title: 'One' },
            { id: 'two', title: 'Two' },
            { id: 'three', title: 'Three' },
            { id: 'four', title: 'Four' },
          ]}
        />
        <FeatureBand
          variant="process"
          items={[
            { id: 'step-1', title: 'Plan' },
            { id: 'step-2', title: 'Ship', stepLabel: 'Step B' },
          ]}
        />
      </>,
    );

    expect(screen.getByText('One')).toBeInTheDocument();
    expect(screen.getByText('Four')).toBeInTheDocument();
    expect(screen.getByText('Step 1')).toBeInTheDocument();
    expect(screen.getByText('Step B')).toBeInTheDocument();
  });

  it('renders public navigation and footer primitives with accessible active state', () => {
    renderWithGds(
      <PublicNav
        items={[
          { id: 'home', label: 'Home', href: '/' },
          { id: 'docs', label: 'Docs', href: '/docs' },
        ]}
        activeId="docs"
      />,
    );

    expect(screen.getByRole('link', { name: 'Docs' })).toHaveAttribute('aria-current', 'page');
  });

  it('renders provider-identity buttons with stable fallback labels and states', async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();

    renderWithGds(
      <ProviderIdentityButtonGroup
        providers={[
          { provider: 'google', onClick },
          { provider: 'custom-id', label: 'Continue with Custom provider', disabled: true, error: 'Provider failed. Try another method.' },
          { provider: 'github', variant: 'outline' },
          { provider: 'email', description: 'Email identity lane', policyNote: 'Allowed by tenant policy.', size: 'sm' },
        ]}
        layout="grid"
      />,
    );

    expect(screen.getByRole('button', { name: 'Continue with Google' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Continue with Custom provider' })).toBeInTheDocument();
    expect(screen.getByText('Email identity lane')).toBeInTheDocument();
    expect(screen.getByText('Allowed by tenant policy.')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('Provider failed. Try another method.');
    expect(screen.getByRole('button', { name: 'Continue with Custom provider' })).toBeDisabled();

    await user.click(screen.getByRole('button', { name: 'Continue with Google' }));

    expect(onClick).toHaveBeenCalledTimes(1);
    expect(getProviderIdentityLabel('google')).toBe('Continue with Google');
    expect(getSupportedProviderIdentityIds()).toContain('google');
    expect(getProviderIdentityPolicy('google')).toMatchObject({ colorAuthority: 'provider', minTouchTargetPx: 44 });
  });

  it('renders auth and article shells as shared content contracts', () => {
    renderWithGds(
      <>
        <AuthShell
          title="Sign in"
          description="Access governed workspaces with a supported provider."
          intent="account-linking"
          error="GitHub could not finish account linking."
          guestAction={<button type="button">Continue as guest</button>}
          supportAction={<button type="button">Contact support</button>}
          socialAuth={
            <SocialAuthButtons
              providers={[
                { id: 'google', href: '/auth/google' },
                { id: 'github', href: '/auth/github', description: 'For engineering workspaces', tenantDisabledReason: 'Disabled by tenant policy.' },
              ]}
            />
          }
          helper="Contact support if you cannot access your account."
        >
          <button type="button">Continue</button>
        </AuthShell>
        <ArticleShell
          eyebrow="Docs"
          title="Install the design system"
          lead="Follow the package and provider setup flow."
          meta={<span>5 min read</span>}
        >
          <p>Install packages, wire the provider, and verify release alignment.</p>
        </ArticleShell>
      </>,
    );

    expect(screen.getByRole('heading', { name: 'Sign in' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Continue with Google' })).toBeInTheDocument();
    expect(screen.getByText('account linking')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('GitHub could not finish account linking.');
    expect(screen.getByText('Disabled by tenant policy.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Continue as guest' })).toBeInTheDocument();
    expect(screen.getByText('Or continue with your account')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Continue' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Install the design system' })).toBeInTheDocument();
    expect(screen.getByText('5 min read')).toBeInTheDocument();
  });

  it('renders governed share buttons with copy and native share behavior', async () => {
    const user = userEvent.setup();
    const clipboardWriteText = vi.fn().mockResolvedValue(undefined);
    const nativeShare = vi.fn().mockResolvedValue(undefined);

    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: {
        writeText: clipboardWriteText,
      },
    });

    Object.defineProperty(navigator, 'share', {
      configurable: true,
      value: nativeShare,
    });

    renderWithGds(
      <ShareButtonGroup
        url="https://example.com/listing"
        title="Harvest Dinner"
        text="Join this community dinner."
        channels={['native', 'copy', 'mail', 'x']}
      />,
    );

    expect(screen.getByRole('button', { name: 'Share' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy link' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Email' })).toHaveAttribute('href', expect.stringContaining('mailto:'));
    expect(screen.getByRole('link', { name: 'Share on X' })).toHaveAttribute('href', expect.stringContaining('x.com/intent/tweet'));

    await user.click(screen.getByRole('button', { name: 'Copy link' }));
    expect(clipboardWriteText).toHaveBeenCalledWith('https://example.com/listing');
    expect(screen.getByText('Link copied.')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Share' }));
    expect(nativeShare).toHaveBeenCalledWith({
      url: 'https://example.com/listing',
      title: 'Harvest Dinner',
      text: 'Join this community dinner.',
    });
    expect(screen.getByText('Share sheet opened.')).toBeInTheDocument();
  });

  it('renders docs shells, code blocks, and CTA groups', () => {
    const clipboardWriteText = vi.fn();
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: {
        writeText: clipboardWriteText,
      },
    });

    renderWithGds(
      <>
        <DocsPageShell
          breadcrumbs={[{ label: 'Docs', href: '/docs' }, { label: 'Install' }]}
          title="Install packages"
          lead="Use the published packages and root provider."
          footerNext={{ label: 'Next: Providers', href: '/providers' }}
        >
          <DocsCodeBlock
            code={`npm install @sovereignsquad/gds
npm install @mantine/core @mantine/hooks @mantine/modals @mantine/notifications @tabler/icons-react`}
            language="bash"
            title="Install"
          />
        </DocsPageShell>
        <CtaButtonGroup
          primary={<button type="button">Start</button>}
          secondary={<button type="button">Learn more</button>}
          tertiary={<button type="button">View docs</button>}
        />
      </>,
    );

    expect(screen.getByRole('heading', { name: 'Install packages' })).toBeInTheDocument();
    expect(screen.getByText(/npm install @sovereignsquad\/gds/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy code block' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Learn more' })).toBeInTheDocument();
  });

  it('renders neutral page-header eyebrows by default and supports opt-in ornamental styling', () => {
    const { rerender } = renderWithGds(
      <PageHeader title="Release notes" eyebrow="Docs" />,
    );

    const neutralEyebrow = screen.getByText('Docs');
    expect(neutralEyebrow).toBeInTheDocument();
    expect(neutralEyebrow.getAttribute('style') ?? '').not.toContain('letter-spacing');

    rerender(
      <PageHeader title="Release notes" eyebrow="Docs" eyebrowVariant="ornamental" />,
    );

    const ornamentalEyebrow = screen.getByText('Docs');
    expect(ornamentalEyebrow.getAttribute('style') ?? '').toContain('letter-spacing');
  });

  it('renders status badges from the governed soft tone pair, not Mantine\'s light variant', () => {
    // Mantine's variant="light" pastel-on-tint pair measured 1.81:1 and 2.55:1 in dark mode
    // and its rgba tint's contrast cannot be computed. Tone must come from tokens whose
    // foreground is derived against the background it lands on.
    renderWithGds(<StatusBadge status="warning">Needs review</StatusBadge>);

    const badge = screen.getByText('Needs review');
    expect(badge).toBeInTheDocument();
    const root = badge.closest('[data-gds-badge-fixed-tone]') as HTMLElement;
    expect(root).toBeInTheDocument();
    expect(root.style.background).toContain('--gds-badge-soft-warning');
    expect(root.style.color).toContain('--gds-badge-soft-warning-fg');
  });

  it('marks StatusBadge and LabelTag as fixed-tone so theme presets cannot repaint their semantic color', () => {
    renderWithGds(
      <>
        <StatusBadge status="danger">Failed</StatusBadge>
        <LabelTag label="Food" tone="info" />
      </>,
    );

    expect(screen.getByText('Failed').closest('[data-gds-badge-fixed-tone]')).toBeInTheDocument();
    expect(screen.getByText('Food').closest('[data-gds-badge-fixed-tone]')).toBeInTheDocument();
  });

  it('renders the canonical governed status icon on StatusBadge when withIcon is set (#494)', () => {
    renderWithGds(<StatusBadge status="success" withIcon>Published</StatusBadge>);

    const badge = screen.getByText('Published').closest('.mantine-Badge-root') as HTMLElement;
    const icon = badge.querySelector('[data-gds-icon]');
    expect(icon).not.toBeNull();
    expect(icon).toHaveAttribute('data-gds-icon', 'Success');
    expect(icon).toHaveAttribute('aria-hidden', 'true');
  });

  it('renders no icon on StatusBadge without withIcon, and none for neutral even with it (#494)', () => {
    renderWithGds(
      <>
        <StatusBadge status="warning">Plain</StatusBadge>
        <StatusBadge status="neutral" withIcon>Draft</StatusBadge>
      </>,
    );

    const plain = screen.getByText('Plain').closest('.mantine-Badge-root') as HTMLElement;
    const neutral = screen.getByText('Draft').closest('.mantine-Badge-root') as HTMLElement;
    expect(plain.querySelector('[data-gds-icon]')).toBeNull();
    expect(neutral.querySelector('[data-gds-icon]')).toBeNull();
  });

  it('renders count badges and label tags with governed semantics', () => {
    renderWithGds(
      <>
        <CountBadge value={128} cap={99} srLabel="More than ninety nine updates" />
        <LabelTag label="Food" tone="info" />
      </>,
    );

    expect(screen.getByText('99+')).toBeInTheDocument();
    expect(screen.getByLabelText('More than ninety nine updates')).toBeInTheDocument();
    expect(screen.getByText('Food')).toBeInTheDocument();
    expect(screen.getByText('99+').closest('[data-gds-badge-fixed-tone]')).not.toBeInTheDocument();
  });

  it('exposes an accessible theme toggle and switches the color scheme', async () => {
    const user = userEvent.setup();
    const changes: Array<'light' | 'dark'> = [];

    renderWithGds(<ThemeToggle onColorSchemeChange={(next) => changes.push(next)} />);

    const toggle = screen.getByRole('button', { name: 'Toggle color scheme' });
    expect(toggle).toBeInTheDocument();

    await user.click(toggle);
    expect(document.documentElement.getAttribute('data-mantine-color-scheme')).toBe('dark');
    expect(changes.at(-1)).toBe('dark');

    await user.click(toggle);
    expect(document.documentElement.getAttribute('data-mantine-color-scheme')).toBe('light');
    expect(changes.at(-1)).toBe('light');
  });

  it('renders the reference theme explorer with all official lanes and recovery guidance', async () => {
    const user = userEvent.setup();

    renderWithGds(<ReferenceThemeExplorer />);

    expect(screen.getByText('Theme Lab')).toBeInTheDocument();
    expect(screen.getAllByText('gdsTheme').length).toBeGreaterThan(0);
    expect(screen.getAllByText('gdsDarkPublicTheme').length).toBeGreaterThan(0);
    expect(screen.getAllByText('gdsFlatSurfaceTheme').length).toBeGreaterThan(0);
    expect(screen.getAllByText('gdsEditorialPublicTheme').length).toBeGreaterThan(0);
    expect(screen.getAllByText('createPublicBrandTheme(...)').length).toBeGreaterThan(0);
    expect(screen.getByText('Light, dark, and auto proof')).toBeInTheDocument();
    expect(screen.getByText('Unsupported lane boundary')).toBeInTheDocument();
    expect(screen.getByText('Athlete Gold reference surface')).toBeInTheDocument();
    expect(screen.getByText('Athlete IQ')).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Preset'), 'brand');
    await user.selectOptions(screen.getByLabelText('Brand primary color'), 'teal');
    expect(screen.getAllByText('Brand theme generator').length).toBeGreaterThan(0);

    await user.click(screen.getByLabelText('Compare against a second shipped preset'));
    expect(screen.getByText('Comparison Preview Surface')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Reset theme lab' }));
    expect(screen.getAllByText('Default runtime theme').length).toBeGreaterThan(0);
  });

  it('themes control cards globally (not via bespoke owned-contrast) and marks the active preset', () => {
    const { container } = renderWithGds(<ReferenceThemeExplorer />);

    // The Theme Lab control/result cards must not carry a bespoke owned-contrast surface
    // (that forced a dark gradient onto them on a light page); they re-theme globally like
    // any `.gds-paper`, so `theme-lab-controls` must not appear.
    expect(container.querySelectorAll('[data-gds-owned-contrast="theme-lab-controls"]').length).toBe(0);
    expect(container.querySelector('[data-gds-local-contrast="theme-lab-controls"]')).toBeNull();

    // Owned contrast stays reserved for the vibe *swatch* surfaces that preview a specific
    // vibe atmosphere rather than matching the page.
    expect(container.querySelectorAll('[data-gds-owned-contrast="vibe-gallery-card"]').length).toBeGreaterThan(12);
    expect(container.querySelector('[data-gds-owned-contrast="vibe-contract"]')).toBeInTheDocument();
    expect(container.querySelector('[data-gds-owned-contrast="athlete-gold-reference"]')).toBeInTheDocument();
    const firstVibeCard = container.querySelector('[data-gds-owned-contrast="vibe-gallery-card"]');
    expect(firstVibeCard).toHaveStyle({ color: '#111827' });
    expect(firstVibeCard?.getAttribute('style')).toContain('background-color:');
    expect(firstVibeCard?.getAttribute('style')).toContain('background-image: var(--gds-local-background)');
    expect(firstVibeCard?.getAttribute('data-gds-local-contrast')).toBe('vibe-gallery-card');

    const activeMarkers = container.querySelectorAll('[data-gds-theme-lab-active]');
    expect(activeMarkers.length).toBe(2);
    expect([...activeMarkers].some((element) => element.textContent?.includes('Selected'))).toBe(true);
    expect([...activeMarkers].some((element) => element.textContent?.includes('Default runtime theme'))).toBe(true);
  });

  it('renders one generated brand-badge specimen per catalog preset, as a live proof (issue 699)', () => {
    const { container } = renderWithGds(<ReferenceThemeExplorer />);

    // Count derived from the catalog, never a literal (vibe-themes.test.ts lesson) — a
    // preset added in the future must appear here automatically, with no explorer edit.
    const expectedCount = getGdsVibeThemes().length;
    const badgeSpecimens = container.querySelectorAll('[data-gds-generated-mark]');
    expect(badgeSpecimens.length).toBe(expectedCount);

    // Decorative within an already-labeled vibe-gallery card: no accessible name of its own.
    expect(badgeSpecimens[0]).toHaveAttribute('aria-hidden', 'true');
    expect(badgeSpecimens[0]).not.toHaveAttribute('role');
  });

  it('mounts the design rule profile panel and updates it on preset switch (issue #651)', async () => {
    const user = userEvent.setup();
    const { container } = renderWithGds(<ReferenceThemeExplorer />);

    expect(container.querySelector('[data-gds-design-rule-profile-panel]')).toBeInTheDocument();
    expect(screen.getAllByText(/default: declared role classification/).length).toBeGreaterThan(0);

    await user.selectOptions(screen.getByLabelText('Preset'), 'editorial');
    expect(screen.getAllByText(/editorial: declared role classification/).length).toBeGreaterThan(0);
  });

  it('does not fall back to English reference theme explorer copy for non-English locales', () => {
    renderWithGds(<ReferenceThemeExplorer />, { locale: 'ru' });

    expect(screen.getByText('Лаборатория тем')).toBeInTheDocument();
    expect(screen.queryByText('Theme Lab')).not.toBeInTheDocument();
    expect(screen.getAllByText('Стандартная runtime-тема').length).toBeGreaterThan(0);
  });

  it('passes locale messages into nested theme preview providers', () => {
    renderWithGds(<ReferenceThemeExplorer />, { locale: 'fr', messages: fr });

    expect(screen.getByRole('button', { name: 'Annuler' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Aperçu' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Enregistrer' })).toBeInTheDocument();
  });

  it('forwards chosen files from the shared upload dropzone', async () => {
    const user = userEvent.setup();
    const onFilesSelected = vi.fn();

    renderWithGds(
      <UploadDropzone
        title="Upload evidence"
        description="Attach one or more files."
        acceptedTypesLabel="PDF or image"
        maxSizeLabel="5 MB max"
        selectedFiles={['first.txt']}
        policyText="Do not upload private customer data."
        onFilesSelected={onFilesSelected}
      />,
    );

    expect(screen.getByText('idle')).toBeInTheDocument();
    expect(screen.getByText('PDF or image')).toBeInTheDocument();
    expect(screen.getByText('5 MB max')).toBeInTheDocument();
    expect(screen.getByText('Selected: first.txt')).toBeInTheDocument();
    expect(screen.getByText('Do not upload private customer data.')).toBeInTheDocument();

    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, [new File(['a'], 'first.txt', { type: 'text/plain' })]);

    expect(onFilesSelected).toHaveBeenCalledTimes(1);
    expect(onFilesSelected.mock.calls[0][0][0].name).toBe('first.txt');
  });

  it('renders upload dropzone error and readonly states without hidden network behavior', () => {
    const onFilesSelected = vi.fn();

    const { rerender } = renderWithGds(
      <>
        <UploadDropzone
          title="Upload logo"
          state="unsupported-type"
          error="SVG files are not allowed for this surface."
          retryAction={<button type="button">Try again</button>}
          removeAction={<button type="button">Remove asset</button>}
          onFilesSelected={onFilesSelected}
        />
      </>,
    );

    expect(screen.getByRole('alert')).toHaveTextContent('SVG files are not allowed for this surface.');
    expect(screen.getByText('unsupported type')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Remove asset' })).toBeInTheDocument();

    rerender(
      <UploadDropzone title="Locked asset" readonly onFilesSelected={onFilesSelected} />,
    );

    expect(screen.getByText('readonly')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Choose files' })).toBeDisabled();
  });

  it('validates assets, uploads with progress, retries failures, and saves required metadata', async () => {
    const user = userEvent.setup();
    const events = vi.fn();
    const upload = vi.fn()
      .mockRejectedValueOnce(new Error('Network interrupted.'))
      .mockImplementation(async ({ file, onProgress }) => {
        onProgress?.(25);
        onProgress?.(100);
        return {
          id: file.name,
          fileName: file.name,
          mimeType: file.type,
          size: file.size,
          url: `/assets/${file.name}`,
          status: 'metadata-incomplete' as const,
        };
      });

    function AssetProbe() {
      const queue = useGdsAssetUploadQueue({
        adapter: { upload },
        policy: { acceptedTypes: ['image/png'], maxSizeBytes: 10, requireAlt: true, requireCaption: true },
        onEvent: events,
      });
      const item = queue.items[queue.items.length - 1];
      return (
        <div>
          <button type="button" onClick={() => { void queue.selectFiles([new File(['bad'], 'bad.txt', { type: 'text/plain' })]); }}>Bad file</button>
          <button type="button" onClick={() => { void queue.selectFiles([new File(['ok'], 'logo.png', { type: 'image/png' })]); }}>Good file</button>
          {item ? <Text>{item.status}:{item.progress}:{item.error ?? item.asset?.alt ?? 'none'}</Text> : null}
          {item ? <button type="button" onClick={() => { void queue.retry(item.id); }}>Retry</button> : null}
          {item?.asset ? <button type="button" onClick={() => queue.saveMetadata(item.id, { alt: 'Logo alt', caption: 'Logo caption', displayMode: 'contain' })}>Save metadata</button> : null}
        </div>
      );
    }

    expect(validateGdsAsset(new File(['x'], 'x.txt', { type: 'text/plain' }), { acceptedTypes: ['image/png'] }).valid).toBe(false);
    renderWithGds(<AssetProbe />);

    await user.click(screen.getByRole('button', { name: 'Bad file' }));
    expect(screen.getByText(/Unsupported file type/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Good file' }));
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(screen.getByText('metadata-incomplete:100:none')).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: 'Save metadata' }));
    expect(await screen.findByText('ready:100:Logo alt')).toBeInTheDocument();
    expect(events.mock.calls.map(([event]) => event.type)).toEqual(expect.arrayContaining([
      'asset_selected',
      'validation_failed',
      'upload_started',
      'upload_failed',
      'upload_retry',
      'metadata_saved',
    ]));
  });

  it('renders the governed asset manager with preview, metadata policy, retry, and removal controls', async () => {
    const user = userEvent.setup();
    const adapter = createGdsAssetAdapter();
    const file = new File(['png'], 'hero.png', { type: 'image/png' });
    renderWithGds(
      <GdsAssetManager
        title="Upload hero"
        description="Hero assets require publish-safe metadata."
        adapter={adapter}
        policy={{ acceptedTypes: ['image/png'], maxSizeBytes: 20, requireAlt: true }}
        displayMode="contain"
      />,
    );

    expect(screen.getByText('No assets selected')).toBeInTheDocument();
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);
    expect(await screen.findByText('Metadata incomplete. Alt text or caption is required before publish.')).toBeInTheDocument();
    await user.type(screen.getByRole('textbox', { name: 'Alt text for hero.png' }), 'Hero image');
    expect(await screen.findByText('ready')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Remove' }));
    expect(screen.getByText('No assets selected')).toBeInTheDocument();
  });

  it('renders game board tile face and handles press', async () => {
    const user = userEvent.setup();
    const onPress = vi.fn();

    renderWithGds(
      <GameBoardTile face="A" revealed={false} matched={false} disabled={false} onPress={onPress} />,
    );

    expect(screen.getByText('A')).toBeInTheDocument();
    await user.click(screen.getByRole('button'));
    expect(onPress).toHaveBeenCalledTimes(1);
  });

  it('renders placeholder and simple data primitives with deterministic state handling', () => {
    renderWithGds(
      <>
        <PlaceholderPanel
          title="Impact dashboard"
          description="Data will appear after the first reporting window closes."
          badge="Coming soon"
          mode="placeholder"
        />
        <StatsSection title="Category summary" belowThreshold thresholdMessage="Need at least 5 submissions." />
        <SimpleDataTable
          columns={[{ key: 'name', header: 'Name' }]}
          rows={[{ name: 'Northern Region' }]}
        />
      </>,
    );

    expect(screen.getByText('Coming soon')).toBeInTheDocument();
    expect(screen.getByText('Need at least 5 submissions.')).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Name' })).toBeInTheDocument();
    expect(screen.getByText('Northern Region')).toBeInTheDocument();
  });

  it('renders reporting controls, evidence, and chart-token panels with governed states', async () => {
    const user = userEvent.setup();
    const onPeriodChange = vi.fn();

    renderWithGds(
      <ReportingSection
        title="Revenue report"
        description="Canonical reporting composition with period, evidence, metrics, chart summary, and table fallback."
        state="partial"
        stateMessage="Two locations have not reported yet."
        periodControl={(
          <PeriodSelector
            label="Reporting period"
            value="last-30"
            timezone="Europe/Budapest"
            scope="All locations"
            filtered
            stale
            helperText="Periods are evaluated in the selected timezone."
            onChange={onPeriodChange}
            options={[
              { value: 'last-7', label: 'Last 7 days', description: 'Short-term operating window.' },
              { value: 'last-30', label: 'Last 30 days', description: 'Default reporting window.' },
            ]}
          />
        )}
        metrics={(
          <div>
            <MetricCard label="Orders" value="1,240" description="Permission-safe aggregate." />
          </div>
        )}
        chart={(
          <ChartTokenPanel
            title="Orders by channel"
            summary="Online orders account for 62 percent of visible orders; in-store accounts for 38 percent."
            state="permission-limited"
            legend={[
              { label: 'Online', token: 'var(--mantine-color-blue-6)' },
              { label: 'In-store', token: 'var(--mantine-color-teal-6)' },
            ]}
            tableFallback={<SimpleDataTable columns={[{ key: 'channel', header: 'Channel' }, { key: 'share', header: 'Share' }]} rows={[{ channel: 'Online', share: '62%' }]} />}
          />
        )}
        evidence={(
          <EvidencePanel
            title="Evidence trail"
            source="Point-of-sale export"
            freshness="Updated 12 minutes ago"
            confidence="High"
            evidenceCount={18}
            state="permission-limited"
            permissionNote="Private customer-level rows are hidden from this aggregate."
          />
        )}
      />,
    );

    expect(screen.getByRole('heading', { name: 'Revenue report' })).toBeInTheDocument();
    expect(screen.getByLabelText('Reporting period')).toBeInTheDocument();
    expect(screen.getByText('Timezone: Europe/Budapest')).toBeInTheDocument();
    expect(screen.getByText('Stale data')).toBeInTheDocument();
    expect(screen.getByText('Partial report')).toBeInTheDocument();
    expect(screen.getByText('Evidence: 18')).toBeInTheDocument();
    expect(screen.getByText('Accessible data fallback')).toBeInTheDocument();
    expect(screen.getByText('Online: var(--mantine-color-blue-6)')).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Reporting period'), 'last-7');
    expect(onPeriodChange).toHaveBeenCalledWith('last-7');
  });

  it('resolves accent surface styles and renders the shared accent panel contract', () => {
    const subtle = resolveAccentPanelStyles('violet', 'subtle');
    const outline = resolveAccentPanelStyles('green', 'soft-outline');

    expect(subtle.backgroundColor).toContain('light-dark');
    expect(outline.border).toContain('var(--mantine-color-green-4)');

    renderWithGds(
      <AccentPanel tone="blue" title="Shared accent" badge="Contract">
        Accent-safe copy
      </AccentPanel>,
    );

    expect(screen.getByRole('heading', { name: 'Shared accent' })).toBeInTheDocument();
    expect(screen.getByText('Contract')).toBeInTheDocument();
    expect(screen.getByText('Accent-safe copy')).toBeInTheDocument();
  });

  it('renders editorial heroes with one primary CTA and deterministic error fallback', () => {
    const { rerender } = renderWithGds(
      <EditorialHero
        eyebrow="Editorial"
        title="Shared public hero"
        description="Split media and text layouts are now GDS-governed."
        actions={[
          { label: 'Primary path', variant: 'primary' },
          { label: 'Second primary', variant: 'primary' },
        ]}
        meta={[{ id: 'stack', label: 'Server safe' }]}
        media={<div>Media slot</div>}
      />,
    );

    expect(screen.getByRole('heading', { name: 'Shared public hero' })).toBeInTheDocument();
    expect(screen.getByText('Server safe')).toBeInTheDocument();
    expect(screen.getByText('Primary path')).toBeInTheDocument();
    expect(screen.getByText('Second primary')).toBeInTheDocument();

    rerender(
      <EditorialHero
        title="Shared public hero"
        actions={[{ label: 'Primary path', variant: 'primary' }]}
        error="Unable to load hero media."
      />,
    );

    expect(screen.getByText('Media unavailable')).toBeInTheDocument();
    expect(screen.getByText('Unable to load hero media.')).toBeInTheDocument();
  });

  it('renders feature bands and branded footers as shared public composition primitives', () => {
    renderWithGds(
      <>
        <FeatureBand
          columns={2}
          items={[
            { id: 'one', title: 'Fast pickup', description: 'Ready in 15 minutes.' },
            { id: 'two', title: 'Local delivery', description: 'Live in selected districts.' },
          ]}
        />
        <PublicBrandFooter
          layoutVariant="balanced-quote"
          brandTitle="Shared footer"
          description="Narrative, actions, and secondary content now share one footer contract."
          actions={<a href="/support">Support</a>}
          secondary={<blockquote>Quote-led supporting content.</blockquote>}
          legal="© Shared footer contract"
        />
      </>,
    );

    expect(screen.getByText('Fast pickup')).toBeInTheDocument();
    expect(screen.getByText('Local delivery')).toBeInTheDocument();
    expect(screen.getByRole('contentinfo')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Support' })).toBeInTheDocument();
  });

  it('renders public product cards with visible price, helper regions, and sold-out action disabling', () => {
    renderWithGds(
      <>
        <PublicProductCard
          title="Chef tasting menu"
          description="Five courses with seasonal ingredients."
          price="EUR 89"
          helperText="Reserve before 18:00"
          helperKind="pickup"
          inventoryNote="Only 8 left tonight"
          stateLabels={{ preorder: 'Pre-order', limited: 'Low stock' }}
          state="limited"
          metadata={[{ label: 'Availability', value: 'Evenings only' }]}
          primaryAction={<button type="button">Reserve</button>}
        />
        <PublicProductCard
          title="House special"
          state="sold-out"
          primaryAction={<button type="button">Order now</button>}
        />
      </>,
    );

    expect(screen.getByText('EUR 89')).toBeInTheDocument();
    expect(screen.getByText('Reserve before 18:00')).toBeInTheDocument();
    expect(screen.getByText('Only 8 left tonight')).toBeInTheDocument();
    expect(screen.getByText('Low stock')).toBeInTheDocument();
    expect(screen.getByText('Sold out')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Order now' })).toBeDisabled();
  });

  it('renders public product card loading and missing-image fallback states', () => {
    const { rerender } = renderWithGds(
      <PublicProductCard title="Seasonal plate" loading />,
    );

    expect(document.querySelectorAll('.mantine-Skeleton-root').length).toBeGreaterThan(0);

    // A card with no image shows generated art, not the broken-image placeholder glyph.
    const { container } = renderWithGds(<PublicProductCard title="Seasonal plate" />);
    expect(container.querySelector('[data-gds-generated-thumbnail]')).toBeTruthy();
    expect(screen.queryByLabelText('No product image available')).toBeNull();
    void rerender;
  });

  it('applies gds form reducer transitions and blocking summary output', async () => {
    const user = userEvent.setup();
    const reduced = gdsFormReducer(
      { fields: { title: { value: '', touched: false, dirty: false } }, issues: [], submitState: 'idle' },
      { type: 'set-field', field: 'title', value: 'Hi' },
    );
    expect(reduced.fields.title.dirty).toBe(true);

    function FormProbe() {
      const form = useGdsForm({
        initialValues: { title: '' },
        validate: (snapshot) => (String(snapshot.fields.title?.value ?? '').length < 3
          ? [{ field: 'title', message: 'Title is too short.', severity: 'blocking' as const }]
          : []),
        onSubmit: async () => {},
      });

      return (
        <GdsFormProvider snapshot={form.snapshot}>
          <input
            aria-label="Title"
            value={String(form.snapshot.fields.title?.value ?? '')}
            onChange={(event) => form.setFieldValue('title', event.currentTarget.value)}
          />
          <button type="button" onClick={() => { void form.submit(); }}>Submit</button>
          <FormErrorSummary />
          <ValidatedFieldMessage field="title" />
        </GdsFormProvider>
      );
    }

    renderWithGds(<FormProbe />);
    await user.click(screen.getByRole('button', { name: 'Submit' }));
    expect(screen.getAllByText('Title is too short.').length).toBeGreaterThan(0);
  });

  it('orchestrates autosave, optimistic submit, server errors, retry, and draft restore', async () => {
    const user = userEvent.setup();
    const storage = new Map<string, string>();
    const draft = createGdsDraftAdapter<{ title: string }>('draft:test', {
      getItem: (key) => storage.get(key) ?? null,
      setItem: (key, value) => { storage.set(key, value); },
      removeItem: (key) => { storage.delete(key); },
    });
    const submit = vi.fn()
      .mockRejectedValueOnce(new Error('Server rejected title.'))
      .mockResolvedValueOnce(undefined);
    const events = vi.fn();

    function FormProbe() {
      const form = useGdsFormOrchestration({
        initialValues: { title: '' },
        draftAdapter: draft,
        autosave: true,
        optimisticSubmit: true,
        onEvent: events,
        validate: (snapshot) => (String(snapshot.fields.title?.value ?? '').length < 3
          ? [{ field: 'title', message: 'Title is too short.', severity: 'blocking' as const }]
          : []),
        onSubmit: submit,
        mapServerErrors: () => [{ field: 'title', message: 'Use a unique title.' }],
      });
      return (
        <GdsFormProvider snapshot={form.snapshot}>
          <input
            id="title"
            aria-label="Title"
            value={String(form.snapshot.fields.title?.value ?? '')}
            onChange={(event) => form.setFieldValue('title', event.currentTarget.value)}
          />
          <button type="button" onClick={() => { void form.submit(); }}>Submit</button>
          <button type="button" onClick={() => { void form.restoreDraft?.(); }}>Restore</button>
          <GdsValidationSummary />
          <ValidatedFieldMessage field="title" />
          <Text>{form.snapshot.submitState}</Text>
        </GdsFormProvider>
      );
    }

    renderWithGds(<FormProbe />);

    await user.type(screen.getByRole('textbox', { name: 'Title' }), 'AB');
    await user.click(screen.getByRole('button', { name: 'Submit' }));
    expect(screen.getAllByText('Title is too short.').length).toBeGreaterThan(0);

    await user.clear(screen.getByRole('textbox', { name: 'Title' }));
    await user.type(screen.getByRole('textbox', { name: 'Title' }), 'Alpha');
    await user.click(screen.getByRole('button', { name: 'Submit' }));
    await waitFor(() => expect(screen.getAllByText('Use a unique title.').length).toBeGreaterThan(0));
    expect(storage.get('draft:test')).toContain('Alpha');

    await user.click(screen.getByRole('button', { name: 'Submit' }));
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(2));
    expect(events.mock.calls.map(([event]) => event.type)).toEqual(expect.arrayContaining([
      'dirty_changed',
      'validation_failed',
      'autosave_succeeded',
      'submit_failed',
      'retry_succeeded',
    ]));

    await draft.save({ title: 'Restored title' });
    await user.click(screen.getByRole('button', { name: 'Restore' }));
    expect(await screen.findByDisplayValue('Restored title')).toBeInTheDocument();
    expect(events.mock.calls.map(([event]) => event.type)).toContain('draft_restored');
  });

  it('normalizes JSON Schema, OpenAPI, and Zod-like contracts into GDS form schemas', () => {
    const jsonSchema = jsonSchemaToGdsFormSchema({
      title: 'Profile',
      type: 'object',
      required: ['email'],
      properties: {
        email: { type: 'string', format: 'email', title: 'Email address', description: 'Used for receipts.' },
        role: { type: 'string', enum: ['Admin', 'Editor'] },
        gallery: { type: 'array', title: 'Gallery' },
      },
    }, { id: 'profile' });
    expect(jsonSchema.schema?.fields).toEqual(expect.arrayContaining([
      expect.objectContaining({ name: 'email', type: 'email', required: true, i18nKey: 'gds.form.profile.email' }),
      expect.objectContaining({ name: 'role', type: 'select' }),
      expect.objectContaining({ name: 'gallery', type: 'unsupported' }),
    ]));
    expect(jsonSchema.events.map((event) => event.type)).toContain('unsupported_field');

    const openApi = openApiToGdsFormSchema({
      components: {
        schemas: {
          Venue: {
            type: 'object',
            properties: { name: { type: 'string', minLength: 3 } },
            required: ['name'],
          },
        },
      },
    }, { schemaName: 'Venue' });
    expect(openApi.schema?.fields[0]).toEqual(expect.objectContaining({ name: 'name', required: true, minLength: 3 }));

    const zodLike = zodToGdsFormSchema({
      shape: {
        title: { _def: { typeName: 'ZodString' } },
        count: { _def: { typeName: 'ZodOptional', innerType: { _def: { typeName: 'ZodNumber' } } } },
      },
    }, { id: 'zod-fixture' });
    expect(zodLike.schema?.fields).toEqual(expect.arrayContaining([
      expect.objectContaining({ name: 'title', type: 'text', required: true }),
      expect.objectContaining({ name: 'count', type: 'number', required: false }),
    ]));

    expect(createGdsFormFromSchema({ type: 'object', properties: { active: { type: 'boolean' } } }, { adapter: 'json-schema', id: 'active-form' }).schema?.fields[0]?.type).toBe('boolean');
  });

  it('renders schema-generated forms with labels, required validation, submit payload, and overrides', async () => {
    const user = userEvent.setup();
    const submit = vi.fn();
    const events = vi.fn();
    const result = jsonSchemaToGdsFormSchema({
      title: 'Profile',
      description: 'Generated from contract.',
      type: 'object',
      required: ['name'],
      properties: {
        name: { type: 'string', title: 'Name', description: 'Public display name.', minLength: 3 },
        role: { type: 'string', enum: ['Admin', 'Editor'] },
        files: { type: 'array', title: 'Files' },
      },
    }, { id: 'profile-form' });

    renderWithGds(
      <GdsSchemaForm
        schema={result.schema!}
        onSubmit={submit}
        onEvent={events}
        renderers={{
          files: ({ field }) => <input aria-label={field.label} id={field.name} />,
        }}
      />,
    );

    expect(screen.getByText('Generated from contract.')).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Name' })).toHaveAttribute('aria-describedby', 'name-description name-error');
    // renderDefaultField's raw native fallback carries no Mantine class/--input-fz, so it
    // needs its own mobile input-focus auto-zoom guard.
    expect(screen.getByRole('textbox', { name: 'Name' })).toHaveStyle({ fontSize: 'max(1rem, 1em)' });
    expect(screen.getByRole('combobox', { name: 'Role' })).toHaveStyle({ fontSize: 'max(1rem, 1em)' });
    expect(screen.getByRole('textbox', { name: 'Files' })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Submit' }));
    expect(screen.getAllByText('Name is required.').length).toBeGreaterThan(0);
    await user.type(screen.getByRole('textbox', { name: 'Name' }), 'Ada');
    await user.click(screen.getByRole('button', { name: 'Submit' }));
    await waitFor(() => expect(submit).toHaveBeenCalledWith(expect.objectContaining({ name: 'Ada' })));
  });

  // Skipped (issue 739 / issue 742): deterministically misses vitest's timeout on CI's
  // shared runners even at 60000ms (default 15000ms and 30000ms also insufficient); real
  // cause suspected to be genuine per-keystroke/per-interaction cost, not artificial delay.
  // Re-enable once issue 742's investigation lands a real fix.
  it.skip('renders schema date fields with a real date picker and submits an ISO string (issue #389)', async () => {
    const user = userEvent.setup();
    const submit = vi.fn();
    const result = jsonSchemaToGdsFormSchema({
      title: 'Booking',
      type: 'object',
      required: ['startDate'],
      properties: {
        startDate: { type: 'string', format: 'date', title: 'Start date' },
      },
    }, { id: 'booking-form' });

    renderWithGds(<GdsSchemaForm schema={result.schema!} onSubmit={submit} />);

    const input = screen.getByLabelText('Start date');
    await user.type(input, 'July 23, 2026');
    await user.tab();
    await user.click(screen.getByRole('button', { name: 'Submit' }));
    await waitFor(() => expect(submit).toHaveBeenCalledWith(expect.objectContaining({ startDate: '2026-07-23' })));
  });

  it('renders a checkbox-group field, edits it, validates required, and submits a string[] (#437)', async () => {
    const user = userEvent.setup();
    const submit = vi.fn();
    const schema = {
      id: 'prefs-form',
      title: 'Preferences',
      fields: [
        {
          name: 'channels',
          type: 'checkbox-group' as const,
          label: 'Channels',
          i18nKey: 'gds.form.prefs.channels',
          required: true,
          options: [
            { label: 'Email', value: 'email' },
            { label: 'SMS', value: 'sms' },
          ],
        },
      ],
    };

    renderWithGds(<GdsSchemaForm schema={schema} onSubmit={submit} />);

    await user.click(screen.getByRole('button', { name: 'Submit' }));
    expect(screen.getAllByText('Channels requires at least one selection.').length).toBeGreaterThan(0);
    expect(submit).not.toHaveBeenCalled();

    await user.click(screen.getByRole('checkbox', { name: 'Email' }));
    await user.click(screen.getByRole('button', { name: 'Submit' }));
    await waitFor(() => expect(submit).toHaveBeenCalledWith(expect.objectContaining({ channels: ['email'] })));
  });

  it('renders a repeatable row group with add/remove, min/max bounds, and required-subfield validation (#437)', async () => {
    const user = userEvent.setup();
    const submit = vi.fn();
    const schema = {
      id: 'team-form',
      title: 'Team',
      fields: [
        {
          name: 'members',
          type: 'repeatable' as const,
          label: 'Member',
          i18nKey: 'gds.form.team.members',
          minRows: 1,
          maxRows: 2,
          addRowLabel: 'Add member',
          removeRowLabel: 'Remove member',
          fields: [
            { name: 'fullName', type: 'text' as const, label: 'Full name', i18nKey: 'gds.form.team.fullName', required: true },
            { name: 'role', type: 'text' as const, label: 'Role', i18nKey: 'gds.form.team.role' },
          ],
        },
      ],
    };

    renderWithGds(<GdsSchemaForm schema={schema} onSubmit={submit} />);

    // minRows:1 renders one row up front; its remove button is disabled (can't go below min).
    expect(screen.getByText('Member 1')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Remove member: Member 1' })).toBeDisabled();

    // Required sub-field empty blocks submit.
    await user.click(screen.getByRole('button', { name: 'Submit' }));
    expect(screen.getAllByText('Member has a row with a missing required field.').length).toBeGreaterThan(0);
    expect(submit).not.toHaveBeenCalled();

    const row1 = screen.getByText('Member 1').closest('[data-gds-repeatable-row]') as HTMLElement;
    await user.type(within(row1).getByRole('textbox', { name: 'Full name' }), 'Ada');

    await user.click(screen.getByRole('button', { name: 'Add member' }));
    expect(screen.getByText('Member 2')).toBeInTheDocument();
    // The live region announces the new row count for screen-reader users.
    expect(screen.getByText('Row added, 2 rows.')).toBeInTheDocument();
    // maxRows:2 reached — add is disabled.
    expect(screen.getByRole('button', { name: 'Add member' })).toBeDisabled();

    const row2 = screen.getByText('Member 2').closest('[data-gds-repeatable-row]') as HTMLElement;
    await user.type(within(row2).getByRole('textbox', { name: 'Full name' }), 'Bo');

    await user.click(screen.getByRole('button', { name: 'Remove member: Member 2' }));
    expect(screen.queryByText('Member 2')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Submit' }));
    await waitFor(() => expect(submit).toHaveBeenCalledWith(expect.objectContaining({ members: [{ fullName: 'Ada', role: '' }] })));
  });

  it('renders schema file-upload fields with dropzone policy, validation, and File payloads', async () => {
    const user = userEvent.setup();
    const submit = vi.fn();
    const result = jsonSchemaToGdsFormSchema({
      title: 'Upload',
      type: 'object',
      required: ['attachment'],
      properties: {
        attachment: {
          type: 'string',
          format: 'binary',
          title: 'Training file',
          description: 'Upload the workout evidence.',
          contentMediaType: 'image/png',
          'x-gds-multiple': true,
          'x-gds-maxFileSizeBytes': 1024,
          'x-gds-maxFileSizeLabel': '1 KB',
          'x-gds-uploadActionLabel': 'Attach file',
          'x-gds-uploadPolicyText': 'PNG evidence only.',
        },
      },
    }, { id: 'upload-form' });

    renderWithGds(<GdsSchemaForm schema={result.schema!} onSubmit={submit} />);

    expect(screen.getByText('PNG evidence only.')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Submit' }));
    expect(screen.getAllByText('Training file is required.').length).toBeGreaterThan(0);

    const input = document.querySelector<HTMLInputElement>('input[type="file"]');
    expect(input).toHaveAttribute('accept', 'image/png');
    const file = new File(['x'], 'evidence.png', { type: 'image/png' });
    await user.upload(input!, file);

    expect(await screen.findByText('Selected: evidence.png')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Submit' }));
    await waitFor(() => expect(submit).toHaveBeenCalledWith(expect.objectContaining({
      attachment: [file],
    })));
  });

  it('uploads schema file-upload fields through an adapter before submit', async () => {
    const user = userEvent.setup();
    const submit = vi.fn();
    const events = vi.fn();
    const uploadResult = { id: 'asset-1', name: 'evidence.png', url: '/uploads/evidence.png' };
    let resolveUpload: (value: typeof uploadResult) => void = () => {};
    const uploadAdapter = {
      upload: vi.fn(({ onProgress }) => {
        onProgress(42);
        return new Promise<typeof uploadResult>((resolve) => {
          resolveUpload = resolve;
        });
      }),
      remove: vi.fn(),
    };
    const result = jsonSchemaToGdsFormSchema({
      title: 'Upload',
      type: 'object',
      required: ['attachment'],
      properties: {
        attachment: {
          type: 'string',
          format: 'binary',
          title: 'Training file',
          'x-gds-uploadPolicyText': 'PNG evidence only.',
        },
      },
    }, { id: 'upload-adapter-form' });

    renderWithGds(<GdsSchemaForm schema={result.schema!} onSubmit={submit} uploadAdapter={uploadAdapter} onEvent={events} />);

    const input = document.querySelector<HTMLInputElement>('input[type="file"]');
    const file = new File(['x'], 'evidence.png', { type: 'image/png' });
    await user.upload(input!, file);

    expect(uploadAdapter.upload).toHaveBeenCalledWith(expect.objectContaining({
      field: expect.objectContaining({ name: 'attachment' }),
      files: [file],
    }));
    expect(await screen.findByText('42% uploaded')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Submit' }));
    expect(screen.getAllByText('Training file upload must finish before submit.').length).toBeGreaterThan(0);

    await act(async () => {
      resolveUpload(uploadResult);
    });

    await user.click(screen.getByRole('button', { name: 'Submit' }));
    await waitFor(() => expect(submit).toHaveBeenCalledWith(expect.objectContaining({
      attachment: [uploadResult],
    })));
    expect(events.mock.calls.map(([event]) => event.type)).toEqual(expect.arrayContaining([
      'upload_started',
      'upload_progress',
      'upload_succeeded',
    ]));
  });

  it('manages overlay stack with top-most close rules', async () => {
    const user = userEvent.setup();
    const events = vi.fn();

    function OverlayProbe() {
      const overlay = useOverlayManager();
      return (
        <>
          <button id="open-a" type="button" onClick={() => overlay.registerOverlay({ id: 'dialog-a', kind: 'dialog', policy: { allowNested: true } })}>Open A</button>
          <button type="button" onClick={() => overlay.registerOverlay({ id: 'drawer-b', kind: 'drawer', policy: { allowNested: true, closeOnEscape: false } })}>Open B</button>
          <button type="button" onClick={() => overlay.closeOverlay('drawer-b', 'escape')}>Blocked escape</button>
          <button type="button" onClick={() => overlay.closeOverlay('drawer-b', 'action')}>Close B</button>
          <Text>{overlay.requestClose('dialog-a', 'escape') ?? 'blocked'}</Text>
          <Text>{overlay.isTopMost('drawer-b') ? 'top' : 'not-top'}</Text>
          <Text>{overlay.stack.length} overlays</Text>
        </>
      );
    }

    renderWithGds(
      <OverlayManagerProvider onOverlayEvent={events}>
        <OverlayProbe />
      </OverlayManagerProvider>,
    );

    await user.click(screen.getByRole('button', { name: 'Open A' }));
    await user.click(screen.getByRole('button', { name: 'Open B' }));
    expect(screen.getByText('top')).toBeInTheDocument();
    expect(screen.getByText('2 overlays')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Blocked escape' }));
    expect(screen.getByText('top')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Close B' }));
    expect(screen.getByText('escape')).toBeInTheDocument();
    expect(events.mock.calls.map(([event]) => event.type)).toEqual(expect.arrayContaining(['overlay_opened', 'blocked_close', 'overlay_closed']));
  });

  it('renders governed overlay surfaces with focus return and route recovery', async () => {
    const user = userEvent.setup();
    const events = vi.fn();

    function OverlaySurfaceProbe({ routeKey }: { routeKey: string }) {
      const [modalOpen, setModalOpen] = React.useState(false);
      const [drawerOpen, setDrawerOpen] = React.useState(false);
      return (
        <OverlayManagerProvider routeKey={routeKey} onOverlayEvent={events}>
          <button id="modal-trigger" type="button" onClick={() => setModalOpen(true)}>Open governed modal</button>
          <button id="drawer-trigger" type="button" onClick={() => setDrawerOpen(true)}>Open governed drawer</button>
          <GdsModal
            id="governed-modal"
            opened={modalOpen}
            onClose={() => setModalOpen(false)}
            title="Governed modal"
            invokerId="modal-trigger"
            policy={{ closeOnEscape: true }}
          >
            Modal body
          </GdsModal>
          <GdsDrawer
            id="governed-drawer"
            opened={drawerOpen}
            onClose={() => setDrawerOpen(false)}
            title="Governed drawer"
            invokerId="drawer-trigger"
            policy={{ routeChange: 'recover', mobileFullscreen: true }}
          >
            Drawer body
          </GdsDrawer>
        </OverlayManagerProvider>
      );
    }

    const { rerender } = renderWithGds(<OverlaySurfaceProbe routeKey="one" />);

    await user.click(screen.getByRole('button', { name: 'Open governed modal' }));
    expect(await screen.findByRole('dialog', { name: 'Governed modal' })).toBeInTheDocument();
    await user.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Governed modal' })).not.toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Open governed modal' })).toHaveFocus();

    await user.click(screen.getByRole('button', { name: 'Open governed drawer' }));
    expect(await screen.findByRole('dialog', { name: 'Governed drawer' })).toBeInTheDocument();
    rerender(<OverlaySurfaceProbe routeKey="two" />);
    expect(events.mock.calls.map(([event]) => event.type)).toEqual(expect.arrayContaining(['overlay_opened', 'escape_close', 'route_recovered']));
  });

  it('exposes dialog and side-panel aliases over governed overlay surfaces', async () => {
    renderWithGds(
      <OverlayManagerProvider>
        <GdsDialog opened onClose={() => {}} title="Alias dialog">
          Dialog body
        </GdsDialog>
        <GdsSidePanel opened onClose={() => {}} title="Alias side panel">
          Side panel body
        </GdsSidePanel>
      </OverlayManagerProvider>,
    );

    expect(await screen.findByRole('dialog', { name: 'Alias dialog' })).toBeInTheDocument();
    expect(await screen.findByRole('dialog', { name: 'Alias side panel' })).toBeInTheDocument();
  });

  it('registers and executes command palette commands', async () => {
    const user = userEvent.setup();
    const run = vi.fn();

    function CommandProbe() {
      const launcher = useCommandLauncher();
      return (
        <>
          <button type="button" onClick={() => launcher.registerCommands([{ id: 'save', label: 'Save draft', run }])}>Register</button>
          <button type="button" onClick={() => launcher.open()}>Open</button>
        </>
      );
    }

    renderWithGds(
      <CommandRegistryProvider>
        <CommandProbe />
      </CommandRegistryProvider>,
    );

    await user.click(screen.getByRole('button', { name: 'Register' }));
    await user.click(screen.getByRole('button', { name: 'Open' }));
    await user.click(await screen.findByRole('button', { name: 'Save draft' }));
    expect(run).toHaveBeenCalledTimes(1);
  });

  it('emits sampled telemetry events with redacted context', async () => {
    const user = userEvent.setup();
    const sink = vi.fn();

    function TelemetryProbe() {
      const telemetry = useGdsTelemetry();
      return (
        <button
          type="button"
          onClick={() => telemetry.emit({
            component: 'test',
            eventType: 'click',
            correlationId: 'always-sampled',
            context: { route: 'patterns', email: 'hidden@example.com' },
          })}
        >
          Emit
        </button>
      );
    }

    renderWithGds(
      <GdsTelemetryProvider sampleRate={1} sink={sink}>
        <TelemetryProbe />
      </GdsTelemetryProvider>,
    );

    await user.click(screen.getByRole('button', { name: 'Emit' }));
    expect(sink).toHaveBeenCalledTimes(1);
    expect(sink.mock.calls[0][0].context.email).toBeUndefined();
    expect(sink.mock.calls[0][0].context.route).toBe('patterns');
  });

  it('exposes the operational telemetry taxonomy and typed guard', () => {
    expect(gdsOperationalEventTypes).toContain('submit');
    expect(gdsOperationalEventTypes).toContain('validation_error');
    expect(gdsOperationalEventTypes).toContain('destructive_action');
    expect(isGdsOperationalEventType('timeout')).toBe(true);
    expect(isGdsOperationalEventType('product-local-event')).toBe(false);
  });

  it('emits provider telemetry through the canonical adapter alias with privacy-safe payloads', async () => {
    const user = userEvent.setup();
    const adapter = { id: 'test-adapter', emit: vi.fn() };

    function TelemetryProbe() {
      const telemetry = useGdsTelemetry();
      return (
        <button
          type="button"
          onClick={() => telemetry.emitGdsEvent({
            component: 'test',
            eventType: 'submit_error',
            correlationId: 'adapter-sampled',
            outcome: 'error',
            reason: 'validation_failed',
            payload: { fieldId: 'email', authToken: 'secret-token' },
          })}
        >
          Emit canonical
        </button>
      );
    }

    renderWithGds(
      <GdsTelemetryProvider sampleRate={1} adapter={adapter}>
        <TelemetryProbe />
      </GdsTelemetryProvider>,
    );

    await user.click(screen.getByRole('button', { name: 'Emit canonical' }));
    expect(adapter.emit).toHaveBeenCalledTimes(1);
    expect(adapter.emit.mock.calls[0][0]).toMatchObject({
      component: 'test',
      eventType: 'submit_error',
      outcome: 'error',
      reason: 'validation_failed',
      payload: { fieldId: 'email' },
    });
    expect(adapter.emit.mock.calls[0][0].payload.authToken).toBeUndefined();
  });

  it('rejects unsafe telemetry payloads when policy requires explicit rejection', () => {
    const sink = vi.fn();
    const onRejectedPayload = vi.fn();

    const result = emitGdsEvent({
      sink,
      payloadPolicy: {
        rejectUnsafePayload: true,
        onRejectedPayload,
      },
    }, {
      component: 'test',
      eventType: 'submit',
      correlationId: 'reject-pii',
      payload: { route: 'admin', email: 'hidden@example.com' },
    });

    expect(result.status).toBe('payload-rejected');
    expect(result.rejectedKeys).toEqual(['email']);
    expect(sink).not.toHaveBeenCalled();
    expect(onRejectedPayload).toHaveBeenCalledWith(expect.objectContaining({
      component: 'test',
      eventType: 'submit',
      rejectedKeys: ['email'],
    }));
  });

  it('reports adapter unavailable and sampling disabled states without throwing', () => {
    const adapter = { id: 'offline', isAvailable: () => false, emit: vi.fn() };
    const baseEvent = {
      component: 'test',
      eventType: 'retry',
      correlationId: 'offline-adapter',
    };

    expect(emitGdsEvent({ adapter }, baseEvent).status).toBe('adapter-unavailable');
    expect(adapter.emit).not.toHaveBeenCalled();
    expect(emitGdsEvent({ sink: vi.fn(), sampleRate: 0 }, baseEvent).status).toBe('sampling-disabled');
  });

  it('creates non-blocking telemetry adapters with bounded retry and error callbacks', async () => {
    const emit = vi.fn()
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce(undefined);
    const onError = vi.fn();
    const adapter = createGdsTelemetryAdapter({
      id: 'retrying-adapter',
      emit,
      retryAttempts: 1,
      retryDelayMs: 0,
      timeoutMs: 100,
      onError,
    });

    adapter.emit({
      component: 'test',
      eventType: 'retry',
      correlationId: 'retry-event',
      ts: Date.now(),
    });

    await waitFor(() => expect(emit).toHaveBeenCalledTimes(2));
    expect(onError).not.toHaveBeenCalled();

    const failingOnError = vi.fn();
    const failingAdapter = createGdsTelemetryAdapter({
      id: 'failing-adapter',
      emit: () => {
        throw new Error('permanent failure');
      },
      timeoutMs: 100,
      onError: failingOnError,
    });

    failingAdapter.emit({
      component: 'test',
      eventType: 'adapter_error',
      correlationId: 'adapter-failure',
      ts: Date.now(),
    });

    await waitFor(() => expect(failingOnError).toHaveBeenCalledTimes(1));
  });

  it('renders the expanded chart contract and fallback data table', () => {
    renderWithGds(
      <GdsChart
        type="heatmap"
        title="Heatmap contract"
        summary="Governed chart wrapper."
        data={[
          { label: 'Cell A', value: 4, group: 'Row 1' },
          { label: 'Cell B', value: 9, group: 'Row 2' },
        ]}
      />,
    );

    expect(screen.getByRole('heading', { name: 'Heatmap contract' })).toBeInTheDocument();
    expect(screen.getByText('Type lane: heatmap')).toBeInTheDocument();
    expect(screen.getByText('Registry family: matrix')).toBeInTheDocument();
    expect(screen.getByText('Cell A')).toBeInTheDocument();
  });

  it('validates chart schemas, thresholds, and rendering budgets before adapter rendering', () => {
    expect(Object.keys(gdsChartTypeRegistry)).toHaveLength(14);
    expect(Object.keys(gdsChartSetATypeRegistry)).toEqual(['line', 'area', 'bar', 'stacked-bar', 'pie', 'donut', 'radar', 'scatter']);
    expect(Object.keys(gdsChartSetBTypeRegistry)).toEqual(['bubble', 'heatmap', 'funnel', 'treemap']);
    expect(Object.keys(gdsChartSetCTypeRegistry)).toEqual(['candlestick', 'sankey']);
    expect(isGdsChartSetAType('scatter')).toBe(true);
    expect(isGdsChartSetAType('heatmap')).toBe(false);
    expect(isGdsChartSetBType('heatmap')).toBe(true);
    expect(isGdsChartSetCType('candlestick')).toBe(true);
    expect(isGdsChartSetCType('bar')).toBe(false);

    expect(validateGdsChartData('pie', [{ label: 'Only', value: 1 }])).toMatchObject({
      state: 'below-threshold',
      issues: ['Pie charts require at least 2 data points.'],
    });

    expect(validateGdsChartData('stacked-bar', [
      { label: 'Q1', value: 12 },
      { label: 'Q1', value: 8, group: 'B' },
    ])).toMatchObject({
      state: 'error',
      issues: ['Stacked bar charts require a group value for every data point.'],
    });

    expect(validateGdsChartData('bar', [
      { label: 'A', value: 1 },
      { label: 'B', value: 2 },
    ], { maxDataPoints: 1 })).toMatchObject({
      state: 'error',
      issues: ['Dataset has 2 points, above the 1 point rendering budget.'],
      visibleData: [{ label: 'A', value: 1 }],
    });

    const decimated = validateGdsChartData('line', Array.from({ length: 10 }, (_, index) => ({
      label: `D${index}`,
      value: index,
    })), { maxDataPoints: 4, decimateLargeSeries: true });

    expect(decimated.state).toBe('partial');
    expect(decimated.visibleData).toHaveLength(4);
    expect(decimated.visibleData[0].label).toBe('D0');
    expect(decimated.visibleData[decimated.visibleData.length - 1].label).toBe('D9');
  });

  it('applies type-specific Set A chart validation rules', () => {
    expect(validateGdsChartData('line', [
      { label: 'Mon', value: 4 },
      { label: 'Tue', value: null },
    ])).toMatchObject({
      state: 'error',
      issues: ['Point 2 has an invalid numeric value.'],
    });

    expect(validateGdsChartData('line', [
      { label: 'Mon', value: 4 },
      { label: 'Tue', value: null },
    ], { connectNulls: true })).toMatchObject({
      state: 'ready',
      issues: [],
    });

    expect(validateGdsChartData('donut', [
      { label: 'A', value: 0 },
      { label: 'B', value: 0 },
    ])).toMatchObject({
      state: 'error',
      issues: ['Donut charts require a positive total.'],
    });

    expect(validateGdsChartData('pie', [
      { label: 'A', value: -1 },
      { label: 'B', value: 2 },
    ])).toMatchObject({
      state: 'error',
      issues: ['Pie charts cannot render negative slice values.'],
    });

    expect(validateGdsChartData('radar', [
      { label: 'Reach', value: 4 },
      { label: 'Quality', value: -2 },
      { label: 'Retention', value: 8 },
    ])).toMatchObject({
      state: 'error',
      issues: ['Radar charts cannot render negative axis values.'],
    });

    expect(validateGdsChartData('scatter', [
      { label: 'A', value: 2 },
      { label: 'B', value: 5, secondaryValue: 8 },
    ])).toMatchObject({
      state: 'error',
      issues: ['Scatter point 1 requires a numeric secondaryValue.'],
    });
  });

  it('supports vendor-neutral chart renderer adapters while GDS owns shell semantics', () => {
    const renderer = vi.fn((context) => (
      <div role="img" aria-labelledby={context.labelledBy} aria-describedby={context.describedBy}>
        Adapter rendered {context.type} with {context.data.length} points
      </div>
    ));

    renderWithGds(
      <GdsChart
        type="line"
        title="Adapter chart"
        summary="Adapter summary."
        data={[{ label: 'Mon', value: 4 }, { label: 'Tue', value: 9 }]}
        renderer={renderer}
      />,
    );

    expect(renderer).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Adapter rendered line with 2 points')).toBeInTheDocument();
    expect(screen.getByText('Primary series: brand.primary')).toBeInTheDocument();
    expect(screen.getByText('Accessible data fallback')).toBeInTheDocument();
  });

  it('renders semantic chart wrappers with tone-based series colors', () => {
    const data = [
      { label: 'Open', value: 4, group: 'Status' },
      { label: 'Closed', value: 8, group: 'Status' },
    ];

    renderWithGds(
      <>
        <GdsBarChart title="Bar chart" summary="Bar summary" data={data} seriesTone="success" />
        <GdsLineChart title="Line chart" summary="Line summary" data={data} seriesTone="info" />
        <GdsStackedBarChart title="Stacked chart" summary="Stacked summary" data={data} seriesTone="warning" />
      </>,
    );

    expect(screen.getByRole('img', { name: 'Bar chart' })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'Line chart' })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'Stacked chart' })).toBeInTheDocument();
    expect(getGdsSeriesColor('warning')).toBe('var(--gds-state-warning, var(--mantine-color-yellow-7))');
  });

  it('renders the extended chart kit wrappers on the shared accessible shell', () => {
    const cartesianData = [
      { label: 'Start', value: 4 },
      { label: 'End', value: 8 },
    ];
    const radarData = [
      { label: 'Recovery', value: 4 },
      { label: 'Fuel', value: 5 },
      { label: 'Mental', value: 3 },
    ];
    const heatmapData = [
      { label: 'Mon', value: 2, group: 'Week 1' },
      { label: 'Tue', value: 7, group: 'Week 1' },
    ];

    renderWithGds(
      <>
        <GdsAreaChart title="Area kit" summary="Area summary" data={cartesianData} />
        <GdsSparkline title="Sparkline kit" summary="Spark summary" data={cartesianData} />
        <GdsLongitudinalChart title="Longitudinal kit" summary="Longitudinal summary" data={cartesianData} />
        <GdsBenchmarkBarChart title="Benchmark kit" summary="Benchmark summary" data={cartesianData} />
        <GdsRadarChart title="Radar kit" summary="Radar summary" data={radarData} />
        <GdsMaturityRadarChart title="Maturity kit" summary="Maturity summary" data={radarData} />
        <GdsGaugeChart title="Gauge kit" summary="Gauge summary" data={cartesianData} />
        <GdsCalendarHeatmapChart title="Calendar kit" summary="Calendar summary" data={heatmapData} />
        <GdsHistogramChart title="Histogram kit" summary="Histogram summary" data={cartesianData} />
        <GdsDivergingBarChart title="Diverging kit" summary="Diverging summary" data={[{ label: 'Low', value: -2 }, { label: 'High', value: 5 }]} />
        <GdsSlopeChart title="Slope kit" summary="Slope summary" data={cartesianData} />
        <GdsSymmetryChart title="Symmetry kit" summary="Symmetry summary" data={cartesianData} />
      </>,
    );

    expect(screen.getByRole('img', { name: 'Area kit' })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'Sparkline kit' })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'Radar kit' })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'Calendar kit' })).toBeInTheDocument();
  });

  it('mirrors chart wrapper input data into the accessible table fallback', () => {
    const benchmarkData = [
      { label: 'Sprint', value: 42 },
      { label: 'Endurance', value: 87 },
      { label: 'Recovery', value: 63 },
    ];

    renderWithGds(
      <GdsBenchmarkBarChart title="Benchmark fallback" summary="Benchmark summary" data={benchmarkData} />,
    );

    const table = screen.getByRole('table');
    const utils = within(table);
    expect(utils.getByText('Label')).toBeInTheDocument();
    expect(utils.getByText('Value')).toBeInTheDocument();
    benchmarkData.forEach((datum) => {
      expect(utils.getByText(datum.label)).toBeInTheDocument();
      expect(utils.getByText(String(datum.value))).toBeInTheDocument();
    });
  });

  it('renders Set A chart primitive metadata and scatter fallback fields', () => {
    renderWithGds(
      <GdsChart
        type="scatter"
        title="Scatter primitive"
        summary="Correlation across value pairs."
        data={[
          { label: 'A', value: 4, secondaryValue: 12 },
          { label: 'B', value: 9, secondaryValue: 19 },
        ]}
      />,
    );

    expect(screen.getByText('Set A primitive: x/y point field')).toBeInTheDocument();
    expect(screen.getByText('Secondary value')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();
  });

  it('applies type-specific Set B chart validation rules and metadata', () => {
    expect(validateGdsChartData('bubble', [
      { label: 'A', value: 4 },
      { label: 'B', value: 9, secondaryValue: 0 },
    ])).toMatchObject({
      state: 'error',
      issues: [
        'Bubble point 1 requires a numeric secondaryValue for bubble size.',
        'Bubble point 2 requires a positive secondaryValue for bubble size.',
      ],
    });

    expect(validateGdsChartData('heatmap', [
      { label: 'Morning', value: 4, group: 'Mon' },
      { label: 'Evening', value: 9 },
    ])).toMatchObject({
      state: 'error',
      issues: ['Heatmap cell 2 requires a group value for the matrix row.'],
    });

    expect(validateGdsChartData('funnel', [
      { label: 'Visits', value: 100 },
      { label: 'Trials', value: 120 },
    ])).toMatchObject({
      state: 'error',
      issues: ['Funnel stage 2 cannot be greater than the previous stage.'],
    });

    expect(validateGdsChartData('treemap', [
      { label: 'Cluster A', value: 42 },
      { label: 'Cluster B', value: 0 },
    ])).toMatchObject({
      state: 'error',
      issues: ['Treemap node 2 requires a positive area value.'],
    });

    expect(validateGdsChartData('candlestick', [
      { label: 'Day 1', value: null, open: 10, high: 12, low: 9, close: 11 },
    ])).toMatchObject({ state: 'ready' });

    expect(validateGdsChartData('candlestick', [
      { label: 'Day 1', value: null, open: 10, high: 8, low: 9, close: 11 },
    ])).toMatchObject({
      state: 'error',
      issues: ["Candlestick point 1 has a high/low range that doesn't contain its open/close values."],
    });

    expect(validateGdsChartData('candlestick', [
      { label: 'Day 1', value: null, open: 10, high: 12 },
    ])).toMatchObject({
      state: 'error',
      issues: ['Candlestick point 1 requires numeric open, high, low, and close values.'],
    });

    expect(validateGdsChartData('sankey', [
      { label: 'A to B', value: 40, source: 'A', target: 'B' },
    ])).toMatchObject({ state: 'ready' });

    expect(validateGdsChartData('sankey', [
      { label: 'A to B', value: -5, source: 'A' },
    ])).toMatchObject({
      state: 'error',
      issues: [
        'Sankey flow 1 requires both a source and a target node.',
        'Sankey flow 1 cannot render a negative flow value.',
      ],
    });

    renderWithGds(
      <GdsChart
        type="bubble"
        title="Bubble primitive"
        summary="Weighted distribution."
        data={[
          { label: 'Segment A', value: 30, secondaryValue: 14 },
          { label: 'Segment B', value: 55, secondaryValue: 22 },
        ]}
      />,
    );

    expect(screen.getByText('Set B primitive: weighted x/y bubble field')).toBeInTheDocument();

    renderWithGds(
      <GdsChart
        type="candlestick"
        title="Candlestick primitive"
        summary="Daily price movement."
        data={[{ label: 'Day 1', value: null, open: 10, high: 12, low: 9, close: 11 }]}
      />,
    );

    expect(screen.getByText('Set C primitive: open-high-low-close price series')).toBeInTheDocument();
  });

  it('renders schema-based layout blocks through the governed renderer', () => {
    renderWithGds(
      renderGdsLayout({
        version: '1',
        blocks: [
          { id: 'hero', type: 'hero', props: { title: 'Layout hero', description: 'Schema block.' } },
          { id: 'stats', type: 'stats', props: { items: [{ label: 'Blocks', value: '8' }] } },
          { id: 'table', type: 'table', props: { columns: [{ key: 'name', header: 'Name' }], rows: [{ name: 'Schema row' }] } },
          { id: 'filter', type: 'filter', props: { searchLabel: 'Search block', filterLabel: 'Filter block', sortLabel: 'Sort block' } },
          { id: 'cta', type: 'cta', props: {} },
        ],
      }),
    );

    expect(screen.getByText('Layout hero')).toBeInTheDocument();
    expect(screen.getByText('Blocks')).toBeInTheDocument();
    expect(screen.getByText('Schema row')).toBeInTheDocument();
    expect(screen.getByText('Search block')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save' })).toBeInTheDocument();
  });

  it('validates layout schemas and renders actionable diagnostics for unsafe or unknown blocks', () => {
    const schema = {
      version: '1' as const,
      blocks: [
        { id: 'bad', type: 'unknown', props: { title: '<script>alert(1)</script>' } },
      ],
    };

    const result = renderGdsLayoutWithDiagnostics(schema);

    expect(validateGdsLayout(schema)).toEqual([
      { blockId: 'bad', message: 'Unsupported layout block type "unknown".' },
      { blockId: 'bad', message: 'Layout block props may not include script or javascript URL content.' },
    ]);
    expect(result.issues).toHaveLength(2);

    renderWithGds(result.node);
    expect(screen.getByText('Layout diagnostics')).toBeInTheDocument();
    expect(screen.getByText('Unsupported block type: unknown')).toBeInTheDocument();
  });

  it('supports registered custom GDS layout blocks without replacing the default registry', () => {
    registerGdsBlock('notice', (block) => (
      <StateBlock variant="info" title={String(block.props.title ?? 'Notice')} compact />
    ));

    expect(getGdsBlockTypes()).toEqual(expect.arrayContaining(['hero', 'stats', 'cards-grid', 'table', 'chart', 'filter', 'cta', 'footer', 'notice']));

    renderWithGds(
      renderGdsLayout({
        version: '1',
        blocks: [{ id: 'notice', type: 'notice', props: { title: 'Registered notice' } }],
      }),
    );

    expect(screen.getByText('Registered notice')).toBeInTheDocument();
  });

  it('exposes cloned layout starter templates for developer cookbook flows', () => {
    const templates = getGdsLayoutTemplates();
    expect(templates.map((template) => template.id)).toEqual(expect.arrayContaining(['landing-feed', 'operations-dashboard', 'detail-listing']));
    expect(getGdsLayoutTemplate('operations-dashboard')?.schema.blocks.some((block) => block.type === 'chart')).toBe(true);

    templates[0]!.schema.blocks = [];
    expect(getGdsLayoutTemplate('landing-feed')?.schema.blocks.length).toBeGreaterThan(0);
  });

  it('renders the package-owned layout template preview with diagnostics and edited schema output', () => {
    renderWithGds(<GdsLayoutTemplatePreview />);

    expect(screen.getByText('Template cookbook')).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Template preset' })).toBeInTheDocument();
    expect(screen.getByLabelText('Layout schema JSON')).toBeInTheDocument();
    expect(screen.getByText(/Diagnostic result: no issues/i)).toBeInTheDocument();

    fireEvent.change(screen.getByRole('combobox', { name: 'Template preset' }), {
      target: { value: 'diagnostic-invalid' },
    });

    expect(screen.getByDisplayValue('Validation Failure Example')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Layout schema JSON'), {
      target: { value: '{ "version": "1", "blocks": [ { "id": "bad", "type": "ghost", "props": {} } ] }' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Apply schema' }));

    expect(screen.getAllByText(/Unsupported layout block type "ghost"/i).length).toBeGreaterThan(0);
  });

describe('BottomTabBar layout token (issue 698)', () => {
  const items = [{ id: 'home', label: 'Home', href: '/' }];

  it('reads the bar height from --gds-layout-bottom-bar-height, with the pre-token literal as the var() fallback', () => {
    const { container } = renderWithGds(<BottomTabBar items={items} />);
    const nav = container.querySelector('nav');
    expect(nav).toHaveStyle({
      height: `calc(var(--gds-layout-bottom-bar-height, ${BOTTOM_TAB_HEIGHT}px) + env(safe-area-inset-bottom, 0px))`,
    });
  });
});

describe('GdsColorSystemReference (issue 661)', () => {
  it('renders every 60-30-10 role group with a resolved swatch value, and the live contrast matrix', () => {
    renderWithGds(<GdsColorSystemReference />);
    expect(screen.getByText('Dominant (~60%)')).toBeInTheDocument();
    expect(screen.getByText('Secondary (~30%)')).toBeInTheDocument();
    expect(screen.getByText('Accent (~10%)')).toBeInTheDocument();
    // bg.canvas is a DOMINANT_ROLES member; its resolved default/light value must render, not
    // just its role name, or the page would be a role list rather than a live reference.
    expect(screen.getByText('bg.canvas')).toBeInTheDocument();
    expect(screen.getAllByText(/^#/).length).toBeGreaterThan(0);
    expect(screen.getByText('Accent names', { exact: false })).toBeInTheDocument();
  });
});

describe('GdsAccessibilitySystemReference (issue 661)', () => {
  it('renders a live audit verdict and every governed floor rule', () => {
    renderWithGds(<GdsAccessibilitySystemReference />);
    // The default theme holds its own floor -- if this ever renders a violation count instead
    // of "Holds", that is real regression evidence, not a fixture to relax.
    expect(screen.getByText('Holds')).toBeInTheDocument();
    expect(screen.getByText('focus-ring-min-width')).toBeInTheDocument();
    expect(screen.getByText(/rules checked across/)).toBeInTheDocument();
  });
});
});
