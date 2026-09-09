// Public barrel for `@sovereignsquad/gds-core`.
//
// Exports are grouped by role (primitives → surfaces → shells → forms → data →
// reporting → runtime) so the surface is navigable without cross-referencing
// COMPONENTS_AND_PATTERNS.md. Grouping is documentation only — every entry is a
// flat `export *`, so ordering carries no runtime or precedence meaning.
//
// Deliberately NOT re-exported here: `GdsRichTextEditor` is intentionally
// subpath-only (`@sovereignsquad/gds-core/rich-text-editor`) so its heavier
// "Content engine" dependency (Tiptap/ProseMirror) stays out of the default
// bundle. Its absence from this barrel is by design, not an omission — see
// DEPENDENCY_GOVERNANCE.md.

// ── Primitives & style utilities ──
export * from './GdsPrimitives';
export * from './Typography';
export * from './StatusBadge';
export * from './StyleUtilities';
export * from './SafeStyles';
export * from './vocabulary';
export * from './icons';
export * from './pictograms';
export * from './badge-shapes';
export * from './GdsBadgeStack';
export * from './GdsBadge';
export * from './GdsAccentContrastMatrix';
export * from './GdsPinSystemReference';
export * from './GdsCountBadge';
export * from './GdsMapPinBadge';
export * from './category-registry';
export * from './generated-art-engine';
export * from './GdsGeneratedThumbnail';
export * from './GdsGeneratedHero';
export * from './GdsRemovableTag';
export * from './GdsSavedIndicator';
export * from './GdsDensity';
export * from './CardContracts';

// ── Layout & structural surfaces ──
export * from './breakpoints';
export * from './LayoutPrimitives';
export * from './LayoutBlocks';
export * from './AccentPanel';
export * from './SectionPanel';
export * from './BrowseSurface';
export * from './BoundedPreviewSurface';
export * from './GdsViewportFrame';
export * from './OwnedContrastSurface';
export * from './SurfacePresentation';

// ── Buttons, actions & chips ──
export * from './SemanticButton';
export * from './ActionBar';
export * from './CtaButtonGroup';
export * from './ProviderIdentityButtons';
export * from './ShareButtonGroup';
export * from './SocialAuthButtons';
export * from './ChoiceChip';
export * from './NumberStepper';
export * from './ThemeToggle';
export * from './GdsCompareButton';

// ── Feedback, empty & async states ──
export * from './EmptyState';
export * from './StateBlock';
export * from './AsyncSurface';
export * from './PlaceholderPanel';
export * from './ConfirmDialog';
export * from './Notifications';
export * from './Notifications.client';
export * from './GdsNotificationBell';
export * from './FeedbackRuntime.client';
export * from './Telemetry.client';

// ── Cards & content surfaces ──
export * from './MetricCard';
export * from './ProgressCard';
export * from './ConsumerSection';
export * from './ConsumerDashboardGrid';
export * from './GameBoardTile';
export * from './EditorialCard';
export * from './EditorialHero';
export * from './FeatureBand';
export * from './ProductCard';
export * from './PublicProductCard';
export * from './PublicFoodCard';
export * from './FoodMenuSection';
export * from './ListingCard';
export * from './ListingPrimitives';
export * from './ListingState.client';
export * from './BrowseSelection.client';
export * from './DetailProfileShell';
export * from './DetailFactsTable';
export * from './ProviderCTA';
export * from './FitScoreChip';
export * from './MeaningBadge';
export * from './TrustLayer';
export * from './TrustLayer.client';
export * from './AISearchCard';
export * from './HeroSearchPanel.client';
export * from './QuickStartCard';

// ── Shells & navigation ──
export * from './PublicShell';
export * from './DiscoveryShell';
export * from './DocsShell';
export * from './DocsPageShell';
export * from './ArticleShell';
export * from './AuthShell';
export * from './PublicNav';
export * from './BottomTabBar';
export * from './SidebarNav';
export * from './PublicSiteFooter';
export * from './PublicBrandFooter';
export * from './GdsBreadcrumbs';
export * from './PageHeader';
export * from './DocsCodeBlock';

// ── Forms & validation ──
export * from './FormField';
export * from './GdsFormControls';
export * from './GdsDateInput.client';
export * from './GdsForm.client';
export * from './GdsSchemaForm.client';
export * from './SearchableSelect';
export * from './FilterDrawer';
export * from './DataToolbar';

// ── Media & upload ──
export * from './UploadDropzone';
export * from './MediaField';
export * from './MediaCard';
export * from './MediaPreviewCard';
export * from './MediaWithFallback';
export * from './GdsAssetManager.client';
export * from './PlaybackSurface';
export * from './PlaybackControls.client';

// ── Data tables & resource management ──
export * from './SimpleDataTable';
export * from './AdvancedDataTable.client';
export * from './GdsDataTable.client';
export * from './GdsResourceManager.client';

// ── Reporting, charts & evidence ──
export * from './StatsSection';
export * from './PeriodSelector';
export * from './EvidencePanel';
export * from './ChartTokenPanel';
export * from './GdsChart';
export * from './SemanticCharts';
export * from './ReportingSection';

// ── Access & identity ──
export * from './AccessSummary';
export * from './AccessRecoveryPanel';
export * from './GdsAccessGate';
export * from './AccessibilityEvidence';

// ── Overlays, command palette & runtime providers ──
export * from './OverlayManager.client';
export * from './CommandPalette.client';
export * from './GdsTour.client';
export * from './GdsI18nRuntime';

// ── Page templates & task patterns ──
export * from './GdsPageTemplates';
export * from './TaskPatterns';
export * from './MaturityCapabilities';
export * from './CreatorTheme';
export * from './LayoutTemplatePreview.client';

// ── Content-design & handoff tooling ──
export * from './GdsContentDesign';
export * from './GdsDesignHandoff';

// ── Kanban ──
export * from './KanbanBoard.client';

// ── Chat surfaces ──
export * from './ChatSurface';

// ── Public capture & flow shells ──
export * from './PublicFlowShell';
export * from './PublicCaptureFlow';
export * from './MapPanel';
export * from './GdsMapBasemapWash';
export * from './GdsMapFilterRail';
export * from './GdsMapPinPreviewCard';
export * from './GdsGeneratedAvatar';
export * from './GdsGeneratedMark';
export * from './GdsLogoLockup';
export * from './GdsIconBadge';
export * from './GdsMeter';
export * from './GdsRatingDisplay';
export * from './dom-phrase-translation';
export * from './useGdsDomPhraseTranslation.client';
export * from './GdsMotionSystemReference';
export * from './GdsShapeElevationSystemReference';
export * from './GdsDensitySpacingSystemReference';
export * from './GdsIconSystemReference';
export * from './GdsPictogramSystemReference';
export * from './GdsTypographySystemReference';
export * from './GdsColorSystemReference';
export * from './GdsAccessibilitySystemReference';
export * from './GdsButtonFeedbackReference';
export * from './GdsDesignRuleProfilePanel';
export * from './GdsInlineLink';

// ── Partner surfaces ──
export * from './PartnerDiscovery';

// ── Reference-site building blocks ──
// ReferenceThemeExplorer lives behind the `./reference-theme-explorer` subpath
// (see src/reference-theme-explorer.ts) — not re-exported here; issue 532.
export * from './ReferenceLocaleNotice';
export * from './ReferenceLinkGrid';
export * from './ReferenceSection';

// ── i18n message packs ──
export * from './locales';
