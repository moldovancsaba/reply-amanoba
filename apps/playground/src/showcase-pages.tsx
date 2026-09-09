import { useState } from 'react';
import {
  AccessSummary,
  ActionBar,
  AsyncSurface,
  AuthShell,
  BodyText,
  BoundedPreviewSurface,
  ConsumerDashboardGrid,
  DetailProfileShell,
  DiscoveryShell,
  DocsPageShell,
  FeatureBand,
  FoodMenuSection,
  GdsBox,
  GdsChart,
  GdsCluster,
  GdsGeneratedThumbnail,
  GdsInline,
  GdsInlineLink,
  GdsMapPinBadge,
  GdsStack,
  ListingCard,
  MapPanel,
  MeaningBadge,
  MediaCard,
  MetricCard,
  PlaybackSurface,
  ProgressCard,
  PublicFlowShell,
  PublicFoodCard,
  GdsSegmentedControl,
  ReferenceLinkGrid,
  ReferenceSection,
  SectionPanel,
  SemanticButton,
  ShareButtonGroup,
  SidebarNav,
  SidebarNavItem,
  SidebarNavSection,
  ProviderIdentityButtonGroup,
  StatsSection,
  GdsLayoutTemplatePreview,
  useGdsBrowseSelection,
} from '@sovereignsquad/gds-core';
import type { AsyncSurfaceState } from '@sovereignsquad/gds-core';
import { DataTable, PageHeader, ResponsiveDataView } from '@sovereignsquad/gds-admin';
import { patternRegistry } from './pattern-registry';
import { SiteTourLauncher } from './SiteTourLauncher';

const catalogEntryCount = patternRegistry.length;

function DemoFooter() {
  return (
    <p>
      Need something missing? <GdsInlineLink href="mailto:moldovancsaba+general.design.system@gmail.com">Request a feature</GdsInlineLink>.
    </p>
  );
}

// Reuses the pin-system reference's own fixture identities (label/summary/accent/icon) verbatim
// rather than inventing new copy: same three places, already governed and already translated.
const BROWSE_SELECTION_LISTINGS: Array<{
  id: string;
  title: string;
  description: string;
  accent: 'ocean' | 'teal' | 'grape';
  icon: 'Location' | 'Habit' | 'Message';
}> = [
  { id: 'pool', title: 'Community pool', description: 'Heated pools with beginner lanes and family sessions.', accent: 'ocean', icon: 'Location' },
  { id: 'studio', title: 'Dance studio', description: 'Ballet and street classes with end-of-term showcases.', accent: 'teal', icon: 'Habit' },
  { id: 'hall', title: 'Riverside hall', description: 'Choir and ensemble sessions in a riverside hall.', accent: 'grape', icon: 'Message' },
];

// Labels reuse AsyncSurface's own governed default titles (already localized package-side)
// rather than restating "empty"/"error" as new site copy.
const BROWSE_SURFACE_STATE_OPTIONS: Array<{ value: AsyncSurfaceState; label: string }> = [
  { value: 'loading', label: 'Loading' },
  { value: 'empty', label: 'No results' },
  { value: 'error', label: 'Unable to load' },
  { value: 'success', label: 'Success' },
];

/**
 * Live proof for `useGdsBrowseSelection`: one hook instance shared between a `ListingCard` list
 * (media-left row tiles) and a set of `GdsMapPinBadge` pins, so selecting either half selects
 * both from the same id — no per-consumer sync code. The surface-state switcher exercises the
 * loading/empty/error/success contract `AsyncSurface` already governs (issue 701).
 */
function CardPinSelectionDemo() {
  const browse = useGdsBrowseSelection({ defaultSelectedId: BROWSE_SELECTION_LISTINGS[0].id });
  const [surfaceState, setSurfaceState] = useState<AsyncSurfaceState>('success');

  return (
    <GdsStack gap="md">
      <GdsSegmentedControl
        value={surfaceState}
        onChange={setSurfaceState}
        ariaLabel="Browse surface state"
        options={BROWSE_SURFACE_STATE_OPTIONS}
      />
      <AsyncSurface
        state={surfaceState}
        onRetry={() => setSurfaceState('success')}
        successContent={
          <GdsStack gap="md">
            <GdsStack gap="sm">
              {BROWSE_SELECTION_LISTINGS.map((listing) => (
                <ListingCard
                  key={listing.id}
                  title={listing.title}
                  description={listing.description}
                  variant="media-left"
                  size="sm"
                  density="compact"
                  mediaSeed={listing.id}
                  selected={browse.isSelected(listing.id)}
                  interactiveMode="surface-button"
                  onSurfaceActivate={() => browse.toggle(listing.id)}
                />
              ))}
            </GdsStack>
            <GdsInline gap="md" wrap="wrap">
              {BROWSE_SELECTION_LISTINGS.map((listing) => (
                <button
                  key={listing.id}
                  type="button"
                  aria-label={listing.title}
                  aria-pressed={browse.isSelected(listing.id)}
                  onClick={() => browse.toggle(listing.id)}
                >
                  <GdsMapPinBadge
                    accent={listing.accent}
                    icon={listing.icon}
                    label={listing.title}
                    filled
                    state={browse.isSelected(listing.id) ? 'selected' : 'idle'}
                  />
                </button>
              ))}
            </GdsInline>
          </GdsStack>
        }
      />
    </GdsStack>
  );
}

export function LiveProofsPage() {
  return (
    <DocsPageShell
      title="Live Proofs"
      eyebrow="Official runtime proof"
      lead="This section is the public runtime showcase for shipped GDS surfaces. Use it to inspect real compositions and interaction contracts before building locally."
    >
      <SiteTourLauncher
        tourId="gds-live-proofs"
        autoStart
        steps={[
          { id: 'live-proofs-families', target: 'live-proofs-families', title: 'Open a live proof family', body: 'Proofs are grouped by purpose — discovery cards, shells, actions/auth, food, playback, analytics. Open the lane you are about to build.', placement: 'bottom' },
          { id: 'live-proofs-howto', target: 'live-proofs-howto', title: 'These are shipped contracts', body: 'Every proof renders the real published primitives — not marketing art or local sandboxes — so each one doubles as a migration target.', placement: 'top' },
        ]}
      />
      <div data-gds-tour-target="live-proofs-families">
      <ReferenceSection
        title="Open a live proof family"
        description="The proofs are separated by purpose so visitors can inspect the exact runtime lane they care about."
      >
        <ReferenceLinkGrid
          items={[
            {
              id: 'surfaces',
              title: 'Discovery & Cards',
              description: 'Listing, media, map, menu, and share surfaces for public and discovery-heavy products.',
              href: '/general-design-system/live-proofs/surfaces',
            },
            {
              id: 'layouts',
              title: 'Shells & Layouts',
              description: 'DiscoveryShell, detail shells, and bounded public flows with governed layout rhythm.',
              href: '/general-design-system/live-proofs/layouts',
            },
            {
              id: 'semantics',
              title: 'Actions & Auth',
              description: 'Semantic actions, social auth, share buttons, and governed interaction states.',
              href: '/general-design-system/live-proofs/semantics',
            },
            {
              id: 'food',
              title: 'Food & Menus',
              description: 'Public food and menu contracts with availability, helper states, and grouped item behavior.',
              href: '/general-design-system/live-proofs/food',
            },
            {
              id: 'playback',
              title: 'Playback & Capture',
              description: 'Playback surfaces and controlled capture/review flows without introducing app-local hardware UI.',
              href: '/general-design-system/live-proofs/playback',
            },
            {
              id: 'analytics',
              title: 'Analytics & Data',
              description: 'Metrics, data views, and operational summaries for analytics-oriented workflows.',
              href: '/general-design-system/live-proofs/analytics',
            },
          ]}
        />
      </ReferenceSection>
      </div>

      <div data-gds-tour-target="live-proofs-howto">
      <ReferenceSection
        title="How to read these proofs"
        description="These are live examples built from shipped GDS packages. They are not mock marketing art and they are not local component sandboxes."
      >
        <FeatureBand
          columns={3}
          items={[
            {
              id: 'shipped',
              title: 'Shipped contracts only',
              description: 'The live proof routes should show the actual primitives we publish, not custom website-only replacements.',
            },
            {
              id: 'bounded',
              title: 'Bounded previews',
              description: 'Contained examples are preferred over fake nested websites so the docs stay readable and honest.',
            },
            {
              id: 'migration',
              title: 'Migration target',
              description: 'Each proof is also a migration target for teams currently using local wrappers and bespoke UI.',
            },
          ]}
        />
      </ReferenceSection>
      </div>

      <DemoFooter />
    </DocsPageShell>
  );
}

export function CardsPage() {
  return (
    <DocsPageShell
      title="Discovery & Cards"
      eyebrow="Live proof family"
      lead="Public discovery surfaces should converge on shared cards, menus, map containment, and governed share affordances."
    >
      <ReferenceSection
        title="Unified listing card"
        description="Use one configurable listing-card contract instead of proliferating event, venue, and community cards."
      >
        <ListingCard
          title="Danube Sunset Run"
          description="A public discovery card with featured treatment, governed metadata rows, and clear save/share affordances."
          featured
          pickBadge
          sponsoredDisclosure="Sponsored placement. Selection criteria belong to the host product."
          price="Free"
          metadata={[
            { id: 'date', label: 'Date', value: 'June 14' },
            { id: 'time', label: 'Time', value: '18:30' },
            { id: 'location', label: 'Location', value: 'Margaret Island' },
          ]}
          saveAction={{ action: 'save' }}
          shareAction={{ action: 'refer' }}
          primaryAction={<GdsInlineLink href="/general-design-system/live-proofs/surfaces">Open listing</GdsInlineLink>}
        />
      </ReferenceSection>

      <ReferenceSection
        title="Card and pin selection sync"
        description="One useGdsBrowseSelection instance drives both halves of a browse split view: select a card or its pin and the other follows."
      >
        <CardPinSelectionDemo />
      </ReferenceSection>

      <ReferenceSection title="Food surfaces" description="Food and menu contracts are first-class public surfaces, not downstream product exceptions.">
        <PublicFoodCard
          title="Smoked paprika chicken bowl"
          description="Canonical public food card with availability state, helper copy, and one clear primary action."
          state="preorder"
          price="€12.50"
          helperText="Pickup window closes at 18:00."
          pickupNote="Today, 17:15-18:00"
          freshnessNote="Prepared in small daily batches"
          markers={[
            { id: 'featured', label: 'Featured', tone: 'positive' },
            { id: 'hot', label: 'Limited batch', tone: 'warning' },
          ]}
          primaryAction={<GdsInlineLink href="/general-design-system/live-proofs/surfaces">Reserve pickup</GdsInlineLink>}
        />
        <FoodMenuSection
          title="Weekly menu"
          description="Grouped menu categories with consistent category rhythm and per-item affordances."
          categories={[
            {
              id: 'lunch',
              title: 'Lunch',
              description: 'Fast pickup dishes for midday orders.',
              items: [
                {
                  id: 'dish-1',
                  title: 'Smoked paprika chicken bowl',
                  state: 'available',
                  price: '€12.50',
                  description: 'Roasted vegetables, herbed rice, and citrus yogurt.',
                  primaryAction: <GdsInlineLink href="/general-design-system/live-proofs/surfaces">Add to order</GdsInlineLink>,
                },
                {
                  id: 'dish-2',
                  title: 'Green falafel plate',
                  state: 'limited',
                  price: '€10.90',
                  description: 'Tahini slaw, pickled onions, and flatbread.',
                  primaryAction: <GdsInlineLink href="/general-design-system/live-proofs/surfaces">Add to order</GdsInlineLink>,
                },
              ],
            },
          ]}
        />
      </ReferenceSection>

      <ReferenceSection title="Map and media containment" description="Embeds and media should render inside the sanctioned GDS containment surfaces.">
        <MapPanel
          title="Meetup route map"
          description="MapPanel keeps third-party embeds inside shared header chrome, loading, and failure behavior."
          empty="No coordinates published yet."
        />
        <MediaCard
          title="Public media card"
          description="Media-led discovery surface for stories, bundles, guides, and catalog promotions."
          status="Published"
          image={(
            <GdsGeneratedThumbnail
              seed="media-card-demo"
              categories={[{ key: 'media', label: 'Media', icon: 'Gallery' }]}
            />
          )}
        />
      </ReferenceSection>

      <ReferenceSection title="Governed sharing" description="Sharing should use the canonical share-button group instead of local icon clusters.">
        <ShareButtonGroup
          url="https://sovereignsquad.github.io/general-design-system/live-proofs/surfaces"
          title="General Design System live proofs"
          text="Inspect the shipped discovery and card surfaces."
          channels={['copy', 'mail', 'linkedin', 'whatsapp']}
        />
      </ReferenceSection>

      <DemoFooter />
    </DocsPageShell>
  );
}

export function FoodMenuPage() {
  return (
    <DocsPageShell
      title="Food & Menu"
      eyebrow="Live proof family"
      lead="Food and menu contracts are first-class discovery surfaces. They should follow the same card, helper, and action rules as any other canonical listing."
    >
      <ReferenceSection
        title="Food cards"
        description="Use the shared PublicFoodCard for menu items with clear availability and helper cues."
      >
        <PublicFoodCard
          title="Smoked paprika chicken bowl"
          description="Balanced, protein-forward dish with transparent prep and pickup expectations."
          state="preorder"
          price="€12.50"
          helperText="Pickup is available today after 17:15."
          pickupNote="17:15-18:00"
          freshnessNote="Made fresh in small batches."
          markers={[
            { id: 'featured', label: 'Featured', tone: 'positive' },
            { id: 'hot', label: 'Limited batch', tone: 'warning' },
          ]}
          primaryAction={<GdsInlineLink href="/general-design-system/live-proofs/food">Reserve pickup</GdsInlineLink>}
        />
      </ReferenceSection>
      <ReferenceSection
        title="Category menus"
        description="FoodMenuSection keeps grouped discovery menus stable in spacing, card rhythm, and CTA placement."
      >
        <FoodMenuSection
          title="Weekly menu"
          description="Grouped menu categories with consistent action and disclosure behavior."
          categories={[
            {
              id: 'lunch',
              title: 'Lunch',
              description: 'Fast pickup dishes for midday orders.',
              items: [
                {
                  id: 'dish-1',
                  title: 'Smoked paprika chicken bowl',
                  state: 'available',
                  price: '€12.50',
                  description: 'Roasted vegetables, herbed rice, and citrus yogurt.',
                  primaryAction: <GdsInlineLink href="/general-design-system/live-proofs/food">Add to order</GdsInlineLink>,
                },
                {
                  id: 'dish-2',
                  title: 'Green falafel plate',
                  state: 'limited',
                  price: '€10.90',
                  description: 'Tahini slaw, pickled onions, and flatbread.',
                  primaryAction: <GdsInlineLink href="/general-design-system/live-proofs/food">Add to order</GdsInlineLink>,
                },
              ],
            },
          ]}
        />
      </ReferenceSection>
      <DemoFooter />
    </DocsPageShell>
  );
}

export function LayoutsPage() {
  return (
    <DocsPageShell
      title="Shells & Layouts"
      eyebrow="Live proof family"
      lead="Application shells, detail shells, and staged public flows should converge on shared structure instead of page-local layout contracts."
    >
      <ReferenceSection
        title="Discovery shell"
        description="Sidebar-first applications should use the canonical shell with governed sidebar IA and semantic actions."
      >
        <BoundedPreviewSurface minHeight="34rem">
          <DiscoveryShell
            header={<BodyText>Catalog workspace</BodyText>}
            sidebar={(
              <SidebarNav ariaLabel="Catalog navigation">
                <SidebarNavSection label="Primary">
                  <SidebarNavItem action="dashboard" href="/general-design-system/live-proofs/layouts" active />
                  <SidebarNavItem action="calendar" href="/general-design-system/patterns/foundations" />
                  <SidebarNavItem action="analytics" href="/general-design-system/live-proofs/analytics" />
                </SidebarNavSection>
                <SidebarNavSection label="Account" pushToBottom>
                  <SidebarNavItem action="settings" href="/general-design-system/governance" />
                  <SidebarNavItem action="logout" component="button" />
                </SidebarNavSection>
              </SidebarNav>
            )}
          >
            <PageHeader
              title="Catalog Operations"
              description="Governed sidebar-first shell"
            />
            <SectionPanel
              title="Contained desktop preview"
              description="This is a real DiscoveryShell contract shown inside a bounded preview rather than a fake nested website."
            >
              <ActionBar
                primary={{ action: 'save', size: 'sm' }}
                secondary={[{ action: 'cancel', size: 'sm' }]}
                tertiary={[{ action: 'preview', size: 'sm' }]}
              />
            </SectionPanel>
          </DiscoveryShell>
        </BoundedPreviewSurface>
      </ReferenceSection>

      <ReferenceSection
        title="Detail profile shell"
        description="Drawer and page detail experiences should share one consistent hero, section stack, and related-items rhythm."
      >
        <DetailProfileShell
          hero={<PageHeader title="Universal SSO" description="OAuth/OIDC provider rollout detail surface" />}
          actions={<ActionBar primary={{ action: 'edit', size: 'sm' }} secondary={[{ action: 'refer', size: 'sm' }]} />}
          sections={[
            <SectionPanel key="overview" title="Overview" description="Shared detail-shell content blocks.">
              <p>Use the same detail contract across page and drawer modes instead of growing product-local profile panels.</p>
            </SectionPanel>,
            <AccessSummary
              key="access"
              title="Access summary"
              roles={['platform-ui', 'maintainers']}
              scope="Public adopters"
              description="The detail shell can mix profile sections with access/readiness information without inventing a second layout contract."
            />,
          ]}
        />
      </ReferenceSection>

      <ReferenceSection
        title="Bounded public flows"
        description="Hardware-adjacent or staged public flows should stay inside the sanctioned public-flow shell."
      >
        <PublicFlowShell
          eyebrow="Capture flow"
          stage={{
            id: 'capture-ready',
            title: 'Review your capture setup',
            description: 'The flow contract governs stage status, actions, and bounded hardware surfaces.',
            status: 'ready',
            body: (
              <SectionPanel title="Before you continue" description="This is where a staged flow explains the next irreversible step.">
                <p>Confirm lighting, permissions, and the intended upload destination before opening a hardware-adjacent step.</p>
              </SectionPanel>
            ),
            actions: [
              { action: 'start', priority: 'primary' },
              { action: 'cancel', priority: 'secondary' },
            ],
          }}
        />
      </ReferenceSection>

      <ReferenceSection
        title="Playback surface"
        description="Video and timed-media playback should render through the canonical playback contract."
      >
        <PlaybackSurface
          title="Product walkthrough"
          state="ready"
          statusMessage="Accessible playback surface with bounded description and media containment."
          media={(
            <GdsBox w={320}>
              <GdsGeneratedThumbnail
                seed="playback-surface-demo"
                categories={[{ key: 'playback', label: 'Playback', icon: 'Play' }]}
              />
            </GdsBox>
          )}
        />
      </ReferenceSection>

      <ReferenceSection
        title="Block-based layout schema"
        description="Developers can compose pages from governed blocks using the shared schema renderer."
      >
        <GdsLayoutTemplatePreview />
      </ReferenceSection>

      <DemoFooter />
    </DocsPageShell>
  );
}

export function VocabularyPage() {
  const [submitFeedback, setSubmitFeedback] = useState<'success' | 'error' | null>(null);
  const [deleteFeedback, setDeleteFeedback] = useState<'success' | 'error' | null>(null);
  const [demoAction, setDemoAction] = useState<'save' | 'delete' | 'preview' | 'add'>('save');
  const [demoBrand, setDemoBrand] = useState<'primary' | 'secondary' | 'accent'>('primary');
  const [demoSize, setDemoSize] = useState<'sm' | 'md' | 'lg'>('md');
  const [demoMeaning, setDemoMeaning] = useState<'attention' | 'validation' | 'info' | 'urgency'>('attention');
  const [demoTone, setDemoTone] = useState<'positive' | 'negative' | 'neutral'>('positive');

  // Two independent state slots (not one shared `feedback`): a shared slot let
  // clicking Delete flash an error state on Submit too, since both buttons read
  // the same value (regression, found in a full-site audit — one asymmetric
  // mask on the Delete button hid Submit's own state from it, but nothing hid
  // Delete's state from Submit).
  const showSubmitFeedback = () => {
    setSubmitFeedback('success');
    setTimeout(() => setSubmitFeedback(null), 1600);
  };
  const showDeleteFeedback = () => {
    setDeleteFeedback('error');
    setTimeout(() => setDeleteFeedback(null), 1600);
  };

  return (
    <DocsPageShell
      title="Actions & Auth"
      eyebrow="Live proof family"
      lead="Semantic actions and canonical auth/share surfaces exist so products do not need local wrappers for buttons, login providers, or social distribution."
    >
      <ReferenceSection title="Semantic action system" description="Use semantic actions instead of free-form button stacks wherever the intent is already known.">
        <ActionBar
          primary={{ action: 'save' }}
          secondary={[{ action: 'cancel' }]}
          tertiary={[{ action: 'preview' }, { action: 'refer' }]}
          iconOnly={[{ action: 'settings' }]}
        />
      </ReferenceSection>

      <ReferenceSection title="Feedback states" description="Interaction states remain visible and consistent without route-local button wrappers.">
        <GdsCluster gap="sm">
          <SemanticButton action="submit" feedbackState={submitFeedback} onClick={showSubmitFeedback} />
          <SemanticButton action="delete" feedbackState={deleteFeedback} onClick={showDeleteFeedback} color="red" />
        </GdsCluster>
      </ReferenceSection>

      <ReferenceSection title="Interactive controls" description="Storybook-parity: toggle the props to see the governed SemanticButton update live inside the catalog — no local wrapper, no external tool.">
        <GdsStack gap="sm">
          <GdsSegmentedControl
            ariaLabel="Action"
            value={demoAction}
            onChange={(next) => setDemoAction(next as 'save' | 'delete' | 'preview' | 'add')}
            options={[
              { value: 'save', label: 'Save' },
              { value: 'delete', label: 'Delete' },
              { value: 'preview', label: 'Preview' },
              { value: 'add', label: 'Add' },
            ]}
          />
          <GdsSegmentedControl
            ariaLabel="Brand variant"
            value={demoBrand}
            onChange={(next) => setDemoBrand(next as 'primary' | 'secondary' | 'accent')}
            options={[
              { value: 'primary', label: 'Primary' },
              { value: 'secondary', label: 'Secondary' },
              { value: 'accent', label: 'Accent' },
            ]}
          />
          <GdsSegmentedControl
            ariaLabel="Size"
            value={demoSize}
            onChange={(next) => setDemoSize(next as 'sm' | 'md' | 'lg')}
            options={[
              { value: 'sm', label: 'Small' },
              { value: 'md', label: 'Medium' },
              { value: 'lg', label: 'Large' },
            ]}
          />
          <SemanticButton action={demoAction} brandVariant={demoBrand} size={demoSize} />
          <BodyText>{`<SemanticButton action="${demoAction}" brandVariant="${demoBrand}" size="${demoSize}" />`}</BodyText>
        </GdsStack>
      </ReferenceSection>

      <ReferenceSection title="Interactive controls — MeaningBadge" description="Storybook-parity: switch the semantic variant to see the governed MeaningBadge re-map its background and foreground tokens live.">
        <GdsStack gap="sm">
          <GdsSegmentedControl
            ariaLabel="Meaning variant"
            value={demoMeaning}
            onChange={(next) => setDemoMeaning(next as 'attention' | 'validation' | 'info' | 'urgency')}
            options={[
              { value: 'attention', label: 'Attention' },
              { value: 'validation', label: 'Validation' },
              { value: 'info', label: 'Info' },
              { value: 'urgency', label: 'Urgency' },
            ]}
          />
          <GdsCluster gap="sm">
            <MeaningBadge variant={demoMeaning} label={`${demoMeaning} status`} />
            <MeaningBadge variant={demoMeaning} label={`${demoMeaning} status`} icon="Star" />
          </GdsCluster>
          <BodyText>{`<MeaningBadge variant="${demoMeaning}" label="${demoMeaning} status" icon="Star" />`}</BodyText>
        </GdsStack>
      </ReferenceSection>

      <ReferenceSection title="Interactive controls — MetricCard" description="Storybook-parity: switch the trend tone to see the governed MetricCard re-map its trend badge across the positive, negative, and neutral data lanes.">
        <GdsStack gap="sm">
          <GdsSegmentedControl
            ariaLabel="Trend tone"
            value={demoTone}
            onChange={(next) => setDemoTone(next as 'positive' | 'negative' | 'neutral')}
            options={[
              { value: 'positive', label: 'Positive' },
              { value: 'negative', label: 'Negative' },
              { value: 'neutral', label: 'Neutral' },
            ]}
          />
          <MetricCard label="Monthly active users" value="12,480" trend={{ label: '+8.2% vs last month', tone: demoTone }} />
          <BodyText>{`<MetricCard label="Monthly active users" value="12,480" trend={{ label: '+8.2% vs last month', tone: "${demoTone}" }} />`}</BodyText>
        </GdsStack>
      </ReferenceSection>

      <ReferenceSection title="Canonical social auth" description="Provider-based login belongs to the shared auth surface, not to custom stacks inside each product.">
        <AuthShell
          title="Sign in to GDS"
          description="Canonical social-auth placement inside the shared auth shell."
          socialAuth={(
            <ProviderIdentityButtonGroup
              layout="grid"
              providers={[
                { provider: 'google', href: '/auth/google' },
                { provider: 'apple', href: '/auth/apple' },
                { provider: 'github', href: '/auth/github' },
                { provider: 'microsoft', href: '/auth/microsoft' },
              ]}
            />
          )}
          helper="You can swap in your product session wiring while preserving the shared auth presentation."
        >
          <SectionPanel title="Email lane" description="Products keep their backend auth implementation. GDS governs the surface contract.">
            <p>This bounded helper block replaces the old pattern of every app inventing a different social-login stack.</p>
          </SectionPanel>
        </AuthShell>
      </ReferenceSection>

      <ReferenceSection title="Share buttons" description="Use the canonical share-button group instead of per-product icon clusters.">
        <ShareButtonGroup
          url="https://sovereignsquad.github.io/general-design-system/live-proofs/semantics"
          title="GDS actions and auth live proof"
          text="Inspect semantic actions and canonical social-auth surfaces."
          channels={['native', 'copy', 'mail', 'x']}
          compact
        />
      </ReferenceSection>

      <DemoFooter />
    </DocsPageShell>
  );
}

export function PlaybackPage() {
  return (
    <DocsPageShell
      title="Playback & Capture"
      eyebrow="Live proof family"
      lead="Playback and capture flows should use shared shells and staging semantics rather than product-local hardware scaffolding."
    >
      <ReferenceSection
        title="Playback surfaces"
        description="Use PlaybackSurface for bounded rich media experiences with explicit metadata and stable action affordances."
      >
        <PlaybackSurface
          title="Product walkthrough"
          state="ready"
          statusMessage="Accessible playback surface with bounded media and clear next actions."
          media={(
            <GdsBox w={320}>
              <GdsGeneratedThumbnail
                seed="playback-surfaces-demo"
                categories={[{ key: 'playback', label: 'Playback', icon: 'Play' }]}
              />
            </GdsBox>
          )}
        />
      </ReferenceSection>

      <ReferenceSection
        title="Capture/review stage"
        description="PublicFlowShell keeps capture, review, consent, and submission states predictable."
      >
        <PublicFlowShell
          eyebrow="Capture review"
          stage={{
            id: 'capture-ready',
            title: 'Review your capture setup',
            description: 'The flow contract governs stage status, actions, and blocked/unblocked transitions.',
            status: 'ready',
            body: (
              <SectionPanel
                title="Capture pre-check"
                description="Verify permissions, upload destination, and preview settings before users enter a production step."
              >
                <p>Do not invent local capture UX. Use this contract for every hardware-adjacent staged flow.</p>
              </SectionPanel>
            ),
            actions: [
              { action: 'start', priority: 'primary' },
              { action: 'cancel', priority: 'secondary' },
            ],
          }}
        />
      </ReferenceSection>

      <DemoFooter />
    </DocsPageShell>
  );
}

export function AnalyticsPage() {
  const rows = [
    { id: '1', surface: 'DiscoveryShell', coverage: 'Live proof', status: 'Adopted' },
    { id: '2', surface: 'ListingCard', coverage: 'Live proof', status: 'Adopted' },
    { id: '3', surface: 'MapPanel', coverage: 'Live proof', status: 'Adopted' },
  ];

  return (
    <DocsPageShell
      title="Analytics & Data"
      eyebrow="Live proof family"
      lead="Operational metrics, shared data views, and threshold-aware analytics should use the canonical GDS surfaces rather than local reporting wrappers."
    >
      <ReferenceSection title="Metric and progress surfaces" description="Shared metrics should remain readable, threshold-aware, and consistent across products.">
        <ConsumerDashboardGrid columns={3}>
          <MetricCard label="Catalog coverage" value={`${catalogEntryCount} entries`} description="Pattern inventory shown on the public site." />
          <ProgressCard label="Reference-site conversion" value="Strict consumer" progress={100} progressLabel="Current state" />
          <MetricCard label="npm line" value="3.0.0" description="Public package and docs release line." />
        </ConsumerDashboardGrid>
      </ReferenceSection>

      <ReferenceSection title="Shared data views" description="ResponsiveDataView and DataTable handle desktop/mobile rhythm without inventing local list shells.">
        <ResponsiveDataView
          data={rows}
          columns={[
            { key: 'surface', label: 'Surface' },
            { key: 'coverage', label: 'Coverage' },
            { key: 'status', label: 'Status' },
          ]}
          renderCard={(item) => (
            <SectionPanel title={item.surface} description={item.coverage}>
              <p>{item.status}</p>
            </SectionPanel>
          )}
        />
        <DataTable
          data={rows}
          columns={[
            { key: 'surface', label: 'Surface' },
            { key: 'coverage', label: 'Coverage' },
            { key: 'status', label: 'Status' },
          ]}
          getRowKey={(row) => row.id}
        />
      </ReferenceSection>

      <ReferenceSection title="Statistics thresholds" description="StatsSection communicates loading, threshold, and empty states instead of leaving analytics surfaces vague.">
        <StatsSection
          title="Adoption threshold example"
          belowThreshold
          thresholdMessage="This report remains hidden until the consumer has enough live traffic to produce stable numbers."
        />
      </ReferenceSection>

      <ReferenceSection title="Expanded chart catalog" description="GDS now exposes a governed 12-type chart contract with shared state and fallback behavior.">
        <ConsumerDashboardGrid columns={3}>
          <GdsChart type="line" title="Line chart" summary="Trend over time." data={[{ label: 'Mon', value: 12 }, { label: 'Tue', value: 19 }]} />
          <GdsChart type="area" title="Area chart" summary="Filled trend continuity." data={[{ label: 'A', value: 7 }, { label: 'B', value: 16 }]} />
          <GdsChart type="bar" title="Bar chart" summary="Category totals." data={[{ label: 'North', value: 44 }, { label: 'South', value: 30 }]} />
          <GdsChart type="stacked-bar" title="Stacked bar chart" summary="Grouped category totals." data={[{ label: 'Q1', value: 34, group: 'A' }, { label: 'Q1', value: 21, group: 'B' }]} />
          <GdsChart type="pie" title="Pie chart" summary="Part-to-whole split." data={[{ label: 'Organic', value: 62 }, { label: 'Paid', value: 38 }]} />
          <GdsChart type="donut" title="Donut chart" summary="Part-to-whole with center context." data={[{ label: 'Web', value: 75 }, { label: 'Store', value: 25 }]} />
          <GdsChart type="radar" title="Radar chart" summary="Multi-dimension profile." data={[{ label: 'Reach', value: 80 }, { label: 'Retention', value: 63 }]} />
          <GdsChart type="scatter" title="Scatter chart" summary="Correlation map." data={[{ label: 'Point A', value: 21, secondaryValue: 11 }, { label: 'Point B', value: 40, secondaryValue: 24 }]} />
          <GdsChart type="bubble" title="Bubble chart" summary="Weighted scatter profile." data={[{ label: 'Segment A', value: 30, secondaryValue: 14 }, { label: 'Segment B', value: 55, secondaryValue: 22 }]} />
          <GdsChart type="heatmap" title="Heatmap" summary="Intensity by matrix cell." data={[{ label: 'Morning', value: 9, group: 'Mon' }, { label: 'Evening', value: 4, group: 'Tue' }]} />
          <GdsChart type="funnel" title="Funnel chart" summary="Stage conversion progression." data={[{ label: 'Visits', value: 100 }, { label: 'Leads', value: 25 }]} />
          <GdsChart type="treemap" title="Treemap" summary="Hierarchical distribution." data={[{ label: 'Cluster A', value: 54 }, { label: 'Cluster B', value: 31 }]} />
        </ConsumerDashboardGrid>
      </ReferenceSection>

      <DemoFooter />
    </DocsPageShell>
  );
}
