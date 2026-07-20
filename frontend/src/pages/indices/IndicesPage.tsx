import { useState } from 'react'
import { BarChartIcon, LayersIcon, SearchIcon, SparklesIcon } from '../../components/icons'
import { PageHeader } from '../../components/PageHeader'
import { Tabs } from '../../components/Tabs'
import { CreateIndexTab } from './CreateIndexTab'
import { ExploreTab } from './ExploreTab'
import { VisualizerTab } from './VisualizerTab'

type Mode = 'create' | 'explore' | 'visualize'

export function IndicesPage() {
  const [mode, setMode] = useState<Mode>('explore')
  return (
    <div>
      <PageHeader icon={LayersIcon} title="Indices">
        Create a new index from scratch, or explore what already exists across all vector stores.
      </PageHeader>

      <Tabs
        tabs={[
          {
            id: 'create',
            label: (
              <span className="inline-flex items-center justify-center gap-1.5 w-full">
                <SparklesIcon size={14} /> Create new index
              </span>
            ),
          },
          {
            id: 'explore',
            label: (
              <span className="inline-flex items-center justify-center gap-1.5 w-full">
                <SearchIcon size={14} /> Explore existing
              </span>
            ),
          },
          {
            id: 'visualize',
            label: (
              <span className="inline-flex items-center justify-center gap-1.5 w-full">
                <BarChartIcon size={14} /> Visualizer
              </span>
            ),
          },
        ]}
        active={mode}
        onChange={setMode}
      />

      {/* All three tabs stay mounted (display-toggled, not unmounted) so switching tabs doesn't
          lose in-progress form state or re-trigger fetches -- mirrors the old console's lazy-build-once behavior. */}
      <div style={{ display: mode === 'create' ? 'block' : 'none' }}>
        <CreateIndexTab />
      </div>
      <div style={{ display: mode === 'explore' ? 'block' : 'none' }}>
        <ExploreTab />
      </div>
      <div style={{ display: mode === 'visualize' ? 'block' : 'none' }}>
        <VisualizerTab />
      </div>
    </div>
  )
}
