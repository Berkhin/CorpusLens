import { useState, type JSX } from 'react'

import { useIsFetching } from '@tanstack/react-query'
import { LayoutGrid, Map as MapIcon, X } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { ExportButton } from '@/features/export/ExportButton'
import { FilterBar } from '@/features/filters/FilterBar'
import { EMPTY_IMAGE_FILTER, type ImageFilter } from '@/features/filters/image-filter'
import { GalleryView } from '@/features/gallery/GalleryView'
import { placeholderGalleryItem, type GalleryItem } from '@/features/gallery/gallery-item'
import { ImageInspectorDialog } from '@/features/inspector/ImageInspectorDialog'
import { ProjectionView } from '@/features/projection/ProjectionView'
import { SearchBar } from '@/features/search/SearchBar'
import { SearchResultsView } from '@/features/search/SearchResultsView'
import { DatasetStatsBar } from '@/features/stats/DatasetStatsBar'
import { useDatasetStats } from '@/features/stats/useDatasetStats'
import type { SearchTarget } from '@/lib/api-client'
import { queryKeys } from '@/lib/query-keys'
import { cn } from '@/lib/utils'

/** Shared measure, so header, main and footer stay on one column edge. */
const SHELL_CONTAINER = 'mx-auto w-full max-w-[1600px] px-6'

/** Translucent treatment shared by the two sticky bars. */
const SHELL_BAR = 'border-border bg-background/85 backdrop-blur-sm'

/** Which representation of the corpus is on screen. */
type ViewMode = 'grid' | 'map'

type ViewTabsProps = {
  view: ViewMode
  onChange: (view: ViewMode) => void
  mapEnabled: boolean
}

/**
 * Grid / map switch.
 *
 * The map tab is hidden rather than disabled when no projection exists: a
 * control that is present but never usable is a worse explanation than its
 * absence, and the empty state inside the view already names the command that
 * creates one for anyone who reaches it directly.
 */
function ViewTabs({ view, onChange, mapEnabled }: ViewTabsProps): JSX.Element {
  return (
    <div role="tablist" aria-label="Corpus view" className="flex items-center gap-1">
      <Button
        role="tab"
        aria-selected={view === 'grid'}
        variant={view === 'grid' ? 'secondary' : 'ghost'}
        size="sm"
        onClick={() => onChange('grid')}
      >
        <LayoutGrid aria-hidden="true" />
        Grid
      </Button>
      {mapEnabled && (
        <Button
          role="tab"
          aria-selected={view === 'map'}
          variant={view === 'map' ? 'secondary' : 'ghost'}
          size="sm"
          onClick={() => onChange('map')}
        >
          <MapIcon aria-hidden="true" />
          Map
        </Button>
      )}
    </div>
  )
}

/**
 * Application shell.
 *
 * Owns four pieces of client state — the submitted query, the corpus filter,
 * which view is showing, and the selected image. Everything else is server
 * state, and that belongs to TanStack Query rather than to a store
 * (CLAUDE.md §4.3).
 *
 * Query and filter are lifted to here rather than held per view because they
 * apply to both: narrowing to one split, searching, and then switching to the
 * map should carry both constraints across, not silently drop them.
 */
export function App(): JSX.Element {
  const [target, setTarget] = useState<SearchTarget | null>(null)
  const [filter, setFilter] = useState<ImageFilter>(EMPTY_IMAGE_FILTER)
  const [view, setView] = useState<ViewMode>('grid')
  const [selectedItem, setSelectedItem] = useState<GalleryItem | null>(null)

  // Read from the cache rather than lifting the search query out of its view,
  // so the header can show progress without owning the request.
  const isSearching = useIsFetching({ queryKey: queryKeys.search.all }) > 0

  const { data: stats } = useDatasetStats()
  const mapEnabled = stats?.projection_available ?? false

  return (
    <div className="flex min-h-svh flex-col bg-background">
      <header className={cn('sticky top-0 z-40 border-b', SHELL_BAR)}>
        <div className={cn(SHELL_CONTAINER, 'flex flex-col gap-4 py-4')}>
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <div className="flex items-baseline gap-3">
              <h1 className="font-heading text-lg font-semibold">CorpusLens</h1>
              <p className="text-sm text-muted-foreground">
                Browse, filter and semantically search the corpus
              </p>
            </div>
            <ViewTabs view={view} onChange={setView} mapEnabled={mapEnabled} />
          </div>

          <SearchBar
            activeQuery={target?.kind === 'text' ? target.query : ''}
            onSubmit={(query) => setTarget({ kind: 'text', query })}
            onClear={() => setTarget(null)}
            isSearching={isSearching}
          />

          {target?.kind === 'image' && (
            <div className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground">Showing neighbours of</span>
              <span className="font-mono text-xs">{target.imageId}</span>
              <Button type="button" variant="ghost" size="sm" onClick={() => setTarget(null)}>
                <X aria-hidden="true" />
                Clear
              </Button>
            </div>
          )}

          <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
            <FilterBar filter={filter} onChange={setFilter} />
            <ExportButton
              scope={{
                filter,
                // Undefined rather than empty so the request omits the field
                // entirely and the backend takes the filtered-slice path.
                query: target?.kind === 'text' ? target.query : undefined,
                similarToImageId: target?.kind === 'image' ? target.imageId : undefined,
              }}
            />
          </div>
        </div>
      </header>

      <main className={cn(SHELL_CONTAINER, 'flex-1 py-8')}>
        {view === 'map' ? (
          <ProjectionView filter={filter} target={target} onSelect={setSelectedItem} />
        ) : target === null ? (
          <GalleryView filter={filter} onSelect={setSelectedItem} />
        ) : (
          <SearchResultsView target={target} filter={filter} onSelect={setSelectedItem} />
        )}
      </main>

      <footer className={cn('sticky bottom-0 border-t', SHELL_BAR)}>
        <div className={cn(SHELL_CONTAINER, 'py-3')}>
          <DatasetStatsBar />
        </div>
      </footer>

      <ImageInspectorDialog
        item={selectedItem}
        onClose={() => setSelectedItem(null)}
        onFindSimilar={(imageId) => {
          setTarget({ kind: 'image', imageId })
          setSelectedItem(null)
        }}
        // Jumping to a neighbour reuses the same dialog: the inspector fetches
        // by id, so a bare item with the id is all it needs.
        onInspect={(imageId) => setSelectedItem(placeholderGalleryItem(imageId))}
      />
    </div>
  )
}
