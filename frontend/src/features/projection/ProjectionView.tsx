import { useMemo, useState, type JSX } from 'react'

import { Map as MapIcon, X } from 'lucide-react'

import { EmptyState, ErrorNotice } from '@/components/StatusPanel'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { MoveToCollectionMenu } from '@/features/collections/MoveToCollectionMenu'
import { ExportButton } from '@/features/export/ExportButton'
import type { ImageFilter } from '@/features/filters/image-filter'
import { placeholderGalleryItem, type GalleryItem } from '@/features/gallery/gallery-item'
import { HoverCard } from '@/features/projection/HoverCard'
import { useImageSearch } from '@/features/search/useImageSearch'
import { ScatterCanvas } from '@/features/projection/ScatterCanvas'
import { assignSplitColours, readScatterPalette } from '@/features/projection/scatter-palette'
import type { ScreenPoint } from '@/features/projection/scatter-viewport'
import { useDebouncedValue } from '@/lib/useDebouncedValue'
import { useElementSize } from '@/lib/useElementSize'
import { usePointPositions, useProjection } from '@/features/projection/useProjection'
import { ApiError, type SearchTarget } from '@/lib/api-client'
import type { ProjectionPoint } from '@/types/api'

type ProjectionViewProps = {
  filter: ImageFilter
  /** Active search, whose hits are highlighted in place on the map. */
  target: SearchTarget | null
  onSelect: (item: GalleryItem) => void
}

type HoverState = { point: ProjectionPoint; at: ScreenPoint } | null

/**
 * How long the cursor must rest on a point before its image is requested.
 *
 * The card itself appears immediately — the id, the split and the position all
 * come from the projection payload already in memory, so there is nothing to
 * wait for. Only the thumbnail and caption, which need a request, wait.
 *
 * Long enough that crossing a dense region costs nothing, short enough that
 * deliberately pointing at a dot still feels immediate. Sweeping the cloud used
 * to fire one request per point crossed, each cancelling the last.
 */
const HOVER_FETCH_DELAY_MS = 300

/** Render the PCA variance figure as the caveat it is. */
function varianceCaption(method: string, ratio: readonly number[] | null): string {
  if (method !== 'pca' || ratio === null || ratio.length === 0) {
    return 't-SNE preserves local neighbourhoods only — cluster sizes and the distances between clusters carry no meaning.'
  }
  const total = ratio.reduce((sum, value) => sum + value, 0)
  const parts = ratio.map((value) => `${(value * 100).toFixed(1)}%`).join(' + ')
  return `These two components explain ${parts} = ${(total * 100).toFixed(1)}% of the variance in CLIP space. That is low: read this map for broad structure, not for clusters.`
}

/**
 * The embedding map.
 *
 * Answers the question a grid cannot: *where in the corpus does this subset
 * live?* Filtered-out points stay on screen, dimmed, and search hits are
 * highlighted in place, so both narrowing operations are legible as regions
 * rather than as a shorter list.
 *
 * **The query itself is not plotted.** It would be easy to project the text
 * embedding onto the same axes and draw a marker, and it would be wrong: CLIP's
 * modality gap puts text vectors in a different region of the space from image
 * vectors, so the marker would land far outside the cloud and invite a reading
 * of "the query is nowhere near my data" that says nothing about retrieval.
 * Highlighting the images it actually retrieved answers the same question
 * honestly.
 */
export function ProjectionView({ filter, target, onSelect }: ProjectionViewProps): JSX.Element {
  const { data, error, status } = useProjection(filter)
  // Same key as the search view's own call, so this shares that cache entry
  // rather than issuing a second forward pass through CLIP's text encoder.
  const { data: hits } = useImageSearch(target, filter)
  const [plotRef, size] = useElementSize()
  const [hover, setHover] = useState<HoverState>(null)
  const [selected, setSelected] = useState<ReadonlySet<number>>(new Set())

  const points = useMemo(() => data?.points ?? [], [data])
  const positions = usePointPositions(points)

  // Which point the cursor is on now, and which it was still on a moment ago.
  // The fetch is allowed only where those agree — that is what "has rested
  // here" means, and it is why a stale id can never reach the card and show
  // the wrong image under the cursor.
  //
  // Debounced here rather than inside HoverCard because the card unmounts every
  // time the cursor crosses empty space. State inside it would reset on each
  // gap, and a sweep through a sparse region would fire immediately again.
  const hoveredId = hover?.point.id ?? null
  const settledId = useDebouncedValue(hoveredId, HOVER_FETCH_DELAY_MS)
  const hoverHasSettled = hoveredId !== null && settledId === hoveredId

  const highlightedIds = useMemo(
    () => new Set((hits?.results ?? []).map((result) => result.image.id)),
    [hits],
  )

  const legend = useMemo(
    () => [
      ...assignSplitColours(
        points.map((point) => point.split),
        readScatterPalette(),
      ),
    ],
    [points],
  )

  const selectedIds = useMemo(
    () =>
      [...selected]
        .map((index) => points[index]?.id)
        .filter((id): id is string => id !== undefined),
    [selected, points],
  )

  if (status === 'pending') return <Skeleton className="h-[70vh] w-full rounded-lg" />

  if (status === 'error') {
    // A 404 is the documented "you have not run the projection step" case, not
    // a fault: say what to run instead of showing a red error box.
    if (error instanceof ApiError && error.status === 404) {
      return (
        <EmptyState
          icon={<MapIcon className="size-6" />}
          title="No projection has been computed"
          hint="Run `python scripts/project.py` (or `docker compose run --rm setup`) to build the map. The gallery and search work without it."
        />
      )
    }
    return <ErrorNotice title="Could not load the projection" error={error} />
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
        <div className="flex flex-wrap items-center gap-4">
          {legend.map(([split, colour]) => (
            <div key={split} className="flex items-center gap-1.5">
              <span
                // The colour is data, not a style choice a class could express.
                style={{ backgroundColor: colour }}
                className="size-2.5 rounded-full"
                aria-hidden="true"
              />
              <span className="text-xs text-muted-foreground">{split}</span>
            </div>
          ))}
          {target !== null && (
            <span className="text-xs text-muted-foreground">
              <span className="mr-1.5 inline-block size-2.5 rounded-full bg-foreground align-middle" />
              {target.kind === 'image'
                ? `neighbours of ${target.imageId}`
                : `hits for “${target.query}”`}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {selected.size > 0 && (
            <>
              {/*
                Scoped, not bare. A rectangle only ever picks up points matching
                the active filter, so "98 selected" beside a dimmed cloud of
                8 000 would invite the reading that the other 5 281 points under
                the rectangle went somewhere. Naming the denominator says which
                population the count is out of.
              */}
              <span className="text-xs text-muted-foreground">
                {selected.size.toLocaleString()} selected
                {data.match_count < data.count &&
                  ` of the ${data.match_count.toLocaleString()} matching the filter`}
              </span>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => setSelected(new Set())}
              >
                <X aria-hidden="true" />
                Clear
              </Button>
              {/*
                The primary bulk-move surface, and nearly free: the map already
                owns a box selection and already derives `selectedIds` for the
                export button beside it. Re-partitioning a corpus one image at a
                time through the inspector is not a workflow anyone would use;
                lassoing a cluster on the embedding map and moving it is the
                point of having a map at all.
              */}
              <MoveToCollectionMenu
                source={{ kind: 'ids', ids: selectedIds }}
                count={selectedIds.length}
                onMoved={() => setSelected(new Set())}
              />
              <ExportButton scope={{ filter, ids: selectedIds }} />
            </>
          )}
        </div>
      </div>

      <div ref={plotRef} className="relative h-[70vh] w-full">
        <ScatterCanvas
          points={points}
          positions={positions}
          size={size}
          highlightedIds={highlightedIds}
          selectedIndices={selected}
          onSelectionChange={(indices) => setSelected(new Set(indices))}
          onHoverChange={(point, at) =>
            setHover(point === null || at === null ? null : { point, at })
          }
          onActivate={(point) =>
            // The map payload carries no URL and no collection (see
            // ProjectionPoint in the backend's domain model); the inspector
            // fetches the record by id and overwrites all of this. The split is
            // the one truthful thing the point carries, so it is the seed.
            onSelect(
              placeholderGalleryItem(point.id, { split: point.split, collection: point.split }),
            )
          }
        />

        {hover !== null && (
          <HoverCard
            imageId={hover.point.id}
            split={hover.point.split}
            at={hover.at}
            container={size}
            mayFetch={hoverHasSettled}
          />
        )}
      </div>

      <div className="space-y-1 text-xs text-muted-foreground">
        <p>
          {data.count.toLocaleString()} images
          {data.match_count < data.count && (
            <>
              {' '}
              · <span className="text-foreground">{data.match_count.toLocaleString()}</span> match
              the filter, the rest are dimmed
            </>
          )}{' '}
          · scroll to zoom, drag to pan, shift-drag to select, click to inspect
        </p>
        <p>{varianceCaption(data.method, data.explained_variance_ratio)}</p>
      </div>
    </div>
  )
}
