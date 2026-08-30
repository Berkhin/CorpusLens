import { useMemo, useState, type JSX } from 'react'

import { useWindowVirtualizer } from '@tanstack/react-virtual'
import { X } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { MoveToCollectionMenu } from '@/features/collections/MoveToCollectionMenu'
import { ExportButton } from '@/features/export/ExportButton'
import type { ImageFilter } from '@/features/filters/image-filter'
import { ImageCard } from '@/features/gallery/ImageCard'
import type { GalleryItem } from '@/features/gallery/gallery-item'
import {
  GRID_GAP,
  chunkIntoRows,
  columnsForWidth,
  rowHeightForWidth,
} from '@/features/gallery/grid-metrics'
import { useElementSize } from '@/lib/useElementSize'

/** Column ramp for the loading skeleton, which is static and needs no windowing. */
const GRID_CLASSES = 'grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5'

/**
 * Rows rendered beyond the viewport on each side.
 *
 * Three rows is roughly a screen of slack at typical card sizes: enough that a
 * fast scroll reaches already-mounted cards whose images have begun loading,
 * without mounting so much that windowing stops paying for itself.
 */
const OVERSCAN_ROWS = 3

type ImageGridProps = {
  items: GalleryItem[]
  onSelect: (item: GalleryItem) => void
  /** Active filter, so an export of the selection carries the same scope. */
  filter: ImageFilter
}

/**
 * Responsive thumbnail grid with a multi-select toolbar, windowed.
 *
 * The selection is local UI state — this component still fetches nothing; the
 * move and export controls own their own mutations. Keeping it here rather than
 * in each view is what gives the browse grid *and* the ranked search results the
 * same affordance from one place: before this, a search result was something you
 * could look at and not act on.
 *
 * The set is stored as ids and intersected with `items` on render, so a change
 * of filter or query drops ids that are no longer on screen without a
 * `useEffect` reacting to a prop it was already handed. **That choice is what
 * made windowing safe to add**: only mounted rows render, so a selection keyed
 * by index would have silently detached as the user scrolled. Ids do not.
 *
 * Rows rather than cards are virtualized, which keeps each row an ordinary CSS
 * grid and leaves alignment to the browser. Row height is derived from the
 * measured container width rather than observed after paint, because the card's
 * aspect ratio is fixed — so nothing reflows underneath the user mid-scroll.
 */
export function ImageGrid({ items, onSelect, filter }: ImageGridProps): JSX.Element {
  const [checked, setChecked] = useState<ReadonlySet<string>>(new Set())
  const [containerRef, { width }] = useElementSize()
  const [node, setNode] = useState<HTMLDivElement | null>(null)

  const columns = columnsForWidth(width)
  const rowHeight = rowHeightForWidth(width, columns)
  const rows = useMemo(() => chunkIntoRows(items, columns), [items, columns])

  // The window virtualizer measures from the top of the document, so it needs
  // this list's distance from it to place row 0 correctly.
  //
  // Read during render rather than kept in state. Everything that can move the
  // grid vertically already re-renders this component — the node attaching, a
  // width change, and the selection toolbar appearing above it (which shifts
  // the grid by its own height and is driven by `checked`). Holding it in state
  // would need an effect to write it, which costs an extra render pass and
  // leaves the first frame misaligned; a layout read here is current by
  // construction.
  const scrollMargin = node?.offsetTop ?? 0

  const virtualizer = useWindowVirtualizer({
    count: rows.length,
    estimateSize: () => rowHeight,
    overscan: OVERSCAN_ROWS,
    scrollMargin,
    // Rows are uniform and their height is known, so identity by index is
    // stable and the virtualizer never needs to re-measure.
    getItemKey: (index) => index,
  })

  const selectedIds = useMemo(
    () => items.filter((item) => checked.has(item.id)).map((item) => item.id),
    [items, checked],
  )

  const toggle = (imageId: string): void => {
    setChecked((current) => {
      const next = new Set(current)
      if (!next.delete(imageId)) next.add(imageId)
      return next
    })
  }

  return (
    <div className="space-y-3">
      {selectedIds.length > 0 && (
        <div className="flex flex-wrap items-center justify-end gap-2">
          <span className="text-xs text-muted-foreground">
            {selectedIds.length.toLocaleString()} selected
          </span>
          <Button type="button" variant="ghost" size="sm" onClick={() => setChecked(new Set())}>
            <X aria-hidden="true" />
            Clear
          </Button>
          <MoveToCollectionMenu
            source={{ kind: 'ids', ids: selectedIds }}
            count={selectedIds.length}
            onMoved={() => setChecked(new Set())}
          />
          <ExportButton scope={{ filter, ids: selectedIds }} />
        </div>
      )}

      <div
        ref={(element: HTMLDivElement | null) => {
          containerRef(element)
          setNode(element)
        }}
      >
        {/*
          A list whose height is the full virtual height, so the page scrollbar
          reflects the whole result set rather than the handful of mounted rows.
          Both dimensions are computed, which is the case CLAUDE.md §5.2 allows
          an inline style for.
        */}
        <ul
          className="relative w-full"
          style={{ height: `${virtualizer.getTotalSize()}px` }}
          aria-label={`${items.length} images`}
        >
          {virtualizer.getVirtualItems().map((virtualRow) => {
            const row = rows[virtualRow.index]
            if (row === undefined) return null
            return (
              <li
                key={virtualRow.key}
                className="absolute top-0 left-0 grid w-full gap-3"
                style={{
                  gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                  // `start` is absolute within the document; subtracting the
                  // list's own offset makes it relative to this container.
                  transform: `translateY(${virtualRow.start - scrollMargin}px)`,
                  height: `${rowHeight - GRID_GAP}px`,
                }}
              >
                {row.map((item) => (
                  <ImageCard
                    key={item.id}
                    item={item}
                    onSelect={onSelect}
                    checked={checked.has(item.id)}
                    onToggleChecked={toggle}
                  />
                ))}
              </li>
            )
          })}
        </ul>
      </div>
    </div>
  )
}

/** Placeholder grid shown while the first page is in flight. */
export function ImageGridSkeleton({ count = 20 }: { count?: number }): JSX.Element {
  return (
    <div className={GRID_CLASSES} aria-hidden="true">
      {Array.from({ length: count }, (_, index) => (
        <Skeleton key={index} className="aspect-4/3 w-full rounded-lg" />
      ))}
    </div>
  )
}
