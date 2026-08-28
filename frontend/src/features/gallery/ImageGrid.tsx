import { useMemo, useState, type JSX } from 'react'

import { X } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { MoveToCollectionMenu } from '@/features/collections/MoveToCollectionMenu'
import { ExportButton } from '@/features/export/ExportButton'
import type { ImageFilter } from '@/features/filters/image-filter'
import { ImageCard } from '@/features/gallery/ImageCard'
import type { GalleryItem } from '@/features/gallery/gallery-item'

/** Shared column ramp, so the grid and its loading skeleton stay aligned. */
const GRID_CLASSES = 'grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5'

type ImageGridProps = {
  items: GalleryItem[]
  onSelect: (item: GalleryItem) => void
  /** Active filter, so an export of the selection carries the same scope. */
  filter: ImageFilter
}

/**
 * Responsive thumbnail grid with a multi-select toolbar.
 *
 * The selection is local UI state — this component still fetches nothing; the
 * move and export controls own their own mutations. Keeping it here rather than
 * in each view is what gives the browse grid *and* the ranked search results the
 * same affordance from one place: before this, a search result was something you
 * could look at and not act on.
 *
 * The set is stored as ids and intersected with `items` on render, so a change
 * of filter or query drops ids that are no longer on screen without a
 * `useEffect` reacting to a prop it was already handed.
 */
export function ImageGrid({ items, onSelect, filter }: ImageGridProps): JSX.Element {
  const [checked, setChecked] = useState<ReadonlySet<string>>(new Set())

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

      <ul className={GRID_CLASSES}>
        {items.map((item) => (
          <li key={item.id}>
            <ImageCard
              item={item}
              onSelect={onSelect}
              checked={checked.has(item.id)}
              onToggleChecked={toggle}
            />
          </li>
        ))}
      </ul>
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
