import type { JSX } from 'react'

import { FilterX, ImageOff, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { EmptyState, ErrorNotice } from '@/components/StatusPanel'
import { useCollectionLabel } from '@/features/collections/useCollectionLabel'
import { isFilterActive, type ImageFilter } from '@/features/filters/image-filter'
import { ImageGrid, ImageGridSkeleton } from '@/features/gallery/ImageGrid'
import { galleryItemFromSummary, type GalleryItem } from '@/features/gallery/gallery-item'
import { useImageList } from '@/features/gallery/useImageList'

type GalleryViewProps = {
  filter: ImageFilter
  onSelect: (item: GalleryItem) => void
}

/** Paginated browse view over the corpus, narrowed by the active filter. */
export function GalleryView({ filter, onSelect }: GalleryViewProps): JSX.Element {
  const { data, error, status, fetchNextPage, hasNextPage, isFetching, isFetchingNextPage } =
    useImageList(filter)
  // Once per view, not once per card: this subscribes to the collections query.
  const collectionLabel = useCollectionLabel()

  if (status === 'pending') return <ImageGridSkeleton />
  if (status === 'error') return <ErrorNotice title="Could not load the dataset" error={error} />

  const items = data.pages.flatMap((page) =>
    page.items.map((summary) => galleryItemFromSummary(summary, collectionLabel)),
  )
  const firstPage = data.pages[0]
  const total = firstPage?.total ?? 0
  const corpusTotal = firstPage?.corpus_total ?? 0
  const filtered = isFilterActive(filter)

  if (items.length === 0) {
    // An empty result under a filter is a normal outcome; an empty result
    // without one means the pipeline never ran. Saying "the index is empty" in
    // the first case would send the user to re-run a 15-minute ingestion for
    // nothing.
    return filtered ? (
      <EmptyState
        icon={<FilterX className="size-6" />}
        title="No images match these filters"
        hint={`None of the ${corpusTotal.toLocaleString()} images in the corpus satisfy every active filter. Try removing one.`}
      />
    ) : (
      <EmptyState
        icon={<ImageOff className="size-6" />}
        title="The index is empty"
        hint="Run scripts/ingest.py to populate data/lancedb before browsing."
      />
    )
  }

  return (
    <div className="space-y-6">
      <p className="text-sm text-muted-foreground">
        Showing <span className="font-medium text-foreground">{items.length}</span> of{' '}
        <span className="font-medium text-foreground">{total.toLocaleString()}</span>
        {filtered ? (
          <>
            {' '}
            matching images
            <span className="text-muted-foreground"> (out of {corpusTotal.toLocaleString()})</span>
          </>
        ) : (
          ' images'
        )}
      </p>

      <ImageGrid items={items} onSelect={onSelect} filter={filter} />

      {hasNextPage && (
        <div className="flex justify-center pt-2">
          <Button
            variant="outline"
            size="lg"
            // An infinite query shares one cache entry across pages, so a second
            // concurrent fetch could clobber the first — hence the isFetching guard.
            onClick={() => void fetchNextPage()}
            disabled={isFetching}
          >
            {isFetchingNextPage && <Loader2 className="animate-spin" />}
            {isFetchingNextPage ? 'Loading…' : 'Load more'}
          </Button>
        </div>
      )}
    </div>
  )
}
