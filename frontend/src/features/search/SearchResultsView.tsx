import type { JSX } from 'react'

import { SearchX } from 'lucide-react'

import { EmptyState, ErrorNotice } from '@/components/StatusPanel'
import { useCollectionLabel } from '@/features/collections/useCollectionLabel'
import { isFilterActive, type ImageFilter } from '@/features/filters/image-filter'
import { ImageGrid, ImageGridSkeleton } from '@/features/gallery/ImageGrid'
import { galleryItemFromSearchResult, type GalleryItem } from '@/features/gallery/gallery-item'
import { useImageSearch } from '@/features/search/useImageSearch'
import type { SearchTarget } from '@/lib/api-client'

type SearchResultsViewProps = {
  target: SearchTarget
  filter: ImageFilter
  onSelect: (item: GalleryItem) => void
}

/** Ranked CLIP results, replacing the browse grid while a query is active. */
export function SearchResultsView({
  target,
  filter,
  onSelect,
}: SearchResultsViewProps): JSX.Element {
  const { data, error, status } = useImageSearch(target, filter)
  const collectionLabel = useCollectionLabel()

  if (status === 'pending') return <ImageGridSkeleton count={10} />
  if (status === 'error') return <ErrorNotice title="Search failed" error={error} />

  const filtered = isFilterActive(filter)

  if (data.results.length === 0) {
    return (
      <EmptyState
        icon={<SearchX className="size-6" />}
        title={
          target.kind === 'image'
            ? `Nothing similar to ${target.imageId}`
            : `No results for “${data.query}”`
        }
        hint={
          filtered
            ? 'The filter is applied before ranking, so nothing was left to rank. Try removing a filter.'
            : 'Every image is scored against the query, so an empty result set usually means the index is empty.'
        }
      />
    )
  }

  const items = data.results.map((result) => galleryItemFromSearchResult(result, collectionLabel))

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <p className="text-sm text-muted-foreground">
          Top <span className="font-medium text-foreground">{data.count}</span>{' '}
          {target.kind === 'image' ? 'neighbours of' : 'matches for'}{' '}
          <span className="font-medium text-foreground">
            {target.kind === 'image' ? target.imageId : `“${data.query}”`}
          </span>
          {filtered ? ' within the active filter' : ''}, ranked by cosine similarity
        </p>
        {/* Worth stating in the UI: a researcher reading 0.31 as "31% match"
            would be misreading it. Noted in docs/api.md under "Distance vs. score". */}
        <p className="text-xs text-muted-foreground">
          CLIP&rsquo;s modality gap keeps absolute text&#8594;image similarities low (good matches
          typically score ~0.20&ndash;0.35). Compare ranks, not magnitudes.
        </p>
      </div>

      <ImageGrid items={items} onSelect={onSelect} filter={filter} />
    </div>
  )
}
