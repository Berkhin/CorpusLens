/** Paginated browsing over the corpus. */

import {
  useInfiniteQuery,
  type UseInfiniteQueryResult,
  type InfiniteData,
} from '@tanstack/react-query'

import type { ImageFilter } from '@/features/filters/image-filter'
import { fetchImagePage } from '@/lib/api-client'
import { queryKeys } from '@/lib/query-keys'
import type { ImagePage } from '@/types/api'

/** Rows per request. Matches the backend's `DEFAULT_PAGE_SIZE`. */
export const GALLERY_PAGE_SIZE = 50

/**
 * Load the corpus one page at a time, accumulating pages in the cache.
 *
 * `initialPageParam` is required by TanStack Query v5 (it was inferred in v4).
 * The next offset is derived from the rows actually returned rather than from
 * the requested limit, which keeps the cursor correct on a short final page.
 *
 * The filter is part of the cache key, so changing it starts a fresh page
 * sequence from offset 0 instead of appending filtered rows to unfiltered ones.
 * Offsets index the filtered sequence server-side, so the cursor stays valid.
 */
export function useImageList(
  filter: ImageFilter,
  pageSize: number = GALLERY_PAGE_SIZE,
): UseInfiniteQueryResult<InfiniteData<ImagePage, number>, Error> {
  return useInfiniteQuery({
    queryKey: queryKeys.dataset.list(pageSize, filter),
    queryFn: ({ pageParam, signal }) =>
      fetchImagePage({ offset: pageParam, limit: pageSize, filter }, signal),
    initialPageParam: 0,
    getNextPageParam: (lastPage) =>
      lastPage.has_more ? lastPage.offset + lastPage.items.length : undefined,
  })
}

/**
 * How many images the active filter matches.
 *
 * Deliberately read out of {@link useImageList}'s own cache entry rather than
 * asked for separately. The bulk move acts on "everything matching the filter",
 * and a control that promises a number the grid disagrees with is worse than
 * one that promises nothing. Same key, same entry, one number.
 *
 * On the map tab the grid is not mounted, so this does issue the first page
 * request on its own. That is one 50-row page over loopback, and it is what
 * buys the guarantee; a dedicated count endpoint would be a second number to
 * keep in agreement.
 *
 * @returns The matching total, or `undefined` until the first page lands.
 */
export function useFilteredTotal(filter: ImageFilter): number | undefined {
  const { data } = useImageList(filter)
  return data?.pages[0]?.total
}
