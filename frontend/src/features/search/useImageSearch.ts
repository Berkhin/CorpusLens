/** CLIP text→image semantic search. */

import { useQuery, type UseQueryResult } from '@tanstack/react-query'

import type { ImageFilter } from '@/features/filters/image-filter'
import { searchImages, type SearchTarget } from '@/lib/api-client'
import { queryKeys } from '@/lib/query-keys'
import type { SearchResponse } from '@/types/api'

/** Hits requested per query. Well under the backend's `MAX_SEARCH_LIMIT` of 100. */
export const SEARCH_RESULT_LIMIT = 30

/**
 * Rank images against a natural-language query.
 *
 * Disabled on an empty query so clearing the search bar drops straight back to
 * the browse view without firing a request. Each distinct query is its own
 * cache entry, so re-running a previous search is instant — worth having, since
 * the backend has to run a CPU forward pass through CLIP's text encoder.
 *
 * @param target What to rank against, or `null` when not searching.
 * @param filter Corpus narrowing applied before ranking, not after it.
 * @param limit Maximum hits to return.
 */
export function useImageSearch(
  target: SearchTarget | null,
  filter: ImageFilter,
  limit: number = SEARCH_RESULT_LIMIT,
): UseQueryResult<SearchResponse, Error> {
  return useQuery({
    queryKey: queryKeys.search.target(target, limit, filter),
    queryFn: ({ signal }) => {
      // Unreachable while `enabled` is false.
      if (target === null) throw new Error('useImageSearch: no search target')
      return searchImages({ target, limit, filter }, signal)
    },
    enabled: target !== null,
  })
}
