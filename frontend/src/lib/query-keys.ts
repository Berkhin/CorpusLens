/**
 * Central query-key factory.
 *
 * Keys are built here rather than inline at each `useQuery` call so that a
 * cache entry can be found and invalidated from anywhere without re-deriving
 * the array literal — and so a typo produces a type error instead of a silent
 * second cache entry.
 *
 * Anything that changes the *result set* belongs in the key, which is why the
 * filter is threaded through `list` and `query`. It goes in as
 * {@link filterCacheKey}'s canonical form so that two selections differing only
 * in the order the user clicked them share one cache entry.
 */

import { filterCacheKey, type ImageFilter } from '@/features/filters/image-filter'
import type { SearchTarget } from '@/lib/api-client'

export const queryKeys = {
  dataset: {
    all: ['dataset'] as const,
    stats: () => [...queryKeys.dataset.all, 'stats'] as const,
    list: (pageSize: number, filter: ImageFilter) =>
      [...queryKeys.dataset.all, 'list', { pageSize, filter: filterCacheKey(filter) }] as const,
    detail: (imageId: string) => [...queryKeys.dataset.all, 'detail', imageId] as const,
  },
  projection: {
    all: ['projection'] as const,
    map: (filter: ImageFilter) =>
      [...queryKeys.projection.all, { filter: filterCacheKey(filter) }] as const,
  },
  search: {
    all: ['search'] as const,
    target: (target: SearchTarget | null, limit: number, filter: ImageFilter) =>
      [...queryKeys.search.all, { target, limit, filter: filterCacheKey(filter) }] as const,
  },
  /**
   * The user's partition overlay.
   *
   * Unlike every other key here this one addresses *mutable* state, so it is
   * the one thing in the app that genuinely goes stale. See
   * `useCollectionMutations` for what has to be invalidated alongside it.
   */
  collections: {
    all: ['collections'] as const,
    list: () => [...queryKeys.collections.all, 'list'] as const,
  },
} as const
