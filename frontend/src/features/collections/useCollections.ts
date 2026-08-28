/** Reading the user's partition of the corpus. */

import { useQuery, type UseQueryResult } from '@tanstack/react-query'

import { fetchCollections } from '@/lib/api-client'
import { queryKeys } from '@/lib/query-keys'
import type { Collection } from '@/types/api'

/**
 * Every collection, with its current size.
 *
 * Sizes come from here rather than from `images_by_split`, because they have to
 * reflect moves — `images_by_split` deliberately never does.
 *
 * This is the one query in the app whose answer can change while the app is
 * open, so the global `staleTime: Infinity` is a promise it cannot keep on its
 * own. `useCollectionMutations` is what invalidates it; nothing else will.
 */
export function useCollections(): UseQueryResult<Collection[], Error> {
  return useQuery({
    queryKey: queryKeys.collections.list(),
    queryFn: ({ signal }) => fetchCollections(signal),
  })
}
