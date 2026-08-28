/** Writing the user's partition of the corpus. */

import { useMutation, useQueryClient, type UseMutationResult } from '@tanstack/react-query'

import {
  createCollection,
  deleteCollection,
  moveImagesToCollection,
  renameCollection,
  resetImageCollection,
  type CollectionMoveSource,
} from '@/lib/api-client'
import { queryKeys } from '@/lib/query-keys'
import type { Collection, CollectionMove } from '@/types/api'

/** Arguments for a rename. */
export type RenameVariables = { collectionId: string; name: string }

/** Arguments for a move: a destination and either an id list or a filter. */
export type MoveVariables = { collectionId: string; source: CollectionMoveSource }

/** Arguments for returning one image to its ground-truth split. */
export type ResetVariables = { collectionId: string; imageId: string }

/**
 * Invalidate everything a collection change can be seen through.
 *
 * This is the subtle part of the whole feature, and it is worth spelling out.
 *
 * Moving an image changes the *result set* of every collection-filtered query —
 * the grid, the map, the ranked search — but it does **not** change any of their
 * cache keys: the filter the user has selected is byte-for-byte the same before
 * and after. Combined with the app-wide `staleTime: Infinity` (correct
 * everywhere else, because the index is built offline and read-only), nothing
 * would ever refetch and the UI would keep serving the pre-move pages.
 *
 * So all four roots are invalidated on every successful write. It over-fetches
 * slightly — a rename cannot change a result set — but the alternative is a
 * per-mutation matrix of which keys a given change can reach, which is exactly
 * the kind of cleverness that goes stale the first time someone adds a field.
 * At this scale the extra requests are a few filtered scans against a local
 * 8 000-row table.
 *
 * Do not "optimise" this down to `collections.all`.
 */
function useInvalidateAfterChange(): () => Promise<void> {
  const queryClient = useQueryClient()
  return async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: queryKeys.collections.all }),
      queryClient.invalidateQueries({ queryKey: queryKeys.dataset.all }),
      queryClient.invalidateQueries({ queryKey: queryKeys.search.all }),
      queryClient.invalidateQueries({ queryKey: queryKeys.projection.all }),
    ])
  }
}

/** Create a collection. Fails with a 409 `ApiError` if the name is taken. */
export function useCreateCollection(): UseMutationResult<Collection, Error, string> {
  const invalidate = useInvalidateAfterChange()
  return useMutation({
    mutationFn: (name: string) => createCollection(name),
    onSuccess: invalidate,
  })
}

/** Rename a collection. Fails with 403 for a built-in, 409 for a duplicate name. */
export function useRenameCollection(): UseMutationResult<Collection, Error, RenameVariables> {
  const invalidate = useInvalidateAfterChange()
  return useMutation({
    mutationFn: ({ collectionId, name }: RenameVariables) => renameCollection(collectionId, name),
    onSuccess: invalidate,
  })
}

/**
 * Delete a collection; its images revert to their ground-truth splits.
 *
 * The caller is responsible for dropping the id from an active filter — see
 * `CollectionManagerDialog`. It is done at the mutation site because that is
 * where the deleted id is known; a `useEffect` pruning ids that no longer
 * resolve would be reacting to state it could have been told about.
 */
export function useDeleteCollection(): UseMutationResult<void, Error, string> {
  const invalidate = useInvalidateAfterChange()
  return useMutation({
    mutationFn: (collectionId: string) => deleteCollection(collectionId),
    onSuccess: invalidate,
  })
}

/** Move images into a collection, by explicit id list or by filter. */
export function useMoveImages(): UseMutationResult<CollectionMove, Error, MoveVariables> {
  const invalidate = useInvalidateAfterChange()
  return useMutation({
    mutationFn: ({ collectionId, source }: MoveVariables) =>
      moveImagesToCollection(collectionId, source),
    onSuccess: invalidate,
  })
}

/** Return one image to its ground-truth split. */
export function useResetImage(): UseMutationResult<void, Error, ResetVariables> {
  const invalidate = useInvalidateAfterChange()
  return useMutation({
    mutationFn: ({ collectionId, imageId }: ResetVariables) =>
      resetImageCollection(collectionId, imageId),
    onSuccess: invalidate,
  })
}
