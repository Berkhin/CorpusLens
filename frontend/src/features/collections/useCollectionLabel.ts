/** Turning a collection id into the name a person recognises. */

import { useCallback } from 'react'

import { useCollections } from '@/features/collections/useCollections'

/**
 * Resolve a collection id to its display name.
 *
 * `null` means "not resolved yet", and callers render nothing rather than
 * falling back to the id. That distinction is the whole point: a built-in's id
 * *is* its split name, so a fallback looks perfectly correct for `train` and
 * renders `376a6824e79b41f8b0df914b0a2baaf4` for a user collection — which is
 * how the raw uuid reached the grid in the first place.
 *
 * An id that resolves to nothing once the list *has* loaded also returns
 * `null`: it means the collection was deleted underneath a cached page, and a
 * blank badge is a better answer there than a hex string.
 */
export type CollectionLabel = (collectionId: string) => string | null

/**
 * A stable id→name resolver over the collection list.
 *
 * A function rather than the map itself, so a caller cannot accidentally hold
 * the map across an invalidation. Call this **once per view**, not once per
 * card: it subscribes to the collections query, and 250 subscriptions to one
 * cache entry would re-render the whole grid on every move.
 */
export function useCollectionLabel(): CollectionLabel {
  const { data } = useCollections()

  return useCallback(
    (collectionId: string) =>
      data?.find((collection) => collection.id === collectionId)?.name ?? null,
    [data],
  )
}
