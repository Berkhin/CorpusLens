/**
 * The corpus narrowing shared by browsing, search and export.
 *
 * One shape, three consumers, three wire formats: repeated query parameters for
 * `GET /api/dataset`, body fields for `POST /api/search`, and a stable object
 * for the TanStack Query cache key. Keeping all three conversions here is what
 * stops a hook from hand-rolling a `URLSearchParams` and drifting from the
 * backend contract.
 *
 * Mirrors `ImageFilter` in backend/app/models/domain.py.
 */

import type { QualityFlag } from '@/types/api'

/** Mirrors `MAX_CAPTION_FILTER_LENGTH` in backend/app/models/schemas.py. */
export const MAX_CAPTION_FILTER_LENGTH = 100

export type ImageFilter = {
  /**
   * Splits to keep. Empty means every split.
   *
   * `splits` is the dataset's own immutable partition; `collections` is the
   * user's working one. Both are real API dimensions and they are deliberately
   * not merged: the leakage figures in the data-quality pass are computed from
   * the splits, so collapsing one into the other would make those numbers
   * describe something other than what they say. The UI currently drives only
   * `collections` — the built-in collections *are* the splits — but the backend
   * still accepts both, and this field is what a "filter by ground truth"
   * control would use.
   */
  splits: readonly string[]
  /**
   * Collections to keep. Empty means every collection.
   *
   * Like `qualityFlag` this is not a property of a row — the backend resolves
   * it against the overlay store — but from here it is just another dimension.
   */
  collections: readonly string[]
  /**
   * Case-insensitive substring a caption must contain. `''` means "no caption
   * filter" — the backend rejects an empty string, so it is dropped rather than
   * sent, and this type stays free of a second empty-ish value to check for.
   */
  captionContains: string
  /**
   * A finding from the offline data-quality pass. `null` means no such
   * narrowing. Unlike the other two this is not a property of a row — the
   * backend resolves it to a set of ids — but from here it is just another
   * filter dimension, which is the point.
   */
  qualityFlag: QualityFlag | null
}

export const EMPTY_IMAGE_FILTER: ImageFilter = {
  splits: [],
  collections: [],
  captionContains: '',
  qualityFlag: null,
}

/** Whether this filter would narrow anything. */
export function isFilterActive(filter: ImageFilter): boolean {
  return (
    filter.splits.length > 0 ||
    filter.collections.length > 0 ||
    filter.captionContains.length > 0 ||
    filter.qualityFlag !== null
  )
}

/**
 * A canonical, order-independent value to hash into a query key.
 *
 * TanStack Query hashes object keys in sorted order but preserves array order,
 * so without sorting, selecting `train` then `test` and `test` then `train`
 * would be two cache entries for one result set.
 */
export function filterCacheKey(filter: ImageFilter): {
  splits: string[]
  collections: string[]
  captionContains: string
  qualityFlag: QualityFlag | null
} {
  return {
    splits: [...filter.splits].sort(),
    collections: [...filter.collections].sort(),
    captionContains: filter.captionContains,
    qualityFlag: filter.qualityFlag,
  }
}

/** Append the filter to a query string, omitting parts that are not set. */
export function appendFilterParams(search: URLSearchParams, filter: ImageFilter): void {
  for (const split of filter.splits) search.append('split', split)
  for (const collection of filter.collections) search.append('collection', collection)
  if (filter.captionContains.length > 0) {
    search.set('caption_contains', filter.captionContains)
  }
  if (filter.qualityFlag !== null) search.set('quality_flag', filter.qualityFlag)
}

/**
 * The filter as JSON body fields.
 *
 * `caption_contains` is `null` rather than absent when unset: the backend
 * declares it optional-and-nullable, and an explicit null documents at the wire
 * level that no filter was intended.
 */
export function filterRequestFields(filter: ImageFilter): {
  splits: string[]
  collections: string[]
  caption_contains: string | null
  quality_flag: QualityFlag | null
} {
  return {
    splits: [...filter.splits],
    collections: [...filter.collections],
    caption_contains: filter.captionContains.length > 0 ? filter.captionContains : null,
    quality_flag: filter.qualityFlag,
  }
}

/** Return the filter with one split toggled on or off. */
export function toggleSplit(filter: ImageFilter, split: string): ImageFilter {
  const splits = filter.splits.includes(split)
    ? filter.splits.filter((candidate) => candidate !== split)
    : [...filter.splits, split]
  return { ...filter, splits }
}

/** Return the filter with a quality flag set, or cleared when it was already active. */
export function toggleQualityFlag(filter: ImageFilter, flag: QualityFlag): ImageFilter {
  return { ...filter, qualityFlag: filter.qualityFlag === flag ? null : flag }
}

/** Return the filter with one collection toggled on or off. */
export function toggleCollection(filter: ImageFilter, collection: string): ImageFilter {
  const collections = filter.collections.includes(collection)
    ? filter.collections.filter((candidate) => candidate !== collection)
    : [...filter.collections, collection]
  return { ...filter, collections }
}

/**
 * Return the filter with one collection removed, if it was selected.
 *
 * Used when a collection is deleted while it is being filtered by. The mutation
 * site already knows which id went away, so it drops it directly rather than a
 * `useEffect` watching for ids that no longer resolve.
 */
export function withoutCollection(filter: ImageFilter, collection: string): ImageFilter {
  if (!filter.collections.includes(collection)) return filter
  return { ...filter, collections: filter.collections.filter((id) => id !== collection) }
}
