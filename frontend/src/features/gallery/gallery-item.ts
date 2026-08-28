/**
 * The view model the image grid renders.
 *
 * Browsing and searching return different payloads — summaries without
 * captions, and ranked hits that carry the full detail record — but they are
 * displayed by the same grid. Normalising both into one shape here keeps that
 * difference out of the components (CLAUDE.md §4.3: no business logic in JSX).
 */

import type { CollectionLabel } from '@/features/collections/useCollectionLabel'
import { resolveImageUrl } from '@/lib/api-client'
import type { ImageDetail, ImageSummary, SearchResult } from '@/types/api'

export type GalleryItem = {
  id: string
  /** Absolute URL, ready for `<img src>`. */
  imageUrl: string
  /** Alternative text; a caption when the payload carried one. */
  alt: string
  /** The dataset's own partition. Immutable ground truth. */
  split: string
  /**
   * Effective collection **id**. Equal to `split` unless the image was moved.
   * This is the value comparisons and mutations use; it is not for display —
   * for a user collection it is a uuid4 hex string.
   */
  collection: string
  /**
   * The same collection's display name, or `null` while the collection list is
   * still loading (or if the collection has since been deleted).
   *
   * Resolved here, at the point the two payload shapes are normalised, rather
   * than inside the card: the lookup needs the collections query, and one
   * subscription per view beats one per card.
   */
  collectionLabel: string | null
  /** Cosine similarity — present only for search hits. */
  score?: number
  /**
   * The full record, when the payload already contained it. Search responses
   * embed every hit's captions, so the inspector can open on a search result
   * without a second round trip.
   */
  detail?: ImageDetail
}

/**
 * Describe an image that arrived without captions.
 *
 * CLAUDE.md §5.2 asks for the caption as alt text, but `GET /api/dataset`
 * returns summaries only. Naming the id keeps the text useful to a screen
 * reader — and unique, which "Dataset image" alone would not be.
 */
function fallbackAlt(summary: ImageSummary): string {
  return `Corpus image ${summary.id} (${summary.split} split)`
}

/** Build a grid item from a browse-view summary. */
export function galleryItemFromSummary(
  summary: ImageSummary,
  collectionLabel: CollectionLabel,
): GalleryItem {
  return {
    id: summary.id,
    imageUrl: resolveImageUrl(summary.image_url),
    alt: fallbackAlt(summary),
    split: summary.split,
    collection: summary.collection,
    collectionLabel: collectionLabel(summary.collection),
  }
}

/** Build a grid item from a ranked search hit. */
export function galleryItemFromSearchResult(
  result: SearchResult,
  collectionLabel: CollectionLabel,
): GalleryItem {
  const { image, score } = result
  // Indexing is guarded rather than asserted: `noUncheckedIndexedAccess` is on,
  // and caption_count is not structurally guaranteed to be non-zero.
  const firstCaption = image.captions[0]

  return {
    id: image.id,
    imageUrl: resolveImageUrl(image.image_url),
    alt: firstCaption ?? fallbackAlt(image),
    split: image.split,
    collection: image.collection,
    collectionLabel: collectionLabel(image.collection),
    score,
    detail: image,
  }
}

/**
 * A placeholder item for an image whose record has not been fetched yet.
 *
 * The map and the inspector's neighbour link both open the dialog from an id
 * alone; the dialog fetches the real record and overwrites all of this. Kept
 * here rather than inline at each call site so "what an unresolved item looks
 * like" has one definition.
 */
export function placeholderGalleryItem(
  imageId: string,
  overrides: Partial<GalleryItem> = {},
): GalleryItem {
  return {
    id: imageId,
    imageUrl: '',
    alt: `Corpus image ${imageId}`,
    split: '',
    collection: '',
    collectionLabel: null,
    ...overrides,
  }
}
