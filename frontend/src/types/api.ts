/**
 * TypeScript mirrors of the API's Pydantic response models.
 *
 * Field names are deliberately snake_case rather than the camelCase CLAUDE.md
 * §5.3 prescribes for TS: these types describe the wire format, and renaming
 * them here would mean maintaining a mapping layer whose only job is to hide
 * that the backend speaks Python. Keeping the names identical makes drift
 * between `backend/app/models/schemas.py` and this file greppable.
 *
 * Source of truth: backend/app/models/schemas.py.
 */

/** An image as it appears in a grid or list view (`ImageSummaryResponse`). */
export type ImageSummary = {
  id: string
  file_name: string
  /**
   * Dataset split: `train`, `validation`, or `test` in the shipped corpus.
   * Immutable ground truth — a collection move never changes it.
   */
  split: string
  /**
   * The collection this image currently sits in. Equal to `split` unless the
   * user has moved it. Shown beside the split wherever the two differ, so a
   * re-partition never quietly redefines what "test set" means.
   */
  collection: string
  /** Root-relative, e.g. `/images/1000268201_693b08cb0e.jpg`. */
  image_url: string
}

/** A single image with all of its reference captions (`ImageDetailResponse`). */
export type ImageDetail = ImageSummary & {
  /** Human reference captions, normally five. */
  captions: string[]
  caption_count: number
}

/** One page of summaries plus the totals a paginator needs (`ImagePageResponse`). */
export type ImagePage = {
  items: ImageSummary[]
  /** Images matching the active filter; this is what paginates. */
  total: number
  /** Images in the whole index, ignoring the filter. Equals `total` when unfiltered. */
  corpus_total: number
  offset: number
  limit: number
  has_more: boolean
}

/** Data-quality findings the offline pass can select (`QualityFlag`). */
export type QualityFlag = 'near-duplicate' | 'cross-split-duplicate' | 'weak-captions'

/** What the offline pass measured about one image (`ImageAnalysisResponse`). */
export type ImageAnalysis = {
  nearest_neighbour_id: string
  /** Near 1.0 is a duplicate or a re-shoot; the corpus median is about 0.83. */
  nearest_neighbour_similarity: number
  /** Median rank of this image under its own captions; null if not measured. */
  caption_rank: number | null
}

/** An image with its captions and, when computed, its measurements (`InspectedImageResponse`). */
export type InspectedImage = {
  image: ImageDetail
  analysis: ImageAnalysis | null
}

/** Corpus recall of its own captions (`CaptionRetrievalResponse`). */
export type CaptionRetrieval = {
  recall_at_1: number
  recall_at_5: number
  recall_at_10: number
  captions: number
}

/**
 * One collection's caption recall (`CollectionCaptionRecallResponse`).
 *
 * A *re-aggregation* of each image's median own-caption rank **against the full
 * corpus** — not a ranking restricted to the collection, which would be a
 * different and harder number needing `analyze.py` re-run per partition.
 *
 * Not comparable with {@link CaptionRetrieval}: the denominator here is
 * `images`, there it is `captions`, and each image contributes the median of
 * its own five rather than each caption counting separately.
 */
export type CollectionCaptionRecall = {
  recall_at_1: number
  recall_at_5: number
  recall_at_10: number
  /** Images in the collection carrying a measured rank; not its size. */
  images: number
}

/** Corpus-level counts (`DatasetStatsResponse`). */
export type DatasetStats = {
  total_images: number
  /**
   * Row count per split; only splits present in the index are listed.
   * Unaffected by collection moves — this is the dataset's own partition.
   */
  images_by_split: Record<string, number>
  /** Row count per effective collection, keyed by collection id. Follows moves. */
  images_by_collection: Record<string, number>
  /**
   * Whether `GET /projection` will succeed. Lets the shell hide the map tab
   * instead of discovering the absence through a failed request.
   */
  projection_available: boolean
  /** Whether the data-quality artefact exists; gates the quality filters. */
  analysis_available: boolean
  near_duplicate_images: number | null
  /**
   * Near-duplicate pairs spanning two **splits** — the evaluation-leakage count
   * against the dataset's own partition. Never moves; that is the point of it.
   */
  cross_split_duplicate_pairs: number | null
  /**
   * The same pairs counted against the user's **collections**. Falls as leaking
   * images are quarantined together, while the figure above holds — which is
   * how the effect of a re-partition becomes visible rather than assumed.
   */
  cross_collection_duplicate_pairs: number | null
  caption_retrieval: CaptionRetrieval | null
  /** Per-collection caption recall, keyed by collection id. */
  caption_recall_by_collection: Record<string, CollectionCaptionRecall> | null
}

/** One ranked hit (`SearchResultResponse`). */
export type SearchResult = {
  /** Search hits carry the *full* detail record, captions included. */
  image: ImageDetail
  /** Cosine similarity in [-1, 1]; higher is a closer match. */
  score: number
}

/** One image's position on the embedding map (`ProjectionPointResponse`). */
export type ProjectionPoint = {
  id: string
  split: string
  /** Normalised to about [-1, 1]. */
  x: number
  /** Same scale as `x` — never scale the axes independently. */
  y: number
  /** False when the image falls outside the active filter; dim, do not drop. */
  matches: boolean
}

/** The whole embedding map (`ProjectionResponse`). */
export type Projection = {
  /** `pca` or `tsne`. */
  method: string
  count: number
  match_count: number
  /**
   * Share of total variance per component, for PCA. `null` for t-SNE, where the
   * quantity is undefined. Low values mean broad structure, not clusters.
   */
  explained_variance_ratio: number[] | null
  points: ProjectionPoint[]
}

/**
 * Formats `POST /api/export` can produce (`ExportFormat`).
 *
 * `csv` flattens captions into fixed columns for spreadsheets and pandas;
 * `jsonl` keeps them as a list and is the lossless one.
 */
export type ExportFormat = 'csv' | 'jsonl'

/** The full ranked result set for one query (`SearchResponse`). */
export type SearchResponse = {
  /** The query as the backend interpreted it, after trimming. */
  query: string
  count: number
  results: SearchResult[]
}

/** Where a collection came from (`CollectionKind`). */
export type CollectionKind = 'builtin' | 'user'

/** How an image came to be in the collection it is in (`CollectionOrigin`). */
export type CollectionOrigin = 'manual' | 'filter' | 'import'

/**
 * Where a collection's most recent members came from (`CollectionProvenanceResponse`).
 *
 * The last batch, not a full history. `detail` carries the filter that matched
 * them, as compact JSON, when `origin` is `filter` — which is what makes the
 * partition reproducible rather than merely present.
 */
export type CollectionProvenance = {
  origin: CollectionOrigin
  detail: string | null
  /** ISO-8601 UTC. */
  moved_at: string
}

/**
 * One partition of the corpus as the researcher sees it (`CollectionResponse`).
 *
 * `split` is the dataset's immutable partition; a collection is the editable
 * overlay on top of it. The three built-ins mirror the splits actually present
 * in the index and cannot be renamed or deleted.
 */
export type Collection = {
  /** A split name for built-ins, a uuid4 hex string for user collections. */
  id: string
  name: string
  kind: CollectionKind
  /** Images currently in this collection, overrides applied. */
  size: number
  /** How its most recent members got here; `null` if nothing ever was. */
  provenance: CollectionProvenance | null
}

/** What a move actually did (`CollectionMoveResponse`). */
export type CollectionMove = {
  moved: number
  unchanged: number
  /** Requested ids that are not in the index. Reported, not stored. */
  unknown: string[]
}
