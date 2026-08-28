/**
 * The single place this app talks to the FastAPI backend.
 *
 * Everything here returns plain domain data or throws {@link ApiError}; no
 * component or hook constructs a URL or reads a `Response` itself.
 */

import {
  appendFilterParams,
  filterRequestFields,
  type ImageFilter,
} from '@/features/filters/image-filter'
import type {
  Collection,
  CollectionMove,
  DatasetStats,
  ExportFormat,
  ImagePage,
  InspectedImage,
  Projection,
  SearchResponse,
} from '@/types/api'

/**
 * Where the API lives. Absolute by default because in development the app is
 * served by Vite on :5173 while the API listens on :8000 — the two origins the
 * backend's CORS policy already whitelists, so no dev proxy is needed.
 *
 * An **empty string is meaningful, not missing**: it makes every URL below
 * root-relative, i.e. same-origin. That is the container build's configuration,
 * where one FastAPI process serves both this bundle and the API on one port.
 * Hence `??` and not `||` — the latter would treat `''` as unset and send the
 * containerised client back to :8000, where nothing is listening.
 */
export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

/** Mirrors `Settings.api_prefix` in backend/app/core/config.py. */
const API_PREFIX = '/api'

/**
 * Mirrors `MAX_QUERY_LENGTH` in backend/app/models/schemas.py.
 *
 * Every bound the backend enforces is mirrored in this module and nowhere else,
 * so drift between the two contracts is a one-file grep.
 */
export const MAX_QUERY_LENGTH = 500

/** Mirrors `MAX_COLLECTION_NAME_LENGTH` in backend/app/models/schemas.py. */
export const MAX_COLLECTION_NAME_LENGTH = 64

/**
 * Status carried by an {@link ApiError} that never reached the server.
 *
 * `fetch` rejects rather than resolving when the connection is refused, so
 * there is no real status to report; `0` matches `XMLHttpRequest`'s convention
 * for the same condition and keeps callers on one error type.
 */
export const NETWORK_ERROR_STATUS = 0

/** A failed request, carrying the status so callers can branch on 404. */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string, options?: ErrorOptions) {
    super(message, options)
    this.name = 'ApiError'
    this.status = status
  }
}

/** Shape FastAPI uses for a single 422 validation failure. */
type ValidationErrorItem = { msg?: unknown; loc?: unknown }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

/**
 * Pull a human-readable message out of an error body.
 *
 * FastAPI is not consistent here: handled errors set `detail` to a string,
 * but request-validation failures (422) set it to an array of objects. Both
 * are flattened to one line so the UI has something to render either way.
 */
function extractDetail(body: unknown, fallback: string): string {
  if (!isRecord(body) || !('detail' in body)) return fallback

  const { detail } = body
  if (typeof detail === 'string') return detail

  if (Array.isArray(detail)) {
    const messages = (detail as ValidationErrorItem[])
      .map((item) => (typeof item.msg === 'string' ? item.msg : null))
      .filter((msg): msg is string => msg !== null)
    if (messages.length > 0) return messages.join('; ')
  }

  return fallback
}

/**
 * Perform the fetch, translating a transport failure into an {@link ApiError}.
 *
 * A refused connection is this app's most likely failure — the backend is a
 * separate process the user starts by hand — and `fetch` reports it as a bare
 * `TypeError: Failed to fetch`, which names neither the cause nor the fix.
 * An abort is re-thrown untouched: TanStack Query cancels superseded queries
 * that way and must keep seeing its own rejection.
 */
async function send(url: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init)
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    throw new ApiError(
      NETWORK_ERROR_STATUS,
      `Could not reach the API at ${API_BASE_URL}. Start it with ` +
        '`uvicorn app.main:app --app-dir backend`.',
      { cause },
    )
  }
}

/**
 * Issue a request and decode its JSON body.
 *
 * @param path Path below the API prefix, e.g. `/dataset/stats`.
 * @param init Extra fetch options; `signal` is supplied by TanStack Query so
 *   an abandoned query (a superseded search, an unmounted dialog) cancels the
 *   in-flight request rather than resolving into a discarded cache entry.
 * @throws ApiError When the response status is not 2xx, or the server could
 *   not be reached at all.
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await send(`${API_BASE_URL}${API_PREFIX}${path}`, init)

  if (!response.ok) {
    // A body is not guaranteed on every error path, so parsing is best-effort.
    const body: unknown = await response.json().catch(() => null)
    throw new ApiError(response.status, extractDetail(body, response.statusText))
  }

  return (await response.json()) as T
}

/**
 * Issue a request whose success carries no body.
 *
 * Separate from {@link request} rather than a special case inside it: the
 * collection deletes answer `204 No Content`, and calling `response.json()` on
 * an empty body throws a parse error that would surface as a failed mutation
 * for a request that actually succeeded.
 */
async function requestNoContent(path: string, init?: RequestInit): Promise<void> {
  const response = await send(`${API_BASE_URL}${API_PREFIX}${path}`, init)

  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null)
    throw new ApiError(response.status, extractDetail(body, response.statusText))
  }
}

/**
 * Issue a request whose body is a file rather than JSON.
 *
 * Separate from {@link request} because only the *success* path differs: an
 * error still arrives as a JSON problem document, so failures are decoded the
 * same way and surface as the same {@link ApiError}.
 */
async function requestBlob(path: string, init?: RequestInit): Promise<Blob> {
  const response = await send(`${API_BASE_URL}${API_PREFIX}${path}`, init)

  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null)
    throw new ApiError(response.status, extractDetail(body, response.statusText))
  }

  return await response.blob()
}

/**
 * Turn a backend `image_url` into something an `<img src>` can load.
 *
 * The API returns image URLs root-relative (`/images/foo.jpg`) on purpose, so
 * the client never composes a path from `file_name` and cannot get the static
 * mount point wrong. All this does is resolve that against the API origin.
 */
export function resolveImageUrl(imageUrl: string): string {
  return `${API_BASE_URL}${imageUrl}`
}

/** Fetch the total image count and per-split breakdown. */
export function fetchDatasetStats(signal?: AbortSignal): Promise<DatasetStats> {
  return request<DatasetStats>('/dataset/stats', { signal })
}

/** Fetch one page of image summaries, optionally narrowed by a filter. */
export function fetchImagePage(
  params: { offset: number; limit: number; filter: ImageFilter },
  signal?: AbortSignal,
): Promise<ImagePage> {
  const search = new URLSearchParams({
    offset: String(params.offset),
    limit: String(params.limit),
  })
  appendFilterParams(search, params.filter)
  return request<ImagePage>(`/dataset?${search.toString()}`, { signal })
}

/**
 * Fetch the whole embedding map.
 *
 * Unpaged by design — a scatter plot with a page missing is not a smaller plot,
 * it is a misleading one. The filter travels along so the backend can mark
 * which points match; non-matching points still come back, to be dimmed.
 *
 * @throws ApiError with status 404 when no projection has been computed.
 */
export function fetchProjection(
  params: { filter: ImageFilter },
  signal?: AbortSignal,
): Promise<Projection> {
  const search = new URLSearchParams()
  appendFilterParams(search, params.filter)
  const query = search.toString()
  return request<Projection>(`/projection${query === '' ? '' : `?${query}`}`, { signal })
}

/** Fetch one image with its captions and, when computed, its quality measurements. */
export function fetchImageDetail(imageId: string, signal?: AbortSignal): Promise<InspectedImage> {
  return request<InspectedImage>(`/dataset/${encodeURIComponent(imageId)}`, { signal })
}

/**
 * What a search is ranking against.
 *
 * A union rather than two nullable fields, because the backend requires exactly
 * one and a shape that cannot express "both" or "neither" is a shape that
 * cannot send a request the API will reject.
 */
export type SearchTarget = { kind: 'text'; query: string } | { kind: 'image'; imageId: string }

/**
 * Rank images by CLIP similarity to a query, or to another image.
 *
 * The filter travels in the body because the backend applies it *before*
 * ranking — asking for 20 hits inside one split returns the best 20 in that
 * split, not whatever of the global best 20 happens to be in it.
 */
export function searchImages(
  params: { target: SearchTarget; limit: number; filter: ImageFilter },
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const { target } = params
  return request<SearchResponse>('/search', {
    method: 'POST',
    // Content-Type is the only header the API's CORS policy allows.
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...(target.kind === 'text' ? { query: target.query } : { image_id: target.imageId }),
      limit: params.limit,
      ...filterRequestFields(params.filter),
    }),
    signal,
  })
}

/**
 * Download a manifest of the current selection.
 *
 * Three sources, resolved server-side in this precedence: an explicit `ids`
 * list, then a `query` (re-ranked on the backend so the file matches a
 * reproducible query rather than whatever the grid had loaded), then everything
 * matching the filter.
 */
export function exportImages(
  params: {
    format: ExportFormat
    filter: ImageFilter
    ids?: readonly string[]
    query?: string
    similarToImageId?: string
    limit?: number
  },
  signal?: AbortSignal,
): Promise<Blob> {
  const body: Record<string, unknown> = {
    format: params.format,
    ...filterRequestFields(params.filter),
  }
  if (params.ids !== undefined && params.ids.length > 0) body.ids = [...params.ids]
  if (params.query !== undefined && params.query.length > 0) body.query = params.query
  if (params.similarToImageId !== undefined) body.image_id = params.similarToImageId
  if (params.limit !== undefined) body.limit = params.limit

  return requestBlob('/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
}

/** List every collection with its current size, built-ins first. */
export function fetchCollections(signal?: AbortSignal): Promise<Collection[]> {
  return request<Collection[]>('/collections', { signal })
}

/** Create a user collection. Rejects with a 409 `ApiError` on a duplicate name. */
export function createCollection(name: string, signal?: AbortSignal): Promise<Collection> {
  return request<Collection>('/collections', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
    signal,
  })
}

/** Rename a user collection. 403 for a built-in, 409 for a duplicate name. */
export function renameCollection(
  collectionId: string,
  name: string,
  signal?: AbortSignal,
): Promise<Collection> {
  return request<Collection>(`/collections/${encodeURIComponent(collectionId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
    signal,
  })
}

/** Delete a user collection; its images revert to their splits. 403 for a built-in. */
export function deleteCollection(collectionId: string, signal?: AbortSignal): Promise<void> {
  return requestNoContent(`/collections/${encodeURIComponent(collectionId)}`, {
    method: 'DELETE',
    signal,
  })
}

/**
 * Which images a move addresses.
 *
 * A union rather than two optional fields, for the same reason
 * {@link SearchTarget} is one: the backend requires exactly one of `ids` and
 * `filter` and rejects a body carrying both, so a shape that cannot express the
 * invalid state cannot send the invalid request.
 *
 * The `filter` arm is what makes "quarantine the 200 weak captions" a single
 * action — that set is scattered across the whole embedding cloud, so no
 * rectangle on the map can approximate it and no one is going to click 200
 * times.
 */
export type CollectionMoveSource =
  | {
      kind: 'ids'
      ids: readonly string[]
      /**
       * How the ids were chosen, recorded as the batch's provenance. Omitted
       * means `manual`. A pasted list and a lassoed one arrive at the API
       * identically, so only the caller can tell them apart.
       */
      origin?: 'manual' | 'import'
    }
  | { kind: 'filter'; filter: ImageFilter }

/**
 * Move images into a collection, reporting what changed and what was unknown.
 *
 * `unknown` is always empty on the filter channel — every id came out of the
 * index server-side — and is the reporting channel that matters when a list of
 * ids was pasted in from somewhere else.
 *
 * A selection larger than the server's ceiling fails with a 413 before anything
 * is written, which surfaces here as an `ApiError`.
 */
export function moveImagesToCollection(
  collectionId: string,
  source: CollectionMoveSource,
  signal?: AbortSignal,
): Promise<CollectionMove> {
  const body =
    source.kind === 'ids'
      ? { ids: [...source.ids], origin: source.origin ?? 'manual' }
      : { filter: filterRequestFields(source.filter) }

  return request<CollectionMove>(`/collections/${encodeURIComponent(collectionId)}/images`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
}

/** Drop one image's override, returning it to its ground-truth split. */
export function resetImageCollection(
  collectionId: string,
  imageId: string,
  signal?: AbortSignal,
): Promise<void> {
  const collection = encodeURIComponent(collectionId)
  return requestNoContent(`/collections/${collection}/images/${encodeURIComponent(imageId)}`, {
    method: 'DELETE',
    signal,
  })
}
