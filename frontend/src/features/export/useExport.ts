/** Downloading the current selection as a file. */

import { useMutation, type UseMutationResult } from '@tanstack/react-query'

import type { ImageFilter } from '@/features/filters/image-filter'
import { exportImages } from '@/lib/api-client'
import type { ExportFormat } from '@/types/api'

export type ExportScope = {
  filter: ImageFilter
  /** Explicit selection; wins over `query` and the filter, server-side. */
  ids?: readonly string[]
  /** Active text search, exported with similarity scores. */
  query?: string
  /** Active by-example search, exported with similarity scores. */
  similarToImageId?: string
}

/**
 * Hand a blob to the browser as a download.
 *
 * There is no declarative API for "save this response to disk", so this is the
 * standard anchor dance. The object URL is revoked immediately after the click:
 * the browser has already taken its own reference to the blob by then, and
 * leaving it alive would pin the file in memory for the life of the document.
 */
function saveBlob(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = fileName
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

/**
 * Name the file after what is in it, so a folder of exports stays legible.
 *
 * The scope is spelled out rather than timestamped: "selection", "search",
 * "train", "filtered" tells the user what they downloaded, where
 * `flickr8k-export-3.csv` would not.
 *
 * The collection branch replaced an equivalent one on `filter.splits`. The UI
 * no longer sets `splits` — the built-in collections are the splits — so that
 * branch had become unreachable, and unreachable code that looks live is worse
 * than none.
 */
function exportFileName(scope: ExportScope, extension: string): string {
  const parts = ['flickr8k']
  if (scope.ids !== undefined && scope.ids.length > 0) parts.push(`selection-${scope.ids.length}`)
  else if (scope.query !== undefined && scope.query.length > 0) parts.push('search')
  else if (scope.similarToImageId !== undefined) parts.push(`similar-${scope.similarToImageId}`)
  else if (scope.filter.collections.length > 0) parts.push(scope.filter.collections.join('-'))
  else if (scope.filter.captionContains.length > 0) parts.push('filtered')
  else parts.push('corpus')
  return `${parts.join('-')}.${extension}`
}

/**
 * Fetch and save a manifest of the current scope.
 *
 * A mutation rather than a query: this has an effect the user asked for at a
 * moment they chose, and caching a download would be meaningless — the point is
 * the file, not the value.
 */
export function useExport(scope: ExportScope): UseMutationResult<void, Error, ExportFormat> {
  return useMutation({
    mutationFn: async (format: ExportFormat) => {
      const blob = await exportImages({
        format,
        filter: scope.filter,
        ids: scope.ids,
        query: scope.query,
        similarToImageId: scope.similarToImageId,
      })
      saveBlob(blob, exportFileName(scope, format))
    },
  })
}
