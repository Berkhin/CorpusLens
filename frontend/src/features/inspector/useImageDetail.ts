/** Single-image inspection: the full record with every reference caption. */

import { useQuery, type UseQueryResult } from '@tanstack/react-query'

import { fetchImageDetail } from '@/lib/api-client'
import { queryKeys } from '@/lib/query-keys'
import type { InspectedImage } from '@/types/api'

/**
 * Fetch one image's detail record.
 *
 * Search responses embed the full caption record for each hit, so this used to
 * be seeded from one to save a request. It no longer is: the endpoint now also
 * returns the image's quality measurements, which a search hit does not carry,
 * and seeding would populate the shared cache entry with `analysis: null` —
 * asserting the absence of a measurement that exists. One extra request over
 * loopback is the cheaper mistake.
 *
 * @param imageId Image to inspect, or `null` when nothing is selected — the
 *   query stays disabled rather than the caller conditionally calling a hook.
 * @param options.enabled Whether this caller is ready to *fetch*. Defaults to
 *   true, which is what the inspector wants: it opens on a deliberate click.
 *   The projection's hover card passes false until the cursor settles, so a
 *   sweep across the cloud does not issue a request per point crossed.
 *
 *   Gating the fetch is deliberately not the same as passing a null id. A
 *   disabled query still reads the cache, so a point whose record was already
 *   fetched renders instantly on re-hover while an unseen one waits out the
 *   delay — the debounce costs nothing where the answer is already known.
 */
export function useImageDetail(
  imageId: string | null,
  options: { enabled?: boolean } = {},
): UseQueryResult<InspectedImage, Error> {
  return useQuery({
    queryKey: queryKeys.dataset.detail(imageId ?? ''),
    queryFn: ({ signal }) => {
      // Unreachable while `enabled` is false; throwing keeps the id non-null
      // without a non-null assertion.
      if (imageId === null) throw new Error('useImageDetail: no image selected')
      return fetchImageDetail(imageId, signal)
    },
    enabled: imageId !== null && (options.enabled ?? true),
  })
}
