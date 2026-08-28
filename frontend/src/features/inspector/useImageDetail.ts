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
 */
export function useImageDetail(imageId: string | null): UseQueryResult<InspectedImage, Error> {
  return useQuery({
    queryKey: queryKeys.dataset.detail(imageId ?? ''),
    queryFn: ({ signal }) => {
      // Unreachable while `enabled` is false; throwing keeps the id non-null
      // without a non-null assertion.
      if (imageId === null) throw new Error('useImageDetail: no image selected')
      return fetchImageDetail(imageId, signal)
    },
    enabled: imageId !== null,
  })
}
