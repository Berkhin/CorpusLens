/** Corpus-level counts for the footer dashboard. */

import { useQuery, type UseQueryResult } from '@tanstack/react-query'

import { fetchDatasetStats } from '@/lib/api-client'
import { queryKeys } from '@/lib/query-keys'
import type { DatasetStats } from '@/types/api'

/** Fetch the total image count and the per-split breakdown. */
export function useDatasetStats(): UseQueryResult<DatasetStats, Error> {
  return useQuery({
    queryKey: queryKeys.dataset.stats(),
    queryFn: ({ signal }) => fetchDatasetStats(signal),
  })
}
