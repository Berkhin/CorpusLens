/** The 2-D embedding map of the whole corpus. */

import { useMemo } from 'react'

import { useQuery, type UseQueryResult } from '@tanstack/react-query'

import type { ImageFilter } from '@/features/filters/image-filter'
import { fetchProjection } from '@/lib/api-client'
import { queryKeys } from '@/lib/query-keys'
import type { Projection } from '@/types/api'

/**
 * Load the point cloud.
 *
 * `staleTime: Infinity` because the projection is a build artefact: it cannot
 * change while the page is open, so any refetch would be pure cost. The filter
 * is in the key, so switching filters is a separate entry rather than a refetch
 * of the same one.
 */
export function useProjection(filter: ImageFilter): UseQueryResult<Projection, Error> {
  return useQuery({
    queryKey: queryKeys.projection.map(filter),
    queryFn: ({ signal }) => fetchProjection({ filter }, signal),
    staleTime: Infinity,
    // A 404 means "not computed", which no amount of retrying will fix.
    retry: false,
  })
}

/**
 * The point coordinates as a flat `[x0, y0, x1, y1, …]` buffer.
 *
 * The hit-test and the renderer both walk every point on every pointer move and
 * every frame. Reading `points[i].x` off an array of objects makes that a
 * pointer chase through 8 000 heap objects; a `Float32Array` makes it a linear
 * scan over 64 KB. Memoised on the array identity so it is built once per
 * response rather than once per render.
 */
export function usePointPositions(points: readonly { x: number; y: number }[]): Float32Array {
  return useMemo(() => {
    const buffer = new Float32Array(points.length * 2)
    for (let index = 0; index < points.length; index += 1) {
      const point = points[index]
      if (point === undefined) continue
      buffer[index * 2] = point.x
      buffer[index * 2 + 1] = point.y
    }
    return buffer
  }, [points])
}
