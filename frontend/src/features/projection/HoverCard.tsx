import type { JSX } from 'react'

import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { useImageDetail } from '@/features/inspector/useImageDetail'
import type { ScreenPoint } from '@/features/projection/scatter-viewport'
import { resolveImageUrl } from '@/lib/api-client'
import { cn } from '@/lib/utils'

/** Card size, in pixels. Fixed so the flip-at-the-edge maths is exact. */
const CARD_WIDTH = 208
const CARD_HEIGHT = 190
const CURSOR_GAP = 14

type HoverCardProps = {
  imageId: string
  split: string
  at: ScreenPoint
  container: { width: number; height: number }
}

/**
 * Thumbnail and first caption for the point under the cursor.
 *
 * The projection payload carries no image URL and no captions — with 8 000
 * points those fields would have added most of a megabyte to a response that
 * ships on every filter change. Instead this fetches the record on demand
 * through the same hook and the same cache entry the inspector uses, so
 * hovering a point warms the dialog that a click then opens.
 */
export function HoverCard({ imageId, split, at, container }: HoverCardProps): JSX.Element {
  const { data, status } = useImageDetail(imageId)

  // Flip across the cursor when the card would otherwise leave the plot.
  const left =
    at.x + CURSOR_GAP + CARD_WIDTH > container.width
      ? at.x - CURSOR_GAP - CARD_WIDTH
      : at.x + CURSOR_GAP
  const top =
    at.y + CURSOR_GAP + CARD_HEIGHT > container.height
      ? Math.max(0, at.y - CURSOR_GAP - CARD_HEIGHT)
      : at.y + CURSOR_GAP

  return (
    <div
      // Position is genuinely dynamic, which is the exception CLAUDE.md §5.2
      // carves out for inline styles.
      style={{ left, top, width: CARD_WIDTH }}
      className={cn(
        'pointer-events-none absolute z-20 overflow-hidden rounded-lg border border-border',
        'bg-popover text-popover-foreground shadow-lg',
      )}
    >
      {status === 'success' ? (
        <img
          src={resolveImageUrl(data.image.image_url)}
          alt={data.image.captions[0] ?? `Flickr8k image ${data.image.id}`}
          className="aspect-4/3 w-full object-cover"
        />
      ) : (
        <Skeleton className="aspect-4/3 w-full rounded-none" />
      )}

      <div className="space-y-1.5 p-2">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate font-mono text-[11px] text-muted-foreground">{imageId}</span>
          <Badge variant="secondary" className="shrink-0">
            {split}
          </Badge>
        </div>
        {status === 'success' ? (
          <p className="line-clamp-2 text-xs leading-snug">{data.image.captions[0]}</p>
        ) : (
          <Skeleton className="h-6 w-full" />
        )}
      </div>
    </div>
  )
}
