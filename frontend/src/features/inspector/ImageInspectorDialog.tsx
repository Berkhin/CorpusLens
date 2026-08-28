import type { JSX } from 'react'

import { Copy, Loader2, Search, Undo2 } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Skeleton } from '@/components/ui/skeleton'
import { ErrorNotice } from '@/components/StatusPanel'
import { MoveToCollectionMenu } from '@/features/collections/MoveToCollectionMenu'
import { useCollectionLabel } from '@/features/collections/useCollectionLabel'
import { useResetImage } from '@/features/collections/useCollectionMutations'
import type { GalleryItem } from '@/features/gallery/gallery-item'
import { useImageDetail } from '@/features/inspector/useImageDetail'
import { resolveImageUrl } from '@/lib/api-client'
import { cn } from '@/lib/utils'
import type { ImageAnalysis } from '@/types/api'

/**
 * Cosine above which a neighbour is worth flagging rather than merely noting.
 * Mirrors the default `--duplicate-threshold` of scripts/analyze.py.
 */
const DUPLICATE_SIMILARITY = 0.95

/** Image beside captions, shared so the skeleton and the loaded state align. */
const INSPECTOR_LAYOUT = 'grid gap-6 md:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]'

/** Placeholder caption rows; Flickr8k supplies five references per image. */
const SKELETON_CAPTION_COUNT = 5

type ImageInspectorDialogProps = {
  /** The selected grid item, or `null` when the inspector is closed. */
  item: GalleryItem | null
  onClose: () => void
  /** Run a by-example search from here; also used to jump to a neighbour. */
  onFindSimilar: (imageId: string) => void
  /** Open another image in this same dialog. */
  onInspect: (imageId: string) => void
}

type NeighbourPanelProps = {
  analysis: ImageAnalysis
  onInspect: (imageId: string) => void
}

/**
 * The measurements from the offline data-quality pass.
 *
 * This is where a near-duplicate stops being a number in a report and becomes
 * one click away from the image it duplicates — which is the only form in which
 * a researcher can actually judge whether it is one.
 */
function NeighbourPanel({ analysis, onInspect }: NeighbourPanelProps): JSX.Element {
  const isDuplicate = analysis.nearest_neighbour_similarity >= DUPLICATE_SIMILARITY

  return (
    <div className="space-y-2 rounded-md border border-border p-3">
      <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
        Data quality
      </h3>

      <div className="flex flex-wrap items-center gap-2 text-sm">
        <Copy className="size-3.5 text-muted-foreground" aria-hidden="true" />
        <span className="text-muted-foreground">Nearest neighbour</span>
        <button
          type="button"
          onClick={() => onInspect(analysis.nearest_neighbour_id)}
          className="font-mono text-xs underline underline-offset-2 hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none"
        >
          {analysis.nearest_neighbour_id}
        </button>
        <Badge
          variant={isDuplicate ? 'destructive' : 'secondary'}
          className="font-mono tabular-nums"
        >
          {analysis.nearest_neighbour_similarity.toFixed(3)}
        </Badge>
      </div>

      {isDuplicate && (
        <p className="text-xs text-destructive">
          Above the {DUPLICATE_SIMILARITY} duplicate threshold — check whether these are the same
          photograph.
        </p>
      )}

      {analysis.caption_rank !== null && (
        <p className="text-xs text-muted-foreground">
          Its own captions rank it{' '}
          <span
            className={cn(
              'font-mono tabular-nums',
              analysis.caption_rank > 100 && 'text-foreground',
            )}
          >
            #{analysis.caption_rank.toLocaleString()}
          </span>{' '}
          in the corpus. A large number means CLIP does not read these captions as describing this
          image.
        </p>
      )}
    </div>
  )
}

/**
 * Full-size image with every reference caption.
 *
 * Built on the Radix-backed dialog primitive, which supplies the focus trap,
 * Escape handling, scroll lock and `aria-modal` wiring.
 */
export function ImageInspectorDialog({
  item,
  onClose,
  onFindSimilar,
  onInspect,
}: ImageInspectorDialogProps): JSX.Element {
  const { data, error, status } = useImageDetail(item?.id ?? null)
  const reset = useResetImage()
  const collectionLabel = useCollectionLabel()
  const detail = data?.image
  // `null` while the collection list loads, or once a collection is deleted.
  // The id stays the value every comparison and mutation below uses; only the
  // rendered text changes, so a pending lookup cannot break the move menu.
  const collectionName = detail === undefined ? null : collectionLabel(detail.collection)

  return (
    <Dialog open={item !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-5xl">
        <DialogHeader>
          <DialogTitle className="font-mono text-sm">{detail?.file_name ?? item?.id}</DialogTitle>
          <DialogDescription>
            {detail === undefined
              ? 'Loading reference captions…'
              : `${detail.caption_count} reference caption${detail.caption_count === 1 ? '' : 's'} · ${detail.split} split`}
          </DialogDescription>
        </DialogHeader>

        {status === 'error' && <ErrorNotice title="Could not load this image" error={error} />}

        {status === 'pending' && (
          <div className={INSPECTOR_LAYOUT}>
            <Skeleton className="aspect-4/3 w-full rounded-lg" />
            <div className="space-y-3">
              {Array.from({ length: SKELETON_CAPTION_COUNT }, (_, index) => (
                <Skeleton key={index} className="h-12 w-full rounded-md" />
              ))}
            </div>
          </div>
        )}

        {status === 'success' && (
          <div className={INSPECTOR_LAYOUT}>
            <img
              src={resolveImageUrl(data.image.image_url)}
              alt={data.image.captions[0] ?? `Flickr8k image ${data.image.id}`}
              className="max-h-[70vh] w-full rounded-lg border border-border object-contain"
            />

            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                {/*
                  When the collection differs from the split, BOTH are shown,
                  as `train → my-holdout`. That visible pairing is what stops a
                  re-partition from quietly corrupting someone's reading of the
                  leakage numbers: those are computed from the split, and the
                  split is still what it always was.
                */}
                {data.image.collection === data.image.split || collectionName === null ? (
                  <Badge variant="secondary">{data.image.split}</Badge>
                ) : (
                  <Badge
                    variant="secondary"
                    title="Dataset split → your collection. The split is unchanged."
                  >
                    {data.image.split} → {collectionName}
                  </Badge>
                )}
                {item?.score !== undefined && (
                  <Badge className="font-mono tabular-nums">
                    similarity {item.score.toFixed(4)}
                  </Badge>
                )}
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => onFindSimilar(data.image.id)}
                  title="Rank the corpus against this image. Costs no inference — its embedding is already indexed."
                >
                  <Search aria-hidden="true" />
                  Find similar
                </Button>
                <MoveToCollectionMenu
                  source={{ kind: 'ids', ids: [data.image.id] }}
                  count={1}
                  currentCollectionId={data.image.collection}
                />
                {data.image.collection !== data.image.split && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    disabled={reset.isPending}
                    onClick={() =>
                      reset.mutate({
                        collectionId: data.image.collection,
                        imageId: data.image.id,
                      })
                    }
                    title={`Drop the override and return this image to the ${data.image.split} split`}
                  >
                    {reset.isPending ? (
                      <Loader2 className="animate-spin" aria-hidden="true" />
                    ) : (
                      <Undo2 aria-hidden="true" />
                    )}
                    Return to {data.image.split}
                  </Button>
                )}
              </div>

              {data.analysis !== null && (
                <NeighbourPanel analysis={data.analysis} onInspect={onInspect} />
              )}

              <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                Reference captions
              </h3>

              {/* Numbered, because the captions are five independent annotations
                  of the same image and researchers refer to them by index. */}
              <ol className="space-y-2">
                {data.image.captions.map((caption, index) => (
                  <li
                    key={`${data.image.id}-caption-${index}`}
                    className="flex gap-3 rounded-md border border-border bg-muted/40 p-3 text-sm"
                  >
                    <span className="font-mono text-xs text-muted-foreground tabular-nums">
                      {index + 1}
                    </span>
                    <span className="leading-relaxed">{caption}</span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
