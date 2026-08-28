import type { JSX } from 'react'

import { AlertTriangle, Database, ShieldCheck } from 'lucide-react'

import { Skeleton } from '@/components/ui/skeleton'

import { useDatasetStats } from '@/features/stats/useDatasetStats'
import { cn } from '@/lib/utils'

type StatItemProps = {
  label: string
  value: number
}

/** One labelled count in the footer. */
function StatItem({ label, value }: StatItemProps): JSX.Element {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-xs tracking-wide text-muted-foreground uppercase">{label}</span>
      <span className="font-mono text-sm font-medium tabular-nums">{value.toLocaleString()}</span>
    </div>
  )
}

type LeakageItemProps = {
  label: string
  pairs: number
  title: string
}

/**
 * One of the two leakage readings.
 *
 * Rendered in the destructive colour only when non-zero: a quarantine that
 * worked should *look* like it worked, and a red zero would undercut the one
 * piece of feedback the whole feature exists to give.
 */
function LeakageItem({ label, pairs, title }: LeakageItemProps): JSX.Element {
  return (
    <div
      className={cn('flex items-center gap-2', pairs > 0 ? 'text-destructive' : 'text-emerald-600')}
      title={title}
    >
      {pairs > 0 ? (
        <AlertTriangle className="size-3.5" aria-hidden="true" />
      ) : (
        <ShieldCheck className="size-3.5" aria-hidden="true" />
      )}
      <span className="text-xs tracking-wide uppercase">{label}</span>
      <span className="font-mono text-sm font-medium tabular-nums">
        {pairs.toLocaleString()} pairs
      </span>
    </div>
  )
}

/**
 * Corpus totals and per-split breakdown.
 *
 * A failure here is deliberately quiet: the stats are contextual, and blocking
 * the gallery behind a broken counter would be the wrong trade.
 */
export function DatasetStatsBar(): JSX.Element | null {
  const { data, status } = useDatasetStats()

  if (status === 'error') return null

  return (
    <div className="flex flex-wrap items-center gap-x-8 gap-y-2">
      <div className="flex items-center gap-2 text-muted-foreground">
        <Database className="size-4" aria-hidden="true" />
        <span className="text-xs font-medium tracking-wide uppercase">Corpus</span>
      </div>

      {status === 'pending' ? (
        <Skeleton className="h-5 w-64" />
      ) : (
        <>
          <StatItem label="Total images" value={data.total_images} />
          {Object.entries(data.images_by_split).map(([split, count]) => (
            <StatItem key={split} label={split} value={count} />
          ))}

          {data.caption_retrieval !== null && (
            <div
              className="flex items-baseline gap-2"
              title={`Fraction of the corpus's own ${data.caption_retrieval.captions.toLocaleString()} captions that rank their image first. The standard text-to-image retrieval metric, measured on this dataset with this model.`}
            >
              <span className="text-xs tracking-wide text-muted-foreground uppercase">
                Caption R@1
              </span>
              <span className="font-mono text-sm font-medium tabular-nums">
                {(data.caption_retrieval.recall_at_1 * 100).toFixed(1)}%
              </span>
            </div>
          )}

          {/*
            Two readings of one finding, side by side, and the pairing is the
            point. SPLIT LEAKAGE is computed from the dataset's own immutable
            partition and must never move — it is what every offline
            measurement is derived from. COLLECTION LEAKAGE is the same pairs
            counted against the partition the user is building, so quarantining
            both sides of a leaking pair drives it to zero while the first holds.
            Showing only the first lets someone act on a finding and never see
            the effect; showing only the second would quietly redefine what
            "test set" means.

            Both are rendered even at zero, unlike before: a figure that
            disappears when it reaches the value you were working towards is a
            figure you cannot use as feedback.
          */}
          {data.cross_split_duplicate_pairs !== null && (
            <LeakageItem
              label="Split leakage"
              pairs={data.cross_split_duplicate_pairs}
              title="Near-duplicate pairs whose two images sit in different dataset splits. Evaluating on those test images measures memorisation, not generalisation. Derived from the immutable split column — collection moves never change it."
            />
          )}
          {data.cross_collection_duplicate_pairs !== null && (
            <LeakageItem
              label="Collection leakage"
              pairs={data.cross_collection_duplicate_pairs}
              title="The same near-duplicate pairs, counted against your collections instead of the dataset splits. Move both sides of a pair into one collection and this falls; separate them and it rises."
            />
          )}
        </>
      )}
    </div>
  )
}
