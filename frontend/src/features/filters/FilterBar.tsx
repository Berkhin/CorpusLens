import { useEffect, useState, type JSX } from 'react'

import { AlertTriangle, Copy, FileWarning, FilterX, Settings2, Tag } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { CollectionChips } from '@/features/collections/CollectionChips'
import { CollectionManagerDialog } from '@/features/collections/CollectionManagerDialog'
import { MoveToCollectionMenu } from '@/features/collections/MoveToCollectionMenu'
import { Chip } from '@/features/filters/Chip'
import { useDebouncedValue } from '@/lib/useDebouncedValue'
import { useFilteredTotal } from '@/features/gallery/useImageList'
import {
  EMPTY_IMAGE_FILTER,
  MAX_CAPTION_FILTER_LENGTH,
  isFilterActive,
  toggleQualityFlag,
  type ImageFilter,
} from '@/features/filters/image-filter'
import { useDatasetStats } from '@/features/stats/useDatasetStats'
import type { QualityFlag } from '@/types/api'

/** Quiet period before a caption keystroke becomes a request. */
const CAPTION_DEBOUNCE_MS = 300

/**
 * The data-quality findings, as controls.
 *
 * Each is a *finding* rather than a property — the offline pass computed it —
 * but presenting them here, beside the collection and caption filters, is what
 * makes them reachable through the grid, the map and the export instead of
 * needing a report view of their own.
 */
const QUALITY_FLAGS: readonly {
  flag: QualityFlag
  label: string
  icon: typeof Copy
  title: string
}[] = [
  {
    flag: 'near-duplicate',
    label: 'Near-duplicates',
    icon: Copy,
    title: 'Images with another image above the duplicate threshold in CLIP space',
  },
  {
    flag: 'cross-split-duplicate',
    label: 'Split leakage',
    icon: AlertTriangle,
    title:
      'Near-duplicates that sit in different splits — evaluating on these measures memorisation',
  },
  {
    flag: 'weak-captions',
    label: 'Weak captions',
    icon: FileWarning,
    title: 'Images their own captions retrieve worst — a review queue for the annotations',
  },
]

type FilterBarProps = {
  filter: ImageFilter
  onChange: (filter: ImageFilter) => void
}

/**
 * Corpus filters, applied to both the browse grid and the ranked search.
 *
 * The **Collection** group replaces what used to be a Split group. The three
 * built-in collections *are* the splits, so nothing is lost — but the sizes now
 * come from `GET /api/collections` and therefore follow the user's moves, which
 * `images_by_split` deliberately does not.
 *
 * Two different combination rules live in this bar, and keeping them in
 * separate visual groups is load-bearing: selections *within* the collection
 * group union with each other, while the group as a whole intersects the
 * caption and quality filters. Merging collection chips into a shared row with
 * the quality flags would put controls with different semantics into one group,
 * which is how a filter bar starts lying about what it is showing.
 *
 * The caption field keeps a local draft and lifts it on a debounce, because
 * every commit costs the backend a filtered scan plus a count. That draft is
 * the reason "Clear filters" lives here and not in the parent: clearing from
 * outside would leave stale text in the box, which the next debounce would
 * helpfully push straight back up.
 *
 * The quality row appears only when the offline analysis exists. A control that
 * can only ever return nothing is worse than no control.
 */
export function FilterBar({ filter, onChange }: FilterBarProps): JSX.Element {
  const { data, status } = useDatasetStats()
  const [captionDraft, setCaptionDraft] = useState(filter.captionContains)
  const [managerOpen, setManagerOpen] = useState(false)
  const debouncedCaption = useDebouncedValue(captionDraft.trim(), CAPTION_DEBOUNCE_MS)
  const matching = useFilteredTotal(filter)
  const filtered = isFilterActive(filter)

  useEffect(() => {
    // Guarded rather than unconditional: `filter` is a fresh object on every
    // parent render, so without this the effect would republish on each one.
    if (debouncedCaption === filter.captionContains) return
    onChange({ ...filter, captionContains: debouncedCaption })
  }, [debouncedCaption, filter, onChange])

  const handleClear = (): void => {
    setCaptionDraft('')
    onChange(EMPTY_IMAGE_FILTER)
  }

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
          Collection
        </span>
        <CollectionChips filter={filter} onChange={onChange} />
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          onClick={() => setManagerOpen(true)}
          title="Create, rename and delete collections"
        >
          <Settings2 aria-hidden="true" />
          <span className="sr-only">Manage collections</span>
        </Button>
      </div>

      <div className="relative min-w-56 flex-1">
        <Tag
          className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground"
          aria-hidden="true"
        />
        <Input
          type="search"
          value={captionDraft}
          onChange={(event) => setCaptionDraft(event.target.value)}
          maxLength={MAX_CAPTION_FILTER_LENGTH}
          placeholder="Caption contains… (literal text, not semantic)"
          aria-label="Filter by caption text"
          className="h-8 pl-8"
        />
      </div>

      {status === 'success' && data.analysis_available && (
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
            Quality
          </span>
          <div className="flex flex-wrap gap-1.5">
            {QUALITY_FLAGS.map(({ flag, label, icon: Icon, title }) => (
              <Chip
                key={flag}
                selected={filter.qualityFlag === flag}
                onToggle={() => onChange(toggleQualityFlag(filter, flag))}
                title={title}
              >
                <Icon className="size-3" aria-hidden="true" />
                {label}
              </Chip>
            ))}
          </div>
        </div>
      )}

      {filtered && (
        <div className="flex items-center gap-2">
          {/*
            The set you are looking at is the set you can move, in one action.
            The count comes from the same `total` the grid renders — the same
            query, the same cache entry — because a control that promises a
            different number from the one on screen is worse than none.

            Shown only when a filter is active. Unfiltered, this would be a
            one-click "move all 8 000 images", which is legal, occasionally
            wanted, and not something to leave sitting in a toolbar aimed at
            nobody.

            "matching the filter", spelled out, because a search can be running
            at the same time: the ranked view then shows twenty hits while this
            still addresses everything the *filter* selects. The move endpoint
            has no ranking channel, and a label reading "Move 2 014…" beside
            twenty results would be read as being about the twenty.
          */}
          <MoveToCollectionMenu
            source={{ kind: 'filter', filter }}
            count={matching ?? 0}
            label={`Move ${(matching ?? 0).toLocaleString()} matching the filter…`}
          />
          <Button type="button" variant="ghost" size="sm" onClick={handleClear}>
            <FilterX aria-hidden="true" />
            Clear filters
          </Button>
        </div>
      )}

      <CollectionManagerDialog
        open={managerOpen}
        onOpenChange={setManagerOpen}
        filter={filter}
        onChange={onChange}
      />
    </div>
  )
}
