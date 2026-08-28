import type { JSX } from 'react'

import { Check } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { GalleryItem } from '@/features/gallery/gallery-item'

type ImageCardProps = {
  item: GalleryItem
  onSelect: (item: GalleryItem) => void
  /** Whether this card is in the multi-select set. */
  checked: boolean
  /** Toggle membership of the multi-select set. */
  onToggleChecked: (imageId: string) => void
}

/**
 * One clickable thumbnail in the grid, with a selection checkbox.
 *
 * The thumbnail is a real `<button>` rather than a click-handled `<div>`, so it
 * is focusable and activates on Enter/Space without re-implementing either
 * behaviour. The checkbox is therefore a **sibling** of that button, not a child
 * — nesting an interactive element inside a button is invalid, and browsers
 * disagree about which one a click reaches.
 *
 * It appears on hover (and on keyboard focus anywhere in the card) rather than
 * behind a selection *mode*. A mode is a state to enter, remember you are in,
 * and leave; checkbox-on-hover leaves the card's primary action — open the
 * inspector — exactly where it was, and puts the second action next to it. The
 * cost is that it is invisible until pointed at, which is why a checked card
 * keeps it visible and why the filter bar carries the bulk affordance for the
 * sets too large to click through.
 */
export function ImageCard({
  item,
  onSelect,
  checked,
  onToggleChecked,
}: ImageCardProps): JSX.Element {
  return (
    <div className="group relative">
      <button
        type="button"
        onClick={() => onSelect(item)}
        className={cn(
          'block w-full overflow-hidden rounded-lg border border-border bg-muted',
          'transition-all hover:border-foreground/25 hover:shadow-md',
          'focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none',
          checked && 'border-primary ring-2 ring-primary/40',
        )}
      >
        <img
          src={item.imageUrl}
          alt={item.alt}
          loading="lazy"
          decoding="async"
          className="aspect-4/3 w-full object-cover transition-transform duration-200 group-hover:scale-[1.03]"
        />

        {item.score !== undefined && (
          <Badge
            variant="default"
            className="absolute bottom-1.5 left-1.5 font-mono tabular-nums shadow-sm"
            title="Cosine similarity to the query"
          >
            {item.score.toFixed(3)}
          </Badge>
        )}

        {/*
          The *effective* collection, by name. Rendering `item.collection` here
          showed a built-in's split name correctly and a user collection's raw
          uuid4, which is a leak that looks right until someone creates their
          first collection. Nothing is rendered until the name resolves, rather
          than flashing the id.
        */}
        {item.collectionLabel !== null && (
          <Badge
            variant="secondary"
            className="absolute right-1.5 bottom-1.5 max-w-[60%] truncate opacity-90"
          >
            {item.collectionLabel}
          </Badge>
        )}

        {item.collection !== item.split && (
          <Badge
            variant="outline"
            className="absolute top-1.5 right-1.5 bg-background/80 opacity-90"
            title={`Moved out of the ${item.split} split`}
          >
            {item.split} →
          </Badge>
        )}
      </button>

      <button
        type="button"
        role="checkbox"
        aria-checked={checked}
        aria-label={`Select image ${item.id}`}
        onClick={() => onToggleChecked(item.id)}
        className={cn(
          'absolute top-1.5 left-1.5 z-10 grid size-6 place-items-center rounded-md border',
          'transition-opacity focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none',
          checked
            ? 'border-transparent bg-primary text-primary-foreground'
            : 'border-border bg-background/85 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100',
        )}
      >
        <Check className="size-3.5" aria-hidden="true" />
      </button>
    </div>
  )
}
