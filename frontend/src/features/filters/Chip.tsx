import type { JSX, ReactNode } from 'react'

import { cn } from '@/lib/utils'

type ChipProps = {
  selected: boolean
  onToggle: () => void
  title?: string
  children: ReactNode
}

/**
 * Shared toggle affordance for the filter bar.
 *
 * Lifted out of `FilterBar` so the collection chips can be their own component
 * without cloning the styling: a collection, a split and a quality finding must
 * read as the same *kind* of control, because they are all "click to narrow".
 * How they combine differs — see `FilterBar` — but that is communicated by
 * grouping, not by making them look like different widgets.
 */
export function Chip({ selected, onToggle, title, children }: ChipProps): JSX.Element {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-pressed={selected}
      title={title}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-colors',
        'focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none',
        selected
          ? 'border-transparent bg-primary text-primary-foreground'
          : 'border-border text-muted-foreground hover:border-foreground/25 hover:text-foreground',
      )}
    >
      {children}
    </button>
  )
}
