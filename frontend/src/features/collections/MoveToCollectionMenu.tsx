import { useState, type JSX } from 'react'

import { Check, FolderInput, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { useCollections } from '@/features/collections/useCollections'
import { useMoveImages } from '@/features/collections/useCollectionMutations'
import { cn } from '@/lib/utils'
import type { CollectionMoveSource } from '@/lib/api-client'

type MoveToCollectionMenuProps = {
  /** Which images to move: an explicit id list, or the filter that selects them. */
  source: CollectionMoveSource
  /**
   * How many images the move addresses. Stated on the control *before* it acts,
   * because the filter channel can move thousands of images the user cannot see
   * on screen. Zero renders nothing.
   */
  count: number
  /** Button text. Defaults to "Move to…", which suits a stated count beside it. */
  label?: string
  /** Effective collection of the selection, when they all share one. */
  currentCollectionId?: string
  /** Called after a successful move, e.g. to clear a map selection. */
  onMoved?: () => void
}

/**
 * Reassign a set of images to a collection.
 *
 * A disclosure button plus a plain list rather than a `DropdownMenu`: the menu
 * primitive is not vendored, and this needs no keyboard-navigable roving focus
 * beyond what a list of buttons already gives. Adding a Radix dependency for
 * one control would be scope the feature does not need.
 *
 * The destination list includes the built-ins, because moving *back* to a split
 * is how a re-partition is undone — the backend clears the override rather than
 * storing a redundant one.
 *
 * The `source` is a union rather than an id array so this one control serves all
 * four surfaces that need it — the map's rectangle, the grid's multi-select, the
 * inspector's single image, and the filter bar's "everything matching" — without
 * the two that are filter-shaped having to materialise thousands of ids in the
 * browser first.
 */
export function MoveToCollectionMenu({
  source,
  count,
  label = 'Move to…',
  currentCollectionId,
  onMoved,
}: MoveToCollectionMenuProps): JSX.Element | null {
  const [open, setOpen] = useState(false)
  const { data: collections } = useCollections()
  const move = useMoveImages()

  if (count === 0) return null

  const handleMove = (collectionId: string): void => {
    move.mutate(
      { collectionId, source },
      {
        onSuccess: () => {
          setOpen(false)
          onMoved?.()
        },
      },
    )
  }

  return (
    <div className="relative">
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => setOpen((current) => !current)}
        disabled={move.isPending}
        aria-expanded={open}
        title={`Reassign ${count.toLocaleString()} image(s) to a collection. The dataset split is not changed.`}
      >
        {move.isPending ? (
          <Loader2 className="animate-spin" aria-hidden="true" />
        ) : (
          <FolderInput aria-hidden="true" />
        )}
        {label}
      </Button>

      {open && collections !== undefined && (
        <div
          role="menu"
          className="absolute right-0 z-20 mt-1 max-h-64 w-56 overflow-y-auto rounded-md border border-border bg-popover p-1 shadow-md"
        >
          {collections.map((collection) => {
            const isCurrent = collection.id === currentCollectionId
            return (
              <button
                key={collection.id}
                type="button"
                role="menuitem"
                onClick={() => handleMove(collection.id)}
                disabled={isCurrent}
                className={cn(
                  'flex w-full items-center justify-between gap-2 rounded-sm px-2 py-1.5 text-left text-xs',
                  isCurrent
                    ? 'text-muted-foreground'
                    : 'hover:bg-accent hover:text-accent-foreground',
                )}
              >
                <span className="truncate">{collection.name}</span>
                {isCurrent ? (
                  <Check className="size-3 shrink-0" aria-hidden="true" />
                ) : (
                  <span className="shrink-0 font-mono tabular-nums opacity-60">
                    {collection.size.toLocaleString()}
                  </span>
                )}
              </button>
            )
          })}
        </div>
      )}

      {move.isError && (
        <span role="alert" className="ml-2 text-xs text-destructive">
          {move.error.message}
        </span>
      )}
    </div>
  )
}
