import { useState, type FormEvent, type JSX } from 'react'

import { Check, Loader2, Pencil, Trash2, X } from 'lucide-react'

import { ErrorNotice } from '@/components/StatusPanel'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Skeleton } from '@/components/ui/skeleton'
import {
  useCreateCollection,
  useDeleteCollection,
  useRenameCollection,
} from '@/features/collections/useCollectionMutations'
import { useCollections } from '@/features/collections/useCollections'
import { withoutCollection, type ImageFilter } from '@/features/filters/image-filter'
import { useDatasetStats } from '@/features/stats/useDatasetStats'
import { MAX_COLLECTION_NAME_LENGTH } from '@/lib/api-client'
import { ImportIdsForm } from '@/features/collections/ImportIdsForm'
import type { Collection, CollectionCaptionRecall, CollectionProvenance } from '@/types/api'

type CollectionManagerDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  filter: ImageFilter
  onChange: (filter: ImageFilter) => void
}

/**
 * Spell out one collection's provenance for its tooltip.
 *
 * The `filter` case is the one that earns the field: "32 images, because of the
 * cross-split-duplicate flag" can be re-derived three weeks later, where "32
 * images" cannot. The stored detail is compact JSON, shown verbatim — it is the
 * request that produced the set, and prettifying it into prose would lose the
 * property that you can paste it back.
 */
function describeProvenance(provenance: CollectionProvenance): string {
  const when = `Last changed ${new Date(provenance.moved_at).toLocaleString()}.`
  if (provenance.origin === 'filter') {
    return `Populated from a filter: ${provenance.detail ?? '{}'}. ${when}`
  }
  if (provenance.origin === 'import') {
    return `Populated from an imported list of ids. ${when}`
  }
  return `Images were picked by hand or selected on the map. ${when}`
}

type CollectionRowProps = {
  collection: Collection
  /** Caption recall over this collection's images, when it was measured. */
  recall: CollectionCaptionRecall | undefined
  onDelete: (collectionId: string) => void
  deleting: boolean
}

/**
 * One collection, with its edit controls and its caption-recall figure.
 *
 * Built-ins render without the edit controls rather than with disabled ones:
 * they are not a capability the user is temporarily lacking, they are a
 * different kind of thing — the dataset's own partition, which the overlay has
 * no authority over.
 *
 * The recall figure is the reason this dialog is worth opening after a move
 * rather than only before one. Its label says "median rank vs. full corpus"
 * because that is what the number is: a re-aggregation of ranks `analyze.py`
 * computed against all 8 000 images, not a ranking re-run inside the
 * collection. Calling it plain "R@1" beside the footer's corpus figure would
 * invite comparing two numbers with different denominators.
 */
function CollectionRow({
  collection,
  recall,
  onDelete,
  deleting,
}: CollectionRowProps): JSX.Element {
  const [draft, setDraft] = useState<string | null>(null)
  const rename = useRenameCollection()

  const submitRename = (event: FormEvent): void => {
    event.preventDefault()
    if (draft === null || draft.trim().length === 0) return
    rename.mutate(
      { collectionId: collection.id, name: draft.trim() },
      { onSuccess: () => setDraft(null) },
    )
  }

  return (
    <li className="flex items-center gap-2 border-b border-border py-2 last:border-b-0">
      {draft === null ? (
        <>
          <span className="flex-1 truncate text-sm">{collection.name}</span>
          {recall !== undefined && (
            <span
              className="font-mono text-xs tabular-nums text-muted-foreground"
              title={`${(recall.recall_at_1 * 100).toFixed(1)}% of this collection's ${recall.images.toLocaleString()} measured images are retrieved first by their own captions, ranked against the whole 8 000-image corpus (R@5 ${(recall.recall_at_5 * 100).toFixed(1)}%, R@10 ${(recall.recall_at_10 * 100).toFixed(1)}%). Each image contributes the median rank of its five captions, so this is not comparable with the corpus caption R@1 in the footer, which counts captions.`}
            >
              R@1 {(recall.recall_at_1 * 100).toFixed(1)}%
            </span>
          )}
          <span className="font-mono text-xs tabular-nums text-muted-foreground">
            {collection.size.toLocaleString()}
          </span>
          {collection.provenance !== null && (
            <span
              className="rounded-full border border-border px-2 py-0.5 text-xs text-muted-foreground"
              title={describeProvenance(collection.provenance)}
            >
              {collection.provenance.origin}
            </span>
          )}
          {collection.kind === 'builtin' ? (
            <span className="text-xs text-muted-foreground">dataset split</span>
          ) : (
            <>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                onClick={() => setDraft(collection.name)}
                title={`Rename ${collection.name}`}
              >
                <Pencil aria-hidden="true" />
                <span className="sr-only">Rename {collection.name}</span>
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                onClick={() => onDelete(collection.id)}
                disabled={deleting}
                title={`Delete ${collection.name}. Its images return to their dataset splits.`}
              >
                <Trash2 aria-hidden="true" />
                <span className="sr-only">Delete {collection.name}</span>
              </Button>
            </>
          )}
        </>
      ) : (
        <form onSubmit={submitRename} className="flex flex-1 items-center gap-2">
          <Input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            maxLength={MAX_COLLECTION_NAME_LENGTH}
            aria-label={`New name for ${collection.name}`}
            className="h-8"
            autoFocus
          />
          <Button type="submit" variant="ghost" size="icon-sm" disabled={rename.isPending}>
            {rename.isPending ? (
              <Loader2 className="animate-spin" aria-hidden="true" />
            ) : (
              <Check aria-hidden="true" />
            )}
            <span className="sr-only">Save name</span>
          </Button>
          <Button type="button" variant="ghost" size="icon-sm" onClick={() => setDraft(null)}>
            <X aria-hidden="true" />
            <span className="sr-only">Cancel rename</span>
          </Button>
        </form>
      )}

      {rename.isError && (
        <span role="alert" className="text-xs text-destructive">
          {rename.error.message}
        </span>
      )}
    </li>
  )
}

/**
 * Create, rename and delete collections.
 *
 * Deleting one that is currently in the active filter drops it from the filter
 * here, at the mutation site, rather than through a `useEffect` watching for
 * ids that stopped resolving. This component already knows exactly which id
 * disappeared; reacting to that fact afterwards would be inferring something it
 * was told.
 */
export function CollectionManagerDialog({
  open,
  onOpenChange,
  filter,
  onChange,
}: CollectionManagerDialogProps): JSX.Element {
  const { data, status, error } = useCollections()
  const { data: stats } = useDatasetStats()
  const [name, setName] = useState('')
  const create = useCreateCollection()
  const remove = useDeleteCollection()

  const submitCreate = (event: FormEvent): void => {
    event.preventDefault()
    if (name.trim().length === 0) return
    create.mutate(name.trim(), { onSuccess: () => setName('') })
  }

  const handleDelete = (collectionId: string): void => {
    remove.mutate(collectionId, {
      onSuccess: () => onChange(withoutCollection(filter, collectionId)),
    })
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Collections</DialogTitle>
          <DialogDescription>
            Your own partition of the corpus. Moving an image never changes its dataset split — the
            two are shown side by side wherever they differ, so the duplicate-leakage figures keep
            meaning what they say.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={submitCreate} className="flex items-center gap-2">
          <Input
            value={name}
            onChange={(event) => setName(event.target.value)}
            maxLength={MAX_COLLECTION_NAME_LENGTH}
            placeholder="New collection name…"
            aria-label="New collection name"
            className="h-8"
          />
          <Button type="submit" size="sm" disabled={create.isPending || name.trim().length === 0}>
            {create.isPending && <Loader2 className="animate-spin" aria-hidden="true" />}
            Create
          </Button>
        </form>

        {create.isError && (
          <span role="alert" className="text-xs text-destructive">
            {create.error.message}
          </span>
        )}
        {remove.isError && (
          <span role="alert" className="text-xs text-destructive">
            {remove.error.message}
          </span>
        )}

        {status === 'pending' && <Skeleton className="h-32 w-full" />}
        {status === 'error' && <ErrorNotice title="Could not load collections" error={error} />}
        {status === 'success' && (
          <>
            <ul className="max-h-64 overflow-y-auto">
              {data.map((collection) => (
                <CollectionRow
                  key={collection.id}
                  collection={collection}
                  recall={stats?.caption_recall_by_collection?.[collection.id]}
                  onDelete={handleDelete}
                  deleting={remove.isPending}
                />
              ))}
            </ul>
            <ImportIdsForm collections={data} />
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}
