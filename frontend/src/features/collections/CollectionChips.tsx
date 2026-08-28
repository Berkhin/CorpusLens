import type { JSX } from 'react'

import { FolderPlus } from 'lucide-react'

import { Skeleton } from '@/components/ui/skeleton'
import { useCollections } from '@/features/collections/useCollections'
import { Chip } from '@/features/filters/Chip'
import { toggleCollection, type ImageFilter } from '@/features/filters/image-filter'

type CollectionChipsProps = {
  filter: ImageFilter
  onChange: (filter: ImageFilter) => void
}

/**
 * The corpus partition, as toggles.
 *
 * Built-ins come first and carry no icon; user collections carry one, so the
 * two kinds are distinguishable without looking like different controls. The
 * distinction matters because only one kind can be renamed or deleted — and
 * because a built-in is ground truth from the dataset while a user collection
 * is something someone decided.
 *
 * Sizes come from `GET /api/collections`, not from `images_by_split`: these
 * have to follow moves, and that one deliberately does not.
 */
export function CollectionChips({ filter, onChange }: CollectionChipsProps): JSX.Element {
  const { data, status } = useCollections()

  if (status !== 'success') return <Skeleton className="h-6 w-40" />

  return (
    <div className="flex flex-wrap gap-1.5">
      {data.map((collection) => (
        <Chip
          key={collection.id}
          selected={filter.collections.includes(collection.id)}
          onToggle={() => onChange(toggleCollection(filter, collection.id))}
          title={
            collection.kind === 'builtin'
              ? `Images whose ground-truth split is "${collection.id}", plus anything moved into it`
              : 'A collection you created'
          }
        >
          {collection.kind === 'user' && <FolderPlus className="size-3" aria-hidden="true" />}
          {collection.name}
          <span className="font-mono tabular-nums opacity-70">
            {collection.size.toLocaleString()}
          </span>
        </Chip>
      ))}
    </div>
  )
}
