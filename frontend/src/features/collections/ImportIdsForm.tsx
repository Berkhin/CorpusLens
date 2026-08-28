import { useState, type ChangeEvent, type FormEvent, type JSX } from 'react'

import { Loader2, Upload } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { useMoveImages } from '@/features/collections/useCollectionMutations'
import {
  MAX_IMPORT_IDS,
  parseImageIdList,
  type ParsedImageIdList,
} from '@/features/collections/image-id-list'
import type { Collection } from '@/types/api'

type ImportIdsFormProps = {
  collections: readonly Collection[]
}

/** Rows the textarea shows before it scrolls. */
const TEXTAREA_ROWS = 3

/** How many rejected or unknown ids to name before trailing off. */
const NAMED_IN_REPORT = 3

function plural(count: number, singular: string, many = `${singular}s`): string {
  return count === 1 ? singular : many
}

/** Name the first few of a list, so the report is specific without being a wall. */
function summarise(values: readonly string[]): string {
  const named = values.slice(0, NAMED_IN_REPORT).join(', ')
  return values.length > NAMED_IN_REPORT ? `${named}…` : named
}

/**
 * Bring a list of image ids in from outside the tool.
 *
 * Images could only ever be added by selecting them *inside* this UI, and the
 * normal direction is the opposite: a training run emits 400 failure-case ids
 * and you want to look at them. Export existed; import did not.
 *
 * It posts through the ordinary move endpoint with `origin: 'import'`, so the
 * imported batch is recorded as such and nothing about the overlay semantics is
 * special-cased for it.
 *
 * **The report is the feature, not decoration.** `CollectionMoveResponse.unknown`
 * was already the right channel for ids that are not in this corpus, and
 * swallowing it would leave "37 of your 400 ids are not in this corpus" — the
 * single most useful thing to learn from an import — invisible. Tokens that
 * cannot be an image id at all are reported alongside it and never sent, since
 * one of them would otherwise 422 the whole batch with a message naming a field
 * index.
 */
export function ImportIdsForm({ collections }: ImportIdsFormProps): JSX.Element {
  const [text, setText] = useState('')
  const [target, setTarget] = useState('')
  const move = useMoveImages()

  const parsed: ParsedImageIdList = parseImageIdList(text)
  const destination = target === '' ? (collections[0]?.id ?? '') : target
  const overCap = parsed.ids.length > MAX_IMPORT_IDS

  const readFile = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = event.target.files?.[0]
    if (file === undefined) return
    setText(await file.text())
    // Clear the input so re-picking the same file fires `change` again.
    event.target.value = ''
  }

  const submit = (event: FormEvent): void => {
    event.preventDefault()
    if (parsed.ids.length === 0 || destination === '' || overCap) return
    move.mutate({
      collectionId: destination,
      source: { kind: 'ids', ids: parsed.ids, origin: 'import' },
    })
  }

  return (
    <form onSubmit={submit} className="space-y-2 rounded-md border border-border p-3">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
          Import ids
        </h3>
        <span className="text-xs text-muted-foreground">
          One per line, or comma-separated. A CSV header row is harmless — it comes back as an
          unknown id.
        </span>
      </div>

      <textarea
        value={text}
        onChange={(event) => setText(event.target.value)}
        rows={TEXTAREA_ROWS}
        placeholder="1000268201_693b08cb0e&#10;1001773457_577c3a7d70"
        aria-label="Image ids to import"
        className="w-full rounded-md border border-border bg-transparent px-2 py-1.5 font-mono text-xs focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none"
      />

      <div className="flex flex-wrap items-center gap-2">
        <input
          type="file"
          accept=".txt,.csv,.tsv,text/plain,text/csv"
          onChange={(event) => void readFile(event)}
          aria-label="Load image ids from a file"
          className="max-w-52 text-xs text-muted-foreground file:mr-2 file:rounded-md file:border file:border-border file:bg-transparent file:px-2 file:py-1 file:text-xs"
        />

        <label className="text-xs text-muted-foreground" htmlFor="import-destination">
          into
        </label>
        <select
          id="import-destination"
          value={destination}
          onChange={(event) => setTarget(event.target.value)}
          className="h-8 rounded-md border border-border bg-transparent px-2 text-xs focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none"
        >
          {collections.map((collection) => (
            <option key={collection.id} value={collection.id}>
              {collection.name}
            </option>
          ))}
        </select>

        <Button
          type="submit"
          size="sm"
          disabled={move.isPending || parsed.ids.length === 0 || overCap}
        >
          {move.isPending ? (
            <Loader2 className="animate-spin" aria-hidden="true" />
          ) : (
            <Upload aria-hidden="true" />
          )}
          Import {parsed.ids.length.toLocaleString()}
        </Button>
      </div>

      {overCap && (
        <p role="alert" className="text-xs text-destructive">
          {parsed.ids.length.toLocaleString()} ids is above the {MAX_IMPORT_IDS.toLocaleString()}
          -image limit for one move. Split the list.
        </p>
      )}

      {(parsed.malformed.length > 0 || parsed.duplicates > 0) && (
        <p className="text-xs text-muted-foreground">
          {parsed.duplicates > 0 &&
            `${parsed.duplicates.toLocaleString()} ${plural(parsed.duplicates, 'repeat')} collapsed. `}
          {parsed.malformed.length > 0 &&
            `${parsed.malformed.length.toLocaleString()} ${
              parsed.malformed.length === 1
                ? 'entry is not an image id'
                : 'entries are not image ids'
            } and will not be sent: ${summarise(parsed.malformed)}`}
        </p>
      )}

      {move.isError && (
        <p role="alert" className="text-xs text-destructive">
          {move.error.message}
        </p>
      )}

      {move.isSuccess && (
        <p className="text-xs">
          <span className="text-muted-foreground">
            {move.data.moved.toLocaleString()} moved, {move.data.unchanged.toLocaleString()} already
            there
          </span>
          {move.data.unknown.length > 0 && (
            <span className="text-destructive">
              {' · '}
              {move.data.unknown.length.toLocaleString()} not in this corpus:{' '}
              <span className="font-mono">{summarise(move.data.unknown)}</span>
            </span>
          )}
        </p>
      )}
    </form>
  )
}
