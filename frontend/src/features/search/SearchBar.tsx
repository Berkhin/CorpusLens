import { useState, type FormEvent, type JSX } from 'react'

import { Loader2, Search, X } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
// Enforced here so an over-long query never 422s; the bound itself is mirrored
// from the backend in one place.
import { MAX_QUERY_LENGTH } from '@/lib/api-client'

type SearchBarProps = {
  /** Query currently driving the results, `''` when browsing. */
  activeQuery: string
  onSubmit: (query: string) => void
  onClear: () => void
  isSearching: boolean
}

/**
 * Semantic search input.
 *
 * The field is local state and only lifts on submit: each search costs a CLIP
 * text-encoder forward pass on the CPU, so searching per keystroke would queue
 * work the user never asked for.
 */
export function SearchBar({
  activeQuery,
  onSubmit,
  onClear,
  isSearching,
}: SearchBarProps): JSX.Element {
  const [draft, setDraft] = useState(activeQuery)

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault()
    const trimmed = draft.trim()
    if (trimmed.length > 0) onSubmit(trimmed)
  }

  const handleClear = (): void => {
    setDraft('')
    onClear()
  }

  return (
    <form onSubmit={handleSubmit} role="search" className="flex w-full items-center gap-2">
      <div className="relative flex-1">
        <Search
          className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground"
          aria-hidden="true"
        />
        <Input
          type="search"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          maxLength={MAX_QUERY_LENGTH}
          placeholder="Describe an image — e.g. “a dog running through shallow water”"
          aria-label="Semantic search query"
          className="h-10 pr-9 pl-9"
        />
        {draft.length > 0 && (
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            onClick={handleClear}
            className="absolute top-1/2 right-1.5 -translate-y-1/2"
          >
            <X aria-hidden="true" />
            <span className="sr-only">Clear search</span>
          </Button>
        )}
      </div>

      <Button type="submit" size="lg" disabled={draft.trim().length === 0 || isSearching}>
        {isSearching ? <Loader2 className="animate-spin" /> : <Search aria-hidden="true" />}
        {isSearching ? 'Searching…' : 'Search'}
      </Button>
    </form>
  )
}
