import type { JSX } from 'react'

import { Download, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { useExport, type ExportScope } from '@/features/export/useExport'

type ExportButtonProps = {
  scope: ExportScope
}

/**
 * Download whatever the user is currently looking at.
 *
 * Two buttons rather than a format dropdown: there are exactly two formats,
 * and a menu to choose between two things costs the user a click and the app a
 * primitive. CSV leads because it is the one that opens in a spreadsheet.
 *
 * The label names the scope, because "Export" alone leaves the user guessing
 * whether they are about to download three images or eight thousand.
 */
export function ExportButton({ scope }: ExportButtonProps): JSX.Element {
  const { mutate, isPending, isError, error } = useExport(scope)

  const selectionSize = scope.ids?.length ?? 0
  const label =
    selectionSize > 0
      ? `Export ${selectionSize.toLocaleString()} selected`
      : scope.query !== undefined || scope.similarToImageId !== undefined
        ? 'Export results'
        : 'Export view'

  return (
    <div className="flex items-center gap-1">
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => mutate('csv')}
        disabled={isPending}
        title="Download as CSV — captions in fixed columns, for spreadsheets and pandas"
      >
        {isPending ? <Loader2 className="animate-spin" /> : <Download aria-hidden="true" />}
        {label}
      </Button>

      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={() => mutate('jsonl')}
        disabled={isPending}
        title="Download as JSONL — captions kept as a list, lossless"
      >
        JSONL
      </Button>

      {isError && (
        <span role="alert" className="text-xs text-destructive">
          {error.message}
        </span>
      )}
    </div>
  )
}
