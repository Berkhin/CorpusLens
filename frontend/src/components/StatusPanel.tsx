import type { JSX, ReactNode } from 'react'

import { AlertCircle } from 'lucide-react'

import { cn } from '@/lib/utils'

type ErrorNoticeProps = {
  title: string
  error: Error
  className?: string
}

/**
 * Failure notice.
 *
 * `role="alert"` so a fetch that fails after the page has settled is announced
 * rather than appearing silently.
 */
export function ErrorNotice({ title, error, className }: ErrorNoticeProps): JSX.Element {
  return (
    <div
      role="alert"
      className={cn(
        'flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-4',
        className,
      )}
    >
      <AlertCircle className="mt-0.5 size-4 shrink-0 text-destructive" />
      <div className="space-y-1">
        <p className="text-sm font-medium text-destructive">{title}</p>
        <p className="text-sm text-muted-foreground">{error.message}</p>
      </div>
    </div>
  )
}

type EmptyStateProps = {
  icon: ReactNode
  title: string
  hint?: string
}

/** Neutral placeholder for a query that succeeded with nothing to show. */
export function EmptyState({ icon, title, hint }: EmptyStateProps): JSX.Element {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border py-16 text-center">
      <div className="text-muted-foreground">{icon}</div>
      <p className="text-sm font-medium">{title}</p>
      {hint !== undefined && <p className="max-w-md text-sm text-muted-foreground">{hint}</p>}
    </div>
  )
}
