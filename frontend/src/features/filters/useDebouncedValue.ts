import { useEffect, useState } from 'react'

/**
 * Follow a value, but only after it has stopped changing for `delayMs`.
 *
 * Used for the caption filter: typing "dog" would otherwise commit `d`, `do`
 * and `dog`, and every commit is a filtered scan plus a second count on the
 * backend.
 *
 * This is one of the `useEffect` cases CLAUDE.md §5.2 does allow — the external
 * system being synchronised with is the timer, and no data is fetched here. The
 * cleanup is what makes it correct: each keystroke cancels the pending commit
 * rather than stacking another one behind it.
 *
 * @param value The value to trail.
 * @param delayMs Quiet period before the trailing value catches up.
 */
export function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [settled, setSettled] = useState(value)

  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), delayMs)
    return () => clearTimeout(timer)
  }, [value, delayMs])

  return settled
}
