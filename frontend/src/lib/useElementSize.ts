import { useEffect, useState } from 'react'

/** An element's measured content box. */
export type ElementSize = { width: number; height: number }

/**
 * Track an element's content box.
 *
 * Shared rather than owned by a feature, because two now need it for unrelated
 * reasons: the projection canvas sizes its backing store from it, and the
 * gallery derives its column count and row height from it. Structurally
 * identical to the projection's own `CanvasSize`, so neither slice has to know
 * about the other's types.
 *
 * **Returns a callback ref, not a `useRef` object, and that is the whole point.**
 * The view this serves returns a skeleton while the projection loads, so the
 * measured element does not exist on first mount. An effect keyed on a ref
 * object would run once against `null`, never attach the observer, and never
 * re-run — leaving the canvas at its default 300×150 and blank. A callback ref
 * held in state changes identity the moment the node attaches, which re-runs
 * the effect exactly then.
 *
 * @returns The ref to attach, and the element's current size.
 */
export function useElementSize(): [(node: HTMLElement | null) => void, ElementSize] {
  const [node, setNode] = useState<HTMLElement | null>(null)
  const [size, setSize] = useState<ElementSize>({ width: 0, height: 0 })

  useEffect(() => {
    if (node === null) return

    const observer = new ResizeObserver(([entry]) => {
      if (entry === undefined) return
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(node)
    return () => observer.disconnect()
  }, [node])

  return [setNode, size]
}
