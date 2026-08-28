import { useEffect, useState } from 'react'

import type { CanvasSize } from '@/features/projection/scatter-viewport'

/**
 * Track an element's content box.
 *
 * The canvas needs it to size its backing store, and the hover card needs it to
 * know when to flip to the other side of the cursor. Measuring once, here, and
 * passing the result down keeps those two from disagreeing during a resize.
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
export function useElementSize(): [(node: HTMLElement | null) => void, CanvasSize] {
  const [node, setNode] = useState<HTMLElement | null>(null)
  const [size, setSize] = useState<CanvasSize>({ width: 0, height: 0 })

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
