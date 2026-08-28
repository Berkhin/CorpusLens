import { useEffect, useMemo, useRef, useState, type JSX, type PointerEvent } from 'react'

import { assignSplitColours, readScatterPalette } from '@/features/projection/scatter-palette'
import { drawScatter } from '@/features/projection/scatter-render'
import {
  IDENTITY_VIEWPORT,
  findNearestPoint,
  panBy,
  pointsInBox,
  zoomAtPoint,
  type CanvasSize,
  type ScreenBox,
  type ScreenPoint,
  type Viewport,
} from '@/features/projection/scatter-viewport'
import { cn } from '@/lib/utils'
import type { ProjectionPoint } from '@/types/api'

/** Wheel notch to zoom factor. One notch is about 10%. */
const ZOOM_PER_WHEEL_UNIT = 0.0015

/** A drag shorter than this is treated as a click, not a pan. */
const CLICK_SLOP_PX = 4

type ScatterCanvasProps = {
  points: readonly ProjectionPoint[]
  positions: Float32Array
  /** Measured by the parent, so the canvas and the hover card agree on it. */
  size: CanvasSize
  /** Ids of the current search hits, drawn on top of everything else. */
  highlightedIds: ReadonlySet<string>
  selectedIndices: ReadonlySet<number>
  onSelectionChange: (indices: number[]) => void
  onHoverChange: (point: ProjectionPoint | null, at: ScreenPoint | null) => void
  onActivate: (point: ProjectionPoint) => void
}

type DragState =
  | { kind: 'none' }
  | { kind: 'pan'; last: ScreenPoint; travelled: number }
  | { kind: 'select'; box: ScreenBox }

/**
 * The embedding map itself: 8 000 points, pan, zoom, hover and box selection.
 *
 * Hand-rolled on a 2-D canvas rather than a plotting library. At this scale a
 * canvas is not a compromise — 8 000 dots redraw in a few milliseconds — and it
 * keeps the dependency list where CLAUDE.md §3 left it while giving exact
 * control over the two behaviours that matter here: dimming filtered-out points
 * and rectangle-selecting for export.
 *
 * **Selection is a rectangle, not a freehand lasso.** A lasso would be nicer
 * and roughly three times the code for a hit test that is only marginally more
 * expressive on a cloud this diffuse.
 */
export function ScatterCanvas({
  points,
  positions,
  size,
  highlightedIds,
  selectedIndices,
  onSelectionChange,
  onHoverChange,
  onActivate,
}: ScatterCanvasProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [viewport, setViewport] = useState<Viewport>(IDENTITY_VIEWPORT)
  const [drag, setDrag] = useState<DragState>({ kind: 'none' })
  const [hovered, setHovered] = useState(-1)

  const splits = useMemo(() => points.map((point) => point.split), [points])
  const matches = useMemo(() => points.map((point) => point.matches), [points])

  const highlighted = useMemo(() => {
    const indices = new Set<number>()
    if (highlightedIds.size === 0) return indices
    points.forEach((point, index) => {
      if (highlightedIds.has(point.id)) indices.add(index)
    })
    return indices
  }, [points, highlightedIds])

  // Repaint whenever anything visible changes. This is a genuine external
  // system — a canvas is not described by JSX, it is commanded.
  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas === null || size.width === 0) return

    const ratio = window.devicePixelRatio || 1
    canvas.width = Math.round(size.width * ratio)
    canvas.height = Math.round(size.height * ratio)

    const context = canvas.getContext('2d')
    if (context === null) return
    // Draw in CSS pixels; the transform handles the device ratio, so nothing
    // downstream has to know the backing store is larger.
    context.setTransform(ratio, 0, 0, ratio, 0, 0)

    const palette = readScatterPalette()
    drawScatter(
      context,
      {
        positions,
        splits,
        splitColours: assignSplitColours(splits, palette),
        matches,
        highlighted,
        selected: selectedIndices,
        hovered,
      },
      viewport,
      size,
      palette,
      drag.kind === 'select' ? drag.box : null,
    )
  }, [positions, splits, matches, highlighted, selectedIndices, hovered, viewport, size, drag])

  const pointerPosition = (event: PointerEvent<HTMLCanvasElement>): ScreenPoint => {
    const bounds = event.currentTarget.getBoundingClientRect()
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top }
  }

  // Wheel is attached natively rather than through React's onWheel: the
  // synthetic listener is passive, so preventDefault there is ignored and the
  // page scrolls while the user is trying to zoom.
  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas === null) return
    const onWheel = (event: WheelEvent): void => {
      event.preventDefault()
      const bounds = canvas.getBoundingClientRect()
      const anchor = { x: event.clientX - bounds.left, y: event.clientY - bounds.top }
      const factor = Math.exp(-event.deltaY * ZOOM_PER_WHEEL_UNIT)
      setViewport((current) => zoomAtPoint(current, anchor, factor, size))
    }
    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
  }, [size])

  const handlePointerDown = (event: PointerEvent<HTMLCanvasElement>): void => {
    const at = pointerPosition(event)
    event.currentTarget.setPointerCapture(event.pointerId)
    setDrag(
      event.shiftKey
        ? { kind: 'select', box: { from: at, to: at } }
        : { kind: 'pan', last: at, travelled: 0 },
    )
  }

  const handlePointerMove = (event: PointerEvent<HTMLCanvasElement>): void => {
    const at = pointerPosition(event)

    if (drag.kind === 'pan') {
      const deltaX = at.x - drag.last.x
      const deltaY = at.y - drag.last.y
      setViewport((current) => panBy(current, deltaX, deltaY))
      setDrag({
        kind: 'pan',
        last: at,
        travelled: drag.travelled + Math.abs(deltaX) + Math.abs(deltaY),
      })
      return
    }

    if (drag.kind === 'select') {
      setDrag({ kind: 'select', box: { from: drag.box.from, to: at } })
      return
    }

    // Hover: only publish when the point under the cursor actually changes, so
    // a slow drag across the cloud does not re-render on every pixel.
    const index = findNearestPoint(positions, at, viewport, size)
    if (index === hovered) return
    setHovered(index)
    onHoverChange(index < 0 ? null : (points[index] ?? null), index < 0 ? null : at)
  }

  const handlePointerUp = (event: PointerEvent<HTMLCanvasElement>): void => {
    const at = pointerPosition(event)
    event.currentTarget.releasePointerCapture(event.pointerId)

    if (drag.kind === 'select') {
      // `matches` travels with the geometry: a rectangle drawn while a filter is
      // active must pick up only what the user can see inside it, because this
      // selection feeds a bulk collection move.
      onSelectionChange(pointsInBox(positions, matches, drag.box, viewport, size))
    } else if (drag.kind === 'pan' && drag.travelled < CLICK_SLOP_PX) {
      // A press that never moved is a click: open the image under it.
      const index = findNearestPoint(positions, at, viewport, size)
      const point = index < 0 ? undefined : points[index]
      if (point !== undefined) onActivate(point)
    }
    setDrag({ kind: 'none' })
  }

  const handlePointerLeave = (): void => {
    setDrag({ kind: 'none' })
    setHovered(-1)
    onHoverChange(null, null)
  }

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: '100%' }}
      className={cn(
        'touch-none rounded-lg border border-border bg-card',
        drag.kind === 'pan' ? 'cursor-grabbing' : 'cursor-grab',
      )}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerLeave}
    />
  )
}
