/**
 * Drawing the point cloud.
 *
 * Split out of the component so that the component owns events and state while
 * this owns pixels. It is also the hot path: everything here runs on every
 * frame of a pan, so the code is written to touch the canvas state as little as
 * possible.
 */

import type { ScatterPalette } from '@/features/projection/scatter-palette'
import {
  normaliseBox,
  scatterTransform,
  type CanvasSize,
  type ScreenBox,
  type Viewport,
} from '@/features/projection/scatter-viewport'

/** Dot radius in CSS pixels at scale 1, and the ceiling it grows to. */
const BASE_RADIUS = 1.7
const MAX_RADIUS = 5

/** Opacity applied to points the active filter excludes. */
const DIMMED_ALPHA = 0.35

export type ScatterFrame = {
  positions: Float32Array
  /** Split name per point; the renderer buckets by it. */
  splits: readonly string[]
  /** Fill colour per split name. */
  splitColours: ReadonlyMap<string, string>
  /** Whether each point survives the active filter. */
  matches: readonly boolean[]
  /** Indices of current search hits, drawn on top and enlarged. */
  highlighted: ReadonlySet<number>
  /** Indices inside the current box selection. */
  selected: ReadonlySet<number>
  hovered: number
}

/**
 * Radius grows with zoom, but sub-linearly and to a ceiling.
 *
 * Constant-size dots become an unreadable smear when zoomed into a dense
 * region; dots that scale linearly with zoom stay exactly as unreadable.
 */
function pointRadius(scale: number): number {
  return Math.min(MAX_RADIUS, BASE_RADIUS * Math.sqrt(scale))
}

/**
 * Draw one batch of points as a single path.
 *
 * One `fill()` per colour rather than per point: 8 000 individual fills means
 * 8 000 driver round trips, while a handful of batched paths is a few.
 */
function fillBatch(
  context: CanvasRenderingContext2D,
  indices: readonly number[],
  positions: Float32Array,
  originX: number,
  originY: number,
  pixelsPerUnit: number,
  radius: number,
  colour: string,
  alpha: number,
): void {
  if (indices.length === 0) return
  context.globalAlpha = alpha
  context.fillStyle = colour
  context.beginPath()
  for (const index of indices) {
    const x = originX + (positions[index * 2] ?? 0) * pixelsPerUnit
    const y = originY - (positions[index * 2 + 1] ?? 0) * pixelsPerUnit
    context.moveTo(x + radius, y)
    context.arc(x, y, radius, 0, Math.PI * 2)
  }
  context.fill()
  context.globalAlpha = 1
}

/** Screen position of one point, for ring overlays. */
function screenAt(
  positions: Float32Array,
  index: number,
  originX: number,
  originY: number,
  pixelsPerUnit: number,
): { x: number; y: number } {
  return {
    x: originX + (positions[index * 2] ?? 0) * pixelsPerUnit,
    y: originY - (positions[index * 2 + 1] ?? 0) * pixelsPerUnit,
  }
}

function strokeRings(
  context: CanvasRenderingContext2D,
  indices: Iterable<number>,
  positions: Float32Array,
  originX: number,
  originY: number,
  pixelsPerUnit: number,
  radius: number,
  colour: string,
  lineWidth: number,
): void {
  context.strokeStyle = colour
  context.lineWidth = lineWidth
  context.beginPath()
  for (const index of indices) {
    const { x, y } = screenAt(positions, index, originX, originY, pixelsPerUnit)
    context.moveTo(x + radius, y)
    context.arc(x, y, radius, 0, Math.PI * 2)
  }
  context.stroke()
}

/**
 * Render the whole scene.
 *
 * Draw order is the visual priority order: excluded points first so everything
 * else sits above them, then the ordinary cloud grouped by split, then search
 * hits, then the selection outline, then the hover ring.
 */
export function drawScatter(
  context: CanvasRenderingContext2D,
  frame: ScatterFrame,
  viewport: Viewport,
  size: CanvasSize,
  palette: ScatterPalette,
  selectionBox: ScreenBox | null,
): void {
  context.clearRect(0, 0, size.width, size.height)

  const { originX, originY, pixelsPerUnit } = scatterTransform(viewport, size)
  const radius = pointRadius(viewport.scale)

  const excluded: number[] = []
  const byColour = new Map<string, number[]>()
  const count = frame.positions.length / 2

  for (let index = 0; index < count; index += 1) {
    if (frame.matches[index] !== true) {
      excluded.push(index)
      continue
    }
    const split = frame.splits[index] ?? ''
    const colour = frame.splitColours.get(split) ?? palette.dimmed
    const bucket = byColour.get(colour)
    if (bucket === undefined) byColour.set(colour, [index])
    else bucket.push(index)
  }

  fillBatch(
    context,
    excluded,
    frame.positions,
    originX,
    originY,
    pixelsPerUnit,
    radius,
    palette.dimmed,
    DIMMED_ALPHA,
  )

  for (const [colour, indices] of byColour) {
    fillBatch(
      context,
      indices,
      frame.positions,
      originX,
      originY,
      pixelsPerUnit,
      radius,
      colour,
      0.85,
    )
  }

  if (frame.highlighted.size > 0) {
    fillBatch(
      context,
      [...frame.highlighted],
      frame.positions,
      originX,
      originY,
      pixelsPerUnit,
      radius * 2,
      palette.highlight,
      1,
    )
  }

  if (frame.selected.size > 0) {
    strokeRings(
      context,
      frame.selected,
      frame.positions,
      originX,
      originY,
      pixelsPerUnit,
      radius * 2.2,
      palette.accent,
      1.2,
    )
  }

  if (frame.hovered >= 0) {
    strokeRings(
      context,
      [frame.hovered],
      frame.positions,
      originX,
      originY,
      pixelsPerUnit,
      radius * 3,
      palette.accent,
      2,
    )
  }

  if (selectionBox !== null) {
    const { x, y, width, height } = normaliseBox(selectionBox)
    context.globalAlpha = 0.15
    context.fillStyle = palette.accent
    context.fillRect(x, y, width, height)
    context.globalAlpha = 1
    context.strokeStyle = palette.accent
    context.lineWidth = 1
    context.setLineDash([4, 3])
    context.strokeRect(x, y, width, height)
    context.setLineDash([])
  }
}
