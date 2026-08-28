/**
 * The coordinate maths behind the embedding map.
 *
 * Kept as pure functions, separate from the canvas component, for two reasons:
 * the component stays about drawing and events rather than algebra, and this
 * half is the half that is easy to get subtly wrong — an inverted axis or a
 * zoom that drifts under the cursor is invisible in a code review and obvious
 * in use.
 *
 * World coordinates arrive from the backend already normalised to roughly
 * `[-1, 1]` on both axes, with **one** scale factor applied to both. Everything
 * here preserves that: the aspect ratio of the projection carries meaning, so
 * the two axes are never scaled independently.
 */

/** A position in projection space, as the backend supplies it. */
export type WorldPoint = { x: number; y: number }

/** A position in CSS pixels relative to the canvas element. */
export type ScreenPoint = { x: number; y: number }

export type CanvasSize = { width: number; height: number }

/**
 * Pan and zoom state.
 *
 * `offset` is in screen pixels and applies *after* scaling, which is what makes
 * {@link zoomAtPoint} expressible as a single subtraction.
 */
export type Viewport = { scale: number; offsetX: number; offsetY: number }

export const IDENTITY_VIEWPORT: Viewport = { scale: 1, offsetX: 0, offsetY: 0 }

/** Fraction of the shorter axis left as margin at scale 1, so points near the edge stay clickable. */
const EDGE_PADDING = 0.92

export const MIN_SCALE = 0.5
export const MAX_SCALE = 40

/** Pointer distance, in CSS pixels, within which a point counts as hovered. */
export const HOVER_RADIUS_PX = 12

/**
 * Pixels per world unit before the viewport's own zoom.
 *
 * Derived from the shorter axis so the cloud fits whatever the container's
 * aspect ratio happens to be, and shared by both axes so the plot is never
 * stretched.
 */
function baseScale(size: CanvasSize): number {
  return (Math.min(size.width, size.height) / 2) * EDGE_PADDING
}

/**
 * The world-to-screen mapping, hoisted out of the loops that use it.
 *
 * `screenX = originX + world.x * pixelsPerUnit`
 * `screenY = originY - world.y * pixelsPerUnit`
 *
 * The renderer, the hit test and the box query each walk thousands of points
 * per frame, so all three inline that arithmetic rather than calling a function
 * per point — but they all derive it **here**. That matters more than it looks:
 * if the drawing code and the hit test ever computed the transform separately,
 * points would render in one place and respond to the cursor in another, and
 * the bug would present as "clicking is slightly off" rather than as anything
 * greppable.
 */
export type ScatterTransform = { originX: number; originY: number; pixelsPerUnit: number }

export function scatterTransform(viewport: Viewport, size: CanvasSize): ScatterTransform {
  return {
    originX: size.width / 2 + viewport.offsetX,
    originY: size.height / 2 + viewport.offsetY,
    pixelsPerUnit: baseScale(size) * viewport.scale,
  }
}

/** Project a world position onto the canvas. */
export function worldToScreen(
  point: WorldPoint,
  viewport: Viewport,
  size: CanvasSize,
): ScreenPoint {
  const { originX, originY, pixelsPerUnit } = scatterTransform(viewport, size)
  return {
    x: originX + point.x * pixelsPerUnit,
    // Negated: screen y grows downward, and a plot whose y axis points down
    // would mirror the projection for no reason a reader could guess.
    y: originY - point.y * pixelsPerUnit,
  }
}

/** Recover the world position under a canvas coordinate. */
export function screenToWorld(
  point: ScreenPoint,
  viewport: Viewport,
  size: CanvasSize,
): WorldPoint {
  const { originX, originY, pixelsPerUnit } = scatterTransform(viewport, size)
  return {
    x: (point.x - originX) / pixelsPerUnit,
    y: -(point.y - originY) / pixelsPerUnit,
  }
}

export function clampScale(scale: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale))
}

/**
 * Zoom by `factor`, keeping whatever is under `anchor` exactly where it is.
 *
 * Zooming towards the viewport centre instead would make the point of interest
 * slide away from the cursor, which is the difference between a map that feels
 * direct and one that feels like it is fighting you.
 */
export function zoomAtPoint(
  viewport: Viewport,
  anchor: ScreenPoint,
  factor: number,
  size: CanvasSize,
): Viewport {
  const scale = clampScale(viewport.scale * factor)
  if (scale === viewport.scale) return viewport

  const world = screenToWorld(anchor, viewport, size)
  const pixelsPerUnit = baseScale(size) * scale
  return {
    scale,
    offsetX: anchor.x - size.width / 2 - world.x * pixelsPerUnit,
    offsetY: anchor.y - size.height / 2 + world.y * pixelsPerUnit,
  }
}

/** Translate the view by a screen-space delta. */
export function panBy(viewport: Viewport, deltaX: number, deltaY: number): Viewport {
  return { ...viewport, offsetX: viewport.offsetX + deltaX, offsetY: viewport.offsetY + deltaY }
}

/**
 * Index of the point nearest `pointer`, or `-1` when none is close enough.
 *
 * A linear scan. At 8 000 points that is ~8 000 subtractions per pointer move —
 * tens of microseconds, comfortably inside a frame — so a quadtree would be
 * structure bought with complexity and paid for by nobody. Distances are
 * compared squared to skip 8 000 square roots.
 */
export function findNearestPoint(
  positions: Float32Array,
  pointer: ScreenPoint,
  viewport: Viewport,
  size: CanvasSize,
  radiusPx: number = HOVER_RADIUS_PX,
): number {
  const { originX, originY, pixelsPerUnit } = scatterTransform(viewport, size)
  const limit = radiusPx * radiusPx

  let best = -1
  let bestDistance = limit
  for (let index = 0; index < positions.length; index += 2) {
    const screenX = originX + (positions[index] ?? 0) * pixelsPerUnit
    const screenY = originY - (positions[index + 1] ?? 0) * pixelsPerUnit
    const dx = screenX - pointer.x
    const dy = screenY - pointer.y
    const distance = dx * dx + dy * dy
    if (distance <= bestDistance) {
      bestDistance = distance
      best = index / 2
    }
  }
  return best
}

/** A rectangle in screen space, given by two opposite corners. */
export type ScreenBox = { from: ScreenPoint; to: ScreenPoint }

/** Normalise a drag rectangle so it has positive width and height. */
export function normaliseBox(box: ScreenBox): {
  x: number
  y: number
  width: number
  height: number
} {
  const x = Math.min(box.from.x, box.to.x)
  const y = Math.min(box.from.y, box.to.y)
  return {
    x,
    y,
    width: Math.abs(box.to.x - box.from.x),
    height: Math.abs(box.to.y - box.from.y),
  }
}

/**
 * Indices of every point inside a screen-space rectangle *that matches the filter*.
 *
 * `matches` is required rather than optional, so that adding a caller cannot
 * silently reintroduce a geometry-only selection. It is the same per-point flag
 * the renderer uses to dim non-matching points, so what the rectangle picks up
 * is exactly what the user can see inside it — and the selection feeds a bulk
 * write, where the difference is not cosmetic. Measured on the real corpus with
 * the Weak-captions filter active (200 of 8 000 matching), one drag over the
 * dense lobe returned **5 379** points, of which 98 actually matched.
 *
 * With no filter active every point matches and this is a no-op, which is the
 * property that makes the fix additive rather than a behaviour change.
 *
 * A point whose index is beyond `matches` is treated as non-matching: the two
 * arrays are built from the same `points` array by the caller, so a
 * disagreement is a bug, and dropping is the safe direction for a write.
 */
export function pointsInBox(
  positions: Float32Array,
  matches: readonly boolean[],
  box: ScreenBox,
  viewport: Viewport,
  size: CanvasSize,
): number[] {
  const { x, y, width, height } = normaliseBox(box)
  const { originX, originY, pixelsPerUnit } = scatterTransform(viewport, size)

  const selected: number[] = []
  for (let index = 0; index < positions.length; index += 2) {
    if (matches[index / 2] !== true) continue
    const screenX = originX + (positions[index] ?? 0) * pixelsPerUnit
    const screenY = originY - (positions[index + 1] ?? 0) * pixelsPerUnit
    if (screenX >= x && screenX <= x + width && screenY >= y && screenY <= y + height) {
      selected.push(index / 2)
    }
  }
  return selected
}
