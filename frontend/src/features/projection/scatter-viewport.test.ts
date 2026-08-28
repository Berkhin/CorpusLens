/**
 * Unit tests for the box query behind the map's rectangle selection.
 *
 * `scatter-viewport.ts` was extracted as pure functions precisely so this could
 * exist, and until now it did not. The behaviour under test is the one that was
 * silently wrong: {@link pointsInBox} selected by geometry alone, ignoring the
 * per-point `matches` flag the renderer already uses to dim filtered-out points.
 * On the real corpus, with the Weak-captions filter active (200 of 8 000
 * matching), one drag over the dense lobe selected 5 379 points — and that
 * selection is what the "Move to…" button writes.
 *
 * Coordinates are chosen so the assertions can be read without running the
 * arithmetic: with a 200x200 canvas and the identity viewport, the origin is at
 * (100, 100) and one world unit is 92 px (see `EDGE_PADDING`).
 */

import { describe, expect, it } from 'vitest'

import {
  IDENTITY_VIEWPORT,
  pointsInBox,
  worldToScreen,
  type CanvasSize,
  type ScreenBox,
  type WorldPoint,
} from '@/features/projection/scatter-viewport'

const SIZE: CanvasSize = { width: 200, height: 200 }

/** Four points, one per quadrant, well clear of each other and of the axes. */
const WORLD: readonly WorldPoint[] = [
  { x: 0.5, y: 0.5 }, // 0 — top right
  { x: -0.5, y: 0.5 }, // 1 — top left
  { x: -0.5, y: -0.5 }, // 2 — bottom left
  { x: 0.5, y: -0.5 }, // 3 — bottom right
]

function positionsOf(points: readonly WorldPoint[]): Float32Array {
  const flat = new Float32Array(points.length * 2)
  points.forEach((point, index) => {
    flat[index * 2] = point.x
    flat[index * 2 + 1] = point.y
  })
  return flat
}

/** A rectangle covering the whole canvas, i.e. every point. */
const WHOLE_CANVAS: ScreenBox = {
  from: { x: 0, y: 0 },
  to: { x: SIZE.width, y: SIZE.height },
}

/** A rectangle covering the right half only, i.e. points 0 and 3. */
const RIGHT_HALF: ScreenBox = {
  from: { x: SIZE.width / 2, y: 0 },
  to: { x: SIZE.width, y: SIZE.height },
}

const POSITIONS = positionsOf(WORLD)
const ALL_MATCH: readonly boolean[] = WORLD.map(() => true)

describe('pointsInBox', () => {
  it('selects every point in the rectangle when no filter is active', () => {
    // The no-regression case: with nothing filtered out, every point matches
    // and the result is the pure geometry the function returned before.
    expect(pointsInBox(POSITIONS, ALL_MATCH, WHOLE_CANVAS, IDENTITY_VIEWPORT, SIZE)).toEqual([
      0, 1, 2, 3,
    ])
  })

  it('selects by geometry, not by index order', () => {
    expect(pointsInBox(POSITIONS, ALL_MATCH, RIGHT_HALF, IDENTITY_VIEWPORT, SIZE)).toEqual([0, 3])
  })

  it('excludes non-matching points that fall inside the rectangle', () => {
    // The defect this function existed to have: points 1 and 2 are inside the
    // rectangle and outside the filter, and a bulk move must not touch them.
    const matches = [true, false, false, true]

    expect(pointsInBox(POSITIONS, matches, WHOLE_CANVAS, IDENTITY_VIEWPORT, SIZE)).toEqual([0, 3])
  })

  it('returns nothing when the rectangle holds only non-matching points', () => {
    const matches = [false, true, true, false]

    expect(pointsInBox(POSITIONS, matches, RIGHT_HALF, IDENTITY_VIEWPORT, SIZE)).toEqual([])
  })

  it('returns nothing for a zero-area rectangle', () => {
    // A shift-click with no drag. The point sits exactly on the corner, so this
    // also pins the boundary as inclusive-but-empty rather than accidentally
    // selecting whatever the click landed on.
    const empty: ScreenBox = { from: { x: 0, y: 0 }, to: { x: 0, y: 0 } }

    expect(pointsInBox(POSITIONS, ALL_MATCH, empty, IDENTITY_VIEWPORT, SIZE)).toEqual([])
  })

  it('returns nothing when the rectangle misses the cloud entirely', () => {
    const offCloud: ScreenBox = { from: { x: 195, y: 195 }, to: { x: 200, y: 200 } }

    expect(pointsInBox(POSITIONS, ALL_MATCH, offCloud, IDENTITY_VIEWPORT, SIZE)).toEqual([])
  })

  it('normalises a rectangle dragged up and to the left', () => {
    // Dragging bottom-right to top-left gives a box whose `to` is above and left
    // of its `from`; the selection must not depend on the drag direction.
    const backwards: ScreenBox = { from: RIGHT_HALF.to, to: RIGHT_HALF.from }

    expect(pointsInBox(POSITIONS, ALL_MATCH, backwards, IDENTITY_VIEWPORT, SIZE)).toEqual([0, 3])
  })

  it('follows the viewport, so a panned map selects what is drawn under the box', () => {
    // The transform is shared with the renderer; this asserts the box query uses
    // it rather than raw world coordinates. Panning right pushes the right-hand
    // pair past the canvas edge, and they stop being selectable.
    const panned = { scale: 1, offsetX: 60, offsetY: 0 }
    const onScreen = worldToScreen(WORLD[0] as WorldPoint, panned, SIZE)
    expect(onScreen.x).toBeGreaterThan(SIZE.width)

    expect(pointsInBox(POSITIONS, ALL_MATCH, WHOLE_CANVAS, panned, SIZE)).toEqual([1, 2])
  })

  it('treats a point beyond the end of the matches array as non-matching', () => {
    // The two arrays are derived from the same `points` array, so a length
    // mismatch is a bug — and dropping is the safe direction when the result
    // feeds a write.
    expect(pointsInBox(POSITIONS, [true, true], WHOLE_CANVAS, IDENTITY_VIEWPORT, SIZE)).toEqual([
      0, 1,
    ])
  })
})
