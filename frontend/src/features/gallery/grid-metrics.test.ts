import { describe, expect, it } from 'vitest'

import {
  GRID_GAP,
  chunkIntoRows,
  columnsForWidth,
  rowHeightForWidth,
} from '@/features/gallery/grid-metrics'

describe('columnsForWidth', () => {
  it('widens the grid as the container grows', () => {
    expect(columnsForWidth(400)).toBe(2)
    expect(columnsForWidth(700)).toBe(3)
    expect(columnsForWidth(1000)).toBe(4)
    expect(columnsForWidth(1400)).toBe(5)
  })

  it('never returns fewer than two columns, including before measurement', () => {
    // The container measures 0 on the render before the ResizeObserver fires.
    // A zero here would divide by zero in rowHeightForWidth and blank the grid.
    expect(columnsForWidth(0)).toBe(2)
  })

  it('treats each breakpoint as inclusive of its own width', () => {
    expect(columnsForWidth(591)).toBe(2)
    expect(columnsForWidth(592)).toBe(3)
    expect(columnsForWidth(975)).toBe(3)
    expect(columnsForWidth(976)).toBe(4)
  })
})

describe('rowHeightForWidth', () => {
  it('derives height from the 4:3 card plus one gap', () => {
    // Four columns of 100px with three 12px gaps between them.
    const width = 4 * 100 + 3 * GRID_GAP
    expect(rowHeightForWidth(width, 4)).toBe(75 + GRID_GAP)
  })

  it('stays positive at zero width so the virtualizer can divide by it', () => {
    expect(rowHeightForWidth(0, 2)).toBeGreaterThan(0)
  })
})

describe('chunkIntoRows', () => {
  it('groups items into rows of the given width', () => {
    expect(chunkIntoRows([1, 2, 3, 4, 5, 6], 3)).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ])
  })

  it('leaves a short final row rather than padding it', () => {
    // Padding would render placeholder cards the user could try to click.
    expect(chunkIntoRows([1, 2, 3, 4, 5], 3)).toEqual([
      [1, 2, 3],
      [4, 5],
    ])
  })

  it('returns no rows for an empty list', () => {
    expect(chunkIntoRows([], 4)).toEqual([])
  })

  it('does not loop forever on a non-positive column count', () => {
    expect(chunkIntoRows([1, 2], 0)).toEqual([[1], [2]])
  })
})
