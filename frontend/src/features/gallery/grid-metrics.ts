/**
 * Geometry for the virtualized image grid.
 *
 * Windowing needs numbers CSS was previously allowed to keep to itself. A
 * virtualizer positions rows absolutely, so it has to know how many cards sit
 * in a row and how tall a row is *before* React renders one — which a
 * `grid-cols-*` utility ramp cannot answer from JavaScript.
 *
 * These are pure functions of the measured container width, kept out of the
 * component so the arithmetic is unit-testable and so the one place breakpoints
 * are defined is greppable.
 */

/** Gap between cards, in pixels. Must track the `gap-3` utility on the grid. */
export const GRID_GAP = 12

/**
 * Card aspect ratio as height ÷ width. Tracks `aspect-4/3` on the thumbnail,
 * which is what makes a row's height derivable rather than measurable.
 */
const CARD_ASPECT = 3 / 4

/**
 * Column-count breakpoints, as `[minimum container width, columns]`.
 *
 * These mirror the ramp the grid used before it was virtualized
 * (`grid-cols-2 sm:3 lg:4 xl:5`) but are keyed on the **container** rather than
 * the viewport, because that is what the virtualizer can measure. The shell
 * wraps content in `max-w-[1600px] px-6`, so a container is roughly 48px
 * narrower than the viewport at the sizes where these fire; the thresholds are
 * shifted to match, which keeps the rendered layout the same as before.
 *
 * Ordered widest-first so the first match wins.
 */
const COLUMN_BREAKPOINTS: ReadonlyArray<readonly [number, number]> = [
  [1232, 5],
  [976, 4],
  [592, 3],
]

/** Columns below the narrowest breakpoint. */
const MINIMUM_COLUMNS = 2

/**
 * How many cards fit in one row at this container width.
 *
 * @param width Measured container width in pixels. Zero before the first
 *   measurement lands, which yields the minimum rather than a division by zero.
 * @returns A column count of at least {@link MINIMUM_COLUMNS}.
 */
export function columnsForWidth(width: number): number {
  const match = COLUMN_BREAKPOINTS.find(([minimum]) => width >= minimum)
  return match?.[1] ?? MINIMUM_COLUMNS
}

/**
 * Height of one grid row, including the gap beneath it.
 *
 * Derived rather than measured. The thumbnail has a fixed aspect ratio and the
 * badges over it are absolutely positioned, so a row's height is a function of
 * its column width alone — which means the virtualizer never has to re-measure
 * after paint, and rows never shift under the user mid-scroll.
 *
 * @param width Measured container width in pixels.
 * @param columns Column count from {@link columnsForWidth}.
 * @returns Row height in pixels, always positive so the virtualizer can divide
 *   by it safely.
 */
export function rowHeightForWidth(width: number, columns: number): number {
  const available = width - GRID_GAP * (columns - 1)
  const cardWidth = Math.max(available / columns, 1)
  return cardWidth * CARD_ASPECT + GRID_GAP
}

/**
 * Group a flat item list into rows of `columns` items.
 *
 * The virtualizer works in rows, not cards: virtualizing cards individually
 * would make every row a separate absolutely-positioned element and lose the
 * grid's own alignment. The final row is short rather than padded, so nothing
 * renders a placeholder card.
 *
 * @param items Items in display order.
 * @param columns Cards per row; values below one are treated as one so a
 *   mid-measurement render cannot produce an infinite loop.
 * @returns One array per row.
 */
export function chunkIntoRows<T>(items: readonly T[], columns: number): T[][] {
  const perRow = Math.max(columns, 1)
  const rows: T[][] = []
  for (let index = 0; index < items.length; index += perRow) {
    rows.push(items.slice(index, index + perRow))
  }
  return rows
}
