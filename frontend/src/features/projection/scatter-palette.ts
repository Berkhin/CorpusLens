/**
 * Canvas colours, taken from the same CSS variables Tailwind uses.
 *
 * A canvas cannot carry a class, so the alternative would be hardcoding hex
 * values here — which would then be the one part of the app that ignores the
 * theme. Reading the computed custom properties keeps the map in step with
 * everything around it, including the `.dark` variant, for the cost of one
 * `getComputedStyle` per redraw.
 */

/** Chart slots defined in src/styles/tailwind.css. Splits are assigned in order. */
const CHART_SLOTS = 5

export type ScatterPalette = {
  /** One colour per split, indexed by the split's position in sorted order. */
  splitColours: readonly string[]
  /** Points outside the active filter. */
  dimmed: string
  /** Search hits. */
  highlight: string
  /** Outline for selected points and for the selection rectangle. */
  accent: string
  /** Fill wash inside the selection rectangle. */
  accentWash: string
}

function readVariable(styles: CSSStyleDeclaration, name: string, fallback: string): string {
  const value = styles.getPropertyValue(name).trim()
  return value === '' ? fallback : value
}

/**
 * Snapshot the palette for one redraw.
 *
 * Read off the document element, where `:root` declares the variables and where
 * the `.dark` class lands. Deliberately not off the canvas: that would mean
 * touching a ref, and the React Compiler is right to reject a render that
 * depends on one.
 */
export function readScatterPalette(): ScatterPalette {
  const styles = getComputedStyle(document.documentElement)
  const splitColours = Array.from({ length: CHART_SLOTS }, (_unused, index) =>
    readVariable(styles, `--chart-${index + 1}`, '#6b7280'),
  )
  return {
    splitColours,
    // Not `--border`: in dark mode that is a 10%-alpha white, which under the
    // renderer's own dimming alpha disappears entirely.
    dimmed: readVariable(styles, '--muted-foreground', '#9ca3af'),
    // Search hits take the ink colour rather than a sixth hue: maximum contrast
    // against every categorical slot in both themes, no unvalidated colour
    // added to a set that was validated as three, and no red-means-error
    // reading. Size carries the rest — hits draw at twice the radius.
    highlight: readVariable(styles, '--foreground', '#111827'),
    accent: readVariable(styles, '--foreground', '#111827'),
    accentWash: readVariable(styles, '--accent', '#f3f4f6'),
  }
}

/**
 * Map split names onto stable colour slots.
 *
 * Sorted so that `train` always gets the same colour regardless of the order
 * the backend happened to return points in — a legend that reshuffles between
 * loads is worse than no legend.
 */
export function assignSplitColours(
  splits: Iterable<string>,
  palette: ScatterPalette,
): Map<string, string> {
  const sorted = [...new Set(splits)].sort()
  return new Map(
    sorted.map((split, index) => [
      split,
      palette.splitColours[index % palette.splitColours.length] ?? palette.dimmed,
    ]),
  )
}
