/**
 * Parsing a pasted or uploaded list of image ids.
 *
 * The normal direction of traffic is the opposite of what the UI supported: a
 * training run emits 400 failure-case ids and you want to look at them. So the
 * input is whatever that run happened to write — one per line, comma-separated,
 * a CSV column with a header row, something with a trailing newline.
 *
 * Kept as a pure function, separate from the form, because the interesting
 * behaviour is entirely in the parsing and none of it is in the rendering.
 */

/**
 * Mirrors `IMAGE_ID_PATTERN` in backend/app/models/schemas.py.
 *
 * Applied here so a stray token becomes a line in the report rather than a 422
 * that names a field index and fails the whole batch. The backend still
 * enforces it; this only decides what is worth sending.
 */
const IMAGE_ID_PATTERN = /^[A-Za-z0-9._-]+$/

/**
 * Mirrors `MAX_COLLECTION_MOVE_IMAGES` in backend/app/models/schemas.py.
 *
 * This is the *body* limit, which is why it is the corpus size rather than the
 * much smaller `CORPUSLENS_MAX_COLLECTION_OVERRIDES` the backend really enforces.
 * A paste under this length can still be refused with a 413 once it is clear
 * how many overrides it would leave behind — that answer needs the server's
 * view of the current overlay, so it is not second-guessed here.
 */
export const MAX_IMPORT_IDS = 8000

/** Anything that separates two ids in a file someone else wrote. */
const SEPARATORS = /[\s,;]+/

export type ParsedImageIdList = {
  /** Well-formed ids, de-duplicated, in the order they first appeared. */
  ids: string[]
  /** Tokens that cannot be an image id, in the order they appeared. */
  malformed: string[]
  /** How many repeats were collapsed. */
  duplicates: number
}

/**
 * Split a pasted list into what can be sent and what cannot.
 *
 * Quotes are stripped before validation, so a CSV column written by
 * `csv.writer` parses without asking the user to clean it up first.
 *
 * @param text Raw pasted or uploaded content.
 */
export function parseImageIdList(text: string): ParsedImageIdList {
  const ids: string[] = []
  const malformed: string[] = []
  const seen = new Set<string>()
  let duplicates = 0

  for (const raw of text.split(SEPARATORS)) {
    const token = raw.replace(/^["']|["']$/g, '')
    if (token.length === 0) continue
    if (!IMAGE_ID_PATTERN.test(token)) {
      malformed.push(token)
      continue
    }
    if (seen.has(token)) {
      duplicates += 1
      continue
    }
    seen.add(token)
    ids.push(token)
  }

  return { ids, malformed, duplicates }
}
