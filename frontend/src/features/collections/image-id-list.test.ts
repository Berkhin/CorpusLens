/**
 * Unit tests for the imported-id parser.
 *
 * The input is a file someone else's training run wrote, so the cases that
 * matter are the untidy ones: a CSV header, quoted values, a trailing newline,
 * the same id twice.
 */

import { describe, expect, it } from 'vitest'

import { parseImageIdList } from '@/features/collections/image-id-list'

const DOG = '1000268201_693b08cb0e'
const SLIDE = '1001773457_577c3a7d70'

describe('parseImageIdList', () => {
  it('reads one id per line, ignoring blank lines and trailing newlines', () => {
    expect(parseImageIdList(`${DOG}\n\n${SLIDE}\n`)).toEqual({
      ids: [DOG, SLIDE],
      malformed: [],
      duplicates: 0,
    })
  })

  it('reads a comma-separated list', () => {
    expect(parseImageIdList(`${DOG}, ${SLIDE}`).ids).toEqual([DOG, SLIDE])
  })

  it('strips the quotes a CSV writer adds', () => {
    expect(parseImageIdList(`"${DOG}";'${SLIDE}'`).ids).toEqual([DOG, SLIDE])
  })

  it('collapses repeats and counts them, keeping first-seen order', () => {
    const parsed = parseImageIdList(`${SLIDE}\n${DOG}\n${SLIDE}\n${SLIDE}`)

    expect(parsed.ids).toEqual([SLIDE, DOG])
    expect(parsed.duplicates).toBe(2)
  })

  it('separates tokens that cannot be an image id instead of sending them', () => {
    // A CSV header row is the common case, and `image_id` is itself a
    // well-formed id — it just is not in the corpus, so the backend reports it
    // through `unknown[]`. The other two are not, and never leave the browser.
    const parsed = parseImageIdList(`image_id\n${DOG}\n/etc/passwd\n../../secret`)

    expect(parsed.ids).toEqual(['image_id', DOG])
    expect(parsed.malformed).toEqual(['/etc/passwd', '../../secret'])
  })

  it('treats whitespace as a separator, so a space-delimited list parses', () => {
    expect(parseImageIdList(`${DOG} ${SLIDE}`).ids).toEqual([DOG, SLIDE])
  })

  it('returns nothing for empty or whitespace-only input', () => {
    expect(parseImageIdList('   \n\t ')).toEqual({ ids: [], malformed: [], duplicates: 0 })
  })
})
