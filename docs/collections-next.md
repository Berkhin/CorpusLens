# Collections: where the overlay stops, and what comes after

Status: **proposal**. Nothing here is built. It exists because the three items
below are schema and lifecycle decisions rather than bug fixes, and because the
first of them has now been settled by measurement rather than by argument.

The question this answers is the one the feature was built to test: *should a
researcher get editable splits, or a collection overlay laid over immutable
ones?*

---

## 1. The answer: both, at different stages of the same lifecycle

**The overlay is correct, and the reason is falsifiability.** `split` is what
`scripts/analyze.py` derives cross-split duplicate leakage from. Exercised on
the real corpus: quarantining the 32 flagged images left `images_by_split` at
`{train: 6000, validation: 1000, test: 1000}` and moved
`images_by_collection` instead, which is what let the effect be measured
honestly — cross-*collection* leakage fell 22 → 8. Had `split` been editable,
that number would have moved because the baseline moved, and "I fixed the
leakage" would be indistinguishable from "I relabelled it away".

Three consequences follow from the same root and are worth stating separately:

- Flickr8k's split is a **published benchmark**. Overwrite it and the numbers
  stop being comparable to the literature.
- The API is a **pure reader** of an index built by idempotent offline scripts
  (CLAUDE.md §4.2). Let it write to LanceDB and the next `ingest.py --force`
  silently discards the researcher's work.
- An overlay has an **undo** — deleting a collection returns its images to
  their splits, free, via `ON DELETE CASCADE`. An edited split has nothing to
  revert to.

**But the overlay does not scale to a whole-corpus re-partition, and that is
now a number rather than an opinion.** Membership is not a column, so it
compiles to id literals in every filtered query. Measured on the 8 000-row
table (see the table in `repositories/filters.py`):

| overrides | predicate | filtered `count_rows` |
|---|---|---|
| 0 | 0 KB | 22 ms |
| 1 000 | 24 KB | 222 ms |
| 8 000 | 194 KB | 1 637 ms |

No encoding rescues the tail. The selection can be written as its members, as
its complement, or as split-plus-deltas; a corpus split evenly between two
collections still names at least 4 000 ids under all three. That is roughly a
2× saving against a 38× regression, so the honest conclusion is that this is a
**boundary, not an optimisation target**.

Hence the lifecycle split, which is the actual recommendation:

| | overlay (`collections.db`) | new ingested index |
|---|---|---|
| **Is for** | the draft: quarantine, holdout, review queue | the commit: a partition you intend to train or publish on |
| **Scale** | up to `max_collection_overrides` (1 000) | the whole corpus |
| **Membership is** | an id literal per moved image | a `split` column again |
| **Measurable by** | the API's cross-collection figures | `analyze.py`, in full |
| **Reversible** | yes, per image | no — but the previous index still exists |

`scripts/export_split.py` already writes the overlay out as a manifest. The
missing half is an ingestion mode that consumes one, producing a **new**
LanceDB table whose `split` is the researcher's partition — leaving the
original beside it so the two can be compared. That, not more overlay
machinery, is what "editable splits" should mean here.

**Not doing** the predicate optimisation is part of this proposal, not an
omission from it: it only pays off in the regime the boundary above declares
out of scope.

---

## 2. Tags are a different thing from a partition

`image_collection.image_id` is a `PRIMARY KEY`, so an image sits in exactly one
collection. That is right for a **partition** (train / val / test / holdout —
one-of-N, exhaustive) and wrong for a **label** (`duplicate`, `weak-caption`,
`mislabelled`, `review` — overlapping, sparse).

The two are routinely true at once: an image can be a near-duplicate *and*
weakly captioned *and* in a holdout. Today, moving it to `quarantine` removes
it from `train`, which is correct under one reading and destructive under the
other. One mechanism is doing two jobs and can only do one.

**Proposed:** a second store beside the overlay — many-to-many, following the
same rules that make the overlay safe (its own file, never the index,
filterable through the same predicate path). Tags are sparse by nature, so the
cost model differs from §1 and the tight override ceiling need not apply.

Open question worth settling before building: whether the quality flags
(`near-duplicate`, `weak-captions`) become *system* tags in that store rather
than a parallel filter dimension. They behave like tags in every respect except
that they are computed rather than asserted.

---

## 3. Colour the map by collection

`ProjectionPoint` deliberately carries no `collection`, and the map colours by
ground-truth `split` permanently. The current docstring argues this protects
ground truth. It does not — it only withholds the second view. After a
re-partition the map, which is this tool's strongest view, cannot show the
partition just built; the only signal is binary dim/bright `matches` for one
filter at a time.

**Proposed:** a `colour by: split | collection` toggle, both labelled. Ground
truth is protected by keeping both available, not by suppressing one.

The real obstacle is honest and is a design problem rather than a config bump:
`scatter-palette.ts` is a fixed three-colour scheme chosen for categorical
distinctness, and N arbitrary user collections have no such palette. A
plausible answer is to colour the selected collections and grey the rest,
which matches how the filter already reads — but it should be decided, not
defaulted into.

---

## Priority

§1's missing half (ingest-from-manifest) is the one that turns the feature into
something a training run can consume, and it is the one to build first. §2 is
the larger correctness gap but is additive and can wait. §3 is small and
cosmetic until §1 exists, because before that there is rarely a partition worth
looking at.
