# Backend API — contract and design decisions

Recorded per CLAUDE.md §7. Covers the serving layer only; the offline pipeline is
documented in `scripts/ingest.py` and `scripts/project.py`.

## Running

```bash
uvicorn app.main:app --app-dir backend --reload   # http://localhost:8000
```

Interactive schema at `/docs`, raw OpenAPI at `/openapi.json`.

The process refuses to start if `data/images/` or the LanceDB table is missing, with a
message naming `scripts/ingest.py`. The API is a pure reader of what that script
produces (CLAUDE.md §4.2) and never writes to the index.

`data/projection.json` and `data/analysis.json` are treated differently on purpose: both
are **optional**. If it is
absent or unparseable the process starts normally, `/api/dataset/stats` reports
`projection_available: false`, and `/api/projection` answers `404` with a message naming
`scripts/project.py`; `analysis_available` reports the other, and the quality filters
resolve to nothing without it. A missing optional capability should degrade one feature,
not the application.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/dataset/stats` | Corpus counts, both partitions, both leakage figures |
| `GET` | `/api/dataset?offset=&limit=` | Paginated summaries (`limit` ≤ 200, default 50) |
| `GET` | `/api/dataset/{id}` | One image: `{image, analysis}`; `404` if absent |
| `POST` | `/api/search` | CLIP search by text or by example; body carries exactly one of `query` / `image_id` |
| `GET` | `/api/projection` | Every image's 2-D position and filter-match flag; `404` if not computed |
| `POST` | `/api/export` | CSV/JSONL manifest; body `{"format", "ids", "query", "limit", "splits", "collections", "caption_contains"}` |
| `GET` | `/api/collections` | Every collection with its current size, built-ins first |
| `POST` | `/api/collections` | Create one; `201`, or `409` on a duplicate name |
| `PATCH` | `/api/collections/{id}` | Rename; `409` duplicate, `403` built-in, `404` unknown |
| `DELETE` | `/api/collections/{id}` | `204`; `403` built-in; members revert to their split |
| `POST` | `/api/collections/{id}/images` | Move images in, by `ids` **or** by `filter`; returns `{moved, unchanged, unknown}`; `413` above the cap |
| `DELETE` | `/api/collections/{id}/images/{image_id}` | Drop one override; the image returns to its split |
| `GET` | `/images/{file_name}` | Original JPEGs, served by `StaticFiles` |

Every image object carries a root-relative `image_url` (e.g. `/images/123_abc.jpg`), so
the client never composes paths itself. Relative rather than absolute so the same
response is valid whether the browser reaches the API directly or through the Vite proxy.

### Filtering

`GET /api/dataset` and `GET /api/projection` take `split` (repeatable), `collection`
(repeatable), `caption_contains` and `quality_flag`; `POST /api/search` and `POST /api/export`
take the same four as `splits`, `collections`, `caption_contains` and `quality_flag` in the
body. All resolve to one `ImageFilter` domain object, and only
`app/repositories/filters.py` turns it into SQL.

These properties are worth stating because each fails silently rather than loudly:

- **Search pre-filters.** `where(expr, prefilter=True)` narrows the candidate set *before*
  the k-NN scan. With a post-filter, `limit=20` inside one split would return the
  survivors of the global top 20 — fewer than 20 results, with no error.
- **`%` and `_` in a caption filter are literal.** They are escaped and the predicate
  declares `ESCAPE '\'`. Verified against the real table: unescaped, `%` matches all
  8 000 rows.
- **A filtered page reports two totals.** `total` counts matches and drives pagination;
  `corpus_total` is the unfiltered size, so the UI can say "2 014 of 8 000".
- **A quality flag is not a predicate on a row.** It comes from an offline artefact, so it
  is resolved to an id set once, at the route boundary, and compiled into an ordinary
  `id IN (…)`. That is what lets it intersect the other two filters, paginate, and reach
  export without any of those paths knowing it exists. Requested with no analysis loaded,
  it resolves to the *empty* set and compiles to `false` — unsatisfiable, not ignored.
- **A collection is not a predicate on a row either**, and it is the *second* such
  dimension. It resolves at the same boundary, but into its own `CollectionSelection`
  rather than into `ids` — that channel already belongs to the quality flag, and whichever
  resolver ran second would silently overwrite the first. Keeping them in separate fields
  is what makes "cross-split duplicates inside my holdout" an intersection rather than a
  coin toss. An unknown or deleted collection id resolves to nothing, on the same
  unsatisfiable-not-ignored principle.
- **The collection predicate must stay parenthesised.** Membership is "the image's split,
  unless an override says otherwise", which compiles to:

  ```sql
  ( split IN (split_names) AND id NOT IN (excluded) )  OR  id IN (moved_in)
  ```

  `AND` binds tighter than `OR`, so without the outer parentheses the `" AND ".join` that
  attaches the caption clause produces `… OR (id IN (…) AND caption LIKE …)` — the caption
  filter then applies only to the moved-in branch and everything else comes back
  unfiltered. Measured against the real corpus: a `train` + `dog` query returns **1 527
  rows parenthesised and 5 999 without**. It is silent, and it is wrong in the direction
  that looks like success. `backend/tests/test_filters.py` asserts the parentheses on the
  emitted string, and fails without them.
- **With zero overrides the emitted SQL is byte-identical to what it was before
  collections existed** — `split IN ('train')`, nothing more. That is what makes the
  feature additive rather than a rewrite, and it has its own test.

### `GET /api/dataset/stats` — every figure that can be read two ways, reported both ways

| Ground truth (never moves) | The user's partition (follows moves) |
|---|---|
| `images_by_split` | `images_by_collection` |
| `cross_split_duplicate_pairs` | `cross_collection_duplicate_pairs` |
| `caption_retrieval` (corpus-wide) | `caption_recall_by_collection` |

Reporting only the left column lets a researcher act on a finding and never see whether the
action worked — which, for a data engineer, is the job. Reporting only the right column would
quietly redefine what "test set" means. **The pairing is the point**, and it is what the
footer renders.

`cross_collection_duplicate_pairs` costs nothing and needs **no re-run of `analyze.py`**.
Each entry in `duplicate_pairs[]` already carries both member ids *and* both their splits, so
mapping each side through the collection overlay is two dictionary lookups per pair — 52
pairs on the shipped corpus. Quarantine both sides of a leaking pair into one collection and
this falls; separate a same-split duplicate pair and it rises. Both directions are tested.

**It does not go to zero, and that is the honest answer, not a bug.** Measured on the live
corpus: moving all 32 `cross-split-duplicate` images into one collection takes the figure from
**22 to 8** while `cross_split_duplicate_pairs` stays at 22. Two things happen at once. Pairs
whose *both* members were flagged end up together and stop crossing; but a quarantined image
that also near-duplicates an image which stayed behind creates a *new* crossing, because the
quarantine boundary is itself a boundary. A number that only ever went down would be measuring
the intent rather than the partition.

**`caption_recall_by_collection` is a re-aggregation, and is named so it cannot be mistaken
for the harder thing.** `images[id].caption_rank` in the artefact is the median position of
that image when each of its own captions is used as a query **against the whole corpus**.
Restricting those existing numbers to a subset and counting is free, and it answers "how well
does this collection's annotation hold up, ranked against everything?".

It is **not** "R@k with the gallery restricted to collection X" — ranking 200 images against
200 rather than against 8 000 is a different and much better-looking number, and would need
`analyze.py` re-run per partition. Nothing here approximates it.

It is also **not comparable with `caption_retrieval`**, which sits beside it: that one's
denominator is *captions* (40 000, each scored individually), this one's is *images* (each
contributing the median of its own five). Hence the different field name (`images`, not
`captions`) and the spelled-out tooltip in the manager dialog. Collections with no measured
image are **omitted** rather than reported as zero, because zero reads as "these annotations
are terrible" rather than "this was not measured".

Membership for all of this comes from one `(id, split)` projection joined against the
overrides — measured at ~93 ms on the 8 000-row corpus, the same as the single-column
`count_by_split`, because the scan is dominated by fixed overhead. That replaced a
`get_many_by_id` over every overridden id, which at a full re-partition meant an 8 000-entry
`IN` list to answer a question a projection already answers.

### `GET /api/dataset/{id}`

The response is **nested**: `{"image": {…captions…}, "analysis": {…} | null}` rather than
the record at the top level. The two halves come from different places — captions from the
index, measurements from an optional artefact — and nesting keeps "no analysis has been
computed" expressible as `null` instead of as a record with zeroed fields.

`analysis` carries the image's nearest neighbour, the cosine to it, and `caption_rank`:
the median position of this image when each of its own captions is used as a query.
`caption_rank` is itself nullable, because `scripts/analyze.py --no-captions` produces
everything else without it.

### Search by example

`POST /api/search` accepts `image_id` in place of `query`, and a `model_validator` rejects
a body carrying both or neither. This is the cheapest thing the API does: the image's
embedding was computed during ingestion, so there is **no inference at all** — a keyed read
and the same brute-force scan a text search ends with. The query image is dropped from its
own results, and one extra candidate is requested so the caller still gets the count it
asked for.

### `GET /api/projection`

Returns the whole cloud in one response, unpaged. A scatter plot missing a page is not a
smaller plot, it is a wrong one. On the full corpus that is ~695 KB of JSON, which
`GZipMiddleware` compresses to ~174 KB — one request, cached for the session.

Points that fall *outside* the filter are still returned, flagged `matches: false`, so the
client can dim rather than drop them. Where a subset sits relative to the rest of the
corpus is the question the view exists to answer.

The payload deliberately carries no `image_url` and no captions: at 8 000 points those
fields would add most of a megabyte of near-duplicate strings. The client fetches the full
record from `/api/dataset/{id}` when the user hovers a point, which reuses the cache entry
the inspector already fills.

`explained_variance_ratio` is `null` for t-SNE rather than zero — for t-SNE the quantity
is undefined, and a zero would read as "explains nothing".

### `POST /api/export`

`POST` rather than `GET` because a region selected on the map is potentially thousands of
ids, which no URL should carry. Three sources, in precedence order: an explicit `ids` list,
then `query` (re-ranked server-side, so the file corresponds to a reproducible query rather
than to whatever the client had loaded), then everything matching the filter.

The filter is **not** re-applied to an explicit `ids` list: the user selected those exact
images, and dropping some because a filter moved underneath would be a surprise.

The response is a `StreamingResponse` generated lazily and paged out of the store 500 rows
at a time, so a whole-corpus export never materialises 8 000 records at once.

## Collections

The corpus ships partitioned by the dataset's own `split` column. A researcher needs their
own partition too — a hand-curated holdout, a cluster lassoed off the embedding map — and
needs to move images *out* of `train`, not merely tag them. The governing rule:

> **`split` is immutable ground truth. `collection` is a user-editable overlay on top of
> it.**

Every image has exactly one *effective collection*, defaulting to its split. Moving one
writes an **override row**; the LanceDB table is never touched. This is not fussiness:
`scripts/analyze.py` computes cross-split duplicate leakage from the real splits and the
`cross-split-duplicate` filter reports it, so a re-partition that overwrote `split` would
make that measurement quietly wrong. Both travel together — in `ImageSummary`,
`ImageDetail`, the CSV/JSONL manifest, and side by side in the inspector wherever they
differ.

**The three built-ins are derived from the splits actually present in the index**, not
hardcoded. A `--limit`ed ingestion run holds only `train`, and offering a `test` collection
that could only ever be empty would be a lie about the data — the same reasoning behind
`images_by_split` listing only what it found.

**`images_by_split` never moves; `images_by_collection` follows the user.** Reporting both
from `/api/dataset/stats` is what keeps a re-partition legible instead of silently
redefining what "test set" means.

**Storage.** `data/collections.db`, stdlib `sqlite3` — no new dependency, and the engine
enforces the two invariants that would otherwise be hand-rolled: a `UNIQUE … COLLATE
NOCASE` index on the name, and `FOREIGN KEY … ON DELETE CASCADE` on the overrides, which
makes "deleting a collection returns its images to their splits" free and correct. The
built-ins are stored as real rows precisely so the cascade has something to point at.
`PRAGMA foreign_keys` is set on **every** connection: it is per-connection, defaults off,
and without it the cascade silently does nothing. Connections are short-lived and
per-operation, because the service layer runs them on anyio's worker pool.

This makes the API a writer — the one carve-out to "pure reader", scoped in CLAUDE.md §4.2
to a store the API owns rather than to the index. Unlike the projection and analysis
repositories, the collection store is **not** a startup snapshot: its contents change while
the process runs, so every method reads it.

**Orphaned overrides are ignored, not counted.** Re-running ingestion with different ids
leaves override rows pointing at images that no longer exist. `get_many_by_id` omits them,
and that omission is the signal — counting them would inflate every collection with images
that are not there. They are otherwise harmless (an id in an `IN` list matching no row), so
they are not garbage-collected behind the user's back.

**Cost.** The two id lists in the collection predicate are bounded by the number of images
*moved*, not by the corpus, so in normal use they are a handful. A user who re-partitions
everything would produce a predicate of a few hundred kilobytes; that is stated in
`filters.py` and deliberately not pre-optimised.

### Moving images: two channels, one ceiling

`POST /api/collections/{id}/images` takes **exactly one of** `ids` and `filter`, enforced by
a `model_validator` in the same way `POST /api/search` requires exactly one of `query` and
`image_id`. Both a missing source and two sources are a `422`, so the service never has to
answer a request that means two things. An **empty** `ids` array counts as absent: it is the
field's default, and treating it as "an explicit selection of nothing" would make a bare
`{"filter": …}` a two-source request.

```jsonc
{"ids": ["1000268201_693b08cb0e", "…"]}          // an explicit selection
{"filter": {"quality_flag": "weak-captions"}}    // everything that filter lists
```

The filter carries the same four dimensions every listing endpoint takes and is resolved
through the same `FilterResolverDep`, so the set that moves is exactly the set the same
filter shows in the grid — quality flags and collection membership included, neither of
which is a property of a stored row. Those are precisely the sets worth quarantining, and
they are scattered across the whole embedding cloud, so no rectangle drawn on the map can
approximate them.

The four filter fields are declared once, in `_CorpusFilterFields`, and inherited by
`SearchRequest`, `_SelectionRequest` and `CollectionMoveFilter`. Three identical copies is
the shape that drifts, and "the set you are looking at is the set you can move" only holds
while all three stay identical.

The service resolves the filter to ids and hands them to the **same** move path an explicit
selection takes, so `{moved, unchanged, unknown}` and the
move-to-your-own-split-clears-the-override rule keep working unchanged. `unknown` is
therefore always empty on the filter channel — every id came out of the index a moment ago —
and is still reported, because one shape for both channels is worth more than a field that
sometimes is not there.

**The cap is 8 000 images per move, and it applies to both channels** (`413` above it,
naming the count and the ceiling). The bound that matters is
not the request body: a filter-driven move sends four short fields and can address the whole
corpus. It is what the move *leaves behind*. Every override becomes a literal in the id lists
`filters.py` embeds in every later filtered query, and the cost is linear in the total.
Measured on the real 8 000-row table: a filtered `count_rows` takes 22 ms with no override,
222 ms at 1 000, and 1.6 s once the corpus is fully re-partitioned into a 194 KB predicate.

**So the ceiling is on the accumulated overlay, not on one request** — checking the batch
bounds nothing, because the same total is reachable as eight moves of a thousand. It is
`CORPUSLENS_MAX_COLLECTION_OVERRIDES`, default **1 000**: the last size at which a filtered
query is still interactive, with room for every set the overlay is *for* — the 32 cross-split
duplicates, the 200 weak captions, a holdout of a few hundred. `MAX_COLLECTION_MOVE_IMAGES`
in `schemas.py` separately bounds the request array to the corpus size; a body under that is
still refused with a `413` when it would push the total over.

The check runs *after* the reset/move split is known and counts what the overlay would hold
afterwards, so a move that only returns images to their splits always succeeds — a store at
the ceiling can still be emptied. Re-partitioning a whole corpus is a different operation and
belongs in a newly ingested index; see [`collections-next.md`](./collections-next.md).

**The destination is checked before the selection.** A move whose every image resolves to a
*reset* — each one going back to its own split — never reaches the store's foreign key, so
without an up-front existence check a typo'd destination would answer `200` and silently do
nothing. Same for a filter that matches nothing.

### Provenance

A partition without a recorded reason is not reproducible, and reproducibility is the
deliverable. Three weeks on, a collection holding 32 images says nothing about the flag and
threshold that put them there. So every assignment records **how it was made**, alongside the
`moved_at` the store had been writing since it existed and exposing nowhere:

| `origin` | Set by | `detail` |
|---|---|---|
| `manual` | default; a hand-picked image or a rectangle on the map | `null` |
| `filter` | the server, on the filter channel | the filter, as compact JSON |
| `import` | the client, on a pasted or uploaded id list | `null` |

`import` has to come from the client, because a pasted list and a lassoed one arrive at the
API identically — hence the `origin` field on the request, which accepts only `manual` and
`import`; `filter` is the server's to record.

The stored `detail` is serialised from the **request**, not from the resolved `ImageFilter`.
By resolution time a quality flag has been expanded into the ids it happened to mean today,
and `{"quality_flag":"cross-split-duplicate"}` is the reproducible record — the 32 ids are
not. Only fields the caller actually set are included (`model_dump_json(exclude_defaults=True)`),
so the record reads as the filter rather than as a wall of empty lists.

`GET /api/collections` reports the provenance of each collection's **most recent** batch, not
a history. That is the question worth answering cheaply, and it fits how collections are
actually populated — one filter, one import, one lasso. A collection built from several
batches reports the last of them; that is stated rather than implied.

**Migration.** SQLite has **no** `ALTER TABLE … ADD COLUMN IF NOT EXISTS` — verified against
the engine bundled with this Python 3.12.13 (SQLite 3.53.1), where it is a syntax error near
`EXISTS`. The `IF NOT EXISTS` trick the rest of the schema leans on therefore cannot add a
column to a table that already exists, so the migration reads `PRAGMA table_info` and adds
what is missing, which is idempotent in the same way. Also verified there: a plain
`ADD COLUMN … NOT NULL DEFAULT 'manual'` succeeds on a populated table and backfills the
existing rows. Those rows read as `manual`, which is truthful — the filter and import
channels did not exist when they were written. A test builds a store on the old schema and
asserts it opens, keeps its data, and reports that.

Reading it back uses SQLite's documented min/max special case: with exactly one `MAX()`
aggregate, bare columns in the same `SELECT` take their values from the row that produced it.
Verified on the same engine rather than assumed; in any other engine that would be an
unspecified row.

### Importing an id list

There is no import endpoint. A pasted list posts through `POST /api/collections/{id}/images`
with `origin: "import"`, because it *is* a move — inventing a second write path for the same
operation would be two places for the overlay semantics to drift apart.

`CollectionMoveResponse.unknown[]` was already the right reporting channel for ids that are
not in this corpus, and the client surfaces it rather than swallowing it: "37 of your 400 ids
are not in this corpus" is the single most useful thing an import can tell you. Tokens that
cannot be an image id at all are filtered out in the browser and reported alongside it,
because one of them would otherwise `422` the whole batch with a message naming a field index.

## Getting the partition out: `scripts/export_split.py`

Everything above keeps the partition inside the tool. A collection reached disk only as a
CSV downloaded through a browser, no training run could be pointed at
`data/collections.db` in any supported way, and nothing under `scripts/` knew collections
existed — so the output of the work was a file in `~/Downloads`.

```bash
python scripts/export_split.py                 # data/splits.json
python scripts/export_split.py --format csv    # data/splits.csv, for pandas
python scripts/export_split.py --force         # rewrite regardless
```

It performs the same join the serving layer does — *split, unless an override says
otherwise* — and writes image id → effective collection **with the ground-truth split kept
beside it**, plus a header block naming every collection, its size, and the provenance of its
most recent batch.

- **Read-only with respect to both stores.** The index is opened for reading like every other
  script here, and `collections.db` is opened through SQLite's `mode=ro` URI, which refuses
  writes at the engine rather than by convention — verified on this Python's SQLite 3.53.1.
  This is the sanctioned bridge out; it does not make `scripts/` a writer.
- **It must therefore tolerate a store it cannot migrate.** A `collections.db` written before
  the provenance columns existed exports without them and says so, rather than demanding the
  API be started first — the offline bridge depending on the online one would be the wrong
  way round for the file a training run reads.
- **Idempotent on the partition, not the clock.** Without `--force` it compares everything but
  `generated_at` and keeps the existing file when nothing has changed. Comparing whole files
  would rewrite on every run to record that nothing happened.
- **CSV rows carry ids only.** The header rides above them as `# ` comment lines so
  `pd.read_csv(path, comment="#")` works, and a collection name containing `#` would break
  exactly that reader — so names live in the header block, joinable on `collection_id`. JSON
  is the format that keeps the provenance as data.
- Written atomically through a temporary sibling, like the other artefacts, so a reader never
  sees half a file.

## Decisions

**Layering.** `api/routes → services → repositories → LanceDB`, per CLAUDE.md §4.1.
`repositories/image_repository.py` is the only module importing `lancedb`; services import
no web framework and are testable without an HTTP client. The initial task spec proposed a
flat `main.py` / `schemas.py` / `api.py`; the layered structure was chosen instead after
raising the conflict.

**Singletons live in the lifespan.** `app/core/lifespan.py` loads the CLIP bi-encoder and
opens the table once and hangs a frozen `AppResources` dataclass off `app.state`. Loading
the model costs ~5 s of CPU here — per-request loading is not an option, and import-time
loading would make the module unimportable without a populated `data/`. A typed container
on one well-known attribute keeps the otherwise-untyped `app.state` checkable by mypy.

**Blocking work is offloaded.** LanceDB's client is synchronous and torch inference is
CPU-bound. Routes are `async`, and services push both onto anyio's worker-thread pool, so
a scan or an encode cannot stall the event loop.

**Query encoding is the one runtime inference.** CLAUDE.md §2 permits it: a short string is
a ~tens-of-milliseconds forward pass, against ~15 minutes for the 8k-image corpus.
`normalize_embeddings=True` is passed explicitly — it is *not* the sentence-transformers
default, it matches what ingestion applied to the image side, and cosine ranking is only
meaningful when both sides are unit length. `encode()` is used rather than 5.x's
`encode_query()`, which would apply a retrieval prompt template CLIP has no notion of.

**Distance vs. score.** LanceDB returns `_distance`; with the cosine metric on unit vectors
that is exactly `1 - cosine_similarity`. The repository inverts it so nothing above that
layer knows the store reports a distance. Note that CLIP's modality gap makes absolute
text↔image similarities low (~0.2–0.35 for good matches) — rank order is the signal, not
the magnitude.

**Exact by default, approximate only where it is free.** The shipped corpus carries no ANN
index — at ~8k rows a brute-force scan is ~21 ms and exact, and an index tuned back to
honest recall costs the same (see `scripts/build_index.py` for the measurements). Searches
therefore return exact nearest neighbours.

Where a corpus is large enough that `scripts/build_index.py` has built an index, the
repository picks per query:

| Situation | Path | Recall |
|---|---|---|
| No index on the table | exact scan | exact |
| Index, no filter | IVF-PQ + `refine_factor` | ~1.000 measured |
| Index, filter leaving ≤ `CORPUSLENS_EXACT_SCAN_CEILING` rows | index bypassed | exact |

The third row is a correctness rule, not a tuning choice. An IVF pre-filter is applied
within the probed partitions, so a selective filter starves the candidate pool: measured on
a 200k corpus, filtering to 20% of rows dropped recall@20 from 1.000 to 0.71 while still
returning a full page. Since a selective filter also makes the exact scan cheap, the exact
path is taken instead. `CLAUDE.md` §4.4 is the contract.

**Pagination ordering** is the table's scan order — the order ingestion wrote rows in
(train, then validation, then test), verified stable across repeated calls. This is a
property of a static, never-compacted, read-only table rather than a sort guarantee from
LanceDB; an explicit sort key would be required if the index ever became writable at
runtime.

**Search is `POST`.** The query is body-shaped input that will grow options (split filter,
image→image), and a `GET` would invite caching of results that depend on the index rather
than on the URL alone.

**Input hardening.** Image ids are constrained to `^[A-Za-z0-9._-]+$` at the route (a
malformed id is a `422`, never reaching the store), and the repository additionally escapes
single quotes before embedding an id in a filter expression — defence in depth for the one
place a caller-supplied string reaches a query expression.

**CORS** allows `http://localhost:5173` and `http://127.0.0.1:5173` with credentials off —
the API has no auth and sets no cookies. Methods and headers are enumerated rather than
wildcarded. Override with `CORPUSLENS_CORS_ALLOW_ORIGINS`, which takes either a
comma-separated list or a JSON array.

**Bind address.** `CORPUSLENS_HOST` defaults to `127.0.0.1`. Because the API has no
authentication, binding `0.0.0.0` exposes the corpus to the local network; the container
image opts into it because the container boundary is what limits reach there. The setting
is read by `python -m app` — the `uvicorn` CLI takes `--host`/`UVICORN_HOST` instead.

## Dependency note

`fastapi` is installed plain, deliberately **not** `fastapi[standard]`: that extra pulls in
`fastapi-cloud-cli`, which CLAUDE.md §2 rules out. `uvicorn` is likewise unextra'd —
`uvicorn[standard]`'s uvloop/httptools buy nothing measurable for a single-user localhost
tool, and both ship native code that would need macOS x86_64 wheels.

## Verification status

All endpoints exercised against the live server on the **full 8 000-image index**: stats,
first/tail pagination pages, detail, `404` on an unknown id, `422` on a malformed id, static
image delivery, path-traversal rejection, CORS preflight (allowed and refused origins),
search-payload bounds, and two semantic queries returning sensible cross-modal rankings.

The filtering, projection and export paths were verified the same way, against the real
store rather than the double:

| Check | Result |
|---|---|
| `count_rows(filter=…)` for `split = 'test'` | 1 000 of 8 000 |
| `caption_contains=dog` | 2 014 rows; with `split=train`, 242 |
| `caption_contains=%` (literal) | 0 rows — escaping holds |
| `split=TEST` (malformed) | `422` |
| Filtered k-NN, `limit=5`, `splits=["test"]` | 5 hits, all in `test` — pre-filter confirmed |
| `GET /api/projection` | 695 KB raw, 174 KB gzipped, 0.28 s |
| `GET /api/projection?split=test` | `count` 8 000, `match_count` 1 000 |
| Export of the `train`+`dog` slice | 1 528 rows in 0.37 s, `score` column empty |
| Ranked export | scores present and descending; coordinates populated |
| `analyze.py --no-captions` on the full corpus | 52 pairs above 0.95, **22 crossing a split**, ~4 s |
| Full `analyze.py` | R@1 29.82%, R@5 53.29%, R@10 63.28% over 40 000 captions, ~9 min |
| `quality_flag=near-duplicate` / `cross-split-duplicate` / `weak-captions` | 63 / 32 / 200 images |
| `quality_flag=cross-split-duplicate&split=test` | 10 images — the flag intersects the split filter |
| `POST /search` with `image_id` of the duplicated kayak photo | its byte-identical twin first, cosine 1.0 |

The collection overlay was verified the same way, against the real 8 000-image index,
after 200 real `train` images were moved into a new collection:

| Check | Result |
|---|---|
| Built-ins seeded from the index | `train` 6 000 · `validation` 1 000 · `test` 1 000 |
| Duplicate name, case-insensitive (`MY HOLDOUT` vs `My holdout`) | `409` |
| Rename / delete a built-in | `403` |
| Move 200 images, then repeat it | `{moved: 200}`, then `{moved: 0, unchanged: 200}` — idempotent |
| **`images_by_split` after the move** | **unchanged** — `train` 6 000 · `validation` 1 000 · `test` 1 000 |
| `images_by_collection` after the move | `train` 5 800 · holdout 200 · rest unchanged |
| Filter by the new collection | 200 of 8 000; `train` now reports 5 800 |
| **Compound predicate as a vector pre-filter** (`prefilter=True`) | parses; 5 hits in 0.07 s, excluded id did not leak |
| `count_rows(filter=…)` on the compound predicate | accepted |
| **Parenthesisation** — `train`+holdout, `caption_contains=dog` | **1 528**; caption alone 2 014; unparenthesised would give 5 999 |
| Search with `collections=[holdout]` | all 3 hits inside the collection — pre-filter confirmed |
| `GET /projection?collection=…` | `count` 8 000, `match_count` 200 |
| Unknown collection id in a filter | `total` 0 — unsatisfiable, not ignored |
| Delete the collection | `204`; all 199 remaining members reverted, `images_by_collection` back to the splits |
| CORS preflight for `DELETE` | `GET, POST, PATCH, DELETE, OPTIONS` |
| `data/collections.db` + `-wal`/`-shm` sidecars | land under the gitignored, Docker-mounted `data/`; `git status` clean |

`ruff check`, `ruff format --check` and `mypy --strict` all pass.

The manual pass is backed by an automated one: `backend/tests/` (174 cases, ~6 s) drives the
same contract through `TestClient`, with only `lancedb.connect` and
`ClipEmbeddingService.load` stubbed — the lifespan, dependency wiring, services, repository
row mapping and Pydantic serialization all run for real. Note that `ClipEmbeddingService`
itself is *not* stubbed, only the checkpoint behind it, so the encoding contract the shared
space depends on (`normalize_embeddings` and the rest) is exercised. Fixtures and the
in-memory LanceDB double live in `backend/tests/conftest.py`.

`test_architecture.py` covers the seams instead of the behaviour: it defines a vector store
and an encoder that inherit from nothing, injects them with `app.dependency_overrides`, and
asserts the real routes serve their data unchanged. That is what keeps "swap LanceDB for
Qdrant without touching a router" an enforced property rather than a claim in a README.

`test_export_split.py` is the one module that does not go through the HTTP boundary: the
offline export's join, its two renderers and its idempotence check are pure functions, and
that is the shape worth testing directly. Its LanceDB projection is not covered — four lines
of query builder, where a double would assert our idea of lancedb rather than lancedb.

That double **evaluates** filter predicates instead of recording them, and raises on any
clause shape it does not recognise. Without that, every filtering test would pass whether
or not the filter did anything. It splits conjunctions at parenthesis depth zero and models
a parenthesised group as a disjunction of conjunctions of the leaf shapes it already knows —
which is exactly what a collection selection is, and nothing else emits one. Dropping the
parentheses from the predicate makes it fail loudly rather than quietly widen the result.

`test_filters.py` exists because two properties of the collection predicate are about the
*text* rather than the rows it selects — the no-regression guarantee and the
parenthesisation — and neither is visible from the API boundary until it is already wrong.

**Not yet covered:** the repository is tested only against that hand-written double, so no
automated check would notice if lancedb's real query-builder semantics drifted from what it
models (vector dimensionality, filter parsing and scan ordering are all approximated) —
including whether the real engine still accepts the compound collection predicate as a
vector pre-filter, which is why that check is in the manual table above rather than the
suite. Path-traversal rejection and CORS preflight remain manually verified — both are handled by
middleware and static-file mounting that `TestClient` does not exercise the same way. There
is no suite below the API boundary.
