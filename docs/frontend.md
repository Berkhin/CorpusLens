# Frontend — contract and design decisions

Recorded per CLAUDE.md §7. Covers the React client only; the API contract is in
[`api.md`](./api.md) and the offline pipeline in `scripts/ingest.py`.

## Running

```bash
npm --prefix frontend install
npm --prefix frontend run dev          # http://localhost:5173
```

The backend must already be serving on `http://localhost:8000` — the client is a pure
consumer of the API and has no fixtures or mock mode.

| Script | Purpose |
|---|---|
| `npm run dev` | Vite dev server on a pinned port 5173 |
| `npm run build` | `tsc -b` then a production bundle into `dist/` |
| `npm run typecheck` | `tsc -b` alone |
| `npm run test` / `test:watch` | Vitest |
| `npm run lint` | ESLint (flat config) |
| `npm run format` / `format:check` | Prettier |

## Structure

```
src/
├── App.tsx                  # shell: header (tabs, search, filters, export) / main / footer
├── main.tsx                 # QueryClient + providers
├── components/ui/           # shadcn primitives — vendored, not hand-edited
├── components/StatusPanel.tsx
├── features/
│   ├── gallery/             # browse grid, cards, pagination hook, view model
│   ├── search/              # search bar, ranked results, search hook
│   ├── filters/             # collection, caption-text and quality filters; wire encoding
│   ├── collections/         # the user's corpus partition: chips, manager, move menu
│   ├── projection/          # embedding map: canvas, viewport maths, palette, hover
│   ├── export/              # CSV/JSONL manifest button, with its mutation
│   ├── inspector/           # detail dialog + detail hook
│   └── stats/               # footer counts
├── lib/                     # api-client, query-keys, cn()
├── styles/tailwind.css      # Tailwind v4 CSS-first entry
└── types/                   # wire-format mirrors of the Pydantic DTOs
```

Every feature slice owns its own hooks, so the scaffold's `src/hooks/` and `src/pages/`
never acquired a file and have been removed (CLAUDE.md §7: no speculative structure).
Components render, hooks fetch (CLAUDE.md §4.3).

## Decisions

**Server state is TanStack Query; client state is four `useState` calls.** `App` owns the
search target, the corpus filter, which view is showing, and the selected image — and
nothing else. Query and filter live there rather than per view because they apply to all
of them: narrowing to one split, searching, then switching to the map must carry both
constraints across. The task brief asked for hooks or a light fetch
wrapper and warned off Redux-style stores; TanStack Query is a server-cache rather than a
global store, and CLAUDE.md §3 mandates it, so the conflict was raised and resolved in its
favour. Search progress reaches the header through `useIsFetching` rather than by lifting the
query out of its view.

**`staleTime: Infinity`.** The index is built offline and the API only reads it, so no
response can change while the app is open — *with one carve-out since collections landed;
see "Collections are the one thing that goes stale" below.* Refetching would be pure waste — most sharply for
search, where a cache miss costs a CPU forward pass through CLIP's text encoder. Retries are
suppressed for any status below 500: a 404 or a 422 is a verdict, not a hiccup.

**Search hits seed the detail cache.** `SearchResultResponse.image` is a full
`ImageDetailResponse`, captions included, so opening the inspector on a search result passes
that record to `useImageDetail` as `initialData` and renders with no second request. Opening
the same image later from the browse grid is then served from the same cache entry. A grid
item carries no captions and does fetch.

**One grid, two sources.** Browse returns summaries; search returns ranked details. Both are
normalised into a `GalleryItem` view model in `features/gallery/gallery-item.ts`, so the grid
and card components never branch on where a row came from.

**Image URLs are resolved, not composed.** The API returns `image_url` root-relative on
purpose (see `api.md`); the client only prefixes the API origin. Composing `/images/` +
`file_name` in the client would duplicate a decision that belongs to the backend's static
mount. `API_BASE_URL` defaults to `http://localhost:8000` and is overridable via
`VITE_API_BASE_URL` (see `frontend/.env.example`). Vite inlines the value at build time
rather than reading it at runtime, so changing it means rebuilding, not restarting.

**No dev proxy.** The API already whitelists `localhost:5173` and `127.0.0.1:5173` for CORS,
so the client calls it cross-origin directly. Vite's port is pinned with `strictPort: true`
— a silent fallback to 5174 would surface as an opaque CORS failure rather than a bind error.

**Pagination cursor uses rows returned, not the requested limit.** `getNextPageParam` returns
`offset + items.length`, which stays correct on a short final page. `initialPageParam` is
required by TanStack Query v5. The Load More button is guarded on `isFetching` because an
infinite query shares one cache entry across pages.

**Similarity is shown, and framed.** Scores are rendered to three decimals in a badge, with a
line under the result header explaining that CLIP's modality gap keeps good text→image
matches around 0.20–0.35 and that rank order — not magnitude — is the signal. A researcher
reading `0.31` as "31% match" would be misreading it.

**Wire types stay snake_case.** `types/api.ts` mirrors `backend/app/models/schemas.py` field
for field. Renaming to camelCase would buy idiomatic TS at the cost of a mapping layer whose
only job is to hide that the backend speaks Python; identical names keep drift greppable.

**Collections are the one thing that goes stale.** They are the only mutable state the API
exposes, and a move is invisible to the cache: it changes the *result set* of every
collection-filtered query while leaving the cache key byte-for-byte identical, because the
filter the user selected did not change. Under `staleTime: Infinity` nothing would refetch
and the grid would keep serving pre-move pages. So every successful create/rename/delete/
move/reset invalidates `collections.all`, `dataset.all`, `search.all` **and**
`projection.all`. That over-fetches slightly — a rename cannot change a result set — but the
alternative is a per-mutation matrix of which keys a given change can reach, which goes
stale the first time someone adds a field. `useCollectionMutations` says so in a comment,
because it reads like something to optimise away and is not.

**Import goes through the move endpoint, and its report is the feature.** The manager dialog
takes a pasted or uploaded id list and posts it as an ordinary move with `origin: 'import'` —
inventing a second write path for the same operation would be two places for the overlay
semantics to drift. `parseImageIdList` splits on whitespace, commas and semicolons, strips the
quotes a CSV writer adds, collapses repeats, and separates tokens that cannot be an image id
so one of them cannot `422` the whole batch. What comes back is rendered rather than
swallowed: `moved`, `unchanged`, and — the one that matters — `unknown`, because "37 of your
400 ids are not in this corpus" is the reason someone imports a list in the first place. The
parser is the second unit-tested module.

**Provenance is a label on the row, and the filter case is why it exists.** Each collection
shows how its most recent batch was made — `manual`, `filter` or `import` — with the stored
filter JSON verbatim in the tooltip. Verbatim rather than prettified into prose: it is the
request that produced the set, and you can paste it back.

**Deleting a filtered-on collection prunes the filter at the mutation site.** The manager
dialog's delete handler drops the id via `onChange`, rather than a `useEffect` watching for
ids that stopped resolving. The component already knows exactly which id disappeared;
reacting to that afterwards would be inferring something it was told.

**`splits` stays in the filter type, unused by the UI.** The built-in collections *are* the
splits, so the collection group covers the ground, and the UI no longer sets `splits`. The
field remains because it mirrors a real API dimension the backend still accepts — and
because the two mean genuinely different things: `split` is the dataset's immutable
partition, which the leakage figures are computed from, and `collection` is the user's
working one. A comment in `image-filter.ts` says so, so nobody later "cleans up" one into
the other.

**Three bulk-move surfaces, one control.** `MoveToCollectionMenu` takes a
`CollectionMoveSource` — a discriminated union of `{kind: 'ids'}` and `{kind: 'filter'}`,
mirroring `SearchTarget` for the same reason: the backend requires exactly one and a client
shape that cannot express the invalid state cannot send the invalid request. That one
control then serves the map's rectangle, the grid's multi-select, the inspector's single
image, and the filter bar's "everything matching" — and the two filter-shaped surfaces never
have to materialise thousands of ids in the browser to use it.

It always states the count it will move *before* it acts, because the filter channel can
address images that are not on screen.

**"Move N matching the filter…" reads its count from the grid's own query.** `useFilteredTotal`
is `useImageList(filter)` with everything but `pages[0].total` thrown away — the same key,
so the same cache entry, so the same number the grid renders. A separate count endpoint would
be a second number to keep in agreement, and the failure mode of disagreeing is a control
that promises 200 and moves 5 000. On the map tab the grid is not mounted, so this does cost
one 50-row page request; that is what buys the guarantee.

The label spells out "matching the filter" because a search can be running at the same time:
the ranked view then shows twenty hits while this still addresses everything the *filter*
selects. The move endpoint has no ranking channel.

It is shown only when a filter is active. Unfiltered it would be a one-click "move all
8 000 images" sitting in a toolbar aimed at nobody.

**The grid has multi-select: a checkbox on hover, not a selection mode.** A mode is a state
to enter, remember you are in, and leave; a checkbox leaves the card's primary action —
open the inspector — exactly where it was and puts the second action next to it. It is a
**sibling** of the card's `<button>`, never a child: nesting an interactive element inside a
button is invalid and browsers disagree about which one a click reaches. The selection lives
in `ImageGrid`, so the ranked search results get it too — before this, a search hit was
something a researcher could look at and not act on. The set is stored as ids and
intersected with `items` on render, so changing the filter drops ids that left the screen
without a `useEffect` reacting to a prop it was already handed. The cost is that it is
invisible until pointed at, so there is no touch affordance; the filter bar carries the bulk
action for sets too large to click through anyway.

**Collection ids are never rendered.** `useCollectionLabel` resolves an id to its display
name and returns `null` until the collection list loads. `ImageCard` and the inspector
render nothing rather than falling back to the id, because the fallback *looks correct*:
a built-in's id is its split name, so `train` reads fine and a user collection renders
`376a6824e79b41f8b0df914b0a2baaf4`. The id remains the value every comparison and mutation
uses; only the text changes. The hook is called once per view and the name is stamped onto
the `GalleryItem` in the mapper — one subscription to the collections query per view rather
than one per card, which at 250 cards would re-render the whole grid on every move.

**The map still colours by split, not collection.** `ProjectionPoint` carries no collection
field. The palette has five chart slots and the map is the one view that still shows the
corpus as the dataset actually partitions it; collection membership only reaches it through
the binary `matches` flag, for one filter at a time. That is a real gap rather than a
protection — see the "colour by" proposal in the PR description.

**Vitest, for the pure modules only.** Added as a dev dependency (peer range
`vite: ^6 || ^7 || ^8`, verified against the installed Vite 8.2.1) and configured in
`vite.config.ts` through `defineConfig` from `vitest/config` — one config file, so the `@`
alias is defined once. `environment: 'node'`: what is under test is arithmetic over a
`Float32Array` and string parsing, and jsdom would be a dependency bought for nothing —
the canvas component needs a real browser rather than a simulated one to be worth testing.
There is still no component or e2e suite; the UI verification below is scripted but manual.

## Deviations from the scaffold and the docs

**oxlint → ESLint + Prettier.** The current `create-vite` react-ts template ships oxlint.
CLAUDE.md §3 specifies ESLint + Prettier, so oxlint was removed and a flat config added.
`src/components/ui/**` is excluded from both, since lint autofixes and reformatting would
amount to hand-editing vendored files.

**`"strict": true` set explicitly.** The template's `tsconfig.app.json` no longer includes
it. Added alongside `noUncheckedIndexedAccess` and `noImplicitOverride` per CLAUDE.md §5.2.

**`baseUrl` omitted.** The shadcn Vite install guide instructs adding `"baseUrl": "."`
next to `paths`, but TypeScript 6.0 — the version this project installs — errors on `baseUrl`
as deprecated (TS5101). `paths` alone resolves relative to the tsconfig, so it is omitted
rather than silenced with `ignoreDeprecations`.

**API client at `lib/api-client.ts`.** The task brief named `src/api/client.ts`; CLAUDE.md
§4.3 assigns the API client to `lib/`. Placed per CLAUDE.md for consistency with the two
larger conflicts above.

## Verified against the current stack

Checked against official docs and the installed versions, per CLAUDE.md §6:

- **Tailwind v4.3.3** — Vite plugin install path: `@tailwindcss/vite` in `plugins`, single
  `@import "tailwindcss"` in CSS, no `tailwind.config.js`, no PostCSS/autoprefixer.
- **shadcn CLI 4.16.2** — `init` now takes `-b/--base` (base/radix/aria) and `-p/--preset`;
  the older `--base-color` flag no longer exists. Initialised with `-b radix -p nova`, which
  uses lucide-react as its icon set.
- **TanStack Query 5.101.4** — `useInfiniteQuery` requires `initialPageParam`;
  `getNextPageParam(lastPage, allPages, lastPageParam)`.
- **React 19.2.8**, **Vite 8.2.1**, **TypeScript 6.0.2**.

**The search target is a union, not two nullable fields.** `SearchTarget` is
`{kind: 'text', query}` or `{kind: 'image', imageId}`. The backend requires exactly one and
rejects a body carrying both, so a client shape that cannot express the invalid state is a
client that cannot send the invalid request. It also means the ranked views, the map's
highlight and the export scope all switch on one discriminant rather than each re-deriving
"are we searching by text or by image".

**The quality findings are filters, not a report.** `near-duplicate`, `cross-split-duplicate`
and `weak-captions` sit in `FilterBar` beside the split chips, and the backend resolves each
to a set of ids. That is the whole reason there is no "Data quality" tab: a finding
delivered as a filter arrives already composed with the split and caption filters, already
paginated by the existing grid, and already exportable. The row is hidden unless
`analysis_available` — a control that can only ever return nothing is worse than no control.

**The inspector no longer seeds from a search hit.** It used to, to save a request. The
detail endpoint now also returns the image's quality measurements, which a search hit does
not carry, so seeding would populate the shared cache entry with `analysis: null` —
asserting the absence of a measurement that exists. One extra request over loopback is the
cheaper mistake.

## The map view

`features/projection/` is the one slice that is not a list of cards, and it is split so
that no file mixes concerns that fail differently:

| File | Owns |
|---|---|
| `useProjection.ts` | The query (`staleTime: Infinity` — a build artefact cannot change mid-session) and the `Float32Array` the hot loops walk |
| `scatter-viewport.ts` | Pure maths: world↔screen, hit test, box query. The one unit-tested module — `scatter-viewport.test.ts` |
| `scatter-render.ts` | Pixels — batched fills, one path per colour |
| `scatter-palette.ts` | Reading theme colours out of CSS custom properties |
| `ScatterCanvas.tsx` | Events and interaction state |
| `HoverCard.tsx`, `ProjectionView.tsx` | Preview, legend, caption, selection toolbar |

Three decisions in there are load-bearing:

**One transform, derived in one place.** `scatterTransform()` produces the origin and
scale that the renderer, the hit test and the box query all use. They each inline the
arithmetic — they run over thousands of points per frame — but none of them *derives* it.
If any of the three computed its own, points would draw in one place and respond to the
cursor in another, and the symptom would be "clicking is slightly off" rather than
anything greppable.

**The box query selects by filter, not only by geometry.** `pointsInBox` takes the per-point
`matches` array — the same flag the renderer uses to dim filtered-out points — as a
**required** parameter, so a new caller cannot silently reintroduce geometry-only selection.
It was geometry-only, and that fed the bulk move: measured on the live 8 000-image index with
Weak captions active (200 matching), one shift-drag over the dense right-hand lobe selected
**5 379** points, 98 of which actually matched. The same drag now selects 98. With no filter
active every point matches and the behaviour is unchanged, which is what makes the fix
additive rather than a change of meaning. The selection label names its denominator —
"98 selected of the 200 matching the filter" — because a bare count beside a dimmed cloud of
8 000 invites the reading that the other 5 281 went somewhere.

**`useElementSize` returns a callback ref, not a `useRef` object.** The view renders a
skeleton while the projection loads, so the measured element does not exist on first
mount. An effect keyed on a ref object runs once against `null`, never attaches the
observer, and never re-runs — the canvas stays at its default 300×150 and paints nothing.
The headless check therefore asserts on the canvas's backing-store size rather than
trusting a screenshot, because a blank canvas and a canvas of the wrong size look alike.

**Split colours come from `--chart-1…5`.** Shadcn's neutral base defines those five slots
as a greyscale ramp, which cannot carry identity — three splits would be three
indistinguishable greys — so this theme defines them as hues instead, in fixed order,
validated as a set against both theme surfaces for colour-vision deficiency and contrast. Identity is never carried by
colour alone: the legend and the hover card both name the split in text.

## Verification status

`typecheck`, `lint`, `format:check` and `build` all pass. The UI was driven in headless
Chrome over the Chrome DevTools Protocol against the live API and the **full 8 000-image
index**:

- gallery renders 25 cards; footer reports 25 total / 25 train
- inspector opened from a grid card fetches and shows all five captions; Escape closes it
- `a dog running through shallow water` returns 25 ranked hits, top score 0.344 on a dog
  wading in water, descending sensibly
- inspector opened from a search hit shows captions plus `similarity 0.3440`
- Load More paginates 10 → 20 → 25 and disappears on the short final page (temporarily
  reducing `GALLERY_PAGE_SIZE`, since 25 rows never fill the real 50-row page)
- the map paints all 8 000 points (canvas backing store matches its CSS box; 62 625
  non-transparent pixels) with the three splits in distinct hues
- selecting the `test` split leaves the caption reading "8 000 images · 1 000 match the
  filter, the rest are dimmed"
- hovering a point shows its thumbnail, id, split badge and first caption
- shift-dragging a rectangle selects 4 118 points and reveals "Export 4 118 selected"
- with a search active, its 30 hits render in the ink colour at double radius and sit in
  one region of the cloud rather than scattered through it
- the quality row appears only with an analysis loaded; selecting "Split leakage" narrows
  the grid to 32 images and the footer reports "22 pairs"
- the inspector shows the nearest neighbour as a clickable id, flags it in the destructive
  colour above the 0.95 threshold, and reports the image's own caption rank
- "Find similar" switches the ranked view to that image's neighbours

Re-verified after the Stage 1 changes, same harness, against the live index (with the
mutable collection store pointed at a throwaway copy so the timing experiments could not
touch real state):

| Check | Before | After |
|---|---|---|
| Shift-drag over the dense lobe, Weak captions active (200 match) | **5 379 selected** | **98 selected of the 200 matching the filter** |
| Grid badge for an image in a user collection | `376a6824e79b41f8b0df914b0a2baaf4` | `new test` |
| Filter bar, that collection selected | no bulk affordance | `Move 1 matching the filter…` |
| Grid multi-select | absent | 50 checkboxes, 3 clicked → `3 selected · Clear · Move to… · Export 3 selected` |

And directly against the API on the full corpus: a filter-driven move of the 200
weak-caption images returned `{moved: 200}` in 0.38 s; repeating it returned
`{moved: 0, unchanged: 200}`; `images_by_split` stayed 6 000 / 1 000 / 1 000 throughout and
`cross_split_duplicate_pairs` stayed 22.

**Not yet covered:** the only automated frontend tests are the 16 Vitest cases over
`pointsInBox` and `parseImageIdList` — the two modules that are pure functions. There is no
component suite (no React Testing Library) and no e2e, so nothing
guards the wiring the table above checks by hand. The `ErrorNotice` paths were not exercised against a real
failure (the backend never errored during the run), and the app has only been viewed at
1600×1100 — the responsive ramp below `sm` is unverified, and the map assumes a pointer:
there is no touch pinch-zoom. Only the light theme was exercised; the dark palette was
validated numerically but never rendered. The caption/vocabulary statistics named as a goal
in `CLAUDE.md` are still unbuilt on both sides, so the UI has no surface for them.
