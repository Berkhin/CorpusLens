# CorpusLens — Full Guide

> The long-form manual: setup in detail, the API surface, design decisions and an honest
> account of what is and isn't done. For the short version, start at the
> [README](../README.md).

**A local, offline visualization and exploration tool for image-captioning corpora — an
interactive map of the CLIP embedding space, semantic search, filtering, caption
inspection and subset export, running entirely on one machine. Ships with Flickr8k as
the reference corpus.**

---

## Why this exists

Flickr8k pairs ~8,000 photographs with five human-written captions each. It is a standard
benchmark for image captioning and vision–language grounding, but it ships as a directory
of JPEGs beside a text file — which makes the questions a CV researcher actually asks
about a corpus tedious to answer:

- *Which images does a vision–language model consider close to "a dog running through
  shallow water"?* — a question keyword search over captions cannot answer, because the
  captions may never use those words.
- *What do the reference captions for this particular image look like side by side?* —
  annotation disagreement and degenerate captions are visible instantly in a grid, and
  invisible in a text file.
- *Where does CLIP's notion of similarity diverge from the human annotation?* — the tool
  surfaces the ranked similarity next to the captions, so the two can be compared directly.

- *Is my `test` split drawn from the same distribution as `train`, or does it sit in its
  own corner of the embedding space?* — a question that is a glance at a coloured scatter
  plot and an afternoon of scripting otherwise.
- *Which subset do I actually want to train on, and how do I get it out of here?* — a
  region drawn on the map or a filter applied to the grid, exported as a manifest.

The tool is a browsing and probing surface for exactly that: an interactive projection of
the whole corpus, natural-language semantic search over a pre-built CLIP index, per-image
caption inspection, split and caption-text filters that apply to all of it, and CSV/JSONL
export of whatever you narrowed it down to — with no cloud service, no external database,
and no network access after the one-time ingestion step.

### What it does today

| Capability | Detail |
|---|---|
| **Gallery browsing** | Paginated grid over the full corpus, incremental "Load more" |
| **Image inspector** | Full-size image with all of its reference captions |
| **Semantic search** | Free-text query → CLIP-ranked images, with similarity scores |
| **Embedding map** | Interactive 2-D projection (PCA or t-SNE) of all 8 000 CLIP vectors — pan, zoom, hover for a preview, click to inspect, shift-drag to select a region |
| **Search by example** | Any image becomes the query. Costs **no inference**: its embedding is already in the index |
| **Data-quality findings** | Near-duplicates, near-duplicates that straddle a split boundary, and images their own captions fail to retrieve — each reachable as a filter, so it composes with everything else |
| **Filtering** | By collection, by literal caption text and by data-quality finding, applied to the grid, the ranking *and* the map — and applied **before** ranking, so "top 20 in `test`" means exactly that |
| **Collections** | Re-partition the corpus into your own named groups. Move a rectangle lassoed on the map, a multi-selection in the grid, one image from the inspector — or **everything matching the active filter**, in one action, which is the only practical way to quarantine a set like "the 200 weakest captions" that is scattered across the whole embedding cloud. `split` stays immutable ground truth and is shown beside the collection wherever they differ |
| **Import** | Paste or upload a list of ids — a training run's 400 failure cases — straight into a collection, and be told which of them are not in this corpus |
| **Provenance** | Each collection records how its most recent batch was made: by hand, by import, or by a filter — with the filter itself stored, so the partition can be re-derived rather than merely inspected |
| **Export** | The current view, a search ranking (with scores), a grid selection, or a region on the map → CSV or JSONL manifest. `scripts/export_split.py` writes the whole partition as a `splits.json`/`.csv` a training script can read |
| **Corpus statistics** | Total image count and per-split breakdown — with every quality figure reported **twice**, against the dataset's partition and against yours, so acting on a finding and measuring the effect are the same loop |

The map is what makes the browsing features worth having: a filter or a query stops being
a shorter list and becomes a *region*, so you can see where in the corpus your subset
actually lives.

**On the shipped corpus the quality pass finds things.** 52 near-duplicate pairs covering
63 images, including one pair of byte-identical JPEGs filed under two ids with two
different caption sets — and **22 of those pairs straddle a split boundary**, which means
an evaluation on those test images is measuring memorisation. Corpus-wide, its own captions
retrieve their image at R@1 29.8% / R@5 53.3% / R@10 63.3% under this CLIP checkpoint.

See [Scope and honest status](#scope-and-honest-status) for what is deliberately *not*
built.

---

## Architecture

The system splits into a **one-time offline pipeline** and a **read-only serving path**.
That split is the central design decision: CLIP inference on an Intel CPU runs at roughly
10 images/s, so embedding the corpus takes ~15 minutes — far too slow to live inside a
request handler. All image embedding happens ahead of time; the API only ever *queries*
what the pipeline produced.

```
  ONE-TIME, OFFLINE                              SERVING (read-only)
  ┌───────────────────────────┐                  ┌────────────────────────────┐
  │ scripts/ingest.py         │                  │ React 19 + Vite + TS       │
  │                           │                  │ Tailwind v4 + Shadcn UI    │
  │  HF datasets              │                  │ TanStack Query             │
  │   └ jxie/flickr8k         │                  │ hand-rolled canvas scatter │
  │  → write JPEGs            │                  └─────────────┬──────────────┘
  │  → CLIP ViT-B/32 encode   │                       HTTP / JSON, :5173→:8000
  │    (CPU, 512-d, batched)  │                  ┌─────────────▼──────────────┐
  │  → write vectors+captions │                  │ FastAPI (Uvicorn, :8000)   │
  └────────────┬──────────────┘                  │ routes → services → repos  │
               │                                 │  • CLIP text encoder,      │
  ┌────────────▼──────────────┐                  │    loaded once at startup  │
  │ scripts/project.py        │                  │  • LanceDB handle, opened  │
  │  → PCA (numpy SVD) or     │                  │    once at startup         │
  │    t-SNE → 2-D coords     │                  │  • projection, read once   │
  ├───────────────────────────┤                  │  • analysis,   read once   │
  │ scripts/analyze.py        │                  └─────────────▲──────────────┘
  │  → nearest neighbours     │                                │
  │  → duplicate pairs        │                                │
  │  → caption retrieval R@k  │                                │
  └────────────┬──────────────┘                                │
               │                                               │
               ▼                                               │
   data/images/ data/lancedb/ projection.json  analysis.json ──┘
   (JPEGs)      (vectors)     (id → x, y)      (findings)
```

The projection and the analysis are **separate offline steps, not part of ingestion**, for
the same reason: each is a property of the corpus *as a whole* — a point's position depends
on every other point, and so does its nearest neighbour — while ingestion is incremental
and resumable, so values computed mid-run would be silently wrong once more rows landed.
Keeping them separate also means recomputing costs seconds rather than a re-embed. Both
artefacts are **optional**: without them the map view and the quality filters are absent
and everything else serves normally.

**Almost nothing is embedded at request time.** A text query costs one short forward pass,
tens of milliseconds on CPU. Search *by example* costs nothing at all — that image's vector
was computed during ingestion, so finding its neighbours is a keyed read and a scan. Text
and images land in the same 512-dimensional CLIP space, so either kind of query is ranked
against image vectors by cosine similarity directly.

### Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| **Backend** | Python 3.12, FastAPI, Pydantic v2, Uvicorn | Async I/O, typed request/response models, OpenAPI for free at `/docs` |
| **Inference** | `sentence-transformers` `clip-ViT-B-32`, CPU-only `torch` | Shared 512-d image/text embedding space; runs locally with no paid API and no GPU |
| **Vector store** | **LanceDB**, embedded | Zero-setup: it is a *directory*, not a service — no daemon, no port, nothing to provision alongside the app. Its columnar Lance format keeps captions and metadata in the same table as the vectors, so one query returns everything a result card needs |
| **Frontend** | React 19, TypeScript (strict), Vite, Tailwind CSS v4, Shadcn UI, TanStack Query | Server state (caching, pagination, retries) owned by the query layer; components stay render-only |
| **Analysis** | `numpy` matrix products over the stored vectors | The whole 8 000-by-8 000 similarity matrix is one chunked matmul, under a second. Only the caption half needs the model, which is why it is separately skippable |
| **Projection** | `numpy.linalg.svd` for PCA; `scikit-learn` for optional t-SNE | PCA in four visible lines keeps the centring, the sign convention and the explained-variance ratio in our own code, where the last of those is the number the view is honest about |
| **Scatter plot** | Hand-rolled 2-D canvas, no plotting library | 8 000 dots redraw in single-digit milliseconds; a WebGL library would be a large dependency for scale this project does not have, and would fight the two behaviours that matter — dimming filtered-out points and rectangle-selecting for export |
| **Quality** | Ruff, mypy `--strict`, ESLint, Prettier, `tsc -b` | Enforced across `backend/` and `scripts/` |

The full engineering contract — layering rules, coding standards, and the constraints that
shaped these choices — is in [`CLAUDE.md`](../CLAUDE.md).

---

## Running it

Two paths. **Docker is the one to use if you just want to see it work**; the native path
is for developing on it.

### With Docker — two commands

```bash
docker compose run --rm setup   # once: download, embed, project. ~15 min, ~3 GB
docker compose up app           # then open http://localhost:8000
```

One container serves the API and the compiled frontend on a single port, so there is no
Node to install, no Python version to match, and no CORS hop. The corpus, the LanceDB
table and the Hugging Face cache land in `./data` on the host, so they survive a
rebuild.

Add `--limit 100` to the setup command for a two-minute end-to-end check before
committing to the full run.

Two things worth knowing: the image is **~2.4 GB** (CPU `torch` alone unpacks to 706 MB),
and the setup step is the same CPU-bound embedding pass it is natively — the container
removes the *setup* friction, not the arithmetic. If the `app` container exits saying the
data directory is empty, you have not run the setup profile yet.

`up app` names the service on purpose. A bare `docker compose up` starts the *contributor*
pair instead — `backend` on :8000 with `--reload` and the Vite dev server on :5173, both
with the source bind-mounted from the host. That is the containerised equivalent of the
native path below, and `make up` is the shorthand for it.

The `torch==2.2.2` and `lancedb==0.25.3` pins have Linux wheels for both `x86_64` and
`aarch64`, so the same `requirements.txt` builds natively on an Intel or an Apple Silicon
host — no emulation, and no container-specific dependency fork.

### Natively

Everything below — prerequisites, environment setup, and the two dev servers — describes
the native path.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.12** | **Exactly 3.12.** `torch`'s last macOS x86_64 wheel is 2.2.2, which supports cp38–cp312 only — 3.13+ cannot install torch on an Intel Mac. |
| **Node.js 20+** | Verified on Node 24.18. `npm` ships with it. |
| **~4 GB free disk** | ≈1 GB Hugging Face download cache, ≈1 GB prepared dataset in `data/raw/`, ≈1 GB extracted JPEGs in `data/images/`, ≈580 MB CLIP weights, plus the LanceDB table. |
| **Network, once** | Only during ingestion, to pull the dataset and CLIP weights. Everything afterwards runs fully offline. |
| **[`uv`](https://docs.astral.sh/uv/)** *(optional)* | Recommended for a fast, reproducible install. A plain `venv` + `pip` path is given below. |

Target hardware is **macOS on Intel CPU**. There is no CUDA or MPS code path anywhere in
this project; `torch_device` is pinned to `"cpu"` as a validated literal.

---

## Setup

All commands are run from the repository root.

### 1. Python environment

With `uv` (recommended):

```bash
uv venv backend/.venv --python 3.12
uv pip install --python backend/.venv/bin/python -r requirements.txt
uv pip install --python backend/.venv/bin/python -r requirements-dev.txt   # optional: ruff, mypy, pytest
source backend/.venv/bin/activate
```

Or with the stdlib tooling:

```bash
python3.12 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional
```

CPU wheels are the default on macOS — no `--index-url` is needed. Every version in
`requirements.txt` is pinned, with the hardware ceiling that forced each pin documented
in the file itself.

### 2. Build the index (one-time)

This downloads Flickr8k and the CLIP weights, writes the JPEGs to `data/images/`, encodes
every image on the CPU, and stores the 512-d vectors with their captions in LanceDB.

```bash
# Fast smoke test — 100 images, ~10 s of embedding. Start here.
python scripts/ingest.py --limit 100

# Full corpus — ~8k images, ~15 minutes of embedding on an Intel CPU.
python scripts/ingest.py
```

The script is **idempotent and resumable**: rows already in the table are skipped, so an
interrupted run can simply be restarted. `--limit N` always takes a stable prefix, so a
limited run and a full run agree on record *i* — meaning you can smoke-test with `--limit`
and later run the full ingest without `--force`.

Useful flags: `--force` (drop and rebuild), `--encode-batch-size N` (images per forward
pass, and the cap on decoded images held in memory), `--read-batch-size N` (rows read per
iteration; does not affect memory), `--threads N` (cap torch's CPU threads), `--verbose`.
Run `python scripts/ingest.py --help` for the full list.

The device is detected rather than configured: CUDA if present, then Apple's MPS, then CPU.
Batch image encoding is where an accelerator pays — measured 81 img/s on MPS against
37 img/s on CPU on the reference machine.

**Corpora past ~50k images** should follow ingestion with `python scripts/build_index.py`,
which builds an ANN index if the corpus is large enough to benefit and declines with an
explanation if it is not. `--status` reports what exists without changing anything.

> Those timings are embedding time. The **first** run also downloads ~1 GB of dataset and
> ~580 MB of CLIP weights, which dominates it; both are cached, so every later run —
> including the full ingest after a `--limit` smoke test — skips straight to encoding.

### 3. Measure data quality (optional)

Builds `data/analysis.json`. Skip it and everything else still works — the quality filters
simply do not appear.

```bash
python scripts/analyze.py --no-captions   # duplicates and split leakage. ~4 s
python scripts/analyze.py                 # adds caption retrieval. ~9 min
```

The vector arithmetic is free: the full 8 000-by-8 000 similarity matrix takes under a
second. What costs is re-encoding all 40 000 captions to measure how well they retrieve
their own images — 75 captions/s on this CPU. `--no-captions` gives you the duplicate and
leakage findings without it; the `weak-captions` filter and the R@k figures then stay
hidden rather than being shown as zeros.

`docker compose run --rm setup` runs the fast half. Set `ANALYZE_CAPTIONS=1` on it for the
whole thing.

### 4. Project the embeddings (one-time, seconds)

Builds `data/projection.json`, the 2-D coordinates the map view reads. Skip it and
everything else still works — the map tab simply does not appear.

```bash

python scripts/project.py                 # PCA.   0.5 s of arithmetic, ~3 s including imports
python scripts/project.py --method tsne   # t-SNE. ~18 s natively, ~45 s in the container
```

Which of the two to build is not obvious, and the measurements behind the answer are in
[Two projections, and which one is honest](#two-projections-and-which-one-is-honest) below.
Short version: PCA is the default because it can quantify its own distortion, t-SNE draws
the more useful map, and the two are not interchangeable in the way the names suggest.

Re-run with `--force` after any ingest that changed the corpus (`docker/setup.sh` does
this automatically). Without `--force` the script keeps an existing file whose method and
row count already match — note that **`--perplexity` and `--seed` are not part of that
check**, so changing either without `--force` silently keeps the old map.

**Switching between the two while the app runs**, which is what a demo needs:

```bash
docker/projection.sh tsne     # compute (once), swap, restart, verify
docker/projection.sh pca      # swap back — ~17 s, straight from the cache
```

Each variant is kept as `data/projection.<method>.json`, so only the first switch pays the
computation. The restart is unavoidable: the API reads the artefact once at startup and
holds it in memory. The script asserts that the API really is serving the method you asked
for before it reports success. Pass `--recompute` to bypass the cache, or `--rebuild` to
rebuild the image first — the latter is almost never needed, because the projection is data
mounted from `./data` rather than anything baked into the image.

### 5. Get the partition back out (any time, seconds)

Once you have moved images into collections, this writes the effective partition to a file a
training script can read — image id → collection, with the ground-truth split kept beside it
and the provenance of each collection in a header block.

```bash
python scripts/export_split.py                 # data/splits.json
python scripts/export_split.py --format csv    # data/splits.csv, for pandas
```

Read-only with respect to both the index and `data/collections.db`, and idempotent: without
`--force` it keeps an existing file that already describes the same partition. See
[`docs/api.md`](./api.md#getting-the-partition-out-scriptsexport_splitpy).

### 6. Start the backend

```bash
source backend/.venv/bin/activate          # if not already active
uvicorn app.main:app --app-dir backend --reload
```

- API: <http://localhost:8000>
- Interactive OpenAPI docs: <http://localhost:8000/docs>

`--reload` belongs to the uvicorn CLI, which is why that is the development command. It
does not read the bind address from settings, though; to serve on a host or port set in
`.env`, use `PYTHONPATH=backend python -m app` instead.

The process **refuses to start** if `data/images/` or the LanceDB table is missing, with an
error naming `scripts/ingest.py`. That is intentional: a server that serves empty pages is
harder to diagnose than one that will not boot.

### 7. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

The port is pinned with `strictPort` because the backend's CORS policy whitelists exactly
`localhost:5173` and `127.0.0.1:5173`; a silent fallback to 5174 would surface as an opaque
CORS failure. If you need to run the API elsewhere, copy `frontend/.env.example` to
`frontend/.env`, set `VITE_API_BASE_URL`, and add the new Vite origin to
`CORPUSLENS_CORS_ALLOW_ORIGINS` on the backend.

### Configuration

Backend settings are environment-driven via `pydantic-settings`, read from the process
environment or a repository-root `.env`, each prefixed `CORPUSLENS_` (see
`backend/app/core/config.py`). Every path, port, model id, and bound has a documented
default — **no configuration is required to run**.

`backend/.env.example` lists every variable with its default and the reasoning behind it.
To change something, copy it to the **repository root** — that is where `pydantic-settings`
looks, not `backend/`:

```bash
cp backend/.env.example .env
```

Two notes on the settings that are easy to get wrong:

- **`CORPUSLENS_HOST` defaults to `127.0.0.1`, not `0.0.0.0`.** The API has no
  authentication, so binding every interface would publish the corpus to the local
  network. The container image sets `0.0.0.0`, where the container boundary is what
  limits reach. It is read by `python -m app`; the `uvicorn` CLI ignores it and takes
  `--host` or `UVICORN_HOST`.
- **`CORPUSLENS_CLIP_MODEL_ID` must match what the corpus was embedded with.** A mismatch
  does not raise — it produces a "shared" embedding space that isn't shared, and search
  silently degrades to noise. The offline scripts read the same variable from the process
  environment (they do not parse `.env`, being standalone per §4.2), so changing it means
  `export CORPUSLENS_CLIP_MODEL_ID=... && python scripts/ingest.py --force`.

---

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/dataset/stats` | Corpus counts and quality figures, reported against **both** the dataset's partition and the user's |
| `GET` | `/api/dataset?offset=&limit=` | Paginated image summaries (`limit` ≤ 200, default 50) |
| `GET` | `/api/dataset/{id}` | One image with all reference captions; `404` if absent |
| `POST` | `/api/search` | CLIP search by text **or** by example image; body carries exactly one of `query` / `image_id` |
| `GET` | `/api/projection` | Every image's 2-D position, its split, and whether it matches the filter |
| `POST` | `/api/export` | CSV or JSONL manifest of a selection, a ranking, or the filtered slice |
| `GET` | `/api/collections` | Every collection with its current size, built-ins first |
| `POST` | `/api/collections` | Create a collection; `409` on a duplicate name |
| `PATCH` | `/api/collections/{id}` | Rename one; `403` for a built-in |
| `DELETE` | `/api/collections/{id}` | Delete one; its images revert to their splits |
| `POST` | `/api/collections/{id}/images` | Move images in, by explicit `ids` **or** by `filter`; reports `{moved, unchanged, unknown}`; `413` above 8 000 |
| `DELETE` | `/api/collections/{id}/images/{image_id}` | Return one image to its split |
| `GET` | `/images/{file_name}` | Original JPEGs, served by `StaticFiles` |

Every listing endpoint accepts the same four filter parameters: `split` (repeatable),
`collection` (repeatable), `caption_contains`, and `quality_flag` (`near-duplicate`,
`cross-split-duplicate` or `weak-captions`).

A **collection** is the researcher's own partition, laid over the dataset's `split` rather
than replacing it: every image's effective collection defaults to its split, and moving one
writes an override in a small SQLite store the API owns at `data/collections.db`. The
LanceDB index is never written to, which is what keeps the cross-split duplicate-leakage
figures meaningful. The three built-in collections mirror the splits actually present in the
index.

The move endpoint takes **exactly one of** `ids` and `filter`, the way `POST /api/search`
takes exactly one of `query` and `image_id`. The filter channel is the one that matters: the
sets worth quarantining — `weak-captions`, `cross-split-duplicate`, a caption match — are
defined by a predicate and scattered across the whole embedding cloud, so no rectangle drawn
on the map can approximate them and nobody is clicking 200 times. Both channels share one
8 000-image ceiling, because the cost of a move is the override rows it leaves behind rather
than the bytes it arrived in.

Contract details and the reasoning behind each decision: [`docs/api.md`](./api.md).
Frontend structure and decisions: [`docs/frontend.md`](./frontend.md).

---

## Development commands

`make` wraps the commands below; `make help` lists every target. `make lint` and
`make test` run exactly what CI runs, in the same order, so passing them locally means
passing [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

```bash
make up          # containers: backend :8000 + frontend :5173, hot reload
make lint        # ruff + mypy + tsc + eslint
make test        # pytest
make format      # ruff format + prettier
```

The targets below are what those wrap, and work the same without Docker or `make`.

```bash
# Backend (from the repo root, venv active)
ruff check . && ruff format --check .
mypy                     # strict mode; config in pyproject.toml
pytest                   # 174 contract, filter, export and architecture tests; ~6 s, no dataset required

# Frontend (from frontend/)
npm run typecheck        # tsc -b
npm run test             # Vitest (16 cases over the pure modules)
npm run lint             # ESLint flat config
npm run format:check     # Prettier
npm run build            # tsc -b && vite build
```

---

## Design decisions

**Modularity — a one-directional dependency chain.** The backend is layered
`api/routes → services → repositories → LanceDB`, and the direction is enforced by what
each layer is allowed to import. Routes handle HTTP only: parse, call exactly one service,
map to a response model, pick a status code. Services hold the business logic and import no
web framework at all — which is what makes ranking and statistics testable without an HTTP
client. `repositories/image_repository.py` is the *only* module in the codebase that
imports `lancedb`. The frontend mirrors this: vertical feature slices
(`features/gallery`, `features/search`, `features/stats`, `features/inspector`), data
fetching confined to hooks, components that only render.

**The store and the encoder are interfaces, not classes.** Both pieces of infrastructure
are declared as `typing.Protocol` — `VectorRepository` in `repositories/vector_db.py`,
`EmbeddingService` in `services/embedding.py` — and every service and route names the
Protocol. `core/lifespan.py` is the only module that picks an implementation
(`LanceDBImageRepository`, `ClipEmbeddingService`), so putting Qdrant or a different
encoder behind the same API is a new class plus one line, with nothing above it recompiled
or rewritten. Protocols rather than ABCs because structural typing lets an adapter in a
separate distribution conform without importing this package at all.

That claim is tested rather than asserted: `backend/tests/test_architecture.py` defines an
in-memory store and a stub encoder that inherit from nothing, injects them through
`app.dependency_overrides`, and drives the real routes end to end.

**Expensive resources are owned by the application lifespan.** The CLIP bi-encoder (~5 s to
load) and the LanceDB handle are created once at startup and hung off `app.state` in a
frozen, typed dataclass, then injected into routes with `Depends`. No module-level
singletons, no per-request model loading, and `app.state` stays checkable by mypy. The
providers in `api/deps.py` are one-line indirections over that bundle precisely so they can
be overridden per test without rebuilding the lifespan.

**Blocking work is offloaded.** LanceDB's client is synchronous and torch inference is
CPU-bound. Routes are `async` and services push both onto the worker-thread pool, so a
table scan or a query encode cannot stall the event loop.

**LanceDB over a vector-DB server.** For a single-user local tool, a Pinecone/Weaviate/
Qdrant deployment is infrastructure with no payoff. LanceDB is embedded and file-backed —
`data/lancedb/` is a directory, not a service — so nothing has to be provisioned, and the
container that does exist is a convenience rather than a dependency.
Because its columnar format stores captions and metadata alongside the vectors, one query
returns a fully renderable result; there is no second lookup to join metadata back on.

**No ANN index at this scale, deliberately — and it is measured, not assumed.** At ~8k rows
a brute-force scan over 8k × 512 float32 takes ~21 ms and is *exact*. An IVF_PQ index over
the same corpus is genuinely faster at 6 ms, but returns only 0.695 recall@20: product
quantization discards a third of the true neighbours. `nprobes` does not fix that (sweeping
1→256 moves recall by 0.008, because the loss is quantization rather than partition
pruning); `refine_factor`, which re-ranks candidates against the full-precision vectors,
does — and at `refine_factor=10` the query costs 20 ms for 0.997 recall, which is the exact
scan again to within noise.

So the corpus ships unindexed because the index buys nothing here, not because approximate
search is unsupported. Past roughly 50k rows the arithmetic inverts — at 200k the same
configuration is 6.6 ms against 204 ms at no measured recall cost — and
`scripts/build_index.py` builds the index then. See §4.4 of `CLAUDE.md` for the exactness
contract that governs how the API uses one.

**Correctness in the embedding space.** `normalize_embeddings=True` is passed explicitly on
both the ingestion and the query side. It is *not* the `sentence-transformers` default, and
cosine ranking is only meaningful when both sides are unit length — a silent mismatch here
degrades search into noise rather than failing loudly. Similarly, the query encoder is
pinned to the same checkpoint the images were embedded with, because a mismatch produces a
"shared" space that isn't shared.

**Usability for the researcher.** The interface is built around the thing a researcher
does repeatedly: type a concept, look at what came back, open a result, read the human
captions against the model's similarity score. Search results carry their similarity
inline, and the inspector is reachable from both the gallery and a search hit, so the
model's judgement and the human annotation are always one click from each other. Note that
CLIP's modality gap keeps absolute text↔image scores low (~0.2–0.35 even for good matches)
— **rank order is the signal, not the magnitude**; the UI presents scores accordingly.

**Hardening at the edge.** Image ids are constrained to `^[A-Za-z0-9._-]+$` at the route, so
a crafted id becomes a `422` and never reaches a filter expression or a filesystem path;
the repository escapes quotes as a second layer. CORS enumerates its methods and headers
rather than wildcarding, with credentials off.

### Two projections, and which one is honest

The map view can be built two ways, and the choice is more interesting than "fast one and
pretty one". **Neither touches search.** Ranking runs on the full 512 dimensions, in
LanceDB, at query time; these two only exist so that a 512-dimensional point can be drawn
on a screen that has two.

**They are not peers — one stands on the other.** PCA is reachable alone: `--method pca`
projects 512 → 2 and those two numbers are the map. t-SNE is not: `_tsne()` first calls the
same PCA to reduce 512 → 50 (which retains 64.4% of the variance, against 14.4% for two
components), and scikit-learn's `init="pca"` then derives the starting layout from that
before a thousand gradient steps move the points. Removing PCA would take t-SNE with it.

**The 512 → 50 step is only worth doing for t-SNE.** PCA components are nested, so running
PCA to 50 and then to 2 returns the first two components unchanged — measured on this
corpus, the two paths agree to `corr = 1.0000000000`, `max|Δ| = 1.8e-07`, i.e. float32
noise. For t-SNE the same step is not redundant but a denoise-and-speedup.

**What each one preserves, measured on the real 8 000 vectors:**

| | PCA | t-SNE |
|---|---|---|
| Neighbours kept (of each image's true top-10 in 512-d, how many land in its top-10 on the map) | **2.4%** | **32.1%** |
| Pairwise-distance correlation with 512-d (Pearson / Spearman, 2 000-point sample) | +0.393 / +0.361 | **+0.470 / +0.445** |
| Reports its own distortion | **yes** — 8.6% + 5.8% = 14.4% | no — the quantity does not exist |
| Reproducible | bitwise, by construction | bitwise, via `init="pca"` |
| Cost on this CPU | 0.5 s | 17.5 s |

**Measuring it corrected three things this guide used to claim.** t-SNE was described as
costing "1–3 minutes" — it costs 18 seconds, because of the 50-dimensional pre-reduction.
It was described as having no global meaning — that warning applies to `init="random"`;
with the PCA initialisation actually in use, t-SNE tracks the original distances *better*
than PCA does. And `--seed` was described as what makes a run reproducible — it is not:
with `init="pca"` there is no randomness left for it to control, and seeds 0, 1 and 12345
produce bitwise-identical output (switching to `init="random"` makes them differ by 18.16,
which is how that was confirmed).

**PCA nevertheless stays the default**, for one reason that survives the table: it is the
only one of the two that can state how much of the corpus it is failing to show. A map that
silently discards 510 dimensions and offers no error bar is a worse default than a blurry
map with "14.4%" printed under it. Three smaller properties come with it — the projection
is a fixed linear map, so a new vector could be placed without recomputing, whereas
`sklearn`'s `TSNE` has no `.transform()` at all; PCA distances remain metric, while t-SNE
expands sparse regions and compresses dense ones, so a distance correlation of 0.45 across
the whole cloud still does not make any individual gap readable; and PCA needs no
`scikit-learn`.

**The honest reading is that they answer different questions** — t-SNE to *find* a group,
PCA to reason about spread — and the 2.4% figure is the argument for building both rather
than for trusting either alone. `docker/projection.sh` switches between them in about 17
seconds so the difference can be looked at rather than argued about.

*Caveat on the method:* pairwise-distance correlation is a crude proxy for
"interpretability", and the neighbour metric uses a single `k = 10`. The 13× gap on
neighbour retention is far too large to be an artefact of either choice; the distance
correlations, at 0.39 against 0.47, are close enough that they should be read as "neither
is good" rather than as a ranking.

---

## Scope and honest status

**Automated:** 170 pytest cases in `backend/tests/` cover the HTTP contract — stats,
pagination windows and bounds, caption retrieval, `404`/`422`/`413` handling, search by text
and by example, filtering including the quality flags and collections, collection CRUD,
moves by id *and* by filter, provenance and its schema migration, export in both formats,
the projection endpoint with and without an artefact, payload validation,
and the `normalize_embeddings` invariant that silently skews every ranking if it is ever
dropped. `test_export_split.py` is the one module below the HTTP boundary, covering the
offline partition export's join, its renderers and its idempotence check.
They drive the real lifespan, wiring, services, repository mapping and serialization; only
the LanceDB connection and the CLIP encoder are replaced by in-memory doubles, so the
suite needs no dataset and finishes in about six seconds.

**Frontend:** 16 Vitest cases, over the two modules that are pure functions — the map's box
query (`pointsInBox`) and the imported-id parser. `ruff check`, `ruff format --check`,
`mypy --strict`, `npm run test`, `npm run lint`, `npm run typecheck` and `npm run build` all
pass.

Nine invariants are worth calling out because they fail *silently* rather than loudly, and
each has a test whose only job is to catch that:

- **Search pre-filters.** With a post-filter, asking for the top 20 inside one split would
  return the survivors of the global top 20 — usually fewer than 20, with the tail of the
  ranking missing.
- **Caption-filter wildcards are literal.** Unescaped, a typed `%` matches all 8 000 rows.
- **A quality flag with no analysis matches nothing**, rather than being ignored. Silently
  dropping it would show the whole corpus under a heading claiming to list duplicates.
- **Search by example runs no inference.** The test asserts the encoder is never called;
  if that ever regressed it would show up only as latency.
- **The collection predicate stays parenthesised.** `AND` binds tighter than `OR`, so
  losing the parentheses lets the collection group swallow the caption filter: on the real
  corpus, 1 527 matching rows become 5 999. Wrong in the direction that looks like success.
- **A move never changes `images_by_split`.** The dataset's partition is what the
  cross-split leakage figures are computed from; if a re-partition moved it, those numbers
  would quietly start describing something else.
- **With no image moved, the emitted SQL is byte-identical to what it was before
  collections existed.** That is what makes the feature additive, and it is asserted on the
  predicate string rather than end to end.
- **A rectangle on the map selects only what matches the active filter.** It used to select
  by geometry alone: measured live with the Weak-captions filter on (200 of 8 000 matching),
  one drag selected **5 379** points, and "Move to…" sat next to that count. Nothing about
  it looked wrong.
- **A store written before provenance existed still opens.** SQLite has no
  `ADD COLUMN IF NOT EXISTS`, so the `IF NOT EXISTS` schema cannot add a column to a table
  that is already there; the migration reads `PRAGMA table_info` instead, and a test builds
  a store on the old schema to prove it.

The in-memory LanceDB double **parses and evaluates** filter expressions rather than
recording and ignoring them, and raises on any clause shape it does not model — otherwise
every filtering test above would be a test of nothing.

**Manual**, per [`docs/api.md`](./api.md) and [`docs/frontend.md`](./frontend.md):
endpoints were exercised against a live server on the real 8 000-image index (including
path-traversal rejection and CORS preflight, which sit in middleware the test client
bypasses), and the UI was driven in headless Chrome over the Chrome DevTools Protocol —
grid, filters, the quality flags, the map's render, pan/zoom, hover preview, rectangle
selection, search highlighting, the inspector's data-quality panel, search by example, and
both export formats. The collection overlay was exercised the same way against the real
index — 200 images moved, the splits verified unmoved, the compound predicate confirmed as a
working vector pre-filter, and the over-cap refusal confirmed to return `413` before any
images were reassigned.

**Deliberate design limits, stated because they are easy to misread:**

- **The default 2-D map shows 14.4% of the variance.** Two PCA components of a 512-d CLIP
  space cannot show more, and only 2.4% of each image's true nearest neighbours survive the
  projection. The figure is printed under the plot rather than buried here, and
  `docker/projection.sh tsne` swaps in the projection that keeps 32.1% of them instead —
  see [Two projections, and which one is honest](#two-projections-and-which-one-is-honest).
- **The text query is not plotted on the map.** Projecting the query's embedding onto axes
  fitted to *image* vectors would place it far outside the cloud — CLIP's modality gap puts
  the two in different regions — and invite a conclusion about retrieval that the picture
  does not support. The images it retrieved are highlighted instead.
- **Selection is a rectangle, not a freehand lasso.** Roughly a third of the code for
  nearly all of the usefulness on a cloud this diffuse.
- **CSV flattens captions into five fixed columns.** That is the full width of Flickr8k's
  schema, and `caption_count` travels beside them so truncation would be visible. JSONL is
  the lossless format.
- **"Weak captions" is the worst 2.5% of the corpus, not a rank threshold.** A cutoff like
  "rank worse than 100" means something different in a corpus of 8 000 than in one of
  80 000; a fraction gives a review queue of predictable size either way.
- **Duplicate detection is embedding-based, at cosine 0.95.** It finds re-shoots and crops
  as well as exact repeats, which is the intent — but it is a judgement call with a
  threshold, not a checksum. The inspector puts the pair one click apart so the call stays
  with the reader.

**Measured, and left as it is:**

- **A full re-partition makes filtered queries ~20× slower.** Every override becomes a
  literal in the id lists `filters.py` embeds in each filtered query. Measured on the live
  index: `?collection=train&caption_contains=dog` takes **0.18 s** with 200 overrides and
  **3.5 s** with all 8 000 — a predicate of roughly 200 KB. The 8 000-image move ceiling does
  **not** bound this, and cannot: two moves of 4 000 reach the same state. Fixing it means
  materialising an id set and paying a full scan, which `filters.py` invited someone to
  measure before trading for. This is that measurement; the trade is not made here.
- **Collection leakage does not fall to zero when you quarantine.** Moving all 32
  cross-split-duplicate images into one collection takes it from 22 pairs to 8, not to 0,
  because a quarantined image whose near-duplicate stayed behind now straddles the new
  boundary. That is the honest reading of the partition, not a bug in the counter.

**Not built, and knowingly so:**

- **A collection is a partition, not a tag.** `image_collection.image_id` is a PRIMARY KEY,
  so an image sits in exactly one collection. Right for train/val/test/holdout; wrong for
  `duplicate`, `weak-caption`, `mislabelled`, which overlap — and moving an image into
  "quarantine" currently removes it from "train", which is correct for one meaning and
  destructive for the other. A second many-to-many store alongside the partition overlay is
  proposed but not built.
- **No re-split generator.** The operation a data engineer actually performs is "produce a
  new 80/10/10, seed 42, stratified by X, excluding tag:duplicate" — one recorded,
  re-runnable action. Everything here is the substrate for it; it is not built.
- **The map still colours by ground-truth split only.** After re-partitioning, the tool's
  best view cannot show the partition you just built — you get binary dim/bright `matches`
  for one filter at a time. The palette is a fixed three-colour scheme and N arbitrary user
  collections is a real design problem, not a colour-count bump.
- **Limited test coverage below the API boundary and none in the browser.** The repository is
  exercised only through an in-memory double, so nothing automated asserts that our
  assumptions about LanceDB's real query builder still hold. The frontend suite covers the
  two pure modules and nothing else — no component tests, no e2e — so the wiring is verified
  by a scripted-but-manual pass.
- **Provenance is the last batch, not a history.** A collection built from several moves
  reports the most recent one. Per-image audit trail is a different feature.
- **Caption-level statistics** — length distributions, vocabulary, token frequencies — are
  not implemented. Statistics today are corpus counts, retrieval recall and the
  duplicate findings.
- **No URL state**, and browse navigation is "Load more" only — no offset jump and no
  virtualisation. A query, a filter, a map viewport or an image is therefore not
  bookmarkable or shareable, and reaching the tail of the corpus takes many clicks.
- **Responsive layout below `sm`** is unverified; the app has been viewed at desktop widths.
  The map in particular assumes a pointer — there is no touch pinch-zoom.

---

## License and data

Flickr8k is distributed by its original authors under their own terms. **This repository
contains no dataset images or captions** — everything is downloaded locally during
ingestion and is git-ignored, along with the model weights and the LanceDB directory.
