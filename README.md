<h1 align="center">CorpusLens</h1>

<p align="center">
  <strong>A local, offline visual search and data-curation workbench for image–caption corpora.</strong><br>
  Map the CLIP embedding space, search it in plain English, and carve out the subset you actually want to train on.
</p>

<!-- Replace `Berkhin/corpuslens` with the real repository path once published. -->
<p align="center">
  <a href="https://github.com/Berkhin/corpuslens/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Berkhin/corpuslens/actions/workflows/ci.yml/badge.svg"></a>
  <a href="./LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-3776AB.svg?logo=python&logoColor=white">
  <img alt="Node 20+" src="https://img.shields.io/badge/node-20%2B-5FA04E.svg?logo=nodedotjs&logoColor=white">
  <img alt="Runs offline" src="https://img.shields.io/badge/runs-100%25%20offline-success.svg">
</p>

<!-- Insert a GIF or Screenshot of the UI here -->

---

## Why this project?

Image–caption datasets ship as a folder of JPEGs and a text file. CorpusLens turns that
into something you can actually interrogate — on your laptop, with no account, no cluster,
and no network after first run.

**🪶 Lightweight by construction.** The vector store is [LanceDB](https://lancedb.com),
embedded and in-process — a *directory*, not a service. No daemon, no port, no container to
provision alongside the app. Vectors, captions and metadata live in one table, so a single
query returns everything a result card needs.

**🔌 Modular, swap-friendly architecture.** Infrastructure is named by `typing.Protocol`,
never by implementation. `VectorRepository` and `EmbeddingService` are contracts; the LanceDB
and CLIP classes are just the ones wired in today. Swapping the store or the encoder is one
new class and two lines in the composition root — and test doubles conform without inheriting
anything.

**🐳 Great developer experience.** `make setup && make up` and you're running, with hot reload
on both ends. `make lint` and `make test` run exactly what CI runs, in the same order — green
locally means green on the PR. Docker is fully supported but never mandatory: the bare
`uvicorn` + `vite` path is a first-class citizen.

**🔒 Genuinely offline.** Network is used once, during ingestion, to pull the dataset and CLIP
weights. After that, everything — inference included — runs on your CPU. No cloud APIs, no
telemetry, no data leaving the machine.

**🔎 Built for curation, not just browsing.** Semantic text→image and image→image search,
an interactive 2-D map of the whole corpus, near-duplicate and split-leakage detection,
composable filters, user-defined collections that re-partition the corpus without mutating
ground truth, and CSV/JSONL export of whatever you narrowed down to.

> Ships with **Flickr8k** (~8k images, ~40k captions) as the reference corpus.

---

## Quick Start

**Prerequisites:** Docker with Compose v2, ~4 GB free disk.

```bash
# 1. One time: download the corpus, embed it on CPU, build the index (~15 min, ~3 GB)
make setup

# 2. Start the backend and frontend with hot reload
make up
```

Open **http://localhost:5173** — the API and its OpenAPI docs are on
**http://localhost:8000/docs**.

<details>
<summary><strong>In a hurry?</strong> Smoke-test with 100 images in about two minutes.</summary>

```bash
make setup ARGS='--limit 100'
```

`--limit` always takes a stable prefix, so a limited run and a full run agree on record *i*.
You can run the full ingest later without `--force`.

</details>

Prefer no Docker? The native `uvicorn` + `vite` path is fully supported —
see [the full guide](./docs/guide.md#setup). Run `make help` to list every target.

---

## Architecture

The system splits into a **one-time offline pipeline** and a **read-only serving path**.
That split is the central design decision: CLIP inference on CPU runs at roughly 10 images/s,
far too slow to live inside a request handler. Images are embedded ahead of time; the API only
ever *queries* the result.

```
  OFFLINE (one time)                        SERVING (read-only)

  scripts/ingest.py                         React 19 · Vite · TypeScript
    HF datasets → JPEGs                     Tailwind v4 · Shadcn UI
    CLIP ViT-B/32 → 512-d vectors           TanStack Query
    └─ write to LanceDB                              │ HTTP/JSON
                                                     ▼
  scripts/project.py   → projection.json    FastAPI · Uvicorn
  scripts/analyze.py   → analysis.json      routes → services → repositories
            │                                        │
            └────────────►  data/  ◄─────────────────┘
                 images · lancedb · json artefacts
```

| Layer | Choice | Why |
|---|---|---|
| **Frontend** | React 19, TypeScript (strict), Vite, Tailwind v4, Shadcn UI, TanStack Query | Server state owned by the query layer; components stay render-only |
| **API** | Python 3.12, FastAPI, Pydantic v2, Uvicorn | Typed models end to end, OpenAPI for free at `/docs` |
| **Embeddings** | OpenAI CLIP `ViT-B/32` via `sentence-transformers`, CPU-only `torch` | Shared 512-d image/text space — no GPU, no paid API |
| **Vector store** | LanceDB, embedded | Zero-setup; columnar Lance keeps vectors and captions in one table |
| **Projection** | `numpy.linalg.svd` (PCA) · `scikit-learn` (t-SNE) | PCA in four visible lines keeps the explained-variance ratio honest |

### Layering and dependency injection

Dependencies point one way, and only one way:

```
api/routes  →  services  →  repositories  →  (LanceDB / filesystem)
                    ↓
              models (Pydantic)
```

- **`api/routes/`** — HTTP only: parse, call one service, map to a response model.
- **`services/`** — all business logic, and **framework-agnostic**: a service never imports
  `fastapi`, which is what makes it unit-testable without an HTTP client.
- **`repositories/`** — the only code that touches LanceDB or the filesystem. Domain-shaped
  methods in, domain objects out; driver types never leak upward.

Wiring is FastAPI's `Depends`, expressed as annotated aliases in
[`api/deps.py`](backend/app/api/deps.py) (`VectorRepositoryDep`, `SearchServiceDep`, …), so a
route declares *what it needs*, not *how to build it*. The expensive singletons — the CLIP
model and the LanceDB handle — are created **once** at startup in
[`core/lifespan.py`](backend/app/core/lifespan.py), which is the single composition root: the
only module in the project that names a concrete implementation. Everything else names the
Protocol. There are no module-level global singletons.

The API is a **pure reader** of the index the offline scripts produce, so `split` stays
immutable ground truth. The one mutable store — user collections — is a separate SQLite
overlay at `data/collections.db`, never a write back into the index.

📚 **Deeper reading:** [Full guide](./docs/guide.md) · [API reference](./docs/api.md) ·
[Frontend notes](./docs/frontend.md) · [Engineering contract](./CLAUDE.md)

---

## Environment Variables

Every value is **optional** — the app runs with no configuration at all. To change something,
copy [`backend/.env.example`](backend/.env.example) to a repository-root `.env`. All backend
variables are prefixed `CORPUSLENS_` so generic names can't collide with another tool's.

| Variable | Default | Purpose |
|---|---|---|
| `CORPUSLENS_HOST` | `127.0.0.1` | Bind interface. Loopback on purpose — the API has no auth, so `0.0.0.0` would publish your corpus to the local network. |
| `CORPUSLENS_PORT` | `8000` | API port. |
| `CORPUSLENS_DATA_DIR` | `./data` | Root of the corpus. Images, index, projection, analysis and collections all derive from it — relocating everything is this one line. |
| `CORPUSLENS_CLIP_MODEL_ID` | `clip-ViT-B-32` | Text-query encoder. **Must match the checkpoint the images were embedded with** — a mismatch doesn't raise, it silently degrades search to noise. |
| `CORPUSLENS_TORCH_DEVICE` | `cpu` | Only accepted value; anything else fails startup validation. |
| `CORPUSLENS_CORS_ALLOW_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Browser origins allowed to call the API. Comma-separated or JSON array. |
| `CORPUSLENS_LOG_LEVEL` | `INFO` | Root log level. |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Frontend → API origin. Set in `frontend/.env`; inlined at **build** time, so changing it means rebuilding. |

Definitions and the reasoning behind each default live in
[`backend/app/core/config.py`](backend/app/core/config.py), which is the source of truth.

---

## Contributing

Contributions are welcome — issues, bug reports and pull requests alike.
Start with **[CONTRIBUTING.md](./CONTRIBUTING.md)** for dev setup, the `make` targets, and PR
conventions. New to the codebase? [`CLAUDE.md`](./CLAUDE.md) is the engineering contract that
explains *why* the code looks the way it does.

---

## License

Released under the **[MIT License](./LICENSE)**.

Flickr8k is distributed by its original authors under their own terms. **This repository
contains no dataset images or captions** — they are downloaded locally during ingestion and
are git-ignored, along with the model weights and the LanceDB directory.
