# CLAUDE.md — Project Rules & Engineering Contract

> This file is the **single source of truth** for how code is written in this repository.
> Read it in full before proposing or generating any code. If a request conflicts with
> a rule here, surface the conflict explicitly instead of silently deviating.

---

## 1. Project Goal

Build a **local web-based visualization and exploration tool for the Flickr8k dataset**,
targeted at **Computer Vision researchers**. The tool must let a researcher:

- Browse the image/caption corpus with fast, paginated grid views.
- Inspect a single image with all of its (typically five) reference captions.
- Run **semantic search** over the dataset via CLIP embeddings — both
  text→image and image→image — backed by a local vector index.
- View dataset-level statistics (caption length distributions, vocabulary,
  token frequencies, per-split counts) to support data-quality assessment.

Optimize for **clarity, correctness, and demonstrable engineering judgment** over
feature count.

---

## 2. Core Constraints (Non-Negotiable)

| Constraint | Rule |
|---|---|
| **Local only** | Everything runs on `localhost`. No deployment targets. Docker is a **supported but optional** run path — `docker compose up` must work, and so must the bare `uvicorn` + `vite` path, with neither becoming the only way in. |
| **No cloud services** | No AWS/GCP/Azure, no hosted inference, no remote object storage. |
| **No external vector DB** | LanceDB **embedded** only. No Pinecone, Weaviate, Qdrant server, Milvus, pgvector. |
| **No paid APIs** | No OpenAI/Anthropic/Cohere/Replicate calls at runtime. All inference is local. |
| **No telemetry** | Disable analytics/telemetry in any library that ships it. |
| **Hardware** | **macOS on Intel CPU.** No CUDA, no MPS, no Apple Silicon assumptions. All ML is CPU-bound `torch` (install CPU wheels). Never suggest `device="mps"` or `device="cuda"`. |
| **Network** | Allowed **once**, during a one-time ingestion step, to pull the Flickr8k dataset and CLIP weights from Hugging Face. After that, the app must run fully offline. |

**Performance implication:** CPU-only CLIP inference over ~8k images is slow (minutes,
not seconds). Embedding is therefore a **one-time offline batch job** in `scripts/`,
never something triggered by an HTTP request. The API only ever *queries* a pre-built index.
The single exception: encoding a short user query string at search time (fast, ~tens of ms).

---

## 3. Technology Stack (Exact)

### Backend
- **Python 3.12+**
- **FastAPI** — HTTP API, async endpoints
- **Pydantic v2** — all request/response/config models; `pydantic-settings` for config
- **Uvicorn** — ASGI server (dev: `--reload`)
- **Ruff** — lint + format (replaces black/isort/flake8)
- **mypy** — static type checking, strict mode
- **pytest** — testing

### Data / ML (all local)
- **Hugging Face `datasets`** — Flickr8k acquisition and iteration
- **`sentence-transformers`** with **`clip-ViT-B-32`** — 512-dim image & text embeddings in a shared space
- **`lancedb`** — embedded, file-backed vector database at `data/lancedb/`
- **`torch`** — CPU build only
- **Pillow** — image I/O and thumbnail generation
- **`scikit-learn`** — the t-SNE implementation in `scripts/project.py`, and nothing else.
  PCA there is plain `numpy.linalg.svd`, which keeps the centring, the sign convention
  and the explained-variance ratio visible in our own code. Already present transitively
  via `sentence-transformers`; declared directly because we import it.

### Frontend
- **React 19**
- **Vite** — dev server and build
- **TypeScript** — `strict: true`
- **Tailwind CSS** — utility-first styling
- **Shadcn UI** — component primitives, vendored into `src/components/ui/`
- **TanStack Query** — server state, caching, pagination
- **ESLint + Prettier**

**Do not introduce any dependency not listed above without explicitly proposing it first**,
with a one-paragraph justification and confirmation that it satisfies Section 2.

---

## 4. Architecture & Separation of Concerns

### 4.1 Backend layering — strict, one-directional

```
api/routes  →  services  →  repositories  →  (LanceDB / filesystem)
     ↓            ↓              ↓
           models (Pydantic)
```

- **`api/routes/`** — HTTP concerns only. Parse and validate input, call exactly one
  service, map results to a response model, choose the status code. **No business logic,
  no direct LanceDB access, no filesystem access.**
- **`services/`** — all business logic: search orchestration, ranking, statistics
  computation, query embedding. **Framework-agnostic** — a service must not import
  `fastapi` or touch `Request`/`Response` objects. This makes services unit-testable
  without an HTTP client.
- **`repositories/`** — the *only* place that talks to LanceDB, the filesystem, or the
  HF cache. Expose domain-shaped methods (`search_by_vector`, `get_image_by_id`),
  never leak driver types or raw table handles upward.

**Swappable infrastructure.** The two pieces of infrastructure the app depends on are
named by `typing.Protocol`, never by their implementation:

| Contract | Module | Implementation |
|---|---|---|
| `VectorRepository` | `repositories/vector_db.py` | `LanceDBImageRepository` |
| `EmbeddingService` | `services/embedding.py` | `ClipEmbeddingService` |

Services and routes name only the Protocol. `core/lifespan.py` is the sole module that
picks an implementation, so replacing the store or the encoder is a change to those two
lines plus one new class. Protocols rather than ABCs deliberately: structural typing lets
an out-of-tree adapter conform without importing this package, and lets test doubles
conform without inheriting anything. Implementations in this repository pin conformance
statically (see `_assert_conformance` in `image_repository.py`), because a Protocol is
otherwise only checked where a value is *used* as one.
- **`models/`** — Pydantic models. Keep API-facing DTOs distinct from internal domain
  models when they diverge; do not expose storage schemas directly over HTTP.
- **`core/`** — settings (`pydantic-settings`, env-driven), logging setup, app lifespan
  (load the CLIP model and open the LanceDB connection **once** at startup, not per request).

**Dependency injection:** use FastAPI's `Depends` to inject repositories and services
into routes. No module-level global singletons except those owned by the lifespan context.

### 4.2 Offline pipeline (`scripts/`)

Ingestion, embedding, and index-building live in standalone, re-runnable CLI scripts —
**not** in the API. Each script must be idempotent, log progress, and support resuming
or `--force` re-computation. **The API is a pure reader of the corpus index these scripts
produce** — it never writes to the LanceDB table, and `split` therefore stays immutable
ground truth that `scripts/analyze.py` can measure cross-split leakage from.

**The one exception**, added with user collections: the API owns a small mutable store of
its own at `data/collections.db` (stdlib `sqlite3`), holding user-created collections and
the per-image overrides that re-partition the corpus. It is an *overlay* — every image's
effective collection defaults to its split, and moving one writes an override rather than
touching the index. Any future writable state must follow the same rule: a separate store,
never the index. See `app/repositories/collection_repository.py` and `docs/api.md`.

### 4.3 Frontend structure

- **`features/<feature>/`** — vertical slices (e.g. `gallery/`, `search/`, `stats/`),
  each owning its components, hooks, and API-call functions.
- **`components/ui/`** — Shadcn primitives only. Do not hand-edit unless customizing
  deliberately; note any customization in a comment.
- **`lib/`** — shared utilities, API client, query-key factories.
- **`types/`** — shared TypeScript types, kept in sync with backend Pydantic models.
- **Data fetching lives in hooks**, never inline in a component body. Components render;
  hooks fetch.
- **No business logic in JSX.** Derive values above the return, or in a hook.

---

## 5. Coding Standards

### 5.1 Python — mandatory

- **Type hints on every function signature** — parameters and return type, no exceptions.
  `mypy --strict` must pass. No bare `Any`; if unavoidable, add `# type: ignore[<code>]`
  with a one-line reason.
- **Modern syntax**: `list[str]`, `dict[str, int]`, `X | None` — not `List`, `Dict`, `Optional`.
- **Docstrings** on every public module, class, and function. Google style. Explain
  *why* and document non-obvious parameters, not restatements of the signature.
- **No mutable default arguments.** No `except:` or bare `except Exception: pass`.
- **`pathlib.Path`** for all paths. Never string concatenation, never `os.path`.
- **Structured logging** via the stdlib `logging` module. **No `print()`** outside `scripts/`
  (where CLI progress output is acceptable).
- **Configuration via `pydantic-settings`**, read from environment / `.env`.
  **No hardcoded paths, ports, model names, or magic numbers** — they go in settings or
  a module-level `Final` constant with a descriptive name.
- **Errors**: raise domain-specific exceptions in services; translate them to
  `HTTPException` at the route boundary only.
- Functions stay small and single-purpose. If a function needs a section comment to
  explain its parts, split it.

### 5.2 TypeScript — mandatory

- **`strict: true`**, plus `noUncheckedIndexedAccess` and `noImplicitOverride`.
- **`any` is forbidden.** Use `unknown` and narrow. Casting with `as` requires a comment
  justifying it.
- **Explicit return types** on all exported functions and hooks.
- **`type` for object shapes, `interface` for extensible contracts** — be consistent.
- **Functional components only**, with typed props. No class components. No `React.FC`.
- **No `useEffect` for data fetching** — TanStack Query owns server state.
  `useEffect` is for genuine synchronization with external systems only.
- **Named exports** preferred over default exports (better refactoring and search).
- **Tailwind**: compose classes with the `cn()` helper. No inline `style` objects except
  for genuinely dynamic values (e.g. computed grid dimensions). No arbitrary values
  where a scale token exists.
- **Accessibility**: semantic HTML, `alt` text on every image (use the caption),
  keyboard-navigable interactive elements.

### 5.3 Shared

- **Small, focused modules.** A file over ~300 lines is a signal to split it.
- **Comments explain *why*, not *what*.** Delete commented-out code.
- **No TODOs without an owner and a concrete next action.**
- **Naming**: `snake_case` (Python), `camelCase` / `PascalCase` (TS), `SCREAMING_SNAKE` for
  constants. Names describe intent, not type.
- **Never commit secrets, dataset files, model weights, or the LanceDB directory.**

---

## 6. Documentation Research Protocol — MANDATORY

The libraries in this stack (React 19, Tailwind, Shadcn UI, FastAPI, Pydantic v2, LanceDB,
`sentence-transformers`, HF `datasets`) evolve quickly, and several have shipped breaking
changes. **Training-data recall is not an acceptable source for a library-specific API.**

**Before proposing, generating, or debugging any library-specific implementation, you MUST:**

1. **Consult the current official documentation for that library** (as of 2026) — the
   official docs site, the versioned API reference, the changelog/migration guide, or the
   library's own repository. Prefer the official source over blog posts, Stack Overflow,
   or tutorials.
2. **Verify against the version actually installed in this project** — check
   `requirements.txt` / `pyproject.toml` / `package.json`, or run
   `pip show <pkg>` / `npm ls <pkg>`. Docs for a different major version do not count.
3. **State what you verified.** When introducing a library API, name the source and the
   version it applies to in one line (e.g. *"Per the LanceDB Python docs for v0.x,
   `Table.search()` now …"*).
4. **If you cannot verify, say so explicitly** and mark the code as unverified rather
   than presenting a guess as fact. Never invent a parameter, method, or config key.

**Areas where this is especially critical:**

- **LanceDB** — table creation, schema definition, index type/params, and the
  `search()` / query-builder API have all changed across releases.
- **Pydantic v2** — validator decorators, `model_config`, and serialization differ
  substantially from v1. Never emit v1-era code.
- **React 19** — the Actions API, `use()`, ref-as-prop, and removed legacy APIs.
- **Tailwind** — the v3→v4 configuration model (CSS-first config vs `tailwind.config.js`)
  determines the entire setup. Confirm which applies before writing config.
- **Shadcn UI** — the CLI, its React 19 / Tailwind compatibility, and the init flow.
- **`sentence-transformers`** — CLIP model loading, `encode()` signature, and
  normalization defaults (this directly affects whether cosine similarity is correct).

---

## 7. Working Agreement for the AI

- **Plan before coding.** For any non-trivial change, state the files you will touch and
  the approach, then implement.
- **Respect the layering in Section 4.** If a task seems to require crossing a boundary,
  say so and propose where the logic belongs instead.
- **One concern per change.** Do not opportunistically refactor unrelated code.
- **Do not scaffold speculative abstractions.** Build what the current step needs.
- **Report honestly.** If something is untested, partially working, or was skipped, say
  so plainly with the evidence. Never claim a step succeeded without running it.
- **Ask when genuinely ambiguous**; otherwise make the reasonable call and state the
  assumption.
- **Keep this file current.** When an architectural decision is made, record it here or
  in `docs/`.
