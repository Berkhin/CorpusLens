# Contributing to CorpusLens

Thanks for taking the time — issues, bug reports, docs fixes and pull requests are all
welcome. This page gets you from a fresh clone to a green PR.

New to the codebase? [`CLAUDE.md`](./CLAUDE.md) is the engineering contract: layering rules,
coding standards, and the constraints behind them. Skim it before your first change — it
answers most "why is it like this?" questions.

---

## Setting up

Two supported paths. **Neither is the blessed one** — Docker is the fastest way in, native is
better for tight edit/run loops. CI exercises the native toolchain; `make test-docker`
exercises the container one.

### Option A — Docker (fastest)

**Needs:** Docker with Compose v2, ~4 GB free disk.

```bash
git clone https://github.com/Berkhin/corpuslens.git
cd corpuslens

make start     # download, embed, project (~15 min, ~3 GB), then bring the stack up
```

That is the default goal, so a bare `make` does the same. It runs `setup` then `up`; both
remain available separately, which is what you want after a reboot (`make up` alone) or after
changing the pipeline (`make setup` alone).

Use `make setup ARGS='--limit 100' && make up` for a ~2 minute end-to-end check first — the
limit bounds embedding, not the download, so expect ~2.7 GB regardless
([#3](https://github.com/Berkhin/CorpusLens/issues/3)). Source is bind-mounted from the host,
so your editor drives the containers directly.

### Option B — Native

**Needs:** **Python 3.12 exactly**, Node.js 20+ (CI uses 24).

> Python 3.12 is a ceiling, not a preference: `torch`'s last macOS x86_64 wheel is 2.2.2,
> which supports cp38–cp312 only.

```bash
# Backend — from the repository root
python3.12 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Frontend
npm --prefix frontend install
```

Build the index once, then run the two servers in separate terminals:

```bash
python scripts/ingest.py --limit 100     # or omit --limit for the full corpus
python scripts/project.py                # optional: enables the map view
python scripts/analyze.py --no-captions  # optional: enables quality filters

uvicorn app.main:app --app-dir backend --reload   # API    → http://localhost:8000
npm --prefix frontend run dev                     # Client → http://localhost:5173
```

`--reload` is a uvicorn CLI flag, which is why that's the dev command. It ignores the host and
port from `.env`; to honour those, use `PYTHONPATH=backend python -m app` instead.

Full detail — flags, resumability, timings, the optional artefacts — is in
[`docs/guide.md`](./docs/guide.md#setup).

---

## Make targets

`make` on its own prints this list. The split is deliberate: the Docker targets drive Compose,
the quality targets run the **host** toolchain so they work without Docker and finish in
seconds.

| Target | What it does |
|---|---|
| `make start` | **The default goal.** `setup` then `up` — everything, from a clean clone. |
| `make setup` | One-time corpus download, embedding and projection. Takes `ARGS='--limit 100'`. |
| `make up` | Start backend + frontend in the background, with hot reload. |
| `make down` | Stop and remove the containers. |
| `make logs` | Follow container logs. |
| `make build` | Build or rebuild the images. |
| **`make lint`** | **Ruff (lint + format check), mypy `--strict`, `tsc -b`, ESLint.** |
| **`make test`** | **Backend `pytest`.** No dataset, weights or LanceDB store needed. |
| `make test-frontend` | Vitest unit tests. |
| `make test-docker` | Both suites *inside* the running containers — catches Linux/Node-24 failures a macOS host won't reproduce. Requires `make up` first. |
| `make format` | Auto-fix with Ruff and Prettier. Run this before `make lint`. |
| `make format-check` | Verify formatting without writing. |
| `make clean` | Remove containers and the `node_modules` volume. **Keeps `./data`.** |

**`make lint && make test` runs exactly what CI runs, in the same order.** Green locally means
green on the PR.

The test suite is hermetic by construction — `conftest.py` stubs out LanceDB and the CLIP
encoder, so it needs no dataset, no weights and no network.

---

## Making a change

### Branch naming

Branch off `main` as `<type>/<short-kebab-slug>`:

```
feat/collection-import      fix/search-empty-query
docs/api-examples           refactor/filter-resolver
chore/bump-lancedb          test/export-service
```

Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`.

### Commits

[Conventional Commits](https://www.conventionalcommits.org/) — `feat: add collection import`,
`fix: handle empty search query`. Present tense, imperative mood. Explain *why* in the body if
the *what* isn't self-evident.

### Code standards

Enforced by `make lint`; the full rules are in [`CLAUDE.md`](./CLAUDE.md) §5.

**Python** — type hints on every signature (`mypy --strict` must pass), Google-style docstrings
on public APIs, `pathlib.Path` over `os.path`, stdlib `logging` (no `print()` outside
`scripts/`), config via `pydantic-settings` rather than hardcoded values.

**TypeScript** — `strict: true`, `any` forbidden (use `unknown` and narrow), explicit return
types on exports, functional components only, named exports. Data fetching lives in TanStack
Query hooks, never in `useEffect`.

**Both** — respect the layering (`routes → services → repositories`). If a change seems to
need a boundary crossing, say so in the PR and propose where the logic belongs instead.
Comments explain *why*, not *what*.

### Adding a dependency

Propose it in the issue or PR description first, with a paragraph on why it's needed and
confirmation that it satisfies the project's constraints: **local only, no cloud services, no
external vector DB, no paid APIs, no telemetry, CPU-only torch.**

### Working with library APIs

This stack moves fast, and several parts have shipped breaking changes. When you touch a
library-specific API — LanceDB, Pydantic v2, React 19, Tailwind v4, Shadcn, `sentence-transformers` —
check the **current official docs for the version actually installed**, and mention what you
verified in the PR. Don't ship an API call you couldn't confirm.

---

## Opening a pull request

Before you push:

- [ ] `make format` — then `make lint` passes clean
- [ ] `make test` passes; new behaviour has a test, fixed bugs have a regression test
- [ ] Docs updated if you changed setup, config, or the API surface
- [ ] Commits follow the convention above
- [ ] No dataset files, model weights, `data/` contents or secrets committed

In the PR description, cover **what** changed, **why**, and **how you verified it**. Screenshots
or a short clip for UI changes are very welcome. Small, focused PRs get reviewed faster —
please keep unrelated refactors in their own branch.

CI runs the backend job (Ruff, mypy, pytest) and the frontend job (tsc, ESLint, Prettier,
Vitest) on every pull request. **Both must be green before merge.**

If something's broken, unclear, or just took you longer than it should have — open an issue.
That's useful feedback, not noise.
