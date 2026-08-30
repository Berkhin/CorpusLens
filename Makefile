# Task runner for CorpusLens.
#
# Two kinds of target, and the split is deliberate (CLAUDE.md §2: Docker is a
# supported but optional run path, and neither way in may become the only one):
#
#   * `up` / `down` / `build` / `logs` / `setup` drive Docker Compose.
#   * `lint` / `test` / `format` run the host toolchain directly, so they work
#     without Docker and finish in seconds rather than minutes.
#
# The quality targets run exactly what .github/workflows/ci.yml runs, in the
# same order. `make lint && make test` passing locally means CI passes.

SHELL := /bin/bash
.DEFAULT_GOAL := start

COMPOSE := docker compose

# Passed to the `user:` keys in docker-compose.yml so containers write files as
# the caller rather than as root. Both must be *exported*: `UID` is a shell
# variable that bash and zsh never export, and `GID` is normally unset, so
# without these two lines Compose would silently fall back to its 1000 default.
# Computed with `id` rather than read from the environment for the same reason.
export UID := $(shell id -u)
export GID := $(shell id -g)

# Prefer the project venv documented in requirements.txt; fall back to whatever
# `python3` is on PATH so the targets still work in a container or in CI.
VENV_PYTHON := backend/.venv/bin/python
PYTHON      := $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python3)

# `--prefix` rather than `cd frontend &&`: each make recipe line is its own
# shell, so a bare `cd` would not survive to the next one.
NPM := npm --prefix frontend

.PHONY: start help up down build logs setup lint lint-backend lint-frontend \
        test test-backend test-frontend test-docker format format-check clean

# ---------------------------------------------------------------------------
# The one command
# ---------------------------------------------------------------------------

# The default goal, so a bare `make` on a fresh clone does the whole thing.
#
# Ordered dependencies rather than one recipe: `setup` must finish before `up`
# starts, because the API refuses to boot without a corpus. Both halves stay
# runnable on their own — `make up` after a reboot should not re-run a
# fifteen-minute job.
#
# Safe to repeat. Ingestion skips records already in the table, so on a machine
# that is already set up this re-derives the projection and the quality
# measurements (seconds) and then starts the stack.
start: setup up ## Set up the corpus if needed, then start the app

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help: ## Show this help
	@echo 'CorpusLens — available targets:'
	@echo
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo 'First run: make start   (one time, ~15 min, downloads ~3 GB)'

# ---------------------------------------------------------------------------
# Docker Compose
# ---------------------------------------------------------------------------

up: ## Start backend + frontend in the background (hot reload)
	$(COMPOSE) up -d
	@echo 'Frontend  http://localhost:5173'
	@echo 'API docs  http://localhost:8000/docs'

down: ## Stop and remove the containers
	$(COMPOSE) down

build: ## Build (or rebuild) the images
	$(COMPOSE) build

logs: ## Follow the container logs
	$(COMPOSE) logs -f

# Extra arguments reach scripts/ingest.py, e.g. `make setup ARGS='--limit 100'`.
#
# `data/` is created here, by the caller, rather than left to Docker. A bind
# mount whose source is missing is created by the daemon as root, which the
# containers — now running as the host user — could not then write to. Creating
# it first is what makes that ownership work on the very first run.
#
# `app` is built explicitly because it is the only service carrying `build:` for
# this image; `setup` declares `image:` alone to keep two builders off one tag.
setup: ## One-time corpus download, embedding and projection (~15 min)
	@mkdir -p data
	$(COMPOSE) --profile app build app
	$(COMPOSE) run --rm setup $(ARGS)

# ---------------------------------------------------------------------------
# Quality — same commands, same order, as CI
# ---------------------------------------------------------------------------

lint: lint-backend lint-frontend ## Lint and type-check everything

lint-backend: ## Ruff (lint + format check) and mypy --strict
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m mypy

lint-frontend: ## tsc --build and ESLint
	$(NPM) run typecheck
	$(NPM) run lint

test: test-backend ## Run the backend test suite

test-backend: ## pytest — no dataset, weights or LanceDB store required
	$(PYTHON) -m pytest

test-frontend: ## vitest, for the frontend unit tests
	$(NPM) run test

# Both suites against the image's own dependency set rather than the host's,
# which is the point: it catches a Linux/Node-24 failure that the macOS host
# would not reproduce. `exec` needs the containers already up, so this checks
# first and says so rather than failing on a bare "service not running".
#
# -T disables TTY allocation. Without it this breaks the moment it runs
# anywhere without a terminal — a CI job, or a `make -j` build.
test-docker: ## Run pytest and vitest inside the running containers
	@for svc in backend frontend; do \
		$(COMPOSE) ps --services --status running 2>/dev/null | grep -qx $$svc \
			|| { echo "make test-docker: '$$svc' is not running — start it with 'make up'"; exit 1; }; \
	done
	$(COMPOSE) exec -T backend python -m pytest
	$(COMPOSE) exec -T frontend npm run test

# `ruff check --fix` runs first because import order is a lint rule in Ruff (the
# "I" set), not a format rule: `ruff format` alone leaves imports unsorted and
# `make lint` then fails on a tree that was just formatted.
format: ## Auto-format with Ruff and Prettier
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .
	$(NPM) run format

format-check: ## Verify formatting without writing changes
	$(PYTHON) -m ruff format --check .
	$(NPM) run format:check

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

# Drops the node_modules volume too, which is the fix when a frontend
# dependency changes: the named volume keeps the copy made when it was first
# created and will not otherwise pick up a new package-lock.json. Leaves ./data
# alone — that is a bind mount, and re-creating it means another 15 minutes.
clean: ## Remove containers and the node_modules volume (keeps ./data)
	$(COMPOSE) down --volumes --remove-orphans
