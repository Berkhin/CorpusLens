# syntax=docker/dockerfile:1
#
# Development image for the FastAPI backend.
#
# This is the *contributor* image, not the one users run. The root `Dockerfile`
# stays the single-image production path — SPA compiled in and served by FastAPI
# on one port. This one installs dependencies and nothing else: the source is
# bind-mounted by docker-compose.yml so `--reload` picks up host edits.
#
# Build context is the repository root (see docker-compose.yml), so the COPY
# paths below are repo-relative.

FROM python:3.12-slim

# HF_HOME points inside the mounted data volume so the ~580 MB of CLIP weights
# are downloaded once and survive a rebuild. PYTHONPATH mirrors the `--app-dir
# backend` that the documented host command uses, so `app.*` imports resolve the
# same way in both. Telemetry off per CLAUDE.md §2.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/data/hf \
    PYTHONPATH=/app/backend

WORKDIR /app

# torch first and separately, from PyTorch's CPU index: on Linux the default
# PyPI wheel bundles CUDA and weighs ~2 GB, and this container has no GPU to
# reach. Unpinned because requirements.txt caps torch at 2.2.2 only for macOS
# x86_64 and asks Linux for >=2.4 — pinning 2.2.2 here would install a wheel the
# next layer replaces with the CUDA build this line exists to avoid.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# torch is already satisfied by the layer above, so this does not pull it again.
# requirements-dev.txt is included because this is the image a contributor
# without a host venv would run `ruff`/`mypy`/`pytest` in.
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt -r requirements-dev.txt

EXPOSE 8000

# The uvicorn CLI rather than `python -m app`, because `--reload` belongs to the
# CLI (backend/app/__main__.py says as much). That CLI ignores CORPUSLENS_HOST, so
# the bind address is passed explicitly — 0.0.0.0 is safe here in a way it is
# not on the host, since the container boundary is what limits reach.
#
# watchfiles is deliberately not installed: uvicorn then falls back to
# StatReload, which polls mtimes (verified in uvicorn/supervisors/__init__.py).
# That is exactly what a macOS→Linux bind mount needs, as inotify events do not
# cross it. --reload-dir keeps the poll off the mounted data/ directory.
CMD ["uvicorn", "app.main:app", \
     "--app-dir", "backend", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--reload-dir", "backend/app"]
