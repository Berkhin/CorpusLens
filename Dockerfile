# syntax=docker/dockerfile:1
#
# Single-image build of CorpusLens: the React SPA is compiled in one
# stage and served as static files by the FastAPI process in the next, so the
# whole tool is one container on one port. That is deliberate — the point of
# containerising a local research tool is that someone else can run it without
# installing Python, Node, or matching either version.
#
# Nothing here reaches the network at runtime. The one-time corpus download
# happens in the `setup` service of docker-compose.yml (CLAUDE.md §2).


# --------------------------------------------------------------------------- #
# Stage 1 — compile the SPA                                                     #
# --------------------------------------------------------------------------- #
FROM node:24-alpine AS web

WORKDIR /web

# Manifests first so `npm ci` is cached independently of source edits.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

# An *empty* base URL means "same origin", which is exactly right here: one
# process serves both the SPA and the API, so the client must not hardcode the
# http://localhost:8000 that the split Vite dev setup relies on.
#
# Written to .env.production rather than passed as a build ARG because that file
# is Vite's documented, mode-scoped input for a production build — it does not
# depend on how Vite happens to treat prefixed variables in the process
# environment.
RUN printf 'VITE_API_BASE_URL=\n' > .env.production \
    && npm run build


# --------------------------------------------------------------------------- #
# Stage 2 — Python runtime                                                      #
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

# HF_HOME points inside the mounted data volume so the ~580 MB of CLIP weights
# are downloaded once and survive an image rebuild. Telemetry off per CLAUDE.md §2.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/data/hf \
    PYTHONPATH=/app/backend \
    CORPUSLENS_FRONTEND_DIST_DIR=/app/frontend-dist \
    CORPUSLENS_HOST=0.0.0.0 \
    CORPUSLENS_PORT=8000

WORKDIR /app

# torch is installed first and separately, from PyTorch's CPU index, because on
# Linux the default PyPI wheel bundles CUDA and weighs ~2 GB. This image has no
# GPU to reach, so it takes the CPU build deliberately — a CUDA deployment would
# override this line and let `resolve_device` find the accelerator.
#
# Unpinned, unlike the reference environment. requirements.txt caps torch at
# 2.2.2 only for macOS x86_64 (the last wheel published for it) and asks every
# other platform for >=2.4, so naming 2.2.2 here would install a wheel the next
# layer immediately replaces — from PyPI, i.e. with the CUDA build this line
# exists to avoid. lancedb 0.25.3 publishes manylinux wheels for x86_64 and
# aarch64 alike, so this builds natively on either host.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Everything else from PyPI. torch is already satisfied by the layer above, so
# this does not pull it again; it does resolve numpy, which follows the same
# platform split.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY docker/ ./docker/
COPY --from=web /web/dist ./frontend-dist

EXPOSE 8000

# `python -m app` rather than the uvicorn CLI so the bind address comes from
# CORPUSLENS_HOST / CORPUSLENS_PORT above — the same settings object the rest of the
# configuration goes through. The uvicorn CLI reads neither.
CMD ["python", "-m", "app"]
