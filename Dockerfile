# syntax=docker/dockerfile:1
#
# Single-image build of the Flickr8k Explorer: the React SPA is compiled in one
# stage and served as static files by the FastAPI process in the next, so the
# whole tool is one container on one port. That is deliberate — the point of
# containerising a local research tool is that someone else can run it without
# installing Python, Node, or matching either version.
#
# Nothing here reaches the network at runtime. The one-time corpus download
# happens in the `setup` service of compose.yaml (CLAUDE.md §2).


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
    FLICKR8K_FRONTEND_DIST_DIR=/app/frontend-dist \
    FLICKR8K_HOST=0.0.0.0 \
    FLICKR8K_PORT=8000

WORKDIR /app

# torch is installed first and separately, from PyTorch's CPU index, because on
# Linux the default PyPI wheel bundles CUDA and weighs ~2 GB — useless here and
# ruled out by CLAUDE.md §2.
#
# The pin is the same 2.2.2 as requirements.txt: verified that the CPU index
# carries torch-2.2.2+cpu-cp312-linux_x86_64 *and* torch-2.2.2-cp312-manylinux
# aarch64, and that lancedb 0.25.3 publishes manylinux wheels for both arches.
# So the container builds natively on an Intel or an Apple Silicon host and the
# dependency set never forks from the one used on the host.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2

# Everything else from PyPI. torch is already satisfied (PEP 440 treats the
# local version 2.2.2+cpu as matching ==2.2.2), so this does not pull it again.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY docker/ ./docker/
COPY --from=web /web/dist ./frontend-dist

EXPOSE 8000

# `python -m app` rather than the uvicorn CLI so the bind address comes from
# FLICKR8K_HOST / FLICKR8K_PORT above — the same settings object the rest of the
# configuration goes through. The uvicorn CLI reads neither.
CMD ["python", "-m", "app"]
