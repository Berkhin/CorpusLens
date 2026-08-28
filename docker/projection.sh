#!/usr/bin/env bash
#
# Switch the map between its two projections while the app is running.
#
# Written for a live demo, where the cost that matters is the wait in front of
# an audience. Each variant is computed once and kept as
# `data/projection.<method>.json`; switching afterwards is a copy plus a
# restart, so the second and every later switch costs only the ~15 s the CLIP
# encoder needs to load.
#
# A restart is unavoidable, not an oversight: the API reads the artefact once at
# startup (`app/core/lifespan.py`) and holds it in memory, because re-reading a
# file that cannot change during a session would be pure waste.
#
# Usage:
#   docker/projection.sh tsne              # switch to t-SNE
#   docker/projection.sh pca               # switch back
#   docker/projection.sh tsne --recompute  # ignore the cache and recompute
#   docker/projection.sh pca  --rebuild    # rebuild the image first
#
# Note that --rebuild is almost never what you want: the projection is *data*,
# mounted from ./data, and it is not baked into the image. Rebuild only after
# changing code.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Where to reach the running container. Overridable for a non-default published
# port: `API_URL=http://localhost:9000 docker/projection.sh tsne`.
API_URL="${API_URL:-http://localhost:8000}"

method=""
rebuild=0
recompute=0

for argument in "$@"; do
  case "$argument" in
    pca | tsne) method="$argument" ;;
    --rebuild) rebuild=1 ;;
    --recompute) recompute=1 ;;
    *)
      echo "unknown argument: $argument" >&2
      echo "usage: docker/projection.sh {pca|tsne} [--rebuild] [--recompute]" >&2
      exit 2
      ;;
  esac
done

if [ -z "$method" ]; then
  echo "usage: docker/projection.sh {pca|tsne} [--rebuild] [--recompute]" >&2
  exit 2
fi

cached="data/projection.${method}.json"

if [ "$rebuild" -eq 1 ]; then
  echo "==> Rebuilding the image"
  docker compose build app
fi

if [ "$recompute" -eq 1 ] || [ ! -f "$cached" ]; then
  echo "==> Computing the ${method} projection (PCA ~5 s, t-SNE ~45 s in the container)"
  # The setup service's entrypoint is the full offline pipeline; overriding it
  # runs just this one step against the same image and the same mounted data.
  docker compose run --rm --entrypoint python setup \
    scripts/project.py --method "$method" --force
  cp data/projection.json "$cached"
else
  echo "==> Reusing the cached ${method} projection ($cached)"
fi

cp "$cached" data/projection.json

echo "==> Restarting the API so it picks the artefact up at startup"
docker compose up -d --force-recreate app

echo "==> Waiting for startup (the CLIP encoder loads first)"
until curl -sf "${API_URL}/api/dataset/stats" >/dev/null 2>&1; do sleep 1; done

# Assert rather than announce: a demo that silently keeps serving the old map is
# the one failure mode this script exists to prevent.
served="$(curl -s "${API_URL}/api/projection" |
  python3 -c 'import json, sys; print(json.load(sys.stdin)["method"])')"

if [ "$served" != "$method" ]; then
  echo "==> FAILED: asked for ${method}, the API is serving ${served}" >&2
  exit 1
fi

echo "==> Serving ${served}. Open ${API_URL} and switch to the map tab."
