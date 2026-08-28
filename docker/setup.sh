#!/usr/bin/env bash
#
# One-time offline preparation, run inside the container by the `setup` service.
#
# Both steps are idempotent, so re-running this after an interrupted attempt
# resumes rather than starting over. Any arguments given to
# `docker compose run --rm setup` are forwarded to the ingestion script — most
# usefully `--limit N` for a fast end-to-end check.

set -euo pipefail

echo "==> Ingesting the corpus and embedding it with CLIP (this is the slow part)"
python scripts/ingest.py "$@"

# The projection is derived from every vector in the table, so it is recomputed
# from scratch whenever ingestion has just changed the corpus.
echo "==> Projecting the embeddings to 2-D"
python scripts/project.py --force

# The data-quality pass is opt-in even here: its caption-retrieval half re-encodes
# every caption in the corpus, which roughly doubles the setup time. Pass
# ANALYZE_CAPTIONS=1 to include it.
echo "==> Measuring duplicates and split leakage"
if [ "${ANALYZE_CAPTIONS:-0}" = "1" ]; then
  python scripts/analyze.py
else
  python scripts/analyze.py --no-captions
  echo "    (skipped caption retrieval; re-run with ANALYZE_CAPTIONS=1 for R@k and"
  echo "     the weak-captions filter — it adds about nine minutes)"
fi

echo "==> Done. Start the app with: docker compose up"
