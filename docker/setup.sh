#!/usr/bin/env bash
#
# One-time offline preparation, run inside the container by the `setup` service.
#
# Both steps are idempotent, so re-running this after an interrupted attempt
# resumes rather than starting over. Any arguments given to
# `docker compose run --rm setup` are forwarded to the ingestion script — most
# usefully `--limit N` for a fast end-to-end check.
#
# `--limit N` bounds the *embedding*, not the download. Hugging Face resolves a
# split slice by fetching whole parquet shards, and the CLIP checkpoint is
# fetched in full regardless, so even `--limit 20` writes about 2.7 GB into
# data/ (measured: 1.1 GB of shards, 1.6 GB of weights). It saves CPU minutes,
# not bandwidth. Making the limit reach the download is tracked upstream as
# "[Enhancement] Lazy streaming for dataset downloads".

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

# `make up` rather than `docker compose up app`: the README's path is the
# two-container dev stack with hot reload on :5173, and `make start` runs this
# script and then that. Naming the single-image `app` profile here sent anyone
# following the output to a different topology than the one documented.
echo "==> Done. Start the app with: make up   (or 'make start' to do both)"
