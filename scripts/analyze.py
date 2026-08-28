#!/usr/bin/env python3
"""Offline data-quality pass over the ingested corpus.

Reads the CLIP vectors ``scripts/ingest.py`` produced and writes
``data/analysis.json``: for every image its nearest neighbour, the list of
near-duplicate pairs, and — unless ``--no-captions`` — how well each image's own
captions retrieve it.

**What this is for.** A gallery answers "what is in this dataset". These three
measurements answer "what is *wrong* with it", which is the question a
researcher has before they train on it:

* **Near-duplicates.** Flickr8k contains them. At the default threshold this
  finds 52 pairs covering 63 images — including one pair of byte-identical
  JPEGs stored under two different ids, with two different caption sets.
* **Cross-split duplicates.** Twenty-two of those pairs straddle a split
  boundary, train-to-test or train-to-validation. That is evaluation leakage: a
  model can score on a test image it effectively trained on.
* **Caption agreement.** Ranking every image against each of its own captions
  gives the standard recall-at-k for the corpus, and per image a rank that says
  whether its annotations describe it at all. The tail of that distribution is
  a review queue.

**Cost.** The vector arithmetic is free — under a second for the full
8 000-by-8 000 similarity matrix. Encoding the 40 000 captions is not: measured at
75 captions/s on the target CPU, roughly **9 minutes**. That is why this is a
separate, opt-in script rather than part of ingestion, and why ``--no-captions``
exists for the parts that cost nothing.

The artefact is optional in exactly the way ``projection.json`` is: without it
the application serves normally and the quality filters simply do not appear.

Typical use::

    python scripts/analyze.py --no-captions   # duplicates only, ~2 s
    python scripts/analyze.py                 # everything, ~9 min
"""

from __future__ import annotations

import os

# Hugging Face reads these at import time (see scripts/ingest.py).
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final, cast

import lancedb
import numpy as np
from numpy.typing import NDArray

LOGGER: Final = logging.getLogger("analyze")

TABLE_NAME: Final = "images"
OUTPUT_FILE_NAME: Final = "analysis.json"

#: Must be the checkpoint the corpus was embedded with, so this reads the same
#: environment variable the API's ``Settings.clip_model_id`` does — a divergence
#: between the two would compare vectors from two different spaces and report
#: quality figures that mean nothing. Read from the process environment, not
#: from ``.env``: these scripts stay standalone (CLAUDE.md §4.2) and do not
#: import the backend package, so exporting the variable is what propagates it.
CLIP_MODEL_ID: Final = os.environ.get("CORPUSLENS_CLIP_MODEL_ID", "clip-ViT-B-32")

TORCH_DEVICE: Final = "cpu"

#: Cosine above which two images are reported as near-duplicates. 0.95 is well
#: clear of the corpus median nearest-neighbour similarity (0.83), so it selects
#: genuine repeats and re-shoots rather than merely similar scenes.
DEFAULT_DUPLICATE_THRESHOLD: Final = 0.95

#: Images per chunk when multiplying the corpus against itself. The full matrix
#: is 8 000 squared float32 — 256 MB — which is enough to matter on a laptop; a chunk
#: is 32 MB and the loop costs nothing.
_SIMILARITY_CHUNK: Final = 1024

#: Captions per chunk when scoring them against every image, on the same
#: reasoning: 40 000 by 8 000 float32 would be 1.3 GB materialised at once.
_CAPTION_CHUNK: Final = 2048

#: Ceiling on how many pairs the artefact lists. A sane threshold produces
#: dozens; a careless one could produce millions, and the artefact is loaded
#: into memory at startup. Truncation is logged, never silent.
_MAX_REPORTED_PAIRS: Final = 2000

#: Recall thresholds reported for the corpus, matching the convention used in
#: the image-text retrieval literature.
_RECALL_AT: Final = (1, 5, 10)


def _configure_logging(*, verbose: bool) -> None:
    """Send structured, timestamped logs to stderr so tqdm keeps stdout."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector to parse; defaults to ``sys.argv[1:]``.

    Returns:
        The populated argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Measure duplicates, split leakage and caption agreement in the index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-captions",
        action="store_true",
        help="Skip the caption-retrieval pass, the only slow part (~9 min). "
        "Duplicate detection still runs and takes about two seconds.",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=DEFAULT_DUPLICATE_THRESHOLD,
        help="Cosine above which a pair counts as a near-duplicate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Captions per CLIP forward pass.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root of the local data directory.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args(argv)
    if not 0.0 < args.duplicate_threshold <= 1.0:
        parser.error("--duplicate-threshold must be in (0, 1]")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")
    return args


class Corpus:
    """Everything the analysis reads out of the index, loaded once.

    A small carrier rather than four parallel lists threaded through every
    function: the alignment between ids, splits, captions and vector rows is an
    invariant, and keeping them together is what makes it hard to break.
    """

    def __init__(
        self,
        ids: list[str],
        splits: list[str],
        captions: list[list[str]],
        vectors: NDArray[np.float32],
    ) -> None:
        """Bind the columns, which must all be in the same row order.

        Args:
            ids: Corpus image ids, in table scan order.
            splits: Split per row.
            captions: Reference captions per row.
            vectors: ``(n, 512)`` unit-length image embeddings.
        """
        self.ids = ids
        self.splits = splits
        self.captions = captions
        self.vectors = vectors

    @property
    def size(self) -> int:
        """Number of images."""
        return len(self.ids)


def _read_corpus(lancedb_dir: Path) -> Corpus:
    """Load ids, splits, captions and vectors from the index.

    Args:
        lancedb_dir: Directory backing the embedded database.

    Returns:
        The loaded corpus.

    Raises:
        FileNotFoundError: If the database or table is absent.
    """
    if not lancedb_dir.is_dir():
        raise FileNotFoundError(
            f"{lancedb_dir} does not exist. Run `python scripts/ingest.py` first."
        )

    db = lancedb.connect(lancedb_dir)
    if TABLE_NAME not in db.table_names():
        raise FileNotFoundError(
            f"Table {TABLE_NAME!r} not found in {lancedb_dir}. "
            "Run `python scripts/ingest.py` first."
        )

    table = db.open_table(TABLE_NAME)
    projection = table.search().select(["id", "split", "captions", "vector"]).limit(None).to_arrow()
    vectors = projection.column("vector").to_numpy(zero_copy_only=False)
    return Corpus(
        ids=cast(list[str], projection.column("id").to_pylist()),
        splits=cast(list[str], projection.column("split").to_pylist()),
        captions=cast(list[list[str]], projection.column("captions").to_pylist()),
        vectors=np.stack(vectors).astype(np.float32, copy=False),
    )


def _similarity_pass(
    vectors: NDArray[np.float32], threshold: float
) -> tuple[NDArray[np.int64], NDArray[np.float32], list[tuple[int, int, float]]]:
    """Scan the corpus against itself once, collecting both results it can give.

    Cosine reduces to a dot product because the ingestion script stores
    unit-length vectors — that invariant is what makes this one matrix multiply
    rather than a normalisation pass first.

    Two outputs, deliberately different in kind:

    * the **nearest neighbour** of each image, which is a per-image signal;
    * **every** pair above the threshold, which is a corpus-level finding.

    The pair list must be exhaustive rather than derived from the
    nearest-neighbour result. In a cluster of three near-identical images each
    one has a single argmax, so pairs built that way silently drop the rest of
    the cluster — and for leakage detection a missed pair is the one failure
    mode that matters.

    Args:
        vectors: ``(n, 512)`` unit-length embeddings.
        threshold: Cosine above which a pair is recorded.

    Returns:
        Nearest-neighbour index per row, similarity to it, and every
        ``(row, column, similarity)`` above the threshold with ``row < column``.
    """
    count = vectors.shape[0]
    neighbour = np.zeros(count, dtype=np.int64)
    similarity = np.zeros(count, dtype=np.float32)
    pairs: list[tuple[int, int, float]] = []

    for start in range(0, count, _SIMILARITY_CHUNK):
        stop = min(start + _SIMILARITY_CHUNK, count)
        block = vectors[start:stop] @ vectors.T
        # Mask the diagonal: every image is its own nearest neighbour at 1.0,
        # which is true and useless.
        block[np.arange(stop - start), np.arange(start, stop)] = -1.0
        neighbour[start:stop] = block.argmax(axis=1)
        similarity[start:stop] = block.max(axis=1)

        rows, columns = np.where(block > threshold)
        for local_row, column in zip(rows, columns, strict=True):
            row = start + int(local_row)
            # Each unordered pair appears twice in a full scan; keep one.
            if row < int(column):
                pairs.append((row, int(column), float(block[local_row, column])))

    pairs.sort(key=lambda pair: pair[2], reverse=True)
    if len(pairs) > _MAX_REPORTED_PAIRS:
        LOGGER.warning(
            "%d pairs exceed the threshold; reporting the %d most similar. "
            "Raise --duplicate-threshold to narrow this instead of truncating.",
            len(pairs),
            _MAX_REPORTED_PAIRS,
        )
        pairs = pairs[:_MAX_REPORTED_PAIRS]

    return neighbour, similarity, pairs


def _describe_pairs(corpus: Corpus, pairs: list[tuple[int, int, float]]) -> list[dict[str, object]]:
    """Turn index pairs into the records the artefact carries.

    Args:
        corpus: The loaded corpus.
        pairs: ``(row, column, similarity)`` triples, already ordered.

    Returns:
        One record per pair, each flagged with whether it crosses a split.
    """
    return [
        {
            "a": corpus.ids[left],
            "b": corpus.ids[right],
            "a_split": corpus.splits[left],
            "b_split": corpus.splits[right],
            "similarity": round(score, 5),
            # The finding that matters: a near-duplicate spanning two splits
            # means a model can be evaluated on an image it trained on.
            "cross_split": corpus.splits[left] != corpus.splits[right],
        }
        for left, right, score in pairs
    ]


def _caption_ranks(corpus: Corpus, batch_size: int) -> tuple[NDArray[np.int64], dict[str, float]]:
    """Rank every image against each of its own captions.

    This is the standard text→image retrieval evaluation, run over the corpus's
    own annotations: encode a caption, score it against all images, and record
    where the image it belongs to landed. Corpus-wide that yields recall-at-k;
    per image it yields a number that says whether its captions describe it.

    The per-image figure is the **median** rank across its captions, not the
    best. One caption can be lazy without the annotation set being wrong; a poor
    median means none of the five really describe the image, which is the case
    worth surfacing.

    Args:
        corpus: The loaded corpus.
        batch_size: Captions per CLIP forward pass.

    Returns:
        Median rank per image (1 is perfect), and the corpus recall-at-k.
    """
    from sentence_transformers import SentenceTransformer

    flat_captions: list[str] = []
    owner_rows: list[int] = []
    for row, captions in enumerate(corpus.captions):
        for caption in captions:
            flat_captions.append(caption)
            owner_rows.append(row)

    LOGGER.info(
        "Encoding %d caption(s) with %r — the slow part, roughly %.0f min at 75/s",
        len(flat_captions),
        CLIP_MODEL_ID,
        len(flat_captions) / 75 / 60,
    )
    model = SentenceTransformer(CLIP_MODEL_ID, device=TORCH_DEVICE)
    # normalize_embeddings is load-bearing here for the same reason it is in
    # ingest.py: the ranking below is a dot product standing in for a cosine.
    encoded = model.encode(
        flat_captions,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device=TORCH_DEVICE,
    )
    caption_vectors = cast(NDArray[np.float32], encoded).astype(np.float32, copy=False)

    owners = np.asarray(owner_rows, dtype=np.int64)
    ranks = np.zeros(len(flat_captions), dtype=np.int64)

    LOGGER.info("Ranking each caption against all %d images", corpus.size)
    for start in range(0, len(flat_captions), _CAPTION_CHUNK):
        stop = min(start + _CAPTION_CHUNK, len(flat_captions))
        scores = caption_vectors[start:stop] @ corpus.vectors.T
        truth = scores[np.arange(stop - start), owners[start:stop]]
        # Rank without sorting: how many images beat the correct one, plus one.
        ranks[start:stop] = (scores > truth[:, np.newaxis]).sum(axis=1) + 1

    recall = {f"r_at_{k}": round(float((ranks <= k).mean()), 4) for k in _RECALL_AT}
    recall["captions"] = float(len(flat_captions))

    median_rank = np.zeros(corpus.size, dtype=np.int64)
    for row in range(corpus.size):
        owned = ranks[owners == row]
        median_rank[row] = int(np.median(owned)) if owned.size else 0

    return median_rank, recall


def _build_document(
    corpus: Corpus,
    neighbour: NDArray[np.int64],
    similarity: NDArray[np.float32],
    pairs: list[dict[str, object]],
    threshold: float,
    caption_rank: NDArray[np.int64] | None,
    recall: dict[str, float] | None,
) -> dict[str, object]:
    """Assemble the JSON document the API will serve.

    Args:
        corpus: The loaded corpus.
        neighbour: Nearest-neighbour index per row.
        similarity: Similarity to that neighbour.
        pairs: Near-duplicate pairs.
        threshold: Threshold those pairs were selected with.
        caption_rank: Median own-caption rank per image, or ``None`` if skipped.
        recall: Corpus recall-at-k, or ``None`` if skipped.

    Returns:
        A JSON-serialisable mapping.
    """
    images: dict[str, dict[str, object]] = {}
    for row, image_id in enumerate(corpus.ids):
        entry: dict[str, object] = {
            "nn_id": corpus.ids[int(neighbour[row])],
            "nn_similarity": round(float(similarity[row]), 5),
        }
        if caption_rank is not None:
            entry["caption_rank"] = int(caption_rank[row])
        images[image_id] = entry

    document: dict[str, object] = {
        "corpus_size": corpus.size,
        "duplicate_threshold": threshold,
        "duplicate_pairs": pairs,
        "images": images,
    }
    if recall is not None:
        document["caption_retrieval"] = recall
    return document


def _write_atomically(output_path: Path, document: dict[str, object]) -> None:
    """Write via a temporary sibling, then rename.

    The API reads this at startup; a half-written file would be a parse error at
    exactly the wrong moment, and a rename is atomic on the local filesystems
    this tool targets.
    """
    temporary = output_path.with_name(f".{output_path.name}.partial")
    temporary.write_text(json.dumps(document), encoding="utf-8")
    temporary.replace(output_path)


def analyze(args: argparse.Namespace) -> int:
    """Run the analysis and write the result.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The number of near-duplicate pairs found.
    """
    data_dir: Path = args.data_dir
    corpus = _read_corpus(data_dir / "lancedb")
    LOGGER.info("Read %d image(s) of %d dimension(s)", corpus.size, corpus.vectors.shape[1])

    neighbour, similarity, index_pairs = _similarity_pass(corpus.vectors, args.duplicate_threshold)
    pairs = _describe_pairs(corpus, index_pairs)
    crossing = sum(1 for pair in pairs if pair["cross_split"])
    LOGGER.info(
        "Near-duplicates above %.2f: %d pair(s), of which %d cross a split boundary",
        args.duplicate_threshold,
        len(pairs),
        crossing,
    )
    if crossing:
        LOGGER.warning(
            "%d near-duplicate pair(s) span two splits — evaluation on those test "
            "images is measuring memorisation, not generalisation",
            crossing,
        )

    caption_rank: NDArray[np.int64] | None = None
    recall: dict[str, float] | None = None
    if args.no_captions:
        LOGGER.info("--no-captions: skipping the caption-retrieval pass")
    else:
        caption_rank, recall = _caption_ranks(corpus, args.batch_size)
        LOGGER.info(
            "Caption→image retrieval over the corpus: %s",
            ", ".join(f"R@{k}={recall[f'r_at_{k}']:.1%}" for k in _RECALL_AT),
        )

    document = _build_document(
        corpus, neighbour, similarity, pairs, args.duplicate_threshold, caption_rank, recall
    )
    output_path = data_dir / OUTPUT_FILE_NAME
    _write_atomically(output_path, document)
    LOGGER.info("Wrote %s", output_path)
    return len(pairs)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        A process exit code: 0 on success, 1 on a missing index, 130 if
        interrupted.
    """
    args = _parse_args(argv)
    _configure_logging(verbose=args.verbose)
    try:
        analyze(args)
    except FileNotFoundError as error:
        LOGGER.error("%s", error)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted — rerun the same command")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
