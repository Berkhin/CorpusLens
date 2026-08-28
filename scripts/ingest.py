#!/usr/bin/env python3
"""One-time, offline ingestion of the Flickr8k dataset.

Pulls ``jxie/flickr8k`` through Hugging Face ``datasets``, writes each image's
original JPEG bytes to ``data/images/`` (so FastAPI can serve them statically
later), encodes every image with CLIP ``clip-ViT-B-32`` on the CPU, and stores
the resulting 512-d vectors alongside their five reference captions in an
embedded LanceDB table under ``data/lancedb/``.

This script is the *only* place CLIP image inference happens. Measured on the
target machine (Core i9-9980HK, 8 torch threads) throughput is ~10 images/s, so
a full pass over the ~8k corpus takes roughly 15 minutes — far too slow to sit
inside a request handler, which is why the serving API never embeds images per
request and only queries the index this script produces (CLAUDE.md §2).

Idempotent and resumable: records already present in the table are skipped, so
an interrupted run can simply be restarted. ``--force`` rebuilds from scratch;
``--limit N`` processes only the first N records for a fast end-to-end check.

Typical use::

    python scripts/ingest.py --limit 100    # ~10s of embedding — smoke test
    python scripts/ingest.py                # full corpus — ~15 min on CPU
"""

from __future__ import annotations

import os

# Hugging Face reads these at import time, so they must be set before the
# `datasets` / `sentence_transformers` imports below. `setdefault` keeps any
# value the operator already exported. Telemetry is disabled per CLAUDE.md §2.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Final, cast

import lancedb
import numpy as np
import torch
from datasets import Dataset, load_dataset
from datasets import Image as HfImage
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import Table
from numpy.typing import NDArray
from PIL import Image as PilImage
from PIL.Image import Image as PilImageType
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

LOGGER: Final = logging.getLogger("ingest")

HF_DATASET_ID: Final = os.environ.get("FLICKR8K_HF_DATASET_ID", "jxie/flickr8k")

#: The checkpoint the whole corpus is embedded with, and therefore the one the
#: API must encode queries with. It reads the same environment variable through
#: ``Settings.clip_model_id``; a divergence yields a shared space that is not
#: shared, and search degrades to noise instead of failing. Read from the
#: process environment, not from ``.env``: this script stays standalone
#: (CLAUDE.md §4.2) and does not import the backend package, so exporting the
#: variable is what propagates it to both sides.
CLIP_MODEL_ID: Final = os.environ.get("FLICKR8K_CLIP_MODEL_ID", "clip-ViT-B-32")

#: CLIP ViT-B/32 projects images and text into a shared 512-d space. Pinned to
#: the schema rather than derived, so a model whose width differs fails at the
#: first insert rather than writing a table the API cannot query.
EMBEDDING_DIM: Final = 512

#: Flickr8k's canonical splits, in the order they are ingested.
SPLITS: Final = ("train", "validation", "test")

#: The dataset stores the five references as flat columns, not as a list.
CAPTION_COLUMNS: Final = tuple(f"caption_{index}" for index in range(5))

TABLE_NAME: Final = "images"
DEFAULT_BATCH_SIZE: Final = 32

#: CLAUDE.md §2 forbids CUDA/MPS: this project targets an Intel Mac, and torch
#: 2.2.2 (the last build with a macOS x86_64 wheel) has no usable accelerator
#: here anyway. The device is a constant, never auto-detected.
TORCH_DEVICE: Final = "cpu"


# lancedb ships no py.typed marker, so `LanceModel` resolves to `Any` and strict
# mode rejects subclassing it.
class ImageRecord(LanceModel):  # type: ignore[misc]
    """One Flickr8k image: its file on disk, its captions, and its CLIP vector.

    This is both the LanceDB table schema and the ingestion write model. The
    ``vector`` column materializes as Arrow ``fixed_size_list<float>[512]``
    (float32), which is what LanceDB's cosine metric expects.

    Attributes:
        id: Flickr photo id — the original filename without its extension.
        file_name: Basename under ``data/images/``, safe to join onto the
            static-files root.
        split: Source split (``train`` / ``validation`` / ``test``). Not in the
            minimal spec, but the statistics view needs per-split counts and
            recovering it later would mean a second pass over the dataset.
        captions: The five human reference captions, in dataset column order.
        vector: L2-normalized CLIP image embedding.
    """

    id: str
    file_name: str
    split: str
    captions: list[str]
    # `Vector` builds the Arrow type dynamically, which mypy cannot evaluate as
    # a static annotation.
    vector: Vector(EMBEDDING_DIM)  # type: ignore[valid-type]


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
        description="Ingest Flickr8k into data/images/ and an embedded LanceDB table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N records (across splits, train first). "
        "Intended for fast end-to-end testing; omit to ingest everything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and rebuild the table instead of resuming.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Images per CLIP forward pass.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="Cap torch's CPU threads. Default: let torch decide.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root of the local data directory.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be a positive integer")
    return args


def _load_splits(cache_dir: Path, limit: int | None) -> list[tuple[str, Dataset]]:
    """Load each split, honouring a global record budget.

    ``--limit`` is applied with Hugging Face slice syntax (``train[:N]``), always
    as a *prefix*, so record ``i`` of a limited run is the same record as ``i``
    of a full run. That keeps generated ids stable, which is what lets a limited
    smoke-test run and a later full run share one table.

    The image column is cast to ``decode=False`` so we get the original encoded
    JPEG bytes plus the source filename instead of a decoded PIL object. Both
    matter: the bytes let us persist the file losslessly, and the filename is
    Flickr8k's canonical photo id, which PIL discards on decode.

    Args:
        cache_dir: Directory for the Hugging Face download cache.
        limit: Total records to load across all splits, or ``None`` for all.

    Returns:
        ``(split_name, dataset)`` pairs, skipping splits past the budget.
    """
    splits: list[tuple[str, Dataset]] = []
    remaining = limit

    for split in SPLITS:
        if remaining is not None and remaining <= 0:
            break
        spec = split if remaining is None else f"{split}[:{remaining}]"
        LOGGER.info("Loading %s split %r ...", HF_DATASET_ID, spec)
        dataset = load_dataset(HF_DATASET_ID, split=spec, cache_dir=str(cache_dir))
        if not isinstance(dataset, Dataset):  # pragma: no cover - defensive
            raise TypeError(f"Expected a Dataset for split {spec!r}, got {type(dataset)!r}")
        dataset = dataset.cast_column("image", HfImage(decode=False))
        splits.append((split, dataset))
        if remaining is not None:
            remaining -= dataset.num_rows

    return splits


def _open_table(db: lancedb.DBConnection, *, force: bool) -> Table:
    """Open the images table, creating it (or recreating it) as needed.

    Args:
        db: An open LanceDB connection.
        force: Drop any existing table first.

    Returns:
        A table conforming to :class:`ImageRecord`.
    """
    if force and TABLE_NAME in db.table_names():
        LOGGER.warning("--force: dropping existing table %r", TABLE_NAME)
        db.drop_table(TABLE_NAME)

    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        LOGGER.info("Opened table %r (%d existing rows)", TABLE_NAME, table.count_rows())
        return table

    LOGGER.info("Creating table %r", TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=ImageRecord)


def _existing_ids(table: Table) -> set[str]:
    """Return the ids already stored, so a resumed run can skip them.

    Projects the ``id`` column only — pulling full rows would drag every vector
    into memory for no reason.
    """
    if table.count_rows() == 0:
        return set()
    projection = table.search().select(["id"]).limit(None).to_arrow()
    return set(projection.column("id").to_pylist())


def _record_identity(raw_image: dict[str, object], split: str, index: int) -> tuple[str, str]:
    """Derive a stable ``(id, file_name)`` pair for one dataset row.

    Prefers the dataset's own filename (Flickr8k's photo id, e.g.
    ``2513260012_03d33305cf.jpg``) and falls back to a positional name if the
    parquet ever omits it.

    Only the basename is kept. These names come from dataset metadata and are
    later joined onto a static-files root, so any directory component is
    stripped to keep a crafted path from escaping ``data/images/``.

    Args:
        raw_image: The undecoded image struct, with ``bytes`` and ``path`` keys.
        split: Split the row came from.
        index: Row index within the split.

    Returns:
        The record id and its on-disk file name.
    """
    source_path = raw_image.get("path")
    name = Path(str(source_path)).name if source_path else ""
    if not name or name in {".", ".."}:
        name = f"{split}_{index:05d}.jpg"
    return Path(name).stem, name


def _extract_captions(batch: dict[str, list[object]], offset: int) -> list[str]:
    """Collect the five reference captions for one row, dropping blanks."""
    captions: list[str] = []
    for column in CAPTION_COLUMNS:
        values = batch.get(column)
        if not values:
            continue
        raw = values[offset]
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            captions.append(text)
    return captions


def _write_image(destination: Path, payload: bytes) -> None:
    """Persist image bytes verbatim, atomically.

    Writing to a temporary sibling and renaming means an interrupted run can
    never leave a half-written JPEG that a later resume would treat as done.
    """
    if destination.exists():
        return
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def _encode_images(
    model: SentenceTransformer,
    images: list[PilImageType],
    batch_size: int,
) -> NDArray[np.float32]:
    """Embed images into the shared CLIP space.

    ``normalize_embeddings=True`` is essential and *not* the default: LanceDB's
    cosine metric — and the dot-product shortcut the search service will use —
    are only correct on unit-length vectors.

    Args:
        model: The loaded CLIP bi-encoder.
        images: Decoded RGB images.
        batch_size: Images per forward pass.

    Returns:
        An ``(len(images), EMBEDDING_DIM)`` float32 array of unit vectors.
    """
    embeddings = model.encode(
        images,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device=TORCH_DEVICE,
    )
    # `encode` is typed as a union because its return type depends on runtime
    # flags; `convert_to_numpy=True` pins it to an ndarray.
    array = cast(NDArray[np.float32], embeddings)
    return array.astype(np.float32, copy=False)


def _ingest_batch(
    batch: dict[str, list[object]],
    *,
    split: str,
    start_index: int,
    images_dir: Path,
    known_ids: set[str],
    model: SentenceTransformer,
    batch_size: int,
) -> list[dict[str, object]]:
    """Turn one dataset batch into LanceDB rows, skipping what already exists.

    Images are always written if missing — even for rows already in the table —
    so a deleted ``data/images/`` can be repaired without re-embedding.

    Args:
        batch: Columnar slice from ``Dataset.iter``.
        split: Split being processed.
        start_index: Index of ``batch``'s first row within the split.
        images_dir: Destination for JPEG files.
        known_ids: Ids already in the table; mutated as new rows are staged.
        model: The loaded CLIP bi-encoder.
        batch_size: Images per forward pass.

    Returns:
        Rows to append, empty when the whole batch was already ingested.
    """
    raw_images = cast(list[dict[str, object]], batch["image"])

    pending_ids: list[str] = []
    pending_names: list[str] = []
    pending_offsets: list[int] = []
    decoded: list[PilImageType] = []

    for offset, raw_image in enumerate(raw_images):
        record_id, file_name = _record_identity(raw_image, split, start_index + offset)
        payload = raw_image.get("bytes")
        if not isinstance(payload, bytes):
            LOGGER.warning("Row %s/%d has no image bytes; skipping", split, start_index + offset)
            continue

        _write_image(images_dir / file_name, payload)

        if record_id in known_ids:
            continue

        # CLIP's preprocessor expects RGB; Flickr8k contains a few greyscale JPEGs.
        with PilImage.open(BytesIO(payload)) as handle:
            decoded.append(handle.convert("RGB"))
        pending_ids.append(record_id)
        pending_names.append(file_name)
        pending_offsets.append(offset)

    if not decoded:
        return []

    vectors = _encode_images(model, decoded, batch_size)

    rows: list[dict[str, object]] = []
    for position, record_id in enumerate(pending_ids):
        rows.append(
            {
                "id": record_id,
                "file_name": pending_names[position],
                "split": split,
                "captions": _extract_captions(batch, pending_offsets[position]),
                "vector": vectors[position].tolist(),
            }
        )
        known_ids.add(record_id)
    return rows


def ingest(args: argparse.Namespace) -> int:
    """Run the full ingestion pipeline.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The number of records newly written to the table.
    """
    data_dir: Path = args.data_dir
    images_dir = data_dir / "images"
    lancedb_dir = data_dir / "lancedb"
    hf_cache_dir = data_dir / "raw"
    for directory in (images_dir, lancedb_dir, hf_cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    LOGGER.info(
        "Device %r, torch %s, %d thread(s)",
        TORCH_DEVICE,
        torch.__version__,
        torch.get_num_threads(),
    )

    splits = _load_splits(hf_cache_dir, args.limit)
    total_rows = sum(dataset.num_rows for _, dataset in splits)
    if total_rows == 0:
        LOGGER.warning("Nothing to ingest")
        return 0
    LOGGER.info(
        "Loaded %d record(s) across %d split(s): %s",
        total_rows,
        len(splits),
        ", ".join(f"{name}={dataset.num_rows}" for name, dataset in splits),
    )

    LOGGER.info("Loading CLIP model %r (first run downloads weights) ...", CLIP_MODEL_ID)
    model = SentenceTransformer(CLIP_MODEL_ID, device=TORCH_DEVICE)

    db = lancedb.connect(lancedb_dir)
    table = _open_table(db, force=args.force)
    known_ids = _existing_ids(table)
    if known_ids:
        LOGGER.info("Resuming: %d record(s) already ingested", len(known_ids))

    written = 0
    with tqdm(total=total_rows, unit="img", desc="Embedding", file=sys.stderr) as progress:
        for split, dataset in splits:
            start_index = 0
            for batch in dataset.iter(batch_size=args.batch_size):
                typed_batch = cast("dict[str, list[object]]", batch)
                rows = _ingest_batch(
                    typed_batch,
                    split=split,
                    start_index=start_index,
                    images_dir=images_dir,
                    known_ids=known_ids,
                    model=model,
                    batch_size=args.batch_size,
                )
                if rows:
                    table.add(rows)
                    written += len(rows)
                batch_rows = len(typed_batch["image"])
                start_index += batch_rows
                progress.update(batch_rows)
                progress.set_postfix(split=split, written=written, refresh=False)

    LOGGER.info("Wrote %d new record(s); table now holds %d", written, table.count_rows())
    LOGGER.info("Images: %s", images_dir)
    LOGGER.info("LanceDB: %s (table %r)", lancedb_dir, TABLE_NAME)
    # Deliberately no ANN index. At ~8k rows a brute-force cosine scan over
    # 8k x 512 float32 (~16 MB) is single-digit milliseconds and exact, whereas
    # LanceDB's IVF_PQ defaults (256 partitions, 96 sub-vectors) would put ~31
    # rows in each partition and quantize the vectors — strictly worse recall
    # for no latency win. Revisit only if the corpus grows by orders of magnitude.
    LOGGER.info("No ANN index built: exact search is faster and lossless at this scale")
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        A process exit code: 0 on success, 1 on failure, 130 if interrupted.
    """
    args = _parse_args(argv)
    _configure_logging(verbose=args.verbose)
    try:
        ingest(args)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted — rerun the same command to resume")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
