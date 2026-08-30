#!/usr/bin/env python3
"""One-time, offline ingestion of the Flickr8k dataset.

Pulls ``jxie/flickr8k`` through Hugging Face ``datasets``, writes each image's
original JPEG bytes to ``data/images/`` (so FastAPI can serve them statically
later), encodes every image with CLIP ``clip-ViT-B-32`` on whichever device is
available, and stores the resulting 512-d vectors alongside their five reference
captions in an embedded LanceDB table under ``data/lancedb/``.

This script is the *only* place CLIP image inference happens. Measured on the
reference machine (Core i9-9980HK, CPU, 8 torch threads) throughput is
~10 images/s, so a full pass over the ~8k corpus takes roughly 15 minutes — far
too slow to sit inside a request handler, which is why the serving API never
embeds images per request and only queries the index this script produces
(CLAUDE.md §2). A CUDA or MPS device shortens the wall clock; it does not change
that division of labour.

Memory is bounded by ``--encode-batch-size``, not by how much is read at a time:
decoding streams in encode-sized chunks, so pointing this at a corpus of large
photographs does not put the whole read batch in RAM. See :func:`_ingest_batch`.

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

HF_DATASET_ID: Final = os.environ.get("CORPUSLENS_HF_DATASET_ID", "jxie/flickr8k")

#: The checkpoint the whole corpus is embedded with, and therefore the one the
#: API must encode queries with. It reads the same environment variable through
#: ``Settings.clip_model_id``; a divergence yields a shared space that is not
#: shared, and search degrades to noise instead of failing. Read from the
#: process environment, not from ``.env``: this script stays standalone
#: (CLAUDE.md §4.2) and does not import the backend package, so exporting the
#: variable is what propagates it to both sides.
CLIP_MODEL_ID: Final = os.environ.get("CORPUSLENS_CLIP_MODEL_ID", "clip-ViT-B-32")

#: CLIP ViT-B/32 projects images and text into a shared 512-d space. Pinned to
#: the schema rather than derived, so a model whose width differs fails at the
#: first insert rather than writing a table the API cannot query.
EMBEDDING_DIM: Final = 512

#: Flickr8k's canonical splits, in the order they are ingested.
SPLITS: Final = ("train", "validation", "test")

#: The dataset stores the five references as flat columns, not as a list.
CAPTION_COLUMNS: Final = tuple(f"caption_{index}" for index in range(5))

TABLE_NAME: Final = "images"

#: Images per CLIP forward pass. Also the ceiling on how many decoded images are
#: resident at once — the two were one number until it turned out they should
#: not be; see :func:`_ingest_batch`.
DEFAULT_ENCODE_BATCH_SIZE: Final = 32

#: Rows per Arrow read from the dataset. Larger than the encode batch because a
#: bigger read amortizes parquet overhead, and — now that decoding streams — it
#: no longer costs memory to raise.
DEFAULT_READ_BATCH_SIZE: Final = 256

#: Longest edge to decode JPEGs at. CLIP ViT-B/32 resizes to 224x224 regardless,
#: so decoding a 4000px original at full size is work done only to throw away.
#: Requesting a little above the model's input keeps the final resample's
#: quality; see :func:`_decode_image`.
DECODE_TARGET_PIXELS: Final = 448


def _resolve_device() -> str:
    """Pick the torch device for the embedding pass.

    The batch-workload counterpart to ``app.services.embedding.resolve_device``.
    Duplicated rather than imported because the offline scripts are standalone
    by contract (CLAUDE.md §4.2) and do not depend on the backend package —
    keep the two in step if the detection order ever changes.

    **This one always takes MPS when it is usable, and the serving path does
    not.** That is deliberate, not drift. Measured on the reference machine (an
    Intel Mac whose AMD GPU supports MPS), a 64-image batch runs at 81 img/s on
    MPS against 37 img/s on CPU — a 2.2x win, because a batch amortizes the
    host-to-device transfer over enough arithmetic to pay for it. The same
    transfer makes a single short text encode 3.7x *slower* on MPS, which is
    why the API reaches the opposite conclusion for its workload. See
    ``resolve_device`` there for the full table.

    Returns:
        ``"cuda"``, ``"mps"`` or ``"cpu"``, degrading rather than raising when
        an accelerator is present but unusable.
    """
    if torch.cuda.is_available():
        return "cuda"
    # `is_built()` distinguishes "this wheel has no MPS backend" from "no MPS
    # hardware"; `is_available()` alone conflates them on some builds.
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


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
        "--encode-batch-size",
        type=int,
        default=DEFAULT_ENCODE_BATCH_SIZE,
        help="Images per CLIP forward pass, and the most decoded images held at once.",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=DEFAULT_READ_BATCH_SIZE,
        help="Rows read from the dataset per iteration. Does not affect memory.",
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
    if args.encode_batch_size < 1:
        parser.error("--encode-batch-size must be a positive integer")
    if args.read_batch_size < 1:
        parser.error("--read-batch-size must be a positive integer")
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


def _decode_image(payload: bytes) -> PilImageType:
    """Decode one image to RGB, no larger than the encoder needs.

    ``draft()`` asks libjpeg to decode at 1/2, 1/4 or 1/8 scale directly,
    skipping DCT coefficients rather than computing them and discarding the
    result. Since CLIP resizes to 224x224 anyway, decoding a 4000px original at
    full resolution is pure waste — of time, and of the memory that made a large
    batch dangerous. It is a no-op for formats that do not support it and for
    images already below the target, so it costs nothing on the reference
    corpus, whose images are ~500x375.

    Args:
        payload: Encoded image bytes.

    Returns:
        The decoded image in RGB. Flickr8k contains a few greyscale JPEGs and
        CLIP's preprocessor expects three channels, so the conversion is not
        optional.
    """
    with PilImage.open(BytesIO(payload)) as handle:
        handle.draft("RGB", (DECODE_TARGET_PIXELS, DECODE_TARGET_PIXELS))
        return handle.convert("RGB")


def _encode_images(
    model: SentenceTransformer,
    images: list[PilImageType],
    batch_size: int,
    device: str,
) -> NDArray[np.float32]:
    """Embed images into the shared CLIP space.

    ``normalize_embeddings=True`` is essential and *not* the default: LanceDB's
    cosine metric — and the dot-product shortcut the search service will use —
    are only correct on unit-length vectors.

    Args:
        model: The loaded CLIP bi-encoder.
        images: Decoded RGB images.
        batch_size: Images per forward pass.
        device: Resolved torch device.

    Returns:
        An ``(len(images), EMBEDDING_DIM)`` float32 array of unit vectors.
    """
    embeddings = model.encode(
        images,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device=device,
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
    encode_batch_size: int,
    device: str,
) -> list[dict[str, object]]:
    """Turn one dataset batch into LanceDB rows, skipping what already exists.

    Images are always written if missing — even for rows already in the table —
    so a deleted ``data/images/`` can be repaired without re-embedding.

    **Decoding streams in encode-sized chunks.** This function used to decode
    every image in the read batch before encoding any of them, which tied peak
    memory to the read size. That was harmless on the reference corpus, whose
    images are ~500x375, and dangerous on anything larger: a 256-row read of
    4000x3000 photographs is ~8.8 GB of decoded RGB resident at once, and the
    failure mode is an OOM kill partway through a job measured in tens of
    minutes. Decoding a chunk, encoding it, then dropping it caps residency at
    ``encode_batch_size`` images regardless of how much is read at a time.

    Args:
        batch: Columnar slice from ``Dataset.iter``.
        split: Split being processed.
        start_index: Index of ``batch``'s first row within the split.
        images_dir: Destination for JPEG files.
        known_ids: Ids already in the table; mutated as new rows are staged.
        model: The loaded CLIP bi-encoder.
        encode_batch_size: Images per forward pass, and the cap on decoded
            images held simultaneously.
        device: Resolved torch device.

    Returns:
        Rows to append, empty when the whole batch was already ingested.
    """
    raw_images = cast(list[dict[str, object]], batch["image"])

    # Identify and persist first, decode later: writing every file even for rows
    # already in the table is what lets a deleted data/images/ be repaired
    # without re-embedding, and it costs nothing to separate from the CLIP pass.
    pending: list[tuple[str, str, int, bytes]] = []
    for offset, raw_image in enumerate(raw_images):
        record_id, file_name = _record_identity(raw_image, split, start_index + offset)
        payload = raw_image.get("bytes")
        if not isinstance(payload, bytes):
            LOGGER.warning("Row %s/%d has no image bytes; skipping", split, start_index + offset)
            continue

        _write_image(images_dir / file_name, payload)

        if record_id in known_ids:
            continue
        pending.append((record_id, file_name, offset, payload))

    rows: list[dict[str, object]] = []
    for chunk_start in range(0, len(pending), encode_batch_size):
        chunk = pending[chunk_start : chunk_start + encode_batch_size]
        decoded = [_decode_image(payload) for *_, payload in chunk]
        vectors = _encode_images(model, decoded, encode_batch_size, device)

        for position, (record_id, file_name, offset, _) in enumerate(chunk):
            rows.append(
                {
                    "id": record_id,
                    "file_name": file_name,
                    "split": split,
                    "captions": _extract_captions(batch, offset),
                    "vector": vectors[position].tolist(),
                }
            )
            known_ids.add(record_id)

        # Drop the decoded chunk before decoding the next one. Without this the
        # list stays referenced for the whole loop and the cap above is fiction.
        decoded.clear()

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

    # Unlike the serving process, this one wants every core it can get: it runs
    # a single long batch job with no concurrent requests to trade against, so
    # torch's own default of one thread per core is right here.
    device = _resolve_device()
    LOGGER.info(
        "Device %r, torch %s, %d thread(s)",
        device,
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
    model = SentenceTransformer(CLIP_MODEL_ID, device=device)

    db = lancedb.connect(lancedb_dir)
    table = _open_table(db, force=args.force)
    known_ids = _existing_ids(table)
    if known_ids:
        LOGGER.info("Resuming: %d record(s) already ingested", len(known_ids))

    written = 0
    with tqdm(total=total_rows, unit="img", desc="Embedding", file=sys.stderr) as progress:
        for split, dataset in splits:
            start_index = 0
            for batch in dataset.iter(batch_size=args.read_batch_size):
                typed_batch = cast("dict[str, list[object]]", batch)
                rows = _ingest_batch(
                    typed_batch,
                    split=split,
                    start_index=start_index,
                    images_dir=images_dir,
                    known_ids=known_ids,
                    model=model,
                    encode_batch_size=args.encode_batch_size,
                    device=device,
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
    # Deliberately no ANN index here — but the earlier reasoning in this comment
    # was wrong about *why*, and the corrected version is worth recording.
    #
    # Measured on the reference machine at 8 000 rows: IVF-PQ is genuinely
    # faster than the exact scan (6.0 ms against 22.5 ms), not slower as this
    # comment used to claim. What it does is lose a third of the true
    # neighbours — recall@20 of 0.695. `nprobes` cannot fix that, because the
    # loss is quantization rather than partition pruning; sweeping it from 1 to
    # 256 moved recall by 0.008. Only `refine_factor`, which re-ranks against
    # the full-precision vectors, does — and at the setting that restores
    # honest recall (0.997) the query costs 20.0 ms, which is the exact scan
    # again to within noise.
    #
    # So the conclusion stands and the corpus ships unindexed: at this size an
    # index buys nothing. It stops being true somewhere around 50k rows, and
    # `scripts/build_index.py` is where that decision now lives, with the
    # crossover measurements in its module docstring.
    LOGGER.info(
        "No ANN index built: at %d rows an exact scan matches a recall-corrected "
        "index for latency and is lossless. Run `python scripts/build_index.py` "
        "if this corpus grows past ~50k rows.",
        table.count_rows(),
    )
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
