#!/usr/bin/env python3
"""Write the current partition to a file a training script can read.

Collections reach disk today only as a CSV downloaded through the browser or a
ZIP of JPEGs. Nothing under ``scripts/`` knows they exist, and no training run
can be pointed at ``data/collections.db`` in any supported way — so the output
of re-partitioning a corpus is a file in ``~/Downloads``. This is the sanctioned
bridge out.

It joins the two stores the serving layer joins, using the same rule:

    every image's effective collection is its ``split``, unless an override says
    otherwise.

**Read-only with respect to both.** The LanceDB table is opened for reading, as
every script here does, and ``collections.db`` is opened with SQLite's
``mode=ro`` URI — verified against the engine bundled with this Python 3.12.13
(SQLite 3.53.1) to refuse writes at the engine rather than by convention. This
does not make ``scripts/`` a writer of the overlay any more than ``analyze.py``
makes it a writer of the index.

Two formats, and the difference is not cosmetic:

* ``json`` (default) carries the header block — every collection with its size
  and the provenance of its most recent batch — beside the per-image mapping.
  That provenance is what makes the partition *reproducible* rather than merely
  present: ``{"quality_flag": "cross-split-duplicate"}`` can be re-derived, a
  list of 32 ids cannot.
* ``csv`` is for pandas. Its rows carry ids only, never free text, so the same
  header block can ride above them as ``#`` comment lines and
  ``pd.read_csv(path, comment="#")`` cannot be broken by a ``#`` in someone's
  collection name.

Typical use::

    python scripts/export_split.py                      # data/splits.json
    python scripts/export_split.py --format csv         # data/splits.csv
    python scripts/export_split.py --force              # rewrite regardless
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import lancedb

LOGGER: Final = logging.getLogger("export_split")

TABLE_NAME: Final = "images"
COLLECTIONS_FILE_NAME: Final = "collections.db"

#: Output stem; the format supplies the extension.
OUTPUT_STEM: Final = "splits"

#: Prefix for the header lines above a CSV body. ``pandas.read_csv(comment="#")``
#: reads it, and the rows below carry no free text, so no value can contain it.
CSV_COMMENT: Final = "# "

#: Columns of the CSV body. Deliberately no ``collection_name``: a name is free
#: text and could hold the comment character, which would make the header
#: unreadable by the very reader it exists for. Names are in the header block,
#: joinable on ``collection_id``.
CSV_COLUMNS: Final = ("image_id", "split", "collection_id")


def _configure_logging(*, verbose: bool) -> None:
    """Send structured, timestamped logs to stderr so stdout stays clean."""
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
        description="Export the effective corpus partition (splits plus collection overrides).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="json keeps the provenance header as data; csv puts it in '#' comment lines.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write. Defaults to <data-dir>/splits.<format>.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite even when the existing file already describes this partition.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root of the local data directory.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def _read_corpus(lancedb_dir: Path) -> dict[str, str]:
    """Read every image's ground-truth split from the index.

    Args:
        lancedb_dir: Directory backing the embedded database.

    Returns:
        Image id to split name, in table scan order.

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

    table = db.open_table(TABLE_NAME).search().select(["id", "split"]).limit(None).to_arrow()
    ids = cast(list[str], table.column("id").to_pylist())
    splits = cast(list[str], table.column("split").to_pylist())
    return dict(zip(ids, splits, strict=True))


def _read_overlay(path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Read the collections and their overrides, without opening for writing.

    An absent store is not an error: it means nobody has created a collection
    yet, and the effective partition is exactly the dataset's own.

    Args:
        path: Location of the SQLite file.

    Returns:
        The collection rows (with the provenance of each one's most recent
        batch) and the image-to-collection overrides.
    """
    if not path.is_file():
        LOGGER.info("No collection store at %s — exporting the dataset splits alone", path)
        return [], {}

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        # The provenance columns are added by the API when it opens the store,
        # and this script is read-only — so it must work against a store the new
        # API has never touched rather than demanding one be started first.
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(image_collection)")}
        has_provenance = {"origin", "origin_detail"} <= columns
        if not has_provenance:
            LOGGER.info(
                "%s predates the provenance columns; exporting without them. Start the API "
                "once to migrate it.",
                path,
            )

        # Bare columns beside a single MAX() take their values from the row that
        # produced it — SQLite's documented min/max special case, and the same
        # query the API uses.
        provenance = (
            {
                row["collection_id"]: {
                    "origin": row["origin"],
                    "detail": row["origin_detail"],
                    "moved_at": row["moved_at"],
                }
                for row in connection.execute(
                    "SELECT collection_id, origin, origin_detail, MAX(moved_at) AS moved_at "
                    "FROM image_collection GROUP BY collection_id"
                )
            }
            if has_provenance
            else {}
        )
        collections = [
            {
                "id": row["id"],
                "name": row["name"],
                "kind": row["kind"],
                "provenance": provenance.get(row["id"]),
            }
            for row in connection.execute(
                "SELECT id, name, kind FROM collections ORDER BY kind, name COLLATE NOCASE"
            )
        ]
        overrides = {
            row["image_id"]: row["collection_id"]
            for row in connection.execute("SELECT image_id, collection_id FROM image_collection")
        }
    finally:
        connection.close()
    return collections, overrides


def _build_document(
    splits: dict[str, str], collections: list[dict[str, Any]], overrides: dict[str, str]
) -> dict[str, Any]:
    """Join the two stores into the artefact.

    Args:
        splits: Image id to ground-truth split.
        collections: Collection rows with their provenance.
        overrides: Image id to overridden collection id.

    Returns:
        A JSON-serialisable mapping, without its ``generated_at`` stamp — that
        is added last, so two runs over an unchanged partition can be compared.
    """
    images = {
        image_id: {"split": split, "collection": overrides.get(image_id, split)}
        for image_id, split in splits.items()
    }
    # Seeded with every split so one emptied by a move is still listed, at zero,
    # exactly as GET /api/collections reports it.
    sizes: dict[str, int] = {split: 0 for split in set(splits.values())}
    for entry in images.values():
        sizes[entry["collection"]] = sizes.get(entry["collection"], 0) + 1

    orphans = sum(1 for image_id in overrides if image_id not in splits)
    if orphans:
        LOGGER.warning(
            "%d override(s) point at images that are not in the index and were skipped; "
            "re-running ingestion with different ids leaves those behind",
            orphans,
        )

    return {
        "corpus_size": len(splits),
        "collections": [
            {**collection, "size": sizes.get(collection["id"], 0)} for collection in collections
        ],
        "images": images,
    }


def _render_json(document: dict[str, Any]) -> str:
    """Render the artefact as JSON, header block and all."""
    return json.dumps(document, indent=2, sort_keys=False) + "\n"


def _render_csv(document: dict[str, Any]) -> str:
    """Render the artefact as CSV with the header block in comment lines.

    The header is one compact JSON object per key rather than prose, so a reader
    that wants it can strip the prefix and parse it, and pandas can ignore the
    whole thing with ``comment="#"``.
    """
    buffer = io.StringIO()
    for key in ("generated_at", "corpus_size", "collections"):
        buffer.write(f"{CSV_COMMENT}{key}: {json.dumps(document[key], separators=(',', ':'))}\n")

    # `\n`, not csv's default `\r\n`: the comment lines above are written with
    # `\n`, and a file mixing the two reads back translated under Python's
    # universal newlines — which would make `_is_current` see a difference on
    # every run and rewrite a file that had not changed.
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for image_id, entry in cast(dict[str, dict[str, str]], document["images"]).items():
        writer.writerow((image_id, entry["split"], entry["collection"]))
    return buffer.getvalue()


def _is_current(output_path: Path, document: dict[str, Any], export_format: str) -> bool:
    """Whether the file on disk already describes this exact partition.

    Compared on everything but ``generated_at``, so re-running against an
    unchanged partition is a no-op rather than a file whose only change is its
    timestamp. That is what makes the script idempotent in the sense
    CLAUDE.md §4.2 asks for.

    Args:
        output_path: File that may already exist.
        document: The artefact just built, without its stamp.
        export_format: ``json`` or ``csv``.

    Returns:
        True when nothing would change.
    """
    if not output_path.is_file():
        return False
    try:
        existing = output_path.read_text(encoding="utf-8")
    except OSError:
        return False

    stamped = {"generated_at": "", **document}
    if export_format == "json":
        try:
            parsed: Any = json.loads(existing)
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, dict):
            return False
        return {**parsed, "generated_at": ""} == stamped
    return _render_csv(stamped) == _strip_generated_at(existing)


def _strip_generated_at(rendered: str) -> str:
    """Blank the ``generated_at`` comment line of a rendered CSV.

    Args:
        rendered: The file as read from disk.

    Returns:
        The same text with that one header value emptied, so the rest can be
        compared.
    """
    lines = rendered.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(f"{CSV_COMMENT}generated_at:"):
            lines[index] = f'{CSV_COMMENT}generated_at: ""\n'
            break
    return "".join(lines)


def _write_atomically(output_path: Path, rendered: str) -> None:
    """Write via a temporary sibling, then rename.

    A training script may be reading this file; a rename is atomic on the local
    filesystems this tool targets, so it never sees half of one.
    """
    temporary = output_path.with_name(f".{output_path.name}.partial")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output_path)


def export_split(args: argparse.Namespace) -> Path | None:
    """Build and write the partition artefact.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The path written, or ``None`` when an up-to-date file was kept.
    """
    data_dir: Path = args.data_dir
    output_path: Path = args.output or data_dir / f"{OUTPUT_STEM}.{args.format}"

    splits = _read_corpus(data_dir / "lancedb")
    collections, overrides = _read_overlay(data_dir / COLLECTIONS_FILE_NAME)
    LOGGER.info(
        "Read %d image(s), %d collection(s), %d override(s)",
        len(splits),
        len(collections),
        len(overrides),
    )

    document = _build_document(splits, collections, overrides)
    if not args.force and _is_current(output_path, document, args.format):
        LOGGER.info("%s already describes this partition — pass --force to rewrite", output_path)
        return None

    stamped = {"generated_at": datetime.now(UTC).isoformat(), **document}
    rendered = _render_json(stamped) if args.format == "json" else _render_csv(stamped)
    _write_atomically(output_path, rendered)
    LOGGER.info("Wrote %s (%d bytes)", output_path, len(rendered.encode("utf-8")))
    for collection in cast(list[dict[str, Any]], document["collections"]):
        LOGGER.debug("  %-24s %6d", collection["name"], collection["size"])
    return output_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        A process exit code: 0 on success, 1 on a missing index, 130 if
        interrupted.
    """
    args = _parse_args(argv)
    _configure_logging(verbose=args.verbose)
    try:
        export_split(args)
    except FileNotFoundError as error:
        LOGGER.error("%s", error)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted — rerun the same command")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
