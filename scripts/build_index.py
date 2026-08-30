#!/usr/bin/env python3
"""Build the optional ANN index over an already-ingested corpus.

Separate from ``scripts/ingest.py`` on purpose. Ingestion is the expensive,
resumable pass that produces the corpus; indexing is a cheap, idempotent
decision *about* that corpus which a user may want to revisit — after growing
the dataset, or after deciding the recall trade is worth it. Keeping them apart
means neither forces the other to be re-run.

It lives in ``scripts/`` rather than behind an endpoint because the API is a
pure reader of the index (CLAUDE.md §4.2). A background task that mutated the
table would break the property that makes ``split`` trustworthy ground truth
for the leakage analysis, and would put a multi-minute write on the same file
handle a request is reading through.

**Whether to index at all is a real question, and the answer is usually no.**
Measured on the reference machine (Core i9-9980HK, CPU):

===========  ==========================  ===========  ==========
rows         configuration               recall@20    latency
===========  ==========================  ===========  ==========
8 000        exact scan                  1.000         22.5 ms
8 000        IVF-PQ, no refine           0.695          6.0 ms
8 000        IVF-PQ refine_factor=10     0.997         20.0 ms
200 000      exact scan                  1.000        204 ms
200 000      IVF-PQ refine_factor=10     1.000          6.6 ms
===========  ==========================  ===========  ==========

At the reference corpus size the index is a wash: tuned back to honest recall it
lands within 2 ms of the exact scan. At 200k it is a 31x win at no measured
recall cost. That crossover is what :data:`MIN_WORTHWHILE_ROWS` encodes.

Typical use::

    python scripts/build_index.py              # build if the corpus is large enough
    python scripts/build_index.py --status     # report, change nothing
    python scripts/build_index.py --force      # build regardless of size
    python scripts/build_index.py --drop       # return to exact search
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Final

import lancedb
from lancedb.table import Table

LOGGER: Final = logging.getLogger("build_index")

TABLE_NAME: Final = "images"

#: The column ``scripts/ingest.py`` writes embeddings to.
VECTOR_COLUMN: Final = "vector"

#: Index name LanceDB derives from the column, and therefore what
#: ``index_stats`` and ``drop_index`` must be given.
INDEX_NAME: Final = f"{VECTOR_COLUMN}_idx"

#: Product quantization trains one centroid per code at ``num_bits=8``, so it
#: needs at least 2**8 rows to fit. Verified against lancedb 0.25.3, which
#: raises "Not enough rows to train PQ. Requires 256 rows but only N available"
#: below this. A corpus this small scans exactly in about a millisecond, so the
#: floor costs nothing.
MIN_INDEXABLE_ROWS: Final = 256

#: Below this an exact scan stays comfortably interactive (~1 ms per 1 000 rows
#: measured, so ~50 ms here) and is lossless, which beats any approximate index.
#: See the table in the module docstring for the measurements behind the choice.
MIN_WORTHWHILE_ROWS: Final = 50_000

#: Rows per IVF partition to aim for. LanceDB sizes the partition count and the
#: PQ sub-vector count from this, which is what keeps the index sensible across
#: three orders of magnitude of corpus size instead of tuned for one. Verified
#: present on ``create_index`` in lancedb 0.25.3.
TARGET_PARTITION_SIZE: Final = 1_000

#: Cosine, matching what the repository queries with and what ingestion
#: normalized for. A metric mismatch between index and query does not raise —
#: it silently ranks by the wrong geometry.
METRIC: Final = "cosine"


def _configure_logging(*, verbose: bool) -> None:
    """Send structured, timestamped logs to stderr."""
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
        description="Build or drop the ANN index over the ingested corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build even when the corpus is below the size where an index helps.",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Remove the index and return to exact search.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report what exists and what would happen, then exit without writing.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root of the local data directory.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args(argv)
    if args.drop and args.force:
        parser.error("--drop and --force contradict each other")
    return args


def _open_table(data_dir: Path) -> Table:
    """Open the ingested images table.

    Args:
        data_dir: Root of the local data directory.

    Returns:
        The open table.

    Raises:
        SystemExit: If ingestion has not been run against this directory. This
            is an operator error with a known fix, so it exits with the command
            to run rather than raising a traceback at them.
    """
    lancedb_dir = data_dir / "lancedb"
    if not lancedb_dir.is_dir():
        raise SystemExit(
            f"No LanceDB directory at {lancedb_dir}. Run `python scripts/ingest.py` first."
        )

    db = lancedb.connect(lancedb_dir)
    if TABLE_NAME not in db.table_names():
        raise SystemExit(
            f"Table {TABLE_NAME!r} not found in {lancedb_dir}. "
            "Run `python scripts/ingest.py` first."
        )
    return db.open_table(TABLE_NAME)


def _existing_index(table: Table) -> Any | None:  # noqa: ANN401 - lancedb ships no py.typed
    """Return the vector index on the table, if one has been built.

    Args:
        table: The open images table.

    Returns:
        The index descriptor, or ``None`` when the corpus is served by exact
        search.
    """
    indices = list(table.list_indices())
    return indices[0] if indices else None


def _report_status(table: Table) -> None:
    """Print what exists and what this script would do about it.

    Args:
        table: The open images table.
    """
    rows = table.count_rows()
    index = _existing_index(table)

    print(f"Corpus:  {rows:,} rows")
    if index is None:
        print("Index:   none — every search is an exact scan")
    else:
        print(f"Index:   {index}")
        print(f"Stats:   {table.index_stats(INDEX_NAME)}")

    if rows < MIN_INDEXABLE_ROWS:
        print(f"Verdict: cannot index below {MIN_INDEXABLE_ROWS:,} rows (PQ training floor)")
    elif rows < MIN_WORTHWHILE_ROWS:
        print(
            f"Verdict: exact search is the better choice below {MIN_WORTHWHILE_ROWS:,} rows; "
            "use --force to index anyway"
        )
    else:
        print("Verdict: large enough that an index is worth building")


def build_index(table: Table, *, force: bool) -> bool:
    """Build the IVF-PQ index if the corpus is large enough to benefit.

    Args:
        table: The ingested images table.
        force: Build even when the corpus is below :data:`MIN_WORTHWHILE_ROWS`.
            The hard PQ training floor is never overridden — below it there is
            no index to build, not merely an unwise one.

    Returns:
        True if an index was built, False if the corpus was left to exact
        scanning. Not an error either way: no index is the correct
        configuration at the reference corpus size.
    """
    rows = table.count_rows()

    if rows < MIN_INDEXABLE_ROWS:
        LOGGER.info(
            "%d rows is below the %d-row PQ training floor. Exact search is the only "
            "option and is near-instant at this size — nothing to do.",
            rows,
            MIN_INDEXABLE_ROWS,
        )
        return False

    if rows < MIN_WORTHWHILE_ROWS and not force:
        LOGGER.info(
            "%d rows scan exactly in well under 50 ms. An index here would quantize "
            "vectors for a latency win nobody can perceive — skipping. Use --force to "
            "build anyway.",
            rows,
        )
        return False

    LOGGER.info("Building IVF-PQ over %d rows (roughly 40 s per 200k on CPU) ...", rows)
    table.create_index(
        metric=METRIC,
        index_type="IVF_PQ",
        vector_column_name=VECTOR_COLUMN,
        target_partition_size=TARGET_PARTITION_SIZE,
        # Rebuild in place rather than erroring, so re-running after ingesting
        # more images is the obvious command and not a drop-then-build dance.
        replace=True,
    )
    LOGGER.info("Index built: %s", table.index_stats(INDEX_NAME))
    LOGGER.info(
        "Searches are now approximate when unfiltered. The API re-ranks with "
        "refine_factor and falls back to an exact scan under a selective filter; "
        "see CLAUDE.md §4.4."
    )
    return True


def drop_index(table: Table) -> bool:
    """Remove the vector index, returning the corpus to exact search.

    Args:
        table: The ingested images table.

    Returns:
        True if an index was removed, False if there was none.
    """
    if _existing_index(table) is None:
        LOGGER.info("No index to drop; searches are already exact")
        return False

    table.drop_index(INDEX_NAME)
    LOGGER.info("Dropped %r — every search is now an exact scan", INDEX_NAME)
    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        A process exit code: 0 on success, 130 if interrupted.
    """
    args = _parse_args(argv)
    _configure_logging(verbose=args.verbose)

    try:
        table = _open_table(args.data_dir)
        if args.status:
            _report_status(table)
        elif args.drop:
            drop_index(table)
        else:
            build_index(table, force=args.force)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted — the table is unchanged; rerun to try again")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
