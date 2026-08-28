#!/usr/bin/env python3
"""Project the CLIP embeddings down to two dimensions for the map view.

Reads every vector out of the LanceDB table ``scripts/ingest.py`` built and
writes ``data/projection.json``: one ``[x, y]`` pair per image id, plus the
metadata the UI needs to describe what it is showing.

**Why this is a separate script and not part of ingestion.** A projection is a
property of the corpus *as a whole* — every point's position depends on every
other point. Ingestion is incremental and resumable, so coordinates computed
during a partial run would be silently wrong the moment more rows were appended.
Keeping it separate also means re-projecting costs half a second instead of a
re-embed, and needs no change to the table schema.

**Two methods, and when each is right.**

``pca`` (default) is a linear projection onto the two directions of greatest
variance. It takes under a second, is deterministic, and — the part that matters
for honesty — reports how much of the variance those two directions actually
capture. On Flickr8k that is about 14%, which is *low*: the map shows broad
structure, not tight clusters. The UI prints the number for exactly this reason.

``tsne`` produces far more legible clusters by preserving local neighbourhoods:
measured on this corpus it keeps 32.1% of each image's true ten nearest
neighbours against PCA's 2.4%. It costs about 18 seconds — the 50-dimensional
pre-reduction below is what makes it that cheap — and it cannot report its own
distortion, which is the reason it is not the default. Cluster *sizes* and the
gaps between clusters still are not interpretable. Use it to *find* groups, and
PCA to reason about spread.

Typical use::

    python scripts/project.py                 # PCA, ~0.5 s
    python scripts/project.py --method tsne   # t-SNE, ~18 s on this CPU

``docker/projection.sh`` swaps the two in a running container, for a demo.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final, cast

import lancedb
import numpy as np
from numpy.typing import NDArray

LOGGER: Final = logging.getLogger("project")

TABLE_NAME: Final = "images"
OUTPUT_FILE_NAME: Final = "projection.json"

#: The map is two-dimensional. Named rather than inlined because it appears in
#: the SVD slice, the t-SNE constructor and the output metadata.
TARGET_DIMENSIONS: Final = 2

#: Decimal places kept in the output. Coordinates are normalised into [-1, 1],
#: so five places resolve about a hundred-thousandth of the plot width — far
#: below one screen pixel, and it keeps the file around a third of a megabyte.
COORDINATE_PRECISION: Final = 5

#: t-SNE is run on a PCA-reduced matrix rather than the raw 512 dimensions.
#: This is the standard preprocessing step: it removes most of the noise, cuts
#: the runtime by roughly an order of magnitude, and barely moves the result.
TSNE_PCA_DIMENSIONS: Final = 50
DEFAULT_PERPLEXITY: Final = 30.0
TSNE_MAX_ITERATIONS: Final = 1000
DEFAULT_SEED: Final = 0


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
        description="Project CLIP embeddings to 2-D for the map view.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=("pca", "tsne"),
        default="pca",
        help="pca: fast, linear, reports explained variance. "
        "tsne: slower, better-separated clusters, no global meaning.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when an up-to-date projection already exists.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=DEFAULT_PERPLEXITY,
        help="t-SNE neighbourhood size. Ignored by --method pca.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="t-SNE random seed. Currently inert: with init='pca' there is no "
        "randomness left to control, and every seed yields identical output. "
        "Kept because it is what would matter if init ever became 'random'.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root of the local data directory.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args(argv)
    if args.perplexity <= 0:
        parser.error("--perplexity must be positive")
    return args


def _read_embeddings(lancedb_dir: Path) -> tuple[list[str], NDArray[np.float32]]:
    """Load every id and vector from the index.

    Args:
        lancedb_dir: Directory backing the embedded database.

    Returns:
        Ids in scan order, and the matching ``(n, 512)`` matrix.

    Raises:
        FileNotFoundError: If the database or table is absent — i.e. ingestion
            has not been run against this ``data/`` directory.
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
    projection = table.search().select(["id", "vector"]).limit(None).to_arrow()
    ids = cast(list[str], projection.column("id").to_pylist())
    # `zero_copy_only=False` because a fixed-size-list column materialises as an
    # object array of per-row ndarrays, which cannot be viewed without a copy.
    vectors = projection.column("vector").to_numpy(zero_copy_only=False)
    matrix = np.stack(vectors).astype(np.float32, copy=False)
    return ids, matrix


def _principal_components(
    matrix: NDArray[np.float32], components: int
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Compute a PCA projection and the variance each component explains.

    Implemented directly on top of ``numpy.linalg.svd`` rather than through
    scikit-learn: it is four lines, it makes the centring and the sign
    convention visible instead of implied, and the explained-variance ratio —
    the number this whole view is honest about — falls straight out of the
    singular values.

    Args:
        matrix: ``(n, d)`` embeddings.
        components: How many components to keep.

    Returns:
        The ``(n, components)`` projection, and the fraction of total variance
        each kept component explains.
    """
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centred, full_matrices=False)

    basis = right_vectors[:components]
    # SVD fixes each component only up to a sign, and which sign LAPACK returns
    # can change with the input. Without a convention the map would mirror
    # itself between runs for no visible reason, so each component is oriented
    # to make its largest-magnitude loading positive.
    dominant = np.abs(basis).argmax(axis=1)
    signs = np.sign(basis[np.arange(components), dominant])
    signs[signs == 0] = 1.0
    basis = basis * signs[:, np.newaxis]

    variances = np.square(singular_values, dtype=np.float64)
    explained = variances[:components] / variances.sum()
    return (centred @ basis.T).astype(np.float32, copy=False), explained


def _tsne(matrix: NDArray[np.float32], perplexity: float, seed: int) -> NDArray[np.float32]:
    """Run Barnes-Hut t-SNE, after a linear pre-reduction.

    Verified against the installed scikit-learn 1.9.0: the iteration count is
    ``max_iter``. The old ``n_iter`` was renamed in 1.5 and removed in 1.7, so
    code written from memory of the older API fails outright here.

    Args:
        matrix: ``(n, d)`` embeddings.
        perplexity: Effective neighbourhood size.
        seed: Random seed, so the layout is reproducible.

    Returns:
        The ``(n, 2)`` embedding.
    """
    from sklearn.manifold import TSNE

    reduced_dimensions = min(TSNE_PCA_DIMENSIONS, matrix.shape[1], matrix.shape[0])
    reduced, _ = _principal_components(matrix, reduced_dimensions)
    LOGGER.info(
        "Reduced %d-d to %d-d before t-SNE; running with perplexity %.1f (this is the slow part)",
        matrix.shape[1],
        reduced_dimensions,
        perplexity,
    )

    model = TSNE(
        n_components=TARGET_DIMENSIONS,
        perplexity=perplexity,
        max_iter=TSNE_MAX_ITERATIONS,
        random_state=seed,
        init="pca",
    )
    embedded = cast(NDArray[np.float32], model.fit_transform(reduced))
    return embedded.astype(np.float32, copy=False)


def _normalise(coordinates: NDArray[np.float32]) -> NDArray[np.float32]:
    """Centre the cloud and scale it into roughly ``[-1, 1]``.

    A **single** scale factor is applied to both axes rather than one per axis.
    Stretching each axis to fill the range independently would make the plot
    look better and mean less: the relative spread of the two components is
    part of what the projection is saying.

    Args:
        coordinates: ``(n, 2)`` raw projection.

    Returns:
        The same points, centred on the origin and scaled to fit the unit box.
    """
    # Annotated because numpy's stubs type `.mean()` as Any, which would make
    # the whole expression — and this function's return — untyped.
    centred: NDArray[np.float32] = coordinates - coordinates.mean(axis=0, keepdims=True)
    extent = float(np.abs(centred).max())
    if extent == 0.0:  # pragma: no cover - degenerate, e.g. a one-row table
        return centred
    scaled: NDArray[np.float32] = centred / extent
    return scaled


def _build_document(
    ids: list[str],
    coordinates: NDArray[np.float32],
    *,
    method: str,
    explained: NDArray[np.float64] | None,
) -> dict[str, object]:
    """Assemble the JSON document the API will serve.

    Args:
        ids: Image ids, aligned with ``coordinates``.
        coordinates: Normalised ``(n, 2)`` positions.
        method: Which projection produced them.
        explained: Per-component variance ratios, for PCA only.

    Returns:
        A JSON-serialisable mapping.
    """
    points = {
        image_id: [
            round(float(coordinates[index, 0]), COORDINATE_PRECISION),
            round(float(coordinates[index, 1]), COORDINATE_PRECISION),
        ]
        for index, image_id in enumerate(ids)
    }
    document: dict[str, object] = {
        "method": method,
        "dimensions": TARGET_DIMENSIONS,
        "count": len(ids),
        "points": points,
    }
    if explained is not None:
        document["explained_variance_ratio"] = [
            round(float(value), 6) for value in explained[:TARGET_DIMENSIONS]
        ]
    return document


def _is_up_to_date(output_path: Path, *, method: str, row_count: int) -> bool:
    """Whether an existing projection already describes this corpus.

    Checks the method and the row count, which is what changes when ingestion
    adds images. A corpus edited in place without changing its size would defeat
    this — hence ``--force``, and hence ``docker/setup.sh`` always passing it
    after an ingest.

    Args:
        output_path: Candidate existing file.
        method: Method that would be run now.
        row_count: Rows currently in the table.

    Returns:
        True when recomputing would produce the same thing.
    """
    if not output_path.is_file():
        return False
    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Existing %s is unreadable; recomputing", output_path.name)
        return False
    return bool(existing.get("method") == method and existing.get("count") == row_count)


def _write_atomically(output_path: Path, document: dict[str, object]) -> None:
    """Write the document via a temporary sibling, then rename.

    The API reads this file at startup. A half-written file would be a parse
    error at exactly the wrong moment, and a rename is atomic on the local
    filesystems this tool targets.
    """
    temporary = output_path.with_name(f".{output_path.name}.partial")
    temporary.write_text(json.dumps(document), encoding="utf-8")
    temporary.replace(output_path)


def project(args: argparse.Namespace) -> int:
    """Run the projection and write the result.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The number of points written; 0 when an up-to-date file was kept.
    """
    data_dir: Path = args.data_dir
    output_path = data_dir / OUTPUT_FILE_NAME

    ids, matrix = _read_embeddings(data_dir / "lancedb")
    LOGGER.info("Read %d vector(s) of %d dimension(s)", matrix.shape[0], matrix.shape[1])

    if not args.force and _is_up_to_date(output_path, method=args.method, row_count=len(ids)):
        LOGGER.info(
            "%s is already current for %d rows; use --force to recompute", output_path, len(ids)
        )
        return 0

    explained: NDArray[np.float64] | None = None
    if args.method == "pca":
        raw, explained = _principal_components(matrix, TARGET_DIMENSIONS)
        LOGGER.info(
            "PCA explained variance: %s (total %.1f%%)",
            ", ".join(f"{value:.2%}" for value in explained),
            100.0 * float(explained.sum()),
        )
        LOGGER.info(
            "Two components of a CLIP space capture little of it — the map shows "
            "broad structure, not clean clusters. Try --method tsne for separation."
        )
    else:
        raw = _tsne(matrix, args.perplexity, args.seed)

    document = _build_document(ids, _normalise(raw), method=args.method, explained=explained)
    _write_atomically(output_path, document)
    LOGGER.info("Wrote %d point(s) to %s", len(ids), output_path)
    return len(ids)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        A process exit code: 0 on success, 1 on a missing index, 130 if
        interrupted.
    """
    args = _parse_args(argv)
    _configure_logging(verbose=args.verbose)
    try:
        project(args)
    except FileNotFoundError as error:
        LOGGER.error("%s", error)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted — rerun the same command")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
