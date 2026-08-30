"""Recall guarantees for vector search, against a real LanceDB table.

The rest of the suite runs against the doubles in ``conftest.py``, which is the
right trade nearly everywhere: they are fast, and what they exercise is this
application's logic. They cannot defend the claim in CLAUDE.md §4.4, though,
because that claim is about *LanceDB's* behaviour — how an IVF-PQ index scores,
and how it behaves under a pre-filter. A double that reimplemented that would
only ever confirm what it was written to believe.

So this module builds a genuine index over a synthetic corpus and measures. It
is the slowest module in the suite (a few seconds, dominated by index
construction) and it is the only one that would catch an upstream change to
LanceDB's defaults silently degrading search quality — the failure mode that
otherwise ships green.

**Scope, stated honestly.** The fixture corpus is 2 000 rows, which at the
script's ``target_partition_size`` yields very few IVF partitions, so ``nprobes``
is effectively exhaustive here and partition pruning is *not* what is under
test. What is under test is the part that broke in review: PQ distortion, the
``refine_factor`` that corrects it, and the pre-filter bypass. The pruning
numbers behind §4.4 came from a 200 000-row corpus that has no business in a
test suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import lancedb
import numpy as np
import pyarrow as pa
import pytest
from numpy.typing import NDArray

from app.models.domain import ImageFilter
from app.repositories.image_repository import LanceDBImageRepository
from scripts.build_index import build_index

#: Enough rows to clear the 256-row PQ training floor with room to spare, while
#: keeping index construction inside a couple of seconds.
_ROWS: Final = 2_000
_DIMENSIONS: Final = 512
_CLUSTERS: Final = 40
_TOP_K: Final = 20

#: How far a point sits from its centroid, as a fraction of the centroid's own
#: unit length. Comfortably below 1 so cluster membership dominates position —
#: the property real embeddings have and that IVF depends on. See
#: :func:`_clustered_corpus` for why this is a norm and not a per-dimension
#: sigma.
_JITTER_NORM: Final = 0.35

#: Floor for unfiltered ANN recall. The measured value with ``refine_factor=10``
#: is 1.000 on this fixture; the gap is slack for k-means initialisation, which
#: is seeded by LanceDB rather than by us. A regression that matters — losing
#: the refine pass, say — lands at ~0.7 and trips this comfortably.
_MIN_ANN_RECALL: Final = 0.90


def _clustered_corpus(seed: int = 0) -> NDArray[np.float32]:
    """Generate unit vectors with genuine cluster structure.

    Structure is the whole point. Uniformly random directions in 512 dimensions
    are near-equidistant, which makes the "true" top-k an arbitrary choice among
    ties and drives measured recall toward zero for *any* index — an artefact of
    the fixture, not a finding about the code. Sampling tightly around a set of
    centroids reproduces the property real CLIP embeddings have and that IVF
    relies on.

    The jitter scale is chosen in terms of its *norm*, not per-dimension, and
    that distinction is the whole fixture. Per-dimension noise accumulates as
    ``sigma * sqrt(dimensions)``, so an innocuous-looking sigma of 0.15 across
    512 dimensions displaces each point by ~3.4 — more than three times the
    unit-length centroid it was meant to sit near. The clusters dissolve, every
    point becomes near-equidistant from every other, and measured recall
    collapses for reasons that have nothing to do with the code under test.

    Args:
        seed: Seed for reproducibility across runs.

    Returns:
        An ``(_ROWS, _DIMENSIONS)`` array of unit-length float32 vectors.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((_CLUSTERS, _DIMENSIONS)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    assignments = rng.integers(0, _CLUSTERS, size=_ROWS)
    raw_jitter = rng.standard_normal((_ROWS, _DIMENSIONS)).astype(np.float32)
    raw_jitter /= np.linalg.norm(raw_jitter, axis=1, keepdims=True)

    vectors = centroids[assignments] + raw_jitter * _JITTER_NORM
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return cast("NDArray[np.float32]", vectors.astype(np.float32, copy=False))


def _write_table(lancedb_dir: Path, vectors: NDArray[np.float32]) -> None:
    """Create an images table matching the ingestion schema.

    Only the columns this module reads are populated; the repository projects
    by name, so the absence of the rest is invisible to it.

    Args:
        lancedb_dir: Directory to create the embedded database in.
        vectors: One row per image, unit length.
    """
    rows = vectors.shape[0]
    # A two-way split with the minority class small enough that filtering to it
    # is genuinely selective — which is the condition the bypass exists for.
    splits = ["train"] * int(rows * 0.8) + ["test"] * (rows - int(rows * 0.8))

    table = pa.table(
        {
            "id": pa.array([f"img-{index:05d}" for index in range(rows)]),
            "file_name": pa.array([f"img-{index:05d}.jpg" for index in range(rows)]),
            "split": pa.array(splits),
            "captions": pa.array([[f"caption for image {index}"] for index in range(rows)]),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1), type=pa.float32()), _DIMENSIONS
            ),
        }
    )
    lancedb.connect(lancedb_dir).create_table("images", table)


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A LanceDB directory holding an unindexed synthetic corpus.

    Module-scoped: building it twice would double the slowest part of the
    suite, and every test here treats it as read-only.
    """
    lancedb_dir = tmp_path_factory.mktemp("recall") / "lancedb"
    _write_table(lancedb_dir, _clustered_corpus())
    return lancedb_dir


@pytest.fixture(scope="module")
def indexed_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same corpus, with an IVF-PQ index built over it.

    A separate directory rather than an index added to :func:`corpus_dir`, so
    the indexed and exact repositories can be compared within one test.
    ``force=True`` because 2 000 rows is far below the size at which the script
    would choose to index — which is itself the behaviour asserted elsewhere.
    """
    lancedb_dir = tmp_path_factory.mktemp("recall-indexed") / "lancedb"
    _write_table(lancedb_dir, _clustered_corpus())

    table = lancedb.connect(lancedb_dir).open_table("images")
    assert build_index(table, force=True), "fixture corpus should be indexable"
    return lancedb_dir


def _recall(actual: list[str], expected: list[str]) -> float:
    """Fraction of the exact top-k that an approximate ranking recovered."""
    return len(set(actual) & set(expected)) / len(expected)


def test_index_presence_is_detected(corpus_dir: Path, indexed_dir: Path) -> None:
    """The repository routes on what the table actually has, not on config."""
    assert LanceDBImageRepository.open(corpus_dir, "images")._has_ann_index is False
    assert LanceDBImageRepository.open(indexed_dir, "images")._has_ann_index is True


def test_unindexed_search_is_exact(corpus_dir: Path) -> None:
    """Without an index every query is a full scan, so ranking is ground truth.

    Establishes that the baseline the other tests compare against is itself
    trustworthy rather than merely self-consistent.
    """
    repository = LanceDBImageRepository.open(corpus_dir, "images")
    query = repository.get_vector_by_id("img-00000")
    assert query is not None

    first = [hit.image.id for hit in repository.search_by_vector(query, _TOP_K)]
    second = [hit.image.id for hit in repository.search_by_vector(query, _TOP_K)]
    assert first == second
    # The query vector is in the corpus, so it must rank itself first.
    assert first[0] == "img-00000"


def test_unfiltered_ann_recall_stays_above_floor(corpus_dir: Path, indexed_dir: Path) -> None:
    """The refine pass keeps approximate search honest when unfiltered.

    This is the test that fails if ``refine_factor`` is dropped, defaulted away,
    or stops being applied — the measured recall without it is ~0.70.
    """
    exact = LanceDBImageRepository.open(corpus_dir, "images")
    approximate = LanceDBImageRepository.open(indexed_dir, "images")

    recalls: list[float] = []
    for index in range(0, 200, 20):
        query = exact.get_vector_by_id(f"img-{index:05d}")
        assert query is not None
        truth = [hit.image.id for hit in exact.search_by_vector(query, _TOP_K)]
        got = [hit.image.id for hit in approximate.search_by_vector(query, _TOP_K)]
        recalls.append(_recall(got, truth))

    mean_recall = float(np.mean(recalls))
    assert mean_recall >= _MIN_ANN_RECALL, (
        f"unfiltered ANN recall {mean_recall:.3f} fell below {_MIN_ANN_RECALL}; "
        "the refine pass is the usual cause"
    )


def test_selective_filter_bypasses_the_index_and_stays_exact(
    corpus_dir: Path, indexed_dir: Path
) -> None:
    """A filtered query on an indexed table returns the *exact* ranking.

    The regression this exists for is subtle and silent: an IVF pre-filter
    applies within probed partitions, so a selective filter starves the
    candidate pool and returns a full page of plausible-but-wrong results.
    Measured at 0.71 recall on a 200k corpus before the bypass existed.

    Asserting equality rather than a recall floor is deliberate — under the
    bypass the two paths run the same scan, so anything less than identical
    means the routing did not fire.
    """
    exact = LanceDBImageRepository.open(corpus_dir, "images")
    approximate = LanceDBImageRepository.open(indexed_dir, "images")
    image_filter = ImageFilter(splits=("test",))

    for index in range(0, 200, 20):
        query = exact.get_vector_by_id(f"img-{index:05d}")
        assert query is not None
        truth = [hit.image.id for hit in exact.search_by_vector(query, _TOP_K, image_filter)]
        got = [hit.image.id for hit in approximate.search_by_vector(query, _TOP_K, image_filter)]
        assert got == truth, "a filtered query on an indexed table must not be approximate"


def test_filter_above_the_ceiling_takes_the_ann_path(indexed_dir: Path) -> None:
    """The bypass is bounded, so a loose filter still gets the index.

    Guards the other direction: a ceiling high enough to swallow every query
    would quietly turn the index off. Set it to zero and the same filtered query
    must now route to ANN, which is observable because the two paths disagree.
    """
    routed_to_ann = LanceDBImageRepository.open(indexed_dir, "images", exact_scan_ceiling=0)
    query = routed_to_ann.get_vector_by_id("img-00000")
    assert query is not None

    # `_apply_search_strategy` is the unit under test; asserting on the builder
    # it returns is what distinguishes "took the ANN path" from "happened to
    # agree with the exact path", which on this fixture it often does.
    table = routed_to_ann._require_table()
    builder = routed_to_ann._apply_search_strategy(
        table.search(query).metric("cosine"), "split IN ('test')"
    )
    # `nprobes()` sets a min/max pair rather than one field, verified against
    # the installed lancedb 0.25.3 source.
    assert builder._minimum_nprobes == routed_to_ann._nprobes
    assert builder._maximum_nprobes == routed_to_ann._nprobes
    assert builder._refine_factor == routed_to_ann._refine_factor
