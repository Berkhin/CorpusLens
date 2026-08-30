"""The LanceDB implementation of :class:`~app.repositories.vector_db.VectorRepository`.

This is the only module in the application that talks to LanceDB. Everything
above this layer works in terms of :mod:`app.models.domain` types and never sees
a table handle, a query builder or a ``_distance`` column (CLAUDE.md §4.1) —
services and routes name the Protocol in ``vector_db.py``, not this class, so
replacing it is a one-line change in :mod:`app.core.lifespan`.

All methods here are **blocking**. LanceDB's Python client is synchronous and a
brute-force scan is CPU work, so callers in the service layer push these onto a
worker thread rather than running them on the event loop.

Verified against the installed versions on this machine — lancedb 0.25.3,
pyarrow 25.0.0, Python 3.12.13 — by introspecting the real table rather than
recalling an API (CLAUDE.md §6):

* ``Table.search(vector).metric("cosine")`` — ``metric`` accepts
  ``Literal["l2", "cosine", "dot"]``.
* ``LanceEmptyQueryBuilder`` (``search()`` with no argument) exposes
  ``where`` / ``select`` / ``limit`` / ``offset``, which is what backs
  pagination and id lookup.
* Scoring queries project ``_distance``. 0.25.3 auto-adds it and logs a
  deprecation warning when a projection omits it, so it is requested
  explicitly below — that silences the warning today and keeps the behaviour
  correct when the auto-projection is removed.
* ``Table.count_rows(filter=...)`` accepts the same expression syntax, which is
  what makes a filtered page report a filtered total.
* ``list_indices()`` reports the vector indices on a table, and the query
  builder exposes ``nprobes`` / ``refine_factor`` / ``bypass_vector_index``.
  Those four are what let one code path serve both an indexed and an unindexed
  corpus; see :meth:`LanceDBImageRepository._apply_search_strategy`.

Predicate strings themselves are built in :mod:`app.repositories.filters`; this
module only decides which query to run and how to map rows back to the domain.

**The API never writes to this table.** It is a pure reader of what
``scripts/ingest.py`` produces, which is what makes the ``split`` column
trustworthy ground truth and lets the scan order below be treated as stable.
User-created collections re-partition the corpus *without* touching it: they
live in a separate store owned by :mod:`app.repositories.collection_repository`
and are applied as an overlay above this layer. That separation is deliberate —
``scripts/analyze.py`` derives cross-split duplicate leakage from these splits,
and a re-partition that overwrote them would make the measurement quietly wrong.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, cast

import lancedb
import numpy as np
from numpy.typing import NDArray

from app.exceptions import DatasetUnavailableError
from app.models.domain import ImageDetail, ImageFilter, ImageSummary, SearchHit
from app.repositories.filters import (
    build_filter_expression,
    build_id_equality_expression,
    build_id_membership_expression,
)
from app.repositories.vector_db import VectorRepository

LOGGER: Final = logging.getLogger(__name__)

_SUMMARY_COLUMNS: Final = ["id", "file_name", "split"]
_DETAIL_COLUMNS: Final = ["id", "file_name", "split", "captions"]

#: LanceDB reports similarity as a distance; with the cosine metric that is
#: ``1 - cosine_similarity``. The ingestion script writes unit-length vectors
#: and the query encoder normalizes too, so inverting it is exact rather than
#: an approximation.
_DISTANCE_COLUMN: Final = "_distance"

#: Fallbacks for the search tunables, used when a caller constructs this class
#: directly. The application passes them explicitly from ``Settings``, which is
#: where the reasoning behind each value is documented.
_DEFAULT_NPROBES: Final = 20
_DEFAULT_REFINE_FACTOR: Final = 10
_DEFAULT_EXACT_SCAN_CEILING: Final = 50_000


def _detect_ann_index(table: Any) -> bool:  # noqa: ANN401 - lancedb ships no py.typed
    """Report whether a vector index exists on the table.

    Resolved once at construction rather than per query: the API never writes
    to the table, so an index cannot appear or vanish while the process runs,
    and ``list_indices()`` is a metadata read we should not repeat 20 times a
    second.

    A table with no index is the normal case at the reference corpus size and
    is not a degraded state — it is the exact-search path.

    Args:
        table: An open LanceDB table, or a test double standing in for one.

    Returns:
        True when at least one index is present. Any failure to introspect is
        reported as False: falling back to an exact scan is always correct,
        merely slower, whereas assuming an index that is not there would send
        unsupported ``nprobes`` calls at the query builder.
    """
    lister = getattr(table, "list_indices", None)
    if lister is None:
        return False
    try:
        return bool(list(lister()))
    # Deliberately broad: see the docstring — any failure to introspect means
    # "scan exactly", which is always correct and merely slower.
    except Exception:
        LOGGER.warning("Could not read index metadata; using exact search", exc_info=True)
        return False


class LanceDBImageRepository:
    """Read-only access to the ingested corpus index, backed by LanceDB.

    Satisfies :class:`~app.repositories.vector_db.VectorRepository` structurally
    — there is no base class to inherit, by design; see that module. Conformance
    is asserted at the bottom of this file so a drifting signature is a
    type-check failure here rather than at some distant call site.

    The API never writes: the table is produced offline by ``scripts/ingest.py``
    (CLAUDE.md §4.2), so this class exposes queries only.
    """

    def __init__(
        self,
        table: Any,  # noqa: ANN401 - lancedb ships no py.typed
        *,
        nprobes: int = _DEFAULT_NPROBES,
        refine_factor: int = _DEFAULT_REFINE_FACTOR,
        exact_scan_ceiling: int = _DEFAULT_EXACT_SCAN_CEILING,
    ) -> None:
        """Wrap an already-open LanceDB table.

        Prefer :meth:`open`; this constructor exists so tests can inject a
        double without a database on disk.

        Args:
            table: An open ``lancedb.table.Table``.
            nprobes: IVF partitions to probe when the ANN path is taken.
            refine_factor: Candidate multiplier re-ranked against full-precision
                vectors. The recall lever; see :meth:`search_by_vector`.
            exact_scan_ceiling: Survivor count below which a filtered query
                bypasses the index and scans exactly.
        """
        self._table = table
        self._nprobes = nprobes
        self._refine_factor = refine_factor
        self._exact_scan_ceiling = exact_scan_ceiling
        self._has_ann_index = _detect_ann_index(table)
        #: Memoized whole-corpus split projection; see :meth:`split_by_id`.
        self._split_cache: dict[str, str] | None = None

    @classmethod
    def open(
        cls,
        lancedb_dir: Path,
        table_name: str,
        *,
        nprobes: int = _DEFAULT_NPROBES,
        refine_factor: int = _DEFAULT_REFINE_FACTOR,
        exact_scan_ceiling: int = _DEFAULT_EXACT_SCAN_CEILING,
    ) -> LanceDBImageRepository:
        """Connect to the embedded database and open the images table.

        Args:
            lancedb_dir: Directory backing the embedded database.
            table_name: Table written by the ingestion script.
            nprobes: IVF partitions to probe when the ANN path is taken.
            refine_factor: Candidate multiplier for the ANN re-ranking pass.
            exact_scan_ceiling: Survivor count below which a filtered query
                scans exactly instead of using the index.

        Returns:
            A repository bound to that table.

        Raises:
            DatasetUnavailableError: If the directory or table is absent — i.e.
                ingestion has not been run against this ``data/`` directory.
        """
        if not lancedb_dir.is_dir():
            raise DatasetUnavailableError(
                f"LanceDB directory {lancedb_dir} does not exist. "
                "Run `python scripts/ingest.py` first."
            )

        connection = lancedb.connect(lancedb_dir)
        if table_name not in connection.table_names():
            raise DatasetUnavailableError(
                f"Table {table_name!r} not found in {lancedb_dir}. "
                "Run `python scripts/ingest.py` first."
            )

        table = connection.open_table(table_name)
        repository = cls(
            table,
            nprobes=nprobes,
            refine_factor=refine_factor,
            exact_scan_ceiling=exact_scan_ceiling,
        )
        LOGGER.info(
            "Opened LanceDB table %r (%d rows, %s)",
            table_name,
            table.count_rows(),
            "ANN index present" if repository._has_ann_index else "exact search",
        )
        return repository

    def close(self) -> None:
        """Release the table handle.

        lancedb 0.25.3 exposes no ``close()`` on either the connection or the
        table — the embedded store holds no socket and releases its file
        handles when the object is collected. Dropping the reference is
        therefore the whole of teardown; this method exists so the lifespan has
        an explicit, honest place to do it.
        """
        self._table = None

    def count(self, image_filter: ImageFilter | None = None) -> int:
        """Return the number of images matching a filter.

        Args:
            image_filter: Narrowing to apply, or ``None`` for the whole corpus.

        Returns:
            Matching row count. Unfiltered this is table metadata and free;
            filtered it is a scan, which at this corpus size is still
            milliseconds.
        """
        return int(self._require_table().count_rows(filter=build_filter_expression(image_filter)))

    def count_by_split(self) -> dict[str, int]:
        """Return the number of images per split.

        Projects the ``split`` column alone and tallies it in Python. At
        Flickr8k's scale (~8k rows) that reads a few hundred kilobytes; pulling
        whole rows would drag every 512-d vector along for a count.

        Returns:
            Split name to row count, ordered most-populous first. Splits absent
            from the index are absent from the mapping — a ``--limit``ed
            ingestion run legitimately contains only ``train``.
        """
        table = self._require_table()
        projection = table.search().select(["split"]).limit(None).to_arrow()
        splits = cast(list[str], projection.column("split").to_pylist())
        return dict(Counter(splits).most_common())

    def split_by_id(self) -> dict[str, str]:
        """Return every image's ground-truth split, keyed by id.

        The two-column counterpart to :meth:`count_by_split`, and measured at
        the same cost: on the 8 000-row corpus both take ~93 ms, because the
        scan is dominated by fixed overhead rather than by the extra string
        column. That is what makes "which collection is each image effectively
        in" answerable without a query per image — the overlay store knows only
        the *overrides*, so the default half of every image's membership lives
        here.

        **Memoized after the first call.** This is reached from
        ``CollectionService.membership()``, and so from both ``/dataset/stats``
        and ``/collections`` — two endpoints the client requests on load. Paying
        a whole-corpus scan on each was ~93 ms at the reference size but is
        linear: ~2.3 s per dashboard load at 200k rows, ~12 s at a million.

        Caching is sound rather than merely convenient because the API never
        writes to this table (CLAUDE.md §4.2), so the mapping cannot change
        while the process runs. The overlay that *does* change is a separate
        store and is read fresh on every request. The cost is residency —
        roughly 50 bytes per image, so ~50 MB at a million images, which is the
        scale at which this should become an eviction cache instead.

        Returns:
            Image id to split name, for the whole corpus. The same dict object
            on every call, so callers must treat it as read-only.
        """
        if self._split_cache is not None:
            return self._split_cache

        table = self._require_table()
        projection = table.search().select(["id", "split"]).limit(None).to_arrow()
        ids = cast(list[str], projection.column("id").to_pylist())
        splits = cast(list[str], projection.column("split").to_pylist())
        self._split_cache = dict(zip(ids, splits, strict=True))
        return self._split_cache

    def list_ids(self, image_filter: ImageFilter | None, *, limit: int) -> list[str]:
        """Return the ids of the images a filter selects.

        Projects the ``id`` column alone, on the same reasoning as
        :meth:`count_by_split`: pulling whole rows would drag a 512-d vector
        along per image to collect a string. This is what turns a filter into
        something the collection overlay can act on, since an override is keyed
        by id and the overlay store knows nothing about the index.

        Args:
            image_filter: Narrowing to apply, or ``None`` for the whole corpus.
            limit: Maximum ids to return. Required rather than defaulted: every
                caller has a ceiling in mind, and an accidental whole-corpus
                read here becomes an 8 000-entry ``IN`` list downstream.

        Returns:
            Matching ids in table scan order, empty when ``limit`` is not
            positive — the honest answer for "give me none of them", and one
            LanceDB's builder would otherwise have to be asked to express.
        """
        if limit < 1:
            return []
        table = self._require_table()
        query = table.search().select(["id"])
        expression = build_filter_expression(image_filter)
        if expression is not None:
            query = query.where(expression)
        projection = query.limit(limit).to_arrow()
        return cast(list[str], projection.column("id").to_pylist())

    def list_summaries(
        self,
        *,
        offset: int,
        limit: int,
        image_filter: ImageFilter | None = None,
    ) -> list[ImageSummary]:
        """Return one page of image summaries.

        Ordering is the table's scan order, which for this append-only index is
        the order ``scripts/ingest.py`` wrote rows in (train, then validation,
        then test). Verified stable across repeated calls on the installed
        version. That is a property of a static, never-compacted table rather
        than a sort guarantee from LanceDB — it holds precisely because the API
        is a pure reader, and would need an explicit sort key if the index ever
        became writable at runtime.

        Args:
            offset: Rows to skip. Callers validate it is non-negative.
            limit: Maximum rows to return.
            image_filter: Narrowing to apply before paging, so ``offset`` walks
                the filtered sequence rather than the whole corpus.

        Returns:
            Summaries for the requested window; shorter than ``limit`` at the
            end of the matching set, empty past it.
        """
        table = self._require_table()
        query = table.search().select(_SUMMARY_COLUMNS)
        expression = build_filter_expression(image_filter)
        if expression is not None:
            query = query.where(expression)
        rows = query.offset(offset).limit(limit).to_list()
        return [_to_summary(row) for row in cast(list[dict[str, Any]], rows)]

    def get_by_id(self, image_id: str) -> ImageDetail | None:
        """Fetch one image with its captions.

        Args:
            image_id: Corpus image id.

        Returns:
            The record, or ``None`` if no row carries that id.
        """
        table = self._require_table()
        rows = (
            table.search()
            .where(build_id_equality_expression(image_id))
            .select(_DETAIL_COLUMNS)
            .limit(1)
            .to_list()
        )
        typed_rows = cast(list[dict[str, Any]], rows)
        if not typed_rows:
            return None
        return _to_detail(typed_rows[0])

    def list_details(
        self,
        *,
        offset: int,
        limit: int,
        image_filter: ImageFilter | None = None,
    ) -> list[ImageDetail]:
        """Return one page of records with their captions.

        The caption-carrying counterpart to :meth:`list_summaries`, used by the
        export path — a manifest without captions would be of little use to
        anyone building a training subset. Ordering and paging semantics are
        identical.

        Args:
            offset: Rows to skip.
            limit: Maximum rows to return.
            image_filter: Narrowing to apply before paging.

        Returns:
            Detail records for the requested window.
        """
        table = self._require_table()
        query = table.search().select(_DETAIL_COLUMNS)
        expression = build_filter_expression(image_filter)
        if expression is not None:
            query = query.where(expression)
        rows = query.offset(offset).limit(limit).to_list()
        return [_to_detail(row) for row in cast(list[dict[str, Any]], rows)]

    def get_many_by_id(self, image_ids: Sequence[str]) -> dict[str, ImageDetail]:
        """Fetch several records at once, keyed by id.

        A mapping rather than a list because the store returns scan order and
        the caller — exporting a selection or a ranking — usually needs its own
        order preserved. Ids that match nothing are simply absent from the
        result, which is what lets the caller distinguish them.

        Args:
            image_ids: Ids to look up. An empty sequence short-circuits: ``IN
                ()`` is a syntax error, and there is nothing to ask for.

        Returns:
            Id to record, for every id that exists.
        """
        if not image_ids:
            return {}
        table = self._require_table()
        rows = (
            table.search()
            .where(build_id_membership_expression(image_ids))
            .select(_DETAIL_COLUMNS)
            .limit(len(image_ids))
            .to_list()
        )
        details = [_to_detail(row) for row in cast(list[dict[str, Any]], rows)]
        return {detail.id: detail for detail in details}

    def get_vector_by_id(self, image_id: str) -> NDArray[np.float32] | None:
        """Fetch one image's stored embedding.

        This is what makes image-to-image search cost nothing: the vector was
        computed during ingestion, so finding an image's neighbours needs no
        inference at all — unlike a text query, which still costs a forward
        pass through CLIP's text encoder.

        Args:
            image_id: Corpus image id.

        Returns:
            The unit-length 512-d embedding, or ``None`` if no row carries that
            id.
        """
        table = self._require_table()
        rows = (
            table.search()
            .where(build_id_equality_expression(image_id))
            .select(["vector"])
            .limit(1)
            .to_list()
        )
        typed_rows = cast(list[dict[str, Any]], rows)
        if not typed_rows:
            return None
        return np.asarray(typed_rows[0]["vector"], dtype=np.float32)

    def search_by_vector(
        self,
        vector: NDArray[np.float32],
        limit: int,
        image_filter: ImageFilter | None = None,
    ) -> list[SearchHit]:
        """Rank images by cosine similarity to a query vector.

        Exactness is preserved wherever it is affordable, which at the corpus
        sizes this tool targets is nearly everywhere. Three regimes, chosen per
        query by :meth:`_apply_search_strategy`; CLAUDE.md §4.4 is the contract
        and carries the measurements.

        Args:
            vector: Unit-length query embedding in CLIP's shared space.
            limit: Maximum hits to return.
            image_filter: Narrowing applied **before** ranking. Pre-filtering is
                not an optimisation here but a correctness requirement: a
                post-filter would take the global top ``limit`` and then discard
                from it, so asking for 20 results within one split would
                usually return fewer than 20 — silently, and with the tail of
                the ranking missing.

        Returns:
            Hits ordered by decreasing similarity. Exact unless the corpus is
            both indexed and unfiltered — or indexed and filtered so loosely
            that more than ``exact_scan_ceiling`` rows survive.
        """
        table = self._require_table()
        expression = build_filter_expression(image_filter)

        query = self._apply_search_strategy(table.search(vector).metric("cosine"), expression)
        query = query.select([*_DETAIL_COLUMNS, _DISTANCE_COLUMN])
        if expression is not None:
            query = query.where(expression, prefilter=True)

        rows = query.limit(limit).to_list()
        return [_to_hit(row) for row in cast(list[dict[str, Any]], rows)]

    def _apply_search_strategy(
        self,
        query: Any,  # noqa: ANN401 - lancedb ships no py.typed
        expression: str | None,
    ) -> Any:  # noqa: ANN401 - lancedb ships no py.typed
        """Choose between the exact scan and the ANN path for one query.

        The interesting case is the third one, and it is a correctness fix
        rather than a tuning knob. An IVF pre-filter is applied *within the
        probed partitions*, so a selective filter starves the candidate pool
        and the shortfall cannot be probed away. Measured on a 200k-row corpus
        with a filter selecting 20% of rows: recall@20 fell from an exact 1.000
        to 0.71, and raising ``nprobes`` from 20 to 200 moved it only to 0.75 —
        while every configuration still returned a full page of results, so
        nothing downstream could detect the loss.

        The saving grace is that the same selectivity which breaks the index
        also makes the exact scan cheap: few survivors is little to scan. So a
        filtered query below the ceiling takes the exact path and keeps the
        guarantee ``vector_db.VectorRepository`` documents.

        ``count_rows`` on the predicate costs a scan of its own, but a
        projection-free one, and it is what lets the decision rest on the real
        survivor count rather than on a guess about the predicate's shape — the
        collection overlay compiles to id literals whose selectivity is not
        readable from the expression.

        Args:
            query: A LanceDB vector query builder, freshly created.
            expression: The compiled pre-filter, or ``None`` when unfiltered.

        Returns:
            The same builder with the chosen strategy applied.
        """
        if not self._has_ann_index:
            # Exact by construction — there is no index to bypass.
            return query

        if expression is None:
            return query.nprobes(self._nprobes).refine_factor(self._refine_factor)

        survivors = int(self._require_table().count_rows(filter=expression))
        if survivors <= self._exact_scan_ceiling:
            LOGGER.debug("Filter leaves %d row(s); bypassing the index for exactness", survivors)
            return query.bypass_vector_index()

        LOGGER.debug("Filter leaves %d row(s); using the ANN path", survivors)
        return query.nprobes(self._nprobes).refine_factor(self._refine_factor)

    def _require_table(self) -> Any:  # noqa: ANN401 - lancedb ships no py.typed
        """Return the open table, or fail loudly if it was already closed."""
        if self._table is None:
            raise DatasetUnavailableError("Repository has been closed")
        return self._table


def _to_summary(row: dict[str, Any]) -> ImageSummary:
    """Build a summary from one projected LanceDB row.

    ``collection`` is seeded from the row's own split. This module knows nothing
    about the overlay store and must not — a service stamps the effective value
    afterwards. Seeding it here rather than defaulting the field means a
    forgotten stamp shows the image in its real split instead of an empty
    string.
    """
    return ImageSummary(
        id=str(row["id"]),
        file_name=str(row["file_name"]),
        split=str(row["split"]),
        collection=str(row["split"]),
    )


def _to_detail(row: dict[str, Any]) -> ImageDetail:
    """Build a detail record from one projected LanceDB row.

    ``collection`` is seeded from the split; see :func:`_to_summary`.
    """
    captions = cast(list[Any], row.get("captions") or [])
    return ImageDetail(
        id=str(row["id"]),
        file_name=str(row["file_name"]),
        split=str(row["split"]),
        collection=str(row["split"]),
        captions=[str(caption) for caption in captions],
    )


def _to_hit(row: dict[str, Any]) -> SearchHit:
    """Build a ranked hit, converting cosine distance back to similarity."""
    distance = float(row[_DISTANCE_COLUMN])
    return SearchHit(image=_to_detail(row), score=1.0 - distance)


def _assert_conformance(repository: LanceDBImageRepository) -> VectorRepository:
    """Statically assert this class satisfies the storage contract.

    A Protocol is only checked where a value is *used* as one, so without this
    a signature drifting out of line with ``VectorRepository`` would surface as
    an error in whichever service happened to pass it — or not at all, if the
    method were merely renamed. Naming the requirement here makes mypy report it
    against the class that broke it. Never called at runtime.
    """
    return repository
