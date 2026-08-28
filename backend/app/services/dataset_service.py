"""Business logic for browsing the corpus.

Framework-agnostic by contract (CLAUDE.md §4.1): nothing here imports FastAPI or
touches a ``Request``, so the whole layer is unit-testable without an HTTP
client.

Every method is ``async`` but the work underneath is blocking — LanceDB's client
is synchronous. Calls are pushed onto anyio's worker-thread pool so a scan can
never stall the event loop while other requests are in flight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Final

import anyio.to_thread

from app.exceptions import ImageNotFoundError
from app.models.domain import (
    CollectionCaptionRecall,
    CollectionOverlay,
    DatasetStats,
    DuplicatePair,
    ImageFilter,
    ImagePage,
    InspectedImage,
)
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.vector_db import VectorRepository
from app.services.collection_service import CollectionService

#: Ranks the per-collection caption recall is reported at, matching the
#: convention in the image-text retrieval literature and what ``analyze.py``
#: reports corpus-wide.
_RECALL_AT: Final = (1, 5, 10)


def _count_cross_collection_pairs(
    pairs: Sequence[DuplicatePair], overlay: CollectionOverlay
) -> int:
    """Count near-duplicate pairs whose two images sit in different collections.

    The corpus-level figure beside this one is computed from ``split`` and must
    never move. This one has to, or the feature lets a researcher act on leakage
    without ever seeing whether the action worked.

    Costs nothing: each pair already carries both ids and both splits, so the
    effective collection of each side is a dictionary lookup. On the shipped
    corpus that is 52 pairs.

    Args:
        pairs: Every near-duplicate pair the analysis found.
        overlay: The current override state.

    Returns:
        How many pairs straddle a collection boundary.
    """
    return sum(
        1
        for pair in pairs
        if overlay.effective(pair.a, pair.a_split) != overlay.effective(pair.b, pair.b_split)
    )


def _caption_recall_by_collection(
    membership: Mapping[str, Sequence[str]], ranks: Mapping[str, int]
) -> dict[str, CollectionCaptionRecall]:
    """Re-aggregate the per-image caption ranks over each collection.

    A filter and a count over numbers ``analyze.py`` already computed against
    the whole corpus — see :class:`~app.models.domain.CollectionCaptionRecall`
    for why that is honest and for the harder number it is *not*.

    Args:
        membership: Collection id to its member image ids.
        ranks: Image id to median own-caption rank, for measured images only.

    Returns:
        One entry per collection that has at least one measured image.
        Collections with none are omitted rather than reported as zero, which
        would read as "these annotations are terrible" instead of "this was
        not measured".
    """
    recalls: dict[str, CollectionCaptionRecall] = {}
    for collection_id, members in membership.items():
        measured = [ranks[image_id] for image_id in members if image_id in ranks]
        if not measured:
            continue
        hits = {k: sum(1 for rank in measured if rank <= k) / len(measured) for k in _RECALL_AT}
        recalls[collection_id] = CollectionCaptionRecall(
            recall_at_1=hits[1],
            recall_at_5=hits[5],
            recall_at_10=hits[10],
            images=len(measured),
        )
    return recalls


class DatasetService:
    """Read-side operations over the ingested corpus index."""

    def __init__(
        self,
        repository: VectorRepository,
        collections: CollectionService,
        analysis: AnalysisRepository | None = None,
    ) -> None:
        """Bind the service to its sources.

        Args:
            repository: Source of image records.
            collections: The user's partition overlay, used to stamp each
                record's effective collection.
            analysis: Data-quality measurements, when the offline pass has been
                run. Optional throughout: every method below degrades to the
                pre-analysis answer rather than failing.
        """
        self._repository = repository
        self._collections = collections
        self._analysis = analysis

    async def get_stats(self) -> DatasetStats:
        """Compute corpus-level counts and quality figures for the dashboard.

        Every figure that can be read two ways is reported **both** ways, side
        by side. ``images_by_split`` never moves and ``images_by_collection``
        follows the user; ``cross_split_duplicate_pairs`` never moves and
        ``cross_collection_duplicate_pairs`` follows the user. Reporting only
        the first of each pair lets a researcher act on a finding without ever
        seeing the effect; reporting only the second would quietly redefine what
        "test set" means. The pairing is the point.

        Returns:
            Totals, both partitions, and — when the offline analysis exists —
            the duplicate and caption-retrieval figures against each.
        """
        total = await anyio.to_thread.run_sync(self._repository.count)
        by_split = await anyio.to_thread.run_sync(self._repository.count_by_split)
        membership = await self._collections.membership()
        by_collection = {
            collection_id: len(members) for collection_id, members in membership.items()
        }

        if self._analysis is None:
            return DatasetStats(
                total_images=total,
                images_by_split=by_split,
                images_by_collection=by_collection,
            )

        overlay = await self._collections.overlay()
        ranks = self._analysis.caption_ranks
        return DatasetStats(
            total_images=total,
            images_by_split=by_split,
            images_by_collection=by_collection,
            near_duplicate_images=len(self._analysis.ids_for("near-duplicate")),
            cross_split_duplicate_pairs=self._analysis.cross_split_pair_count,
            cross_collection_duplicate_pairs=_count_cross_collection_pairs(
                self._analysis.duplicate_pairs, overlay
            ),
            caption_retrieval=self._analysis.caption_retrieval,
            caption_recall_by_collection=(
                None if not ranks else _caption_recall_by_collection(membership, ranks)
            ),
        )

    async def list_images(
        self,
        *,
        offset: int,
        limit: int,
        image_filter: ImageFilter | None = None,
    ) -> ImagePage:
        """Return one page of image summaries.

        Totals are fetched alongside the page because a grid needs them to size
        its scrollbar and to report how much of the corpus survived the filter,
        and both counts are cheap at this scale.

        Args:
            offset: Rows to skip; must be non-negative.
            limit: Maximum rows to return; must be positive.
            image_filter: Narrowing to apply to both the page and its total.

        Returns:
            The requested window, the matching total that paginates it, and the
            unfiltered corpus total.

        Raises:
            ValueError: If ``offset`` is negative or ``limit`` is not positive.
                Bounds are also enforced by the route's query-parameter schema;
                this guards the service's own contract for non-HTTP callers.
        """
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit < 1:
            raise ValueError("limit must be positive")

        rows = await anyio.to_thread.run_sync(
            partial(
                self._repository.list_summaries,
                offset=offset,
                limit=limit,
                image_filter=image_filter,
            )
        )
        overlay = await self._collections.overlay()
        items = [overlay.decorate_summary(summary) for summary in rows]
        corpus_total = await anyio.to_thread.run_sync(self._repository.count)
        # Only pay for the second scan when a filter could actually change the
        # answer; unfiltered, the two counts are the same number by definition.
        total = (
            corpus_total
            if image_filter is None or image_filter.is_empty
            else await anyio.to_thread.run_sync(partial(self._repository.count, image_filter))
        )
        return ImagePage(
            items=items,
            total=total,
            corpus_total=corpus_total,
            offset=offset,
            limit=limit,
        )

    async def get_image(self, image_id: str) -> InspectedImage:
        """Fetch a single image with its captions and its measurements.

        Args:
            image_id: Corpus image id.

        Returns:
            The full record, with quality measurements attached when they
            exist. A missing analysis leaves ``analysis`` ``None`` rather than
            failing: inspecting an image must not depend on an optional
            offline step.

        Raises:
            ImageNotFoundError: If no row carries that id. The route maps this
                to a 404; the service stays free of HTTP semantics.
        """
        detail = await anyio.to_thread.run_sync(self._repository.get_by_id, image_id)
        if detail is None:
            raise ImageNotFoundError(image_id)
        overlay = await self._collections.overlay()
        return InspectedImage(
            detail=overlay.decorate_detail(detail),
            analysis=None if self._analysis is None else self._analysis.get(image_id),
        )
