"""Turning a corpus selection into a downloadable manifest.

A researcher who has narrowed the corpus — by split, by caption text, by drawing
a box on the projection, or by running a query — needs that slice *outside* the
browser: as the input list to a training run, or as a table to open in pandas.
This service produces it.

Framework-agnostic by contract (CLAUDE.md §4.1): it yields text, and the route
is what knows about ``Content-Disposition``. Output is generated lazily so a
whole-corpus export never assembles 8 000 records in memory at once.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from collections.abc import AsyncIterator, Sequence
from functools import partial
from typing import Final

import anyio.to_thread

from app.models.domain import (
    ExportFormat,
    ExportRecord,
    ImageAnalysis,
    ImageDetail,
    ImageFilter,
)
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.projection_repository import ProjectionRepository
from app.repositories.vector_db import VectorRepository
from app.services.collection_service import CollectionService
from app.services.search_service import SearchService

LOGGER: Final = logging.getLogger(__name__)

#: Rows read from the store per batch when streaming an unbounded export. Large
#: enough that an 8 000-row export is a handful of scans, small enough that the
#: generator stays incremental.
_PAGE_SIZE: Final = 500

#: Caption columns the CSV writer emits. Flickr8k's schema supplies at most five
#: references per image (``CAPTION_COLUMNS`` in ``scripts/ingest.py``), so this
#: is the full width rather than a truncation. ``caption_count`` travels beside
#: them anyway, so a reader can always tell whether anything was dropped — and
#: JSONL carries the list verbatim if a future corpus ever exceeds this.
_CSV_CAPTION_COLUMNS: Final = 5

#: Column order for the manifest. ``collection`` sits beside ``split`` rather
#: than replacing it: a researcher exporting a re-partitioned corpus needs the
#: working assignment *and* the ground truth in the same file, because the
#: leakage figures in ``analysis.json`` are computed from the latter.
_CSV_HEADER: Final = [
    "id",
    "file_name",
    "split",
    "collection",
    "score",
    "x",
    "y",
    "nn_id",
    "nn_similarity",
    "caption_rank",
    "caption_count",
    *(f"caption_{index + 1}" for index in range(_CSV_CAPTION_COLUMNS)),
]


class ExportService:
    """Streams the current selection as CSV or JSONL."""

    def __init__(
        self,
        repository: VectorRepository,
        search_service: SearchService,
        collections: CollectionService,
        projection: ProjectionRepository | None = None,
        analysis: AnalysisRepository | None = None,
    ) -> None:
        """Bind the service to its sources.

        Args:
            repository: Source of records for the id and filter modes.
            search_service: Used only by the ranked mode, so that an exported
                ranking is reproducible from the request rather than from
                whatever the client happened to have on screen.
            collections: The user's partition overlay. Only the two
                repository-backed modes need it — the ranked modes go through
                the search service, which has already stamped its hits.
            projection: Map coordinates, when they exist. Optional because the
                export is useful without them; the ``x``/``y`` columns are then
                simply empty rather than the endpoint failing.
            analysis: Data-quality measurements, on the same terms. A manifest
                that carries them is one a researcher can filter offline —
                "drop everything with a near-duplicate" becomes a pandas
                expression rather than a second trip through the UI.
        """
        self._repository = repository
        self._search_service = search_service
        self._collections = collections
        self._projection = projection
        self._analysis = analysis

    def _record(self, image: ImageDetail, score: float | None = None) -> ExportRecord:
        """Attach whatever optional measurements exist to a record."""
        coordinates = None if self._projection is None else self._projection.coordinates()
        position = None if coordinates is None else coordinates.get(image.id)
        return ExportRecord(
            image=image,
            score=score,
            position=position,
            analysis=None if self._analysis is None else self._analysis.get(image.id),
        )

    async def stream(
        self,
        *,
        export_format: ExportFormat,
        image_filter: ImageFilter,
        image_ids: Sequence[str] | None = None,
        query: str | None = None,
        similar_to_image_id: str | None = None,
        limit: int,
    ) -> AsyncIterator[str]:
        """Render the selected records as a text stream.

        Args:
            export_format: Output format.
            image_filter: Narrowing applied in the ranked and whole-slice modes.
            image_ids: Explicit selection, which takes precedence over
                everything else.
            query: Free-text query for a ranked export.
            similar_to_image_id: Export this image's neighbours, ranked.
            limit: Row ceiling for the ranked modes.

        Yields:
            Chunks of the finished document, header first for CSV.
        """
        records = self.iter_records(
            image_filter=image_filter,
            image_ids=image_ids,
            query=query,
            similar_to_image_id=similar_to_image_id,
            limit=limit,
        )
        if export_format == "csv":
            async for chunk in render_csv(records):
                yield chunk
            return
        async for chunk in _render_jsonl(records):
            yield chunk

    def iter_records(
        self,
        *,
        image_filter: ImageFilter,
        image_ids: Sequence[str] | None,
        query: str | None,
        similar_to_image_id: str | None,
        limit: int,
    ) -> AsyncIterator[ExportRecord]:
        """Pick the record source implied by the request.

        Precedence is explicit-selection, then a ranking, then the filtered
        slice. An id list is the most specific thing a caller can say, so it
        wins; the filter is deliberately *not* re-applied to it, because the
        client selected those exact images and silently dropping some would be
        surprising.

        Args:
            image_filter: Narrowing for the ranked and whole-slice modes.
            image_ids: Explicit selection, highest precedence.
            query: Free-text query for a ranked export.
            similar_to_image_id: Export this image's neighbours, ranked.
            limit: Row ceiling for the ranked modes.

        Returns:
            An async iterator over the selected records.
        """
        if image_ids:
            return self._iter_selected(image_ids)
        if query is not None:
            return self._iter_ranked(query, limit, image_filter)
        if similar_to_image_id is not None:
            return self._iter_neighbours(similar_to_image_id, limit, image_filter)
        return self._iter_slice(image_filter)

    async def _iter_selected(self, image_ids: Sequence[str]) -> AsyncIterator[ExportRecord]:
        """Yield the requested ids, in the order they were given.

        Preserving caller order matters: the client sends search hits in rank
        order, and the store would return them in scan order.
        """
        found = await anyio.to_thread.run_sync(self._repository.get_many_by_id, image_ids)
        overlay = await self._collections.overlay()
        missing = 0
        for image_id in image_ids:
            detail = found.get(image_id)
            if detail is None:
                missing += 1
                continue
            yield self._record(overlay.decorate_detail(detail))
        if missing:
            LOGGER.warning("Export skipped %d id(s) not present in the index", missing)

    async def _iter_ranked(
        self,
        query: str,
        limit: int,
        image_filter: ImageFilter,
    ) -> AsyncIterator[ExportRecord]:
        """Yield ranked hits, carrying their similarity scores."""
        hits = await self._search_service.search(query, limit, image_filter)
        for hit in hits:
            yield self._record(hit.image, hit.score)

    async def _iter_neighbours(
        self,
        image_id: str,
        limit: int,
        image_filter: ImageFilter,
    ) -> AsyncIterator[ExportRecord]:
        """Yield the neighbours of one image, ranked, with their similarities."""
        hits = await self._search_service.search_by_image(image_id, limit, image_filter)
        for hit in hits:
            yield self._record(hit.image, hit.score)

    async def _iter_slice(self, image_filter: ImageFilter) -> AsyncIterator[ExportRecord]:
        """Yield every record matching the filter, a page at a time."""
        overlay = await self._collections.overlay()
        offset = 0
        while True:
            page = await anyio.to_thread.run_sync(
                partial(
                    self._repository.list_details,
                    offset=offset,
                    limit=_PAGE_SIZE,
                    image_filter=image_filter,
                )
            )
            if not page:
                return
            for detail in page:
                yield self._record(overlay.decorate_detail(detail))
            if len(page) < _PAGE_SIZE:
                return
            offset += len(page)


def _format_score(score: float | None) -> str:
    """Render a similarity for CSV, leaving the cell empty when unranked."""
    return "" if score is None else f"{score:.6f}"


def _format_position(position: tuple[float, float] | None) -> tuple[str, str]:
    """Render map coordinates for CSV, leaving both cells empty when unprojected."""
    if position is None:
        return ("", "")
    return (f"{position[0]:.5f}", f"{position[1]:.5f}")


def _format_analysis(analysis: ImageAnalysis | None) -> tuple[str, str, str]:
    """Render the quality measurements for CSV, empty where they do not exist."""
    if analysis is None:
        return ("", "", "")
    return (
        analysis.nearest_neighbour_id,
        f"{analysis.nearest_neighbour_similarity:.5f}",
        "" if analysis.caption_rank is None else str(analysis.caption_rank),
    )


async def render_csv(records: AsyncIterator[ExportRecord]) -> AsyncIterator[str]:
    """Serialise records as CSV, one row per image.

    Uses :mod:`csv` rather than string joining so that a caption containing a
    comma, a quote or a newline — Flickr8k has all three — is quoted correctly
    instead of corrupting the file.

    Args:
        records: The records to serialise.

    Yields:
        The header row, then one row per record.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    def take() -> str:
        """Drain the buffer, returning what the writer just put in it."""
        chunk = buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
        return chunk

    writer.writerow(_CSV_HEADER)
    yield take()

    async for record in records:
        captions = list(record.image.captions[:_CSV_CAPTION_COLUMNS])
        captions += [""] * (_CSV_CAPTION_COLUMNS - len(captions))
        writer.writerow(
            [
                record.image.id,
                record.image.file_name,
                record.image.split,
                record.image.collection,
                _format_score(record.score),
                *_format_position(record.position),
                *_format_analysis(record.analysis),
                len(record.image.captions),
                *captions,
            ]
        )
        yield take()


async def _render_jsonl(records: AsyncIterator[ExportRecord]) -> AsyncIterator[str]:
    """Serialise records as newline-delimited JSON, one object per image."""
    async for record in records:
        payload: dict[str, object] = {
            "id": record.image.id,
            "file_name": record.image.file_name,
            "split": record.image.split,
            "collection": record.image.collection,
            "captions": record.image.captions,
        }
        if record.score is not None:
            payload["score"] = record.score
        if record.position is not None:
            payload["x"], payload["y"] = record.position
        if record.analysis is not None:
            payload["nn_id"] = record.analysis.nearest_neighbour_id
            payload["nn_similarity"] = record.analysis.nearest_neighbour_similarity
            if record.analysis.caption_rank is not None:
                payload["caption_rank"] = record.analysis.caption_rank
        yield json.dumps(payload, ensure_ascii=False) + "\n"
