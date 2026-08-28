"""Assembling the embedding map from coordinates plus corpus metadata.

The projection artefact holds only ``id -> (x, y)``. Everything else a map needs
— which split a point belongs to, whether it survives the active filter — lives
in the index. Joining the two is this service's whole job.

Framework-agnostic (CLAUDE.md §4.1).
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Final

import anyio.to_thread

from app.exceptions import ProjectionUnavailableError
from app.models.domain import ImageFilter, Projection, ProjectionPoint
from app.repositories.projection_repository import ProjectionRepository
from app.repositories.vector_db import VectorRepository

LOGGER: Final = logging.getLogger(__name__)


class ProjectionService:
    """Builds the full point cloud for the map view."""

    def __init__(
        self,
        repository: VectorRepository,
        projection: ProjectionRepository | None,
    ) -> None:
        """Bind the service to its sources.

        Args:
            repository: Source of splits and of filter evaluation.
            projection: Coordinates, or ``None`` when the offline projection
                step has not been run.
        """
        self._repository = repository
        self._projection = projection

    async def get_projection(self, image_filter: ImageFilter | None = None) -> Projection:
        """Return every projected image, flagged against the filter.

        The whole cloud is returned in one response rather than paged. That is a
        deliberate consequence of what the view is for: a scatter plot with a
        page missing is not a smaller scatter plot, it is a misleading one. At
        8 000 points the payload compresses to a couple of hundred kilobytes
        over loopback, and it never changes within a session.

        Args:
            image_filter: Narrowing to evaluate. Non-matching points are still
                returned, marked ``matches=False``, so the client can dim them
                instead of hiding where the subset sits in the corpus.

        Returns:
            The point cloud, its method, and how many points matched.

        Raises:
            ProjectionUnavailableError: If no projection has been computed. The
                route maps this to a 404; the service stays free of HTTP.
        """
        if self._projection is None:
            raise ProjectionUnavailableError(
                "No projection has been computed. Run `python scripts/project.py`."
            )

        coordinates = self._projection.coordinates()
        total = await anyio.to_thread.run_sync(self._repository.count)
        summaries = await anyio.to_thread.run_sync(
            partial(self._repository.list_summaries, offset=0, limit=total)
        )

        matching_ids = await self._matching_ids(image_filter, total)

        points: list[ProjectionPoint] = []
        for summary in summaries:
            position = coordinates.get(summary.id)
            # An id in the table but not in the artefact means the corpus grew
            # after the projection was computed. Skipping is right: inventing a
            # position would put the point somewhere meaningless, and the count
            # mismatch is what `--force` in docker/setup.sh exists to prevent.
            if position is None:
                continue
            points.append(
                ProjectionPoint(
                    id=summary.id,
                    split=summary.split,
                    x=position[0],
                    y=position[1],
                    matches=matching_ids is None or summary.id in matching_ids,
                )
            )

        if len(points) < len(summaries):
            LOGGER.warning(
                "%d image(s) have no projected position; re-run scripts/project.py --force",
                len(summaries) - len(points),
            )

        match_count = sum(1 for point in points if point.matches)
        return Projection(
            method=self._projection.method,
            explained_variance_ratio=self._projection.explained_variance_ratio,
            points=points,
            match_count=match_count,
        )

    async def _matching_ids(self, image_filter: ImageFilter | None, total: int) -> set[str] | None:
        """Resolve the filter to a set of ids, or ``None`` when it keeps everything.

        ``None`` rather than "every id" so the caller can skip a set lookup per
        point in the common unfiltered case.
        """
        if image_filter is None or image_filter.is_empty:
            return None
        matches = await anyio.to_thread.run_sync(
            partial(
                self._repository.list_summaries,
                offset=0,
                limit=total,
                image_filter=image_filter,
            )
        )
        return {summary.id for summary in matches}
