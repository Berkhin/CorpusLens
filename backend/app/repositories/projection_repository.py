"""The only module that reads the pre-computed projection artefact.

``scripts/project.py`` writes ``data/projection.json``; this reads it once at
startup and holds the coordinates in memory. At 8 000 points that is a few
hundred kilobytes of floats — small enough that re-reading per request would be
pure waste, and small enough that keeping it resident costs nothing.

**A missing file is not a failure.** Unlike the LanceDB table, the projection is
optional: an operator who ran ingestion but not projection should get a working
gallery and search with the map view greyed out, not a process that refuses to
start. :meth:`ProjectionRepository.load` therefore returns ``None`` rather than
raising, and the lifespan treats that as a supported state.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

LOGGER: Final = logging.getLogger(__name__)

_METHOD_KEY: Final = "method"
_POINTS_KEY: Final = "points"
_EXPLAINED_VARIANCE_KEY: Final = "explained_variance_ratio"


class ProjectionRepository:
    """In-memory access to the 2-D coordinates of every embedded image."""

    def __init__(
        self,
        *,
        method: str,
        coordinates: Mapping[str, tuple[float, float]],
        explained_variance_ratio: tuple[float, ...] | None,
    ) -> None:
        """Hold an already-parsed projection.

        Args:
            method: Which algorithm produced the coordinates.
            coordinates: Image id to ``(x, y)``.
            explained_variance_ratio: Per-component variance share, PCA only.
        """
        self._method = method
        self._coordinates = coordinates
        self._explained_variance_ratio = explained_variance_ratio

    @classmethod
    def load(cls, path: Path) -> ProjectionRepository | None:
        """Read a projection from disk.

        Args:
            path: Location of the artefact.

        Returns:
            The loaded repository, or ``None`` when the file is absent or
            unusable. A malformed file is logged and treated as absent: the
            failure belongs to the offline pipeline, and refusing to serve the
            gallery over it would punish the wrong thing.
        """
        if not path.is_file():
            LOGGER.info("No projection at %s — the map view will be unavailable", path)
            return None

        try:
            document: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Could not read the projection at %s; disabling the map view", path)
            return None

        if not isinstance(document, dict) or not isinstance(document.get(_POINTS_KEY), dict):
            LOGGER.warning("Projection at %s has no points object; disabling the map view", path)
            return None

        coordinates: dict[str, tuple[float, float]] = {}
        for image_id, position in document[_POINTS_KEY].items():
            if not isinstance(position, list) or len(position) != 2:
                continue
            coordinates[str(image_id)] = (float(position[0]), float(position[1]))

        explained = document.get(_EXPLAINED_VARIANCE_KEY)
        ratio = tuple(float(value) for value in explained) if isinstance(explained, list) else None
        method = str(document.get(_METHOD_KEY, "unknown"))

        LOGGER.info(
            "Loaded %d projected point(s) from %s (method %r)", len(coordinates), path, method
        )
        return cls(method=method, coordinates=coordinates, explained_variance_ratio=ratio)

    @property
    def method(self) -> str:
        """Algorithm that produced the coordinates."""
        return self._method

    @property
    def explained_variance_ratio(self) -> tuple[float, ...] | None:
        """Per-component share of total variance, or ``None`` for t-SNE."""
        return self._explained_variance_ratio

    def coordinates(self) -> Mapping[str, tuple[float, float]]:
        """Return the id-to-position mapping.

        Returned as a read-only mapping rather than copied: it is shared by
        every request and never mutated.
        """
        return self._coordinates
