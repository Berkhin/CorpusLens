"""Logging setup for the API process.

Kept separate from ``main`` so tests and scripts can configure logging without
constructing an ASGI app.
"""

from __future__ import annotations

import logging
from typing import Final

_LOG_FORMAT: Final = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT: Final = "%H:%M:%S"


def configure_logging(level: str) -> None:
    """Install a stderr handler on the root logger.

    ``force=True`` replaces handlers installed by Uvicorn's own bootstrap, so
    application and server logs share one format instead of interleaving two.

    Args:
        level: Level name such as ``"INFO"`` or ``"DEBUG"``.
    """
    logging.basicConfig(
        level=level.upper(),
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        force=True,
    )
