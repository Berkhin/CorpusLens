"""Run the API with the bind address taken from settings.

``uvicorn app.main:app`` remains the documented development command, and it is
the better one there: ``--reload`` belongs to the CLI. This module exists so the
bind address can come from the same place as every other setting — the process
environment or the repository-root ``.env`` — because uvicorn's CLI reads
neither, and a ``CORPUSLENS_PORT`` that only some entry points honoured would be
worse than no setting at all.

Run with::

    python -m app                     # honours CORPUSLENS_HOST / CORPUSLENS_PORT

The container image uses this form for exactly that reason.
"""

from __future__ import annotations

import uvicorn

from app.core.config import get_settings


def main() -> None:
    """Serve the application on the configured interface and port."""
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_config=None,  # `configure_logging` in `create_app` already owns this.
    )


if __name__ == "__main__":
    main()
