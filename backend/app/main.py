"""ASGI application factory and entry point.

Run in development with::

    uvicorn app.main:app --reload --app-dir backend

Everything expensive is created by the lifespan in :mod:`app.core.lifespan`;
this module only assembles middleware, mounts and routers.
"""

from __future__ import annotations

import logging
from typing import Final

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import api_router
from app.core.config import Settings, get_settings
from app.core.lifespan import lifespan
from app.core.logging import configure_logging

LOGGER: Final = logging.getLogger(__name__)

_TITLE: Final = "Flickr8k Explorer API"
_DESCRIPTION: Final = (
    "Local, offline API over the Flickr8k corpus: paginated browsing, per-image captions, "
    "dataset statistics, CLIP text→image semantic search, split and caption-text filtering, "
    "a 2-D projection of the embedding space, and CSV/JSONL export of any selection."
)
_VERSION: Final = "0.1.0"

#: Responses below this many bytes are sent uncompressed. Gzip costs more CPU
#: than it saves bandwidth on a small JSON body over loopback; it earns its keep
#: on the one large payload this API serves, the full projection cloud.
_GZIP_MINIMUM_SIZE: Final = 1000


def _configure_cors(app: FastAPI, settings: Settings) -> None:
    """Permit the Vite dev server to call the API from the browser.

    ``allow_credentials`` stays off: the API has no auth and sets no cookies, so
    enabling it would only force Starlette to echo a specific origin instead of
    a wildcard for no benefit. Methods and headers are listed rather than
    wildcarded so the preflight response says exactly what is allowed.

    ``PATCH`` and ``DELETE`` are here for the collection endpoints. Omitting
    them does not fail loudly: the request never leaves the browser, and the
    console reports a CORS error that names nothing useful about which method
    was refused.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )


def _mount_images(app: FastAPI, settings: Settings) -> None:
    """Serve the ingested JPEGs so the client can render them directly.

    ``check_dir=False`` because the lifespan already validates the directory
    and raises a message that names the fix; StaticFiles' own error at
    construction time would fire before that and read as a stack trace instead.

    Serving originals rather than thumbnails is a deliberate deferral: Flickr8k
    images are small (a few hundred KB) and the browser is on localhost, so a
    resize pipeline would be optimising a link with no latency to save.
    """
    app.mount(
        settings.images_url_prefix,
        StaticFiles(directory=settings.images_dir, check_dir=False),
        name="images",
    )


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Serve the compiled SPA from the application root, if one was built.

    Only the container image sets ``frontend_dist_dir``. In development Vite
    owns the client on its own port and this is a no-op, so the two topologies
    stay independent of each other.

    **Registration order is load-bearing.** Starlette matches routes in the
    order they were added, and a mount at ``/`` matches every path beneath it.
    Calling this last is what keeps ``/api/*`` and ``/images/*`` reachable; move
    it above ``include_router`` and the SPA swallows the whole API.

    ``html=True`` serves ``index.html`` for the bare root. It does not add a
    catch-all rewrite for unknown paths, which is correct today: the client has
    no router, so there are no deep links to fall back for.
    """
    dist_dir = settings.frontend_dist_dir
    if dist_dir is None:
        return
    if not dist_dir.is_dir():
        LOGGER.warning("frontend_dist_dir %s does not exist; serving API only", dist_dir)
        return

    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")
    LOGGER.info("Serving the compiled frontend from %s", dist_dir)


def create_app() -> FastAPI:
    """Build the FastAPI application.

    A factory rather than a module-level construction so tests can build an app
    against overridden settings without importing side effects.

    Returns:
        The configured application, ready to serve.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=_TITLE,
        description=_DESCRIPTION,
        version=_VERSION,
        lifespan=lifespan,
    )

    _configure_cors(app, settings)
    # Added after CORS so that CORS stays the outermost layer: Starlette applies
    # middleware in reverse registration order, and the preflight response must
    # not arrive gzipped.
    app.add_middleware(GZipMiddleware, minimum_size=_GZIP_MINIMUM_SIZE)

    _mount_images(app, settings)
    app.include_router(api_router, prefix=settings.api_prefix)
    # Last: this mount claims "/" and would shadow everything above it.
    _mount_frontend(app, settings)

    LOGGER.debug("Application assembled with API prefix %s", settings.api_prefix)
    return app


#: Module-level instance for ``uvicorn app.main:app``.
app = create_app()
