"""Application startup and shutdown — where expensive resources are owned.

The CLIP model and the LanceDB table are process-wide singletons created
exactly once here and hung off ``app.state`` (CLAUDE.md §4.1). Loading the
bi-encoder costs seconds of CPU on this machine; doing it per request would be
indefensible, and doing it at import time would make the module unimportable
without a populated ``data/`` directory.

This module is the application's composition root: it is the one place allowed
to know about every layer at once, so that the layers need not know about each
other.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Final

import anyio.to_thread
import torch
from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.exceptions import DatasetUnavailableError
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.collection_repository import CollectionRepository
from app.repositories.image_repository import LanceDBImageRepository
from app.repositories.projection_repository import ProjectionRepository
from app.repositories.vector_db import VectorRepository
from app.services.embedding import ClipEmbeddingService, EmbeddingService

LOGGER: Final = logging.getLogger(__name__)

#: Attribute on ``app.state`` holding :class:`AppResources`.
RESOURCES_ATTRIBUTE: Final = "resources"


@dataclass(frozen=True, slots=True)
class AppResources:
    """Long-lived objects shared by every request.

    Bundled into one typed container rather than scattered across ``app.state``
    attributes: ``state`` is untyped by nature, so a single well-known
    attribute holding a frozen dataclass gives the dependency layer something
    mypy can actually check.

    Attributes:
        settings: The resolved configuration this process started with.
        embedder: Projects queries into the corpus's embedding space. Typed as
            the Protocol, not as the CLIP implementation: this bundle is the
            only place an implementation is named, and naming it here too would
            put the coupling back.
        image_repository: Open handle to the ingested index, likewise typed as
            its Protocol.
        projection_repository: The 2-D map, or ``None`` when the optional
            projection step has not been run. Typed optional rather than
            defaulted to an empty projection so that "not computed" and
            "computed and empty" stay distinguishable.
        analysis_repository: Data-quality measurements, or ``None`` on the same
            terms.
        collection_repository: The user's partition overlay. **Not** optional,
            unlike the two above: it is a store this process owns and creates on
            first open rather than an artefact an offline script may or may not
            have produced.
    """

    settings: Settings
    embedder: EmbeddingService
    image_repository: VectorRepository
    projection_repository: ProjectionRepository | None
    analysis_repository: AnalysisRepository | None
    collection_repository: CollectionRepository


def _verify_data_layout(settings: Settings) -> None:
    """Fail fast when the offline pipeline has not been run.

    The API is a pure reader of what ``scripts/ingest.py`` produces
    (CLAUDE.md §4.2). A missing corpus is an operator error, and a process that
    refuses to start with a clear message beats one that serves empty pages.

    Args:
        settings: Resolved configuration.

    Raises:
        DatasetUnavailableError: If the images directory is missing.
    """
    if not settings.images_dir.is_dir():
        raise DatasetUnavailableError(
            f"Images directory {settings.images_dir} does not exist. "
            "Run `python scripts/ingest.py` first."
        )


def _configure_concurrency(settings: Settings) -> None:
    """Stop the two thread pools in this process from multiplying.

    Two pools size themselves independently and compose badly. anyio's worker
    pool defaults to 40 threads on the assumption that each one blocks on I/O
    and mostly waits; every task this application sends there is instead
    CPU-bound torch or Lance work. torch then defaults to one intra-op thread
    per core. On the 8-core reference machine that is up to 40 x 8 = 320
    runnable threads competing for 8 cores, where each individual query is
    *slower* than it would be with a smaller pool.

    It is invisible in single-user localhost testing — the usage this tool was
    built for — and shows up the moment anyone points a benchmark at it, which
    is a thing open-sourcing invites.

    The fix is to make each request cheap and let the pool bound concurrency:
    one intra-op thread per forward pass, and a pool near the core count, so
    excess load queues instead of thrashing. Both are overridable for the
    single-user machine running one large batch, where the opposite is right.

    Args:
        settings: Resolved configuration supplying both bounds.
    """
    if settings.torch_num_threads is not None:
        torch.set_num_threads(settings.torch_num_threads)

    if settings.worker_threads is not None:
        # RunVar-backed, so it must be set from inside the event loop — which
        # the lifespan is. Setting it at import time would silently not apply.
        anyio.to_thread.current_default_thread_limiter().total_tokens = settings.worker_threads

    LOGGER.info(
        "torch %s, %d intra-op thread(s), %d worker thread(s)",
        torch.__version__,
        torch.get_num_threads(),
        anyio.to_thread.current_default_thread_limiter().total_tokens,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create resources before the first request and release them after the last.

    The loads below are synchronous and deliberately not offloaded to a worker
    thread: no request is served until this function reaches its ``yield``, so
    there is no event loop to keep responsive yet, and blocking here is what
    makes a startup failure observable as a failed startup.

    Args:
        app: The application whose ``state`` will hold the resources.

    Yields:
        Control to the server for the lifetime of the process.
    """
    settings = get_settings()
    _verify_data_layout(settings)

    _configure_concurrency(settings)

    # The two lines that choose implementations. Everything downstream is typed
    # against `EmbeddingService` and `VectorRepository`, so swapping either — a
    # different encoder, Qdrant instead of LanceDB — is confined to here.
    embedder = ClipEmbeddingService.load(settings.clip_model_id, settings.torch_device)
    repository = LanceDBImageRepository.open(
        settings.lancedb_dir,
        settings.lancedb_table_name,
        nprobes=settings.search_nprobes,
        refine_factor=settings.search_refine_factor,
        exact_scan_ceiling=settings.exact_scan_ceiling,
    )
    projection = ProjectionRepository.load(settings.projection_path)
    analysis = AnalysisRepository.load(settings.analysis_path)

    # Warms the repository's own split cache, so the first dashboard request
    # does not pay for the whole-corpus scan. The cache lives in the
    # implementation rather than here on purpose: `deps.py` injects the
    # repository precisely so that overriding it reaches every service, and a
    # copy of its data hanging off this bundle would route around that.
    LOGGER.info("Cached ground-truth splits for %d image(s)", len(repository.split_by_id()))

    # Seeded from the splits actually in the index, so a --limit'ed ingestion
    # never offers a collection that can only ever be empty.
    collections = CollectionRepository.open(
        settings.collections_path, repository.count_by_split().keys()
    )

    # setattr rather than `app.state.resources = ...` so the attribute name
    # lives in one constant shared with the dependency that reads it back.
    setattr(
        app.state,
        RESOURCES_ATTRIBUTE,
        AppResources(
            settings=settings,
            embedder=embedder,
            image_repository=repository,
            projection_repository=projection,
            analysis_repository=analysis,
            collection_repository=collections,
        ),
    )
    LOGGER.info("Startup complete — serving images from %s", settings.images_dir)

    try:
        yield
    finally:
        repository.close()
        LOGGER.info("Shutdown complete")
