"""Semantic search: encode a text query, then rank images against it.

This is the one place the API runs CLIP inference. CLAUDE.md §2 permits it as
the single exception to "embedding is an offline batch job": encoding a short
string is a ~tens-of-milliseconds forward pass over a handful of tokens,
nothing like the ~15 minutes an 8k-image pass costs on this CPU.

Framework-agnostic (CLAUDE.md §4.1) — no FastAPI import anywhere below.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import partial
from typing import Final

import anyio.to_thread

from app.exceptions import ImageNotFoundError
from app.models.domain import ImageFilter, SearchHit
from app.repositories.vector_db import VectorRepository
from app.services.collection_service import CollectionService
from app.services.embedding import EmbeddingService

LOGGER: Final = logging.getLogger(__name__)


class SearchService:
    """Text→image retrieval over the pre-built CLIP index.

    Named against two Protocols and no implementation: it neither knows which
    store holds the vectors nor which model produced them, only that both speak
    the same space.
    """

    def __init__(
        self,
        repository: VectorRepository,
        embedder: EmbeddingService,
        collections: CollectionService,
    ) -> None:
        """Bind the service to its index and encoder.

        Args:
            repository: Source of vectors to rank.
            embedder: Projects a query into the space the corpus was embedded
                in. Loaded once at application startup, never per request.
            collections: The user's partition overlay, used to stamp each hit's
                effective collection.
        """
        self._repository = repository
        self._embedder = embedder
        self._collections = collections

    async def _decorate(self, hits: list[SearchHit]) -> list[SearchHit]:
        """Stamp the effective collection onto each hit's image.

        Args:
            hits: Ranked results straight from the repository.

        Returns:
            The same hits with ``image.collection`` resolved.
        """
        overlay = await self._collections.overlay()
        return [replace(hit, image=overlay.decorate_detail(hit.image)) for hit in hits]

    async def search(
        self,
        query: str,
        limit: int,
        image_filter: ImageFilter | None = None,
    ) -> list[SearchHit]:
        """Rank images by similarity to a natural-language query.

        Args:
            query: Free-text description; already trimmed and length-checked by
                the request schema.
            limit: Maximum hits to return.
            image_filter: Restricts the candidate set before ranking, so
                "the best 20 matches in the test split" means exactly that.

        Returns:
            Hits ordered by decreasing cosine similarity, possibly empty if the
            index is empty or nothing survives the filter.
        """
        # Offloaded because the forward pass is blocking CPU work and the
        # contract in `EmbeddingService` is deliberately synchronous — deciding
        # how to get it off the event loop is this layer's job, not the model's.
        vector = await anyio.to_thread.run_sync(self._embedder.embed_text, query)
        hits = await anyio.to_thread.run_sync(
            partial(self._repository.search_by_vector, vector, limit, image_filter)
        )
        LOGGER.debug("Query %r returned %d hit(s)", query, len(hits))
        return await self._decorate(hits)

    async def search_by_image(
        self,
        image_id: str,
        limit: int,
        image_filter: ImageFilter | None = None,
    ) -> list[SearchHit]:
        """Rank images by similarity to another image in the corpus.

        The cheapest operation this API offers: the query image's embedding was
        computed during ingestion, so this runs **no inference at all** — just a
        keyed read and the same brute-force scan a text search ends with.

        Args:
            image_id: The image to find neighbours of.
            limit: Maximum hits to return.
            image_filter: Restricts the candidate set before ranking.

        Returns:
            Hits ordered by decreasing similarity, **excluding the query image
            itself**. Returning it would waste a result slot on the trivially
            perfect match; one extra candidate is requested so the caller still
            gets the number of neighbours it asked for.

        Raises:
            ImageNotFoundError: If no row carries that id.
        """
        vector = await anyio.to_thread.run_sync(self._repository.get_vector_by_id, image_id)
        if vector is None:
            raise ImageNotFoundError(image_id)

        hits = await anyio.to_thread.run_sync(
            partial(self._repository.search_by_vector, vector, limit + 1, image_filter)
        )
        neighbours = [hit for hit in hits if hit.image.id != image_id][:limit]
        LOGGER.debug("Image %r returned %d neighbour(s)", image_id, len(neighbours))
        return await self._decorate(neighbours)
