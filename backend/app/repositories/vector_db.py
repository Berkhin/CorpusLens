"""The contract every vector store must satisfy to back this API.

This module is the seam between the application and whatever holds the vectors.
It imports no storage library — only domain types — so swapping LanceDB for
Qdrant, pgvector or an in-memory double means writing one class that satisfies
:class:`VectorRepository` and changing one line in
:mod:`app.core.lifespan`. Nothing in ``services/`` or ``api/`` changes, because
nothing there names an implementation.

**Why a Protocol and not an ABC.** Structural typing means an implementation
does not have to import this package or inherit from anything: an adapter
written in a separate distribution satisfies the contract by shape alone, and
the test doubles in ``tests/conftest.py`` satisfy it without a registration
step. An ABC would buy runtime enforcement that mypy already gives statically,
at the cost of making every implementer depend on us. The one thing a Protocol
does not do is check an implementation *eagerly* — a class only fails to
type-check where it is passed as a ``VectorRepository`` — so implementations in
this repository assert their own conformance; see the bottom of
:mod:`app.repositories.image_repository`.

**Every method here is blocking.** Vector search is CPU work and the reference
implementation's client is synchronous, so service-layer callers push these onto
a worker thread rather than running them on the event loop. Making the contract
async would force that decision on implementations that have nothing to await.

**Read-only, deliberately.** There is no ``insert`` or ``upsert``: the index is
built offline by ``scripts/ingest.py`` and the API is a pure reader of it
(CLAUDE.md §4.2). That is what keeps the ``split`` column trustworthy ground
truth for cross-split leakage analysis. Mutable state the API owns — the user's
collection overlay — lives in a separate store behind its own repository, and
any future writable state must follow the same rule rather than widening this
interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from app.models.domain import ImageDetail, ImageFilter, ImageSummary, SearchHit


@runtime_checkable
class VectorRepository(Protocol):
    """Read access to a corpus of images, their captions and their embeddings.

    ``runtime_checkable`` so a composition root can assert conformance at
    startup if it wants to. Note the standard caveat: ``isinstance`` against a
    Protocol checks that the attributes *exist*, never that their signatures
    match. Static checking remains the real guarantee.
    """

    def count(self, image_filter: ImageFilter | None = None) -> int:
        """Return the number of images matching a filter.

        Args:
            image_filter: Narrowing to apply, or ``None`` for the whole corpus.

        Returns:
            Matching row count.
        """
        ...

    def count_by_split(self) -> dict[str, int]:
        """Return the number of images per ground-truth split.

        Returns:
            Split name to row count. Splits absent from the index are absent
            from the mapping — a ``--limit``ed ingestion legitimately holds only
            ``train``.
        """
        ...

    def split_by_id(self) -> dict[str, str]:
        """Return every image's ground-truth split, keyed by id.

        The default half of every image's collection membership: the overlay
        store knows only the overrides, so it needs this to resolve the rest.

        Returns:
            Image id to split name, for the whole corpus.
        """
        ...

    def list_ids(self, image_filter: ImageFilter | None, *, limit: int) -> list[str]:
        """Return the ids of the images a filter selects.

        Args:
            image_filter: Narrowing to apply, or ``None`` for the whole corpus.
            limit: Maximum ids to return. Required rather than defaulted so an
                accidental whole-corpus read cannot become an 8 000-entry ``IN``
                list downstream.

        Returns:
            Matching ids in scan order; empty when ``limit`` is not positive.
        """
        ...

    def list_summaries(
        self,
        *,
        offset: int,
        limit: int,
        image_filter: ImageFilter | None = None,
    ) -> list[ImageSummary]:
        """Return one page of image summaries.

        Args:
            offset: Rows to skip.
            limit: Maximum rows to return.
            image_filter: Narrowing applied before paging, so ``offset`` walks
                the filtered sequence rather than the whole corpus.

        Returns:
            Summaries for the requested window, in a stable order.
        """
        ...

    def list_details(
        self,
        *,
        offset: int,
        limit: int,
        image_filter: ImageFilter | None = None,
    ) -> list[ImageDetail]:
        """Return one page of records with their captions.

        Args:
            offset: Rows to skip.
            limit: Maximum rows to return.
            image_filter: Narrowing applied before paging.

        Returns:
            Detail records for the requested window.
        """
        ...

    def get_by_id(self, image_id: str) -> ImageDetail | None:
        """Fetch one image with its captions.

        Args:
            image_id: Corpus-unique image id.

        Returns:
            The record, or ``None`` if nothing carries that id.
        """
        ...

    def get_many_by_id(self, image_ids: Sequence[str]) -> dict[str, ImageDetail]:
        """Fetch several records at once, keyed by id.

        A mapping rather than a list because the store returns scan order while
        the caller usually needs its own order preserved.

        Args:
            image_ids: Ids to look up; an empty sequence returns an empty map.

        Returns:
            Id to record, for every id that exists. Missing ids are simply
            absent, which is what lets the caller detect them.
        """
        ...

    def get_vector_by_id(self, image_id: str) -> NDArray[np.float32] | None:
        """Fetch one image's stored embedding.

        This is what makes image-to-image search free of inference: the vector
        was computed during ingestion.

        Args:
            image_id: Corpus-unique image id.

        Returns:
            The unit-length embedding, or ``None`` if nothing carries that id.
        """
        ...

    def search_by_vector(
        self,
        vector: NDArray[np.float32],
        limit: int,
        image_filter: ImageFilter | None = None,
    ) -> list[SearchHit]:
        """Rank images by cosine similarity to a query vector.

        Args:
            vector: Unit-length query embedding in the same space the corpus was
                embedded in.
            limit: Maximum hits to return.
            image_filter: Narrowing applied **before** ranking. Implementations
                must pre-filter, not post-filter: taking the global top ``limit``
                and then discarding would silently return fewer results than
                asked for, with the tail of the ranking missing.

        Returns:
            Hits ordered by decreasing similarity, scored so that 1.0 is
            identical — implementations converting from a distance metric must
            invert it.
        """
        ...

    def close(self) -> None:
        """Release any handle the store holds.

        Called once by the lifespan on shutdown. Implementations with nothing to
        release should still provide it and make it idempotent.
        """
        ...
