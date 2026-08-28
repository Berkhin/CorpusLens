"""Tests for the seams themselves, rather than for what runs through them.

The claim this module defends is the one the layering exists to support: an
alternative vector store or an alternative encoder can be dropped in **without
touching a router, a service, or a model**. That is easy to assert in a README
and easy to break silently, so it is exercised here instead — the doubles below
inherit from nothing and import no storage library, exactly as a third-party
``QdrantRepository`` in a separate distribution would.

If a future change makes a service reach past
:class:`~app.repositories.vector_db.VectorRepository` for a LanceDB-specific
method, these tests fail while the rest of the suite — which runs against a
double wired in at the *lifespan*, one layer lower — would still pass.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final, cast

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from app.api.deps import get_embedding_service, get_vector_repository
from app.models.domain import ImageDetail, ImageFilter, ImageSummary, SearchHit
from app.repositories.image_repository import LanceDBImageRepository
from app.repositories.vector_db import VectorRepository
from app.services.embedding import ClipEmbeddingService, EmbeddingService

#: An id that exists in no fixture, so a response carrying it can only have come
#: from the substitute store.
ALIEN_ID: Final = "not-a-flickr-id"
ALIEN_DIMENSIONS: Final = 3


class InMemoryVectorRepository:
    """A complete vector store in fifty lines, inheriting from nothing.

    Stands in for the hypothetical ``QdrantRepository``: it satisfies
    :class:`VectorRepository` structurally, and the application accepts it
    without a registration step, a base class, or an import in either direction.
    """

    def __init__(self, records: list[ImageDetail]) -> None:
        """Hold the corpus this store will serve.

        Args:
            records: Rows in scan order.
        """
        self._records = records
        self.closed = False

    def _summary(self, record: ImageDetail) -> ImageSummary:
        """Project a record down to its summary form."""
        return ImageSummary(
            id=record.id,
            file_name=record.file_name,
            split=record.split,
            collection=record.collection,
        )

    def count(self, image_filter: ImageFilter | None = None) -> int:
        """Return the corpus size, ignoring filters this double need not model."""
        return len(self._records)

    def count_by_split(self) -> dict[str, int]:
        """Tally the records by split."""
        counts: dict[str, int] = {}
        for record in self._records:
            counts[record.split] = counts.get(record.split, 0) + 1
        return counts

    def split_by_id(self) -> dict[str, str]:
        """Map every id to its split."""
        return {record.id: record.split for record in self._records}

    def list_ids(self, image_filter: ImageFilter | None, *, limit: int) -> list[str]:
        """Return ids in scan order, up to ``limit``."""
        return [record.id for record in self._records][: max(limit, 0)]

    def list_summaries(
        self, *, offset: int, limit: int, image_filter: ImageFilter | None = None
    ) -> list[ImageSummary]:
        """Return one page of summaries."""
        return [self._summary(record) for record in self._records[offset : offset + limit]]

    def list_details(
        self, *, offset: int, limit: int, image_filter: ImageFilter | None = None
    ) -> list[ImageDetail]:
        """Return one page of records."""
        return self._records[offset : offset + limit]

    def get_by_id(self, image_id: str) -> ImageDetail | None:
        """Look one record up by id."""
        return next((record for record in self._records if record.id == image_id), None)

    def get_many_by_id(self, image_ids: Sequence[str]) -> dict[str, ImageDetail]:
        """Look several records up at once."""
        wanted = set(image_ids)
        return {record.id: record for record in self._records if record.id in wanted}

    def get_vector_by_id(self, image_id: str) -> NDArray[np.float32] | None:
        """Return a fixed vector for any known id."""
        if self.get_by_id(image_id) is None:
            return None
        return np.ones(ALIEN_DIMENSIONS, dtype=np.float32)

    def search_by_vector(
        self,
        vector: NDArray[np.float32],
        limit: int,
        image_filter: ImageFilter | None = None,
    ) -> list[SearchHit]:
        """Return every record, perfectly scored, in scan order."""
        return [SearchHit(image=record, score=1.0) for record in self._records[:limit]]

    def close(self) -> None:
        """Record that teardown reached this store."""
        self.closed = True


class StubEmbeddingService:
    """An encoder with no model behind it, satisfying :class:`EmbeddingService`."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.embedded: list[str] = []

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Record the query and return a fixed unit vector."""
        self.embedded.append(text)
        return np.ones(ALIEN_DIMENSIONS, dtype=np.float32)

    def embed_image(self, image_bytes: bytes) -> NDArray[np.float32]:
        """Return a fixed unit vector; unused by any route."""
        return np.ones(ALIEN_DIMENSIONS, dtype=np.float32)


def overrides(client: TestClient) -> dict[Callable[..., Any], Callable[..., Any]]:
    """Return the override map of the application behind a test client.

    ``TestClient.app`` is annotated as a bare ASGI callable, which has no
    ``dependency_overrides``. The object really is a ``FastAPI`` — the ``client``
    fixture builds it with ``create_app`` — so the narrowing is safe and is done
    here once rather than at every call site.
    """
    return cast(FastAPI, client.app).dependency_overrides


@pytest.fixture
def alien_records() -> list[ImageDetail]:
    """A one-record corpus that shares no id with the LanceDB fixtures."""
    return [
        ImageDetail(
            id=ALIEN_ID,
            file_name=f"{ALIEN_ID}.jpg",
            split="train",
            collection="train",
            captions=["a record served by a store that is not LanceDB"],
        )
    ]


def test_the_reference_implementations_satisfy_their_protocols() -> None:
    """The shipped classes are checked against the contracts they claim.

    ``isinstance`` on a ``runtime_checkable`` Protocol only verifies that the
    attributes exist — signatures are mypy's job, and
    ``image_repository._assert_conformance`` is where that is pinned. This
    catches the coarser regression: a method renamed or dropped outright.
    """
    assert isinstance(LanceDBImageRepository(table=None), VectorRepository)
    assert isinstance(ClipEmbeddingService(model=None, device="cpu"), EmbeddingService)
    assert isinstance(InMemoryVectorRepository([]), VectorRepository)
    assert isinstance(StubEmbeddingService(), EmbeddingService)


def test_the_vector_store_can_be_swapped_without_touching_a_route(
    client: TestClient, alien_records: list[ImageDetail]
) -> None:
    """A foreign store reaches the wire through the unmodified routing layer.

    This is the ``QdrantRepository`` scenario in miniature. Overriding the one
    provider re-points every service built for the request, so the assertion
    that matters is that the *response body* changes while no application code
    does.
    """
    store = InMemoryVectorRepository(alien_records)
    overrides(client)[get_vector_repository] = lambda: store

    try:
        page = client.get("/api/dataset")
        detail = client.get(f"/api/dataset/{ALIEN_ID}")
    finally:
        overrides(client).clear()

    assert page.status_code == 200
    assert [item["id"] for item in page.json()["items"]] == [ALIEN_ID]
    assert detail.status_code == 200
    assert detail.json()["image"]["captions"] == ["a record served by a store that is not LanceDB"]


def test_the_encoder_can_be_swapped_without_touching_a_route(
    client: TestClient, alien_records: list[ImageDetail]
) -> None:
    """Search runs end to end against an encoder that loads no weights.

    Both seams are overridden at once because a search exercises both, and
    doing so proves they are genuinely independent injection points rather than
    one dependency wearing two names.
    """
    encoder = StubEmbeddingService()
    store = InMemoryVectorRepository(alien_records)
    overrides(client)[get_vector_repository] = lambda: store
    overrides(client)[get_embedding_service] = lambda: encoder

    try:
        response = client.post("/api/search", json={"query": "anything at all"})
    finally:
        overrides(client).clear()

    assert response.status_code == 200
    assert encoder.embedded == ["anything at all"]
    assert [hit["image"]["id"] for hit in response.json()["results"]] == [ALIEN_ID]


def test_overriding_one_seam_leaves_the_other_in_place(
    client: TestClient, alien_records: list[ImageDetail], fake_clip_model: object
) -> None:
    """Swapping the store must not quietly swap the encoder too.

    Guards against a refactor that collapses both providers onto one object:
    with only the repository overridden, the real ``ClipEmbeddingService`` from
    the lifespan must still be the thing doing the encoding.
    """
    store = InMemoryVectorRepository(alien_records)
    overrides(client)[get_vector_repository] = lambda: store

    try:
        response = client.post("/api/search", json={"query": "still the real encoder"})
    finally:
        overrides(client).clear()

    assert response.status_code == 200
    # The lifespan's encoder recorded the call, so it — not a stub — ran.
    calls = fake_clip_model.encode_calls  # type: ignore[attr-defined] # test double
    assert [call["sentences"] for call in calls] == ["still the real encoder"]
