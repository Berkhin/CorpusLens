"""Endpoints for the user's own partition of the corpus.

HTTP concerns only (CLAUDE.md §4.1): validate input, call exactly one service,
map the result onto a response model, pick a status code. The overlay semantics
live in :mod:`app.services.collection_service`; the storage in
:mod:`app.repositories.collection_repository`.

These are the API's only mutating endpoints. They write to a small store this
process owns — never to the corpus index, whose ``split`` column stays the
immutable ground truth every offline measurement is derived from.
"""

from __future__ import annotations

from typing import Annotated, Any, Final

from fastapi import APIRouter, HTTPException, Path, status

from app.api.deps import CollectionServiceDep, FilterResolverDep
from app.exceptions import (
    BuiltinCollectionError,
    CollectionMoveTooLargeError,
    CollectionNotFoundError,
    DuplicateCollectionNameError,
)
from app.models.schemas import (
    CollectionCreateRequest,
    CollectionId,
    CollectionMoveRequest,
    CollectionMoveResponse,
    CollectionRenameRequest,
    CollectionResponse,
    ErrorResponse,
    ImageId,
)

router = APIRouter(prefix="/collections", tags=["collections"])

#: Error bodies shared by the routes that address one collection by id.
_LOOKUP_RESPONSES: Final[dict[int | str, dict[str, Any]]] = {
    status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
    status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
}


def _not_found(error: CollectionNotFoundError) -> HTTPException:
    """Translate a missing collection into a 404."""
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))


def _forbidden(error: BuiltinCollectionError) -> HTTPException:
    """Translate an attempt to edit a built-in into a 403.

    403 rather than 400: the request is well-formed and the collection exists —
    it is the *authority* to change it that is missing, because it mirrors a
    dataset split rather than something the user made.
    """
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(error))


def _conflict(error: DuplicateCollectionNameError) -> HTTPException:
    """Translate a name collision into a 409."""
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))


def _too_large(error: CollectionMoveTooLargeError) -> HTTPException:
    """Translate an over-sized move into a 413.

    413 rather than 400 or 422: the request is well-formed and every field is
    valid — it is the *size* of what it addresses that is refused.
    """
    return HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(error))


@router.get(
    "",
    response_model=list[CollectionResponse],
    summary="List collections with their current sizes",
)
async def list_collections(service: CollectionServiceDep) -> list[CollectionResponse]:
    """Return every collection, built-ins first.

    Sizes reflect moves, which is why the filter bar reads them from here rather
    than from ``images_by_split`` — that one deliberately never moves.
    """
    collections = await service.list_collections()
    return [CollectionResponse.from_domain(collection) for collection in collections]


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a collection",
    responses={status.HTTP_409_CONFLICT: {"model": ErrorResponse}},
)
async def create_collection(
    service: CollectionServiceDep, payload: CollectionCreateRequest
) -> CollectionResponse:
    """Create an empty user collection.

    Raises:
        HTTPException: 409 when the name is already taken, compared
            case-insensitively against built-ins as well as user collections.
    """
    try:
        collection_id = await service.create(payload.name)
    except DuplicateCollectionNameError as error:
        raise _conflict(error) from error

    created = next(
        (item for item in await service.list_collections() if item.id == collection_id), None
    )
    if created is None:  # pragma: no cover — only reachable if deleted mid-request.
        raise _not_found(CollectionNotFoundError(collection_id))
    return CollectionResponse.from_domain(created)


@router.patch(
    "/{collection_id}",
    response_model=CollectionResponse,
    summary="Rename a collection",
    responses={**_LOOKUP_RESPONSES, status.HTTP_409_CONFLICT: {"model": ErrorResponse}},
)
async def rename_collection(
    service: CollectionServiceDep,
    payload: CollectionRenameRequest,
    collection_id: Annotated[CollectionId, Path(description="Collection to rename.")],
) -> CollectionResponse:
    """Rename a user collection.

    Raises:
        HTTPException: 404 when unknown, 403 for a built-in, 409 on a duplicate
            name.
    """
    try:
        await service.rename(collection_id, payload.name)
    except CollectionNotFoundError as error:
        raise _not_found(error) from error
    except BuiltinCollectionError as error:
        raise _forbidden(error) from error
    except DuplicateCollectionNameError as error:
        raise _conflict(error) from error

    renamed = next(
        (item for item in await service.list_collections() if item.id == collection_id), None
    )
    if renamed is None:  # pragma: no cover — only reachable if deleted mid-request.
        raise _not_found(CollectionNotFoundError(collection_id))
    return CollectionResponse.from_domain(renamed)


@router.delete(
    "/{collection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a collection, returning its images to their splits",
    responses=_LOOKUP_RESPONSES,
)
async def delete_collection(
    service: CollectionServiceDep,
    collection_id: Annotated[CollectionId, Path(description="Collection to delete.")],
) -> None:
    """Delete a user collection.

    Its members are not deleted — they revert to their ground-truth split, which
    is the correct undo and comes free from the store's cascade.

    Raises:
        HTTPException: 404 when unknown, 403 for a built-in.
    """
    try:
        await service.delete(collection_id)
    except CollectionNotFoundError as error:
        raise _not_found(error) from error
    except BuiltinCollectionError as error:
        raise _forbidden(error) from error


@router.post(
    "/{collection_id}/images",
    response_model=CollectionMoveResponse,
    summary="Move images into a collection, by id or by filter",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
    },
)
async def move_images(
    service: CollectionServiceDep,
    payload: CollectionMoveRequest,
    resolve_filter: FilterResolverDep,
    collection_id: Annotated[CollectionId, Path(description="Destination collection.")],
) -> CollectionMoveResponse:
    """Reassign images to a collection, named either explicitly or by filter.

    Moving to a built-in is allowed and is how an image is put back: the target
    being the image's own split clears the override rather than storing one.

    The filter is resolved through the same ``FilterResolverDep`` every listing
    endpoint uses, so "move everything matching what I am looking at" moves
    exactly the set that filter lists — quality flags and collection membership
    included, neither of which is a property of a stored row.

    Raises:
        HTTPException: 404 when the destination collection does not exist, 413
            when the move would re-assign more images than the ceiling allows.
    """
    try:
        result = (
            await service.move_images(collection_id, payload.ids, origin=payload.origin)
            if payload.filter is None
            else await service.move_matching(
                collection_id,
                resolve_filter(payload.filter.to_filter()),
                payload.filter.describe(),
            )
        )
    except CollectionNotFoundError as error:
        raise _not_found(error) from error
    except CollectionMoveTooLargeError as error:
        raise _too_large(error) from error

    moved, unchanged, unknown = result
    return CollectionMoveResponse(moved=moved, unchanged=unchanged, unknown=unknown)


@router.delete(
    "/{collection_id}/images/{image_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove an image's override, returning it to its split",
)
async def reset_image(
    service: CollectionServiceDep,
    collection_id: Annotated[CollectionId, Path(description="Collection to remove it from.")],
    image_id: Annotated[ImageId, Path(description="Image to reset.")],
) -> None:
    """Drop one image's override.

    ``collection_id`` is part of the path for REST shape — the override is
    unique per image, so the image id alone identifies what to delete. Removing
    an image that was not in that collection is therefore a no-op rather than an
    error, which keeps the button idempotent under a double click.
    """
    await service.reset_image(image_id)
