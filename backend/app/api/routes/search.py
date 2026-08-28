"""Semantic search endpoint.

POST rather than GET: the query is a body-shaped input — it already carries the
corpus filter alongside the text, and would grow an image→image field next — and
a GET would invite caching of results that depend on the index rather than the
URL alone.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import FilterResolverDep, SearchServiceDep, SettingsDep
from app.exceptions import ImageNotFoundError
from app.models.schemas import ErrorResponse, SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search over the CLIP index, by text or by example image",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def search_images(
    payload: SearchRequest,
    service: SearchServiceDep,
    settings: SettingsDep,
    resolve_filter: FilterResolverDep,
) -> SearchResponse:
    """Rank images by CLIP similarity to a query or to another image.

    The payload carries exactly one target, enforced by the schema. Text costs a
    forward pass through CLIP's text encoder; an example image costs none at
    all, because its embedding is already in the index.

    Any filter narrows the candidate set *before* ranking, so the response holds
    the best ``limit`` matches within the filter rather than the survivors of
    the best ``limit`` matches overall.

    Raises:
        HTTPException: 404 when ``image_id`` names an image the index does not
            hold.
    """
    image_filter = resolve_filter(payload.to_filter())
    if payload.image_id is not None:
        try:
            hits = await service.search_by_image(payload.image_id, payload.limit, image_filter)
        except ImageNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        return SearchResponse.from_domain(
            f"similar to {payload.image_id}", hits, settings.images_url_prefix
        )

    # The schema guarantees one target is set, so this branch has a query.
    query = payload.query or ""
    hits = await service.search(query, payload.limit, image_filter)
    return SearchResponse.from_domain(query, hits, settings.images_url_prefix)
