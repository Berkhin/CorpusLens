"""Embedding-map endpoint.

Returns the entire point cloud in one response. Unpaged on purpose: a scatter
plot missing a page is not a smaller plot, it is a wrong one. The payload is
large by this API's standards, which is why ``GZipMiddleware`` is installed in
``app.main``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import ImageFilterDep, ProjectionServiceDep
from app.exceptions import ProjectionUnavailableError
from app.models.schemas import ErrorResponse, ProjectionResponse

router = APIRouter(tags=["projection"])


@router.get(
    "/projection",
    response_model=ProjectionResponse,
    summary="2-D projection of every image embedding",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def get_projection(
    service: ProjectionServiceDep,
    image_filter: ImageFilterDep,
) -> ProjectionResponse:
    """Return the map of the corpus in CLIP space.

    Points outside the filter come back too, marked ``matches=false``, so the
    client can dim rather than drop them — where a subset sits relative to the
    rest of the corpus is the question this view exists to answer.

    Raises:
        HTTPException: 404 when no projection artefact exists. A 404 rather
            than a 503: the resource genuinely is not there, and the message
            names the command that creates it.
    """
    try:
        projection = await service.get_projection(image_filter)
    except ProjectionUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    return ProjectionResponse.from_domain(projection)
