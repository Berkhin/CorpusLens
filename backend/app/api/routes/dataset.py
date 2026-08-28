"""Dataset browsing endpoints.

HTTP concerns only (CLAUDE.md §4.1): validate input, call exactly one service,
map the domain result onto a response model, pick a status code. No LanceDB, no
filesystem, no ranking logic.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, status

from app.api.deps import DatasetServiceDep, ImageFilterDep, ResourcesDep, SettingsDep
from app.exceptions import ImageNotFoundError
from app.models.schemas import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    DatasetStatsResponse,
    ErrorResponse,
    ImageId,
    ImagePageResponse,
    InspectedImageResponse,
)

router = APIRouter(prefix="/dataset", tags=["dataset"])


# Registered before "/{image_id}": FastAPI matches routes in declaration order,
# so a dynamic segment declared first would swallow "stats" as an id.
@router.get(
    "/stats",
    response_model=DatasetStatsResponse,
    summary="Corpus-level counts",
)
async def get_dataset_stats(
    service: DatasetServiceDep,
    resources: ResourcesDep,
) -> DatasetStatsResponse:
    """Return the corpus counts, plus what the client needs to render controls.

    The availability flags ride along here rather than living on their own
    endpoint so the client learns them in a request it already makes on load,
    and can render the navigation correctly the first time instead of
    discovering a missing artefact through an empty view.
    """
    stats = await service.get_stats()
    return DatasetStatsResponse.from_domain(
        stats,
        projection_available=resources.projection_repository is not None,
        analysis_available=resources.analysis_repository is not None,
    )


@router.get(
    "",
    response_model=ImagePageResponse,
    summary="List images (paginated)",
)
async def list_images(
    service: DatasetServiceDep,
    settings: SettingsDep,
    image_filter: ImageFilterDep,
    offset: Annotated[int, Query(ge=0, description="Rows to skip.")] = 0,
    limit: Annotated[
        int,
        Query(ge=1, le=MAX_PAGE_SIZE, description="Maximum rows to return."),
    ] = DEFAULT_PAGE_SIZE,
) -> ImagePageResponse:
    """Return one page of image summaries plus the matching and corpus totals.

    ``offset`` walks the *filtered* sequence, so paging stays consistent while a
    filter is active.
    """
    page = await service.list_images(offset=offset, limit=limit, image_filter=image_filter)
    return ImagePageResponse.from_domain(page, settings.images_url_prefix)


@router.get(
    "/{image_id}",
    response_model=InspectedImageResponse,
    summary="Image detail with all reference captions and quality measurements",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def get_image(
    service: DatasetServiceDep,
    settings: SettingsDep,
    image_id: Annotated[ImageId, Path(description="Flickr photo id.")],
) -> InspectedImageResponse:
    """Return a single image with its captions and, if computed, its analysis.

    Raises:
        HTTPException: 404 when the id is not in the index. This is the only
            layer that speaks HTTP; the service raises a domain error, and its
            own message is reused rather than composed a second time here.
    """
    try:
        inspected = await service.get_image(image_id)
    except ImageNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    return InspectedImageResponse.from_domain(inspected, settings.images_url_prefix)
