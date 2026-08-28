"""Manifest download endpoint.

POST rather than GET because the request carries a selection: a box drawn on the
projection can be thousands of ids, which no URL should have to hold. The reply
is a file, so this is also the one route that returns something other than JSON.
"""

from __future__ import annotations

from typing import Final

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import ExportServiceDep, FilterResolverDep
from app.models.domain import ExportFormat
from app.models.schemas import ExportRequest

router = APIRouter(tags=["export"])

#: Media types per format. ``application/x-ndjson`` is the registered type for
#: newline-delimited JSON; ``application/json`` would be wrong, since the body
#: is a sequence of documents rather than one.
_MEDIA_TYPES: Final[dict[ExportFormat, str]] = {
    "csv": "text/csv; charset=utf-8",
    "jsonl": "application/x-ndjson; charset=utf-8",
}

_FILENAME_STEM: Final = "corpuslens-export"


@router.post(
    "/export",
    summary="Download the current selection as CSV or JSONL",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/csv": {}, "application/x-ndjson": {}},
            "description": "The manifest, streamed as it is generated.",
        }
    },
)
async def export_images(
    payload: ExportRequest,
    service: ExportServiceDep,
    resolve_filter: FilterResolverDep,
) -> StreamingResponse:
    """Stream a manifest of the selected images.

    The response is generated lazily rather than buffered: a whole-corpus export
    is 8 000 records with five captions each, and there is no reason to hold all
    of it in memory before the first byte goes out.
    """
    stream = service.stream(
        export_format=payload.format,
        image_filter=resolve_filter(payload.to_filter()),
        image_ids=payload.ids,
        query=payload.query,
        similar_to_image_id=payload.image_id,
        limit=payload.limit,
    )
    return StreamingResponse(
        stream,
        media_type=_MEDIA_TYPES[payload.format],
        headers={
            # Suggested only: the browser client names the file itself, which
            # keeps this header out of the CORS `expose_headers` list. It is set
            # anyway so that curl -OJ and the OpenAPI "try it out" button behave.
            "Content-Disposition": f'attachment; filename="{_FILENAME_STEM}.{payload.format}"'
        },
    )
