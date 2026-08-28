"""HTTP route modules, aggregated into a single API router."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import collections, dataset, export, projection, search

api_router = APIRouter()
api_router.include_router(dataset.router)
api_router.include_router(search.router)
api_router.include_router(export.router)
api_router.include_router(projection.router)
api_router.include_router(collections.router)

__all__ = ["api_router"]
