"""Dependency wiring for the route layer.

Services are constructed per request from the singletons the lifespan owns.
They are stateless wrappers around those singletons, so this costs an object
allocation and buys constructor injection — no module-level globals outside the
lifespan context (CLAUDE.md §4.1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Annotated

from fastapi import Depends, Query, Request

from app.core.config import Settings, get_settings
from app.core.lifespan import RESOURCES_ATTRIBUTE, AppResources
from app.models.domain import CollectionSelection, ImageFilter, QualityFlag
from app.models.schemas import CaptionNeedle, CollectionId, SplitName
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.collection_repository import CollectionRepository
from app.repositories.vector_db import VectorRepository
from app.services.collection_service import CollectionService
from app.services.dataset_service import DatasetService
from app.services.embedding import EmbeddingService
from app.services.export_service import ExportService
from app.services.projection_service import ProjectionService
from app.services.search_service import SearchService


def get_resources(request: Request) -> AppResources:
    """Pull the startup-created resource bundle off the application state.

    Args:
        request: The in-flight request, used only to reach ``app.state``.

    Returns:
        The resources created by the lifespan.

    Raises:
        RuntimeError: If the lifespan never ran. This is a programming error —
            typically an app constructed without its ``lifespan`` — not a
            client fault, so it is deliberately not a 4xx.
    """
    resources = getattr(request.app.state, RESOURCES_ATTRIBUTE, None)
    if not isinstance(resources, AppResources):
        raise RuntimeError(
            "Application resources are unavailable; the lifespan did not run. "
            "Construct the app via app.main.create_app()."
        )
    return resources


def resolve_quality_flag(
    image_filter: ImageFilter, analysis: AnalysisRepository | None
) -> ImageFilter:
    """Turn a data-quality flag into the id set it stands for.

    The flag is the one filter dimension that is not a property of a stored
    row — it comes from an offline artefact — so it cannot be compiled into a
    predicate directly. Resolving it here, once, at the edge, means every layer
    below sees an ordinary id restriction and the flag composes with the split
    and caption filters, with pagination, and with export for free.

    A flag requested when no analysis exists resolves to the **empty** set, not
    to "no restriction": the filter is unsatisfiable, and reporting no matches
    is honest where silently ignoring it would not be.

    Args:
        image_filter: Filter as parsed from the request.
        analysis: The loaded analysis, if any.

    Returns:
        An equivalent filter with ``quality_flag`` folded into ``ids``.
    """
    if image_filter.quality_flag is None:
        return image_filter
    selected = frozenset[str]() if analysis is None else analysis.ids_for(image_filter.quality_flag)
    # Sorted so the same request always produces the same predicate string.
    return replace(image_filter, quality_flag=None, ids=tuple(sorted(selected)))


def resolve_collections(
    image_filter: ImageFilter, collections: CollectionRepository
) -> ImageFilter:
    """Turn requested collection ids into something the store can be asked.

    The second filter dimension that is not a property of a row. Unlike the
    quality flag it does **not** compile into ``ids``: that channel is already
    the quality flag's, and whichever resolver ran second would silently
    overwrite the first. A collection resolves into its own
    :class:`~app.models.domain.CollectionSelection` instead, and the two then
    intersect through the ordinary ``AND`` between clauses.

    ``excluded_ids`` is every overridden id whose target is *not* selected —
    including images whose split was never selected either. Those are redundant
    but harmless: ``id NOT IN (…)`` only ever removes rows the split clause
    already admitted. Computing the minimal set would mean asking the index for
    each moved image's split, which is a query this boundary should not make.

    An unknown or deleted collection id contributes nothing, so a filter naming
    only unknown ids resolves to "keep nothing" — unsatisfiable, not ignored,
    exactly as a quality flag with no analysis behaves.

    Args:
        image_filter: Filter as parsed from the request.
        collections: The overlay store.

    Returns:
        An equivalent filter with ``collection_selection`` filled in.
    """
    if not image_filter.collections:
        return image_filter

    selected = set(image_filter.collections)
    kinds = collections.kinds()
    assignments = collections.overlay().assignments

    # A built-in collection's id *is* its split name, which is what lets the
    # selection be matched against the ``split`` column directly.
    split_names = tuple(sorted(name for name in selected if kinds.get(name) == "builtin"))
    moved_in = tuple(
        sorted(image_id for image_id, target in assignments.items() if target in selected)
    )
    excluded = tuple(
        sorted(image_id for image_id, target in assignments.items() if target not in selected)
    )

    return replace(
        image_filter,
        collection_selection=CollectionSelection(
            split_names=split_names, moved_in_ids=moved_in, excluded_ids=excluded
        ),
    )


def resolve_filter(
    image_filter: ImageFilter,
    analysis: AnalysisRepository | None,
    collections: CollectionRepository,
) -> ImageFilter:
    """Fold both non-row filter dimensions in, once, at the edge.

    Args:
        image_filter: Filter as parsed from the request.
        analysis: The loaded analysis, if any.
        collections: The overlay store.

    Returns:
        A filter every layer below can treat as ordinary predicates.
    """
    return resolve_collections(resolve_quality_flag(image_filter, analysis), collections)


def get_filter_resolver(resources: ResourcesDep) -> Callable[[ImageFilter], ImageFilter]:
    """Provide filter resolution to routes whose filter arrives in a body.

    ``GET`` routes get an already-resolved filter from :func:`get_image_filter`.
    ``POST`` routes build theirs from the payload, so they need the resolution
    step as something they can apply — one call, not logic in the route.
    """
    return lambda image_filter: resolve_filter(
        image_filter, resources.analysis_repository, resources.collection_repository
    )


def get_image_filter(
    resources: ResourcesDep,
    split: Annotated[
        list[SplitName] | None,
        Query(description="Repeatable. Restrict to these splits; omit for all."),
    ] = None,
    caption_contains: Annotated[
        CaptionNeedle | None,
        Query(description="Case-insensitive substring that a caption must contain."),
    ] = None,
    quality_flag: Annotated[
        QualityFlag | None,
        Query(description="Restrict to a finding from the offline data-quality pass."),
    ] = None,
    collection: Annotated[
        list[CollectionId] | None,
        Query(description="Repeatable. Restrict to these collections; omit for all."),
    ] = None,
) -> ImageFilter:
    """Assemble the shared corpus filter from query parameters.

    A dependency function rather than a Pydantic model annotated with
    ``Query()``: verified on the installed FastAPI 0.141.1 that a query-params
    *model* stops binding as soon as the route declares any sibling query
    parameter — it degrades into a required scalar named after the argument, so
    ``/dataset?split=train&limit=5`` would 422. Plain parameters compose, and a
    ``list[...]`` parameter still collects a repeated key.

    Returning the domain type directly keeps the translation out of every route
    that filters.

    Args:
        resources: Startup singletons, for the analysis the flag resolves against.
        split: Repeated ``split`` query parameters, if any. The dataset's own
            immutable partition, kept alongside ``collection`` rather than
            replaced by it — one is ground truth, the other is the user's
            working overlay, and a researcher needs to ask about both.
        caption_contains: Substring to require in at least one caption.
        quality_flag: Data-quality finding to narrow to.
        collection: Repeated ``collection`` query parameters, if any.

    Returns:
        The filter to hand to a service, with the quality flag and the
        collections already resolved; empty when no parameter is given.
    """
    return resolve_filter(
        ImageFilter(
            splits=tuple(split or ()),
            caption_contains=caption_contains,
            quality_flag=quality_flag,
            collections=tuple(collection or ()),
        ),
        resources.analysis_repository,
        resources.collection_repository,
    )


ResourcesDep = Annotated[AppResources, Depends(get_resources)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
ImageFilterDep = Annotated[ImageFilter, Depends(get_image_filter)]
FilterResolverDep = Annotated[Callable[[ImageFilter], ImageFilter], Depends(get_filter_resolver)]


def get_vector_repository(resources: ResourcesDep) -> VectorRepository:
    """Provide the corpus index.

    A one-line indirection over the resource bundle, and worth its existence:
    it is the override point. ``app.dependency_overrides[get_vector_repository]``
    swaps the store for the whole application — every service composed below
    receives the substitute — without a monkeypatch or a rebuilt lifespan.
    """
    return resources.image_repository


def get_embedding_service(resources: ResourcesDep) -> EmbeddingService:
    """Provide the query encoder, loaded once at startup.

    The singleton lives in the lifespan-owned bundle; this only hands it out.
    Overridable on the same terms as :func:`get_vector_repository`.
    """
    return resources.embedder


VectorRepositoryDep = Annotated[VectorRepository, Depends(get_vector_repository)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]


def get_collection_service(
    resources: ResourcesDep,
    repository: VectorRepositoryDep,
) -> CollectionService:
    """Build the collection service for this request.

    The index arrives as a dependency rather than being read off ``resources``
    so that overriding :func:`get_vector_repository` reaches every service, not
    just the ones a test remembered to rebuild.
    """
    return CollectionService(
        repository,
        resources.collection_repository,
        max_overrides=resources.settings.max_collection_overrides,
    )


def get_dataset_service(
    resources: ResourcesDep,
    repository: VectorRepositoryDep,
) -> DatasetService:
    """Build the dataset service for this request."""
    return DatasetService(
        repository,
        get_collection_service(resources, repository),
        resources.analysis_repository,
    )


def get_search_service(
    resources: ResourcesDep,
    repository: VectorRepositoryDep,
    embedder: EmbeddingServiceDep,
) -> SearchService:
    """Build the search service for this request.

    Both collaborators are Protocol-typed and injected, which is what makes
    "swap the store" and "swap the encoder" independent changes.
    """
    return SearchService(
        repository=repository,
        embedder=embedder,
        collections=get_collection_service(resources, repository),
    )


def get_export_service(
    resources: ResourcesDep,
    repository: VectorRepositoryDep,
    embedder: EmbeddingServiceDep,
) -> ExportService:
    """Build the export service for this request.

    Composed from the search service rather than the encoder directly, so a
    ranked export ranks by exactly the same code path as the search endpoint.
    The sibling builders are called as plain functions, not re-declared as
    dependencies: they are stateless wrappers, and threading the already-
    resolved ``repository`` and ``embedder`` through them keeps one instance of
    each per request while preserving the override.
    """
    return ExportService(
        repository=repository,
        search_service=get_search_service(resources, repository, embedder),
        collections=get_collection_service(resources, repository),
        projection=resources.projection_repository,
        analysis=resources.analysis_repository,
    )


def get_projection_service(
    resources: ResourcesDep,
    repository: VectorRepositoryDep,
) -> ProjectionService:
    """Build the projection service for this request.

    Constructed even when no projection was loaded: the service reports the
    absence as a domain error the route can turn into a 404, which keeps the
    "not computed" case out of the dependency layer.
    """
    return ProjectionService(
        repository=repository,
        projection=resources.projection_repository,
    )


CollectionServiceDep = Annotated[CollectionService, Depends(get_collection_service)]
DatasetServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
ExportServiceDep = Annotated[ExportService, Depends(get_export_service)]
ProjectionServiceDep = Annotated[ProjectionService, Depends(get_projection_service)]
