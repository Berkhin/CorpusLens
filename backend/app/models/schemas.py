"""Pydantic v2 request/response models — the API's public contract.

These DTOs are the shape the React client sees, and they are kept separate from
the internal domain models in :mod:`app.models.domain` (CLAUDE.md §4.1). The
concrete divergence: every image DTO carries an ``image_url`` the client can
drop straight into ``<img src>``, which is a transport concern the storage
layer has no business knowing about.

The validation bounds below are module-level ``Final`` constants rather than
literals buried in ``Field`` calls, so the limits are greppable and stated once.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Final, Literal, Self
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.models.domain import (
    CaptionRetrieval,
    Collection,
    CollectionCaptionRecall,
    CollectionKind,
    CollectionOrigin,
    CollectionProvenance,
    DatasetStats,
    ExportFormat,
    ImageAnalysis,
    ImageDetail,
    ImageFilter,
    ImagePage,
    ImageSummary,
    InspectedImage,
    Projection,
    ProjectionPoint,
    QualityFlag,
    SearchHit,
)

#: Longest accepted search string. CLIP's text encoder truncates at 77 tokens,
#: so anything beyond a short phrase is silently ignored by the model anyway;
#: this bound just stops a caller from posting a megabyte to find that out.
MAX_QUERY_LENGTH: Final = 500

DEFAULT_SEARCH_LIMIT: Final = 20
MAX_SEARCH_LIMIT: Final = 100

DEFAULT_PAGE_SIZE: Final = 50
MAX_PAGE_SIZE: Final = 200

#: Ceiling on an explicit export selection. A box drawn on the projection can
#: cover thousands of points; this bounds the resulting request body and the
#: ``IN`` list it becomes, and is far above any selection a person makes by hand.
MAX_EXPORT_IDS: Final = 5000

#: Ceiling on a *ranked* export. Higher than ``MAX_SEARCH_LIMIT`` because an
#: export is a file being written once, not a grid being rendered — but still
#: bounded, since past a point a ranking is no longer a ranking.
MAX_EXPORT_ROWS: Final = 1000
DEFAULT_EXPORT_ROWS: Final = 100

#: Ids are filename stems from the ingestion script. Constraining them to this
#: charset at the edge means a crafted id can never reach the store's filter
#: expression or be joined onto a filesystem path.
IMAGE_ID_PATTERN: Final = r"^[A-Za-z0-9._-]+$"

ImageId = Annotated[str, StringConstraints(pattern=IMAGE_ID_PATTERN, min_length=1, max_length=128)]

#: Split names are data, not a fixed enum: a ``--limit``ed ingestion run holds
#: only ``train``, and nothing stops the pipeline from being pointed at a corpus
#: with different names. So the constraint is a shape, not a membership test —
#: it exists to keep a crafted value out of the store's filter expression, and
#: an unknown-but-well-formed split simply matches nothing.
SPLIT_NAME_PATTERN: Final = r"^[a-z]+$"

#: Longest accepted caption substring. Generous for a keyword or short phrase,
#: which is what this filter is for; the semantic search handles sentences.
MAX_CAPTION_FILTER_LENGTH: Final = 100

SplitName = Annotated[
    str, StringConstraints(pattern=SPLIT_NAME_PATTERN, min_length=1, max_length=32)
]
CaptionNeedle = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_CAPTION_FILTER_LENGTH),
]

#: Collection ids are either a split name (for the three built-ins) or a
#: ``uuid4`` hex string. Deliberately *not* reusing ``SPLIT_NAME_PATTERN``,
#: which is ``^[a-z]+$`` and would reject every digit in a uuid. The constraint
#: matters because a built-in's id reaches the store's filter expression through
#: ``split IN (…)`` — a user collection's id never does, but a single type for
#: both is one fewer thing to get wrong.
COLLECTION_ID_PATTERN: Final = r"^[a-z0-9_-]+$"

CollectionId = Annotated[
    str, StringConstraints(pattern=COLLECTION_ID_PATTERN, min_length=1, max_length=64)
]

#: Longest accepted collection name. Long enough for a descriptive label
#: ("holdout — cross-split duplicates"), short enough to render as a chip.
MAX_COLLECTION_NAME_LENGTH: Final = 64

#: Collection names are free text: they never reach a query expression, because
#: every write goes through a bound parameter and every read matches on id.
#: Whitespace is stripped so " train" cannot masquerade as a second "train".
CollectionName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_COLLECTION_NAME_LENGTH),
]

#: Ceiling on the id array of one move request. Raised from the 5 000 that
#: matched ``MAX_EXPORT_IDS`` once a move could also be driven by a *filter*,
#: which addresses sets — ``weak-captions``, ``cross-split-duplicate`` — that are
#: scattered across the whole corpus and can legitimately be all of it.
#:
#: This bounds the *body* only — one array cannot be longer than the corpus.
#: The ceiling that actually governs is ``Settings.max_collection_overrides``,
#: applied in the service to the total the store would hold afterwards, because
#: the cost is the accumulated override rows rather than the bytes any one
#: request arrived in. A body under this limit is still refused when it would
#: push that total over. See :class:`~app.exceptions.CollectionMoveTooLargeError`.
MAX_COLLECTION_MOVE_IMAGES: Final = 8000


def _build_image_url(file_name: str, images_url_prefix: str) -> str:
    """Compose the browser-facing URL for a stored image.

    Args:
        file_name: Basename under the images directory.
        images_url_prefix: Mount point of the static files app.

    Returns:
        A root-relative URL. Relative rather than absolute so the same response
        works behind the Vite dev proxy and when the API is hit directly.
    """
    return f"{images_url_prefix.rstrip('/')}/{quote(file_name)}"


class _ApiModel(BaseModel):
    """Base for every DTO on the wire.

    ``frozen=True`` is declared once here rather than on each model: a DTO is
    built, serialized and discarded, so nothing has a reason to mutate one.
    Pydantic v2 merges ``model_config`` into subclasses — verified against the
    installed pydantic 2.13.4 — so a subclass adding its own key (see
    :class:`SearchRequest`) keeps this one.
    """

    model_config = ConfigDict(frozen=True)


class ImageSummaryResponse(_ApiModel):
    """An image as it appears in a grid or list view."""

    id: str = Field(description="Flickr photo id.")
    file_name: str = Field(description="Basename of the JPEG on disk.")
    split: str = Field(description="Dataset split the image belongs to. Immutable ground truth.")
    collection: str = Field(
        description="Collection the image currently sits in. Equals split unless it was moved."
    )
    image_url: str = Field(description="Root-relative URL serving the full-size JPEG.")

    @classmethod
    def from_domain(cls, summary: ImageSummary, images_url_prefix: str) -> ImageSummaryResponse:
        """Map an internal summary onto its wire representation."""
        return cls(
            id=summary.id,
            file_name=summary.file_name,
            split=summary.split,
            collection=summary.collection,
            image_url=_build_image_url(summary.file_name, images_url_prefix),
        )


class ImageDetailResponse(_ApiModel):
    """A single image with all of its reference captions."""

    id: str = Field(description="Flickr photo id.")
    file_name: str = Field(description="Basename of the JPEG on disk.")
    split: str = Field(description="Dataset split the image belongs to. Immutable ground truth.")
    collection: str = Field(
        description="Collection the image currently sits in. Equals split unless it was moved."
    )
    image_url: str = Field(description="Root-relative URL serving the full-size JPEG.")
    captions: list[str] = Field(description="Human reference captions, normally five.")
    caption_count: int = Field(description="Number of captions present on this record.")

    @classmethod
    def from_domain(cls, detail: ImageDetail, images_url_prefix: str) -> ImageDetailResponse:
        """Map an internal detail record onto its wire representation."""
        return cls(
            id=detail.id,
            file_name=detail.file_name,
            split=detail.split,
            collection=detail.collection,
            image_url=_build_image_url(detail.file_name, images_url_prefix),
            captions=detail.captions,
            caption_count=len(detail.captions),
        )


class ImageAnalysisResponse(_ApiModel):
    """What the offline data-quality pass measured about one image."""

    nearest_neighbour_id: str = Field(description="Most similar other image in the corpus.")
    nearest_neighbour_similarity: float = Field(
        description="Cosine to it. Near 1.0 is a duplicate or a re-shoot; the corpus median "
        "is about 0.83."
    )
    caption_rank: int | None = Field(
        description="Median position of this image when each of its own captions is used as a "
        "query. 1 is perfect. Null when the caption pass was skipped."
    )

    @classmethod
    def from_domain(cls, analysis: ImageAnalysis) -> ImageAnalysisResponse:
        """Map internal measurements onto their wire representation."""
        return cls(
            nearest_neighbour_id=analysis.nearest_neighbour_id,
            nearest_neighbour_similarity=analysis.nearest_neighbour_similarity,
            caption_rank=analysis.caption_rank,
        )


class InspectedImageResponse(_ApiModel):
    """One image with its captions and, when computed, its quality measurements."""

    image: ImageDetailResponse
    analysis: ImageAnalysisResponse | None = Field(
        description="Null when no analysis artefact exists, which is the normal state before "
        "scripts/analyze.py has been run."
    )

    @classmethod
    def from_domain(
        cls, inspected: InspectedImage, images_url_prefix: str
    ) -> InspectedImageResponse:
        """Map an inspected image onto its wire representation."""
        return cls(
            image=ImageDetailResponse.from_domain(inspected.detail, images_url_prefix),
            analysis=(
                None
                if inspected.analysis is None
                else ImageAnalysisResponse.from_domain(inspected.analysis)
            ),
        )


class CaptionRetrievalResponse(_ApiModel):
    """How well the corpus's own captions retrieve their images."""

    recall_at_1: float = Field(description="Fraction of captions whose image ranks first.")
    recall_at_5: float = Field(description="Fraction ranking in the top five.")
    recall_at_10: float = Field(description="Fraction ranking in the top ten.")
    captions: int = Field(description="Captions the figures were computed over.")

    @classmethod
    def from_domain(cls, retrieval: CaptionRetrieval) -> CaptionRetrievalResponse:
        """Map internal recall figures onto their wire representation."""
        return cls(
            recall_at_1=retrieval.recall_at_1,
            recall_at_5=retrieval.recall_at_5,
            recall_at_10=retrieval.recall_at_10,
            captions=retrieval.captions,
        )


class CollectionCaptionRecallResponse(_ApiModel):
    """One collection's caption recall, re-aggregated from per-image ranks.

    Field names are deliberately *not* those of
    :class:`CaptionRetrievalResponse`'s sibling concept: the denominator here is
    ``images``, not ``captions``, and each image contributes the **median** of
    its own captions' ranks rather than each caption contributing separately.
    Shown as the same metric with a different filter, the two would mislead.
    """

    recall_at_1: float = Field(
        description="Fraction of this collection's measured images whose median own-caption "
        "rank against the FULL corpus is 1."
    )
    recall_at_5: float = Field(description="Fraction ranking 5 or better, on the same basis.")
    recall_at_10: float = Field(description="Fraction ranking 10 or better, on the same basis.")
    images: int = Field(
        description="Images in this collection that carry a measured rank. Not the collection's "
        "size — analyze.py --no-captions measures none of them."
    )

    @classmethod
    def from_domain(cls, recall: CollectionCaptionRecall) -> CollectionCaptionRecallResponse:
        """Map internal per-collection recall onto its wire representation."""
        return cls(
            recall_at_1=recall.recall_at_1,
            recall_at_5=recall.recall_at_5,
            recall_at_10=recall.recall_at_10,
            images=recall.images,
        )


class ImagePageResponse(_ApiModel):
    """One page of image summaries, with the totals a paginator needs."""

    items: list[ImageSummaryResponse]
    total: int = Field(description="Images matching the active filter; this is what paginates.")
    corpus_total: int = Field(
        description="Images in the whole index, ignoring the filter. Equals total when unfiltered."
    )
    offset: int = Field(description="Echo of the requested offset.")
    limit: int = Field(description="Echo of the requested limit.")
    has_more: bool = Field(description="True when further pages follow this one.")

    @classmethod
    def from_domain(cls, page: ImagePage, images_url_prefix: str) -> ImagePageResponse:
        """Map an internal page onto its wire representation."""
        return cls(
            items=[
                ImageSummaryResponse.from_domain(item, images_url_prefix) for item in page.items
            ],
            total=page.total,
            corpus_total=page.corpus_total,
            offset=page.offset,
            limit=page.limit,
            has_more=page.offset + len(page.items) < page.total,
        )


class DatasetStatsResponse(_ApiModel):
    """Corpus-level counts for the researcher dashboard."""

    total_images: int = Field(description="Total rows in the index.")
    images_by_split: dict[str, int] = Field(
        description="Row count per split. Only splits present in the index are listed. "
        "Unaffected by collection moves — this is the dataset's own partition."
    )
    images_by_collection: dict[str, int] = Field(
        description="Row count per effective collection, keyed by collection id. Reflects moves."
    )
    projection_available: bool = Field(
        description="Whether GET /projection will succeed. Lets the client hide the map view "
        "instead of discovering its absence through a failed request."
    )
    analysis_available: bool = Field(
        description="Whether the data-quality artefact exists. Gates the quality filters the "
        "same way projection_available gates the map."
    )
    near_duplicate_images: int | None = Field(
        default=None, description="Images appearing in at least one near-duplicate pair."
    )
    cross_split_duplicate_pairs: int | None = Field(
        default=None,
        description="Near-duplicate pairs spanning two splits — the evaluation-leakage count "
        "against the dataset's own partition. Unaffected by collection moves, by construction.",
    )
    cross_collection_duplicate_pairs: int | None = Field(
        default=None,
        description="The same pairs counted against the *user's* partition. Falls as leaking "
        "images are quarantined, while the figure above holds — which is how a researcher sees "
        "that the move they made had the effect they wanted.",
    )
    caption_retrieval: CaptionRetrievalResponse | None = Field(
        default=None, description="Corpus recall of its own captions, if it was measured."
    )
    caption_recall_by_collection: dict[str, CollectionCaptionRecallResponse] | None = Field(
        default=None,
        description="Per-collection caption recall, keyed by collection id. A re-aggregation of "
        "each image's median own-caption rank against the FULL corpus — not a ranking restricted "
        "to the collection, and not comparable with caption_retrieval, whose denominator is "
        "captions rather than images.",
    )

    @classmethod
    def from_domain(
        cls,
        stats: DatasetStats,
        *,
        projection_available: bool,
        analysis_available: bool,
    ) -> DatasetStatsResponse:
        """Map internal statistics onto their wire representation."""
        return cls(
            total_images=stats.total_images,
            images_by_split=stats.images_by_split,
            images_by_collection=stats.images_by_collection,
            projection_available=projection_available,
            analysis_available=analysis_available,
            near_duplicate_images=stats.near_duplicate_images,
            cross_split_duplicate_pairs=stats.cross_split_duplicate_pairs,
            cross_collection_duplicate_pairs=stats.cross_collection_duplicate_pairs,
            caption_retrieval=(
                None
                if stats.caption_retrieval is None
                else CaptionRetrievalResponse.from_domain(stats.caption_retrieval)
            ),
            caption_recall_by_collection=(
                None
                if stats.caption_recall_by_collection is None
                else {
                    collection_id: CollectionCaptionRecallResponse.from_domain(recall)
                    for collection_id, recall in stats.caption_recall_by_collection.items()
                }
            ),
        )


class ProjectionPointResponse(_ApiModel):
    """One image's position on the embedding map.

    Deliberately lean: no ``file_name`` and no ``image_url``, because this
    payload carries every image in the corpus and those two fields would add
    roughly half a megabyte of near-identical strings. The client fetches the
    full record — URL and captions together — from ``/dataset/{id}`` when the
    user actually hovers a point, reusing a cache the inspector already fills.
    """

    id: str = Field(description="Flickr photo id.")
    split: str = Field(description="Dataset split, for colouring the map.")
    x: float = Field(description="Horizontal position, normalised to about [-1, 1].")
    y: float = Field(description="Vertical position, on the same scale as x.")
    matches: bool = Field(description="Whether this image satisfies the active filter.")

    @classmethod
    def from_domain(cls, point: ProjectionPoint) -> ProjectionPointResponse:
        """Map an internal point onto its wire representation."""
        return cls(id=point.id, split=point.split, x=point.x, y=point.y, matches=point.matches)


class ProjectionResponse(_ApiModel):
    """The whole embedding map, with what a reader needs to interpret it."""

    method: str = Field(description="Projection algorithm: pca or tsne.")
    count: int = Field(description="Points returned — every projected image, filtered or not.")
    match_count: int = Field(description="How many of them satisfy the active filter.")
    explained_variance_ratio: list[float] | None = Field(
        description="Share of total variance per component, for PCA. Null for t-SNE, where the "
        "quantity is undefined. Low values mean the map shows broad structure, not clusters."
    )
    points: list[ProjectionPointResponse]

    @classmethod
    def from_domain(cls, projection: Projection) -> ProjectionResponse:
        """Map an internal projection onto its wire representation."""
        ratio = projection.explained_variance_ratio
        return cls(
            method=projection.method,
            count=len(projection.points),
            match_count=projection.match_count,
            explained_variance_ratio=None if ratio is None else list(ratio),
            points=[ProjectionPointResponse.from_domain(point) for point in projection.points],
        )


class _CorpusFilterFields(_ApiModel):
    """The four narrowing dimensions every endpoint that filters accepts.

    Declared once and inherited rather than restated per request model. The
    three consumers — search, export and a filter-driven collection move — must
    take *the same* four dimensions or the "quarantine what I am looking at"
    guarantee stops holding, and three identical copies is exactly the shape
    that drifts. Field descriptions are phrased neutrally ("Restrict to…")
    because the same words now serve ranking, exporting and moving.
    """

    splits: list[SplitName] = Field(
        default_factory=list,
        description="Restrict to these splits. Empty means every split.",
    )
    caption_contains: CaptionNeedle | None = Field(
        default=None,
        description="Restrict to images with this substring in a caption.",
    )
    quality_flag: QualityFlag | None = Field(
        default=None, description="Restrict to a data-quality finding."
    )
    collections: list[CollectionId] = Field(
        default_factory=list,
        description="Restrict to these collections. Empty means every collection.",
    )

    def to_filter(self) -> ImageFilter:
        """Project the narrowing fields onto the domain filter type.

        Keeps every route free of translation: it hands the service a domain
        object, never a bag of request fields.
        """
        return ImageFilter(
            splits=tuple(self.splits),
            caption_contains=self.caption_contains,
            quality_flag=self.quality_flag,
            collections=tuple(self.collections),
        )


class SearchRequest(_CorpusFilterFields):
    """A natural-language query against the CLIP index."""

    #: Merged onto the base's ``frozen=True``. Unknown fields are rejected so a
    #: client that misspells ``limit`` gets a 422 rather than a silent default.
    model_config = ConfigDict(extra="forbid")

    query: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH),
        ]
        | None
    ) = Field(default=None, description="Free-text description of the images to retrieve.")
    image_id: ImageId | None = Field(
        default=None,
        description="Search by example instead: rank images by similarity to this one. Costs no "
        "inference at all — the image's embedding is already in the index.",
    )
    limit: int = Field(
        default=DEFAULT_SEARCH_LIMIT,
        ge=1,
        le=MAX_SEARCH_LIMIT,
        description="Maximum number of ranked results to return.",
    )

    @model_validator(mode="after")
    def _exactly_one_target(self) -> SearchRequest:
        """Require exactly one of ``query`` and ``image_id``.

        Enforced here rather than by branching in the service: neither is
        individually required, but a request with both is ambiguous and one
        with neither has nothing to rank against. Making that a 422 keeps the
        service free of a case that cannot be answered.

        Raises:
            ValueError: If both or neither field is set.
        """
        if (self.query is None) == (self.image_id is None):
            raise ValueError("Provide exactly one of 'query' or 'image_id'")
        return self


class SearchResultResponse(_ApiModel):
    """One ranked hit: the matched image plus its similarity to the query."""

    image: ImageDetailResponse
    score: float = Field(
        description="Cosine similarity to the query in [-1, 1]; higher is a closer match."
    )

    @classmethod
    def from_domain(cls, hit: SearchHit, images_url_prefix: str) -> SearchResultResponse:
        """Map an internal hit onto its wire representation."""
        return cls(
            image=ImageDetailResponse.from_domain(hit.image, images_url_prefix),
            score=hit.score,
        )


class SearchResponse(_ApiModel):
    """The full ranked result set for one query."""

    query: str = Field(description="The query as it was interpreted, after trimming.")
    count: int = Field(description="Number of results returned.")
    results: list[SearchResultResponse]

    @classmethod
    def from_domain(
        cls,
        query: str,
        hits: Sequence[SearchHit],
        images_url_prefix: str,
    ) -> SearchResponse:
        """Map a ranked hit list onto its wire representation.

        Present so the search route maps its result with a single call, like
        every dataset route does, rather than assembling the payload itself.

        Args:
            query: The query as the schema interpreted it, after trimming.
            hits: Hits in the order the service ranked them.
            images_url_prefix: Mount point of the static files app.

        Returns:
            The serialisable result set, with ``count`` derived from ``hits``.
        """
        return cls(
            query=query,
            count=len(hits),
            results=[SearchResultResponse.from_domain(hit, images_url_prefix) for hit in hits],
        )


class _SelectionRequest(_CorpusFilterFields):
    """The corpus selection a manifest export is built from.

    Three mutually exclusive sources, in precedence order:

    1. ``ids`` — the exact images the user selected.
    2. ``query`` — a ranked export, re-run server-side so the file corresponds
       to a reproducible query rather than to whatever the UI had loaded.
    3. neither — every image matching the filter.

    The filter applies to cases 2 and 3. It is deliberately *not* re-applied to
    an explicit id list: the user picked those images, and quietly dropping some
    of them because a filter moved underneath would be a surprise.

    Kept as a base rather than folded into :class:`ExportRequest`: the
    precedence rules above are about *which images*, not about what is emitted,
    and separating them leaves one place to change if a second consumer of the
    same selection appears.
    """

    model_config = ConfigDict(extra="forbid")

    ids: Annotated[list[ImageId], Field(max_length=MAX_EXPORT_IDS)] = Field(
        default_factory=list,
        description="Explicit selection, exported in the order given. Wins over query and filter.",
    )
    query: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH),
        ]
        | None
    ) = Field(default=None, description="Free-text query for a ranked export with scores.")
    limit: int = Field(
        default=DEFAULT_EXPORT_ROWS,
        ge=1,
        le=MAX_EXPORT_ROWS,
        description="Row ceiling for a ranked export. Ignored by the other two modes.",
    )
    image_id: ImageId | None = Field(
        default=None,
        description="Export the neighbours of this image, ranked. Mutually exclusive with query.",
    )

    @model_validator(mode="after")
    def _one_ranking_target_at_most(self) -> Self:
        """Reject a request that asks for two different rankings at once.

        Raises:
            ValueError: If both ``query`` and ``image_id`` are set.
        """
        if self.query is not None and self.image_id is not None:
            raise ValueError("Provide at most one of 'query' or 'image_id'")
        return self


class ExportRequest(_SelectionRequest):
    """A CSV or JSONL manifest to generate from the current selection."""

    format: ExportFormat = Field(
        default="csv",
        description="csv for spreadsheets and pandas; jsonl to keep captions as a list.",
    )


class CollectionProvenanceResponse(_ApiModel):
    """Where a collection's most recent members came from."""

    origin: CollectionOrigin = Field(
        description="manual for a hand-picked or lassoed batch, filter for everything matching "
        "a filter, import for a pasted or uploaded id list."
    )
    detail: str | None = Field(
        description="The filter that matched them, as compact JSON, when origin is filter. This "
        "is what makes the partition reproducible: 32 images with a recorded "
        '{"quality_flag": "cross-split-duplicate"} can be re-derived; 32 images cannot.'
    )
    moved_at: str = Field(description="ISO-8601 UTC instant of that batch.")

    @classmethod
    def from_domain(cls, provenance: CollectionProvenance) -> CollectionProvenanceResponse:
        """Map internal provenance onto its wire representation."""
        return cls(
            origin=provenance.origin,
            detail=provenance.detail,
            moved_at=provenance.moved_at,
        )


class CollectionResponse(_ApiModel):
    """One partition of the corpus as the researcher sees it."""

    id: str = Field(
        description="Stable id. For a built-in this is the dataset split name; for a user "
        "collection, a uuid4 hex string."
    )
    name: str = Field(description="Display name, unique case-insensitively.")
    kind: CollectionKind = Field(
        description="builtin for the dataset's own splits, user for created ones. Only user "
        "collections can be renamed or deleted."
    )
    size: int = Field(description="Images currently in this collection, overrides applied.")
    provenance: CollectionProvenanceResponse | None = Field(
        default=None,
        description="How its most recent members got here. Null when nothing has ever been "
        "moved into it, which is the normal state of an untouched built-in.",
    )

    @classmethod
    def from_domain(cls, collection: Collection) -> CollectionResponse:
        """Map an internal collection onto its wire representation."""
        return cls(
            id=collection.id,
            name=collection.name,
            kind=collection.kind,
            size=collection.size,
            provenance=(
                None
                if collection.provenance is None
                else CollectionProvenanceResponse.from_domain(collection.provenance)
            ),
        )


class CollectionCreateRequest(_ApiModel):
    """A new user collection."""

    model_config = ConfigDict(extra="forbid")

    name: CollectionName = Field(description="Display name for the new collection.")


class CollectionRenameRequest(_ApiModel):
    """A new name for an existing user collection."""

    model_config = ConfigDict(extra="forbid")

    name: CollectionName = Field(description="Replacement display name.")


class CollectionMoveFilter(_CorpusFilterFields):
    """The corpus narrowing a filter-driven move addresses.

    The same four dimensions every listing endpoint takes, inherited rather than
    restated — which is the whole point: the set a researcher is *looking at*
    must be exactly the set they can move, and that only holds if both are
    described by the same fields.
    """

    model_config = ConfigDict(extra="forbid")

    def describe(self) -> str:
        """Render the filter as the provenance record for this move.

        Compact JSON of only the fields the caller actually set, so a stored
        provenance reads ``{"quality_flag":"cross-split-duplicate"}`` rather than
        a wall of empty lists. Serialised from the *request* rather than from
        the resolved :class:`~app.models.domain.ImageFilter`, because by then a
        quality flag has been expanded into the ids it happened to mean today —
        and the reproducible record is the flag, not the ids.

        Returns:
            A JSON object string, ``{}`` for a filter that narrows nothing.
        """
        return self.model_dump_json(exclude_defaults=True)


class CollectionMoveRequest(_ApiModel):
    """Images to move into a collection, named either explicitly or by filter.

    **Exactly one** of ``ids`` and ``filter`` is required, enforced the same way
    :class:`SearchRequest` requires exactly one of ``query`` and ``image_id``:
    a request carrying both is ambiguous and one carrying neither has nothing to
    move, so both are a 422 rather than a case the service has to answer.

    The filter channel exists because the sets a data engineer actually moves —
    ``weak-captions``, ``near-duplicate``, ``cross-split-duplicate``, a caption
    match — are defined by a predicate and scattered across the whole embedding
    cloud, so no rectangle on the map can approximate them.
    """

    model_config = ConfigDict(extra="forbid")

    ids: Annotated[list[ImageId], Field(max_length=MAX_COLLECTION_MOVE_IMAGES)] = Field(
        default_factory=list,
        description="Images to move. Ids not in the index are reported back, not stored.",
    )
    filter: CollectionMoveFilter | None = Field(
        default=None,
        description="Move every image matching this filter instead of naming them. "
        "Resolved server-side, so the set moved is the set the same filter lists.",
    )
    origin: Literal["manual", "import"] = Field(
        default="manual",
        description="How the ids were chosen, recorded as the batch's provenance. Only "
        "meaningful with 'ids': a filter-driven move records 'filter' and the filter itself. "
        "The client has to say, because a pasted list and a lassoed one arrive identically.",
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> CollectionMoveRequest:
        """Require exactly one of ``ids`` and ``filter``.

        An **empty** ``ids`` array counts as absent. It has to: it is the
        default, so treating it as "an explicit selection of nothing" would make
        a bare ``{"filter": …}`` a request with two sources.

        Raises:
            ValueError: If both or neither source is present.
        """
        if bool(self.ids) == (self.filter is not None):
            raise ValueError("Provide exactly one of 'ids' or 'filter'")
        return self


class CollectionMoveResponse(_ApiModel):
    """What a move actually did.

    Reported rather than inferred: a client that moved 300 lassoed points needs
    to know whether any of them were already there or are not in the index,
    and a bare 204 would make both invisible.
    """

    moved: int = Field(description="Images whose collection changed.")
    unchanged: int = Field(description="Images already in the target collection.")
    unknown: list[str] = Field(description="Requested ids that are not in the index. Not stored.")


class ErrorResponse(_ApiModel):
    """Body returned with a 4xx status, declared so it appears in the schema."""

    detail: str = Field(description="Human-readable explanation of the failure.")
