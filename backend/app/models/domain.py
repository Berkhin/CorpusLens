"""Internal domain models — the vocabulary shared by repositories and services.

These are deliberately distinct from the API DTOs in :mod:`app.models.schemas`
(CLAUDE.md §4.1). They describe what is *stored*: a record knows its
``file_name`` on disk but nothing about the URL a browser would fetch it from.
Turning one into the other is a presentation concern and happens at the route
boundary.

Frozen slotted dataclasses rather than Pydantic models: nothing here crosses a
trust boundary, so validation would only cost cycles on the read path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

#: File formats the export endpoint can produce. ``csv`` is the spreadsheet and
#: pandas format; ``jsonl`` is the lossless one, because it carries the caption
#: list as a list instead of flattening it into fixed columns.
ExportFormat = Literal["csv", "jsonl"]

#: Data-quality selections the offline analysis makes available. Each resolves
#: to a set of image ids, so they compose with the split and caption filters
#: through the same query path rather than needing one of their own.
QualityFlag = Literal["near-duplicate", "cross-split-duplicate", "weak-captions"]

#: Where a collection came from. ``builtin`` ones mirror the splits actually
#: present in the index and cannot be renamed or deleted; ``user`` ones are
#: created through the API.
CollectionKind = Literal["builtin", "user"]

#: How an image came to be in the collection it is in. Recorded per assignment,
#: because a partition without a recorded reason is not reproducible — three
#: weeks later, 32 images in a "quarantine" collection say nothing about the
#: flag and threshold that put them there.
#:
#: ``manual`` — picked by hand, or lassoed on the map.
#: ``filter`` — everything matching a filter, which is stored alongside.
#: ``import`` — pasted or uploaded as a list of ids from outside the tool.
CollectionOrigin = Literal["manual", "filter", "import"]


@dataclass(frozen=True, slots=True)
class ImageAnalysis:
    """What the offline data-quality pass measured about one image.

    Attributes:
        nearest_neighbour_id: The most similar *other* image in the corpus.
        nearest_neighbour_similarity: Cosine to it. Near 1.0 means a duplicate
            or a re-shoot; the corpus median is around 0.83.
        caption_rank: Median position of this image when each of its own
            captions is used as a query against the whole corpus. 1 is perfect.
            ``None`` when the caption pass was skipped — distinct from a bad
            rank, and the reason this is optional rather than defaulted.
    """

    nearest_neighbour_id: str
    nearest_neighbour_similarity: float
    caption_rank: int | None = None


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    """Two images the embedding space cannot tell apart.

    Attributes:
        a: One image's id.
        b: The other's.
        a_split: Split of ``a``.
        b_split: Split of ``b``.
        similarity: Cosine between them.
        cross_split: Whether the two sit in different splits. This is the
            finding worth acting on — a near-duplicate spanning train and test
            means an evaluation on that test image measures memorisation.
    """

    a: str
    b: str
    a_split: str
    b_split: str
    similarity: float
    cross_split: bool


@dataclass(frozen=True, slots=True)
class CaptionRetrieval:
    """How well the corpus's own captions retrieve their images.

    The standard text-to-image recall-at-k, computed over the dataset's
    annotations rather than a held-out benchmark: it says how self-consistent
    the corpus is under the same model the tool searches with.

    Attributes:
        recall_at_1: Fraction of captions whose image ranks first.
        recall_at_5: Fraction ranking in the top five.
        recall_at_10: Fraction ranking in the top ten.
        captions: How many captions the figures were computed over.
    """

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    captions: int


@dataclass(frozen=True, slots=True)
class CollectionCaptionRecall:
    """How well one collection's images are retrieved by their own captions.

    **A re-aggregation, not a re-measurement, and the distinction is the whole
    point.** ``scripts/analyze.py`` recorded, per image, the median position of
    that image when each of its own captions is used as a query **against the
    full corpus**. Restricting that existing per-image number to a subset and
    counting is free. It answers "how well does this collection's annotation
    hold up, ranked against everything?" — which is the question a quarantine
    is checked against.

    It is **not** "R@k with the gallery restricted to this collection". That is
    a different and harder number — ranking 200 images against 200 rather than
    against 8 000 — and would need `analyze.py` re-run per partition. Nothing
    here approximates it, and the names are kept apart so the two cannot be
    confused.

    It is also **not comparable with** :class:`CaptionRetrieval`. That one's
    denominator is *captions* (40 000 of them) and each is scored individually;
    this one's denominator is *images*, each contributing the median of its own
    five. The two therefore answer to different scales and must never be shown
    as the same metric with a different filter.

    Attributes:
        recall_at_1: Fraction of this collection's measured images whose median
            own-caption rank is 1.
        recall_at_5: Fraction ranking 5 or better.
        recall_at_10: Fraction ranking 10 or better.
        images: How many of the collection's images carry a measured rank. Not
            the collection's size: ``analyze.py --no-captions`` produces none,
            and an image the analysis predates carries none either.
    """

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    images: int


@dataclass(frozen=True, slots=True)
class CollectionProvenance:
    """How a collection last came to hold what it holds.

    The **most recent** assignment into it, not a full history. That is the
    question worth answering cheaply — "where did this set come from?" — and it
    is right for the way collections are actually populated: one filter, one
    import, one lasso. A collection built up over several batches reports the
    last of them, which is stated rather than implied.

    Attributes:
        origin: Whether the last batch was picked by hand, matched by a filter,
            or imported as a list of ids.
        detail: The filter that matched them, serialised as compact JSON, for
            ``filter``; ``None`` otherwise. This is the field that makes a
            partition reproducible: "32 images, cross-split-duplicate flag"
            can be re-derived, "32 images" cannot.
        moved_at: ISO-8601 UTC instant of that batch. Written since the store
            existed and, until now, exposed by no endpoint.
    """

    origin: CollectionOrigin
    detail: str | None
    moved_at: str


@dataclass(frozen=True, slots=True)
class Collection:
    """One partition of the corpus as the researcher sees it.

    Attributes:
        id: Stable identifier. For a built-in this **is** the split name, which
            is what lets the filter resolver match it against the ``split``
            column directly; for a user collection it is a ``uuid4`` hex string.
        name: Display name. Unique case-insensitively across both kinds.
        kind: Whether this mirrors a dataset split or was created by the user.
        size: How many images currently sit in it, overrides applied.
        provenance: Where its most recent members came from, or ``None`` when
            nothing has ever been moved into it — which is the normal state of
            a built-in nobody has touched.
    """

    id: str
    name: str
    kind: CollectionKind
    size: int
    provenance: CollectionProvenance | None = None


@dataclass(frozen=True, slots=True)
class CollectionSelection:
    """A collection filter resolved into terms the store can actually answer.

    A collection is not a column. Membership is "the image's split, unless an
    override says otherwise", which is two facts from two different stores, so
    the selection has to be flattened into id sets at the boundary before any
    predicate can be built.

    Kept as its own value object rather than three loose fields on
    :class:`ImageFilter` because the three are only meaningful together: any one
    of them alone describes half a condition.

    Attributes:
        split_names: The built-in collections selected. These are split names,
            matched against the ``split`` column.
        moved_in_ids: Ids overridden *into* one of the selected collections.
            Re-added regardless of their split.
        excluded_ids: Ids overridden *out of* a selected built-in. Deliberately
            not "every overridden id": one moved *into* a selected collection is
            re-added by ``moved_in_ids`` anyway, so leaving it out keeps the
            ``IN`` list shorter at no cost to correctness.
    """

    split_names: tuple[str, ...] = ()
    moved_in_ids: tuple[str, ...] = ()
    excluded_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CollectionOverlay:
    """The override state, read fresh per request.

    Only overrides are stored. An image absent from :attr:`assignments` sits in
    its ground-truth split, which is what makes the zero-override case identical
    to the behaviour before collections existed.

    Attributes:
        assignments: Image id to collection id, **overrides only**.
    """

    assignments: Mapping[str, str]

    def effective(self, image_id: str, split: str) -> str:
        """Return the collection this image actually belongs to.

        Args:
            image_id: The image to resolve.
            split: Its ground-truth split, used when no override exists.

        Returns:
            The override target, or ``split`` when the image was never moved.
        """
        return self.assignments.get(image_id, split)

    def decorate_summary(self, summary: ImageSummary) -> ImageSummary:
        """Stamp the effective collection onto a summary.

        Args:
            summary: Record as the repository read it, ``collection`` still
                defaulted to its split.

        Returns:
            The same record with ``collection`` resolved.
        """
        return replace(summary, collection=self.effective(summary.id, summary.split))

    def decorate_detail(self, detail: ImageDetail) -> ImageDetail:
        """Stamp the effective collection onto a detail record.

        Args:
            detail: Record as the repository read it.

        Returns:
            The same record with ``collection`` resolved.
        """
        return replace(detail, collection=self.effective(detail.id, detail.split))


@dataclass(frozen=True, slots=True)
class ImageFilter:
    """A narrowing of the corpus, shared by browsing, search and export.

    Carries *intent* only. Turning it into a store query expression is the
    repository's job (CLAUDE.md §4.1) — nothing above that layer should know
    the captions are matched with SQL ``LIKE``.

    Attributes:
        splits: Splits to keep. Empty means "every split", which is why this is
            a tuple rather than a set with a sentinel: the empty case is the
            default and reads naturally.
        caption_contains: Case-insensitive substring that at least one of the
            image's captions must contain. ``None`` disables the check.
            Deliberately lexical — it is the complement to semantic search, not
            a worse version of it: a researcher comparing what CLIP retrieves
            for "dog" against what annotators actually wrote "dog" about needs
            both, and needs them to be different mechanisms.
        quality_flag: A finding from the offline analysis to narrow to. Unlike
            the other two this is not a property of a row, so it cannot be
            expressed as a predicate on the table; it is resolved into ``ids``
            at the route boundary. Everything downstream then treats it as an
            ordinary id filter, which is what lets it compose with the other
            two, with pagination and with export for free.
        ids: Exact ids to keep. ``None`` means no id restriction; an **empty
            tuple means keep nothing**, which is the honest answer when a
            quality flag was requested and the analysis that would satisfy it
            has not been computed. Collapsing those two cases into one would
            turn an unsatisfiable filter into a silently ignored one.
        collections: Collection ids to keep, as *requested*. Like
            ``quality_flag`` this is not a property of a row, but unlike it the
            resolution lands in :attr:`collection_selection` rather than in
            ``ids`` — the two dimensions would otherwise fight over the single
            id channel, and the second one resolved would silently win.
        collection_selection: ``collections`` resolved against the overlay
            store at the route boundary. ``None`` means unresolved, which for a
            non-empty ``collections`` would be a wiring bug rather than "keep
            everything".
    """

    splits: tuple[str, ...] = ()
    caption_contains: str | None = None
    quality_flag: QualityFlag | None = None
    ids: tuple[str, ...] | None = None
    collections: tuple[str, ...] = ()
    collection_selection: CollectionSelection | None = None

    @property
    def is_empty(self) -> bool:
        """True when this filter would keep every row.

        Lets callers skip building an expression, and lets the service skip a
        second row count when the filtered total cannot differ from the corpus
        total.

        ``collections`` is read here rather than ``collection_selection``
        because the *intent* is what makes the filter non-empty: a selection
        that resolves to nothing still has to narrow to nothing. Reporting a
        collection-only filter as empty would make ``list_images`` skip the
        filtered count and report the corpus total as ``total`` — broken
        pagination — and make ``ProjectionService`` mark every point as
        matching.
        """
        return (
            not self.splits
            and not self.caption_contains
            and self.quality_flag is None
            and self.ids is None
            and not self.collections
        )


@dataclass(frozen=True, slots=True)
class ImageSummary:
    """Minimal record for grid/list views.

    Attributes:
        id: Corpus image id (the filename without its extension).
        file_name: Basename under the images directory.
        split: Source split — ``train``, ``validation`` or ``test``. Immutable
            ground truth from the dataset itself.
        collection: The partition the researcher currently has this image in.
            Equal to ``split`` unless it has been moved. Deliberately **not**
            defaulted: the repository sets it from the row's own split, so an
            overlay stamp that never happens degrades to the truthful value
            instead of leaking an empty string into the UI.
    """

    id: str
    file_name: str
    split: str
    collection: str


@dataclass(frozen=True, slots=True)
class ImageDetail:
    """A single image with its full set of reference captions.

    Attributes:
        id: Corpus image id.
        file_name: Basename under the images directory.
        split: Source split, immutable ground truth.
        collection: Effective collection; see :class:`ImageSummary`.
        captions: The human reference captions, in dataset column order.
            Flickr8k supplies five; blank ones were dropped at ingestion, so
            treat the count as "usually five", not "always five".
    """

    id: str
    file_name: str
    split: str
    collection: str
    captions: list[str]


@dataclass(frozen=True, slots=True)
class InspectedImage:
    """A single image together with whatever has been measured about it.

    The two halves come from different places — the captions from the index,
    the measurements from an optional offline artefact — and pairing them here
    means the route asks one service one question instead of assembling the
    answer itself.

    Attributes:
        detail: The record and its captions.
        analysis: Quality measurements, or ``None`` when none were computed.
    """

    detail: ImageDetail
    analysis: ImageAnalysis | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked result of a vector query.

    Attributes:
        image: The matched record, captions included.
        score: Cosine similarity in ``[-1, 1]``, higher is better. Derived from
            LanceDB's ``_distance`` at the repository boundary so nothing above
            it has to know the store reports a distance rather than a score.
    """

    image: ImageDetail
    score: float


@dataclass(frozen=True, slots=True)
class ExportRecord:
    """One row of an exported manifest.

    Distinct from :class:`SearchHit` because ``score`` is genuinely optional
    here: exporting a filtered slice of the corpus produces records that were
    never ranked against anything, and a sentinel score would invite a reader to
    compare it with a real one.

    Attributes:
        image: The record being exported, captions included.
        score: Cosine similarity, when the export came from a ranked query.
        position: The image's ``(x, y)`` on the embedding map, when one has been
            computed. Included so a selection lassoed off the map can be
            replotted outside the tool without re-deriving the projection.
        analysis: Quality measurements, when computed. Carried so a manifest can
            be filtered offline — dropping everything with a near-duplicate
            becomes one pandas expression instead of a second pass through the
            UI.
    """

    image: ImageDetail
    score: float | None = None
    position: tuple[float, float] | None = None
    analysis: ImageAnalysis | None = None


@dataclass(frozen=True, slots=True)
class ProjectionPoint:
    """One image's position on the 2-D embedding map.

    Attributes:
        id: Corpus image id, the key back to the full record.
        split: Source split, so the map can be coloured by it.
        x: Horizontal coordinate, normalised into roughly ``[-1, 1]``.
        y: Vertical coordinate, on the same scale as ``x`` — the projection
            applies one scale factor to both axes, so the aspect ratio carries
            meaning and the client must not stretch them independently.
        matches: Whether this image satisfies the active filter. Non-matching
            points are still returned: seeing *where in the corpus* a filtered
            subset sits is the question a map answers and a grid cannot.

    There is deliberately no ``collection`` field. The map colours by
    ground-truth split on a fixed three-colour palette, and collection
    membership already reaches the client through ``matches`` when a collection
    filter is active. Extending the palette to N arbitrary user names is a
    different feature, and colouring the map by a mutable overlay would lose the
    one view that still shows the corpus as the dataset actually partitions it.
    """

    id: str
    split: str
    x: float
    y: float
    matches: bool


@dataclass(frozen=True, slots=True)
class Projection:
    """The whole embedding map, plus what is needed to read it honestly.

    Attributes:
        method: ``pca`` or ``tsne``.
        explained_variance_ratio: Per-component share of total variance, for
            PCA only. ``None`` for t-SNE, where the quantity does not exist —
            which is itself worth surfacing rather than defaulting to zero.
        points: Every image in the index.
        match_count: How many points satisfy the active filter.
    """

    method: str
    explained_variance_ratio: tuple[float, ...] | None
    points: list[ProjectionPoint]
    match_count: int


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Corpus-level counts for the researcher dashboard.

    Attributes:
        total_images: Rows in the index.
        images_by_split: Row count per split. Only splits actually present are
            listed — a ``--limit``ed ingestion run may contain just ``train``.
            **Unaffected by moves**: this is the ground-truth partition, and it
            staying still while ``images_by_collection`` shifts is the visible
            proof that a re-partition did not corrupt the leakage numbers.
        images_by_collection: Row count per effective collection, overrides
            applied. Keyed by collection id.
        near_duplicate_images: Images appearing in at least one near-duplicate
            pair, or ``None`` when no analysis has been computed.
        cross_split_duplicate_pairs: Near-duplicate pairs spanning two splits —
            the leakage count. ``None`` when no analysis has been computed.
            **This one never moves**, by construction: it is derived from the
            immutable ``split`` column.
        cross_collection_duplicate_pairs: The same pairs, counted against the
            *user's* partition instead. Both members are mapped through the
            overlay, so quarantining one side of a leaking pair makes this fall
            while the figure above holds — which is the only way a researcher
            can see that the action they took had the effect they wanted.
            ``None`` when no analysis has been computed.
        caption_retrieval: Corpus recall of its own captions, or ``None`` when
            the analysis was run with ``--no-captions``.
        caption_recall_by_collection: The same measurement re-aggregated per
            collection; see :class:`CollectionCaptionRecall` for what it is and,
            more importantly, what it is not. ``None`` when the caption pass was
            skipped.
    """

    total_images: int
    images_by_split: dict[str, int]
    images_by_collection: dict[str, int]
    near_duplicate_images: int | None = None
    cross_split_duplicate_pairs: int | None = None
    cross_collection_duplicate_pairs: int | None = None
    caption_retrieval: CaptionRetrieval | None = None
    caption_recall_by_collection: dict[str, CollectionCaptionRecall] | None = None


@dataclass(frozen=True, slots=True)
class ImagePage:
    """One page of summaries plus the totals a paginator needs.

    Attributes:
        items: Summaries for the requested window.
        total: Rows matching the active filter, not the size of this page. This
            is what paginates.
        corpus_total: Rows in the whole index, ignoring the filter. Equal to
            ``total`` when no filter is active. Kept alongside rather than
            derived so the UI can say "2 014 of 8 000" without a second call.
        offset: Echo of the requested offset.
        limit: Echo of the requested limit.
    """

    items: list[ImageSummary]
    total: int
    corpus_total: int
    offset: int
    limit: int
