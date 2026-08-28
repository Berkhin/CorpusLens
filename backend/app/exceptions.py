"""Domain exceptions raised by the service layer.

These carry no HTTP semantics on purpose. Routes are the only layer that
translates them into ``HTTPException`` responses (CLAUDE.md §5.1), which keeps
services usable from tests and future CLIs without a web framework.
"""

from __future__ import annotations


class Flickr8kError(Exception):
    """Base class for every error this application raises deliberately."""


class ImageNotFoundError(Flickr8kError):
    """No image with the requested id exists in the index."""

    def __init__(self, image_id: str) -> None:
        """Record the id that missed so the route can echo it back.

        Args:
            image_id: The identifier that was looked up.
        """
        super().__init__(f"No image with id {image_id!r}")
        self.image_id = image_id


class ProjectionUnavailableError(Flickr8kError):
    """The 2-D projection has not been computed for this data directory.

    Unlike :class:`DatasetUnavailableError` this is *not* a startup failure. The
    map view is an optional capability layered on an optional artefact, so the
    application serves everything else and reports this per request, letting the
    client hide the view rather than the whole app breaking.
    """


class DatasetUnavailableError(Flickr8kError):
    """The on-disk artefacts the API reads are missing or unreadable.

    Raised at startup rather than per request: the API is a pure reader of the
    corpus index ``scripts/ingest.py`` produces (CLAUDE.md §4.2), so a missing
    index is an operator error to surface immediately, not a condition to
    degrade through.
    """


class CollectionNotFoundError(Flickr8kError):
    """No collection with the requested id exists in the overlay store."""

    def __init__(self, collection_id: str) -> None:
        """Record the id that missed so the route can echo it back.

        Args:
            collection_id: The identifier that was looked up.
        """
        super().__init__(f"No collection with id {collection_id!r}")
        self.collection_id = collection_id


class DuplicateCollectionNameError(Flickr8kError):
    """A collection with that name already exists.

    Names are compared case-insensitively: two collections called ``Holdout``
    and ``holdout`` would be indistinguishable in the filter bar, and a
    researcher who cannot tell which one they are filtering by has a worse
    problem than a rejected create.
    """

    def __init__(self, name: str) -> None:
        """Record the name that collided.

        Args:
            name: The requested collection name.
        """
        super().__init__(f"A collection named {name!r} already exists")
        self.name = name


class CollectionMoveTooLargeError(Flickr8kError):
    """A move would leave more overrides in the store than the ceiling allows.

    The ceiling is not about the request body — a filter-driven move sends four
    short fields and can still address the whole corpus. It is about what the
    move *leaves behind*: every override becomes a literal in the id lists that
    :mod:`app.repositories.filters` embeds in every subsequent filtered query.

    Measured against the real 8 000-row table, ``count_rows`` on a filtered
    query costs 22 ms with no override in play and rises linearly with the id
    list: 46 ms at 100 overrides, 222 ms at 1 000, 847 ms at 4 000, and 1.6 s
    once the whole corpus is re-partitioned (a 194 KB predicate). Vector search
    with the same pre-filter tracks it. That is the cost this bound exists to
    keep knowable rather than discovering it as a slow gallery.

    **The bound is therefore on the accumulated total, not on one request.**
    Checking the batch size instead bounds nothing at all: the same 8 000
    overrides are reachable as eight moves of a thousand, which is what the
    previous wording of this message actively suggested doing.
    """

    def __init__(self, count: int, maximum: int) -> None:
        """Record both numbers so the message can name them.

        Args:
            count: Overrides the store would hold once the move was applied.
            maximum: Configured ceiling.
        """
        super().__init__(
            f"This move would leave {count} images assigned to a collection, above the "
            f"limit of {maximum}. Narrow the filter, or return images to their splits "
            "first — splitting it into smaller moves reaches the same total and is "
            "refused the same way."
        )
        self.count = count
        self.maximum = maximum


class BuiltinCollectionError(Flickr8kError):
    """A built-in collection was asked to do something only user ones can.

    The three built-ins mirror the dataset's own splits, which are immutable
    ground truth (see :mod:`app.repositories.collection_repository`). Renaming
    or deleting one would make the overlay claim authority it does not have.
    """

    def __init__(self, collection_id: str) -> None:
        """Record the built-in that was targeted.

        Args:
            collection_id: The built-in collection's id, which is a split name.
        """
        super().__init__(
            f"Collection {collection_id!r} is built in from the dataset split and cannot "
            "be renamed or deleted"
        )
        self.collection_id = collection_id
