"""Business logic for the user's partition of the corpus.

Framework-agnostic by contract (CLAUDE.md §4.1). This is the layer that composes
the two stores the overlay model needs: the immutable corpus index, which knows
each image's ground-truth split, and the mutable collection store, which knows
which images the researcher has moved. Neither repository knows about the other,
and keeping the join here is what lets each stay a single-purpose module.

Every repository call underneath is blocking, so all of them are pushed onto
anyio's worker-thread pool, matching the convention in the other services.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import anyio.to_thread

from app.exceptions import CollectionMoveTooLargeError, CollectionNotFoundError
from app.models.domain import Collection, CollectionOrigin, CollectionOverlay, ImageFilter
from app.repositories.collection_repository import CollectionRepository
from app.repositories.vector_db import VectorRepository


class CollectionService:
    """Create, rename, delete and populate the researcher's collections."""

    def __init__(
        self,
        repository: VectorRepository,
        collections: CollectionRepository,
        max_overrides: int,
    ) -> None:
        """Bind the service to both stores.

        Args:
            repository: The corpus index, consulted for ground-truth splits and
                to validate that a moved id actually exists.
            collections: The overlay store.
            max_overrides: Ceiling on how many overrides the store may hold in
                total, checked against the state a move would *leave behind*
                rather than against the size of the move. Bounding the request
                instead is no bound at all: the cost is carried by the
                accumulated rows, so the same total is reachable in batches.
        """
        self._repository = repository
        self._collections = collections
        self._max_overrides = max_overrides

    async def overlay(self) -> CollectionOverlay:
        """Read the current override state.

        Returns:
            The overrides, read fresh — the store changes while the process
            runs, so this must never be cached across requests.
        """
        return await anyio.to_thread.run_sync(self._collections.overlay)

    async def list_collections(self) -> list[Collection]:
        """List every collection with its effective size.

        Returns:
            Built-in collections first, then user ones, each with the number of
            images currently in it.
        """
        sizes = await self.size_by_collection()
        return await anyio.to_thread.run_sync(partial(self._collections.list_collections, sizes))

    async def membership(self) -> dict[str, list[str]]:
        """Group every image in the index by its effective collection.

        One scan of ``(id, split)`` joined against the overrides, rather than a
        lookup per moved image. That matters at the far end of the range this
        feature now allows: a fully re-partitioned corpus has 8 000 overrides,
        and asking the index about them by id would build a 200 KB ``IN`` list
        for what a single projection already answers.

        Every split present in the index appears as a key even when nothing is
        left in it, so a collection emptied by a move stays visible at zero
        rather than vanishing from the filter bar with its images.

        **Only overrides whose image is still in the index are honoured.** An id
        absent from the projection is simply skipped, which is the signal
        wanted: re-running ingestion with different ids leaves orphaned override
        rows behind, and counting them would inflate a collection with images
        that do not exist. They are otherwise harmless, so they are not
        garbage-collected behind the user's back.

        Returns:
            Collection id to the ids of its members, in table scan order.
        """
        splits = await anyio.to_thread.run_sync(self._repository.split_by_id)
        assignments = (await self.overlay()).assignments

        grouped: dict[str, list[str]] = {split: [] for split in set(splits.values())}
        for image_id, split in splits.items():
            grouped.setdefault(assignments.get(image_id, split), []).append(image_id)
        return grouped

    async def size_by_collection(self) -> dict[str, int]:
        """Count images per effective collection.

        Returns:
            Row count per collection id, derived from :meth:`membership` so the
            counts and the memberships can never disagree.
        """
        return {
            collection_id: len(members)
            for collection_id, members in (await self.membership()).items()
        }

    async def create(self, name: str) -> str:
        """Create a user collection.

        Args:
            name: Display name.

        Returns:
            The new collection's id.
        """
        return await anyio.to_thread.run_sync(self._collections.create, name)

    async def rename(self, collection_id: str, name: str) -> None:
        """Rename a user collection.

        Args:
            collection_id: Collection to rename.
            name: New display name.
        """
        await anyio.to_thread.run_sync(self._collections.rename, collection_id, name)

    async def delete(self, collection_id: str) -> None:
        """Delete a user collection, returning its images to their splits.

        Args:
            collection_id: Collection to delete.
        """
        await anyio.to_thread.run_sync(self._collections.delete, collection_id)

    async def move_matching(
        self, collection_id: str, image_filter: ImageFilter, filter_description: str
    ) -> tuple[int, int, list[str]]:
        """Move every image a filter selects into a collection.

        The sets worth quarantining — the weak captions, the near-duplicates,
        the images whose captions mention a thing — are defined by a predicate
        and scattered across the corpus, so naming them as ids means either 200
        clicks or a rectangle that picks up the wrong images. This resolves the
        predicate to ids and hands them to :meth:`move_images` unchanged, so the
        ``{moved, unchanged, unknown}`` contract and the
        move-to-own-split-clears-the-override rule keep working exactly as they
        do for an explicit selection.

        ``unknown`` is therefore always empty here: every id came out of the
        index a moment ago. It is still reported, because the two channels
        answer with one shape.

        Counting before listing costs a second scan of a local 8 000-row table —
        milliseconds — and buys an error message that can name the real number
        instead of "more than the limit".

        Args:
            collection_id: Destination collection.
            image_filter: Narrowing, already resolved at the route boundary.
            filter_description: The filter as the caller expressed it, recorded
                as this batch's provenance. Passed in rather than derived from
                ``image_filter``, which by this point has had its quality flag
                expanded into a list of ids — the useful record is
                ``{"quality_flag": "cross-split-duplicate"}``, not the 32 ids it
                happened to mean today.

        Returns:
            The same ``(moved, unchanged, unknown)`` triple as :meth:`move_images`.

        Raises:
            CollectionMoveTooLargeError: If the resulting overlay would exceed
                the configured ceiling. Raised by :meth:`move_images`, which is
                the single place the bound is enforced.
            CollectionNotFoundError: If the destination does not exist.
        """
        count = await anyio.to_thread.run_sync(partial(self._repository.count, image_filter))
        image_ids = await anyio.to_thread.run_sync(
            partial(self._repository.list_ids, image_filter, limit=count)
        )
        return await self.move_images(
            collection_id, image_ids, origin="filter", origin_detail=filter_description
        )

    async def move_images(
        self,
        collection_id: str,
        image_ids: Sequence[str],
        origin: CollectionOrigin = "manual",
        origin_detail: str | None = None,
    ) -> tuple[int, int, list[str]]:
        """Move images into a collection.

        Ids are validated against the index rather than stored on trust: an
        override pointing at an image that does not exist would be invisible in
        the UI and would quietly lengthen every filter predicate.

        Moving an image into the built-in collection matching its **own** split
        clears the override instead of writing one. The redundant row would mean
        the same thing, but it would also put the id into ``excluded_ids`` for
        every unrelated query, and "moved back to where it started" should leave
        no trace.

        Args:
            collection_id: Destination collection.
            image_ids: Images to move.
            origin: How this batch was selected, recorded as its provenance.
            origin_detail: The filter that selected it, when there was one.

        Returns:
            A triple of ``(moved, unchanged, unknown)`` — how many images changed
            collection, how many were already there, and the ids that are not in
            the index. Re-running the same move therefore reports everything as
            ``unchanged``, which is what makes the operation observably
            idempotent rather than merely harmless.

        Raises:
            CollectionMoveTooLargeError: If the overlay would end up holding
                more overrides than the configured ceiling. Checked on the
                resulting total rather than on this batch, and after the
                reset/move split is known, so returning images to their splits
                always succeeds and can bring a full store back under the bound.
            CollectionNotFoundError: If the destination does not exist. Checked
                up front rather than left to the store, because a move whose
                every image resolves to a *reset* never reaches the store's own
                foreign key — so a typo'd destination would answer 200.
        """
        requested = list(dict.fromkeys(image_ids))
        known_collections = await anyio.to_thread.run_sync(self._collections.kinds)
        if collection_id not in known_collections:
            raise CollectionNotFoundError(collection_id)

        known = await anyio.to_thread.run_sync(self._repository.get_many_by_id, requested)
        unknown = [image_id for image_id in requested if image_id not in known]
        assignments = (await self.overlay()).assignments

        to_move: list[str] = []
        to_reset: list[str] = []
        unchanged = 0
        for image_id, image in known.items():
            if assignments.get(image_id, image.split) == collection_id:
                unchanged += 1
            elif image.split == collection_id:
                to_reset.append(image_id)
            else:
                to_move.append(image_id)

        # The bound is on what the store will hold afterwards, not on this
        # batch. `to_move` adds a row only for an image that has none yet;
        # `to_reset` removes one only from an image that has one.
        prospective = (
            len(assignments)
            + sum(1 for image_id in to_move if image_id not in assignments)
            - sum(1 for image_id in to_reset if image_id in assignments)
        )
        if prospective > self._max_overrides:
            raise CollectionMoveTooLargeError(prospective, self._max_overrides)

        moved = await anyio.to_thread.run_sync(
            self._collections.move_images, collection_id, to_move, origin, origin_detail
        )
        moved += await anyio.to_thread.run_sync(self._collections.reset_images, to_reset)
        return moved, unchanged, unknown

    async def reset_image(self, image_id: str) -> None:
        """Drop one image's override, returning it to its ground-truth split.

        Args:
            image_id: Image to reset.
        """
        await anyio.to_thread.run_sync(self._collections.reset_images, [image_id])
