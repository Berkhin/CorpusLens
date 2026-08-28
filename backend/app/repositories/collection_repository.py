"""The only module that talks to the collection overlay store.

``split`` is immutable ground truth written once by ``scripts/ingest.py``.  A
*collection* is the researcher's own partition laid over it: every image has
exactly one effective collection, defaulting to its split, and moving an image
records an **override** here rather than touching the LanceDB table.  Keeping
both is not fussiness — ``scripts/analyze.py`` derives cross-split duplicate
leakage from the real splits, and a re-partition that overwrote them would make
that measurement quietly wrong.

Unlike the projection and analysis repositories, this one is **not** a snapshot
loaded at startup.  Its contents change while the process runs, so every method
reads the store.  A cached-at-startup view would serve stale membership for the
life of the process, which for a store the user edits from the UI is the one
failure mode that must not exist.

Storage is stdlib :mod:`sqlite3` — no new dependency, and the engine enforces
the two invariants that would otherwise be hand-rolled: name uniqueness, and
"deleting a collection returns its images to their splits" via
``ON DELETE CASCADE``.  Every method opens a short-lived connection: SQLite
connections are not safe to share across threads, and the service layer runs
these on ``anyio``'s worker pool.

All methods are **blocking**, following the repository convention in
:mod:`app.repositories.image_repository`; callers in the service layer push them
onto a worker thread.

Verified against Python 3.12.13's bundled SQLite: ``PRAGMA foreign_keys`` is
per-connection and defaults **off**, so it is set on every connection — without
it the cascade silently does nothing and deleting a collection would strand its
overrides pointing at a row that no longer exists. ``journal_mode``, by
contrast, is a property of the file and is set once at :meth:`open`; see
``_JOURNAL_MODE`` for why it is not WAL.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Self

from app.exceptions import (
    BuiltinCollectionError,
    CollectionNotFoundError,
    DuplicateCollectionNameError,
)
from app.models.domain import (
    Collection,
    CollectionOrigin,
    CollectionOverlay,
    CollectionProvenance,
)

LOGGER: Final = logging.getLogger(__name__)

#: Journal mode for the store — SQLite's default rollback journal, set
#: explicitly rather than left implicit.
#:
#: **Deliberately not WAL.** WAL would let a read proceed during a write, which
#: sounds right for a store the browser hits concurrently. But WAL coordinates
#: its readers through an mmap'd ``-shm`` file, and ``data/`` is a Docker bind
#: mount on macOS — where that mapping is unreliable. In WAL mode this store
#: raised ``sqlite3.OperationalError: disk I/O error`` from the container as
#: soon as two connections overlapped, while the identical code worked outside
#: it. Docker is a supported run path (CLAUDE.md §2), so the store must work
#: there.
#:
#: Nothing is really given up: the writes here are single rows against a table
#: measured in kilobytes, from one local user. ``_BUSY_TIMEOUT_SECONDS`` covers
#: the contention WAL would otherwise have absorbed.
_JOURNAL_MODE: Final = "DELETE"

#: How long a connection waits for a lock before raising. Without it, two
#: overlapping requests — a move and the listing that refreshes after it — make
#: the loser fail instantly instead of waiting the millisecond it needs.
_BUSY_TIMEOUT_SECONDS: Final = 5.0

#: Kind marker for a collection derived from a dataset split.
_BUILTIN: Final = "builtin"

#: Kind marker for a collection the user created.
_USER: Final = "user"

#: Default origin, and the value backfilled onto rows written before the column
#: existed. Honest for those: they were all hand-picked or lassoed, because the
#: filter and import channels did not exist yet.
_DEFAULT_ORIGIN: Final = "manual"

#: Schema. Applied on every open; every statement is ``IF NOT EXISTS`` so this
#: doubles as the migration for an existing file. Columns added *after* a store
#: was first created cannot be expressed that way — see ``_ADDED_COLUMNS``.
_SCHEMA: Final = f"""
CREATE TABLE IF NOT EXISTS collections (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  kind       TEXT NOT NULL CHECK (kind IN ('builtin', 'user')),
  created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS collections_name_unique
  ON collections (name COLLATE NOCASE);

CREATE TABLE IF NOT EXISTS image_collection (
  image_id      TEXT PRIMARY KEY,
  collection_id TEXT NOT NULL REFERENCES collections (id) ON DELETE CASCADE,
  moved_at      TEXT NOT NULL,
  origin        TEXT NOT NULL DEFAULT '{_DEFAULT_ORIGIN}'
                CHECK (origin IN ('manual', 'filter', 'import')),
  origin_detail TEXT
);
CREATE INDEX IF NOT EXISTS image_collection_by_collection
  ON image_collection (collection_id);
"""

#: Columns added to ``image_collection`` after the store shipped, with the
#: definition to add them with.
#:
#: SQLite has **no** ``ALTER TABLE … ADD COLUMN IF NOT EXISTS`` — verified
#: against the engine bundled with this Python 3.12.13 (SQLite 3.53.1), where it
#: is a syntax error near ``EXISTS``. So the ``IF NOT EXISTS`` trick the rest of
#: the schema relies on cannot be used here, and the migration is guarded by
#: reading ``PRAGMA table_info`` instead. Also verified there: a plain
#: ``ADD COLUMN … NOT NULL DEFAULT 'manual'`` succeeds on a populated table and
#: backfills the existing rows, so an existing ``collections.db`` keeps working
#: and its pre-provenance assignments read as ``manual``.
_ADDED_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("origin", f"TEXT NOT NULL DEFAULT '{_DEFAULT_ORIGIN}'"),
    ("origin_detail", "TEXT"),
)


def _now() -> str:
    """Return the current instant as an ISO-8601 UTC string."""
    return datetime.now(UTC).isoformat()


def _as_origin(value: str) -> CollectionOrigin:
    """Narrow a stored origin string to the domain literal.

    The ``CHECK`` constraint already restricts what can be written, but a store
    migrated from a build without it could in principle hold anything. An
    unrecognised value reads as ``manual`` rather than raising: the provenance
    is a label on a listing, not something a request depends on, and failing the
    whole collection list over it would be the wrong trade.

    Args:
        value: The ``origin`` column as stored.

    Returns:
        The matching literal, or ``manual``.
    """
    if value == "filter":
        return "filter"
    if value == "import":
        return "import"
    return "manual"


class CollectionRepository:
    """Read and write the user's partition of the corpus."""

    def __init__(self, path: Path) -> None:
        """Bind the repository to a database file.

        Args:
            path: Location of the SQLite file. Created on first
                :meth:`open`; its parent directory must already exist.
        """
        self._path = path

    @classmethod
    def open(cls, path: Path, builtin_splits: Iterable[str]) -> Self:
        """Create or migrate the store and seed the built-in collections.

        The built-ins are derived from the splits actually present in the index
        rather than hardcoded: a ``--limit``ed ingestion run holds only
        ``train``, and offering a ``test`` collection that can only ever be
        empty would be a lie about the data.

        Args:
            path: Location of the SQLite file.
            builtin_splits: Split names found in the index.

        Returns:
            A repository ready to serve.
        """
        repository = cls(path)
        with repository._connect() as connection:
            # Journal mode is a property of the *file*, not of a connection, so
            # it is set once here rather than on every open. Setting it per
            # connection is not merely wasteful — see the note on _connect.
            connection.execute(f"PRAGMA journal_mode = {_JOURNAL_MODE}")
            connection.executescript(_SCHEMA)
            repository._add_missing_columns(connection)
            repository._seed_builtins(connection, builtin_splits)
        LOGGER.info("Collection store ready at %s", path)
        return repository

    @staticmethod
    def _add_missing_columns(connection: sqlite3.Connection) -> None:
        """Bring an older ``image_collection`` table up to the current schema.

        ``CREATE TABLE IF NOT EXISTS`` is a no-op on a store that already has
        the table, so columns added later never arrive that way, and SQLite
        offers no ``ADD COLUMN IF NOT EXISTS`` to lean on (see
        ``_ADDED_COLUMNS``). Reading the current column list and adding what is
        missing is idempotent in the same way the rest of the schema is.

        Args:
            connection: Open connection, inside a transaction.
        """
        present = {row["name"] for row in connection.execute("PRAGMA table_info(image_collection)")}
        for column, definition in _ADDED_COLUMNS:
            if column in present:
                continue
            # The column name and definition are module constants, never input.
            connection.execute(f"ALTER TABLE image_collection ADD COLUMN {column} {definition}")
            LOGGER.info("Migrated collection store: added image_collection.%s", column)

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived connection with the pragmas this schema needs.

        Returns:
            A connection with foreign keys enforced, a busy timeout set, and
            rows returned as :class:`sqlite3.Row`.
        """
        connection = sqlite3.connect(self._path, timeout=_BUSY_TIMEOUT_SECONDS)
        connection.row_factory = sqlite3.Row
        # Per-connection, and off by default — without it the cascade that
        # returns images to their splits on delete silently does nothing. This
        # one genuinely does have to be set every time.
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _seed_builtins(connection: sqlite3.Connection, builtin_splits: Iterable[str]) -> None:
        """Insert a built-in row per split, leaving existing rows alone.

        Splits that appear later (a re-ingestion widening the corpus) are added;
        one that disappears is never deleted, because its override rows would
        cascade away with it and silently discard the user's work.

        Args:
            connection: Open connection, inside a transaction.
            builtin_splits: Split names found in the index.
        """
        created_at = _now()
        for split in sorted(builtin_splits):
            connection.execute(
                "INSERT OR IGNORE INTO collections (id, name, kind, created_at) "
                "VALUES (?, ?, ?, ?)",
                (split, split, _BUILTIN, created_at),
            )

    def list_collections(self, sizes: Mapping[str, int]) -> list[Collection]:
        """List every collection, built-ins first.

        Sizes are passed in rather than computed here: an effective size depends
        on which overridden ids still exist in the corpus index, and this module
        deliberately knows nothing about the index (CLAUDE.md §4.1). Composing
        the two stores is the service's job.

        Args:
            sizes: Row count per collection id. Missing keys count as zero.

        Returns:
            Built-in collections first, each group ordered by name, each
            carrying the provenance of its most recent assignment.
        """
        with self._connect() as connection:
            rows = connection.execute(
                # 'builtin' sorts before 'user' alphabetically, which is the
                # order the filter bar wants; the tie-break keeps each group
                # stable by name.
                "SELECT id, name, kind FROM collections ORDER BY kind, name COLLATE NOCASE"
            ).fetchall()
            provenance = self._read_provenance(connection)
        return [
            Collection(
                id=row["id"],
                name=row["name"],
                kind=_BUILTIN if row["kind"] == _BUILTIN else _USER,
                size=sizes.get(row["id"], 0),
                provenance=provenance.get(row["id"]),
            )
            for row in rows
        ]

    @staticmethod
    def _read_provenance(connection: sqlite3.Connection) -> dict[str, CollectionProvenance]:
        """Read the most recent assignment into each collection.

        The bare ``origin`` and ``origin_detail`` columns alongside
        ``MAX(moved_at)`` are SQLite's documented min/max special case: with
        exactly one such aggregate, the bare columns take their values from the
        row that produced it. Verified against the engine bundled with this
        Python 3.12.13 (SQLite 3.53.1) rather than assumed — in any other
        engine this would be an unspecified row.

        Args:
            connection: Open connection.

        Returns:
            Collection id to its latest provenance, omitting collections that
            have never had anything moved into them.
        """
        rows = connection.execute(
            "SELECT collection_id, origin, origin_detail, MAX(moved_at) AS moved_at "
            "FROM image_collection GROUP BY collection_id"
        ).fetchall()
        return {
            row["collection_id"]: CollectionProvenance(
                origin=_as_origin(row["origin"]),
                detail=row["origin_detail"],
                moved_at=row["moved_at"],
            )
            for row in rows
        }

    def create(self, name: str) -> str:
        """Create a user collection.

        Args:
            name: Display name, already stripped and length-checked upstream.

        Returns:
            The new collection's id.

        Raises:
            DuplicateCollectionNameError: If the name is taken, compared
                case-insensitively.
        """
        collection_id = uuid.uuid4().hex
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO collections (id, name, kind, created_at) VALUES (?, ?, ?, ?)",
                    (collection_id, name, _USER, _now()),
                )
        except sqlite3.IntegrityError as error:
            raise DuplicateCollectionNameError(name) from error
        return collection_id

    def rename(self, collection_id: str, name: str) -> None:
        """Rename a user collection.

        Args:
            collection_id: Collection to rename.
            name: New display name.

        Raises:
            CollectionNotFoundError: If no such collection exists.
            BuiltinCollectionError: If it is built in from a split.
            DuplicateCollectionNameError: If the new name is taken.
        """
        with self._connect() as connection:
            self._require_user_collection(connection, collection_id)
            try:
                connection.execute(
                    "UPDATE collections SET name = ? WHERE id = ?", (name, collection_id)
                )
            except sqlite3.IntegrityError as error:
                raise DuplicateCollectionNameError(name) from error

    def delete(self, collection_id: str) -> None:
        """Delete a user collection, returning its images to their splits.

        The revert is the ``ON DELETE CASCADE`` on ``image_collection``: dropping
        the collection drops its override rows, and an image with no override is
        by definition in its split again. That is why the built-ins are real
        rows rather than a synthetic prefix — the constraint needs something to
        point at.

        Args:
            collection_id: Collection to delete.

        Raises:
            CollectionNotFoundError: If no such collection exists.
            BuiltinCollectionError: If it is built in from a split.
        """
        with self._connect() as connection:
            self._require_user_collection(connection, collection_id)
            connection.execute("DELETE FROM collections WHERE id = ?", (collection_id,))

    def move_images(
        self,
        collection_id: str,
        image_ids: Sequence[str],
        origin: CollectionOrigin = "manual",
        origin_detail: str | None = None,
    ) -> int:
        """Move images into a collection, overriding any current assignment.

        Moving an image into the built-in collection matching its **own** split
        is recorded as a deletion rather than a row: the override would be
        redundant, and storing it would make ``all_overridden_ids`` grow with
        entries that change nothing while lengthening every filter predicate.
        That case is handled by :meth:`reset_images`; this method assumes the
        caller has already split the two.

        Args:
            collection_id: Destination collection.
            image_ids: Images to move. Ids not in the index must be filtered out
                by the caller — this layer does not know what the index holds.
            origin: How this batch was selected.
            origin_detail: The filter that selected it, serialised, when
                ``origin`` is ``filter``. Stored verbatim and never parsed here.

        Returns:
            How many rows were written.

        Raises:
            CollectionNotFoundError: If the destination does not exist.
        """
        if not image_ids:
            return 0
        moved_at = _now()
        with self._connect() as connection:
            self._require_collection(connection, collection_id)
            connection.executemany(
                "INSERT INTO image_collection "
                "(image_id, collection_id, moved_at, origin, origin_detail) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (image_id) DO UPDATE SET collection_id = excluded.collection_id, "
                "moved_at = excluded.moved_at, origin = excluded.origin, "
                "origin_detail = excluded.origin_detail",
                [
                    (image_id, collection_id, moved_at, origin, origin_detail)
                    for image_id in image_ids
                ],
            )
        return len(image_ids)

    def reset_images(self, image_ids: Sequence[str]) -> int:
        """Drop overrides, returning images to their ground-truth splits.

        Args:
            image_ids: Images to reset.

        Returns:
            How many overrides were actually removed.
        """
        if not image_ids:
            return 0
        # The only interpolated text is a run of ``?`` placeholders; every value
        # still travels as a bound parameter.
        placeholders = ", ".join("?" for _ in image_ids)
        with self._connect() as connection:
            cursor = connection.execute(
                f"DELETE FROM image_collection WHERE image_id IN ({placeholders})",
                tuple(image_ids),
            )
        return cursor.rowcount

    def overlay(self) -> CollectionOverlay:
        """Read the whole override map.

        Bounded by the number of images the user has *moved*, not by the corpus,
        so reading it whole per request is cheap in normal use.

        Returns:
            The override state.
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT image_id, collection_id FROM image_collection"
            ).fetchall()
        return CollectionOverlay(
            assignments={row["image_id"]: row["collection_id"] for row in rows}
        )

    def kinds(self) -> dict[str, str]:
        """Return every collection id mapped to its kind.

        Returns:
            Mapping of collection id to ``builtin`` or ``user``.
        """
        with self._connect() as connection:
            rows = connection.execute("SELECT id, kind FROM collections").fetchall()
        return {row["id"]: row["kind"] for row in rows}

    @staticmethod
    def _require_collection(connection: sqlite3.Connection, collection_id: str) -> str:
        """Return a collection's kind, raising when it does not exist.

        Args:
            connection: Open connection.
            collection_id: Collection to look up.

        Returns:
            The collection's kind.

        Raises:
            CollectionNotFoundError: If no such collection exists.
        """
        row = connection.execute(
            "SELECT kind FROM collections WHERE id = ?", (collection_id,)
        ).fetchone()
        if row is None:
            raise CollectionNotFoundError(collection_id)
        kind: str = row["kind"]
        return kind

    @classmethod
    def _require_user_collection(cls, connection: sqlite3.Connection, collection_id: str) -> None:
        """Assert a collection exists and is user-created.

        Args:
            connection: Open connection.
            collection_id: Collection to check.

        Raises:
            CollectionNotFoundError: If no such collection exists.
            BuiltinCollectionError: If it is built in from a split.
        """
        if cls._require_collection(connection, collection_id) == _BUILTIN:
            raise BuiltinCollectionError(collection_id)
