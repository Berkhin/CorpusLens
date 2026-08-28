"""Tests for the offline partition export.

The one module here that does *not* go through the HTTP boundary. It exists
because ``scripts/export_split.py`` is the sanctioned bridge out of the tool —
the file a training run actually reads — and everything in it except the LanceDB
projection is a pure join or a renderer, which is exactly the shape worth
testing directly.

``_read_corpus`` is left to the manual pass in ``docs/api.md``: it is four lines
of lancedb query builder, and a double for it would assert our idea of lancedb
rather than lancedb.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Final

import pytest

from scripts.export_split import (
    CSV_COMMENT,
    _build_document,
    _is_current,
    _read_overlay,
    _render_csv,
    _render_json,
    export_split,
)
from tests.conftest import CLIMB_ID, DOG_ID, MISSING_ID, SLIDE_ID

SPLITS: Final[dict[str, str]] = {DOG_ID: "train", SLIDE_ID: "train", CLIMB_ID: "validation"}

#: The store's schema as the API creates it, provenance columns included.
_SCHEMA: Final = """
CREATE TABLE collections (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE image_collection (
  image_id TEXT PRIMARY KEY,
  collection_id TEXT NOT NULL REFERENCES collections (id) ON DELETE CASCADE,
  moved_at TEXT NOT NULL,
  origin TEXT NOT NULL DEFAULT 'manual',
  origin_detail TEXT
);
"""

#: The same schema before provenance existed. The script opens the store
#: read-only and therefore cannot migrate it, so it has to cope.
_PRE_PROVENANCE_SCHEMA: Final = """
CREATE TABLE collections (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE image_collection (
  image_id TEXT PRIMARY KEY,
  collection_id TEXT NOT NULL REFERENCES collections (id) ON DELETE CASCADE,
  moved_at TEXT NOT NULL
);
"""


def _write_store(path: Path, schema: str, *, with_provenance: bool) -> None:
    """Build a collection store on disk, on the given schema."""
    with sqlite3.connect(path) as connection:
        connection.executescript(schema)
        connection.execute(
            "INSERT INTO collections VALUES ('train', 'train', 'builtin', '2026-01-01T00:00:00')"
        )
        connection.execute(
            "INSERT INTO collections VALUES ('q', 'Quarantine', 'user', '2026-01-01T00:00:00')"
        )
        if with_provenance:
            connection.execute(
                "INSERT INTO image_collection VALUES (?, 'q', ?, 'filter', ?)",
                (CLIMB_ID, "2026-02-01T00:00:00+00:00", '{"quality_flag":"cross-split-duplicate"}'),
            )
        else:
            connection.execute(
                "INSERT INTO image_collection VALUES (?, 'q', ?)",
                (CLIMB_ID, "2026-02-01T00:00:00+00:00"),
            )


def test_the_document_applies_overrides_over_the_splits() -> None:
    """The same rule the serving layer uses: split, unless an override says otherwise."""
    document = _build_document(SPLITS, [], {CLIMB_ID: "q"})

    assert document["images"] == {
        DOG_ID: {"split": "train", "collection": "train"},
        SLIDE_ID: {"split": "train", "collection": "train"},
        # The ground-truth split travels beside the collection, never replaced
        # by it — that pairing is the whole reason the artefact is useful.
        CLIMB_ID: {"split": "validation", "collection": "q"},
    }


def test_an_emptied_split_is_still_listed_at_zero() -> None:
    """A collection nobody is in is still a collection, and a reader needs to see it."""
    collections: list[dict[str, Any]] = [
        {"id": "validation", "name": "validation", "kind": "builtin", "provenance": None},
        {"id": "q", "name": "Quarantine", "kind": "user", "provenance": None},
    ]

    document = _build_document(SPLITS, collections, {CLIMB_ID: "q"})

    sizes = {entry["id"]: entry["size"] for entry in document["collections"]}
    assert sizes == {"validation": 0, "q": 1}


def test_an_override_for_a_missing_image_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    """Orphaned overrides are reported, not counted — as in the API."""
    document = _build_document(SPLITS, [], {MISSING_ID: "q"})

    assert MISSING_ID not in document["images"]
    assert document["corpus_size"] == 3
    assert "not in the index" in caplog.text


def test_the_overlay_carries_the_filter_that_populated_a_collection(tmp_path: Path) -> None:
    """The provenance is what makes the exported partition reproducible."""
    store = tmp_path / "collections.db"
    _write_store(store, _SCHEMA, with_provenance=True)

    collections, overrides = _read_overlay(store)

    assert overrides == {CLIMB_ID: "q"}
    quarantine = next(item for item in collections if item["id"] == "q")
    assert quarantine["provenance"] == {
        "origin": "filter",
        "detail": '{"quality_flag":"cross-split-duplicate"}',
        "moved_at": "2026-02-01T00:00:00+00:00",
    }


def test_a_store_predating_provenance_still_exports(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The script is read-only, so it cannot migrate — it has to cope instead.

    Requiring the API to be started first would make the offline bridge depend
    on the online one, which is the wrong way round for the artefact a training
    run reads.
    """
    store = tmp_path / "collections.db"
    _write_store(store, _PRE_PROVENANCE_SCHEMA, with_provenance=False)

    collections, overrides = _read_overlay(store)

    assert overrides == {CLIMB_ID: "q"}
    assert all(item["provenance"] is None for item in collections)
    assert "predates the provenance columns" in caplog.text


def test_no_store_at_all_exports_the_dataset_splits(tmp_path: Path) -> None:
    """Nobody has made a collection yet, which is not an error."""
    assert _read_overlay(tmp_path / "collections.db") == ([], {})


def test_the_csv_keeps_free_text_out_of_its_rows(tmp_path: Path) -> None:
    """The header rides in comment lines, so no value may contain the marker.

    A collection *name* can hold anything, including ``#``. Putting names in the
    body would let one of them break ``pd.read_csv(comment="#")`` — the reader
    the comment header exists for.
    """
    collections: list[dict[str, Any]] = [
        {"id": "q", "name": "Sharp # name", "kind": "user", "provenance": None}
    ]
    document = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        **_build_document(SPLITS, collections, {CLIMB_ID: "q"}),
    }

    rendered = _render_csv(document)

    header = [line for line in rendered.splitlines() if line.startswith(CSV_COMMENT)]
    body = list(csv.reader(io.StringIO(rendered.split("image_id,")[1])))
    assert len(header) == 3
    assert "Sharp # name" in "".join(header)
    assert all("#" not in cell for row in body for cell in row)


def test_a_rerun_over_an_unchanged_partition_writes_nothing(tmp_path: Path) -> None:
    """Idempotent per CLAUDE.md §4.2, and on the partition rather than the clock.

    Comparing whole files would make every run differ by its timestamp alone,
    which is a rewrite that says nothing happened.
    """
    document = _build_document(SPLITS, [], {})
    output = tmp_path / "splits.json"
    output.write_text(
        _render_json({"generated_at": "2026-01-01T00:00:00+00:00", **document}), encoding="utf-8"
    )

    assert _is_current(output, document, "json") is True
    assert _is_current(output, _build_document(SPLITS, [], {CLIMB_ID: "q"}), "json") is False


def test_a_rerun_over_an_unchanged_csv_partition_writes_nothing(tmp_path: Path) -> None:
    """Same rule for the CSV, whose timestamp lives in a comment line."""
    document = _build_document(SPLITS, [], {})
    output = tmp_path / "splits.csv"
    output.write_text(
        _render_csv({"generated_at": "2026-01-01T00:00:00+00:00", **document}), encoding="utf-8"
    )

    assert _is_current(output, document, "csv") is True
    assert _is_current(output, _build_document(SPLITS, [], {CLIMB_ID: "q"}), "csv") is False


def test_the_export_refuses_a_data_directory_with_no_index(tmp_path: Path) -> None:
    """A missing index names the script that builds one, as every script here does."""
    args = argparse.Namespace(
        format="json", output=None, force=False, data_dir=tmp_path, verbose=False
    )

    with pytest.raises(FileNotFoundError, match=r"ingest\.py"):
        export_split(args)


def test_the_json_artefact_round_trips(tmp_path: Path) -> None:
    """What is written is what a training script parses back."""
    document = {"generated_at": "2026-01-01T00:00:00+00:00", **_build_document(SPLITS, [], {})}

    assert json.loads(_render_json(document)) == document
