"""Contract tests for the HTTP API.

Every test drives the real application through ``TestClient`` — real lifespan,
real dependency wiring, real services, real repository, real Pydantic
serialization — with only the LanceDB connection and the CLIP encoder replaced
by the doubles in ``conftest.py``. What is asserted here is therefore the
contract the frontend consumes, not the behaviour of a mock.

Ordering within a section runs happy path first, then boundaries, then
rejections.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Final

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.core.config import get_settings
from app.main import create_app
from tests.conftest import (
    CLIMB_ID,
    DOG_ID,
    MISSING_ID,
    PROJECTION_POSITIONS,
    PROJECTION_VARIANCE_RATIO,
    SLIDE_ID,
    FakeClipModel,
    FakeTable,
    install_fake_backends,
)

DOG_QUERY = "a dog running through shallow water"
CLIMB_QUERY = "a man climbing a rock face"

#: The ``image_collection`` schema as it shipped before provenance, used to
#: build a store the migration has to bring forward. Written out rather than
#: derived from the current constant, so a future change to the live schema
#: cannot silently make this test stop testing anything.
_PRE_PROVENANCE_SCHEMA: Final = """
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
  moved_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS image_collection_by_collection
  ON image_collection (collection_id);
"""


# --------------------------------------------------------------------------- #
# GET /api/dataset/stats
# --------------------------------------------------------------------------- #


def test_stats_reports_totals_and_per_split_breakdown(client: TestClient) -> None:
    """Stats expose the corpus total and only the splits actually present."""
    response = client.get("/api/dataset/stats")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "total_images": 3,
        "images_by_split": {"train": 2, "validation": 1},
        # With no image moved, the working partition is the ground-truth one.
        "images_by_collection": {"train": 2, "validation": 1},
        "projection_available": False,
        "analysis_available": False,
        "near_duplicate_images": None,
        "cross_split_duplicate_pairs": None,
        "cross_collection_duplicate_pairs": None,
        "caption_retrieval": None,
        "caption_recall_by_collection": None,
    }


def test_stats_announces_the_projection_when_one_was_computed(
    client_with_projection: TestClient,
) -> None:
    """The flag is what lets the client show the map tab without probing for it."""
    response = client_with_projection.get("/api/dataset/stats")

    assert response.json()["projection_available"] is True


# --------------------------------------------------------------------------- #
# GET /api/dataset
# --------------------------------------------------------------------------- #


def test_list_images_returns_the_corpus_with_root_relative_urls(client: TestClient) -> None:
    """The default page carries every row plus a URL the client can use verbatim."""
    response = client.get("/api/dataset")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert [item["id"] for item in body["items"]] == [DOG_ID, SLIDE_ID, CLIMB_ID]
    assert body["items"][0]["image_url"] == f"/images/{DOG_ID}.jpg"
    assert body["total"] == 3
    assert body["has_more"] is False


def test_list_images_echoes_the_requested_window(client: TestClient) -> None:
    """A partial window reports ``has_more`` against the corpus total, not the page."""
    response = client.get("/api/dataset", params={"offset": 1, "limit": 1})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert [item["id"] for item in body["items"]] == [SLIDE_ID]
    assert (body["offset"], body["limit"], body["total"]) == (1, 1, 3)
    assert body["has_more"] is True


def test_list_images_past_the_end_returns_an_empty_page(client: TestClient) -> None:
    """Paging beyond the corpus is an empty result, not an error."""
    response = client.get("/api/dataset", params={"offset": 99, "limit": 10})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 3
    assert body["has_more"] is False


def test_list_images_accepts_the_maximum_page_size(client: TestClient) -> None:
    """``limit=200`` is the documented ceiling and must be inside the bound."""
    response = client.get("/api/dataset", params={"limit": 200})

    assert response.status_code == status.HTTP_200_OK


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        ({"limit": 0}, "limit below the minimum"),
        ({"limit": 201}, "limit above MAX_PAGE_SIZE"),
        ({"offset": -1}, "negative offset"),
        ({"limit": "many"}, "non-integer limit"),
    ],
)
def test_list_images_rejects_out_of_range_pagination(
    client: TestClient, params: dict[str, Any], reason: str
) -> None:
    """Pagination bounds are enforced at the edge, so no bad window reaches the store."""
    response = client.get("/api/dataset", params=params)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, reason


# --------------------------------------------------------------------------- #
# GET /api/dataset — filtering
# --------------------------------------------------------------------------- #


def test_list_images_filters_by_split(client: TestClient) -> None:
    """One ``split`` keeps only that split, and the total follows the filter."""
    response = client.get("/api/dataset", params={"split": "train"})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert [item["id"] for item in body["items"]] == [DOG_ID, SLIDE_ID]
    assert body["total"] == 2


def test_list_images_accepts_a_repeated_split_parameter(client: TestClient) -> None:
    """``split`` is repeatable and unions the splits rather than overwriting."""
    response = client.get("/api/dataset", params=[("split", "train"), ("split", "validation")])

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["total"] == 3


def test_list_images_reports_the_corpus_total_beside_the_filtered_total(
    client: TestClient,
) -> None:
    """A filtered page still says how large the whole corpus is, for "2 of 3"."""
    response = client.get("/api/dataset", params={"split": "validation"})

    body = response.json()
    assert body["total"] == 1
    assert body["corpus_total"] == 3


def test_list_images_filters_by_caption_substring_ignoring_case(client: TestClient) -> None:
    """The caption filter is a case-insensitive substring over all captions."""
    response = client.get("/api/dataset", params={"caption_contains": "SLIDE"})

    assert response.status_code == status.HTTP_200_OK
    assert [item["id"] for item in response.json()["items"]] == [SLIDE_ID]


def test_list_images_combines_the_split_and_caption_filters(client: TestClient) -> None:
    """Both filters apply together, not one or the other."""
    response = client.get("/api/dataset", params={"split": "train", "caption_contains": "climb"})

    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["corpus_total"] == 3


@pytest.mark.parametrize("needle", ["%", "_"])
def test_list_images_treats_like_wildcards_as_literal_text(client: TestClient, needle: str) -> None:
    """A wildcard the user typed is a character to find, not a pattern to match.

    Unescaped, ``%`` matches every row and ``_`` matches any single character —
    both verified against the real store. Nothing in the fixtures contains
    either character, so a correct implementation returns nothing.
    """
    response = client.get("/api/dataset", params={"caption_contains": needle})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["total"] == 0


def test_list_images_paginates_within_the_filtered_set(client: TestClient) -> None:
    """``offset`` walks the matches, not the underlying corpus."""
    response = client.get("/api/dataset", params={"split": "train", "offset": 1, "limit": 1})

    body = response.json()
    assert [item["id"] for item in body["items"]] == [SLIDE_ID]
    assert body["total"] == 2
    assert body["has_more"] is False


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        ({"split": "TRAIN"}, "uppercase split name"),
        ({"split": "train'; drop"}, "split carrying quote and punctuation"),
        ({"caption_contains": "   "}, "caption needle blank after trimming"),
        ({"caption_contains": "x" * 101}, "caption needle above the length ceiling"),
    ],
)
def test_list_images_rejects_malformed_filters(
    client: TestClient, params: dict[str, Any], reason: str
) -> None:
    """Filter values are constrained at the edge, before they reach a query."""
    response = client.get("/api/dataset", params=params)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, reason


# --------------------------------------------------------------------------- #
# GET /api/dataset/{image_id}
# --------------------------------------------------------------------------- #


def test_get_image_returns_every_reference_caption(client: TestClient) -> None:
    """Detail serves the full caption set with a matching count."""
    response = client.get(f"/api/dataset/{DOG_ID}")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()["image"]
    assert body["id"] == DOG_ID
    assert body["split"] == "train"
    assert body["image_url"] == f"/images/{DOG_ID}.jpg"
    assert len(body["captions"]) == 5
    assert body["caption_count"] == 5
    assert body["captions"][0].startswith("A black dog")


def test_get_image_reports_a_caption_count_below_five(client: TestClient) -> None:
    """``caption_count`` is derived, not assumed to be five (blanks drop at ingestion)."""
    response = client.get(f"/api/dataset/{SLIDE_ID}")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()["image"]
    assert body["caption_count"] == 4
    assert len(body["captions"]) == 4


def test_get_image_returns_404_for_an_unknown_id(client: TestClient) -> None:
    """A well-formed id that is absent is a 404 naming the id, not a 500."""
    response = client.get(f"/api/dataset/{MISSING_ID}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert MISSING_ID in response.json()["detail"]


@pytest.mark.parametrize("image_id", ["bad!id", "id with spaces", "quote'id"])
def test_get_image_returns_422_for_a_malformed_id(client: TestClient, image_id: str) -> None:
    """Ids outside the safe charset are rejected before reaching a filter expression."""
    response = client.get(f"/api/dataset/{image_id}")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


# --------------------------------------------------------------------------- #
# POST /api/search
# --------------------------------------------------------------------------- #


def test_search_ranks_the_semantically_closest_image_first(client: TestClient) -> None:
    """The dog query retrieves the dog image at similarity 1.0."""
    response = client.post("/api/search", json={"query": DOG_QUERY})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["query"] == DOG_QUERY
    assert body["count"] == 3
    assert body["results"][0]["image"]["id"] == DOG_ID
    assert body["results"][0]["score"] == pytest.approx(1.0)


def test_search_ranking_follows_the_query(client: TestClient) -> None:
    """A different query promotes a different image — ranking is computed, not canned."""
    response = client.post("/api/search", json={"query": CLIMB_QUERY})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["results"][0]["image"]["id"] == CLIMB_ID


def test_search_returns_hits_in_descending_similarity(client: TestClient) -> None:
    """Distance is inverted to a score, so results must descend rather than ascend."""
    response = client.post("/api/search", json={"query": DOG_QUERY})

    scores = [result["score"] for result in response.json()["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_results_carry_full_image_detail(client: TestClient) -> None:
    """A hit is renderable on its own: captions and URL arrive with the score."""
    response = client.post("/api/search", json={"query": DOG_QUERY, "limit": 1})

    top = response.json()["results"][0]["image"]
    assert top["image_url"] == f"/images/{DOG_ID}.jpg"
    assert len(top["captions"]) == 5


def test_search_honours_the_requested_limit(client: TestClient) -> None:
    """``limit`` truncates the ranking and is reflected in ``count``."""
    response = client.post("/api/search", json={"query": DOG_QUERY, "limit": 2})

    body = response.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2


def test_search_restricts_the_ranking_to_the_requested_split(client: TestClient) -> None:
    """A split filter narrows the candidates that get ranked at all."""
    response = client.post("/api/search", json={"query": DOG_QUERY, "splits": ["validation"]})

    assert response.status_code == status.HTTP_200_OK
    assert [hit["image"]["id"] for hit in response.json()["results"]] == [CLIMB_ID]


def test_search_applies_the_filter_before_ranking_not_after(client: TestClient) -> None:
    """The filter selects the candidate set; it does not sieve the global top-k.

    This is the whole reason the repository passes ``prefilter=True``. The dog
    query's global best match is in ``train``, so with ``limit=1`` a post-filter
    would take that one hit, discard it for being outside ``validation``, and
    return nothing. Pre-filtering ranks within the split and returns its best.
    """
    response = client.post(
        "/api/search", json={"query": DOG_QUERY, "limit": 1, "splits": ["validation"]}
    )

    body = response.json()
    assert body["count"] == 1
    assert body["results"][0]["image"]["id"] == CLIMB_ID


def test_search_restricts_the_ranking_by_caption_substring(client: TestClient) -> None:
    """The lexical caption filter composes with semantic ranking."""
    response = client.post("/api/search", json={"query": DOG_QUERY, "caption_contains": "slide"})

    assert [hit["image"]["id"] for hit in response.json()["results"]] == [SLIDE_ID]


def test_search_with_a_filter_matching_nothing_returns_an_empty_ranking(
    client: TestClient,
) -> None:
    """An over-narrow filter yields no hits rather than falling back to all of them."""
    response = client.post("/api/search", json={"query": DOG_QUERY, "splits": ["test"]})

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"query": DOG_QUERY, "count": 0, "results": []}


def test_search_trims_the_query_before_encoding(client: TestClient) -> None:
    """Surrounding whitespace is stripped once, at the schema, and never re-added."""
    response = client.post("/api/search", json={"query": f"  {DOG_QUERY}  "})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["query"] == DOG_QUERY
    assert body["results"][0]["image"]["id"] == DOG_ID


def test_search_normalizes_embeddings_for_cosine_ranking(
    client: TestClient, fake_clip_model: FakeClipModel
) -> None:
    """Guards the one flag that silently corrupts ranking if it is ever dropped.

    ``normalize_embeddings=True`` is not the ``sentence-transformers`` default and
    is what makes the stored unit vectors and the query vector comparable by
    cosine. Losing it would not raise anywhere — it would just skew every result
    — so the contract is asserted explicitly rather than left to inspection.
    """
    client.post("/api/search", json={"query": DOG_QUERY})

    assert len(fake_clip_model.encode_calls) == 1
    call = fake_clip_model.encode_calls[0]
    assert call["sentences"] == DOG_QUERY
    assert call["normalize_embeddings"] is True
    assert call["convert_to_numpy"] is True
    assert call["device"] == "cpu"


def test_search_requires_a_query_field(client: TestClient) -> None:
    """A body without ``query`` is a validation error, not an empty search."""
    response = client.post("/api/search", json={})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"query": ""}, "empty query"),
        ({"query": "   "}, "whitespace-only query collapses to empty after trimming"),
        ({"query": "x" * 501}, "query above MAX_QUERY_LENGTH"),
        ({"query": DOG_QUERY, "limit": 0}, "limit below the minimum"),
        ({"query": DOG_QUERY, "limit": 101}, "limit above MAX_SEARCH_LIMIT"),
        ({"query": DOG_QUERY, "top_k": 5}, "unknown field rejected by extra='forbid'"),
    ],
)
def test_search_rejects_invalid_payloads(
    client: TestClient, payload: dict[str, Any], reason: str
) -> None:
    """Every documented bound on the search payload is enforced before inference runs."""
    response = client.post("/api/search", json=payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, reason


def test_rejected_search_never_runs_inference(
    client: TestClient, fake_clip_model: FakeClipModel
) -> None:
    """Validation short-circuits ahead of the encoder — a bad payload costs no CPU."""
    client.post("/api/search", json={"query": ""})

    assert fake_clip_model.encode_calls == []


# --------------------------------------------------------------------------- #
# POST /api/export
# --------------------------------------------------------------------------- #


def _csv_rows(body: str) -> list[dict[str, str]]:
    """Parse an exported CSV body into dict rows, header included."""
    return list(csv.DictReader(io.StringIO(body)))


def _jsonl_rows(body: str) -> list[dict[str, Any]]:
    """Parse an exported JSONL body into objects."""
    return [json.loads(line) for line in body.splitlines() if line]


def test_export_defaults_to_the_whole_corpus_as_csv(client: TestClient) -> None:
    """With no selection, the export is every row, in scan order."""
    response = client.post("/api/export", json={})

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"].startswith("text/csv")
    rows = _csv_rows(response.text)
    assert [row["id"] for row in rows] == [DOG_ID, SLIDE_ID, CLIMB_ID]


def test_export_csv_spreads_captions_across_numbered_columns(client: TestClient) -> None:
    """Captions land in fixed columns, with the real count beside them."""
    response = client.post("/api/export", json={"ids": [DOG_ID]})

    row = _csv_rows(response.text)[0]
    assert row["caption_count"] == "5"
    assert row["caption_1"] == "A black dog runs through shallow water ."
    assert row["caption_5"] == "A black dog is playing in a creek ."


def test_export_csv_pads_the_caption_columns_of_a_short_record(client: TestClient) -> None:
    """A record with four captions still fills the fifth column, with nothing in it."""
    response = client.post("/api/export", json={"ids": [SLIDE_ID]})

    row = _csv_rows(response.text)[0]
    assert row["caption_count"] == "4"
    assert row["caption_5"] == ""


def test_export_leaves_the_score_column_empty_when_nothing_was_ranked(
    client: TestClient,
) -> None:
    """An unranked export must not invent a similarity for its rows."""
    response = client.post("/api/export", json={})

    assert {row["score"] for row in _csv_rows(response.text)} == {""}


def test_export_honours_the_filter(client: TestClient) -> None:
    """The same filter that narrows the grid narrows the manifest."""
    response = client.post("/api/export", json={"splits": ["validation"]})

    assert [row["id"] for row in _csv_rows(response.text)] == [CLIMB_ID]


def test_export_of_an_explicit_selection_keeps_the_requested_order(client: TestClient) -> None:
    """Ids come back in the order asked for, not the order the store scans."""
    response = client.post("/api/export", json={"ids": [CLIMB_ID, DOG_ID]})

    assert [row["id"] for row in _csv_rows(response.text)] == [CLIMB_ID, DOG_ID]


def test_export_of_a_selection_ignores_the_filter(client: TestClient) -> None:
    """An explicit selection wins: the user picked these exact images."""
    response = client.post("/api/export", json={"ids": [DOG_ID], "splits": ["validation"]})

    assert [row["id"] for row in _csv_rows(response.text)] == [DOG_ID]


def test_export_skips_ids_that_are_not_in_the_index(client: TestClient) -> None:
    """A stale id is dropped rather than failing the whole download."""
    response = client.post("/api/export", json={"ids": [MISSING_ID, DOG_ID]})

    assert response.status_code == status.HTTP_200_OK
    assert [row["id"] for row in _csv_rows(response.text)] == [DOG_ID]


def test_export_of_a_query_carries_similarity_scores(client: TestClient) -> None:
    """A ranked export is reproducible from the request and keeps its scores."""
    response = client.post("/api/export", json={"query": DOG_QUERY, "limit": 2})

    rows = _csv_rows(response.text)
    assert [row["id"] for row in rows] == [DOG_ID, SLIDE_ID]
    assert float(rows[0]["score"]) > float(rows[1]["score"])


def test_export_of_a_query_applies_the_filter_before_ranking(client: TestClient) -> None:
    """Ranked export inherits the search endpoint's pre-filter behaviour."""
    response = client.post(
        "/api/export", json={"query": DOG_QUERY, "limit": 1, "splits": ["validation"]}
    )

    assert [row["id"] for row in _csv_rows(response.text)] == [CLIMB_ID]


def test_export_jsonl_keeps_captions_as_a_list(client: TestClient) -> None:
    """JSONL is the lossless format: no fixed columns, no padding."""
    response = client.post("/api/export", json={"format": "jsonl", "ids": [SLIDE_ID]})

    assert response.headers["content-type"].startswith("application/x-ndjson")
    rows = _jsonl_rows(response.text)
    assert len(rows) == 1
    assert rows[0]["captions"] == [
        "Two children play on a red slide .",
        "Kids going down a playground slide .",
        "Children at a playground in the sun .",
        "A child slides down a red slide .",
    ]
    assert "score" not in rows[0]


def test_export_quotes_a_caption_containing_a_comma(client: TestClient) -> None:
    """CSV is written by the csv module, so punctuation cannot shift a column."""
    response = client.post("/api/export", json={"ids": [DOG_ID]})

    # Re-parsing is the assertion: a naive join would put the caption's own
    # separators into the wrong fields and the row would not round-trip.
    row = _csv_rows(response.text)[0]
    assert row["split"] == "train"
    assert row["file_name"] == f"{DOG_ID}.jpg"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"format": "parquet"}, "unsupported format"),
        ({"limit": 0}, "limit below the minimum"),
        ({"limit": 1001}, "limit above MAX_EXPORT_ROWS"),
        ({"ids": ["../etc/passwd"]}, "id outside the permitted charset"),
        ({"ids": ["a"] * 5001}, "selection above MAX_EXPORT_IDS"),
        ({"splits": ["TRAIN"]}, "malformed split"),
        ({"unknown": 1}, "unknown field rejected by extra='forbid'"),
    ],
)
def test_export_rejects_invalid_payloads(
    client: TestClient, payload: dict[str, Any], reason: str
) -> None:
    """Export bounds are enforced at the edge, like every other payload."""
    response = client.post("/api/export", json=payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, reason


# --------------------------------------------------------------------------- #
# GET /api/projection
# --------------------------------------------------------------------------- #


def test_projection_returns_404_when_none_has_been_computed(client: TestClient) -> None:
    """The map is optional, so its absence is a per-request 404, not a dead app."""
    response = client.get("/api/projection")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    # The message has to name the fix: this is the one error an operator who ran
    # ingestion but not projection will actually hit.
    assert "scripts/project.py" in response.json()["detail"]


def test_projection_still_serves_the_rest_of_the_api(client: TestClient) -> None:
    """A missing projection must not degrade browsing or search."""
    assert client.get("/api/dataset").status_code == status.HTTP_200_OK
    assert client.post("/api/search", json={"query": DOG_QUERY}).status_code == status.HTTP_200_OK


def test_projection_returns_every_point_with_its_position(
    client_with_projection: TestClient,
) -> None:
    """Each image comes back with the coordinates the offline script assigned it."""
    response = client_with_projection.get("/api/projection")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["method"] == "pca"
    assert body["count"] == 3
    positions = {point["id"]: [point["x"], point["y"]] for point in body["points"]}
    assert positions == PROJECTION_POSITIONS


def test_projection_reports_how_much_variance_the_axes_explain(
    client_with_projection: TestClient,
) -> None:
    """The honesty number: without it a reader over-reads the layout."""
    response = client_with_projection.get("/api/projection")

    assert response.json()["explained_variance_ratio"] == PROJECTION_VARIANCE_RATIO


def test_projection_carries_the_split_for_colouring(client_with_projection: TestClient) -> None:
    """Points know their split, which is what the map's legend is built from."""
    response = client_with_projection.get("/api/projection")

    splits = {point["id"]: point["split"] for point in response.json()["points"]}
    assert splits[CLIMB_ID] == "validation"


def test_projection_marks_filtered_out_points_instead_of_dropping_them(
    client_with_projection: TestClient,
) -> None:
    """Dimming beats hiding: where a subset sits is what a map is for."""
    response = client_with_projection.get("/api/projection", params={"split": "validation"})

    body = response.json()
    assert body["count"] == 3
    assert body["match_count"] == 1
    matches = {point["id"]: point["matches"] for point in body["points"]}
    assert matches == {DOG_ID: False, SLIDE_ID: False, CLIMB_ID: True}


def test_projection_marks_everything_as_matching_when_unfiltered(
    client_with_projection: TestClient,
) -> None:
    """With no filter every point matches, so nothing renders as dimmed."""
    response = client_with_projection.get("/api/projection")

    body = response.json()
    assert body["match_count"] == body["count"] == 3


@pytest.mark.parametrize(
    "projection_document",
    [
        {
            "method": "tsne",
            "count": 1,
            "points": {DOG_ID: [0.1, 0.2]},
        }
    ],
)
def test_projection_omits_images_the_artefact_does_not_cover(
    client_with_projection: TestClient,
) -> None:
    """A stale projection yields fewer points, not points at invented positions.

    Also covers t-SNE's missing variance ratio: the field is null rather than
    zero, because for t-SNE the quantity does not exist.
    """
    response = client_with_projection.get("/api/projection")

    body = response.json()
    assert body["method"] == "tsne"
    assert [point["id"] for point in body["points"]] == [DOG_ID]
    assert body["explained_variance_ratio"] is None


def test_export_carries_map_coordinates_when_a_projection_exists(
    client_with_projection: TestClient,
) -> None:
    """A selection lassoed off the map exports with the positions it was drawn from."""
    response = client_with_projection.post("/api/export", json={"ids": [CLIMB_ID]})

    row = _csv_rows(response.text)[0]
    assert (float(row["x"]), float(row["y"])) == tuple(PROJECTION_POSITIONS[CLIMB_ID])


def test_export_leaves_coordinates_empty_without_a_projection(client: TestClient) -> None:
    """The columns stay in the header so the format does not depend on optional state."""
    response = client.post("/api/export", json={"ids": [DOG_ID]})

    row = _csv_rows(response.text)[0]
    assert (row["x"], row["y"]) == ("", "")


# --------------------------------------------------------------------------- #
# Data-quality analysis
# --------------------------------------------------------------------------- #


def test_inspecting_an_image_without_an_analysis_reports_none(client: TestClient) -> None:
    """The analysis is optional, so its absence is a null field, not a failure."""
    response = client.get(f"/api/dataset/{DOG_ID}")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["analysis"] is None


def test_inspecting_an_image_surfaces_its_nearest_neighbour(
    client_with_analysis: TestClient,
) -> None:
    """The inspector is where a duplicate becomes one click away from its twin."""
    response = client_with_analysis.get(f"/api/dataset/{DOG_ID}")

    analysis = response.json()["analysis"]
    assert analysis["nearest_neighbour_id"] == SLIDE_ID
    assert analysis["nearest_neighbour_similarity"] == pytest.approx(0.991)
    assert analysis["caption_rank"] == 1


def test_stats_report_the_leakage_count(client_with_analysis: TestClient) -> None:
    """The finding worth acting on gets a number on the dashboard.

    Both readings of it, in fact: with nothing moved, the user's partition is
    the dataset's, so the two leakage figures agree.
    """
    body = client_with_analysis.get("/api/dataset/stats").json()

    assert body["analysis_available"] is True
    assert body["near_duplicate_images"] == 3
    assert body["cross_split_duplicate_pairs"] == 1
    assert body["cross_collection_duplicate_pairs"] == 1
    assert body["caption_retrieval"]["recall_at_1"] == pytest.approx(0.29)
    assert body["caption_retrieval"]["captions"] == 14


def test_quarantining_a_leaking_pair_drops_the_collection_leakage_only(
    client_with_analysis: TestClient,
) -> None:
    """The measurement loop, closed — and the one that must not move, not moving.

    The fixture's leaking pair is DOG (train) and CLIMB (validation). Collecting
    both of them into one user collection means the pair no longer straddles a
    *collection* boundary, so that figure falls to zero — while
    ``cross_split_duplicate_pairs`` and ``images_by_split``, which every offline
    leakage measurement is derived from, hold exactly where they were.

    SLIDE goes along because it near-duplicates DOG too: leaving it behind would
    move that second pair across the new boundary, which is correct behaviour
    and would obscure what this test is about.
    """
    collection_id = _create(client_with_analysis, "Quarantine")
    before = client_with_analysis.get("/api/dataset/stats").json()

    client_with_analysis.post(
        f"/api/collections/{collection_id}/images",
        json={"ids": [DOG_ID, SLIDE_ID, CLIMB_ID]},
    )
    after = client_with_analysis.get("/api/dataset/stats").json()

    assert before["cross_collection_duplicate_pairs"] == 1
    assert after["cross_collection_duplicate_pairs"] == 0
    assert after["cross_split_duplicate_pairs"] == before["cross_split_duplicate_pairs"] == 1
    assert after["images_by_split"] == before["images_by_split"] == {"train": 2, "validation": 1}
    assert after["images_by_collection"] == {"train": 0, "validation": 0, collection_id: 3}


def test_splitting_a_duplicate_pair_apart_raises_the_collection_leakage(
    client_with_analysis: TestClient,
) -> None:
    """The figure is live in both directions, not a one-way "improvement" counter.

    DOG and SLIDE are near-duplicates inside one split, so they contribute
    nothing to the ground-truth leakage count. Move one of them out and the
    user's partition now has a duplicate straddling it — which is exactly the
    mistake this number exists to make visible.
    """
    collection_id = _create(client_with_analysis, "Holdout")

    client_with_analysis.post(f"/api/collections/{collection_id}/images", json={"ids": [SLIDE_ID]})
    body = client_with_analysis.get("/api/dataset/stats").json()

    assert body["cross_collection_duplicate_pairs"] == 2
    assert body["cross_split_duplicate_pairs"] == 1


def test_caption_recall_is_re_aggregated_per_collection(
    client_with_analysis: TestClient,
) -> None:
    """Per-image ranks, filtered and counted over each collection.

    The fixture ranks DOG 1, SLIDE 3 and CLIMB 4 000, so ``train`` retrieves half
    its images first and all of them within five, while ``validation`` retrieves
    none. Note the denominators: this is *images*, where ``caption_retrieval``
    counts *captions* — the two are not the same metric with a filter on it.
    """
    body = client_with_analysis.get("/api/dataset/stats").json()

    recall = body["caption_recall_by_collection"]
    assert recall["train"] == {
        "recall_at_1": pytest.approx(0.5),
        "recall_at_5": pytest.approx(1.0),
        "recall_at_10": pytest.approx(1.0),
        "images": 2,
    }
    assert recall["validation"]["recall_at_1"] == pytest.approx(0.0)
    assert recall["validation"]["images"] == 1


def test_caption_recall_follows_a_move_and_omits_the_unmeasured(
    client_with_analysis: TestClient,
) -> None:
    """A collection with no measured image is omitted, not reported as zero.

    Zero would read as "these annotations are terrible" rather than "this was
    not measured", and the difference matters most for the collection someone
    just created.
    """
    collection_id = _create(client_with_analysis, "Weak")
    empty_id = _create(client_with_analysis, "Untouched")

    client_with_analysis.post(f"/api/collections/{collection_id}/images", json={"ids": [CLIMB_ID]})
    recall = client_with_analysis.get("/api/dataset/stats").json()["caption_recall_by_collection"]

    assert recall[collection_id] == {
        "recall_at_1": pytest.approx(0.0),
        "recall_at_5": pytest.approx(0.0),
        "recall_at_10": pytest.approx(0.0),
        "images": 1,
    }
    # validation is now empty and CLIMB's rank moved with it; both vanish rather
    # than reporting a recall over nothing.
    assert "validation" not in recall
    assert empty_id not in recall


def test_the_quality_figures_are_absent_without_an_analysis(client: TestClient) -> None:
    """Both readings degrade together; neither is invented from the splits alone."""
    body = client.get("/api/dataset/stats").json()

    assert body["cross_collection_duplicate_pairs"] is None
    assert body["caption_recall_by_collection"] is None


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("near-duplicate", [DOG_ID, SLIDE_ID, CLIMB_ID]),
        ("cross-split-duplicate", [DOG_ID, CLIMB_ID]),
        ("weak-captions", [CLIMB_ID]),
    ],
)
def test_quality_flags_narrow_the_gallery(
    client_with_analysis: TestClient, flag: str, expected: list[str]
) -> None:
    """Each finding is reachable as an ordinary filtered listing."""
    response = client_with_analysis.get("/api/dataset", params={"quality_flag": flag})

    assert response.status_code == status.HTTP_200_OK
    assert [item["id"] for item in response.json()["items"]] == expected


def test_a_quality_flag_composes_with_the_split_filter(
    client_with_analysis: TestClient,
) -> None:
    """Resolving the flag to ids is what lets it intersect the other filters."""
    response = client_with_analysis.get(
        "/api/dataset", params={"quality_flag": "cross-split-duplicate", "split": "validation"}
    )

    assert [item["id"] for item in response.json()["items"]] == [CLIMB_ID]


def test_a_quality_flag_without_an_analysis_matches_nothing(client: TestClient) -> None:
    """An unsatisfiable filter returns nothing rather than being ignored.

    Silently dropping it would show the whole corpus under a heading that
    claims to be a list of near-duplicates, which is worse than empty.
    """
    response = client.get("/api/dataset", params={"quality_flag": "near-duplicate"})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["corpus_total"] == 3


def test_quality_flags_narrow_search_and_export(client_with_analysis: TestClient) -> None:
    """The flag reaches the ranking and the manifest, not just the grid."""
    search = client_with_analysis.post(
        "/api/search", json={"query": DOG_QUERY, "quality_flag": "weak-captions"}
    )
    assert [hit["image"]["id"] for hit in search.json()["results"]] == [CLIMB_ID]

    export = client_with_analysis.post("/api/export", json={"quality_flag": "weak-captions"})
    assert [row["id"] for row in _csv_rows(export.text)] == [CLIMB_ID]


def test_export_carries_the_quality_columns(client_with_analysis: TestClient) -> None:
    """A manifest that carries the measurements can be filtered offline."""
    row = _csv_rows(client_with_analysis.post("/api/export", json={"ids": [DOG_ID]}).text)[0]

    assert row["nn_id"] == SLIDE_ID
    assert float(row["nn_similarity"]) == pytest.approx(0.991)
    assert row["caption_rank"] == "1"


def test_export_leaves_the_quality_columns_empty_without_an_analysis(
    client: TestClient,
) -> None:
    """The header does not depend on optional state; only the cells do."""
    row = _csv_rows(client.post("/api/export", json={"ids": [DOG_ID]}).text)[0]

    assert (row["nn_id"], row["nn_similarity"], row["caption_rank"]) == ("", "", "")


def test_rejects_an_unknown_quality_flag(client: TestClient) -> None:
    """The flag is a closed set, validated at the edge."""
    response = client.get("/api/dataset", params={"quality_flag": "suspicious"})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


# --------------------------------------------------------------------------- #
# POST /api/search — by example image
# --------------------------------------------------------------------------- #


def test_search_by_image_ranks_the_corpus_against_it(client: TestClient) -> None:
    """An image's own stored vector is the query; no inference is involved."""
    response = client.post("/api/search", json={"image_id": DOG_ID})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["query"] == f"similar to {DOG_ID}"
    assert body["count"] == 2


def test_search_by_image_excludes_the_query_image(client: TestClient) -> None:
    """Returning the image itself would spend a result slot on a perfect tautology."""
    response = client.post("/api/search", json={"image_id": DOG_ID})

    assert DOG_ID not in [hit["image"]["id"] for hit in response.json()["results"]]


def test_search_by_image_still_fills_the_requested_limit(client: TestClient) -> None:
    """Dropping the query image must not silently shorten the result set."""
    response = client.post("/api/search", json={"image_id": DOG_ID, "limit": 2})

    assert response.json()["count"] == 2


def test_search_by_image_runs_no_inference(
    client: TestClient, fake_clip_model: FakeClipModel
) -> None:
    """The whole point: the vector already exists, so the encoder is never touched."""
    client.post("/api/search", json={"image_id": DOG_ID})

    assert fake_clip_model.encode_calls == []


def test_search_by_image_honours_the_filter(client: TestClient) -> None:
    """Search by example pre-filters exactly as search by text does."""
    response = client.post("/api/search", json={"image_id": DOG_ID, "splits": ["validation"]})

    assert [hit["image"]["id"] for hit in response.json()["results"]] == [CLIMB_ID]


def test_search_by_an_unknown_image_returns_404(client: TestClient) -> None:
    """A stale id is a missing resource, not an empty ranking."""
    response = client.post("/api/search", json={"image_id": MISSING_ID})

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({}, "neither query nor image_id"),
        ({"query": DOG_QUERY, "image_id": DOG_ID}, "both targets at once"),
    ],
)
def test_search_requires_exactly_one_target(
    client: TestClient, payload: dict[str, Any], reason: str
) -> None:
    """Ambiguous and empty requests are both 422 rather than a service branch."""
    response = client.post("/api/search", json=payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, reason


def test_export_can_take_a_neighbour_ranking(client: TestClient) -> None:
    """Export mirrors search: a manifest of an image's neighbours, with scores."""
    response = client.post("/api/export", json={"image_id": DOG_ID, "limit": 2})

    rows = _csv_rows(response.text)
    assert DOG_ID not in [row["id"] for row in rows]
    assert all(row["score"] for row in rows)


# --------------------------------------------------------------------------- #
# Collections
# --------------------------------------------------------------------------- #


def _collections(client: TestClient) -> dict[str, dict[str, Any]]:
    """Read the collection list back, keyed by name.

    Assertions go through the API rather than predicting a ``uuid4`` id, which
    is also how a real client learns what it just created.
    """
    response = client.get("/api/collections")
    assert response.status_code == status.HTTP_200_OK
    return {item["name"]: item for item in response.json()}


def _create(client: TestClient, name: str) -> str:
    """Create a collection and return its id."""
    response = client.post("/api/collections", json={"name": name})
    assert response.status_code == status.HTTP_201_CREATED, response.text
    created: str = response.json()["id"]
    return created


def test_collections_start_as_the_splits_present_in_the_index(client: TestClient) -> None:
    """Built-ins are derived from the data, not hardcoded.

    The fixture corpus holds no ``test`` row, so offering a ``test`` collection
    that could only ever be empty would be a lie about the data — the same
    reason ``images_by_split`` lists only the splits it actually found.
    """
    body = client.get("/api/collections").json()

    assert [(item["id"], item["kind"], item["size"]) for item in body] == [
        ("train", "builtin", 2),
        ("validation", "builtin", 1),
    ]


def test_creating_a_collection_returns_it_empty(client: TestClient) -> None:
    """A new collection exists immediately and holds nothing."""
    response = client.post("/api/collections", json={"name": "My holdout"})

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["kind"] == "user"
    assert response.json()["size"] == 0
    assert "My holdout" in _collections(client)


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("My holdout", "exact duplicate"),
        ("MY HOLDOUT", "case-insensitive duplicate"),
        ("train", "collides with a built-in"),
    ],
)
def test_a_duplicate_collection_name_is_rejected(
    client: TestClient, name: str, reason: str
) -> None:
    """Names are unique case-insensitively, built-ins included.

    Two collections the filter bar renders identically would make it impossible
    to tell which one is being filtered by.
    """
    _create(client, "My holdout")

    response = client.post("/api/collections", json={"name": name})

    assert response.status_code == status.HTTP_409_CONFLICT, reason


def test_renaming_a_collection_keeps_its_id_and_members(client: TestClient) -> None:
    """A rename is cosmetic; the id the filter uses does not move."""
    collection_id = _create(client, "First name")
    client.post(f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]})

    response = client.patch(f"/api/collections/{collection_id}", json={"name": "Second name"})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    provenance = body.pop("provenance")
    assert body == {
        "id": collection_id,
        "name": "Second name",
        "kind": "user",
        "size": 1,
    }
    # A rename does not touch the assignment, so its provenance survives it.
    assert (provenance["origin"], provenance["detail"]) == ("manual", None)


@pytest.mark.parametrize("method", ["patch", "delete"])
def test_a_builtin_collection_cannot_be_renamed_or_deleted(client: TestClient, method: str) -> None:
    """Built-ins mirror immutable ground truth, so the overlay may not edit them."""
    request = getattr(client, method)
    kwargs: dict[str, Any] = {"json": {"name": "renamed"}} if method == "patch" else {}

    response = request("/api/collections/train", **kwargs)

    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_an_unknown_collection_id_is_a_404(client: TestClient) -> None:
    """Editing something that is not there is a missing resource, not a no-op."""
    assert client.delete("/api/collections/nope").status_code == status.HTTP_404_NOT_FOUND


def test_deleting_a_collection_returns_its_images_to_their_split(client: TestClient) -> None:
    """Delete is the undo for a move, and it must be complete.

    Members are not deleted — they revert. The store's ON DELETE CASCADE is what
    provides this, which is why the built-ins are real rows for it to point at.
    """
    collection_id = _create(client, "Holdout")
    client.post(f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]})
    assert client.get(f"/api/dataset/{DOG_ID}").json()["image"]["collection"] == collection_id

    assert (
        client.delete(f"/api/collections/{collection_id}").status_code == status.HTTP_204_NO_CONTENT
    )

    assert client.get(f"/api/dataset/{DOG_ID}").json()["image"]["collection"] == "train"
    assert _collections(client)["train"]["size"] == 2


def test_moving_reports_unknown_ids_instead_of_storing_them(client: TestClient) -> None:
    """An id that is not in the index is reported, not silently recorded.

    A stored override for a non-existent image would be invisible in the UI and
    would lengthen every filter predicate for nothing.
    """
    collection_id = _create(client, "Holdout")

    response = client.post(
        f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID, MISSING_ID]}
    )

    assert response.json() == {"moved": 1, "unchanged": 0, "unknown": [MISSING_ID]}
    assert _collections(client)["Holdout"]["size"] == 1


def test_moving_an_image_to_its_own_split_clears_the_override(client: TestClient) -> None:
    """Moving back to where it started leaves no trace.

    Storing the redundant row would mean the same thing, but it would also put
    the id into the exclusion list of every unrelated collection query.
    """
    collection_id = _create(client, "Holdout")
    client.post(f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]})

    response = client.post("/api/collections/train/images", json={"ids": [DOG_ID]})

    assert response.json()["moved"] == 1
    assert client.get(f"/api/dataset/{DOG_ID}").json()["image"]["collection"] == "train"
    assert _collections(client)["Holdout"]["size"] == 0


def test_moving_is_idempotent(client: TestClient) -> None:
    """Repeating a move changes nothing and says so."""
    collection_id = _create(client, "Holdout")
    client.post(f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]})

    response = client.post(f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]})

    assert response.json() == {"moved": 0, "unchanged": 1, "unknown": []}


def test_resetting_an_image_returns_it_to_its_split(client: TestClient) -> None:
    """The per-image undo, without deleting the collection."""
    collection_id = _create(client, "Holdout")
    client.post(f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]})

    response = client.delete(f"/api/collections/{collection_id}/images/{DOG_ID}")

    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert client.get(f"/api/dataset/{DOG_ID}").json()["image"]["collection"] == "train"


def test_a_move_leaves_the_ground_truth_splits_untouched(client: TestClient) -> None:
    """The whole point of the overlay, asserted directly.

    ``images_by_split`` must not move, because ``scripts/analyze.py`` derives
    cross-split duplicate leakage from those splits. ``images_by_collection``
    is what follows the user's re-partition.
    """
    collection_id = _create(client, "Holdout")
    before = client.get("/api/dataset/stats").json()

    client.post(f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]})
    after = client.get("/api/dataset/stats").json()

    assert after["images_by_split"] == before["images_by_split"] == {"train": 2, "validation": 1}
    assert after["images_by_collection"] == {"train": 1, "validation": 1, collection_id: 1}
    assert client.get(f"/api/dataset/{DOG_ID}").json()["image"]["split"] == "train"


# --------------------------------------------------------------------------- #
# POST /api/collections/{id}/images — the filter channel
# --------------------------------------------------------------------------- #


def test_a_filter_driven_move_matches_the_same_move_by_id(client: TestClient) -> None:
    """The two channels are one operation reached two ways.

    A filter naming ``train`` and an explicit list of the train ids must produce
    the same partition; if they could diverge, "move what I am looking at" would
    be a different promise from "move these".
    """
    by_id = _create(client, "By id")
    by_filter = _create(client, "By filter")

    explicit = client.post(
        f"/api/collections/{by_id}/images", json={"ids": [DOG_ID, SLIDE_ID]}
    ).json()
    # Put them back, so the filter move starts from the state the id move did.
    for image_id in (DOG_ID, SLIDE_ID):
        client.delete(f"/api/collections/{by_id}/images/{image_id}")

    matched = client.post(
        f"/api/collections/{by_filter}/images", json={"filter": {"splits": ["train"]}}
    ).json()

    assert matched == explicit == {"moved": 2, "unchanged": 0, "unknown": []}
    assert _collections(client)["By filter"]["size"] == 2


def test_a_filter_move_resolves_a_quality_flag(client_with_analysis: TestClient) -> None:
    """The dimension that motivated the channel, driven end to end.

    A quality flag is not a property of a stored row — it is resolved to an id
    set at the route boundary — so this is the case that proves the move goes
    through the same ``FilterResolverDep`` as the gallery rather than a second,
    thinner resolution of its own.
    """
    collection_id = _create(client_with_analysis, "Quarantine")

    response = client_with_analysis.post(
        f"/api/collections/{collection_id}/images",
        json={"filter": {"quality_flag": "cross-split-duplicate"}},
    )

    assert response.json()["moved"] == 2
    listed = client_with_analysis.get("/api/dataset", params={"collection": collection_id}).json()
    assert sorted(item["id"] for item in listed["items"]) == sorted([DOG_ID, CLIMB_ID])


def test_a_filter_move_leaves_the_ground_truth_splits_untouched(client: TestClient) -> None:
    """The invariant that matters most, on the channel that can move everything.

    A filter-driven move can address the whole corpus in one request, which is
    exactly when overwriting ``split`` would be tempting and exactly when doing
    so would invalidate every leakage figure derived from it.
    """
    collection_id = _create(client, "Holdout")
    before = client.get("/api/dataset/stats").json()

    client.post(f"/api/collections/{collection_id}/images", json={"filter": {}})
    after = client.get("/api/dataset/stats").json()

    assert after["images_by_split"] == before["images_by_split"] == {"train": 2, "validation": 1}
    # The emptied built-ins stay listed at zero rather than disappearing: they
    # are still collections, and a filter bar that dropped a chip the moment its
    # last image left would strand the user with no way to move anything back.
    assert after["images_by_collection"] == {"train": 0, "validation": 0, collection_id: 3}
    assert client.get(f"/api/dataset/{DOG_ID}").json()["image"]["split"] == "train"


def test_a_filter_matching_nothing_moves_nothing(client: TestClient) -> None:
    """An unsatisfiable filter is a no-op, not an error and not everything."""
    collection_id = _create(client, "Holdout")

    response = client.post(
        f"/api/collections/{collection_id}/images",
        json={"filter": {"caption_contains": "no caption says this"}},
    )

    assert response.json() == {"moved": 0, "unchanged": 0, "unknown": []}
    assert _collections(client)["Holdout"]["size"] == 0


def test_repeating_a_filter_move_reports_everything_unchanged(client: TestClient) -> None:
    """Observable idempotence survives the second channel.

    The filter still selects the same rows the second time — collection
    membership is not what ``splits`` matches on — so every image is found
    already in place rather than moved again.
    """
    collection_id = _create(client, "Holdout")
    client.post(f"/api/collections/{collection_id}/images", json={"filter": {"splits": ["train"]}})

    response = client.post(
        f"/api/collections/{collection_id}/images", json={"filter": {"splits": ["train"]}}
    )

    assert response.json() == {"moved": 0, "unchanged": 2, "unknown": []}


def test_a_filter_move_never_reports_unknown_ids(client: TestClient) -> None:
    """Every id came out of the index a moment ago, so none can be unknown.

    Reported anyway, empty, because both channels answer with one shape.
    """
    collection_id = _create(client, "Holdout")

    response = client.post(
        f"/api/collections/{collection_id}/images", json={"filter": {"splits": ["train"]}}
    )

    assert response.json()["unknown"] == []


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({}, "neither source"),
        ({"ids": []}, "an empty selection is not a source"),
        ({"ids": [DOG_ID], "filter": {"splits": ["train"]}}, "both sources"),
        ({"filter": {"quality_flag": "not-a-flag"}}, "unknown quality flag"),
        ({"filter": {"splits": ["TRAIN"]}}, "malformed split name"),
        ({"filter": {"unknown_field": 1}}, "unknown filter field"),
    ],
)
def test_a_move_must_name_exactly_one_source(
    client: TestClient, payload: dict[str, Any], reason: str
) -> None:
    """Ambiguous and empty requests are rejected, not guessed at."""
    collection_id = _create(client, "Holdout")

    response = client.post(f"/api/collections/{collection_id}/images", json=payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, reason


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"ids": [DOG_ID, SLIDE_ID]}, "explicit selection"),
        ({"filter": {"splits": ["train"]}}, "filter-driven"),
    ],
)
def test_an_oversized_move_is_refused_on_either_channel(
    client_with_a_tiny_move_cap: TestClient, payload: dict[str, Any], reason: str
) -> None:
    """One ceiling, both channels.

    The cost a move leaves behind is the override rows, and those are identical
    whichever way the images were named — so bounding only the request body
    would bound the wrong thing.
    """
    collection_id = _create(client_with_a_tiny_move_cap, "Holdout")

    response = client_with_a_tiny_move_cap.post(
        f"/api/collections/{collection_id}/images", json=payload
    )

    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE, reason
    assert "2" in response.json()["detail"]
    assert _collections(client_with_a_tiny_move_cap)["Holdout"]["size"] == 0


def test_the_ceiling_cannot_be_reached_in_batches(
    client_with_a_tiny_move_cap: TestClient,
) -> None:
    """The bound is on the accumulated overlay, not on one request.

    This is the invariant the ceiling exists for, and the one a per-request
    check silently fails: with a cap of N, two moves of one image each reach
    N + 1 overrides while neither request is individually oversized. Every
    override is an id literal in each subsequent filtered query, so the cost is
    carried by the total — measured on the real corpus, a filtered count goes
    from 22 ms at zero overrides to 1.6 s at a full re-partition. Bounding the
    batch bounds nothing.
    """
    collection_id = _create(client_with_a_tiny_move_cap, "Holdout")
    url = f"/api/collections/{collection_id}/images"

    first = client_with_a_tiny_move_cap.post(url, json={"ids": [DOG_ID]})
    assert first.status_code == status.HTTP_200_OK
    assert _collections(client_with_a_tiny_move_cap)["Holdout"]["size"] == 1

    second = client_with_a_tiny_move_cap.post(url, json={"ids": [SLIDE_ID]})

    assert second.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert _collections(client_with_a_tiny_move_cap)["Holdout"]["size"] == 1


def test_returning_images_to_their_split_is_never_refused(
    client_with_a_tiny_move_cap: TestClient,
) -> None:
    """A full store must still be emptiable.

    The check runs after the reset/move split is known and counts what the
    overlay would hold *afterwards*, so a move that only clears overrides
    lowers the total and passes even at the ceiling. Checking before that split
    would strand a user at the cap with no way back.
    """
    collection_id = _create(client_with_a_tiny_move_cap, "Holdout")
    client_with_a_tiny_move_cap.post(
        f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]}
    )
    assert _collections(client_with_a_tiny_move_cap)["Holdout"]["size"] == 1

    back = client_with_a_tiny_move_cap.post("/api/collections/train/images", json={"ids": [DOG_ID]})

    assert back.status_code == status.HTTP_200_OK
    assert _collections(client_with_a_tiny_move_cap)["Holdout"]["size"] == 0


def test_moving_into_an_unknown_collection_is_a_404_even_when_nothing_moves(
    client: TestClient,
) -> None:
    """The destination is checked before the selection is, not after.

    A move whose every image resolves to a *reset* never reaches the store's
    foreign key, so without an up-front check a typo'd destination would answer
    200 and silently do nothing.
    """
    response = client.post("/api/collections/nope/images", json={"ids": [DOG_ID]})
    matched = client.post("/api/collections/nope/images", json={"filter": {"splits": ["nothing"]}})

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert matched.status_code == status.HTTP_404_NOT_FOUND


# --------------------------------------------------------------------------- #
# Collection provenance
# --------------------------------------------------------------------------- #


def test_a_new_collection_has_no_provenance(client: TestClient) -> None:
    """Nothing has been moved into it, so there is nothing to report."""
    _create(client, "Empty")

    assert _collections(client)["Empty"]["provenance"] is None
    assert _collections(client)["train"]["provenance"] is None


def test_a_filter_move_records_the_filter_that_made_it(client: TestClient) -> None:
    """The field that makes a partition reproducible.

    A collection holding 32 images says nothing three weeks later; one holding
    32 images *because of the cross-split-duplicate flag* can be re-derived.
    Only the fields the caller set are stored, so the record reads as the filter
    rather than as a wall of empty defaults.
    """
    collection_id = _create(client, "Quarantine")

    client.post(
        f"/api/collections/{collection_id}/images",
        json={"filter": {"splits": ["train"], "caption_contains": "dog"}},
    )
    provenance = _collections(client)["Quarantine"]["provenance"]

    assert provenance["origin"] == "filter"
    assert json.loads(provenance["detail"]) == {"splits": ["train"], "caption_contains": "dog"}
    assert provenance["moved_at"].endswith("+00:00")


@pytest.mark.parametrize("origin", ["manual", "import"])
def test_an_id_move_records_the_origin_the_client_declared(client: TestClient, origin: str) -> None:
    """A pasted list and a lassoed one arrive identically, so the client says which."""
    collection_id = _create(client, "Holdout")

    client.post(
        f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID], "origin": origin}
    )

    assert _collections(client)["Holdout"]["provenance"] == {
        "origin": origin,
        "detail": None,
        "moved_at": _collections(client)["Holdout"]["provenance"]["moved_at"],
    }


def test_provenance_reports_the_most_recent_batch(client: TestClient) -> None:
    """The last assignment wins, which is what "where did this set come from" means.

    Reported rather than a full history: collections are populated in one go —
    one filter, one import, one lasso — and a per-image audit trail would be a
    different feature.
    """
    collection_id = _create(client, "Holdout")

    client.post(f"/api/collections/{collection_id}/images", json={"filter": {"splits": ["train"]}})
    client.post(
        f"/api/collections/{collection_id}/images", json={"ids": [CLIMB_ID], "origin": "import"}
    )

    assert _collections(client)["Holdout"]["provenance"]["origin"] == "import"


def test_a_reset_does_not_leave_stale_provenance(client: TestClient) -> None:
    """Provenance lives on the assignment rows, so it goes when they do."""
    collection_id = _create(client, "Holdout")
    client.post(f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]})

    client.delete(f"/api/collections/{collection_id}/images/{DOG_ID}")

    assert _collections(client)["Holdout"]["provenance"] is None


def test_an_existing_store_without_the_provenance_columns_still_opens(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
    fake_table: FakeTable,
    fake_clip_model: FakeClipModel,
) -> None:
    """The schema's IF NOT EXISTS statements cannot add a column, so this is checked.

    SQLite has no ``ALTER TABLE … ADD COLUMN IF NOT EXISTS`` — verified against
    the engine bundled with this Python 3.12 — so the migration reads
    ``PRAGMA table_info`` and adds what is missing. A store written by the
    previous build must keep its data and read back as ``manual``, which is
    truthful: the filter and import channels did not exist when it was written.
    """
    store = data_dir / "collections.db"
    with sqlite3.connect(store) as setup:
        setup.executescript(_PRE_PROVENANCE_SCHEMA)
        setup.execute(
            "INSERT INTO collections (id, name, kind, created_at) VALUES (?, ?, ?, ?)",
            ("old", "Older holdout", "user", "2026-01-01T00:00:00+00:00"),
        )
        setup.execute(
            "INSERT INTO image_collection (image_id, collection_id, moved_at) VALUES (?, ?, ?)",
            (DOG_ID, "old", "2026-01-02T00:00:00+00:00"),
        )

    monkeypatch.setenv("CORPUSLENS_DATA_DIR", str(data_dir))
    get_settings.cache_clear()
    install_fake_backends(mocker, fake_table, fake_clip_model)

    with TestClient(create_app()) as migrated:
        older = _collections(migrated)["Older holdout"]

    get_settings.cache_clear()

    assert older["size"] == 1
    assert older["provenance"] == {
        "origin": "manual",
        "detail": None,
        "moved_at": "2026-01-02T00:00:00+00:00",
    }


def test_filtering_the_gallery_by_collection(client: TestClient) -> None:
    """A collection is an ordinary filter dimension from the client's side."""
    collection_id = _create(client, "Holdout")
    client.post(f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]})

    response = client.get("/api/dataset", params={"collection": collection_id})

    assert [item["id"] for item in response.json()["items"]] == [DOG_ID]
    assert response.json()["total"] == 1
    assert response.json()["corpus_total"] == 3


def test_filtering_by_a_builtin_excludes_images_moved_out_of_it(client: TestClient) -> None:
    """The exclusion half of the predicate: train loses what left it."""
    collection_id = _create(client, "Holdout")
    client.post(f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]})

    response = client.get("/api/dataset", params={"collection": "train"})

    assert [item["id"] for item in response.json()["items"]] == [SLIDE_ID]


def test_filtering_by_a_builtin_includes_images_moved_into_it(client: TestClient) -> None:
    """The union half: an image from another split is re-added by id."""
    client.post("/api/collections/train/images", json={"ids": [CLIMB_ID]})

    response = client.get("/api/dataset", params={"collection": "train"})

    assert [item["id"] for item in response.json()["items"]] == [DOG_ID, SLIDE_ID, CLIMB_ID]
    assert client.get(f"/api/dataset/{CLIMB_ID}").json()["image"]["split"] == "validation"


def test_a_collection_filter_does_not_swallow_the_caption_filter(client: TestClient) -> None:
    """Operator precedence, end to end.

    ``AND`` binds tighter than ``OR``, so without parentheses around the
    collection group the caption clause would apply only to the moved-in branch
    and every other selected image would come back unfiltered. Against the real
    corpus that turned 1 527 rows into 5 999.
    """
    collection_id = _create(client, "Holdout")
    client.post(f"/api/collections/{collection_id}/images", json={"ids": [CLIMB_ID]})

    response = client.get(
        "/api/dataset", params={"collection": ["train", collection_id], "caption_contains": "slide"}
    )

    assert [item["id"] for item in response.json()["items"]] == [SLIDE_ID]


def test_a_collection_filter_intersects_a_quality_flag(client_with_analysis: TestClient) -> None:
    """Two non-row dimensions at once must narrow, not overwrite each other.

    The quality flag compiles into ``ids`` and the collection into its own
    selection, so they meet through the ordinary AND between clauses. If either
    resolver wrote to the other's channel, whichever ran second would win.
    """
    collection_id = _create(client_with_analysis, "Holdout")
    client_with_analysis.post(
        f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID, CLIMB_ID]}
    )

    response = client_with_analysis.get(
        "/api/dataset",
        params={"collection": collection_id, "quality_flag": "weak-captions"},
    )

    # weak-captions is [CLIMB_ID]; the collection is {DOG_ID, CLIMB_ID}.
    assert [item["id"] for item in response.json()["items"]] == [CLIMB_ID]


def test_an_unknown_collection_matches_nothing_rather_than_everything(
    client: TestClient,
) -> None:
    """An unsatisfiable filter is honest; a silently ignored one is not.

    Same contract as a quality flag requested with no analysis loaded.
    """
    response = client.get("/api/dataset", params={"collection": "deleted-yesterday"})

    assert response.json()["items"] == []
    assert response.json()["total"] == 0


def test_search_pre_filters_by_collection(client: TestClient) -> None:
    """A collection narrows the candidate set before ranking, not after.

    Same property the split filter has: with a post-filter, ``limit=1`` would
    take the global best hit, discard it for being outside the collection, and
    return nothing.
    """
    collection_id = _create(client, "Holdout")
    client.post(f"/api/collections/{collection_id}/images", json={"ids": [CLIMB_ID]})

    response = client.post(
        "/api/search", json={"query": DOG_QUERY, "limit": 1, "collections": [collection_id]}
    )

    body = response.json()
    assert body["count"] == 1
    assert body["results"][0]["image"]["id"] == CLIMB_ID


def test_the_projection_marks_only_collection_members_as_matching(
    client_with_projection: TestClient,
) -> None:
    """The map narrows by collection while still drawing the whole corpus."""
    collection_id = _create(client_with_projection, "Holdout")
    client_with_projection.post(f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]})

    body = client_with_projection.get(
        "/api/projection", params={"collection": collection_id}
    ).json()

    assert body["count"] == 3
    assert body["match_count"] == 1
    assert [point["id"] for point in body["points"] if point["matches"]] == [DOG_ID]


def test_export_can_be_filtered_by_collection(client: TestClient) -> None:
    """The manifest honours the same dimension as the grid."""
    collection_id = _create(client, "Holdout")
    client.post(f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]})

    response = client.post("/api/export", json={"collections": [collection_id]})

    assert [row["id"] for row in _csv_rows(response.text)] == [DOG_ID]


def test_the_manifest_carries_the_collection_beside_the_split(client: TestClient) -> None:
    """Both partitions travel in the export, and they differ for a moved image.

    A researcher needs the working assignment *and* the ground truth in one
    file: the leakage figures in ``analysis.json`` are computed from the latter.
    """
    collection_id = _create(client, "Holdout")
    client.post(f"/api/collections/{collection_id}/images", json={"ids": [DOG_ID]})

    row = _csv_rows(client.post("/api/export", json={"ids": [DOG_ID]}).text)[0]

    assert (row["split"], row["collection"]) == ("train", collection_id)


def test_the_jsonl_manifest_carries_the_collection(client: TestClient) -> None:
    """The lossless format carries it too, or the two formats would disagree."""
    row = _jsonl_rows(client.post("/api/export", json={"ids": [DOG_ID], "format": "jsonl"}).text)[0]

    assert (row["split"], row["collection"]) == ("train", "train")


def test_an_override_for_a_missing_image_does_not_inflate_a_collection(
    client: TestClient,
) -> None:
    """Orphaned overrides are ignored in the counts, not counted.

    Re-running ingestion with different ids leaves override rows pointing at
    images that no longer exist. Counting them would inflate every collection
    with images that are not there.
    """
    collection_id = _create(client, "Holdout")

    client.post(f"/api/collections/{collection_id}/images", json={"ids": [MISSING_ID]})

    assert _collections(client)["Holdout"]["size"] == 0
    assert client.get("/api/dataset/stats").json()["images_by_collection"] == {
        "train": 2,
        "validation": 1,
    }
