"""Shared fixtures and in-memory doubles for the API test suite.

The suite runs with **no dataset, no LanceDB store and no CLIP weights on disk**.
Two seams are stubbed, both at the outermost edge of the application:

* ``lancedb.connect`` — replaced with a connection returning :class:`FakeTable`,
  an in-memory stand-in that implements exactly the query-builder chain
  ``LanceDBImageRepository`` calls and nothing else.
* ``app.core.lifespan.SentenceTransformer`` — replaced with
  :class:`FakeClipModel`, which maps a handful of known queries to fixed unit
  vectors.

Stubbing that far out is deliberate: everything between the HTTP boundary and
those two seams is the real code path — the lifespan, dependency wiring,
services, the repository's row→domain mapping, its distance→similarity
inversion, and Pydantic serialization. Mocking the repository or the service
instead would leave most of that untested.

**What this trades away.** :class:`FakeTable` is a hand-written approximation of
LanceDB, so it cannot catch a divergence between our assumptions and the real
store: it does not enforce vector dimensionality (the fixtures use 4-d vectors
for legibility where production uses 512), and it reproduces the observed
scan-order and ``_distance`` semantics of lancedb 0.25.3 rather than
guaranteeing them. Those properties are verified against the real store
manually — see the verification notes in ``docs/api.md``.

Filter expressions are **parsed and evaluated** here rather than recorded and
ignored, because a double that accepts any predicate would turn every filtering
test into a test of nothing. What it evaluates is still a re-implementation, not
DataFusion: it recognises exactly the clause shapes the repository emits and
raises on anything else, so the day the repository learns a new one, the suite
says so instead of quietly passing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from pytest_mock import MockerFixture

from app.core.config import get_settings
from app.main import create_app
from app.services.embedding import ClipEmbeddingService

#: Ids in Flickr8k's ``<photo-id>_<hash>`` shape, so route-level id validation
#: is exercised against realistic input rather than "abc".
DOG_ID: Final = "1000268201_693b08cb0e"
SLIDE_ID: Final = "1001773457_577c3a7d70"
CLIMB_ID: Final = "1002674143_1b742ab4b8"

MISSING_ID: Final = "9999999999_0000000000"

#: Queries the fake encoder knows. Each maps to the basis vector of the image it
#: should retrieve, which is what makes the ranking assertions meaningful rather
#: than a restatement of a canned result list.
_QUERY_VECTORS: Final[dict[str, list[float]]] = {
    "a dog running through shallow water": [1.0, 0.0, 0.0, 0.0],
    "children playing on a slide": [0.0, 1.0, 0.0, 0.0],
    "a man climbing a rock face": [0.0, 0.0, 1.0, 0.0],
}

#: Returned for any query outside the table above: equidistant from every
#: fixture image, so an unknown query still ranks deterministically.
_DEFAULT_QUERY_VECTOR: Final = [0.5, 0.5, 0.5, 0.5]

#: The clause shapes ``_build_filter_expression`` can emit, plus the
#: id-equality lookup. Anything else must raise rather than be ignored — a
#: double that shrugs at a filter it does not understand turns every filtering
#: test into a test of nothing.
_ID_FILTER_PATTERN: Final = re.compile(r"^id = '(?P<value>.*)'$")
#: ``split IN (...)`` narrows the corpus; ``id IN (...)`` fetches an explicit
#: selection for export. Same shape, so one pattern captures the column, and the
#: optional ``NOT`` covers the exclusion half of a collection selection.
_MEMBERSHIP_FILTER_PATTERN: Final = re.compile(
    r"^(?P<column>id|split) (?P<negated>NOT )?IN \((?P<literals>.*)\)$"
)
_CAPTION_FILTER_PATTERN: Final = re.compile(
    r"^LOWER\(array_to_string\(captions, ' '\)\) LIKE '%(?P<needle>.*)%' ESCAPE '\\'$"
)

#: Disjunction separator inside a parenthesised group. Only the collection
#: selection emits one, and only ever inside parentheses — which is the property
#: worth modelling, because an unparenthesised ``OR`` reaching this double means
#: the real engine would have mis-grouped the whole expression.
_OR_SEPARATOR: Final = " OR "

#: Pulls the individual quoted values out of an ``IN`` list.
_SQL_LITERAL_PATTERN: Final = re.compile(r"'((?:[^']|'')*)'")

#: The predicate an empty id restriction compiles to. Modelled explicitly so
#: that "this filter is unsatisfiable" stays distinguishable from "this double
#: does not understand the filter".
_MATCH_NOTHING_CLAUSE: Final = "false"

#: Clauses are joined with this. Splitting on it is safe rather than merely
#: convenient: the repository lowercases the caption needle before embedding it,
#: so no user-supplied text can ever contain the uppercase separator.
#:
#: It must nevertheless be split at **paren depth zero only**. A collection
#: selection contains its own ``AND`` inside parentheses, and a naive
#: ``str.split`` would tear it into two fragments that parse as neither clause.
_CLAUSE_SEPARATOR: Final = " AND "


def _split_top_level(expression: str, separator: str) -> list[str]:
    """Split on a separator, ignoring occurrences inside parentheses.

    Args:
        expression: Filter expression built by the repository.
        separator: ``" AND "`` or ``" OR "``.

    Returns:
        The top-level parts, in order.
    """
    parts: list[str] = []
    depth = 0
    start = 0
    index = 0
    while index < len(expression):
        character = expression[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif depth == 0 and expression[index : index + len(separator)] == separator:
            parts.append(expression[start:index])
            index += len(separator)
            start = index
            continue
        index += 1
    parts.append(expression[start:])
    return parts


def _unwrap_group(clause: str) -> str | None:
    """Return the contents of a fully parenthesised clause, or ``None``.

    Args:
        clause: A single top-level clause.

    Returns:
        The inner expression when the clause is one balanced group, otherwise
        ``None``. ``(a) OR (b)`` is deliberately *not* unwrapped: its first
        ``(`` does not pair with its last ``)``.
    """
    if not (clause.startswith("(") and clause.endswith(")")):
        return None
    depth = 0
    for index, character in enumerate(clause):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return clause[1:-1] if index == len(clause) - 1 else None
    return None


def _unquote(literal: str) -> str:
    """Undo the repository's quote-doubling for one SQL string literal."""
    return literal.replace("''", "'")


def _unescape_like(pattern: str) -> str:
    r"""Undo ``_escape_like_pattern``, recovering the literal the user typed.

    Args:
        pattern: The body of a ``LIKE`` pattern, escaped with ``\``.

    Returns:
        The plain substring to test for.
    """
    result: list[str] = []
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "\\" and index + 1 < len(pattern):
            result.append(pattern[index + 1])
            index += 2
            continue
        result.append(character)
        index += 1
    return "".join(result)


def _cosine_distance(left: NDArray[np.float32], right: NDArray[np.float32]) -> float:
    """Return ``1 - cosine_similarity``, the quantity LanceDB reports as ``_distance``.

    Args:
        left: Query vector.
        right: Stored image vector.

    Returns:
        Distance in ``[0, 2]``; ``0.0`` for identical directions.
    """
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0
    return 1.0 - float(np.dot(left, right) / denominator)


def _matches_clause(row: dict[str, Any], clause: str) -> bool:
    """Evaluate one filter clause against a row.

    Each branch mirrors a shape ``_build_filter_expression`` (or the id lookup)
    produces. Unquoting and un-escaping happen here rather than being assumed
    away, so a test can assert that a value containing a quote or a ``%`` is
    neutralised rather than injected.

    Args:
        row: Candidate row.
        clause: A single predicate, already split off the conjunction.

    Returns:
        Whether the row satisfies this clause.

    Raises:
        AssertionError: If the clause is not one this double can model. Failing
            loudly is the point: silently passing an unmodelled filter would
            make the filtering tests assert nothing.
    """
    if clause == _MATCH_NOTHING_CLAUSE:
        return False

    # A parenthesised group is a disjunction of conjunctions of the leaf shapes
    # below — which is exactly what a collection selection is, and nothing else
    # emits one. Recursing keeps this a composition of modelled shapes rather
    # than a second, looser parser: an unrecognised leaf still raises.
    group = _unwrap_group(clause)
    if group is not None:
        return any(
            _matches_conjunction(row, part) for part in _split_top_level(group, _OR_SEPARATOR)
        )

    id_match = _ID_FILTER_PATTERN.match(clause)
    if id_match is not None:
        return bool(row["id"] == _unquote(id_match.group("value")))

    membership_match = _MEMBERSHIP_FILTER_PATTERN.match(clause)
    if membership_match is not None:
        wanted = {
            _unquote(literal)
            for literal in _SQL_LITERAL_PATTERN.findall(membership_match.group("literals"))
        }
        present = row[membership_match.group("column")] in wanted
        return not present if membership_match.group("negated") else bool(present)

    caption_match = _CAPTION_FILTER_PATTERN.match(clause)
    if caption_match is not None:
        needle = _unescape_like(caption_match.group("needle"))
        haystack = " ".join(row["captions"]).lower()
        return needle in haystack

    raise AssertionError(f"FakeTable cannot evaluate filter clause {clause!r}")


def _matches_conjunction(row: dict[str, Any], expression: str) -> bool:
    """Evaluate a conjunction of clauses against a row.

    Args:
        row: Candidate row.
        expression: One or more clauses joined with ``AND``.

    Returns:
        Whether the row satisfies every clause.
    """
    return all(
        _matches_clause(row, clause) for clause in _split_top_level(expression, _CLAUSE_SEPARATOR)
    )


class FakeQuery:
    """In-memory stand-in for a LanceDB query builder.

    Implements only the chain ``LanceDBImageRepository`` uses — ``metric``, ``where``,
    ``select``, ``offset``, ``limit``, ``to_list``, ``to_arrow`` — and rejects
    anything it does not understand, so a change in the repository that this
    double silently cannot model fails the suite instead of passing it.
    """

    def __init__(self, rows: list[dict[str, Any]], vector: NDArray[np.float32] | None) -> None:
        """Bind the builder to the table's rows and an optional query vector.

        Args:
            rows: Every row in the table, in scan order.
            vector: Query embedding for a scoring search, or ``None`` for a
                plain scan.
        """
        self._rows = rows
        self._vector = vector
        self._where: str | None = None
        self._prefilter: bool | None = None
        self._columns: list[str] | None = None
        self._offset = 0
        self._limit: int | None = None

    def metric(self, name: str) -> FakeQuery:
        """Record the distance metric, asserting it is the one we index for."""
        if name != "cosine":
            raise AssertionError(f"Unexpected metric {name!r}; the index is cosine-only")
        return self

    def where(self, expression: str, prefilter: bool = True) -> FakeQuery:
        """Apply a filter expression.

        Args:
            expression: Predicate built by the repository.
            prefilter: Whether the store applies the filter before the vector
                search rather than after it. Recorded so :meth:`_resolve` can
                assert the repository never asks for the post-filter behaviour,
                which would silently return fewer hits than requested.
        """
        self._where = expression
        self._prefilter = prefilter
        return self

    def select(self, columns: list[str]) -> FakeQuery:
        """Restrict the projection to ``columns``."""
        self._columns = list(columns)
        return self

    def offset(self, count: int) -> FakeQuery:
        """Skip ``count`` rows."""
        self._offset = count
        return self

    def limit(self, count: int | None) -> FakeQuery:
        """Cap the result at ``count`` rows; ``None`` means unbounded."""
        self._limit = count
        return self

    def to_list(self) -> list[dict[str, Any]]:
        """Materialize the query as a list of projected row dicts."""
        return [self._project(row) for row in self._resolve()]

    def to_arrow(self) -> pa.Table:
        """Materialize the query as an Arrow table, as ``count_by_split`` expects."""
        rows = [self._project(row) for row in self._resolve()]
        columns = self._columns or []
        return pa.table({column: [row[column] for row in rows] for column in columns})

    def _resolve(self) -> list[dict[str, Any]]:
        """Apply filtering, scoring, ordering and windowing in LanceDB's order."""
        rows = self._rows
        if self._where is not None:
            if self._vector is not None and not self._prefilter:
                raise AssertionError(
                    "A filtered vector search must pre-filter; post-filtering would "
                    "truncate the ranking to whatever survives the global top-k"
                )
            rows = [row for row in rows if self._matches(row, self._where)]
        if self._vector is not None:
            scored = [(row, _cosine_distance(self._vector, row["vector"])) for row in rows]
            scored.sort(key=lambda pair: pair[1])
            rows = [{**row, "_distance": distance} for row, distance in scored]
        if self._offset:
            rows = rows[self._offset :]
        if self._limit is not None:
            rows = rows[: self._limit]
        return rows

    @staticmethod
    def _matches(row: dict[str, Any], expression: str) -> bool:
        """Evaluate a conjunction of the clause shapes the repository emits.

        Args:
            row: Candidate row.
            expression: Filter expression built by the repository.

        Returns:
            Whether the row satisfies every clause.
        """
        return _matches_conjunction(row, expression)

    def _project(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return only the selected columns, mirroring a real projection."""
        if self._columns is None:
            return {key: value for key, value in row.items() if key != "vector"}
        return {column: row[column] for column in self._columns}


class FakeTable:
    """In-memory stand-in for an open ``lancedb.table.Table``."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        """Store the rows this table will serve, in scan order.

        Args:
            rows: Row dicts carrying ``id``, ``file_name``, ``split``,
                ``captions`` and ``vector``.
        """
        self._rows = rows

    def count_rows(self, filter: str | None = None) -> int:
        """Return the row count, optionally restricted by a filter expression.

        The parameter shadows the builtin because the real
        ``Table.count_rows(filter=...)`` names it that way and the repository
        passes it by keyword; renaming it here would let a signature mismatch
        pass unnoticed.

        Args:
            filter: Predicate to count under, or ``None`` for every row.

        Returns:
            Number of matching rows.
        """
        if filter is None:
            return len(self._rows)
        return sum(1 for row in self._rows if FakeQuery._matches(row, filter))

    def search(self, query: NDArray[np.float32] | None = None) -> FakeQuery:
        """Start a query; passing a vector makes it a scoring search."""
        return FakeQuery(self._rows, query)


class FakeClipModel:
    """Stand-in for the CLIP bi-encoder that never loads a weight file.

    Records the keyword arguments of the most recent call so a test can assert
    the encoding contract the shared embedding space depends on.
    """

    def __init__(self) -> None:
        """Initialise with no recorded calls."""
        self.encode_calls: list[dict[str, object]] = []

    def encode(self, sentences: str, **kwargs: object) -> NDArray[np.float32]:
        """Return the fixed unit vector associated with ``sentences``.

        ``**kwargs`` is typed ``object`` rather than ``Any``: the recorded values
        are only ever compared, never called into, so the wider type costs
        nothing and keeps the file free of ``Any`` escape hatches.

        Args:
            sentences: The query string. Named to match the real
                ``SentenceTransformer.encode`` signature.
            **kwargs: Encoding options; recorded, not honoured.

        Returns:
            A ``(4,)`` float32 vector.
        """
        self.encode_calls.append({"sentences": sentences, **kwargs})
        vector = _QUERY_VECTORS.get(sentences, _DEFAULT_QUERY_VECTOR)
        return np.asarray(vector, dtype=np.float32)


@pytest.fixture
def image_rows() -> list[dict[str, Any]]:
    """Three rows spanning two splits, one with fewer than five captions.

    The caption counts differ on purpose: ``ImageDetail`` documents its captions
    as "usually five, not always", and a fixture where every row has exactly
    five would never exercise that.
    """
    return [
        {
            "id": DOG_ID,
            "file_name": f"{DOG_ID}.jpg",
            "split": "train",
            "captions": [
                "A black dog runs through shallow water .",
                "A dog splashes through a stream .",
                "A wet black dog running in water .",
                "A dog running through water outdoors .",
                "A black dog is playing in a creek .",
            ],
            "vector": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        },
        {
            "id": SLIDE_ID,
            "file_name": f"{SLIDE_ID}.jpg",
            "split": "train",
            "captions": [
                "Two children play on a red slide .",
                "Kids going down a playground slide .",
                "Children at a playground in the sun .",
                "A child slides down a red slide .",
            ],
            "vector": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        },
        {
            "id": CLIMB_ID,
            "file_name": f"{CLIMB_ID}.jpg",
            "split": "validation",
            "captions": [
                "A man climbs a steep rock face .",
                "A climber scales a cliff .",
                "A person climbing a large rock .",
                "A man in a harness on a rock wall .",
                "A rock climber reaching for a hold .",
            ],
            "vector": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        },
    ]


@pytest.fixture
def fake_table(image_rows: list[dict[str, Any]]) -> FakeTable:
    """The in-memory table the stubbed LanceDB connection hands back."""
    return FakeTable(image_rows)


@pytest.fixture
def fake_clip_model() -> FakeClipModel:
    """The encoder double, exposed so tests can inspect how it was called."""
    return FakeClipModel()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """An empty ``data/`` layout under ``tmp_path``.

    The lifespan and ``LanceDBImageRepository.open`` both refuse to start when these
    directories are absent, and that guard is worth keeping in the loop — so the
    directories are created, but stay empty. No dataset file is ever read: the
    JPEGs would only be touched by a request to ``/images/...``, which this suite
    does not make, and the table itself is served from memory.
    """
    (tmp_path / "images").mkdir()
    (tmp_path / "lancedb").mkdir()
    return tmp_path


#: Positions for the fixture rows, as ``scripts/project.py`` would write them.
#: Deliberately spread across quadrants so a test can tell them apart by sign.
PROJECTION_POSITIONS: Final[dict[str, list[float]]] = {
    DOG_ID: [0.1, 0.2],
    SLIDE_ID: [-0.5, 0.4],
    CLIMB_ID: [0.9, -0.3],
}

PROJECTION_VARIANCE_RATIO: Final = [0.0857, 0.0584]


@pytest.fixture
def projection_document() -> dict[str, Any]:
    """A projection artefact covering every fixture row.

    Shaped exactly like the file ``scripts/project.py`` writes, so the
    repository's parsing is exercised rather than bypassed.
    """
    return {
        "method": "pca",
        "dimensions": 2,
        "count": len(PROJECTION_POSITIONS),
        "explained_variance_ratio": PROJECTION_VARIANCE_RATIO,
        "points": dict(PROJECTION_POSITIONS),
    }


@pytest.fixture
def written_projection(data_dir: Path, projection_document: dict[str, Any]) -> Path:
    """Write the projection into the data directory before the app starts.

    The lifespan reads it once at startup, so this has to land on disk before
    the client fixture enters its context — which it does, because ``client``
    depends on this fixture through ``client_with_projection``.
    """
    path = data_dir / "projection.json"
    path.write_text(json.dumps(projection_document), encoding="utf-8")
    return path


#: A duplicate pair inside one split and one that straddles two, so both the
#: "near-duplicate" and the "cross-split" filters have something to select.
ANALYSIS_NN: Final[dict[str, tuple[str, float]]] = {
    DOG_ID: (SLIDE_ID, 0.991),
    SLIDE_ID: (DOG_ID, 0.991),
    CLIMB_ID: (DOG_ID, 0.962),
}

#: CLIMB's captions retrieve it worst, so it is the one the weak-captions
#: filter should pick out of a three-row corpus.
ANALYSIS_CAPTION_RANK: Final[dict[str, int]] = {DOG_ID: 1, SLIDE_ID: 3, CLIMB_ID: 4000}


@pytest.fixture
def analysis_document() -> dict[str, Any]:
    """A data-quality artefact covering every fixture row.

    Shaped exactly like the file ``scripts/analyze.py`` writes, so the
    repository's parsing and its derived id sets are exercised rather than
    bypassed.
    """
    return {
        "corpus_size": 3,
        "duplicate_threshold": 0.95,
        "duplicate_pairs": [
            {
                "a": DOG_ID,
                "b": SLIDE_ID,
                "a_split": "train",
                "b_split": "train",
                "similarity": 0.991,
                "cross_split": False,
            },
            {
                "a": DOG_ID,
                "b": CLIMB_ID,
                "a_split": "train",
                "b_split": "validation",
                "similarity": 0.962,
                "cross_split": True,
            },
        ],
        "caption_retrieval": {"r_at_1": 0.29, "r_at_5": 0.57, "r_at_10": 0.69, "captions": 14},
        "images": {
            image_id: {
                "nn_id": neighbour,
                "nn_similarity": score,
                "caption_rank": ANALYSIS_CAPTION_RANK[image_id],
            }
            for image_id, (neighbour, score) in ANALYSIS_NN.items()
        },
    }


@pytest.fixture
def written_analysis(data_dir: Path, analysis_document: dict[str, Any]) -> Path:
    """Write the analysis into the data directory before the app starts."""
    path = data_dir / "analysis.json"
    path.write_text(json.dumps(analysis_document), encoding="utf-8")
    return path


@pytest.fixture
def client_with_analysis(written_analysis: Path, client: TestClient) -> TestClient:
    """A client whose application found a data-quality artefact at startup.

    The plain ``client`` fixture has none, on purpose: the analysis is optional,
    and most of the suite should keep proving the app works without it.
    """
    assert written_analysis.is_file()
    return client


#: Override ceiling used by :func:`client_with_a_tiny_move_cap`. One, because
#: the fixture corpus has three rows and the refusal has to be reachable
#: without inventing a thousand of them.
TINY_MOVE_CAP: Final = 1


@pytest.fixture
def tiny_move_cap(monkeypatch: pytest.MonkeyPatch) -> int:
    """Shrink ``max_collection_overrides`` before the app is constructed.

    Declared *before* ``client`` in the signature of the fixture below, on the
    same reasoning as ``written_analysis``: the setting is read once when the
    lifespan builds the settings singleton, so it has to be in the environment
    before the client fixture clears the cache.

    Returns:
        The ceiling that will be in force.
    """
    monkeypatch.setenv("CORPUSLENS_MAX_COLLECTION_OVERRIDES", str(TINY_MOVE_CAP))
    return TINY_MOVE_CAP


@pytest.fixture
def client_with_a_tiny_move_cap(tiny_move_cap: int, client: TestClient) -> TestClient:
    """A client whose move ceiling is one image."""
    assert tiny_move_cap == TINY_MOVE_CAP
    return client


@pytest.fixture
def client_with_projection(written_projection: Path, client: TestClient) -> TestClient:
    """A client whose application found a projection at startup.

    The plain ``client`` fixture deliberately has none: the map is optional, and
    the majority of the suite should keep proving the app works without it.
    """
    assert written_projection.is_file()
    return client


def install_fake_backends(
    mocker: MockerFixture, fake_table: FakeTable, fake_clip_model: FakeClipModel
) -> None:
    """Point the lifespan's two implementation choices at in-memory doubles.

    The lifespan is the only place that names ``LanceDBImageRepository`` and
    ``ClipEmbeddingService``, so this is the only place a test has to intercept
    to run the whole stack without a database or a checkpoint. Note what is
    *not* faked: ``ClipEmbeddingService`` itself is real, wrapping a fake model,
    so the encoding contract every ranking depends on — ``normalize_embeddings``
    and the rest — is exercised rather than stubbed over.

    Args:
        mocker: The pytest-mock fixture owning the patches.
        fake_table: In-memory stand-in for the LanceDB table.
        fake_clip_model: In-memory stand-in for the bi-encoder.
    """
    connection = mocker.Mock(name="lancedb_connection")
    connection.table_names.return_value = ["images"]
    connection.open_table.return_value = fake_table
    mocker.patch("lancedb.connect", return_value=connection)
    mocker.patch.object(
        ClipEmbeddingService,
        "load",
        return_value=ClipEmbeddingService(fake_clip_model, "cpu"),
    )


@pytest.fixture
def client(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
    fake_table: FakeTable,
    fake_clip_model: FakeClipModel,
) -> Iterator[TestClient]:
    """Yield a client for an app whose lifespan ran against the doubles.

    ``TestClient`` is entered as a context manager so the real lifespan executes
    — the resource bundle on ``app.state``, the dependency wiring and the
    shutdown path are all covered rather than bypassed with
    ``dependency_overrides``.

    ``get_settings`` is ``lru_cache``d, so its cache is cleared on both sides of
    the test: once so the patched ``CORPUSLENS_DATA_DIR`` is picked up, and once
    afterwards so a cached test configuration cannot leak into another test or
    into an interpreter that later imports the app for real.
    """
    monkeypatch.setenv("CORPUSLENS_DATA_DIR", str(data_dir))
    get_settings.cache_clear()

    install_fake_backends(mocker, fake_table, fake_clip_model)

    with TestClient(create_app()) as test_client:
        yield test_client

    get_settings.cache_clear()
