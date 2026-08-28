r"""Translation of domain filters into LanceDB predicate expressions.

Split out of :mod:`app.repositories.image_repository` so that module is about
*reaching* the table and this one is about *describing* what to keep. It is the
only place that knows the store speaks DataFusion SQL — the layers above work in
:class:`~app.models.domain.ImageFilter` terms and never see a quote.

The dialect notes below were each confirmed by executing them against the real
8 000-row table (CLAUDE.md §6), not recalled:

* ``array_to_string(captions, ' ')`` flattens the list column, which is what
  makes "any caption contains X" expressible at all.
* ``LOWER(...) LIKE '%x%'`` gives the case-insensitive substring match.
* ``ESCAPE '\\'`` is honoured, and is required: an unescaped ``%`` in the needle
  matches every row in the corpus, and an unescaped ``_`` matches any character.
* ``IN`` accepts a parenthesised list of quoted literals.
* ``false`` is accepted as a standalone predicate and matches nothing, which is
  what an empty id restriction has to compile to.
* The collection group below — ``(split IN (…) AND id NOT IN (…)) OR id IN (…)``
  — parses, counts and, critically, works as a **vector pre-filter**. An ``OR``
  inside a pre-filter was not something to assume; it was run against the real
  table before this code was written. See ``docs/api.md``.

**Cost note — measured, on the real 8 000-row table.** The collection group
embeds two id lists as literals on every filtered query, and the cost is linear
in their combined length:

===============  ==========  =================  ===================
overrides        predicate   ``count_rows``     search (pre-filter)
===============  ==========  =================  ===================
0                  0 KB           22 ms                50 ms
100                2.4 KB         46 ms                60 ms
1 000             24 KB          222 ms               245 ms
4 000             97 KB          847 ms               945 ms
8 000 (all)      194 KB        1 637 ms             1 788 ms
===============  ==========  =================  ===================

The earlier note here asked for this measurement before trading the literals for
a materialised id set. The answer is that no encoding rescues the tail: the
selection can be written as its members, as its complement, or as split plus
deltas, and a corpus split evenly between two collections still names at least
4 000 ids whichever is chosen — about a 2x saving on a 38x regression.

So the bound is a product decision rather than an optimisation, and it lives in
``Settings.max_collection_overrides`` (default 1 000, the last row above that is
still interactive). The overlay is the mechanism for a *delta* — a quarantine, a
holdout, a review queue. Re-partitioning a whole corpus is a different operation
and belongs in a newly ingested index, where the partition is a column again
rather than a literal; see ``docs/collections-next.md``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from app.models.domain import CollectionSelection, ImageFilter

#: Escape character introduced by :func:`escape_like_pattern` and declared to
#: the engine with ``ESCAPE``. Backslash is conventional, and SQL string
#: literals give it no meaning of their own, so it survives quoting unchanged.
LIKE_ESCAPE: Final = "\\"

#: Predicate for "keep nothing". Needed because ``IN ()`` is a syntax error, so
#: an empty id restriction cannot be expressed as a membership test.
MATCH_NOTHING: Final = "false"


def escape_sql_literal(value: str) -> str:
    """Escape a string for inclusion between single quotes.

    Callers already constrain these values at the request schema; this is
    defence in depth for the one place caller-supplied text reaches a query
    expression. SQL string literals escape a quote by doubling it.

    Args:
        value: Raw value to embed.

    Returns:
        The value with embedded single quotes doubled.
    """
    return value.replace("'", "''")


def escape_like_pattern(value: str) -> str:
    r"""Neutralise ``LIKE`` wildcards in a user-supplied substring.

    Without this, a caption filter of ``%`` matches the entire corpus and ``_``
    matches any single character — the user typed a literal, not a pattern.
    Verified against the real table: unescaped, ``%`` returns all 8 000 rows;
    escaped, it returns none.

    The escape character itself is doubled first, so a needle containing a
    backslash cannot smuggle in an escape sequence.

    Args:
        value: Raw substring to search for.

    Returns:
        The substring with ``\``, ``%`` and ``_`` escaped, for use in a pattern
        declared with ``ESCAPE '\'``.
    """
    escaped = value.replace(LIKE_ESCAPE, LIKE_ESCAPE * 2)
    for wildcard in ("%", "_"):
        escaped = escaped.replace(wildcard, LIKE_ESCAPE + wildcard)
    return escaped


def _quoted_list(values: Iterable[str]) -> str:
    """Render values as a comma-separated list of escaped SQL literals."""
    return ", ".join(f"'{escape_sql_literal(value)}'" for value in values)


def build_filter_expression(image_filter: ImageFilter | None) -> str | None:
    """Translate a domain filter into a predicate.

    Args:
        image_filter: The narrowing to apply, or ``None``.

    Returns:
        A predicate for ``where()`` / ``count_rows(filter=...)``, or ``None``
        when every row qualifies and the query should carry no filter at all.
    """
    if image_filter is None or image_filter.is_empty:
        return None

    clauses: list[str] = []

    if image_filter.splits:
        clauses.append(f"split IN ({_quoted_list(image_filter.splits)})")

    if image_filter.caption_contains:
        needle = escape_sql_literal(escape_like_pattern(image_filter.caption_contains.lower()))
        clauses.append(
            f"LOWER(array_to_string(captions, ' ')) LIKE '%{needle}%' ESCAPE '{LIKE_ESCAPE}'"
        )

    if image_filter.ids is not None:
        clauses.append(
            build_id_membership_expression(image_filter.ids) if image_filter.ids else MATCH_NOTHING
        )

    if image_filter.collection_selection is not None:
        clauses.append(build_collection_expression(image_filter.collection_selection))

    return " AND ".join(clauses)


def build_collection_expression(selection: CollectionSelection) -> str:
    """Build the predicate for a collection selection.

    A collection is "the image's split, unless an override says otherwise", so
    the predicate has two branches::

        ( split IN (split_names) AND id NOT IN (excluded) )  OR  id IN (moved_in)

    An image with no override matches iff its split is selected. One overridden
    *into* a selected collection is re-added by the right branch whatever its
    split. One overridden *out of* a selected built-in is removed by
    ``excluded``.

    The **parentheses around the whole group are load-bearing**. ``AND`` binds
    tighter than ``OR``, so without them the caller's ``" AND ".join`` produces
    ``… OR id IN (…) AND caption LIKE …``, which the engine reads as
    ``… OR (id IN (…) AND caption LIKE …)`` — the caption filter applies only to
    the moved-in branch and every other selected image comes back unfiltered.
    Measured against the real corpus: a ``train`` + ``dog`` query returns 1 527
    rows parenthesised and 5 999 without. It is silent, and it is wrong in the
    direction that looks like success.

    Args:
        selection: The resolved selection.

    Returns:
        A predicate, always parenthesised or otherwise safe to ``AND`` with its
        neighbours.
    """
    if not selection.split_names:
        # No built-in selected, so nothing is reachable via the split column;
        # only explicitly moved-in ids can match.
        if not selection.moved_in_ids:
            return MATCH_NOTHING
        return build_id_membership_expression(selection.moved_in_ids)

    split_clause = f"split IN ({_quoted_list(selection.split_names)})"
    if selection.excluded_ids:
        split_clause = f"{split_clause} AND id NOT IN ({_quoted_list(selection.excluded_ids)})"

    if not selection.moved_in_ids:
        # With no overrides in play this is byte-identical to the split-only
        # predicate emitted before collections existed — the no-regression
        # guarantee that makes this change additive.
        return split_clause if not selection.excluded_ids else f"({split_clause})"

    moved_in = build_id_membership_expression(selection.moved_in_ids)
    return f"(({split_clause}) OR {moved_in})"


def build_id_equality_expression(image_id: str) -> str:
    """Build the predicate that selects exactly one row by id."""
    return f"id = '{escape_sql_literal(image_id)}'"


def build_id_membership_expression(image_ids: Iterable[str]) -> str:
    """Build the predicate that selects a set of rows by id.

    One ``IN`` rather than a query per id: an export of a few hundred selected
    images would otherwise be a few hundred separate scans.

    Args:
        image_ids: Ids to match. Must be non-empty — an empty ``IN ()`` list is
            a syntax error, and callers know to skip the query entirely.

    Returns:
        The membership predicate.
    """
    return f"id IN ({_quoted_list(image_ids)})"
