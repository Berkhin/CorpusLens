"""Unit tests for the predicate builder.

The rest of the suite drives filtering through ``TestClient`` and lets the
LanceDB double evaluate whatever expression comes out. That proves the filter
*selects* the right rows, but not that the *text* is what we think it is — and
two properties of the collection predicate are about the text itself:

* with no image moved, it must be byte-identical to what the code emitted before
  collections existed, which is what makes this change additive rather than a
  rewrite;
* the group must be parenthesised, or the ``AND`` that joins it to the caption
  clause binds tighter than its own ``OR`` and the caption filter silently stops
  applying to most of the result.

Neither is visible from the API boundary until it is already wrong, so both are
asserted here on the string.
"""

from __future__ import annotations

from app.models.domain import CollectionSelection, ImageFilter
from app.repositories.filters import build_filter_expression

MOVED_ID = "1000268201_693b08cb0e"
OTHER_ID = "1001773457_577c3a7d70"


def test_a_split_filter_is_unchanged_by_the_collection_machinery() -> None:
    """The pre-collections predicate still compiles to exactly its old text."""
    expression = build_filter_expression(ImageFilter(splits=("train",)))

    assert expression == "split IN ('train')"


def test_selecting_a_builtin_with_no_overrides_emits_the_old_split_predicate() -> None:
    """Zero overrides must be byte-identical to filtering by the split directly.

    This is the no-regression guarantee. If this test ever needs updating, the
    change has stopped being additive and every query in the application just
    got a new shape.
    """
    expression = build_filter_expression(
        ImageFilter(
            collections=("train",),
            collection_selection=CollectionSelection(split_names=("train",)),
        )
    )

    assert expression == "split IN ('train')"


def test_a_user_collection_with_no_builtin_selected_is_a_plain_id_list() -> None:
    """Nothing is reachable through the split column, so only moved-in ids match."""
    expression = build_filter_expression(
        ImageFilter(
            collections=("abc123",),
            collection_selection=CollectionSelection(moved_in_ids=(MOVED_ID,)),
        )
    )

    assert expression == f"id IN ('{MOVED_ID}')"


def test_an_empty_selection_is_unsatisfiable_rather_than_ignored() -> None:
    """An unknown or empty collection keeps nothing, exactly like a missing analysis."""
    expression = build_filter_expression(
        ImageFilter(collections=("gone",), collection_selection=CollectionSelection())
    )

    assert expression == "false"


def test_the_collection_group_is_parenthesised() -> None:
    """The OR group must be wrapped before anything is ANDed onto it.

    Measured against the real 8 000-row corpus: a train+dog query returns 1 527
    rows with the parentheses and 5 999 without, because ``AND`` binds tighter
    than ``OR`` and the caption clause ends up applying only to the moved-in
    branch. It fails silently and in the direction that looks like success.
    """
    expression = build_filter_expression(
        ImageFilter(
            caption_contains="dog",
            collections=("train", "abc123"),
            collection_selection=CollectionSelection(
                split_names=("train",),
                moved_in_ids=(MOVED_ID,),
                excluded_ids=(OTHER_ID,),
            ),
        )
    )

    assert expression is not None
    caption, separator, group = expression.partition(" AND (")
    assert separator, "the collection group must be joined with AND"
    assert caption.startswith("LOWER(array_to_string(captions, ' ')) LIKE '%dog%'")
    assert f"({group}" == (
        f"((split IN ('train') AND id NOT IN ('{OTHER_ID}')) OR id IN ('{MOVED_ID}'))"
    )
    # Balanced, so the OR cannot reach past the group and swallow the caption.
    assert expression.count("(") == expression.count(")")


def test_a_collection_filter_is_not_empty() -> None:
    """``is_empty`` must see the collections field.

    Missing this does not produce a wrong predicate — it produces *no* predicate:
    ``build_filter_expression`` short-circuits on an empty filter, pagination
    reports the corpus total, and every point on the map is marked as matching.
    """
    assert not ImageFilter(collections=("train",)).is_empty
    assert ImageFilter().is_empty
