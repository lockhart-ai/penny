"""``parse_draw`` (penny/tools/micro_context_shape.py): the ONE tagged-line
grammar every micro-context customer's parser shares (#1814).

Every existing micro-context test (``test_browse_micro_context.py``,
``test_notify_micro_context.py``) drives ``parse_draw`` indirectly, through a
real customer's ``MicroContextShape`` and a mocked model end to end — none of
them exercises the grammar itself in isolation. These tests build small
synthetic shapes with no customer behind them and call ``parse_draw``
directly, as a pure function: well-formed draws, a missing REQUIRED line, a
malformed line, an unknown tag, and the declared tolerances (ALTERNATIVE,
OPENS_DRAW, the whitespace-delimited DASH separator).
"""

from __future__ import annotations

from penny.tools.micro_context_shape import (
    FieldShape,
    FieldSpec,
    LineAnchor,
    LineRole,
    LineSpec,
    MicroContextShape,
    Separator,
    parse_draw,
)

# A REQUIRED line whose value must be a NAME (a token, not a sentence) — the
# plausibility gate that turns a punctuated "name" into a malformed line.
_NAME_LINE = LineSpec(
    tag="NAME:",
    fields=(FieldSpec(name="value", placeholder="<name>", shape=FieldShape.NAME),),
    role=LineRole.REQUIRED,
)
# An OPTIONAL singular line: absent or malformed is never a violation.
_NOTE_LINE = LineSpec(
    tag="NOTE:",
    fields=(FieldSpec(name="value", placeholder="<note>"),),
    role=LineRole.OPTIONAL,
)
# A PER_ITEM line with two fields split on the whitespace-delimited DASH.
_ITEM_LINE = LineSpec(
    tag="ITEM:",
    fields=(
        FieldSpec(name="key", placeholder="<key>", separator=Separator.DASH),
        FieldSpec(name="detail", placeholder="<detail>"),
    ),
    role=LineRole.PER_ITEM,
)
SAMPLE_SHAPE = MicroContextShape(lines=(_NAME_LINE, _NOTE_LINE, _ITEM_LINE))


def test_well_formed_draw_parses_every_declared_line():
    draw = (
        "NAME: alpha-beta\n"
        "NOTE: a side note\n"
        "ITEM: widget — the first widget\n"
        "ITEM: gadget — the second gadget"
    )
    parsed = parse_draw(draw, SAMPLE_SHAPE)
    assert parsed is not None
    assert parsed.field("NAME:", "value") == "alpha-beta"
    assert parsed.field("NOTE:", "value") == "a side note"
    assert [item.fields["key"] for item in parsed.items] == ["widget", "gadget"]
    assert [item.fields["detail"] for item in parsed.items] == [
        "the first widget",
        "the second gadget",
    ]
    assert parsed.malformed == 0


def test_missing_required_line_invalidates_the_whole_draw():
    """No ``NAME:`` line at all: the draw is a contract violation, not a
    partial result — ``parse_draw`` returns ``None``, the reroll signal."""
    draw = "NOTE: a side note\nITEM: widget — the first widget"
    assert parse_draw(draw, SAMPLE_SHAPE) is None


def test_malformed_required_line_invalidates_the_whole_draw():
    """A ``NAME:`` value that is a sentence, not a token, fails the
    :attr:`FieldShape.NAME` plausibility gate — the required field is
    invalid, so the line itself does not carve, and the draw is rejected
    exactly as if the tag were absent."""
    draw = "NAME: alpha (the first one)\nITEM: widget — detail"
    assert parse_draw(draw, SAMPLE_SHAPE) is None


def test_malformed_optional_line_is_absent_not_a_violation():
    """A blank ``NOTE:`` value fails to carve, but OPTIONAL means malformed
    reads as simply absent — the rest of the draw still parses (#1770:
    absence is never a verdict)."""
    draw = "NAME: alpha\nNOTE:   \nITEM: widget — detail"
    parsed = parse_draw(draw, SAMPLE_SHAPE)
    assert parsed is not None
    assert parsed.field("NAME:", "value") == "alpha"
    assert parsed.field("NOTE:", "value") is None


def test_malformed_per_item_line_is_dropped_and_counted():
    """An ``ITEM:`` line missing its DASH separator cannot carve its second
    field; PER_ITEM drops that one item rather than failing the draw, but the
    drop is COUNTED (#1830) — a customer that mints its own items from an
    offered set relies on the count to notice the gap."""
    draw = "NAME: alpha\nITEM: widget only no dash\nITEM: gadget — second detail"
    parsed = parse_draw(draw, SAMPLE_SHAPE)
    assert parsed is not None
    assert parsed.malformed == 1
    assert [item.fields["key"] for item in parsed.items] == ["gadget"]
    assert parsed.items[0].fields["detail"] == "second detail"


def test_unknown_tag_is_simply_invisible_to_the_grammar():
    """A line whose tag the shape never declared is neither an error nor a
    dropped item — it carries no declared tag, so it is not counted as
    malformed and the rest of the draw parses normally."""
    draw = "NAME: alpha\nUNKNOWN: something nobody declared\nITEM: widget — detail"
    parsed = parse_draw(draw, SAMPLE_SHAPE)
    assert parsed is not None
    assert parsed.field("NAME:", "value") == "alpha"
    assert len(parsed.items) == 1
    assert parsed.malformed == 0


# ── ALTERNATIVE: at least one of the marked lines must match ──────────────────

_ALT_A = LineSpec(
    tag="FOO:", fields=(FieldSpec(name="value", placeholder="<v>"),), role=LineRole.ALTERNATIVE
)
_ALT_B = LineSpec(
    tag="BAR:", fields=(FieldSpec(name="value", placeholder="<v>"),), role=LineRole.ALTERNATIVE
)
ALTERNATIVE_SHAPE = MicroContextShape(lines=(_ALT_A, _ALT_B))


def test_alternative_lines_require_at_least_one_match():
    assert parse_draw("nothing recognizable here", ALTERNATIVE_SHAPE) is None


def test_alternative_satisfied_by_either_line_alone():
    parsed = parse_draw("FOO: hello", ALTERNATIVE_SHAPE)
    assert parsed is not None
    assert parsed.field("FOO:", "value") == "hello"
    assert parsed.field("BAR:", "value") is None


# ── OPENS_DRAW: the tag must open the draw, not merely appear in it ──────────

_OPEN_LINE = LineSpec(
    tag="HEAD:",
    fields=(FieldSpec(name="value", placeholder="<v>"),),
    role=LineRole.REQUIRED,
    anchor=LineAnchor.OPENS_DRAW,
)
OPEN_SHAPE = MicroContextShape(lines=(_OPEN_LINE,))


def test_opens_draw_anchor_matches_on_the_first_line():
    parsed = parse_draw("HEAD: hello", OPEN_SHAPE)
    assert parsed is not None
    assert parsed.field("HEAD:", "value") == "hello"


def test_opens_draw_anchor_rejects_the_tag_appearing_later():
    """The same tag, present but NOT first, is a REQUIRED line that never
    matches — the draw is invalidated exactly as if the tag were absent."""
    draw = "Some preamble line.\nHEAD: hello"
    assert parse_draw(draw, OPEN_SHAPE) is None


# ── DASH separator tolerance: whitespace-delimited, hyphen-in-name-safe ───────


def test_dash_separator_tolerates_an_en_dash_and_never_splits_a_hyphenated_name():
    """The declared separator is an em-dash, but the tolerant DASH split
    accepts the en-dash variant a model actually emits — and the hyphen
    inside the item's own name is never mistaken for it, because the
    separator match requires surrounding whitespace."""
    draw = "NAME: alpha\nITEM: aurora-deck-2 – a detail here"
    parsed = parse_draw(draw, SAMPLE_SHAPE)
    assert parsed is not None
    assert len(parsed.items) == 1
    assert parsed.items[0].fields["key"] == "aurora-deck-2"
    assert parsed.items[0].fields["detail"] == "a detail here"
