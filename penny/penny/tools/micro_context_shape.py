"""The DECLARED output shape of a micro-context, and the one parse that reads it.

A micro-context's wire format is TAGGED LINES, not JSON, so its contract cannot be
a Pydantic ``args_model`` the way a tool call's arguments are: a model would need a
parser either way, and what actually varies between customers is **line grammar** —
which tag opens a line, whether the line repeats once per item, whether its payload
is that line or the whole remainder of the draw, how the payload splits into fields,
and which fields may be absent.  So the shape is a small declarative line grammar
expressed as frozen Pydantic models — the ``validation.conditions`` CATALOG pattern
(the contract is data, walked by one dispatcher), not the ``Tool.args_model``
pattern (a validator class per customer).

Three things fall out of declaring it as data:

1. **The prompt's contract block is RENDERED from the declaration**
   (:func:`render_line`), so the tag, the separators, and the field placeholders the
   model is shown come from the same object the parser walks.  The prompt and the
   parser can no longer be two independent copies of one rule.
2. **One parse** (:func:`parse_draw`) serves every customer; a customer reads named
   fields off a :class:`ParsedDraw` and never partitions a string itself.
3. **A malformed line stops being read as good data.**  Before, a ``PARAM`` line
   whose separator came out as an en-dash where the parser wanted an em-dash
   partitioned to nothing and handed the ENTIRE remainder back as the parameter's
   semantic name — a 60-character "name" carrying its own description, persisted as
   a skill's binding key, with no reroll because the parse "succeeded".  Tolerance
   (below) now reaches that draw, and the plausibility gate
   (:attr:`FieldShape.NAME`) refuses whatever tolerance doesn't.

:class:`LineRole` is what keeps the best-effort rules intact while the strict ones
tighten: a REQUIRED line missing or malformed invalidates the draw (a reroll), while
an OPTIONAL or PER_ITEM line missing OR malformed is simply ABSENT — never a
verdict, never a reroll.  So "absence is never a verdict" (#1770) and "the
labeller's per-candidate lines are best-effort by design" are declared properties of
those lines rather than conventions each bespoke parser had to remember.

Dependency-light leaf: pydantic + ``text_validity`` only, so the tools package and
anything that declares a shape import it without a cycle.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from penny.text_validity import is_blank


class Separator(StrEnum):
    """How one field's payload ENDS — the two delimiters the tagged-line grammar
    uses, and the only two a contract block may render.

    The distinction is load-bearing twice over.  ``COLON`` attaches to the token
    before it (``<current name>:``).  ``DASH`` is WHITESPACE-DELIMITED (`` — ``),
    which is exactly what makes tolerating its variants safe: a hyphen inside a name
    (``memory-2``, ``aurora-deck-2``) carries no spaces around it, so it is never
    mistaken for the separator."""

    COLON = ":"
    DASH = "—"


class FieldShape(StrEnum):
    """What a carved field must LOOK like for the line to be plausible.

    ``TEXT`` is prose — any non-blank value.  ``NAME`` is a token, not a sentence: a
    run of word characters, spaces and hyphens, nothing else.  That is the contract
    block's own instruction ("a single lowercase word or snake_case") relaxed to the
    two things the downstream hardener already fixes (case and spaces), and it is
    what makes a "name" that is really a name-plus-its-own-description — punctuated,
    parenthesised, quoted — a malformed line rather than a value."""

    TEXT = "text"
    NAME = "name"


class LineRole(StrEnum):
    """What a declared line contributes to a valid draw — the four cases every
    customer's contract is built from, and the axis that keeps strictness and
    best-effort tolerance from being confused for each other.

    ``REQUIRED`` — absent or malformed invalidates the whole draw (a reroll, then an
    honest typed failure).  ``ALTERNATIVE`` — at least one of the lines so marked
    must be present and well-formed (the extraction contract's "the first line must
    open with one of these two tags").  ``OPTIONAL`` — a singular line that may be
    absent; malformed reads as absent, never as a violation.  ``PER_ITEM`` — repeats
    once per item; a malformed one is DROPPED, costing that one item its line and
    nothing else."""

    REQUIRED = "required"
    ALTERNATIVE = "alternative"
    OPTIONAL = "optional"
    PER_ITEM = "per_item"


class PayloadSpan(StrEnum):
    """How much of the draw a tagged line's payload covers — the rest of its own
    line, or EVERYTHING after the tag (a digest / item-per-line list served whole)."""

    LINE = "line"
    REMAINDER = "remainder"


class LineAnchor(StrEnum):
    """Where the tag may sit — opening the draw (the extraction contract's
    first-line rule, which also makes its two alternatives mutually exclusive), or
    anywhere among the lines."""

    OPENS_DRAW = "opens_draw"
    ANYWHERE = "anywhere"


class FieldSpec(BaseModel):
    """One named field carved out of a tagged line's payload.

    ``placeholder`` is the ``<…>`` the contract block shows the model, and
    ``separator`` is what ends this field — so the rendered instruction and the
    parse are the same declaration read twice.  The last field of a line declares no
    separator: it takes whatever payload is left."""

    model_config = ConfigDict(frozen=True)

    name: str
    placeholder: str
    shape: FieldShape = FieldShape.TEXT
    required: bool = True
    separator: Separator | None = None


class LineSpec(BaseModel):
    """One tagged line of a micro-context's declared output contract."""

    model_config = ConfigDict(frozen=True)

    tag: str
    fields: tuple[FieldSpec, ...]
    role: LineRole = LineRole.REQUIRED
    span: PayloadSpan = PayloadSpan.LINE
    anchor: LineAnchor = LineAnchor.ANYWHERE


class MicroContextShape(BaseModel):
    """One customer's whole declared output shape — the lines it may emit, what each
    one contributes, and how each one's payload carves into named fields."""

    model_config = ConfigDict(frozen=True)

    lines: tuple[LineSpec, ...]


class ParsedLine(BaseModel):
    """One matched line: the tag that selected it and the named fields carved from
    its payload.  An optional field the draw omitted is simply absent from
    ``fields``."""

    model_config = ConfigDict(frozen=True)

    tag: str
    fields: dict[str, str]


class ParsedDraw(BaseModel):
    """A draw read against its declared shape: the singular lines keyed by tag, and
    the per-item lines in draw order.  Customers read it by FIELD NAME (:meth:`field`)
    — there is no string left to partition."""

    model_config = ConfigDict(frozen=True)

    singles: dict[str, ParsedLine] = {}
    items: tuple[ParsedLine, ...] = ()

    def field(self, tag: str, name: str) -> str | None:
        """The named field of the singular line tagged ``tag``, or ``None`` when that
        line was absent from the draw (or carried no such field, an optional one
        having been omitted)."""
        line = self.singles.get(tag)
        return line.fields.get(name) if line is not None else None


# ── Tolerance: declared ONCE, deliberately ────────────────────────────────────
# What a draw is allowed to get cosmetically wrong.  Each parser used to be
# accidentally strict in a different place; these are the variations we have
# actually seen a model emit, normalised in one spot so every customer tolerates
# the same set and no more.
#
# A separator dash is WHITESPACE-DELIMITED, so folding its variants can never touch
# a hyphen inside a name.  The observed failure — an en-dash where the parser wanted
# an em-dash — is this line.
_DASH_SEPARATOR_RE = re.compile(r"\s+[-‐‑‒–—―−]+\s+")

# Markdown decoration on the line that carries the TAG: a leading list marker, and
# ``**bold**`` anywhere in it (the labeller emits ``**PARAM …**`` lines and
# ``**USER-GIVEN PARAMETERS**`` headers).  Applied to the tag's own line only, so a
# multi-line EXTRACTED value's later lines survive verbatim — a markdown digest is
# the value, not grammar.
_LIST_MARKER_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
_BOLD_DECORATION = "**"

# One matched pair of quotes WRAPPING a whole field value — straight or smart.
# Positional and conservative: a quoted phrase inside a description is untouched.
# This is what lets ``STATE: "apply"`` decide instead of failing membership.
_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
)

# A name is a token, not a sentence: word characters, spaces and hyphens only.
_NAME_RE = re.compile(r"[\w -]+")


def render_line(spec: LineSpec) -> str:
    """The line as the CONTRACT BLOCK shows it to the model — the tag, then each
    field's placeholder followed by the separator that ends it.

    This is what makes the prompt and the parser one declaration instead of two
    copies: the separator the model is told to write is literally the one
    :func:`parse_draw` splits on."""
    body = "".join(
        f"{field.placeholder}{_render_separator(field.separator)}" for field in spec.fields
    )
    return f"{spec.tag} {body}"


def parse_draw(draw: str, shape: MicroContextShape) -> ParsedDraw | None:
    """The one validate step: ``draw`` read against its declared ``shape``.

    ``None`` is a CONTRACT VIOLATION — a REQUIRED line missing or malformed, or no
    ALTERNATIVE matched — which the handler rerolls on the unchanged context and then
    fails honestly.  An OPTIONAL or PER_ITEM line that is missing *or malformed* is
    simply absent from the result, never a violation: that is what keeps "absence is
    never a verdict" true by declaration rather than by convention."""
    lines = draw.strip().splitlines()
    singles: dict[str, ParsedLine] = {}
    for spec in shape.lines:
        if spec.role is LineRole.PER_ITEM:
            continue
        found = _find_line(lines, spec)
        if found is not None:
            singles[spec.tag] = found
        elif spec.role is LineRole.REQUIRED:
            return None
    if not _alternatives_satisfied(shape, singles):
        return None
    return ParsedDraw(singles=singles, items=_find_items(lines, shape))


def _find_line(lines: list[str], spec: LineSpec) -> ParsedLine | None:
    """The first line carrying ``spec``'s tag, carved into its fields — or ``None``
    when no line carries it or the one that does is malformed."""
    candidates = lines[:1] if spec.anchor is LineAnchor.OPENS_DRAW else lines
    for index, line in enumerate(candidates):
        payload = _tag_payload(_undecorate(line), spec.tag)
        if payload is None:
            continue
        if spec.span is PayloadSpan.REMAINDER:
            payload = "\n".join([payload, *lines[index + 1 :]]).strip()
        fields = _carve(payload, spec.fields)
        if fields is not None:
            return ParsedLine(tag=spec.tag, fields=fields)
    return None


def _find_items(lines: list[str], shape: MicroContextShape) -> tuple[ParsedLine, ...]:
    """Every PER_ITEM line, in draw order, carved into its fields.  A malformed one
    is DROPPED — those lines are best-effort by design, so a bad line costs its own
    item a verdict and nothing else.  What changed is that a malformed line is no
    longer accepted as good data."""
    specs = [spec for spec in shape.lines if spec.role is LineRole.PER_ITEM]
    items: list[ParsedLine] = []
    for line in lines:
        undecorated = _undecorate(line)
        for spec in specs:
            payload = _tag_payload(undecorated, spec.tag)
            if payload is None:
                continue
            fields = _carve(payload, spec.fields)
            if fields is not None:
                items.append(ParsedLine(tag=spec.tag, fields=fields))
            break
    return tuple(items)


def _alternatives_satisfied(shape: MicroContextShape, singles: dict[str, ParsedLine]) -> bool:
    """True unless the shape declares ALTERNATIVE lines and none of them matched."""
    alternatives = [spec.tag for spec in shape.lines if spec.role is LineRole.ALTERNATIVE]
    return not alternatives or any(tag in singles for tag in alternatives)


def _carve(payload: str, fields: tuple[FieldSpec, ...]) -> dict[str, str] | None:
    """A payload split into its declared fields, or ``None`` when the line does not
    match the declared shape — a required field missing, blank, or failing its
    :class:`FieldShape`.

    A separator that isn't in the payload ends the line: the field takes what is
    left and every field after it is ABSENT, which is fine only if they are all
    optional (so ``PARAM x: y`` with no description parses, while ``PLACEHOLDER x``
    with no description does not)."""
    values: dict[str, str] = {}
    remaining: str | None = payload
    for spec in fields:
        if remaining is None:
            if spec.required:
                return None
            continue
        value, remaining = _next_field(remaining, spec)
        if _field_is_valid(value, spec):
            values[spec.name] = value
        elif spec.required:
            return None
    return values


def _next_field(remaining: str, spec: FieldSpec) -> tuple[str, str | None]:
    """This field's raw value and what is left after it — split on the separator the
    field declares, or the whole remainder when it declares none."""
    if spec.separator is None:
        return _unquote(remaining.strip()), None
    value, rest = _split_on(remaining, spec.separator)
    return _unquote(value.strip()), rest


def _field_is_valid(value: str, spec: FieldSpec) -> bool:
    """Whether a carved value is plausible as this field — non-blank, and a token
    rather than a sentence when the field declares :attr:`FieldShape.NAME`."""
    if is_blank(value):
        return False
    if spec.shape is FieldShape.NAME:
        return _NAME_RE.fullmatch(value) is not None
    return True


def _split_on(payload: str, separator: Separator) -> tuple[str, str | None]:
    """Partition ``payload`` at the first ``separator``.  The second member is
    ``None`` when the separator is absent, so a caller can tell "no field follows"
    from "an empty one follows".

    The DASH is the tolerant special case — it is whitespace-delimited, so any dash
    variant standing between spaces splits while a hyphen inside a name never does.
    Every other separator is its own literal character."""
    if separator is Separator.DASH:
        found_dash = _DASH_SEPARATOR_RE.search(payload)
        if found_dash is None:
            return payload, None
        return payload[: found_dash.start()], payload[found_dash.end() :]
    before, found, after = payload.partition(separator.value)
    return (before, after) if found else (payload, None)


def _render_separator(separator: Separator | None) -> str:
    """How a separator is WRITTEN in the contract block — the DASH whitespace-
    delimited (the same fact that makes its variants safe to tolerate), every other
    separator attached to the token before it.  A field that ends the line renders
    nothing."""
    if separator is None:
        return ""
    if separator is Separator.DASH:
        return f" {separator.value} "
    return f"{separator.value} "


def _tag_payload(line: str, tag: str) -> str | None:
    """The payload of ``line`` when it opens with ``tag``, else ``None``.

    A tag ending in ``:`` is its own delimiter; any other tag must be followed by
    whitespace, so a ``PARAMETRIC`` line is never read as a ``PARAM`` one."""
    if not line.startswith(tag):
        return None
    rest = line[len(tag) :]
    if tag.endswith(Separator.COLON.value):
        return rest.strip()
    if rest and not rest[0].isspace():
        return None
    return rest.strip()


def _undecorate(line: str) -> str:
    """A line stripped of the markdown decoration a draw is allowed to get wrong: a
    leading list marker and any ``**bold**``."""
    return _LIST_MARKER_RE.sub("", line.strip()).replace(_BOLD_DECORATION, "").strip()


def _unquote(value: str) -> str:
    """A field value with ONE matched pair of wrapping quotes removed."""
    for opener, closer in _QUOTE_PAIRS:
        if len(value) >= 2 and value.startswith(opener) and value.endswith(closer):
            return value[1:-1].strip()
    return value
