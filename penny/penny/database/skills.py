"""The skill substrate — structured steps, provenance-inferred parameters, and the
steps→text render (#1590, stage ④ of #1562 / epic #1554).

A **skill** is a certified-by-execution script distilled from ONE demonstrated
run: an ordered list of structured steps in the ``LoggedToolCall`` shape (#1578)
plus declared *parameters* (SKILL-level inputs, semantically named + described,
not tool-arg echoes — #1668).  It is authored only by the framework — there is no
``skill_create`` tool.  The run-end extractor (``penny.skill_extraction``, #1658)
snapshots a qualifying chat run's own ledger, copying ALL its succeeded,
non-``done`` tool-call ordinals out, never re-emitting them.  Each argument leaf of
a copied call is factored by **provenance** — derived STRUCTURALLY from the ledger,
never by matching the user's prose (#1659):

* a value that **equals, fills whole tokens of, or wraps** a prior selected step's
  result → a **binding** (rendered "the value from step N"), because in the
  source run it *came from* that step (a wrapped result binds too — the arg
  ``Price: $499`` over a browse that returned ``$499``; a value sliced out of the
  middle of a token does NOT, since a fetched page contains most short strings by
  accident — #1809);
* **every other string leaf** → a **candidate** required parameter (the model binds
  it per instantiation); identical values collapse to ONE shared candidate.  A
  parameter is ``required`` by construction — an unbound one is a loud refusal at
  instantiation, never a silent default (no-silent-fallbacks).  Candidates get
  arg-derived names at distill; the run-end labelling micro-context then NAMES each
  one and says what belongs there each run (#1668/#1828).

**Every** unexplained leaf goes through that one process, whatever tool it sits on
and whatever argument it fills (#1783).  A skill is a learned sequence of ARBITRARY
tool calls, so nothing here knows which tools a skill contains: there is no write
target, no privileged argument, no tool whitelist.  Several reads, writes, browses
and plugin calls in one routine are not special cases — they are more leaves through
the same process.

That candidate rule was a **default, not a determination** — it assumed an unexplained
value came from the user, which is false for a value the model derived-and-transformed
from a result or invented outright (neither shares a literal span with what produced
it, so no substring test can reach them, and #1659 ruled prose matching out).  Since
#1828 the question is not asked at all: EVERY leaf is a placeholder, the run-end draw
NAMES each one, and what a routine is called and what must be re-supplied to run it
again are decided from the user's ask alone by a separate draw.  A leaf the draw missed
keeps its arg-derived name — absence is never a drop.

The one leaf nobody binds and the *attachment* fills is the ATTACHMENT MARK
(#1783, :attr:`SkillSubstitution.attachment`): a leaf whose demonstrated value named
one of Penny's own COLLECTIONS.  "Apply this routine to C" is what decides such a
leaf, so the render seam binds it to C (:func:`retarget_writes`) instead of freezing
the demonstration's collection into the recipe.  The mark is registry-derived, not
tool-derived — distillation is handed the collection names and compares VALUES — and
nothing clears it: a destination is never a parameter, so where a routine writes is
always what it is applied to.

This module is pure (no engine, no tool imports): the step/parameter models, the
provenance inference (:func:`distill_steps`), and the load-bearing render
(:func:`render_skill`) that turns steps + bound params into the numbered TEXT
``extraction_prompt`` a collection runs.  The DB store lives in
:mod:`penny.database.skill_store`; the run-end extractor in
:mod:`penny.skill_extraction`; the ``skill_read`` tool in
:mod:`penny.tools.skill_tools`.
"""

from __future__ import annotations

import copy
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def slug_skill_name(name: str) -> str:
    """Normalize a skill name for use as the primary key.

    Unlike a memory slug, a skill name is a human-readable title (``"Watch a page
    field"``) — the read surface shows it and the model names it — so casing and
    spaces are preserved; only surrounding whitespace is trimmed.  Trimming makes
    ``"x "`` and ``"x"`` the one key, so an upsert can't silently fork a skill on a
    stray trailing space."""
    return name.strip()


class SkillSubKind(StrEnum):
    """What a substituted argument leaf resolves to — the closed union of dynamic
    leaves the render substitutes (everything else is a constant)."""

    HOLE = "hole"
    BINDING = "binding"
    PLACEHOLDER = "placeholder"


class SkillSubstitution(BaseModel):
    """One dynamic leaf inside a step's arguments, addressed by its JSON ``path``.

    A leaf NOT covered by any substitution is a constant (rendered verbatim).  A
    ``HOLE`` names the parameter that fills it at instantiation (``parameter`` is
    the parameter's semantic name — the binding key at instantiation); a
    ``BINDING`` names the prior *skill* step (1-based ordinal) whose result flows
    into it; a ``PLACEHOLDER`` carries a one-line ``description`` of what belongs
    there each run — since #1828 that is EVERY leaf the run-end labeller named, and
    the description is what renders there so the demonstrated value is never frozen
    into the recipe.

    ``attachment`` is the ATTACHMENT MARK (#1783): this leaf's demonstrated value
    named one of Penny's own COLLECTIONS, so *what the routine is attached to* is what
    decides it.  Set at distill by comparing values against the registry (never a tool
    name), and since #1828 nothing clears it — a destination is never a parameter — so
    every mark is what :func:`retarget_writes` binds at the render seam.
    """

    path: list[str | int]
    kind: SkillSubKind
    parameter: str | None = None  # set when kind == HOLE — the parameter's semantic name
    step: int | None = None  # set when kind == BINDING — the skill-relative ordinal
    description: str | None = None  # set when kind == PLACEHOLDER — what belongs there
    attachment: bool = False  # the attached collection fills this leaf (#1783)


class SkillStep(BaseModel):
    """One structured step of a skill — the ``LoggedToolCall`` shape (verbatim tool
    name + arguments, copied from the ledger) annotated with its dynamic leaves.

    ``ordinal`` is the step's 1-based position within the skill (what a binding and
    the render number against).  ``source_ordinal`` is the absolute tool-call
    ordinal of the run it was copied from (the provenance/selection anchor, #1578)
    — kept so the skill can always be traced back to the exact call that certified
    it.  ``arguments`` is the call's verbatim argument structure; ``substitutions``
    marks which leaves are parameters/bindings.
    """

    ordinal: int
    source_ordinal: int
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    substitutions: list[SkillSubstitution] = Field(default_factory=list)


class SkillParameter(BaseModel):
    """A declared parameter of a skill — a SKILL-level input, not a tool-arg echo
    (#1668).  ``name`` is its semantic name (the binding key at instantiation —
    ``params={name: value}``, display form == invocation form), ``description`` a
    one-line what-to-supply (``None`` when unlabelled — the run-end naming
    micro-context writes both, falling back to the arg-derived name), and
    ``required`` whether an unbound value is a loud validation error at
    instantiation (#1591/#1659; never a silent default).  Every parameter inferred
    by :func:`distill_steps` is ``required`` by construction: structural provenance
    can't know a safe fallback, so the model must bind each explicitly — an
    out-of-context guess would be untraceable later.  ``required`` stays a declared
    field so a future authoring path can still mark a parameter optional."""

    name: str
    required: bool = True
    description: str | None = None


class SkillDraft(BaseModel):
    """A skill distilled but not yet persisted — the bundle the run-end extractor
    hands to the store.  ``source_run_id`` is the demonstrated run; the triggering
    user message is the ``description`` (reachable through that run too)."""

    name: str
    intent: str
    description: str
    steps: list[SkillStep]
    parameters: list[SkillParameter]
    source_run_id: str


# ── Provenance inference (taint-tracking over one run's selected steps) ────────


class DistillInput(BaseModel):
    """One selected step handed to :func:`distill_steps` — its source-run ordinal,
    tool name, verbatim arguments, and the framed text of its result (used only to
    match bindings; success is decided by the caller before this runs)."""

    source_ordinal: int
    tool: str
    arguments: dict[str, Any]
    result: str


# The framework injects a universal ``reasoning`` think-aloud string into every
# tool call's arguments (``Tool.to_ollama_tool``) — the model's per-run narration
# of *why* it made the call.  It is run narration, never part of the routine, so
# it is stripped from a distilled step outright (#1661): not a parameter, not a baked
# constant, simply absent — the executing model supplies its own reasoning at run
# time.  Only the TOP-LEVEL key is dropped; a nested arg that happens to share the
# name is real routine data and stays.
_REASONING_KEY = "reasoning"


def _without_reasoning(arguments: dict[str, Any]) -> dict[str, Any]:
    """A shallow copy of a logged call's arguments with the top-level ``reasoning``
    think-aloud removed (#1661) — so distillation never sees it as a string leaf
    (a nonsense required parameter) and the stored step never carries or renders it."""
    return {key: value for key, value in arguments.items() if key != _REASONING_KEY}


def _leaf_paths(value: Any, prefix: list[str | int]) -> list[tuple[list[str | int], str]]:
    """Every ``(path, string_value)`` string leaf under ``value``, recursively.

    Only string leaves are provenance-inferred — a number/bool argument
    (``collector_interval_seconds``, ``notify``) can't be a user-utterance
    phrase or a prior text result, so it is always a constant."""
    leaves: list[tuple[list[str | int], str]] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            leaves.extend(_leaf_paths(sub, [*prefix, key]))
    elif isinstance(value, list):
        for index, sub in enumerate(value):
            leaves.extend(_leaf_paths(sub, [*prefix, index]))
    elif isinstance(value, str):
        leaves.append((prefix, value))
    return leaves


def _nearest_key(path: list[str | int]) -> str:
    """The nearest string key up a leaf's path — the inferred parameter's initial
    (arg-derived) name (mirrors the ``{url}`` / ``{field}`` convention: a parameter
    is first named for the argument key it fills, then relabelled semantically by
    the run-end naming micro-context, #1668).  A leaf directly under a list falls
    back to ``"param"``."""
    for part in reversed(path):
        if isinstance(part, str):
            return part
    return "param"


class _ParameterNamer:
    """Assigns a stable parameter name per distinct demonstrated value — the same
    value in two places is ONE parameter (deduped), and two different values never
    collide (a name clash gets a numeric suffix)."""

    def __init__(self) -> None:
        self._by_value: dict[str, str] = {}
        self._used: set[str] = set()

    def name_for(self, value: str, path: list[str | int]) -> str:
        if value in self._by_value:
            return self._by_value[value]
        base = _nearest_key(path)
        name = base
        suffix = 2
        while name in self._used:
            name = f"{base}-{suffix}"
            suffix += 1
        self._by_value[value] = name
        self._used.add(name)
        return name


_MIN_BINDING_OVERLAP = 3

# A tool result reaches distillation FRAMED (``Tool.format_result``): a first-person
# narration line carrying the ``(<tool> result)`` machine tag, then the body, and —
# for a browse-with-``extract`` result — a trailing fetch-handle line pointing at the
# stored full content.  Only the BODY is the routine's real output, so binding
# compares against that PAYLOAD, not the frame (#1665/#1661 item 3): an arg that
# WRAPS the value (``Price: $499`` over a browse that returned ``$499``) can never
# contain the whole frame, so the wraps direction would otherwise never fire —
# ``content`` becomes a nonsense required parameter that leaks into narration.
#
# The tag is the structural anchor (mirrors ``RESULT_TAG = "({tool} result)"`` in
# ``tools/base.py`` — matched structurally, not imported, so this module stays pure):
# a first line ending in ``(<tool> result)`` is the narration and is dropped.
_RESULT_TAG_LINE = re.compile(r"\([\w-]+ result\)\s*$")
# The browse fetch-handle tail (``tools/browse.py`` ``_EXTRACT_HANDLE_CLAUSE``,
# ``"Full page content saved to {handles} — read it there for anything more."``) —
# anchored on its INVARIANT prefix (the ``{handles}`` and phrasing after it vary
# structurally), so the payload isn't inflated by the pointer-to-stored-content line.
_FETCH_HANDLE_PREFIX = "Full page content saved to "


def _result_payload(result: str) -> str:
    """A framed tool result stripped to its PAYLOAD (#1665) — the routine's real
    output, for binding comparison.  Drops the leading narration line ending in the
    ``(<tool> result)`` machine tag and truncates at any trailing browse fetch-handle
    line, then trims surrounding whitespace.  An unframed string passes through
    (nothing matched)."""
    body = result
    first_break = body.find("\n")
    if first_break != -1 and _RESULT_TAG_LINE.search(body[:first_break]):
        body = body[first_break + 1 :]
    handle_at = body.find(_FETCH_HANDLE_PREFIX)
    if handle_at != -1:
        body = body[:handle_at]
    return body.strip()


def _enclosing_token(payload: str, start: int, end: int) -> tuple[int, int]:
    """The bounds of the whitespace-delimited run of ``payload`` enclosing the match at
    ``[start, end)`` — the token the match sits in (the two tokens at either end, when
    the match spans several)."""
    opens_at = start
    while opens_at > 0 and not payload[opens_at - 1].isspace():
        opens_at -= 1
    closes_at = end
    while closes_at < len(payload) and not payload[closes_at].isspace():
        closes_at += 1
    return opens_at, closes_at


def _occurs_as_whole_token(value: str, payload: str) -> bool:
    """Whether ``value`` occupies a WHOLE token of ``payload`` — the copied-out-of-a
    -result test (#1809), as opposed to a coincidence of characters.

    A payload is not always a returned value: a browse without ``extract`` returns a
    PAGE, and a page contains most short strings by accident.  What distinguishes a
    value the model *copied out of* the result from one that merely appears inside it
    is that a copy takes whole tokens — a field, a figure, a phrase — never a fragment
    sliced out of the middle of one.  So the match must fill its enclosing
    whitespace-delimited token(s) up to their PUNCTUATION margins: what sits between
    the token's edge and the match may be punctuation (``is $499.`` and
    ``"Aurora Deck 2"`` are copies of ``$499`` and ``Aurora Deck 2``) but never letters
    or digits.  ``aurora-deck-2`` inside ``https://faux-market.example/aurora-deck-2``
    is the motivating case and fails exactly there — the margin is
    ``https://faux-market.example/``, not punctuation — where plain containment bound
    the write key the assistant chose to the price the step fetched, and the learned
    routine then wrote every cycle's price under a key that was also the price.
    """
    start = payload.find(value)
    while start != -1:
        end = start + len(value)
        opens_at, closes_at = _enclosing_token(payload, start, end)
        margins = f"{payload[opens_at:start]}{payload[end:closes_at]}"
        if not any(character.isalnum() for character in margins):
            return True
        start = payload.find(value, start + 1)
    return False


def _binding_step(value: str, index: int, selected: list[DistillInput]) -> int | None:
    """The skill ordinal (1-based) of the latest PRIOR selected step whose result the
    value flowed from, or ``None`` when none produced it (then it is a parameter).

    Comparison is against each prior result's PAYLOAD (``_result_payload`` — the frame
    stripped off, #1665), not the framed text.  A value binds when it **equals or fills
    whole tokens of** a prior payload (the model copied the tool output verbatim,
    ``_occurs_as_whole_token`` — #1809) OR **contains** a prior payload (it wrapped the
    output — ``Price: $499`` over a returned ``$499``).  Guarded against degenerate
    matches: a blank payload never binds, and the shared content must be non-trivial
    (``_MIN_BINDING_OVERLAP`` chars) so a one-character coincidence can't manufacture a
    binding.  No fuzzy matching and no thresholds — every test here is exact-match, which
    is what lets the wraps direction fire without loosening anything (#1661's LCS
    analysis showed loosening false-binds topic names).

    The two directions are deliberately ASYMMETRIC.  *Wraps* stays plain containment:
    the whole result sits inside the argument, so there is nothing for a coincidence to
    exploit.  *Contained-in* is the direction exposed to bulk — the payload may be an
    entire fetched page, where a short arg turning up somewhere is worth nothing — so it
    additionally demands that the arg fill the tokens it lands in."""
    stripped_value = value.strip()
    for prior in range(index - 1, -1, -1):
        payload = _result_payload(selected[prior].result)
        if not payload:
            continue
        if len(stripped_value) >= _MIN_BINDING_OVERLAP and _occurs_as_whole_token(
            stripped_value, payload
        ):
            return prior + 1
        if len(payload) >= _MIN_BINDING_OVERLAP and payload in stripped_value:
            return prior + 1
    return None


# What belongs in an attachment-marked leaf whose labeller draw never landed (#1777,
# retained as the FALLBACK by #1783) — the collection the routine is attached to.  The
# leaf must not carry the demonstrated collection's name: the ambient recipe is a 0-call
# firing surface where NO instantiation runs, so a baked literal there reads as "this
# routine writes to X" — which is what misled a live run into treating a duplicate
# collection as harmless.  A fresh distillation gets its wording from the labeller like
# every other leaf; this fixed string covers the draws that fail and the legacy skills
# migration 0101 back-filled, which have no draw left to recover.
WRITE_TARGET_DESCRIPTION = "the collection this is set up on"


def distill_steps(
    selected: list[DistillInput], attachment_names: frozenset[str]
) -> tuple[list[SkillStep], list[SkillParameter]]:
    """Factor one run's selected steps into ``(steps, parameters)`` by STRUCTURAL
    provenance — read off the ledger, never by matching the user's prose (#1659).

    ``selected`` is the contiguous, certified slice in run order.  The universal
    ``reasoning`` think-aloud is stripped from each call's arguments FIRST (#1661) —
    run narration, never routine — so it is neither classified nor stored.  Each
    remaining string leaf is then classified the SAME way regardless of which tool it
    sits on or which argument it fills (#1783): a value that **equals / fills whole
    tokens of / wraps** a prior selected step's result is a **binding** (it came from
    that step); **every other** string leaf is a required **parameter**, with identical
    values
    collapsing to one shared parameter.  A non-string leaf (a number/bool) is always
    a constant.  That last rule is a DEFAULT, not a determination — it holds only
    when the user supplied the value, which structure cannot decide; the run-end
    naming micro-context relabels each one semantically AND adjudicates it, turning
    an assistant-produced leaf into a placeholder (#1668/#1770).

    ``attachment_names`` are the names of Penny's own COLLECTIONS (the caller reads them
    off the registry).  An UNEXPLAINED leaf whose demonstrated VALUE is one of them
    additionally carries the attachment mark — see
    :attr:`SkillSubstitution.attachment` — because attaching the routine somewhere is
    what decides a leaf that named a collection.  The mark rides ALONGSIDE the ordinary
    classification: the leaf is still a candidate the labeller adjudicates, and a
    user-supplied verdict clears it.  A leaf that BOUND is never marked, whatever its
    value: its provenance is already explained (it came from a prior step), so nothing
    is left for an attachment to decide, and marking it would let the render seam
    overwrite a real binding with the collection's name."""
    namer = _ParameterNamer()
    steps: list[SkillStep] = []
    parameters: dict[str, SkillParameter] = {}
    for index, inp in enumerate(selected):
        arguments = _without_reasoning(inp.arguments)
        subs = [
            _substitution_for(path, value, index, selected, namer, parameters, attachment_names)
            for path, value in _leaf_paths(arguments, [])
        ]
        steps.append(
            SkillStep(
                ordinal=index + 1,
                source_ordinal=inp.source_ordinal,
                tool=inp.tool,
                arguments=arguments,
                substitutions=subs,
            )
        )
    return steps, list(parameters.values())


def _substitution_for(
    path: list[str | int],
    value: str,
    index: int,
    selected: list[DistillInput],
    namer: _ParameterNamer,
    parameters: dict[str, SkillParameter],
    attachment_names: frozenset[str],
) -> SkillSubstitution:
    """Classify ONE string leaf — the whole of :func:`distill_steps`'s per-leaf rule.

    A value a prior selected step produced is a BINDING and nothing else; anything left
    unexplained is a candidate parameter (registered in ``parameters``, deduped by value
    through ``namer``), marked for the attachment when its value names a collection."""
    producer = _binding_step(value, index, selected)
    if producer is not None:
        return SkillSubstitution(path=path, kind=SkillSubKind.BINDING, step=producer)
    name = namer.name_for(value, path)
    parameters.setdefault(name, SkillParameter(name=name, required=True))
    return SkillSubstitution(
        path=path,
        kind=SkillSubKind.HOLE,
        parameter=name,
        attachment=value in attachment_names,
    )


# ── The render (steps + bound params → the numbered TEXT extraction_prompt) ────


class _Bound(BaseModel):
    """Render sentinel: a parameter filled with its bound value (verbatim)."""

    value: Any


class _Placeholder(BaseModel):
    """Render sentinel: an unbound parameter, shown as ``{name}`` (the with-params
    form)."""

    name: str


class _BindingRef(BaseModel):
    """Render sentinel: a binding, shown as the legible ``the value from step N``."""

    step: int


class _DescribedSlot(BaseModel):
    """Render sentinel: a leaf no user binds — a spot the labeller named (#1770/#1828),
    including one still awaiting its attachment (#1783) — shown in the SAME
    placeholder syntax as an
    unbound parameter but carrying the one-line description of what belongs there,
    never the demonstrated value.  Freezing that value is the failure this exists to
    prevent: a collector re-running the skill would write the demonstration's stale
    phrase into the collection every cycle, forever, and a reader of the recipe would
    believe it acts on the thing the demonstration happened to act on."""

    description: str


def _marker_for(sub: SkillSubstitution, params: dict[str, str]) -> Any:
    # Each kind's payload read defensively, as its siblings are: the extractor reads a
    # blank drawn description as NO label (there would be nothing to render in the leaf's
    # place) and an unlabelled marked leaf falls back to ``WRITE_TARGET_DESCRIPTION``, so
    # an empty one never reaches here in practice.
    if sub.kind == SkillSubKind.PLACEHOLDER:
        return _DescribedSlot(description=sub.description or "")
    if sub.kind == SkillSubKind.HOLE:
        name = sub.parameter or ""
        if name in params:
            return _Bound(value=params[name])
        return _Placeholder(name=name)
    return _BindingRef(step=sub.step or 0)


def _set_at_path(root: Any, path: list[str | int], marker: Any) -> None:
    """Replace the leaf at ``path`` in the deep-copied argument tree with a render
    sentinel."""
    node = root
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = marker


def _render_value(value: Any) -> str:
    """One argument value in the canonical call notation (the ``!r`` projection
    #1578's ``render_tool_call`` uses), with the render sentinels rendered
    legibly: a bound parameter as its value (verbatim, quoted like any literal), an
    unbound parameter as ``{name}``, an assistant-produced leaf as
    ``{<what belongs there>}``, a binding as ``the value from step N``."""
    if isinstance(value, _Bound):
        return repr(value.value)
    if isinstance(value, _Placeholder):
        return f"{{{value.name}}}"
    if isinstance(value, _DescribedSlot):
        return f"{{{value.description}}}"
    if isinstance(value, _BindingRef):
        return f"the value from step {value.step}"
    if isinstance(value, dict):
        inner = ", ".join(f"{key!r}: {_render_value(sub)}" for key, sub in value.items())
        return f"{{{inner}}}"
    if isinstance(value, list):
        return f"[{', '.join(_render_value(sub) for sub in value)}]"
    return repr(value)


def _render_step(step: SkillStep, params: dict[str, str]) -> str:
    """One ``N. tool(args)`` line — the canonical call notation applied faithfully
    (no per-tool compaction, so the recipe is runnable, unlike a run-trace
    summary)."""
    resolved = copy.deepcopy(step.arguments)
    for sub in step.substitutions:
        _set_at_path(resolved, sub.path, _marker_for(sub, params))
    args = ", ".join(f"{key}={_render_value(value)}" for key, value in resolved.items())
    return f"{step.ordinal}. {step.tool}({args})"


def render_skill(steps: list[SkillStep], params: dict[str, str] | None = None) -> str:
    """Render a skill's steps + bound ``params`` into a numbered TEXT recipe — the
    same numbered-tool-call dialect production ``extraction_prompt``s use.

    The load-bearing deliverable (#1590): #1591's ``collection_set`` calls this
    to stamp the collection's ``extraction_prompt`` at creation.  Parameters present
    in ``params`` are substituted with their value verbatim; parameters NOT in
    ``params`` render as ``{name}`` (the with-params form the read surface shows);
    a leaf ``params`` can never bind — an assistant-produced value (#1770), including
    one an attachment has yet to fill (#1783) — renders as ``{<what belongs there>}``
    from its description, never the demonstrated value; bindings render as ``the value from
    step N``; everything else is a constant.  Pure and deterministic — the same steps
    + params always produce the same text.
    """
    params = params or {}
    return "\n".join(_render_step(step, params) for step in steps)


def unbound_required_parameters(
    parameters: list[SkillParameter], params: dict[str, str]
) -> list[SkillParameter]:
    """The required parameters ``params`` doesn't bind — the validation #1591's
    ``collection_set`` runs before rendering (an unbound required parameter is an
    error).  Returns the whole :class:`SkillParameter` (name + description) so the
    refusal can name each parameter AND what to supply (#1668).  Shipped here so the
    rule lives with the skill, tested standalone."""
    return [p for p in parameters if p.required and p.name not in params]


# ── Attachment binding at apply (#1629/#1783) ──────────────────────────────────


def retarget_writes(steps: list[SkillStep], target: str) -> list[SkillStep]:
    """Bind every ATTACHMENT-MARKED leaf to ``target`` — the retarget-at-apply rule
    (#1629, generalized off the tool whitelist by #1783).

    "Apply this routine to collection C" is what decides a leaf that named a collection,
    so each leaf still carrying :attr:`SkillSubstitution.attachment` at instantiation is
    bound to C's own name: the rendered copy takes the value and drops the substitution,
    so the render prints the collection instead of the placeholder.  Nothing here knows
    which tools a skill contains — it reads the MARK distillation left and the labeller
    let stand, so a write, a read of the routine's own collection, or a tool nobody has
    enumerated all bind identically, and a leaf the labeller judged USER-supplied kept no
    mark and stays the parameter the user binds.

    Two leaves marked and unbound therefore both land on C — not a collapse rule, the
    OUTCOME of two verdicts that said "the assistant chose this"; two destinations the
    user named are two parameters and stay distinct.

    Runs at the render/instantiation seam (``render_skill_prompt``), on BOTH the one-call
    ``collection_set(skill=…)`` and the adopt path, so the rendered ``extraction_prompt``
    never lies about the collection it acts on.  Pure — the skill's STORED steps keep
    their verbatim ledger arguments (a skill is target-agnostic at rest); only the
    rendered-into-a-collection copy is bound.  A step with no marked leaf passes through
    untouched.
    """
    retargeted: list[SkillStep] = []
    for step in steps:
        marked = [sub.path for sub in step.substitutions if sub.attachment]
        if not marked:
            retargeted.append(step)
            continue
        arguments = copy.deepcopy(step.arguments)
        for path in marked:
            _set_at_path(arguments, path, target)
        # Each bound leaf now holds the target, so drop the substitution that addressed
        # it — else the render would substitute a marker back over the collection's name.
        substitutions = [sub for sub in step.substitutions if not sub.attachment]
        retargeted.append(
            step.model_copy(update={"arguments": arguments, "substitutions": substitutions})
        )
    return retargeted
