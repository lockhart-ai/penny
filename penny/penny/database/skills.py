"""The skill substrate — structured steps, provenance-inferred holes, and the
steps→text render (#1590, stage ④ of #1562 / epic #1554).

A **skill** is a certified-by-execution script distilled from ONE demonstrated
run: an ordered list of structured steps in the ``LoggedToolCall`` shape (#1578)
plus declared parameter *holes*.  It is authored only by reference to the ledger
(``skill_create(name)`` — name-only): the system snapshots the run immediately
preceding this one, copying ALL its non-``done`` tool-call ordinals out, never
re-emitting them.  Each argument leaf of a copied call is factored by
**provenance** — derived STRUCTURALLY from the ledger, never by matching the
user's prose (#1659):

* a value that **equals, is contained in, or wraps** a prior selected step's
  result → a **binding** (rendered "the value from step N"), because in the
  source run it *came from* that step (a wrapped result binds too — the arg
  ``Price: $499`` over a browse that returned ``$499``);
* the scoped-write **target** argument (``memory`` on a write step) → a
  **constant** owned by write-retarget (#1629), never a parameter;
* **every other string leaf** → a required **hole** (a parameter the model binds
  per instantiation); identical values collapse to ONE shared hole.  A hole is
  ``required`` by construction — an unbound hole is a loud refusal at
  instantiation, never a silent default (no-silent-fallbacks).

This module is pure (no engine, no tool imports): the step/hole models, the
provenance inference (:func:`distill_steps`), and the load-bearing render
(:func:`render_skill`) that turns steps + bound params into the numbered TEXT
``extraction_prompt`` a collection runs.  The DB store lives in
:mod:`penny.database.skill_store`; the ``skill_create`` / ``skill_read`` tools in
:mod:`penny.tools.skill_tools`.
"""

from __future__ import annotations

import copy
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


class SkillSubstitution(BaseModel):
    """One dynamic leaf inside a step's arguments, addressed by its JSON ``path``.

    A leaf NOT covered by any substitution is a constant (rendered verbatim).  A
    ``HOLE`` names the parameter that fills it at instantiation; a ``BINDING``
    names the prior *skill* step (1-based ordinal) whose result flows into it.
    """

    path: list[str | int]
    kind: SkillSubKind
    hole: str | None = None  # set when kind == HOLE
    step: int | None = None  # set when kind == BINDING — the skill-relative ordinal


class SkillStep(BaseModel):
    """One structured step of a skill — the ``LoggedToolCall`` shape (verbatim tool
    name + arguments, copied from the ledger) annotated with its dynamic leaves.

    ``ordinal`` is the step's 1-based position within the skill (what a binding and
    the render number against).  ``source_ordinal`` is the absolute tool-call
    ordinal of the run it was copied from (the provenance/selection anchor, #1578)
    — kept so the skill can always be traced back to the exact call that certified
    it.  ``arguments`` is the call's verbatim argument structure; ``substitutions``
    marks which leaves are holes/bindings.
    """

    ordinal: int
    source_ordinal: int
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    substitutions: list[SkillSubstitution] = Field(default_factory=list)


class SkillHole(BaseModel):
    """A declared parameter of a skill — its ``name`` and whether it is
    ``required`` (an unbound required hole is a loud validation error at
    instantiation, #1591/#1659; never a silent default).  Every hole inferred by
    :func:`distill_steps` is ``required`` by construction: structural provenance
    can't know a safe fallback, so the model must bind each parameter explicitly —
    an out-of-context guess would be untraceable later.  ``required`` stays a
    declared field so a future authoring path can still mark a hole optional."""

    name: str
    required: bool = True


class SkillDraft(BaseModel):
    """A skill distilled but not yet persisted — the bundle ``skill_create`` hands
    to the store.  ``source_run_id`` is the demonstrated run; the triggering user
    message is reachable *through* that run (never copied here)."""

    name: str
    intent: str
    description: str
    steps: list[SkillStep]
    holes: list[SkillHole]
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
# it is stripped from a distilled step outright (#1661): not a hole, not a baked
# constant, simply absent — the executing model supplies its own reasoning at run
# time.  Only the TOP-LEVEL key is dropped; a nested arg that happens to share the
# name is real routine data and stays.
_REASONING_KEY = "reasoning"


def _without_reasoning(arguments: dict[str, Any]) -> dict[str, Any]:
    """A shallow copy of a logged call's arguments with the top-level ``reasoning``
    think-aloud removed (#1661) — so distillation never sees it as a string leaf
    (a nonsense required hole) and the stored step never carries or renders it."""
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
    """The nearest string key up a leaf's path — the inferred hole's name (mirrors
    the ``{url}`` / ``{field}`` convention: a hole is named for the argument key it
    fills).  A leaf directly under a list falls back to ``"param"``."""
    for part in reversed(path):
        if isinstance(part, str):
            return part
    return "param"


class _HoleNamer:
    """Assigns a stable hole name per distinct demonstrated value — the same value
    in two places is ONE parameter (deduped), and two different values never
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


def _binding_step(value: str, index: int, selected: list[DistillInput]) -> int | None:
    """The skill ordinal (1-based) of the latest PRIOR selected step whose result the
    value flowed from, or ``None`` when none produced it (then it is a hole).

    A value binds when it **equals or is contained in** a prior result (the model
    copied the tool output verbatim) OR **contains** a prior result (it wrapped the
    output — ``Price: $499`` over a returned ``$499``).  Guarded against degenerate
    matches: a blank/whitespace prior result never binds, and the shared content must
    be non-trivial (``_MIN_BINDING_OVERLAP`` chars) so a one-character coincidence
    can't manufacture a binding."""
    stripped_value = value.strip()
    for prior in range(index - 1, -1, -1):
        result = selected[prior].result
        stripped_result = result.strip()
        if not stripped_result:
            continue
        if len(stripped_value) >= _MIN_BINDING_OVERLAP and value in result:
            return prior + 1
        if len(stripped_result) >= _MIN_BINDING_OVERLAP and result in value:
            return prior + 1
    return None


def _is_write_target(tool: str, path: list[str | int]) -> bool:
    """The scoped-write target leaf (``memory`` on a ``collection_write`` /
    ``update_entry`` / ``collection_delete_entry`` step) — excluded from provenance
    classification (#1659).  Write-retarget (#1629, :func:`retarget_writes`) rebinds
    it to the attached collection at the render seam, so making it a parameter would
    force the model to bind a value retarget then overwrites; it stays a constant."""
    return tool in SCOPED_WRITE_TOOLS and path == ["memory"]


def distill_steps(selected: list[DistillInput]) -> tuple[list[SkillStep], list[SkillHole]]:
    """Factor one run's selected steps into ``(steps, holes)`` by STRUCTURAL
    provenance — read off the ledger, never by matching the user's prose (#1659).

    ``selected`` is the contiguous, certified slice in run order.  The universal
    ``reasoning`` think-aloud is stripped from each call's arguments FIRST (#1661) —
    run narration, never routine — so it is neither classified nor stored.  Each
    remaining string leaf is classified in order: the scoped-write **target** is a
    retarget-owned constant (skipped); a value that **equals / is contained in /
    wraps** a prior selected step's result is a **binding** (it came from that step);
    **every other** string leaf is a required **hole**, with identical values
    collapsing to one shared hole.  A non-string leaf (a number/bool) is always a
    constant."""
    namer = _HoleNamer()
    steps: list[SkillStep] = []
    holes: dict[str, SkillHole] = {}
    for index, inp in enumerate(selected):
        arguments = _without_reasoning(inp.arguments)
        subs: list[SkillSubstitution] = []
        for path, value in _leaf_paths(arguments, []):
            if _is_write_target(inp.tool, path):
                continue
            producer = _binding_step(value, index, selected)
            if producer is not None:
                subs.append(SkillSubstitution(path=path, kind=SkillSubKind.BINDING, step=producer))
                continue
            name = namer.name_for(value, path)
            holes.setdefault(name, SkillHole(name=name, required=True))
            subs.append(SkillSubstitution(path=path, kind=SkillSubKind.HOLE, hole=name))
        steps.append(
            SkillStep(
                ordinal=index + 1,
                source_ordinal=inp.source_ordinal,
                tool=inp.tool,
                arguments=arguments,
                substitutions=subs,
            )
        )
    return steps, list(holes.values())


# ── The render (steps + bound params → the numbered TEXT extraction_prompt) ────


class _Bound(BaseModel):
    """Render sentinel: a hole filled with its bound parameter value (verbatim)."""

    value: Any


class _Placeholder(BaseModel):
    """Render sentinel: an unbound hole, shown as ``{name}`` (the with-holes form)."""

    name: str


class _BindingRef(BaseModel):
    """Render sentinel: a binding, shown as the legible ``the value from step N``."""

    step: int


def _marker_for(sub: SkillSubstitution, params: dict[str, str]) -> Any:
    if sub.kind == SkillSubKind.HOLE:
        name = sub.hole or ""
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
    legibly: a bound hole as its value (verbatim, quoted like any literal), an
    unbound hole as ``{name}``, a binding as ``the value from step N``."""
    if isinstance(value, _Bound):
        return repr(value.value)
    if isinstance(value, _Placeholder):
        return f"{{{value.name}}}"
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

    The load-bearing deliverable (#1590): #1591's ``collection_create`` calls this
    to stamp the collection's ``extraction_prompt`` at creation.  Holes present in
    ``params`` are substituted with their value verbatim; holes NOT in ``params``
    render as ``{name}`` (the with-holes form the read surface shows); bindings
    render as ``the value from step N``; everything else is a constant.  Pure and
    deterministic — the same steps + params always produce the same text.
    """
    params = params or {}
    return "\n".join(_render_step(step, params) for step in steps)


def unbound_required_holes(holes: list[SkillHole], params: dict[str, str]) -> list[str]:
    """Names of the required holes ``params`` doesn't bind — the validation
    #1591's ``collection_create`` runs before rendering (an unbound required hole
    is an error).  Shipped here so the rule lives with the skill, tested
    standalone; not wired to a runtime surface in this stage."""
    return [hole.name for hole in holes if hole.required and hole.name not in params]


# ── Write-retarget at apply (#1629) ────────────────────────────────────────────

# The scoped-write tools a collector run is pinned to its bound target through
# (``_memory_scope``): their ``memory`` argument must name the collection being
# instantiated, not whatever the demonstration happened to write into.  A skill is
# demonstrated against SOME collection, so its write steps carry that demo target
# as a baked-in constant; instantiating the skill into collection C is what DEFINES
# the write target, so the constant is overwritten with C at the render seam.
SCOPED_WRITE_TOOLS = frozenset({"collection_write", "update_entry", "collection_delete_entry"})


def retarget_writes(steps: list[SkillStep], target: str) -> list[SkillStep]:
    """Bind every scoped-write step's ``memory`` argument to ``target`` — the
    write-retarget-at-apply rule (#1629).

    "Apply this skill to collection C" is what fixes where its writes go, so the
    demo-run constant (or a stray hole/binding) on the ``memory`` argument is
    replaced by ``target`` — the collection's own name.  This runs at the
    render/instantiation seam (``render_skill_prompt``), on BOTH the one-call
    ``collection_create(skill=…)`` and the ``collection_update`` adopt paths, so the
    rendered ``extraction_prompt`` never lies about its write target.  Pure — the
    skill's STORED steps keep their demo constant (a skill is target-agnostic); only
    the rendered-into-a-collection copy is retargeted.  A step that isn't a scoped
    write, or whose call omits ``memory``, passes through untouched.
    """
    retargeted: list[SkillStep] = []
    for step in steps:
        if step.tool not in SCOPED_WRITE_TOOLS or "memory" not in step.arguments:
            retargeted.append(step)
            continue
        arguments = copy.deepcopy(step.arguments)
        arguments["memory"] = target
        # The ``memory`` leaf is now a constant, so drop any hole/binding that
        # addressed it — else the render would substitute a marker back over it.
        substitutions = [sub for sub in step.substitutions if sub.path[:1] != ["memory"]]
        retargeted.append(
            step.model_copy(update={"arguments": arguments, "substitutions": substitutions})
        )
    return retargeted
