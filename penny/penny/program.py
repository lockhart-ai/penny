"""The rendered program's own calls, and when a run has covered them (#1911).

A configured collection's ``extraction_prompt`` is a numbered program — the rendered
form of a taught routine (``render_skill`` writes ``N. tool(args)``), or, for the
legacy hand-authored rows, numbered prose that names its calls in the same notation.
Either way the calls it makes are DATA, known before the cycle starts, so "is this
cycle finished?" is a read of that data against what actually executed — not a
question the model answers with a terminal ``done()``.

That inversion is the point.  The exit used to be a model decision at the end of a
long, tool-flavoured context, and the measured cost was the tail: the cycle had to
survive four more steps to close, and 42 of 49 reroll-exhaustion deaths in one
instrumented run landed there.  A read cannot decay.

Two rules, and both are keyed to the STATE rather than to any tool's identity (a
skill is an arbitrary sequence of calls, and a plugin can add one tomorrow):

- **What the program's calls ARE** is whatever call notation each numbered step
  carries, matched against the surface this cycle actually runs with.  Nothing here
  enumerates tools, so a routine built out of tools this module has never heard of is
  read exactly like one built out of ``browse``.
- **What COVERS them** is a forward-only cursor: each successful call that matches the
  call the cursor stands on advances it, and everything else — a failed call about to
  be retried, a read the model interjected of its own accord — passes without counting
  against coverage.  A program is covered when the cursor passes its last call.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from penny.agents.models import ToolCallRecord

# A numbered step opens a line: ``4. collection_write(...)``.  The same ``^\d+.`` scan
# the prompt assembler used to number its injected steps with, so "what is a step" is
# one answer in both places.
_STEP_RE = re.compile(r"^(\d+)\.\s*(.*)$", re.MULTILINE)

# A call inside a step: an identifier immediately followed by ``(``.  It is searched
# ANYWHERE in the step, not just at its start, because the two program dialects differ
# exactly there — a rendered routine writes ``4. collection_write(memory='x', …)`` while
# a hand-authored row writes ``4. Call collection_write("x", entries=[...]) once with
# all of them batched.``  The identifier is then checked against the live surface, which
# is what keeps prose from manufacturing calls: an ordinary sentence has no
# ``<a-tool-name>(`` in it.
_CALL_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(")


class ProgramCall(BaseModel):
    """One call the program makes: the step it is written on, and the tool it names.

    The ARGUMENTS are deliberately not carried.  A rendered leaf holds a bound value,
    a ``{what belongs there}`` placeholder, or ``the value from step N`` — so what the
    call executes with legitimately differs from what the step says, and matching on
    arguments would make coverage unreachable rather than strict."""

    ordinal: int
    tool: str


def program_calls(extraction_prompt: str, surface: frozenset[str]) -> tuple[ProgramCall, ...]:
    """The ordered calls a stored program makes, read off its numbered steps.

    ``surface`` is the set of tool names this cycle actually runs with — passed in
    rather than imported, so this module names no tool and a plugin's verb
    participates for free.  At most one call per step, the FIRST one: a step is one
    move of the routine, and a later mention in the same step is prose about it (a
    hand-authored step's "``update_entry`` … or ``collection_delete_entry`` …" is one
    step offering two ways, not two calls to cover).

    An empty result means the program names no call this cycle could run — a purely
    prose prompt, or one whose every verb has left the surface.  That is a real state
    with a real consequence (there is no coverage to read, so nothing closes the cycle
    structurally), and the caller states it rather than treating an empty program as
    instantly complete."""
    found: list[ProgramCall] = []
    for match in _STEP_RE.finditer(extraction_prompt):
        tool = _first_call(match.group(2), surface)
        if tool is not None:
            found.append(ProgramCall(ordinal=int(match.group(1)), tool=tool))
    return tuple(found)


def _first_call(step: str, surface: frozenset[str]) -> str | None:
    """The first tool this step calls, or ``None`` when it calls none."""
    for match in _CALL_RE.finditer(step):
        if match.group(1) in surface:
            return match.group(1)
    return None


def covered_calls(program: Sequence[ProgramCall], records: Sequence[ToolCallRecord]) -> int:
    """How many of the program's calls this run has executed, in order.

    The forward-only cursor: walk the executed records once, and advance one place
    whenever a SUCCEEDED record names the tool the cursor stands on.  Everything else
    is passed over — a failed call (the retry that follows is the same step trying
    again, so a failure must not consume it), and a call the program never asked for
    (a read the model interjected to orient itself).

    A tool the program names twice is covered by two successful executions of it, in
    order, which the cursor gets right without knowing that repeats are possible.

    Deliberately ORDERED, per the design: a run that executed step 2 before step 1 has
    not carried out the routine it was given, and the honest consequence is that the
    cycle does not close structurally and the run record says so — rather than a
    lenient set comparison quietly calling a scrambled run complete."""
    cursor = 0
    for record in records:
        if cursor >= len(program):
            break
        if not record.failed and record.tool == program[cursor].tool:
            cursor += 1
    return cursor


def is_covered(program: Sequence[ProgramCall], records: Sequence[ToolCallRecord]) -> bool:
    """Whether every call the program makes has executed, in order — the cycle's
    deterministic end.  An EMPTY program is never covered: there is nothing to read,
    so the read cannot be what ends the cycle."""
    return bool(program) and covered_calls(program, records) == len(program)
