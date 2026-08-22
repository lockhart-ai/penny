"""The rendered program's own calls, and when a run has covered them (#1911).

A configured collection's ``extraction_prompt`` is a numbered program: the rendered
form of a taught routine, which ``render_skill`` writes as ``N. tool(args)``.  The
calls it makes are therefore DATA, known before the cycle starts, so "is this cycle
finished?" is a read of that data against what actually executed — not a question the
model answers with a terminal ``done()``.

There is exactly ONE dialect to read, since #1911's soft reboot dropped the seeded
collections: every program in the registry was rendered by ``render_skill`` from a
routine the user taught.  The prose-tolerant read this replaced (a call found anywhere
in a step, to accommodate ``4. Call collection_write("x", …) once with all of them``)
went with its subjects — there is no hand-authored row left for it to serve, and a
lenient parse over prose can only manufacture calls nobody wrote.

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

from pydantic import BaseModel

# A numbered step opens a line: ``4. collection_write(...)``.  The same ``^\d+.`` scan
# the prompt assembler used to number its injected steps with, so "what is a step" is
# one answer in both places.
_STEP_RE = re.compile(r"^(\d+)\.\s*(.*)$", re.MULTILINE)

# The step's call, which must OPEN the step — the rendered dialect exactly, as
# ``render_skill`` writes it (``4. collection_write(memory='x', …)``).  Anchored rather
# than searched (#1911): a program is a render now, so a step that does not open with
# its call is not a step this framework wrote, and reading one leniently would be
# inventing a program out of prose.
_CALL_RE = re.compile(r"^([a-z_][a-z0-9_]*)\(")


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
    participates for free.  Exactly one call per step, and it must OPEN the step: a
    step is one move of the routine, written the one way the renderer writes it.

    An empty result means this prompt is not a rendered program — its steps open with
    something else, or every verb it names has left the surface.  That is a CONFIG
    DEFECT, not a mode: the collection has no readable job, so nothing can close its
    cycle, and the caller surfaces it rather than falling back to a second way of
    running."""
    found: list[ProgramCall] = []
    for match in _STEP_RE.finditer(extraction_prompt):
        tool = _step_call(match.group(2), surface)
        if tool is not None:
            found.append(ProgramCall(ordinal=int(match.group(1)), tool=tool))
    return tuple(found)


def _step_call(step: str, surface: frozenset[str]) -> str | None:
    """The tool this step opens with, or ``None`` when it opens with anything else."""
    match = _CALL_RE.match(step)
    if match is None or match.group(1) not in surface:
        return None
    return match.group(1)
