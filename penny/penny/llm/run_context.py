"""Ambient run/tool-call linkage for model calls (#2458).

A model call made INSIDE a run or a tool should link itself back to that run
(and, for a call nested under a dispatched tool, that call's own id) with NO
wiring at the call site — adding, moving, or removing a micro-context must be
a one-place change, never a second pass over every caller that happens to sit
beneath it.

Two ``contextvars.ContextVar`` pairs carry this:

- ``_CURRENT_RUN_ID`` / ``_CURRENT_PARENT_RUN_ID`` — which run a model call
  happening right now belongs to, and what run (if any) encloses THAT run.
  Pushed as a pair by :func:`run_scope`, opened by whatever starts a run (the
  chat turn, a collector cycle) and re-opened, with a FRESH id, by a
  micro-context's own draw — which is what makes nesting work at any depth:
  each push captures "whatever was ambient a moment ago" as its own parent,
  then becomes the new ambient value for anything beneath it.
- ``_CURRENT_TOOL_CALL_ID`` — which dispatched tool call is executing right
  now, pushed by :func:`tool_call_scope` around one tool's execution.

Nothing here is keyed to an agent's or a micro-context's name — the push/pop
is pure identity plumbing, read by :class:`~penny.llm.client.LlmClient` at log
time (``LlmClient._log_call``) and nowhere else.

``contextvars`` copy into a new ``asyncio.Task`` at the moment it is created
(``asyncio.gather`` wraps each bare coroutine in one), so N parallel tool
calls — or N parallel micro-context draws inside ONE tool call (browse reads
several pages at once) — each get an independent snapshot: a ``.set()``
inside one never leaks into a sibling or back into the caller once the
``gather`` returns.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_CURRENT_RUN_ID: ContextVar[str | None] = ContextVar("current_run_id", default=None)
_CURRENT_PARENT_RUN_ID: ContextVar[str | None] = ContextVar("current_parent_run_id", default=None)
_CURRENT_TOOL_CALL_ID: ContextVar[str | None] = ContextVar("current_tool_call_id", default=None)


def current_run_id() -> str | None:
    """The run id ambient right now — what an unparented model call's OWN
    ``run_id`` resolves to when the caller supplies none."""
    return _CURRENT_RUN_ID.get()


def current_parent_run_id() -> str | None:
    """The run id a model call happening right now should link back to as its
    PARENT — the enclosing run, or ``None`` for a run's own top-level call."""
    return _CURRENT_PARENT_RUN_ID.get()


def current_tool_call_id() -> str | None:
    """The dispatched tool call executing right now, or ``None`` outside one."""
    return _CURRENT_TOOL_CALL_ID.get()


@contextmanager
def run_scope(run_id: str) -> Iterator[None]:
    """Make ``run_id`` the ambient current run for the duration of the block.

    The run this ENCLOSES (whatever was ambient a moment ago) becomes the
    PARENT every call inside the block links back to — ``None`` when nothing
    was ambient yet, which is the run's own top-level call.

    Idempotent when ``run_id`` is already the ambient one: a starter that
    mints a run id and hands it to something that ALSO opens its own scope
    with that same id (e.g. the channel mints a turn's id and passes it to
    ``ChatAgent.handle``, which opens its own scope too, for callers that
    invoke it directly) collapses to a no-op on the inner call, rather than
    making the run its own parent."""
    if _CURRENT_RUN_ID.get() == run_id:
        yield
        return
    parent_token = _CURRENT_PARENT_RUN_ID.set(_CURRENT_RUN_ID.get())
    run_token = _CURRENT_RUN_ID.set(run_id)
    try:
        yield
    finally:
        _CURRENT_RUN_ID.reset(run_token)
        _CURRENT_PARENT_RUN_ID.reset(parent_token)


@contextmanager
def tool_call_scope(tool_call_id: str) -> Iterator[None]:
    """Make ``tool_call_id`` the ambient current tool call for the duration of
    the block — opened by the tool dispatcher around one dispatched call's
    execution, so anything that call makes (a micro-context draw) links back
    to it with no parameter threaded through the tool's own ``execute``."""
    token = _CURRENT_TOOL_CALL_ID.set(tool_call_id)
    try:
        yield
    finally:
        _CURRENT_TOOL_CALL_ID.reset(token)
