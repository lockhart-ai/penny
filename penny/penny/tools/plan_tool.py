"""Template method for a plugin tool whose single mutating call splits into a
read-only RESOLVE phase and the APPLY call itself (#2947, following #2933's
``mutates`` mark and #2946's ``as_plan_result``).

Unlike a DB write (``collection_write``/``collection_configure``), there is no
local transaction to roll back on an external API call — Zoho and InvoiceNinja
have no "try it and undo it" primitive.  So a plugin tool's plan can never
actually call its mutating endpoint: ``describe`` renders the would-be action
from the RESOLVED inputs alone, never from a committed-then-reverted response.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol

import httpx

from penny.llm.call_link import CallLink
from penny.tools.base import Tool, as_plan_result
from penny.tools.models import ToolResult

logger = logging.getLogger(__name__)


class _HasPlan(Protocol):
    """Structural shape every concrete tool's own args model already has —
    the literal ``plan: bool`` field ``Tool.__init_subclass__`` requires.
    Binding ``ArgsT`` to this (rather than to a concrete base class) means a
    tool's args model keeps whatever base it already has (``ToolArgs`` or a
    plain ``BaseModel``) — this is purely so ``execute`` below can read
    ``args.plan`` with a static type, not a change to any args model's own
    inheritance."""

    plan: bool


class ResolvingMutationTool[ArgsT: _HasPlan](Tool, ABC):
    """``execute`` is defined ONCE here.  Each concrete tool implements:

    - ``resolve(args)``: every READ-ONLY step the call needs — name lookups,
      date parsing, finding the thing to update.  Returns either the tool's
      own resolved shape (whatever ``describe``/``apply`` need), or an early
      ``ToolResult`` refusal (not found, bad date) — itself safe to preview,
      since nothing in it ever ran a mutating call.  ``resolve`` must NEVER
      make a create-if-missing call eagerly (the folder/label/task-list a
      ``MoveEmailsTool``/``ApplyLabelTool``/``CreateTaskTool`` would create if
      absent) — that belongs in ``apply``, after a non-plan call confirms it
      should actually happen.
    - ``describe(resolved)``: the plan headline, built from ``resolved``
      alone.
    - ``apply(resolved)``: the actual mutating call(s).

    ``failure_message`` is the tool's own existing ``*_FAILED`` template (one
    ``{e}``-formatted string) — read-only and mutating client calls share the
    one transport-error catch here, exactly as every one of these tools
    already wraps its whole body in a single ``except`` today.

    Each CONCRETE subclass sets ``mutates = True`` on its own class body —
    never here on the shared abstract base — the same rule every other
    mutating tool follows (``Tool.mutates``'s own docstring): inheriting it
    silently would trip ``Tool.__init_subclass__``'s ``plan``-field check
    against this base's own ``args_model`` (``NoArgs``, no ``plan`` field) at
    IMPORT time, before any real tool is even defined.
    """

    failure_message: str
    args_model: type[ArgsT]

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """Validate, resolve, then either preview or apply — the one shape
        every plugin mutating tool shares."""
        args: ArgsT = self.args_model(**kwargs)
        try:
            resolved = await self.resolve(args)
        except (httpx.HTTPError, RuntimeError) as e:
            logger.exception("%s failed to resolve", self.name)
            return ToolResult(message=self.failure_message.format(e=e), success=False)
        if isinstance(resolved, ToolResult):
            return as_plan_result(resolved) if args.plan else resolved
        if args.plan:
            return as_plan_result(self.describe(resolved))
        return await self._apply_catching_failures(resolved)

    async def _apply_catching_failures(self, resolved: Any) -> ToolResult:
        """The real mutating call(s), under the same transport-error catch as
        ``resolve`` — a failure here reads identically to today's (the whole
        body shared one ``except`` before this split)."""
        try:
            return await self.apply(resolved)
        except (httpx.HTTPError, RuntimeError) as e:
            logger.exception("%s failed to apply", self.name)
            return ToolResult(message=self.failure_message.format(e=e), success=False)

    @abstractmethod
    async def resolve(self, args: Any) -> Any:
        """Every read-only step: name lookups, date parsing, finding the
        target.  Returns the resolved shape, or an early refusal
        ``ToolResult``.  Never a create-if-missing mutating call."""

    @abstractmethod
    def describe(self, resolved: Any) -> ToolResult:
        """The plan headline, built from ``resolved`` alone — there is no
        transaction to roll back on an external call, so this never reads a
        real API response."""

    @abstractmethod
    async def apply(self, resolved: Any) -> ToolResult:
        """The actual mutating call(s)."""
