"""CollectionSandboxRunTool — preview a job's next cycle from chat (#2964).

Model-reachable, unlike ``Collector.run_for`` (the addon's real "run this now" —
deliberately kept off the chat surface since #1911, because a real run both
schedules and immediately runs the job the user only asked to set up). Nothing
a sandbox preview does is real — every mutating call the cycle makes goes
through its plan path, forced by the framework (#2933/#2964) — so letting the
model reach for it reopens nothing that reason guarded against.

A full cycle can run well past a chat tool call's own timeout, so this tool
never awaits the cycle: ``execute`` starts it and returns at once; the actual
report follows as a separate message once the cycle finishes, through the same
queue/drainer path a job's own report rides (``Collector.run_sandbox``,
``notification.build_sandbox_report``, ``ChatAgent.handle_sandbox_report``).
"""

from __future__ import annotations

from typing import Any, Protocol

from penny.llm.call_link import CallLink
from penny.tools.base import Tool
from penny.tools.models import ToolArgs, ToolResult


class SandboxRunner(Protocol):
    """The one capability this tool needs — ``Collector.run_sandbox`` duck-typed.

    A ``Protocol``, not an import of ``Collector`` itself: ``penny.agents.collector``
    already imports from ``penny.tools.memory_tools``, so a tools module importing
    ``Collector`` back would close that cycle. A required constructor parameter
    (CLAUDE.md, "Pass parameters, don't swap state") — no optional/``None`` default
    that would let a future caller forget to wire it; a test double only has to
    satisfy this one async method."""

    async def run_sandbox(self, collection_name: str) -> tuple[bool, str]: ...


class CollectionSandboxRunArgs(ToolArgs):
    """The one argument this tool takes — which job to preview."""

    collection: str


_STARTED_NARRATION = "You started a sandbox preview of '{collection}':"
_FAILED_NARRATION = "You tried to start a sandbox preview of '{collection}' but it didn't work:"


class CollectionSandboxRunTool(Tool):
    """Preview one job's next cycle, with nothing persisted (#2964).

    Not ``mutates``: the tool itself changes no durable job state — it starts a
    background preview and reports honestly that it started, nothing more. The
    preview's OWN tool calls are what go through the forced-plan mechanism,
    inside the cycle this starts — never this tool's own call."""

    name = "collection_sandbox_run"
    description = (
        "Preview what a job's next cycle would do, without actually doing it — "
        "nothing is written, sent, or scheduled; every change the cycle would "
        "make is described, never made. The preview's full report follows as a "
        "separate message once the cycle finishes (it can take a few minutes) — "
        "this call only confirms the preview started. Use this when the user "
        "asks to test, preview, dry-run, or try out a job before relying on it. "
        "There is no tool to run a job for real from chat — a real cycle runs "
        "on the job's own schedule."
    )
    parameters = {
        "type": "object",
        "properties": {
            "collection": {
                "type": "string",
                "description": "The job's (collection's) name to preview.",
            },
        },
        "required": ["collection"],
    }
    args_model = CollectionSandboxRunArgs

    def __init__(self, runner: SandboxRunner) -> None:
        self._runner = runner

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        collection = arguments.get("collection", "a job")
        template = _STARTED_NARRATION if result.success else _FAILED_NARRATION
        return template.format(collection=collection)

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """Start the preview and return at once — never the preview itself.

        ``call_link`` is accepted (every tool's ``execute`` does, #2458) but
        unused: a sandbox cycle mints its OWN top-level link when it starts
        (``Collector.run_sandbox`` → ``CallLink.top``), it is not this call's
        child — the preview outlives this very turn."""
        args = CollectionSandboxRunArgs(**kwargs)
        started, message = await self._runner.run_sandbox(args.collection)
        return ToolResult(message=message, success=started, mutated=False)
