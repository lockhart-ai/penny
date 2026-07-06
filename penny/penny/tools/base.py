"""Base classes for tools."""

import asyncio
import difflib
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import ValidationError

from penny.constants import PennyConstants, ProgressEmoji
from penny.tools.models import NoArgs, ToolArgs, ToolCall, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# Pydantic's error ``type`` discriminator for an ``extra="forbid"`` violation —
# an argument the model passed that the arg model doesn't declare.
EXTRA_FORBIDDEN_ERROR_TYPE = "extra_forbidden"

# Generic first-person narration attached to every tool result — the default the
# framing composes with a retained machine tag + the body.  Per-tool overrides
# (browse, memory, framework-failure) land in #1480–#1482; here every tool speaks
# the same generic line, branching only on success/failure.
RESULT_NARRATION_SUCCESS = "You used `{tool_name}` and here's the result:"
RESULT_NARRATION_FAILURE = "You tried to use `{tool_name}` but it didn't work:"

# The retained machine tag.  The narration now reads as natural first-person prose,
# but a live-model probe measured that pure-prose narration with NO tag RAISED the
# call-as-text bail rate (5/6 vs. 3/6 with a tag): gpt-oss:20b stops parsing the body
# as a tool result and reads it as a fresh instruction (#1332's #1 failure class).  So
# a ``(<tool> result)`` marker stays in the framing — the disambiguation the terse
# ``Result of your `<tool>` call:`` header carried — even as the header reads naturally.
RESULT_TAG = "({tool_name} result)"

# First-person frames for the framework-synthesised failures a tool can't narrate
# itself (#1482) — the ``ToolExecutor`` builds these where there's often no tool
# instance (tool-not-found) or the arguments never validated.  Each is set on the
# ``ToolResult.narration`` field so ``format_result`` leads with it instead of the
# generic per-tool narration; the actionable remedy (the #1414 house template's
# diagnosis + how-to-fix tail) stays in ``ToolResult.message`` verbatim — the narration
# is only the frame around it, never a replacement for the remedy.
FRAMEWORK_NARRATION_NOT_FOUND = "You tried to use `{tool_name}` but there's no such tool."
FRAMEWORK_NARRATION_TIMEOUT = "You tried to use `{tool_name}` but it timed out."
FRAMEWORK_NARRATION_EXCEPTION = "You tried to use `{tool_name}` but it errored: {error}."
FRAMEWORK_NARRATION_INVALID_ARGS = "You tried to use `{tool_name}` but the arguments were wrong:"


class Tool(ABC):
    """Abstract base class for tools."""

    name: str
    description: str
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    # The Pydantic model that validates this tool's call arguments — the tool's
    # "validator" (Django-form style).  ``run`` constructs it before ``execute``,
    # so every validity criterion (required fields, types, custom field/model
    # validators like send_message's half-formed-content check) lives here, on the
    # model, and is enforced uniformly BEFORE the tool runs — never ad-hoc inside
    # ``execute``.  Defaults to ``NoArgs`` for argless tools.  Every arg model
    # subclasses ``ToolArgs`` (``extra="forbid"``), so an unknown parameter is
    # rejected through the envelope instead of being silently dropped.
    args_model: type[ToolArgs] = NoArgs
    timeout: float | None = None  # None = use ToolExecutor's global timeout

    _registry: ClassVar[dict[str, type[Tool]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" in cls.__dict__:
            Tool._registry[cls.name] = cls

    async def run(self, **kwargs: Any) -> ToolResult:
        """Validate the call's arguments via ``args_model``, then ``execute``.

        The single validate-then-handle gate every tool goes through (the
        ``ToolExecutor`` calls this, not ``execute``).  A validation failure
        returns an actionable error tool response and ``execute`` is never reached
        with invalid args — so ``execute`` can trust its inputs and concern itself
        only with the work + any runtime/availability decisions (e.g. mute state)
        that aren't expressible as arg validation.
        """
        try:
            self.args_model(**kwargs)
        except ValidationError as exc:
            return ToolResult(
                narration=FRAMEWORK_NARRATION_INVALID_ARGS.format(tool_name=self.name),
                message=self._validation_error_message(exc),
                success=False,
            )
        return await self.execute(**kwargs)

    def _validation_error_message(self, exc: ValidationError) -> str:
        """Actionable rejection body: each bad field, why, and how to fix it.

        Pairs Pydantic's per-error reason (required / wrong type / custom-validator
        message) with the field's type + description from ``parameters`` (the
        model-facing schema), so the model gets the same rich hint the hand-rolled
        check used to give plus any custom-validator guidance.  The first-person frame
        (``FRAMEWORK_NARRATION_INVALID_ARGS``) is carried on the ``ToolResult.narration``
        field, so this stays the pure remedy — the per-field reasons + the retry
        instruction the model recovers from."""
        properties = self.parameters.get("properties", {})
        parts = [self._format_field_error(error, properties) for error in exc.errors()]
        return f"{'; '.join(parts)}. Call {self.name}(<valid arguments>) again."

    @staticmethod
    def _format_field_error(error: Any, properties: dict[str, Any]) -> str:
        """One ``field (type: description): reason`` line for a Pydantic error.

        Loc-path-aware: a nested error (a field inside a ``collection_write`` batch
        entry) is named by its full path (``entries.0.content``) and described by
        its own schema node, not the top-level field's (#1416)."""
        loc = tuple(error.get("loc") or ())
        if error.get("type") == EXTRA_FORBIDDEN_ERROR_TYPE:
            return Tool._format_unknown_param(loc, properties)
        prop = Tool._schema_at_loc(properties, loc)
        param_type = prop.get("type", "")
        param_desc = prop.get("description", "")
        name = Tool._loc_path(loc)
        if param_type and param_desc:
            descriptor = f"{name} ({param_type}: {param_desc})"
        elif param_type:
            descriptor = f"{name} ({param_type})"
        else:
            descriptor = name
        reason = str(error.get("msg", "invalid value")).removeprefix("Value error, ")
        return f"{descriptor}: {reason}"

    @staticmethod
    def _format_unknown_param(loc: tuple[Any, ...], properties: dict[str, Any]) -> str:
        """One actionable line for an unknown (``extra="forbid"``) parameter.

        A misspelled optional argument used to be silently dropped and the tool
        ran with default behaviour.  Now it reaches the envelope: name the bad
        param (by its full loc path, so a nested batch-entry key reads
        ``entries.0.badkey``), suggest the closest valid *sibling* (resolved from
        the schema node one level up), and list the accepted parameters so the
        model can fix the call rather than lose the argument."""
        parent = Tool._schema_at_loc(properties, loc[:-1])
        valid = list(parent.get("properties", {}).keys())
        bad = str(loc[-1]) if loc else "(argument)"
        path = Tool._loc_path(loc)
        close = difflib.get_close_matches(bad, valid, n=1, cutoff=0.6)
        suggestion = f" — did you mean '{close[0]}'?" if close else ""
        accepted = ", ".join(valid) if valid else "none (this tool takes no parameters)"
        return f"unknown parameter '{path}'{suggestion} (valid parameters: {accepted})"

    @staticmethod
    def _loc_path(loc: tuple[Any, ...]) -> str:
        """Render a Pydantic error ``loc`` as a dotted path (``entries.0.badkey``);
        a top-level error stays a bare field name, a locless error a placeholder."""
        return ".".join(str(part) for part in loc) if loc else "(arguments)"

    @staticmethod
    def _schema_at_loc(properties: dict[str, Any], loc: tuple[Any, ...]) -> dict[str, Any]:
        """Walk the parameters schema down a ``loc`` to the node it points at —
        descending ``properties`` on a field name and ``items`` on a list index —
        so a nested field is described by its OWN schema, not the top-level one's."""
        node: dict[str, Any] = {"properties": properties}
        for part in loc:
            if isinstance(part, int):
                node = node.get("items", {})
            else:
                node = node.get("properties", {}).get(str(part), {})
            if not isinstance(node, dict):
                return {}
        return node

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with already-validated arguments.

        ``run`` validates ``kwargs`` against ``args_model`` before this is
        called, so ``execute`` never sees invalid args.

        Returns:
            A ToolResult carrying the model-facing message plus the
            success/mutated/source_urls signals the agent loop records.
        """
        pass

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        """Return a human-readable status string for this tool call. Override per tool."""
        return f"Using {cls.name}"

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        """First-person narration of THIS tool's result, read by the model as the
        reply to its own call.

        The *result* twin of ``to_action_str`` (the *pre-call* status string):
        registry-dispatched via ``format_result`` exactly as ``to_action_str`` is
        via ``format_status``.  It takes the ``ToolResult`` too, so it branches on
        ``result.success`` and a failure narrates honestly.  Generic default;
        per-tool overrides (browse, memory, framework-failure) land in #1480–#1482.
        """
        return cls._default_result_narration(cls.name, result)

    @staticmethod
    def _default_result_narration(tool_name: str, result: ToolResult) -> str:
        """The generic success/failure narration line — shared by the default
        ``to_result_narration`` and ``format_result``'s unregistered-tool fallback,
        so both speak the one string."""
        template = RESULT_NARRATION_SUCCESS if result.success else RESULT_NARRATION_FAILURE
        return template.format(tool_name=tool_name)

    @classmethod
    def to_progress_emoji(cls, arguments: dict) -> ProgressEmoji:
        """Return an emoji that represents this tool call as in-flight progress.

        Channels that show progress as reactions on the user's message use
        this to morph the reaction as the agent moves through tool calls.
        Override per tool to give a more specific indicator.
        """
        return ProgressEmoji.WORKING

    @classmethod
    def format_status(cls, tool_name: str, arguments: dict) -> str:
        """Dispatch to the matching tool's to_action_str via the class registry."""
        tool_cls = cls._registry.get(tool_name)
        return tool_cls.to_action_str(arguments) if tool_cls else f"Using {tool_name}"

    @classmethod
    def format_result(cls, tool_name: str, arguments: dict, result: ToolResult) -> str:
        """Frame a result as a tagged, first-person reply to the model's OWN call.

        Two parts, composed here so the invariant holds uniformly for every tool —
        current and future — in one place:

        1. A first-person **narration** (``to_result_narration``, registry-dispatched
           just as ``format_status`` dispatches ``to_action_str``; generic default
           here, per-tool overrides in #1480–#1482), branching on ``result.success``.
        2. A retained machine **tag** ``(<tool> result)``.

        The tag is load-bearing, not decorative.  The OpenAI ``role: "tool"`` +
        ``tool_call_id`` envelope already marks this a tool result structurally, but
        gpt-oss:20b doesn't reliably honour that when the body reads like prose — it
        can mistake fetched data (a returned user message that itself reads like an
        instruction) for a fresh directive (#1332's #1 failure class).  The terse
        ``Result of your `<tool>` call:`` header used to carry that disambiguation; a
        live-model probe showed pure-prose narration with no tag RAISED the
        call-as-text bail rate (5/6 vs. 3/6 tagged), so the tag stays even though the
        header now reads naturally.  Read tools additionally lead their body with a
        count + source line (see ``_format_entries``).
        """
        narration = cls._resolve_narration(tool_name, arguments, result)
        tag = RESULT_TAG.format(tool_name=tool_name)
        return f"{narration} {tag}\n{result.message}"

    @classmethod
    def _resolve_narration(cls, tool_name: str, arguments: dict, result: ToolResult) -> str:
        """Pick the first-person frame for a result, in priority order.

        An explicit ``result.narration`` wins — that's how a framework-synthesised
        failure the tool can't narrate itself (tool-not-found, timeout, uncaught
        exception, bad arguments; #1482) injects its specific frame, since a
        tool-not-found has no registered class to dispatch from and the others know
        *why* they failed in a way the generic per-tool line can't.  Otherwise dispatch
        to the tool's ``to_result_narration`` (the #1480–#1481 per-tool overrides), or
        the generic default for an unregistered tool.  This keeps the framework failures
        from being double-framed — the frame lives on the result, not baked into the
        remedy body that ``format_result`` would then narrate a second time."""
        if result.narration is not None:
            return result.narration
        tool_cls = cls._registry.get(tool_name)
        if tool_cls is not None:
            return tool_cls.to_result_narration(arguments, result)
        return cls._default_result_narration(tool_name, result)

    @classmethod
    def format_progress_emoji(cls, tool_name: str, arguments: dict) -> ProgressEmoji:
        """Dispatch to the matching tool's to_progress_emoji via the class registry."""
        tool_cls = cls._registry.get(tool_name)
        return tool_cls.to_progress_emoji(arguments) if tool_cls else ProgressEmoji.WORKING

    def to_definition(self) -> ToolDefinition:
        """Convert to tool definition for prompt."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def to_ollama_tool(self) -> dict[str, Any]:
        """Convert to Ollama tool calling format.

        Injects a ``reasoning`` property so the model can explain why it is
        making this tool call — a structured per-call rationale the run
        record captures for display and safe re-exposure in logs the model
        later reads (raw thinking is never fed back).  The field is stripped
        before the tool executes (see ``Agent._dedup_tool_calls``).  One
        carve-out: a tool that declares ``reasoning`` in its own
        ``parameters`` keeps its hand-written description (browse) —
        injection never overwrites a tool's own declaration.
        """
        params = dict(self.parameters)
        props = dict(params.get("properties", {}))
        if "reasoning" not in props:
            props["reasoning"] = {
                "type": "string",
                "description": (
                    "Explain what you're looking for and what you'll do with the result. "
                    "This is your inner monologue — think out loud."
                ),
            }
        params["properties"] = props
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        """Initialize empty registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_definitions(self) -> list[ToolDefinition]:
        """Get all tool definitions for prompt building."""
        return [tool.to_definition() for tool in self._tools.values()]

    def get_ollama_tools(self) -> list[dict[str, Any]]:
        """Get all tools in Ollama format for tool calling."""
        return [tool.to_ollama_tool() for tool in self._tools.values()]


class ToolExecutor:
    """Executes tools with timeout and error handling."""

    def __init__(self, registry: ToolRegistry, timeout: float = 30.0):
        self.registry = registry
        self.timeout = timeout

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Run a tool call: resolve the tool, then drive it through ``tool.run``
        (which validates args against the tool's ``args_model`` before executing).

        The executor owns only the framework concerns the tool can't: an unknown
        tool name, a timeout, an uncaught exception.  Argument validation lives on
        the tool (its ``args_model``), not here."""
        tool = self.registry.get(tool_call.tool)
        if tool is None:
            return self._tool_not_found_result(tool_call)
        return await self._execute_with_timeout(tool, tool_call)

    def _tool_not_found_result(self, tool_call: ToolCall) -> ToolResult:
        """Build a failed result when the requested tool doesn't exist.

        Framed by ``FRAMEWORK_NARRATION_NOT_FOUND`` (the frame — set on ``narration``
        so ``format_result`` leads with it and doesn't double-frame this
        no-registered-class result via its generic fallback); the did-you-mean +
        available-tools remedy stays the actionable body."""
        logger.error("Tool not found: %s", tool_call.tool)
        available_tools = [t.name for t in self.registry.get_all()]
        available_list = ", ".join(available_tools) if available_tools else "none"
        close = difflib.get_close_matches(tool_call.tool, available_tools, n=1, cutoff=0.6)
        did_you_mean = f"Did you mean '{close[0]}'? " if close else ""
        return ToolResult(
            narration=FRAMEWORK_NARRATION_NOT_FOUND.format(tool_name=tool_call.tool),
            message=(
                f"{did_you_mean}"
                f"Available tools: {available_list}. "
                f"You must ONLY use the tools listed above."
            ),
            success=False,
        )

    async def _execute_with_timeout(self, tool: Tool, tool_call: ToolCall) -> ToolResult:
        """Drive the tool through ``run`` (validate-then-execute) under a timeout.

        ``run`` returns its own ``ToolResult`` — including the actionable failure
        for invalid arguments.  Framework failures the tool can't report (timeout,
        uncaught exception) are synthesised into a failed ``ToolResult`` here.
        """
        effective_timeout = tool.timeout if tool.timeout is not None else self.timeout
        try:
            logger.info("Executing tool: %s", tool_call.tool)
            logger.debug("Tool arguments: %s", tool_call.arguments)
            result = await asyncio.wait_for(
                tool.run(**tool_call.arguments),
                timeout=effective_timeout,
            )
            logger.info("Tool executed successfully: %s", tool_call.tool)
            logger.debug("Tool result: %s", result)
            return result if isinstance(result, ToolResult) else ToolResult(message=str(result))
        except TimeoutError:
            logger.error("Tool execution timeout: %s", tool_call.tool)
            return ToolResult(
                narration=FRAMEWORK_NARRATION_TIMEOUT.format(tool_name=tool_call.tool),
                message=f"It timed out after {effective_timeout}s. "
                f"It may be slow or unavailable — try a simpler request (e.g. one URL or a "
                f"narrower query), or proceed without it rather than retrying the same call.",
                success=False,
            )
        except Exception as e:
            logger.exception("Tool execution error: %s", tool_call.tool)
            return ToolResult(
                narration=FRAMEWORK_NARRATION_EXCEPTION.format(tool_name=tool_call.tool, error=e),
                message=f"Check the arguments you passed against the tool's parameters; if they "
                f"look right, try a different approach{self._finish_clause()} rather than "
                f"repeating the same call.",
                success=False,
            )

    def _finish_clause(self) -> str:
        """`` or call done() to finish`` only when a ``done`` tool is registered.

        The collector shapes carry ``done``; the chat agent does not, so the crash
        envelope must not point a chat run at a tool it can't call."""
        if self.registry.get(PennyConstants.DONE_TOOL_NAME) is not None:
            return " or call done() to finish"
        return ""
