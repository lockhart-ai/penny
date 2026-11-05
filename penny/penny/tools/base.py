"""Base classes for tools."""

import asyncio
import difflib
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import ValidationError

from penny.constants import PennyConstants, ProgressEmoji
from penny.llm.call_link import CallLink
from penny.text_validity import suggest_tool_name
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
# The fixed, tool-name-independent portion of the exception narration below — named
# separately (#2953) so a reader of the promptlog (the eval harness's own
# ``tool_call_raised_exception``) can detect "a tool call escaped with an uncaught
# exception" by this marker alone, never by which tool it was, the same way
# production never keys behaviour to one.  An honest ToolResult failure a tool
# returns itself (a refusal naming why) never carries this text.
FRAMEWORK_TOOL_EXCEPTION_MARKER = "but it errored:"
FRAMEWORK_NARRATION_EXCEPTION = (
    f"You tried to use `{{tool_name}}` {FRAMEWORK_TOOL_EXCEPTION_MARKER} {{error}}."
)
FRAMEWORK_NARRATION_INVALID_ARGS = "You tried to use `{tool_name}` but the arguments were wrong:"


# The plan flag every mutating tool gets automatically (#2933) — a mutating
# tool never declares this by hand on its own args_model or parameters;
# ``Tool.__init_subclass__`` derives both from the tool's own declarations the
# moment ``mutates = True`` is set.
PLAN_ARG_DESCRIPTION = (
    "Validate and describe what this call would do, without doing it — nothing "
    "is written, sent, or changed."
)

# The plan result's headline, built ONCE here so every mutating tool composes
# the SAME disclaimer rather than writing its own — a plan result must never
# read as a done action (the #2915 class of claiming a change that never
# happened).  ``as_plan_result`` wraps whatever a tool already determined the
# real call would do, so the description a plan gives is never a separately
# authored guess that could drift from what apply actually does.
PLAN_MESSAGE_PREFIX = "PLAN — nothing was changed: this call would"


def as_plan_result(result: ToolResult) -> ToolResult:
    """Turn an ordinary mutating-tool result into its plan twin (#2933): the
    fixed headline every mutating tool shares, in front of the SAME
    description the tool built for what the call would do.  Always
    ``mutated=False`` — nothing really changed, whatever the real call would
    have done.

    ``result.success`` decides which headline: a would-succeed description
    reads "this call would — {message}"; a refusal (``success=False`` —
    collection_configure's same-job near-match, a bad schedule, #2946) reads
    "this call would be refused — {message}", and KEEPS ``success=False`` —
    a plan that correctly foresees a refusal is an honest plan, not a failed
    one, but it must never claim the refused call would otherwise succeed.

    Both texts ride the result (#2995): ``plan_description`` is the full plan
    wording just built, ``applied_message`` is the tool's own description the
    call would read as if applied — the wording ``ToolExecutor`` shows a
    sandboxed caller (#2964) in place of this plan, since a sandbox run must
    look live to the job while nothing persists."""
    verb = "" if result.success else "be refused "
    return result.model_copy(
        update={
            "message": f"{PLAN_MESSAGE_PREFIX} {verb}— {result.message}",
            "mutated": False,
            "plan_description": f"{PLAN_MESSAGE_PREFIX} {verb}— {result.message}",
            "applied_message": result.message,
        }
    )


def _plan_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Derive ``parameters`` with the ``plan`` property merged in — a fresh
    dict, so this never mutates a class-level default dict other tools share."""
    properties = parameters.get("properties", {})
    if "plan" in properties:
        return parameters
    merged = dict(parameters)
    merged["properties"] = {
        **properties,
        "plan": {"type": "boolean", "description": PLAN_ARG_DESCRIPTION},
    }
    return merged


def _tool_lead_sentence(description: str) -> str:
    """The lead clause of a tool's ``description`` — up to its first sentence break
    (``". "``) or, lacking one, its first line — so a did-you-mean hint (#2510) can
    carry how the suggested tool works without quoting its whole description."""
    line = description.strip().splitlines()[0].strip()
    end = line.find(". ")
    return line[: end + 1] if end != -1 else line


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
    # The tools THIS tool's own rendered text may direct the model to (#1911).  A
    # result message that says "call ``collection_write(key=…)``" is an ANCHOR: the n≤1
    # discipline says a rendered instruction resolves in one call, so it must never
    # name a tool the surface does not carry.  A job's cycle gets the FIXED
    # chat-minus-lifecycle surface (#2455 naive baseline, not a program-scoped one), so
    # every advised sibling is already present there — but the mark stays useful on its
    # own terms: it is what a tool declares about what its own text points at, enforced
    # by a test that walks each tool class's own string literals for references to any
    # registered tool name and fails when one is not declared here.
    advises: tuple[str, ...] = ()
    # Whether a call to this tool asks Penny's store to KEEP a keyed entry (#2185).  It is
    # the attempt half of the question the run-end writes record answers: the landed half
    # is read off the entries' own run stamps, which no tool can get wrong, but a write the
    # store REFUSED leaves no stamp, and a run whose every attempt was refused would
    # otherwise look exactly like a run that never tried.  Declared here, beside the tool,
    # for the reason ``advises`` is: a tool nobody has written yet says so on itself, and
    # nothing that reads the mark ever names a tool.  Keyed entries only, because that is
    # what the landed half counts — a keyless stream append is not a save.
    stores_entries: bool = False
    # The argument that names the collection a storing tool writes to (#2273) — set
    # beside ``stores_entries`` on every tool that carries one, so a caller can read a
    # call's declared DESTINATION off the tool's own declaration rather than guessing
    # which argument or position holds it. ``None`` for a tool that stores nothing, or
    # one whose destination is implicit (scoped to a single bound collection, with no
    # argument naming it at all).
    destination_param: str | None = None
    # Whether a call to this tool changes durable state or has an outbound side
    # effect — a write, a send, an external create/update/delete (#2933).  Set
    # ``True`` on the tool's OWN class body (never inherited silently, the same
    # way ``stores_entries`` is), and ``Tool.__init_subclass__`` reads it to add
    # a ``plan`` argument to the tool's ``args_model`` and ``parameters``
    # AUTOMATICALLY — nothing is hand-added per tool, so a future mutating tool
    # gets `plan` just by declaring this mark.
    mutates: bool = False

    _registry: ClassVar[dict[str, type[Tool]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" in cls.__dict__:
            Tool._registry[cls.name] = cls
        if cls.mutates:
            # The ``parameters`` half (the model-facing schema) IS derived here,
            # automatically — a plain dict carries no static type for a checker
            # to lose.  The ``args_model`` half is a Pydantic class, and a class
            # synthesised at import time (``pydantic.create_model``) is invisible
            # to the type checker — every ``args.plan`` read below would be an
            # ``unresolved-attribute`` error with no fix visible at the call
            # site.  So each mutating tool's OWN ``args_model`` declares its
            # literal ``plan: bool`` field (statically checkable), and this
            # assertion is the structural guarantee that declaring ``mutates``
            # without it is impossible to ship silently — a tool that sets
            # ``mutates = True`` and forgets the field fails loudly, at import
            # time, naming exactly what is missing.
            if "plan" not in cls.args_model.model_fields:
                raise TypeError(
                    f"{cls.__name__} sets mutates=True but its args_model "
                    f"({cls.args_model.__name__}) declares no 'plan: bool' field"
                )
            cls.parameters = _plan_parameters(cls.parameters)

    async def run(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """Validate the call's arguments via ``args_model``, then ``execute``.

        The single validate-then-handle gate every tool goes through (the
        ``ToolExecutor`` calls this, not ``execute``).  A validation failure
        returns an actionable error tool response and ``execute`` is never reached
        with invalid args — so ``execute`` can trust its inputs and concern itself
        only with the work + any runtime/availability decisions (e.g. mute state)
        that aren't expressible as arg validation.

        ``call_link`` (#2458) is REQUIRED on every tool, not just the ones that draw
        — uniformly, so a new tool that makes a model call internally can't forget
        to accept it (the code owner's ruling: a declared opt-in flag is exactly
        the forgettable shape this replaces).  Forwarded to ``execute`` unchanged;
        most tools never read it.
        """
        try:
            self.args_model(**kwargs)
        except ValidationError as exc:
            return ToolResult(
                narration=FRAMEWORK_NARRATION_INVALID_ARGS.format(tool_name=self.name),
                message=self._validation_error_message(exc),
                success=False,
            )
        return await self.execute(call_link=call_link, **kwargs)

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
        return f"{'; '.join(parts)}. Call {self.name}{{<valid arguments>}} again."

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
        close = difflib.get_close_matches(
            bad, valid, n=1, cutoff=PennyConstants.DID_YOU_MEAN_STRING_CUTOFF
        )
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

    def timeout_report(self, timeout: float) -> str:
        """The ``ToolResult.message`` body for a call that hit its timeout.

        Called by ``ToolExecutor`` AFTER ``asyncio.wait_for`` has cancelled this
        tool's call — so this reads only state the tool wrote to ITSELF before the
        cancellation landed (the cancelled coroutine's own locals are gone by the
        time this runs).  The default here is a fully generic sentence, true of any
        tool whether or not it tracks partial progress.  A tool that accumulates its
        own progress as it runs (browse) overrides this to report what finished
        instead — the recovery guard this issue adds, as a fallback: no new timer,
        the one that just fired is still the only bound (#2367)."""
        return (
            f"It timed out after {timeout}s. It may be slow or unavailable — try a "
            f"simpler request, or proceed without it rather than retrying the same call."
        )

    @abstractmethod
    async def execute(self, call_link: CallLink, **kwargs) -> ToolResult:
        """
        Execute the tool with already-validated arguments.

        ``run`` validates ``kwargs`` against ``args_model`` before this is
        called, so ``execute`` never sees invalid args.

        ``call_link`` is this dispatched call's own linkage (#2458) — required on
        every tool uniformly; a tool that draws internally (a micro-context) passes
        it straight through, every other tool ignores it.

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
    def asks_to_store(cls, tool_name: str) -> bool:
        """Whether a call named ``tool_name`` asked the store to keep an entry — the
        registered tool's own ``stores_entries`` mark, and ``False`` for a name no tool
        answers to (a call to nothing kept nothing, and tried to keep nothing either)."""
        tool_cls = cls._registry.get(tool_name)
        return tool_cls.stores_entries if tool_cls else False

    @classmethod
    def destination_for(cls, tool_name: str, arguments: dict) -> str | None:
        """The collection a call to ``tool_name`` declared as its DESTINATION (#2273) —
        read off the registered tool's own ``destination_param``, never a name or
        position hard-coded here. ``None`` for an unknown tool, a tool that declares no
        destination argument, or a call whose destination argument is absent or not a
        string.

        ``arguments`` may be a LOGGED call's — one made before #2307 renamed
        ``collection_write``'s first argument from ``memory`` to ``collection`` — so a
        miss on the CURRENT ``destination_param`` falls back to ``memory`` rather than
        reading an old call as having named no destination at all."""
        tool_cls = cls._registry.get(tool_name)
        if tool_cls is None or tool_cls.destination_param is None:
            return None
        value = arguments.get(tool_cls.destination_param)
        if value is None and tool_cls.destination_param == "collection":
            value = arguments.get("memory")
        return value if isinstance(value, str) else None

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
        count + source line (see ``format_entries``).
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

        No longer injects a ``reasoning`` property (#2813): gemma thinks
        natively and its thinking is passed back between steps, so a separate
        per-call rationale field cost tokens (schema + the model filling it)
        for no reasoning the loop ever used.  A model may still send a stray
        ``reasoning`` argument for a while out of habit — ``Agent.
        _dedup_tool_calls`` tolerates it (pops it before arg validation so
        ``extra="forbid"`` never rejects it) without depending on its value.
        """
        params = dict(self.parameters)
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

    def __init__(self, registry: ToolRegistry, timeout: float = 30.0, *, forced_plan: bool):
        self.registry = registry
        self.timeout = timeout
        # Whether every ``mutates`` tool call this executor dispatches is forced
        # onto its plan path regardless of what the model passed (#2964's sandbox
        # runs) — a REQUIRED parameter, no default: every construction site states
        # it explicitly (ordinary chat/collector runs say ``False``; a sandboxed
        # collector cycle says ``True``), so a future call site can't forget to
        # wire it the way an optional flag quietly could.  Keyed on the tool's own
        # ``mutates`` mark in ``execute`` below — never on a tool's name — so a
        # future mutating tool is covered with no change here.
        self.forced_plan = forced_plan

    async def execute(self, tool_call: ToolCall, call_link: CallLink) -> ToolResult:
        """Run a tool call: resolve the tool, then drive it through ``tool.run``
        (which validates args against the tool's ``args_model`` before executing).

        The executor owns only the framework concerns the tool can't: an unknown
        tool name, a timeout, an uncaught exception.  Argument validation lives on
        the tool (its ``args_model``), not here.

        ``call_link`` (#2458) is passed straight through — the caller
        (``Agent._execute_single_tool``) already built it via
        ``CallLink.for_tool_call(tool_call.id)``, so the executor forwards it rather
        than deriving anything itself."""
        tool = self.registry.get(tool_call.tool)
        if tool is None:
            return self._tool_not_found_result(tool_call)
        tool_call = self._force_plan(tool, tool_call)
        result = await self._execute_with_timeout(tool, tool_call, call_link)
        return self._unplan_for_sandbox(tool, result)

    def _unplan_for_sandbox(self, tool: Tool, result: ToolResult) -> ToolResult:
        """Show a sandboxed caller the AS-APPLIED wording of a plan-path result
        (#2995): a sandbox run must look live to the job — a forced-plan call's
        result reads like the ordinary result for that call — while nothing
        persists and the plan path still ran underneath.

        The swap carries the plan wording forward on ``plan_description`` (set by
        ``as_plan_result``, the structural mark every plan-path result carries —
        never a string check), so the run record keeps what the call WOULD have
        done and the sandbox report can render it
        (``notification.build_sandbox_report``).  Keyed on ``self.forced_plan`` and
        ``tool.mutates`` — never a tool's name — so a future mutating tool is
        covered with no change here.  A result that never went through a plan path
        (a framework failure: bad arguments, timeout, a crash; a refusal returned
        before the plan branch) carries ``plan_description=None`` and passes
        through untouched — an honest failure needs no re-wording."""
        if not (self.forced_plan and tool.mutates and result.plan_description is not None):
            return result
        # ``applied_message`` is set together with ``plan_description`` by
        # ``as_plan_result`` — read it plainly, so a broken pair fails loudly
        # rather than quietly showing the job the plan wording it must never see.
        assert result.applied_message is not None
        return result.model_copy(update={"message": result.applied_message})

    def _force_plan(self, tool: Tool, tool_call: ToolCall) -> ToolCall:
        """Overwrite ``plan`` to ``True`` on a mutating call, when this executor is
        forcing it (#2964) — whatever the model passed, or omitted.

        A sandbox run never lets a mutating tool actually change anything, so the
        model's own ``plan`` argument (if any) is not consulted, only overwritten:
        a model that passed ``plan=false`` must still get a preview, never a real
        write.  A fresh ``ToolCall`` (never mutating the caller's own dict) keyed on
        ``tool.mutates`` — the structural mark every state-changing tool declares on
        itself (#2933) — so this never names a tool and covers one nobody has
        written yet."""
        if not (self.forced_plan and tool.mutates):
            return tool_call
        return tool_call.model_copy(update={"arguments": {**tool_call.arguments, "plan": True}})

    def _tool_not_found_result(self, tool_call: ToolCall) -> ToolResult:
        """Build a failed result when the requested tool doesn't exist.

        Framed by ``FRAMEWORK_NARRATION_NOT_FOUND`` (the frame — set on ``narration``
        so ``format_result`` leads with it and doesn't double-frame this
        no-registered-class result via its generic fallback); the did-you-mean +
        available-tools remedy stays the actionable body.  The call itself stays
        REJECTED either way (#2510, principles.md §2 item 7) — a suggestion is a
        hint in the reply the model reads next, never a tool the executor ran.

        The suggestion goes through ``suggest_tool_name`` (#1944, #2510), which reads
        a retired name through the rename table, then string distance, then whether a
        real tool name sits inside the malformed one — a guessed ``collection_metadata``
        names ``memory_get`` rather than the character-nearest ``collection_configure``,
        and a guessed ``functions.done`` names ``done``."""
        logger.error("Tool not found: %s", tool_call.tool)
        available_tools = [t.name for t in self.registry.get_all()]
        available_list = ", ".join(available_tools) if available_tools else "none"
        close = suggest_tool_name(tool_call.tool, available_tools)
        did_you_mean = self._did_you_mean_hint(close)
        return ToolResult(
            narration=FRAMEWORK_NARRATION_NOT_FOUND.format(tool_name=tool_call.tool),
            message=(
                f"{did_you_mean}"
                f"Available tools: {available_list}. "
                f"You must ONLY use the tools listed above."
            ),
            success=False,
        )

    def _did_you_mean_hint(self, close: str | None) -> str:
        """The did-you-mean clause, naming ``close``'s own lead sentence (#2510) so
        the model reads how the tool it meant actually works, not just its name."""
        if close is None:
            return ""
        tool = self.registry.get(close)
        description = _tool_lead_sentence(tool.description) if tool else ""
        if description and not description.endswith("."):
            description += "."
        detail = f" {close}: {description}" if description else ""
        return f"Did you mean '{close}'?{detail} "

    async def _execute_with_timeout(
        self, tool: Tool, tool_call: ToolCall, call_link: CallLink
    ) -> ToolResult:
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
                tool.run(call_link, **tool_call.arguments),
                timeout=effective_timeout,
            )
            logger.info("Tool executed successfully: %s", tool_call.tool)
            logger.debug("Tool result: %s", result)
            return result if isinstance(result, ToolResult) else ToolResult(message=str(result))
        except TimeoutError:
            logger.error("Tool execution timeout: %s", tool_call.tool)
            return ToolResult(
                narration=FRAMEWORK_NARRATION_TIMEOUT.format(tool_name=tool_call.tool),
                message=tool.timeout_report(effective_timeout),
                success=False,
            )
        except Exception as e:
            logger.exception("Tool execution error: %s", tool_call.tool)
            return ToolResult(
                narration=FRAMEWORK_NARRATION_EXCEPTION.format(tool_name=tool_call.tool, error=e),
                message="Check the arguments you passed against the tool's parameters; "
                "if they look right, try a different approach rather than repeating the "
                "same call.",
                success=False,
            )
