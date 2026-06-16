"""Base classes for tools."""

import asyncio
import difflib
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from penny.constants import ProgressEmoji
from penny.tools.models import ToolCall, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Abstract base class for tools."""

    name: str
    description: str
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    timeout: float | None = None  # None = use ToolExecutor's global timeout

    _registry: ClassVar[dict[str, type[Tool]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" in cls.__dict__:
            Tool._registry[cls.name] = cls

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool.

        Args:
            **kwargs: Tool parameters

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
    def format_result(cls, tool_name: str, body: str) -> str:
        """Frame a result so the model reads it as the response to ITS own call.

        The OpenAI ``role: "tool"`` + ``tool_call_id`` envelope already marks
        this as a tool result structurally, but smaller local models don't
        reliably honour that primitive when the body reads like prose — they
        can mistake fetched data (e.g. a returned user message) for a fresh
        instruction directed at them.  A one-line content header naming the
        originating tool removes the ambiguity uniformly, for every tool —
        current and future — in one place, so this never has to be solved
        per-tool again.  Read tools additionally lead their body with a
        count + source line (see ``_format_entries``).
        """
        return f"Result of your `{tool_name}` call:\n{body}"

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

        Injects a ``reasoning`` property so the model can explain why
        it is making this tool call.  The field is stripped before the
        tool is actually executed (see ``Agent._execute_single_tool``).
        """
        params = dict(self.parameters)
        props = dict(params.get("properties", {}))
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

    def _validate_arguments(self, tool: Tool, arguments: dict[str, Any]) -> str | None:
        """Validate that all required parameters are present in arguments."""
        parameters = tool.parameters
        required_params = parameters.get("required", [])
        missing_params = [param for param in required_params if param not in arguments]
        if missing_params:
            return self._missing_params_error(missing_params, parameters.get("properties", {}))
        return None

    @staticmethod
    def _missing_params_error(missing: list[str], properties: dict[str, Any]) -> str:
        """Build a hint-rich error message listing missing params with their types/descriptions."""
        hints = []
        for param in missing:
            prop = properties.get(param, {})
            param_type = prop.get("type", "")
            param_desc = prop.get("description", "")
            if param_type and param_desc:
                hints.append(f"{param} ({param_type}: {param_desc})")
            elif param_type:
                hints.append(f"{param} ({param_type})")
            else:
                hints.append(param)
        return (
            f"Missing required parameter(s): {', '.join(hints)}. "
            f"Please call the tool again with all required parameters."
        )

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        tool = self.registry.get(tool_call.tool)
        if tool is None:
            return self._tool_not_found_result(tool_call)
        validation_error = self._validate_arguments(tool, tool_call.arguments)
        if validation_error:
            return self._validation_error_result(tool_call, validation_error)
        return await self._execute_with_timeout(tool, tool_call)

    def _tool_not_found_result(self, tool_call: ToolCall) -> ToolResult:
        """Build a failed result when the requested tool doesn't exist."""
        logger.error("Tool not found: %s", tool_call.tool)
        available_tools = [t.name for t in self.registry.get_all()]
        available_list = ", ".join(available_tools) if available_tools else "none"
        close = difflib.get_close_matches(tool_call.tool, available_tools, n=1, cutoff=0.6)
        suggestion = f" Did you mean '{close[0]}'?" if close else ""
        return ToolResult(
            message=(
                f"Error: Tool '{tool_call.tool}' not found.{suggestion} "
                f"Available tools: {available_list}. "
                f"You must ONLY use the tools listed above."
            ),
            success=False,
        )

    def _validation_error_result(self, tool_call: ToolCall, error: str) -> ToolResult:
        """Build a failed result for argument validation failure."""
        logger.error("Tool call validation failed: %s - %s", tool_call.tool, error)
        return ToolResult(message=f"Error: {error}", success=False)

    async def _execute_with_timeout(self, tool: Tool, tool_call: ToolCall) -> ToolResult:
        """Execute tool with timeout and error handling.

        A tool returns its own ``ToolResult``; a bare string is tolerated and
        wrapped.  Framework failures the tool can't report (timeout, uncaught
        exception) are synthesised into a failed ``ToolResult`` here.
        """
        try:
            logger.info("Executing tool: %s", tool_call.tool)
            logger.debug("Tool arguments: %s", tool_call.arguments)
            effective_timeout = tool.timeout if tool.timeout is not None else self.timeout
            result = await asyncio.wait_for(
                tool.execute(**tool_call.arguments),
                timeout=effective_timeout,
            )
            logger.info("Tool executed successfully: %s", tool_call.tool)
            logger.debug("Tool result: %s", result)
            return result if isinstance(result, ToolResult) else ToolResult(message=str(result))
        except TimeoutError:
            logger.error("Tool execution timeout: %s", tool_call.tool)
            return ToolResult(
                message=f"Error: '{tool_call.tool}' timed out after {effective_timeout}s. "
                f"It may be slow or unavailable — try a simpler request (e.g. one URL or a "
                f"narrower query), or proceed without it rather than retrying the same call.",
                success=False,
            )
        except Exception as e:
            logger.exception("Tool execution error: %s", tool_call.tool)
            return ToolResult(
                message=f"Error: '{tool_call.tool}' failed — {e}. Check the arguments you "
                f"passed against the tool's parameters; if they look right, try a different "
                f"approach or call done to finish rather than repeating the same call.",
                success=False,
            )
