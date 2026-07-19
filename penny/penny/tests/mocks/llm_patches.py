"""Mock for LlmClient — returns LlmResponse objects directly."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import pytest

from penny.llm.models import LlmMessage, LlmResponse, LlmToolCall, LlmToolCallFunction

# Dimension of the deterministic default embedding.  The embedding model is a
# required prerequisite, so every test constructs an embedding client and the
# startup backfill vectorizes seeded memories through this mock.  A single
# fixed dimension (shared by every embed path, including the recall tests'
# anchors) keeps vectors comparable — a mixed-dimension corpus would crash
# cosine similarity.
EMBED_DIM = 4096


def deterministic_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Bag-of-words deterministic embedding for tests.

    Each word picks an axis via SHA-256; the vector is L2-normalised so cosine
    is comparable across strings.  Identical strings map to identical vectors,
    strings sharing words have cosine > 0, and fully-distinct strings map to
    cosine ≈ 0 — so recall behaves realistically (unrelated content doesn't
    spuriously match) instead of collapsing every pair to cosine 1.0.
    """
    vec = [0.0] * dim
    words = text.lower().split() or [text]
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        axis = int.from_bytes(digest[:8], "big") % dim
        vec[axis] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class MockEmbedResponse:
    """Wrapper for embed results to match the embeddings.create() interface."""

    def __init__(self, embeddings: list[list[float]]):
        self.data = [type("Obj", (), {"embedding": e})() for e in embeddings]


class MockLlmClient:
    """Mock for LlmClient that returns configurable responses."""

    def __init__(self, **kwargs: Any):
        self.model = kwargs.get("model", "test-model")
        self.db = kwargs.get("db")
        self.requests: list[dict] = []
        self._response_handler: Callable[[dict, int], LlmResponse] | None = None
        self._request_count = 0
        self._embed_handler: Callable[[str, str | list[str]], list[list[float]]] | None = None
        self.embed_requests: list[dict] = []
        # Browse micro-context calls answered by the built-in intercept — kept off
        # ``requests`` so per-call-count assertions in flow tests stay stable.
        self.micro_requests: list[dict] = []

    def set_response_handler(self, handler: Callable[[dict, int], LlmResponse]) -> None:
        """Set a custom response handler.

        Args:
            handler: Function that takes (request_data, request_count) and returns LlmResponse
        """
        self._response_handler = handler

    def set_default_flow(
        self, final_response: str = "test response", search_query: str = "test query"
    ) -> None:
        """Set up default two-step flow: tool call then final response."""

        def handler(request: dict, count: int) -> LlmResponse:
            if count == 1:
                return self._make_tool_call_response(
                    request, "browse", {"queries": [search_query], "extract": "the page content"}
                )
            return self._make_text_response(request, final_response)

        handler.answers_micro_contexts = False  # default flow: let the intercept serve them
        self._response_handler = handler

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        format: dict | str | None = None,
        agent_name: str | None = None,
        prompt_type: str | None = None,
        run_id: str | None = None,
        run_target: str | None = None,
    ) -> LlmResponse:
        """Mock chat() call."""
        request_data = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "agent_name": agent_name,
            "prompt_type": prompt_type,
            "run_target": run_target,
        }
        # A browse micro-context call (#1588/#1570 — extract is required, so EVERY
        # browse routes page content through one): answer deterministically with the
        # page text tagged EXTRACTED, and do NOT count it toward the flow counter —
        # response handlers keyed on call ordinals (set_default_flow) stay stable.
        # Only when no custom handler claims micro-context calls: a test that sets
        # its own handler (the micro-context contract tests) keeps full control.
        handler_defers = self._response_handler is None or not getattr(
            self._response_handler, "answers_micro_contexts", True
        )
        system = messages[0].get("content", "") if messages else ""
        if (
            handler_defers
            and isinstance(system, str)
            and system.startswith("You are an extraction step.")
        ):
            self.micro_requests.append(request_data)
            user_content = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
            )
            return self._make_text_response(request_data, f"EXTRACTED: {user_content}")
        self.requests.append(request_data)
        self._request_count += 1

        if self._response_handler:
            return self._response_handler(request_data, self._request_count)
        return self._make_text_response(request_data, "default mock response")

    async def generate(
        self,
        prompt: str,
        tools: list[dict] | None = None,
        format: dict | str | None = None,
        agent_name: str | None = None,
        prompt_type: str | None = None,
        run_id: str | None = None,
    ) -> LlmResponse:
        """Mock generate() call — wraps chat like the real client."""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(
            messages,
            tools,
            format,
            agent_name=agent_name,
            prompt_type=prompt_type,
            run_id=run_id,
        )

    def set_embed_handler(
        self, handler: Callable[[str, str | list[str]], list[list[float]]]
    ) -> None:
        """Set a custom embed handler: (model, input) -> list of embedding vectors."""
        self._embed_handler = handler

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        """Mock embed() call."""
        self.embed_requests.append({"model": self.model, "input": text})

        if self._embed_handler:
            return self._embed_handler(self.model, text)

        # Default: deterministic, distinct, dimension-consistent vectors so
        # backfilled seeds and recall-test anchors never clash on dimension.
        texts = [text] if isinstance(text, str) else text
        return [deterministic_embed(t) for t in texts]

    async def close(self) -> None:
        """Mock close."""

    # ── Response builders ────────────────────────────────────────────────

    def _make_text_response(self, request: dict, content: str) -> LlmResponse:
        """Create a text-only response."""
        return LlmResponse(
            message=LlmMessage(role="assistant", content=content),
            model=request.get("model", "test-model"),
        )

    def _make_tool_call_response(
        self, request: dict, tool_name: str, arguments: dict[str, Any]
    ) -> LlmResponse:
        """Create a response with a single tool call."""
        return self._make_parallel_tool_calls_response(request, [(tool_name, arguments)])

    def _make_parallel_tool_calls_response(
        self, request: dict, tool_calls: list[tuple[str, dict[str, Any]]]
    ) -> LlmResponse:
        """Create a response with multiple tool calls in a single turn."""
        return LlmResponse(
            message=LlmMessage(
                role="assistant",
                content="",
                tool_calls=[
                    LlmToolCall(
                        id=f"call_{i}",
                        function=LlmToolCallFunction(name=name, arguments=args),
                    )
                    for i, (name, args) in enumerate(tool_calls)
                ],
            ),
            model=request.get("model", "test-model"),
        )


# Shared instance for tests to configure and inspect
_mock_client: MockLlmClient | None = None


def _create_mock_client(**kwargs: Any) -> MockLlmClient:
    """Factory that returns the shared mock client instance."""
    global _mock_client
    if _mock_client is None:
        _mock_client = MockLlmClient(**kwargs)
    return _mock_client


@pytest.fixture
def mock_llm(monkeypatch: pytest.MonkeyPatch) -> MockLlmClient:
    """Fixture to patch LlmClient with a mock.

    Patches at both the source module and common import locations so that
    tests which construct LlmClient(...) directly get the mock.
    """
    global _mock_client
    _mock_client = MockLlmClient()

    # Patch the internal OpenAI client constructor so any LlmClient(...)
    # constructed during tests gets a mock instead of a real HTTP client.
    # This mirrors the old approach of patching ollama.AsyncClient.
    class _MockOpenAIClient:
        """Stands in for openai.AsyncOpenAI — routes calls to MockLlmClient."""

        def __init__(self, **kwargs: Any):
            pass

        @property
        def chat(self) -> Any:
            return self

        @property
        def completions(self) -> Any:
            return self

        async def create(self, **kwargs: Any) -> Any:
            # Route to the mock's handler, temporarily swapping model
            # so request recording captures the right model name
            messages = kwargs.get("messages", [])
            tools = kwargs.get("tools")
            model = kwargs.get("model", _mock_client.model)
            old_model = _mock_client.model
            _mock_client.model = model
            response = await _mock_client.chat(messages=messages, tools=tools)
            _mock_client.model = old_model
            return _FakeCompletion(response)

        @property
        def embeddings(self) -> Any:
            return _MockEmbeddings()

        async def close(self) -> None:
            pass

    class _MockEmbeddings:
        """Stands in for client.embeddings."""

        async def create(self, **kwargs: Any) -> Any:
            text = kwargs.get("input", "")
            model = kwargs.get("model", "test-model")
            # Override the mock's model so embed_requests records the right one
            old_model = _mock_client.model
            _mock_client.model = model
            result = await _mock_client.embed(text)
            _mock_client.model = old_model
            return type(
                "EmbedResponse",
                (),
                {"data": [type("Item", (), {"embedding": vec})() for vec in result]},
            )()

    monkeypatch.setattr("penny.llm.client.openai.AsyncOpenAI", _MockOpenAIClient)

    # Prevent list_models from hitting the real API. The Ollama image client
    # reports nothing; the OpenAI-compatible LlmClient reports its own configured
    # model as available so the startup preflight resolves the chat + embedding
    # models cleanly instead of hard-failing every test.
    async def _mock_image_list_models(self: object) -> list[str]:
        return []

    async def _mock_llm_list_models(self: Any) -> list[str]:
        return [self.model]

    monkeypatch.setattr(
        "penny.llm.image_client.OllamaImageClient.list_models", _mock_image_list_models
    )
    monkeypatch.setattr("penny.llm.client.LlmClient.list_models", _mock_llm_list_models)

    return _mock_client


class _FakeCompletion:
    """Wraps an LlmResponse to look like an OpenAI ChatCompletion."""

    def __init__(self, response: LlmResponse):
        self._response = response
        self.model = response.model or "test-model"
        self.choices = [_FakeChoice(response)]

    def model_dump(self) -> dict:
        # Mirror a real OpenAI ChatCompletion dump: the message carries its
        # ``tool_calls`` (id + function name + JSON-string arguments) so the logged
        # promptlog ``response`` is faithful — the ledger readers (``read_run_calls``,
        # the run-end skill extractor, #1658) parse tool steps out of exactly this.
        message: dict[str, Any] = {"role": "assistant", "content": self._response.content}
        if self._response.message.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": json.dumps(call.function.arguments),
                    },
                }
                for call in self._response.message.tool_calls
            ]
        return {"model": self.model, "choices": [{"message": message}]}


class _FakeChoice:
    """Wraps an LlmResponse to look like an OpenAI Choice."""

    def __init__(self, response: LlmResponse):
        self.message = _FakeMessage(response)


class _FakeMessage:
    """Wraps an LlmMessage to look like an OpenAI ChatCompletionMessage.

    Mirrors pydantic v2's ``model_extra`` dict for non-standard fields like
    ``reasoning_content`` so the parser doesn't need to special-case the mock.
    """

    def __init__(self, response: LlmResponse):
        self.role = response.message.role
        self.content = response.message.content
        self.model_extra = {"reasoning_content": response.message.thinking}
        self.tool_calls = None
        if response.message.tool_calls:
            self.tool_calls = [_FakeToolCall(tc) for tc in response.message.tool_calls]


class _FakeToolCall:
    """Wraps an LlmToolCall to look like an OpenAI tool call."""

    def __init__(self, tool_call: Any):
        import json

        self.id = tool_call.id
        self.type = "function"
        self.function = type(
            "Fn",
            (),
            {
                "name": tool_call.function.name,
                "arguments": json.dumps(tool_call.function.arguments),
            },
        )()
