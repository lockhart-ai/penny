"""Tests for LlmClient error summarization.

A non-Ollama backend can return a non-JSON error (e.g. a 404 served as an HTML
page). Logging the raw body dumped thousands of characters per occurrence and
buried real signal, so ``_summarize_llm_error`` reports the HTTP status plus a
short, body-free detail instead — and that summary is what propagates through
the raised ``LlmError`` as well.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion
from sqlmodel import select

from penny.config_params import RuntimeParams, token_budget_for
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.models import PromptLog
from penny.llm.call_link import CallLink
from penny.llm.client import (
    LlmClient,
    _extract_model_ids,
    _summarize_httpx_error,
    _summarize_llm_error,
)
from penny.llm.models import (
    LlmConnectionError,
    LlmError,
    LlmFault,
    LlmMessage,
    LlmNotFoundError,
    LlmResponse,
    LlmResponseError,
    LlmTimeoutError,
    ProviderPin,
    fault_for_status,
)

_HTML_ERROR_BODY = (
    f"<!DOCTYPE html><html><head><title>404</title></head><body>{'x' * 5000}</body></html>"
)


def _make_status_error(
    status: int, content_type: str, content: bytes, body: object | None
) -> openai.APIStatusError:
    """Build a real OpenAI status error carrying the given HTTP response."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(
        status, headers={"content-type": content_type}, content=content, request=request
    )
    return openai.NotFoundError("Error code: 404", response=response, body=body)


class TestSummarizeLlmError:
    def test_html_error_body_is_not_dumped(self) -> None:
        """A 404 served as an HTML page is summarized by type + length, never dumped."""
        error = _make_status_error(404, "text/html", _HTML_ERROR_BODY.encode(), body=None)

        summary = _summarize_llm_error(error)

        assert "HTTP 404" in summary
        assert "non-JSON error body" in summary
        assert "text/html" in summary
        assert "<!DOCTYPE" not in summary  # the raw body never leaks into the log
        assert len(summary) < len(_HTML_ERROR_BODY)  # summarized, not dumped

    def test_json_error_surfaces_message_field(self) -> None:
        """A structured JSON error surfaces its short ``message`` field with the status."""
        body = {"error": {"message": "model `foo` not found", "type": "invalid_request_error"}}
        error = _make_status_error(404, "application/json", json.dumps(body).encode(), body=body)

        assert _summarize_llm_error(error) == "HTTP 404: model `foo` not found"

    def test_connection_error_without_response_uses_own_message(self) -> None:
        """An error with no HTTP response (connection/timeout) falls back to its short str."""
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        error = openai.APIConnectionError(message="Connection refused", request=request)

        summary = _summarize_llm_error(error)

        assert "Connection refused" in summary


class TestChatPropagatesSummarizedError:
    """The summarized message — not the raw HTML body — is what reaches the
    raised ``LlmError``, so an onboarding/profile call that 404s surfaces a
    readable reason instead of a wall of HTML."""

    @pytest.mark.asyncio
    async def test_html_404_raises_summarized_not_found(self, monkeypatch) -> None:
        client = LlmClient(
            api_url="http://localhost:11434",
            model="missing-model",
            max_retries=1,
            retry_delay=0.0,
        )
        error = _make_status_error(404, "text/html", _HTML_ERROR_BODY.encode(), body=None)

        async def raise_not_found(**kwargs):
            raise error

        monkeypatch.setattr(client.client.chat.completions, "create", raise_not_found)

        with pytest.raises(LlmNotFoundError) as exc_info:
            await client.chat([{"role": "user", "content": "hi"}], call_link=CallLink.new_top())

        message = str(exc_info.value)
        assert "HTTP 404" in message
        assert "<!DOCTYPE" not in message  # summarized, not the raw HTML body

        await client.close()


def _completion(content: str, finish_reason: str = "stop"):
    """A minimal ChatCompletion-shaped response — what the SDK hands ``chat`` back."""

    class _Message:
        role = "assistant"
        tool_calls = None
        model_extra: dict = {}

        def __init__(self, text: str) -> None:
            self.content = text

    class _Choice:
        def __init__(self, text: str, reason: str) -> None:
            self.message = _Message(text)
            self.finish_reason = reason

    class _Completion:
        model = "m"
        usage = None

        def __init__(self, text: str, reason: str) -> None:
            self.choices = [_Choice(text, reason)]

        def model_dump(self) -> dict:
            return {"choices": [{"message": {"content": content}}]}

    return _Completion(content, finish_reason)


class TestReasoningAlwaysOn:
    """Reasoning is a property of every call, not a setting a run might forget."""

    @pytest.mark.asyncio
    async def test_every_chat_call_asks_for_reasoning(self, monkeypatch) -> None:
        """The same weights reason or do not depending on who serves them.

        Ollama runs a hybrid model with thinking on by default; a gateway serving that
        model defaults it off — a difference nothing in a run records, and one that cost a
        whole eval suite comparing a model against itself. So the switch is sent on every
        call rather than configured per run.
        """
        sent = {}

        async def capture(**kwargs):
            sent.update(kwargs)
            return _completion("ok")

        client = LlmClient(
            api_url="http://localhost:11434", model="m", max_retries=1, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", capture)
        await client.chat([{"role": "user", "content": "hi"}], call_link=CallLink.new_top())

        assert sent["extra_body"] == {"reasoning": {"enabled": True}}
        await client.close()


class TestTokenBudgetResolution:
    """``token_budget_for`` (#2474) maps a call kind's ``agent_name`` onto its own
    configured cap through the same ``RuntimeParams`` lookup chain every other
    runtime param already uses — a DB/env override takes effect the way it does
    for ``MAX_STEPS``, and an unrecognised kind sends no cap rather than a
    guessed number."""

    def test_a_known_call_kind_resolves_its_configured_default(self) -> None:
        runtime = RuntimeParams()
        assert token_budget_for(runtime, PennyConstants.CHAT_AGENT_NAME) == 5500
        assert token_budget_for(runtime, "collector") == 1500
        assert token_budget_for(runtime, PennyConstants.BROWSE_EXTRACT_AGENT_NAME) == 8000

    def test_an_unrecognised_or_absent_kind_sends_no_cap(self) -> None:
        runtime = RuntimeParams()
        assert token_budget_for(runtime, "some-future-microcontext") is None
        assert token_budget_for(runtime, None) is None


class TestTokenBudgetAndFinishReason:
    """Every call can carry its kind's own cap (#2474): the wire shape the probes
    settled on, and the parsed response saying whether a cap — not the model — is
    what ended the draw."""

    @pytest.mark.asyncio
    async def test_a_token_budget_is_sent_under_both_names_or_not_at_all(self, monkeypatch) -> None:
        """Local Ollama's OpenAI-compatible endpoint silently ignores
        ``max_completion_tokens`` — the modern OpenAI field — and only honours the
        deprecated ``max_tokens`` (measured against gemma4:26b-mlx for #2474); sending
        both is what makes one call portable to both without a backend-specific
        branch.  No caller naming a budget is today's behaviour: no cap at all."""
        sent: dict = {}

        async def capture(**kwargs):
            sent.update(kwargs)
            return _completion("ok")

        client = LlmClient(
            api_url="http://localhost:11434", model="m", max_retries=1, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", capture)

        await client.chat(
            [{"role": "user", "content": "hi"}], call_link=CallLink.new_top(), token_budget=1500
        )
        assert sent["max_tokens"] == 1500
        assert sent["max_completion_tokens"] == 1500

        sent.clear()
        await client.chat([{"role": "user", "content": "hi"}], call_link=CallLink.new_top())
        assert "max_tokens" not in sent
        assert "max_completion_tokens" not in sent
        await client.close()

    @pytest.mark.asyncio
    async def test_thinking_off_rides_the_same_passthrough(self, monkeypatch) -> None:
        """``thinking_enabled=False`` sends the switch the probes found actually
        suppresses reasoning — no production caller passes it yet; the follow-up
        issues (#2475/#2476) are the first to."""
        sent: dict = {}

        async def capture(**kwargs):
            sent.update(kwargs)
            return _completion("ok")

        client = LlmClient(
            api_url="http://localhost:11434", model="m", max_retries=1, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", capture)
        await client.chat(
            [{"role": "user", "content": "hi"}],
            call_link=CallLink.new_top(),
            thinking_enabled=False,
        )

        assert sent["extra_body"] == {
            "reasoning": {"enabled": False},
            "reasoning_effort": "none",
        }
        await client.close()

    @pytest.mark.asyncio
    async def test_a_cut_response_reports_its_own_finish_reason(self, monkeypatch) -> None:
        """A response cut off by a cap decodes with ``finish_reason="length"`` rather
        than reading identically to a finished one — the gap #2468's survey found:
        before this field existed, a capped response decoded exactly like a complete
        one, and nothing downstream could tell the two apart."""
        client = LlmClient(
            api_url="http://localhost:11434", model="m", max_retries=1, retry_delay=0.0
        )

        async def capture(**kwargs):
            return _completion("partial answer", finish_reason="length")

        monkeypatch.setattr(client.client.chat.completions, "create", capture)
        response = await client.chat(
            [{"role": "user", "content": "hi"}], call_link=CallLink.new_top(), token_budget=80
        )

        assert response.finish_reason == "length"
        await client.close()


class TestRetryResilience:
    """A remote endpoint fails in bursts, and a stall is not a failure at all until a
    deadline makes it one — so the two things that keep a run alive are pinned here."""

    @pytest.mark.asyncio
    async def test_a_stalled_request_is_retried_on_a_doubling_backoff(self, monkeypatch) -> None:
        """A timing-out call is re-attempted, waiting twice as long before each try.

        The measured failure this guards: an endpoint that stops answering mid-run. Without
        a deadline the SDK waits out its own 600s read and the retry loop never sees a
        failure to retry — so the request must TIME OUT, and the waits must spread rather
        than re-attempting three times into the same bad moment.
        """
        waits: list[float] = []

        async def record_sleep(seconds: float) -> None:
            waits.append(seconds)

        async def always_time_out(**kwargs):
            raise openai.APITimeoutError(request=httpx.Request("POST", "http://localhost/v1"))

        client = LlmClient(
            api_url="http://localhost:11434", model="m", max_retries=3, retry_delay=1.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", always_time_out)
        monkeypatch.setattr("penny.llm.client.asyncio.sleep", record_sleep)

        with pytest.raises(LlmTimeoutError):
            await client.chat([{"role": "user", "content": "hi"}], call_link=CallLink.new_top())

        # Three attempts, so two waits between them — doubling, not flat.
        assert waits == [1.0, 2.0]

        await client.close()

    @pytest.mark.asyncio
    async def test_a_200_carrying_no_completion_is_retried_then_reported(self, monkeypatch) -> None:
        """A gateway can answer 200 with an error payload where a completion belongs.

        The SDK parses it into a ChatCompletion whose `choices` is None — not an openai
        error, so before this it escaped the retry loop as a TypeError at the first
        subscript, killed the turn, and left the provider's own reason nowhere. Measured
        against OpenRouter, that ended 4 of 10 samples with no reply at all.
        """
        attempts = 0

        class _NoChoices:
            choices = None
            model_extra = {"error": {"message": "upstream provider returned an error"}}

        async def answer_without_a_completion(**kwargs):
            nonlocal attempts
            attempts += 1
            return _NoChoices()

        client = LlmClient(
            api_url="http://localhost:11434", model="m", max_retries=3, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", answer_without_a_completion)

        with pytest.raises(LlmResponseError) as exc_info:
            await client.chat([{"role": "user", "content": "hi"}], call_link=CallLink.new_top())

        assert attempts == 3  # retried, not raised on the first one
        # The provider's own words survive, so the run record says WHY rather than "None".
        assert "no choices" in str(exc_info.value)
        assert "upstream provider returned an error" in str(exc_info.value)

        await client.close()

    @pytest.mark.asyncio
    async def test_a_configured_timeout_reaches_the_underlying_client(self) -> None:
        """The deadline is only real if it lands on the HTTP client that makes the call."""
        client = LlmClient(
            api_url="http://localhost:11434",
            model="m",
            max_retries=1,
            retry_delay=0.0,
            timeout=20.0,
        )
        timeout = client.client.timeout
        assert isinstance(timeout, httpx.Timeout)  # a bare float would not bound the read
        assert timeout.read == 20.0
        assert timeout.connect == PennyConstants.LLM_CONNECT_TIMEOUT_SECONDS
        await client.close()


class TestSummarizeHttpxError:
    """``list_embedding_models`` hits a raw ``/v1/embeddings/models`` endpoint
    (not the SDK), so its error summarization takes an ``httpx.Response`` rather
    than an ``openai`` error — same body-free contract as ``_summarize_llm_error``."""

    def test_json_message_field_is_surfaced(self) -> None:
        response = httpx.Response(404, json={"error": {"message": "model `bar` not found"}})

        assert _summarize_httpx_error(response) == "HTTP 404: model `bar` not found"

    def test_html_body_is_summarized_not_dumped(self) -> None:
        response = httpx.Response(
            404, headers={"content-type": "text/html"}, content=_HTML_ERROR_BODY.encode()
        )

        summary = _summarize_httpx_error(response)

        assert "HTTP 404" in summary
        assert "non-JSON error body" in summary
        assert "text/html" in summary
        assert "<!DOCTYPE" not in summary  # the raw body never leaks


class TestExtractModelIds:
    """The fallback endpoint's payload shape varies by provider — a ``data`` or
    ``models`` envelope, dict items keyed by ``id`` or ``name``, or bare strings."""

    def test_openai_data_id_shape(self) -> None:
        payload = {"data": [{"id": "embeddinggemma"}, {"id": "other"}]}

        assert _extract_model_ids(payload) == ["embeddinggemma", "other"]

    def test_models_envelope_with_name_shape(self) -> None:
        payload = {"models": [{"name": "embeddinggemma"}]}

        assert _extract_model_ids(payload) == ["embeddinggemma"]

    def test_bare_list_of_strings(self) -> None:
        assert _extract_model_ids(["a", "b"]) == ["a", "b"]

    def test_unrecognized_items_are_skipped(self) -> None:
        payload = {"data": ["a", {"id": "b"}, {"name": "c"}, {"foo": "bar"}, 123]}

        assert _extract_model_ids(payload) == ["a", "b", "c"]

    def test_non_list_payload_raises(self) -> None:
        with pytest.raises(LlmResponseError):
            _extract_model_ids({"data": {"not": "a list"}})


def _mock_embeddings_endpoint(monkeypatch, handler) -> None:
    """Route the raw httpx client used by ``list_embedding_models`` through an
    httpx ``MockTransport`` — mock at the HTTP boundary, no live network."""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr("penny.llm.client.httpx.AsyncClient", factory)


class TestListEmbeddingModels:
    @pytest.mark.asyncio
    async def test_parses_ids_from_embeddings_endpoint(self, monkeypatch) -> None:
        """A 200 from /v1/embeddings/models is parsed into model ids."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/embeddings/models"
            return httpx.Response(200, json={"data": [{"id": "embeddinggemma"}]})

        _mock_embeddings_endpoint(monkeypatch, handler)
        client = LlmClient(
            api_url="http://localhost:11434", model="m", max_retries=1, retry_delay=0.0
        )

        assert await client.list_embedding_models() == ["embeddinggemma"]

        await client.close()

    @pytest.mark.asyncio
    async def test_http_error_raises_summarized_response_error(self, monkeypatch) -> None:
        """An HTML 404 from the endpoint raises a summarized LlmResponseError, not the body."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404, headers={"content-type": "text/html"}, content=_HTML_ERROR_BODY.encode()
            )

        _mock_embeddings_endpoint(monkeypatch, handler)
        client = LlmClient(
            api_url="http://localhost:11434", model="m", max_retries=1, retry_delay=0.0
        )

        with pytest.raises(LlmResponseError) as exc_info:
            await client.list_embedding_models()

        message = str(exc_info.value)
        assert "HTTP 404" in message
        assert "<!DOCTYPE" not in message  # summarized, not the raw HTML body
        # The status rides as a value, so a 4xx reads as a verdict without parsing the summary.
        assert exc_info.value.fault is LlmFault.CLIENT_ERROR

        await client.close()


class TestFaultClassIsAValueNotASentence:
    """A failure already produced a message; none of them produced a value (#1996).

    So a run that died 188 times of one cause and one that died once of 188 causes read
    identically, and telling them apart meant grepping logs by hand. The class rides on
    the error, and callers decide by reading it.
    """

    @pytest.mark.parametrize(
        ("status", "fault"),
        [
            (429, LlmFault.RATE_LIMITED),
            (500, LlmFault.SERVER_ERROR),
            (502, LlmFault.SERVER_ERROR),
            (400, LlmFault.CLIENT_ERROR),
            (401, LlmFault.CLIENT_ERROR),
            (None, LlmFault.OTHER),
        ],
    )
    def test_a_status_names_its_class(self, status: int | None, fault: LlmFault) -> None:
        assert fault_for_status(status) == fault

    def test_transience_is_what_separates_a_bad_minute_from_a_verdict(self) -> None:
        """The smoke check reads this to decide whether another draw could help."""
        assert LlmFault.SERVER_ERROR.transient
        assert LlmFault.RATE_LIMITED.transient
        assert LlmFault.NO_CHOICES.transient
        assert LlmFault.TIMEOUT.transient
        assert not LlmFault.NOT_FOUND.transient
        assert not LlmFault.CLIENT_ERROR.transient
        # An unrecognised class stays non-transient on purpose: a gate whose whole job is
        # to be decisive must not become flaky over a case nobody has looked at yet.
        assert not LlmFault.OTHER.transient

    def test_each_error_carries_the_class_it_always_is(self) -> None:
        assert LlmNotFoundError("gone").fault == LlmFault.NOT_FOUND
        assert LlmTimeoutError("slow").fault == LlmFault.TIMEOUT
        assert LlmError("who knows").fault == LlmFault.OTHER

    @pytest.mark.asyncio
    async def test_a_rate_limit_reaches_the_caller_as_a_rate_limit(self, monkeypatch) -> None:
        """The one shape that varies takes its class from the status the client read.

        325 rate limits and 325 empty responses are different runs with different fixes,
        and `LlmResponseError` alone said neither.
        """
        request = httpx.Request("POST", "https://gateway.example/api/v1/chat/completions")
        response = httpx.Response(429, request=request, json={"error": {"message": "slow down"}})

        async def refuse(**kwargs):
            raise openai.RateLimitError("Error code: 429", response=response, body=None)

        client = LlmClient(
            api_url="https://gateway.example/api", model="m", max_retries=1, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", refuse)

        with pytest.raises(LlmResponseError) as exc_info:
            await client.chat([{"role": "user", "content": "hi"}], call_link=CallLink.new_top())

        assert exc_info.value.fault == LlmFault.RATE_LIMITED
        await client.close()

    @pytest.mark.asyncio
    async def test_an_empty_completion_reaches_the_caller_as_no_choices(self, monkeypatch) -> None:
        """The class that killed 34 of 48 samples, now nameable and countable."""

        class _NoChoices:
            choices = None
            model_extra = {"provider": "some-upstream"}

        async def answer_without_a_completion(**kwargs):
            return _NoChoices()

        client = LlmClient(
            api_url="https://gateway.example/api", model="m", max_retries=1, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", answer_without_a_completion)

        with pytest.raises(LlmResponseError) as exc_info:
            await client.chat([{"role": "user", "content": "hi"}], call_link=CallLink.new_top())

        assert exc_info.value.fault == LlmFault.NO_CHOICES
        await client.close()


class TestAConfiguredProviderIsPinned:
    """A configured provider is a hard pin: one upstream, no fallback routing (#2143)."""

    def test_a_pin_travels_with_fallbacks_off(self) -> None:
        """Upstreams serving one model are not interchangeable, so nothing routes at random.

        Measured: one gateway adapter replaced every plain-text tool result with `{}`, so a
        call routed there measured a model that never saw its own tool results.
        """
        pin = ProviderPin.pin("Cloudflare")
        assert pin is not None
        assert pin.as_request_field() == {
            "order": ["Cloudflare"],
            "allow_fallbacks": False,
        }

    def test_no_configured_provider_is_no_pin_at_all(self) -> None:
        """A direct endpoint has no upstreams, so nothing is sent — not an empty pin."""
        assert ProviderPin.pin(None) is None
        assert ProviderPin.pin("") is None

    @pytest.mark.asyncio
    async def test_the_pin_reaches_the_request_body(self, monkeypatch) -> None:
        """A pin nothing sends is a pin that does nothing."""
        sent: dict = {}

        async def capture(**kwargs):
            sent.update(kwargs)
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1"))

        client = LlmClient(
            api_url="https://gateway.example/api",
            model="m",
            max_retries=1,
            retry_delay=0.0,
            provider_pin=ProviderPin.pin("Cloudflare"),
        )
        monkeypatch.setattr(client.client.chat.completions, "create", capture)

        with pytest.raises(LlmError):
            await client.chat([{"role": "user", "content": "hi"}], call_link=CallLink.new_top())

        # The reasoning switch still rides the same passthrough — one is not added at the
        # cost of the other, and a run that lost it would be comparing a model to itself.
        assert sent["extra_body"] == {
            "reasoning": {"enabled": True},
            "provider": {"order": ["Cloudflare"], "allow_fallbacks": False},
        }
        await client.close()

    @pytest.mark.asyncio
    async def test_a_client_with_no_pin_sends_the_body_it_always_sent(self, monkeypatch) -> None:
        sent: dict = {}

        async def capture(**kwargs):
            sent.update(kwargs)
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1"))

        client = LlmClient(
            api_url="http://localhost:11434", model="m", max_retries=1, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", capture)

        with pytest.raises(LlmError):
            await client.chat([{"role": "user", "content": "hi"}], call_link=CallLink.new_top())

        assert sent["extra_body"] == {"reasoning": {"enabled": True}}
        await client.close()

    @pytest.mark.asyncio
    async def test_the_upstream_that_answered_comes_back_on_the_response(self, monkeypatch) -> None:
        """The answer says who served it, so a pin the gateway did not honour is visible."""

        class _Answered:
            model = "m"
            model_extra = {"provider": "DeepInfra"}
            choices = [
                type(
                    "Choice",
                    (),
                    {
                        "message": type(
                            "Message",
                            (),
                            {
                                "role": "assistant",
                                "content": "hi",
                                "tool_calls": None,
                                "model_extra": {},
                            },
                        )(),
                        "finish_reason": "stop",
                    },
                )()
            ]

        async def answer(**kwargs):
            return _Answered()

        client = LlmClient(
            api_url="https://gateway.example/api",
            model="m",
            max_retries=1,
            retry_delay=0.0,
            provider_pin=ProviderPin.pin("Cloudflare"),
        )
        monkeypatch.setattr(client.client.chat.completions, "create", answer)

        response = await client.chat(
            [{"role": "user", "content": "hi"}], call_link=CallLink.new_top()
        )

        # It pinned Cloudflare and DeepInfra answered — stated rather than hidden behind an
        # assumption that the pin held.
        assert response.provider == "DeepInfra"
        assert isinstance(response, LlmResponse)
        assert isinstance(response.message, LlmMessage)
        await client.close()


def _tool_call_completion(*tool_calls: tuple[str, str, str]) -> ChatCompletion:
    """A real SDK completion whose one message carries ``(id, name, arguments)`` calls."""
    calls = [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        for call_id, name, arguments in tool_calls
    ]
    message = {"role": "assistant", "content": None, "tool_calls": calls}
    return ChatCompletion.model_validate(
        {
            "id": "completion-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
        }
    )


class TestParallelCallsReturnedAsOneCall:
    """A provider can return parallel calls to one tool as ONE call whose arguments are
    several JSON objects back to back. Read as one value that string is not JSON at all:
    measured on one whole-suite run, 31 of 3,511 calls arrived this way, and each either
    reached its tool with no arguments or was salvaged into a write pairing one object's
    key with the next object's content."""

    _LIKE = '{"collection": "things-im-into", "entries": [{"key": "bouldering", "content": "x"}]}'
    _DISLIKE = '{"collection": "things-i-avoid", "entries": [{"key": "coffee", "content": "y"}]}'

    async def _chat(self, monkeypatch, db: Database, completion: ChatCompletion):
        async def answer(**kwargs):
            return completion

        client = LlmClient(
            api_url="http://localhost:11434", model="m", db=db, max_retries=1, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", answer)
        response = await client.chat(
            [{"role": "user", "content": "hi"}], call_link=CallLink.top("run-1")
        )
        await client.close()
        return response

    @staticmethod
    def _logged_calls(db: Database) -> list[dict]:
        with db.get_session() as session:
            row = session.exec(select(PromptLog)).one()
        return json.loads(row.response)["choices"][0]["message"]["tool_calls"]

    @pytest.mark.asyncio
    async def test_each_object_is_dispatched_and_recorded_as_its_own_call(
        self, monkeypatch, db: Database
    ) -> None:
        completion = _tool_call_completion(
            ("call_a", "collection_write", f"{self._LIKE} {self._DISLIKE}")
        )

        response = await self._chat(monkeypatch, db, completion)

        calls = response.message.tool_calls or []
        assert [call.id for call in calls] == ["call_a", "call_a_2"]
        assert [call.function.name for call in calls] == ["collection_write"] * 2
        # Each call carries its OWN object whole — no key from one beside the other's content.
        assert [call.function.arguments for call in calls] == [
            json.loads(self._LIKE),
            json.loads(self._DISLIKE),
        ]
        # The ledger records the calls that ran, under the ids their results will answer.
        logged = self._logged_calls(db)
        assert [call["id"] for call in logged] == ["call_a", "call_a_2"]
        assert [json.loads(call["function"]["arguments"]) for call in logged] == [
            json.loads(self._LIKE),
            json.loads(self._DISLIKE),
        ]

    @pytest.mark.asyncio
    async def test_a_malformed_tail_is_its_own_call_and_borrows_nothing(
        self, monkeypatch, db: Database
    ) -> None:
        """The salvage regex reads the tail's own text: it has a key and no content of its
        own, so it recovers nothing rather than taking the content of the call before it."""
        tail = '{"collection": "things-i-avoid", "entries": [{"key": "coffee"'
        completion = _tool_call_completion(("call_a", "collection_write", f"{self._LIKE}{tail}"))

        response = await self._chat(monkeypatch, db, completion)

        calls = response.message.tool_calls or []
        assert [call.function.arguments for call in calls] == [json.loads(self._LIKE), {}]
        assert [call["function"]["arguments"] for call in self._logged_calls(db)] == [
            self._LIKE,
            tail,
        ]

    @pytest.mark.asyncio
    async def test_a_call_carrying_one_object_is_untouched(self, monkeypatch, db: Database) -> None:
        completion = _tool_call_completion(
            ("call_a", "collection_write", self._LIKE), ("call_b", "collection_list", "")
        )

        response = await self._chat(monkeypatch, db, completion)

        calls = response.message.tool_calls or []
        assert [(call.id, call.function.arguments) for call in calls] == [
            ("call_a", json.loads(self._LIKE)),
            ("call_b", {}),
        ]
        assert (
            self._logged_calls(db) == completion.model_dump()["choices"][0]["message"]["tool_calls"]
        )


class TestCallLoggedHoweverItEnds:
    """A model call writes ONE ``promptlog`` row at its END, however it ends (#2355
    part A) — completed, cancelled, timed out, or raised — never only on success.
    Before this, a call cancelled by a tool's own timeout, or one that exhausted its
    retries, left no row at all: no input, no duration, no agent name."""

    @staticmethod
    def _one_row(db: Database) -> PromptLog:
        with db.get_session() as session:
            return session.exec(select(PromptLog)).one()

    @pytest.mark.asyncio
    async def test_a_completed_call_is_logged_completed(self, monkeypatch, db: Database) -> None:
        """Unchanged outcome, named explicitly: a successful call's row carries the
        real response and ``call_outcome == "completed"``, with no error fields."""

        async def answer(**kwargs):
            return _completion("ok")

        client = LlmClient(
            api_url="http://localhost:11434", model="m", db=db, max_retries=1, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", answer)

        await client.chat(
            [{"role": "user", "content": "hi"}], agent_name="chat", call_link=CallLink.top("run-ok")
        )
        await client.close()

        row = self._one_row(db)
        assert row.call_outcome == "completed"
        assert row.call_error_class is None
        assert row.call_error_message is None
        assert json.loads(row.response)["choices"][0]["message"]["content"] == "ok"

    @pytest.mark.asyncio
    async def test_a_cancelled_call_leaves_one_row_and_still_propagates(
        self, monkeypatch, db: Database
    ) -> None:
        """A call cancelled from outside (a tool's own timeout, in production) leaves
        ONE row — input, agent name, run id, a duration — marked cancelled, and the
        ``CancelledError`` itself still propagates untouched (a canceller downstream,
        like ``asyncio.wait_for``, must still see it)."""

        async def hang_then_cancel(**kwargs):
            raise asyncio.CancelledError()

        client = LlmClient(
            api_url="http://localhost:11434", model="m", db=db, max_retries=3, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", hang_then_cancel)

        with pytest.raises(asyncio.CancelledError):
            await client.chat(
                [{"role": "user", "content": "hi"}],
                agent_name="browse-extract",
                call_link=CallLink.top("run-cancelled"),
            )
        await client.close()

        row = self._one_row(db)
        assert row.call_outcome == "cancelled"
        assert row.agent_name == "browse-extract"
        assert row.run_id == "run-cancelled"
        assert row.duration_ms is not None
        assert row.response == "{}"  # never NULL — see migration 0114's own note
        assert json.loads(row.messages)[-1]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_a_raised_call_leaves_one_row_with_its_error(
        self, monkeypatch, db: Database
    ) -> None:
        """A call that exhausts its retries on a connection error leaves one row
        carrying the exception's own class and message — never silently lost."""

        async def always_refuse(**kwargs):
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1"))

        client = LlmClient(
            api_url="http://localhost:11434", model="m", db=db, max_retries=2, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", always_refuse)

        with pytest.raises(LlmConnectionError):
            await client.chat(
                [{"role": "user", "content": "hi"}], call_link=CallLink.top("run-raised")
            )
        await client.close()

        row = self._one_row(db)
        assert row.call_outcome == "raised"
        assert row.call_error_class == "LlmConnectionError"
        assert row.call_error_message
        assert row.response == "{}"

    @pytest.mark.asyncio
    async def test_a_timed_out_call_is_logged_timed_out(self, monkeypatch, db: Database) -> None:
        """Exhausting retries on the endpoint's OWN timeout is distinguished from an
        external cancellation and from any other raised error."""

        async def always_time_out(**kwargs):
            raise openai.APITimeoutError(request=httpx.Request("POST", "http://x/v1"))

        client = LlmClient(
            api_url="http://localhost:11434", model="m", db=db, max_retries=2, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", always_time_out)

        with pytest.raises(LlmTimeoutError):
            await client.chat(
                [{"role": "user", "content": "hi"}], call_link=CallLink.top("run-timeout")
            )
        await client.close()

        row = self._one_row(db)
        assert row.call_outcome == "timed_out"
        assert row.call_error_class == "LlmTimeoutError"

    @pytest.mark.asyncio
    async def test_a_retried_then_successful_call_logs_only_one_row(
        self, monkeypatch, db: Database
    ) -> None:
        """One row per CALL, not per network attempt (the proposal this issue
        confirmed): a call retried once before succeeding still writes exactly one
        ``promptlog`` row, marked completed — never a row per attempt, which would
        read as re-rolls to anything that compares inputs (the eval harness's own
        ``draw_rerolled``)."""
        attempts = 0

        async def fail_once_then_answer(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1"))
            return _completion("ok")

        client = LlmClient(
            api_url="http://localhost:11434", model="m", db=db, max_retries=3, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", fail_once_then_answer)

        await client.chat(
            [{"role": "user", "content": "hi"}], call_link=CallLink.top("run-retried")
        )
        await client.close()

        assert attempts == 2  # it DID retry
        row = self._one_row(db)  # but logged exactly once
        assert row.call_outcome == "completed"

    @pytest.mark.asyncio
    async def test_duration_counts_the_whole_call_not_just_the_last_attempt(
        self, monkeypatch, db: Database
    ) -> None:
        """``duration_ms`` is the call's TOTAL wall time across every retried attempt
        — the honest cost the eval harness's token-cost accounting reads — not just
        the attempt that happened to succeed."""
        attempts = 0

        async def slow_then_fast(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                await asyncio.sleep(0.05)
                raise openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1"))
            return _completion("ok")

        client = LlmClient(
            api_url="http://localhost:11434", model="m", db=db, max_retries=3, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", slow_then_fast)

        await client.chat([{"role": "user", "content": "hi"}], call_link=CallLink.top("run-slow"))
        await client.close()

        row = self._one_row(db)
        assert row.duration_ms is not None
        assert row.duration_ms >= 50  # the first (slow, failed) attempt's own time counts


class TestMicroContextDrawLinksToItsParentCall:
    """A draw ``LlmClient.chat`` makes on behalf of a micro-context links back to
    its ``promptlog`` row's ``parent_run_id`` / ``parent_tool_call_id`` purely from
    the EXPLICIT ``call_link`` it is handed (#2458, the code owner's ruling against
    ambient state) — never a parameter ``chat`` derives for itself.  The full
    tool-nested path is covered in ``tests/tools/test_browse_micro_context.py``;
    this pins the plain explicit-link read at the client itself."""

    @pytest.mark.asyncio
    async def test_the_parent_ids_land_on_the_logged_row(self, monkeypatch, db: Database) -> None:
        async def answer(**kwargs):
            return _completion("ok")

        client = LlmClient(
            api_url="http://localhost:11434", model="m", db=db, max_retries=1, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", answer)

        # Two levels, exactly as production nests them: the enclosing chat/collector
        # run ("chat-run-1") is handed to the tool dispatcher, which derives the
        # dispatched call's own link (``for_tool_call``), and a micro-context's OWN
        # draw derives ITS OWN link from that (``child``) — capturing "chat-run-1"
        # as its PARENT and "call-browse-1" as its parent tool call in the process.
        dispatched = CallLink.top("chat-run-1").for_tool_call("call-browse-1")
        draw_link = dispatched.child()
        await client.chat([{"role": "user", "content": "hi"}], call_link=draw_link)
        await client.close()

        with db.get_session() as session:
            row = session.exec(select(PromptLog)).one()
        assert row.parent_run_id == "chat-run-1"
        assert row.parent_tool_call_id == "call-browse-1"

    @pytest.mark.asyncio
    async def test_a_top_level_call_links_to_nothing(self, monkeypatch, db: Database) -> None:
        """A run starter's own top link (``CallLink.new_top()``) lands a FETCHABLE
        ``run_id`` with no parent — the shape every real top-level caller (a chat
        turn, a collector cycle) produces."""

        async def answer(**kwargs):
            return _completion("ok")

        client = LlmClient(
            api_url="http://localhost:11434", model="m", db=db, max_retries=1, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", answer)

        await client.chat([{"role": "user", "content": "hi"}], call_link=CallLink.new_top())
        await client.close()

        with db.get_session() as session:
            row = session.exec(select(PromptLog)).one()
        assert row.run_id is not None
        assert row.parent_run_id is None
