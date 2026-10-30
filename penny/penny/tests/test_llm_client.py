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
from types import SimpleNamespace

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion
from sqlmodel import col, select

from penny.config_params import RuntimeParams, token_budget_for
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.models import PromptLog
from penny.llm.call_link import CallLink
from penny.llm.client import (
    LlmClient,
    _extract_model_ids,
    _image_data_uri,
    _summarize_httpx_error,
    _summarize_llm_error,
)
from penny.llm.models import (
    CallOutcome,
    LlmConnectionError,
    LlmError,
    LlmFault,
    LlmMessage,
    LlmNotFoundError,
    LlmResponse,
    LlmResponseError,
    LlmTimeoutError,
    LlmToolCall,
    LlmToolCallFunction,
    LlmToolParseError,
    PromptTransport,
    ProviderPin,
    RawFaithfulnessOutcome,
    fault_for_status,
    usage_timing_from_response,
)
from penny.llm.thinking_loop import RepeatStreamTracker, ThoughtStreamTracker

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
            await client.chat([LlmMessage(role="user", content="hi")], call_link=CallLink.new_top())

        message = str(exc_info.value)
        assert "HTTP 404" in message
        assert "<!DOCTYPE" not in message  # summarized, not the raw HTML body

        await client.close()


def _completion(content: str, finish_reason: str = "stop", usage: dict | None = None):
    """A minimal ChatCompletion-shaped response — what the SDK hands ``chat`` back.

    ``usage``, when given, lands verbatim in ``model_dump()``'s ``"usage"`` key —
    letting a test hand ``_log_call`` a runner-shaped ``usage`` object exactly as
    ``raw.model_dump()`` would (the SDK's own ``CompletionUsage``/``PromptTokensDetails``
    allow extra fields, so oMLX's non-standard duration keys survive it unchanged).
    Omitted (the default, every existing caller), the dumped dict carries no
    ``"usage"`` key at all — unchanged from before this parameter existed.
    """

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

        def __init__(self, text: str, reason: str, usage: dict | None) -> None:
            self.choices = [_Choice(text, reason)]
            self.usage = usage

        def model_dump(self) -> dict:
            dumped: dict = {"choices": [{"message": {"content": content}}]}
            if usage is not None:
                dumped["usage"] = usage
            return dumped

    return _Completion(content, finish_reason, usage)


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
        await client.chat([LlmMessage(role="user", content="hi")], call_link=CallLink.new_top())

        assert sent["extra_body"] == {
            "reasoning": {"enabled": True},
            "chat_template_kwargs": {"enable_thinking": True},
        }
        await client.close()


class TestTokenBudgetResolution:
    """``token_budget_for`` (#2474) maps a call kind's ``agent_name`` onto its own
    configured cap through the same ``RuntimeParams`` lookup chain every other
    runtime param already uses — a DB/env override takes effect the way it does
    for ``MAX_STEPS``, and an unrecognised kind sends no cap rather than a
    guessed number."""

    def test_a_known_call_kind_resolves_its_configured_default(self) -> None:
        runtime = RuntimeParams()
        assert token_budget_for(runtime, PennyConstants.CHAT_AGENT_NAME) == 16000
        assert token_budget_for(runtime, "collector") == 16000
        assert token_budget_for(runtime, PennyConstants.BROWSE_EXTRACT_AGENT_NAME) == 16000

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
            [LlmMessage(role="user", content="hi")], call_link=CallLink.new_top(), token_budget=1500
        )
        assert sent["max_tokens"] == 1500
        assert sent["max_completion_tokens"] == 1500

        sent.clear()
        await client.chat([LlmMessage(role="user", content="hi")], call_link=CallLink.new_top())
        assert "max_tokens" not in sent
        assert "max_completion_tokens" not in sent
        await client.close()

    @pytest.mark.asyncio
    async def test_thinking_off_rides_the_same_passthrough(self, monkeypatch) -> None:
        """``thinking_enabled=False`` sends ONLY ``reasoning_effort: "none"`` — no
        ``reasoning`` key at all (#2607): on local Ollama a ``reasoning`` object's
        PRESENCE keeps thinking on regardless of its ``enabled`` value, so the
        always-on ``reasoning: {enabled: True}`` must never ride alongside the off
        switch. An ordinary call is unchanged: the always-on key, no
        ``reasoning_effort``. ``chat_template_kwargs.enable_thinking`` (#2761) rides
        both bodies as the field oMLX reads, flipped true/false with the switch."""
        sent: dict = {}

        async def capture(**kwargs):
            sent.update(kwargs)
            return _completion("ok")

        client = LlmClient(
            api_url="http://localhost:11434", model="m", max_retries=1, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", capture)
        await client.chat(
            [LlmMessage(role="user", content="hi")],
            call_link=CallLink.new_top(),
            thinking_enabled=False,
        )

        assert sent["extra_body"] == {
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        assert "reasoning" not in sent["extra_body"]

        sent.clear()
        await client.chat(
            [LlmMessage(role="user", content="hi")],
            call_link=CallLink.new_top(),
        )

        assert sent["extra_body"] == {
            "reasoning": {"enabled": True},
            "chat_template_kwargs": {"enable_thinking": True},
        }
        assert "reasoning_effort" not in sent["extra_body"]
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
            [LlmMessage(role="user", content="hi")], call_link=CallLink.new_top(), token_budget=80
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
            await client.chat([LlmMessage(role="user", content="hi")], call_link=CallLink.new_top())

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
            await client.chat([LlmMessage(role="user", content="hi")], call_link=CallLink.new_top())

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
            await client.chat([LlmMessage(role="user", content="hi")], call_link=CallLink.new_top())

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
            await client.chat([LlmMessage(role="user", content="hi")], call_link=CallLink.new_top())

        assert exc_info.value.fault == LlmFault.NO_CHOICES
        await client.close()


class TestUsageTimingMapping:
    """``usage_timing_from_response`` (#2793) — the one mapping every runner's own
    ``usage`` shape goes through, at write time (``LlmClient._log_call``) and in
    migration 0129's backfill alike. A field the serving runner didn't report is
    ``None``, never a guessed zero — these cases are what prove that, per shape."""

    def test_omlx_shape_seconds_converted_to_ms(self) -> None:
        """oMLX puts everything in ``usage``, durations in SECONDS."""
        timing = usage_timing_from_response(
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "prompt_tokens_details": {"cached_tokens": 20},
                    "prompt_eval_duration": 0.5,
                    "generation_duration": 1.2,
                    "time_to_first_token": 0.1,
                    "total_time": 1.7,
                }
            }
        )
        assert timing.prompt_tokens == 100
        assert timing.completion_tokens == 50
        assert timing.cached_tokens == 20
        assert timing.prefill_ms == 500
        assert timing.generation_ms == 1200

    def test_ollama_native_shape_nanoseconds_converted_to_ms(self) -> None:
        """Ollama's native API names the same two phases differently, in NANOSECONDS —
        told apart from oMLX's shape by ``eval_duration`` being present where oMLX's
        ``generation_duration`` is not, never by a runner name passed in."""
        timing = usage_timing_from_response(
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "prompt_eval_duration": 500_000_000,
                    "eval_duration": 1_200_000_000,
                }
            }
        )
        assert timing.prompt_tokens == 100
        assert timing.completion_tokens == 50
        assert timing.cached_tokens is None  # Ollama native reports no cache breakdown here
        assert timing.prefill_ms == 500
        assert timing.generation_ms == 1200

    def test_openrouter_shape_has_tokens_but_no_duration_split(self) -> None:
        """OpenRouter reports cached tokens but no prefill/generation split at all —
        absence of BOTH duration keys, not a zero-valued one."""
        timing = usage_timing_from_response(
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "prompt_tokens_details": {"cached_tokens": 20},
                }
            }
        )
        assert timing.prompt_tokens == 100
        assert timing.completion_tokens == 50
        assert timing.cached_tokens == 20
        assert timing.prefill_ms is None
        assert timing.generation_ms is None

    def test_no_usage_at_all_maps_every_field_to_none(self) -> None:
        """A response with no ``usage`` object (a cancelled/timed-out/raised call's
        literal ``{}``, or a runner that reports none) is every field NULL."""
        timing = usage_timing_from_response({})
        assert timing.prompt_tokens is None
        assert timing.cached_tokens is None
        assert timing.completion_tokens is None
        assert timing.prefill_ms is None
        assert timing.generation_ms is None

    def test_a_present_usage_missing_one_field_maps_that_field_to_none(self) -> None:
        """A field the runner's ``usage`` simply omits is NULL, not a guessed 0 —
        unlike the legacy ``_extract_token_usage`` input/output tok counts, which
        default to 0 for the addon's older tok/s render."""
        timing = usage_timing_from_response({"usage": {"completion_tokens": 50}})
        assert timing.prompt_tokens is None
        assert timing.completion_tokens == 50


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
            await client.chat([LlmMessage(role="user", content="hi")], call_link=CallLink.new_top())

        # The reasoning switch still rides the same passthrough — one is not added at the
        # cost of the other, and a run that lost it would be comparing a model to itself.
        assert sent["extra_body"] == {
            "reasoning": {"enabled": True},
            "chat_template_kwargs": {"enable_thinking": True},
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
            await client.chat([LlmMessage(role="user", content="hi")], call_link=CallLink.new_top())

        assert sent["extra_body"] == {
            "reasoning": {"enabled": True},
            "chat_template_kwargs": {"enable_thinking": True},
        }
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
            [LlmMessage(role="user", content="hi")], call_link=CallLink.new_top()
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
            [LlmMessage(role="user", content="hi")], call_link=CallLink.top("run-1")
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
            [LlmMessage(role="user", content="hi")],
            agent_name="chat",
            call_link=CallLink.top("run-ok"),
        )
        await client.close()

        row = self._one_row(db)
        assert row.call_outcome == "completed"
        assert row.call_error_class is None
        assert row.call_error_message is None
        assert json.loads(row.response)["choices"][0]["message"]["content"] == "ok"

    @pytest.mark.asyncio
    async def test_a_completed_calls_usage_timing_is_persisted_on_the_row(
        self, monkeypatch, db: Database
    ) -> None:
        """``_log_call`` computes ``usage_timing_from_response`` off the real dumped
        completion and the five columns land on the row (#2793) — the one production
        chokepoint every call, including a reroll, writes through."""

        async def answer(**kwargs):
            return _completion(
                "ok",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "prompt_tokens_details": {"cached_tokens": 20},
                    "prompt_eval_duration": 0.5,
                    "generation_duration": 1.2,
                },
            )

        client = LlmClient(
            api_url="http://localhost:11434", model="m", db=db, max_retries=1, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", answer)

        await client.chat(
            [LlmMessage(role="user", content="hi")],
            agent_name="chat",
            call_link=CallLink.top("run-usage"),
        )
        await client.close()

        row = self._one_row(db)
        assert row.prompt_tokens == 100
        assert row.completion_tokens == 50
        assert row.cached_tokens == 20
        assert row.prefill_ms == 500
        assert row.generation_ms == 1200

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
                [LlmMessage(role="user", content="hi")],
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
        # No response ever landed, so there's no usage to map — every field NULL,
        # never a guessed zero (#2793).
        assert row.prompt_tokens is None
        assert row.prefill_ms is None

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
                [LlmMessage(role="user", content="hi")], call_link=CallLink.top("run-raised")
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
                [LlmMessage(role="user", content="hi")], call_link=CallLink.top("run-timeout")
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
            [LlmMessage(role="user", content="hi")], call_link=CallLink.top("run-retried")
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

        await client.chat(
            [LlmMessage(role="user", content="hi")], call_link=CallLink.top("run-slow")
        )
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
        await client.chat([LlmMessage(role="user", content="hi")], call_link=draw_link)
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

        await client.chat([LlmMessage(role="user", content="hi")], call_link=CallLink.new_top())
        await client.close()

        with db.get_session() as session:
            row = session.exec(select(PromptLog)).one()
        assert row.run_id is not None
        assert row.parent_run_id is None


# ── Frozen request payload (#2563) ──────────────────────────────────────────
#
# Conversation history is moving from ``list[dict]`` to ``list[LlmMessage]`` through
# the agent loop (#2563/#2603/#2604), with the HARD constraint that the request
# payload sent to the model stays byte-identical.  These literals are hand-written —
# not derived by calling ``_translate_messages``/``_build_chat_kwargs`` themselves —
# so a regression in either of those functions is still caught; copying their own
# output into the "expected" side would make the comparison tautological.
#
# One message of every shape in use today: system, user, an assistant tool call
# carrying a thinking passback, its tool result (with the internal
# ``tool_success`` stamp), the injected page-context call (``ChatAgent.
# _inject_page_context``), and a vision user message carrying ``images``.

_FROZEN_SYSTEM = {"role": "system", "content": "You are Penny."}
_FROZEN_USER = {"role": "user", "content": "what should I pack for a trip to Lisbon?"}
_FROZEN_ASSISTANT_TOOL_CALL = {
    "role": "assistant",
    "content": "",
    "reasoning": "checking the forecast first",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "browse",
                "arguments": '{"queries": ["Lisbon weather forecast"]}',
            },
        }
    ],
}
_FROZEN_TOOL_RESULT = {
    "role": "tool",
    "content": "(browse result)\nSunny and mild.",
    "tool_call_id": "call_1",
    PennyConstants.TOOL_RESULT_SUCCESS_KEY: True,
}
# The page-context shape (chat.py's ``_inject_page_context``): arguments is a JSON
# string like every other tool call (#2604; a bare dict before that stage), and its
# tool result carries no ``tool_success`` key.
_FROZEN_PAGE_CONTEXT_CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "page-context-call",
            "type": "function",
            "function": {
                "name": "browse",
                "arguments": '{"queries": ["https://example.com/packing-guide"]}',
            },
        }
    ],
}
_FROZEN_PAGE_CONTEXT_RESULT = {
    "role": "tool",
    "content": "(browse result)\nPacking guide body text.",
    "tool_call_id": "page-context-call",
}
_FROZEN_VISION_USER = {
    "role": "user",
    "content": "what's in this photo?",
    "images": ["c3VpdGNhc2U="],
}

# Today's internal (pre-wire) dict shape, exactly as agent code builds it — what
# ``LlmClient.chat()`` receives and what the prompt log stores verbatim.
FROZEN_INTERNAL_MESSAGES = [
    _FROZEN_SYSTEM,
    _FROZEN_USER,
    _FROZEN_ASSISTANT_TOOL_CALL,
    _FROZEN_TOOL_RESULT,
    _FROZEN_PAGE_CONTEXT_CALL,
    _FROZEN_PAGE_CONTEXT_RESULT,
    _FROZEN_VISION_USER,
]

# The wire shape ``_translate_messages`` produces today: ``tool_success`` stripped,
# ``images`` folded into OpenAI content-parts with each image's real mime type
# (bare base64 with no magic prefix defaults to ``image/jpeg``) — everything else
# untouched.
FROZEN_WIRE_MESSAGES = [
    _FROZEN_SYSTEM,
    _FROZEN_USER,
    _FROZEN_ASSISTANT_TOOL_CALL,
    {"role": "tool", "content": "(browse result)\nSunny and mild.", "tool_call_id": "call_1"},
    _FROZEN_PAGE_CONTEXT_CALL,
    _FROZEN_PAGE_CONTEXT_RESULT,
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "what's in this photo?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,c3VpdGNhc2U="}},
        ],
    },
]

FROZEN_KWARGS = {
    "model": "frozen-test-model",
    "messages": FROZEN_WIRE_MESSAGES,
    "extra_body": {
        "reasoning": {"enabled": True},
        "chat_template_kwargs": {"enable_thinking": True},
    },
}


class TestFrozenRequestPayloadToday:
    """Pins today's wire boundary (``_translate_messages`` + ``_build_chat_kwargs``),
    independent of the #2563 typed-message refactor — this class references no
    ``LlmMessage`` field the refactor adds, so it is unchanged by any stage of it.
    A future change to either function that silently alters the wire payload for
    any of these six shapes fails here first."""

    def test_translate_messages_matches_the_frozen_wire_shape(self) -> None:
        assert LlmClient._translate_messages(FROZEN_INTERNAL_MESSAGES) == FROZEN_WIRE_MESSAGES

    def test_build_chat_kwargs_matches_the_frozen_request(self) -> None:
        client = LlmClient(
            api_url="http://localhost:11434",
            model="frozen-test-model",
            max_retries=1,
            retry_delay=0.0,
        )
        translated = LlmClient._translate_messages(FROZEN_INTERNAL_MESSAGES)
        kwargs = client._build_chat_kwargs(
            translated, tools=None, format=None, token_budget=None, thinking_enabled=True
        )
        assert kwargs == FROZEN_KWARGS


# The same six shapes, built the #2563 way: a plain ``LlmMessage(...)`` for every
# shape a direct caller constructs fresh (system/user/an assistant tool call with a
# thinking passback/its tool result/a vision user turn), and ``LlmMessage.from_dict``
# for the page-context pair — the shape a caller receives off a logged row rather
# than constructs (its builder builds it typed since #2604), so the round trip
# through ``from_dict``/``to_log_dict`` carries it through this boundary unchanged.
TYPED_FROZEN_MESSAGES = [
    LlmMessage(role="system", content=_FROZEN_SYSTEM["content"]),
    LlmMessage(role="user", content=_FROZEN_USER["content"]),
    LlmMessage(
        role="assistant",
        content="",
        thinking="checking the forecast first",
        tool_calls=[
            LlmToolCall(
                id="call_1",
                function=LlmToolCallFunction(
                    name="browse", arguments={"queries": ["Lisbon weather forecast"]}
                ),
            )
        ],
    ),
    LlmMessage(
        role="tool",
        content="(browse result)\nSunny and mild.",
        tool_call_id="call_1",
        tool_success=True,
    ),
    LlmMessage.from_dict(_FROZEN_PAGE_CONTEXT_CALL),
    LlmMessage.from_dict(_FROZEN_PAGE_CONTEXT_RESULT),
    LlmMessage(role="user", content="what's in this photo?", images=["c3VpdGNhc2U="]),
]


class TestTypedMessagesMatchTheFrozenPayload:
    """``LlmMessage`` end to end (#2563): built the typed way, the exact same request
    reaches the model — the HARD constraint this stage must hold."""

    def test_to_log_dict_reproduces_todays_internal_shape(self) -> None:
        """Every typed message's ``to_log_dict()`` is byte-identical to the dict
        literal the agent loop builds for that same shape today — including the
        page-context pair, round-tripped through ``from_dict`` rather than built
        fresh, its ``arguments`` a JSON string like every other call's (#2604)."""
        assert [m.to_log_dict() for m in TYPED_FROZEN_MESSAGES] == FROZEN_INTERNAL_MESSAGES

    @pytest.mark.asyncio
    async def test_chat_sends_the_frozen_wire_payload(self, monkeypatch) -> None:
        """``LlmClient.chat(messages: list[LlmMessage])`` — the live entry point —
        puts exactly the frozen wire messages on the request the SDK receives."""
        sent = {}

        async def capture(**kwargs):
            sent.update(kwargs)
            return _completion("ok")

        client = LlmClient(
            api_url="http://localhost:11434",
            model="frozen-test-model",
            max_retries=1,
            retry_delay=0.0,
        )
        monkeypatch.setattr(client.client.chat.completions, "create", capture)

        await client.chat(TYPED_FROZEN_MESSAGES, call_link=CallLink.new_top())

        assert sent["messages"] == FROZEN_WIRE_MESSAGES
        assert sent["extra_body"] == {
            "reasoning": {"enabled": True},
            "chat_template_kwargs": {"enable_thinking": True},
        }
        await client.close()


class TestImageDataUri:
    """``_image_data_uri`` (#2635): a data URI passes through; bare base64 gets its
    mime type read off the image's magic bytes, defaulting to JPEG."""

    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            ("/9j/4AAQSkZJRg==", "data:image/jpeg;base64,/9j/4AAQSkZJRg=="),
            ("iVBORw0KGgo=", "data:image/png;base64,iVBORw0KGgo="),
            ("R0lGODlhAQ==", "data:image/gif;base64,R0lGODlhAQ=="),
            ("UklGRh4A", "data:image/webp;base64,UklGRh4A"),
            # No magic prefix — assumed JPEG, the most common capture format
            ("c3VpdGNhc2U=", "data:image/jpeg;base64,c3VpdGNhc2U="),
            # A URI the channel already labelled passes through untouched
            ("data:image/webp;base64,UklGRh4A", "data:image/webp;base64,UklGRh4A"),
        ],
    )
    def test_mime_type_comes_from_the_image_itself(self, image: str, expected: str) -> None:
        assert _image_data_uri(image) == expected


# A model id with a vendored chat template (#2829) — matches both oMLX's own id and
# OpenRouter's, via ``raw_template.has_vendored_template``'s substring match.
_GEMMA_MODEL = "google/gemma-4-26b-a4b-it"


class _RawChoice:
    def __init__(self, text: str, reason: str | None) -> None:
        self.text = text
        self.finish_reason = reason

    def model_copy(self, update: dict | None = None):
        update = update or {}
        return _RawChoice(
            update.get("text", self.text), update.get("finish_reason", self.finish_reason)
        )

    def _dump(self) -> dict:
        return {"text": self.text, "finish_reason": self.finish_reason}


class _RawCompletionDouble:
    """A minimal raw-completions-shaped chunk (#2829) — what the SDK hands the
    raw template transport back from ``client.completions.create``.  Carries
    ``model_copy`` too (#2842): the streaming transport reconstructs the final
    completion from the last chunk via ``model_copy``, the same as the real
    SDK's pydantic objects."""

    model = "m"
    model_extra: dict = {}

    def __init__(self, choices: list[_RawChoice], usage: dict | None = None) -> None:
        self.choices = choices
        self.usage = usage

    def model_copy(self, update: dict | None = None):
        update = update or {}
        return _RawCompletionDouble(
            update.get("choices", self.choices), update.get("usage", self.usage)
        )

    def model_dump(self) -> dict:
        dumped: dict = {"choices": [choice._dump() for choice in self.choices]}
        if self.usage is not None:
            dumped["usage"] = self.usage
        return dumped


class _FakeRawStream:
    """A minimal stand-in for ``openai.AsyncStream`` (#2842): any number of
    chunks, no real HTTP underneath — supports exactly the two protocols
    ``LlmClient._stream_raw_completion`` consumes, ``async with`` and
    ``async for``."""

    def __init__(self, chunks: list[_RawCompletionDouble]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> _FakeRawStream:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk

    def __aiter__(self):
        return self._iterate()


def _text_completion(
    text: str, finish_reason: str = "stop", usage: dict | None = None
) -> _FakeRawStream:
    """A minimal raw-completions-shaped STREAM (#2829, streamed since #2842) —
    what the SDK hands the raw template transport back from
    ``client.completions.create`` under ``stream=True``: one chunk carrying
    the whole text, matching a provider that answers in a single piece."""
    return _FakeRawStream([_RawCompletionDouble([_RawChoice(text, finish_reason)], usage)])


def _chunked_text_completion(pieces: list[str], finish_reason: str = "stop") -> _FakeRawStream:
    """A raw-completions STREAM delivered as several small deltas (#2842) —
    what a looping draw's own repeated text looks like arriving over real
    streaming, piece by piece, rather than already whole. Only the LAST
    chunk carries ``finish_reason``, matching a real provider's own stream
    (every earlier chunk's is ``None``, mid-generation)."""
    chunks = [_RawCompletionDouble([_RawChoice(piece, None)]) for piece in pieces[:-1]]
    chunks.append(_RawCompletionDouble([_RawChoice(pieces[-1], finish_reason)]))
    return _FakeRawStream(chunks)


class TestRawTemplateTransport:
    """A model with a vendored chat template (#2829) routes through raw completions
    instead of chat-completions, and its own grammar parses back into the same
    ``LlmResponse`` shape either transport produces."""

    @pytest.mark.asyncio
    async def test_a_vendored_model_calls_raw_completions_not_chat(
        self, monkeypatch, db: Database
    ) -> None:
        captured: dict = {}

        async def chat_create(**kwargs):
            raise AssertionError("the chat transport should not be called")

        async def raw_create(**kwargs):
            captured.update(kwargs)
            return _text_completion(
                "<|channel>thought\nchecking\n<channel|><|tool_call>call:collection_list"
                '{collection:<|"|>jobs<|"|>,k:5}<tool_call|>'
            )

        client = LlmClient(
            api_url="http://localhost:11434",
            model=_GEMMA_MODEL,
            db=db,
            max_retries=1,
            retry_delay=0.0,
        )
        monkeypatch.setattr(client.client.chat.completions, "create", chat_create)
        monkeypatch.setattr(client.client.completions, "create", raw_create)

        response = await client.chat(
            [LlmMessage(role="user", content="list my jobs")],
            agent_name="chat",
            call_link=CallLink.top("run-raw"),
        )
        await client.close()

        assert response.thinking == "checking"
        assert response.message.tool_calls is not None
        call = response.message.tool_calls[0]
        assert call.function.name == "collection_list"
        assert call.function.arguments == {"collection": "jobs", "k": 5}
        assert captured["model"] == _GEMMA_MODEL
        assert "<|turn>user" in captured["prompt"]
        assert captured["stop"] == ["<turn|>", "<|tool_response>"]

        with db.get_session() as session:
            row = session.exec(select(PromptLog)).one()
        assert row.transport == PromptTransport.RAW_TEMPLATE.value
        # The prompt log stays reader-compatible with the chat shape (#2829) —
        # every existing reader of ``promptlog.response`` assumes
        # ``choices[].message``, which the raw wire shape alone doesn't carry.
        logged_message = json.loads(row.response)["choices"][0]["message"]
        assert logged_message["tool_calls"][0]["function"]["name"] == "collection_list"
        assert logged_message["reasoning_content"] == "checking"

    @pytest.mark.asyncio
    async def test_a_looping_thought_is_cut_mid_stream_and_the_turn_continues(
        self, monkeypatch, db: Database
    ) -> None:
        """The client-level wiring for a thinking-loop cut (#2842): the
        detector fires on real STREAMED chunks (not one whole response), the
        stream closes, one more raw call continues the SAME turn from the
        kept thinking — ``chat()`` still returns exactly one ``LlmResponse``,
        the continuation's, and the discarded cut draw lands as its own row,
        discarded, ahead of the continuation's normal completed one.
        """
        opener = "<|channel>thought\n"
        sentence = "The user is asking about todays meeting. "
        pieces = [sentence] * 20

        # Computed independently, off the exact same chunk sequence the fake
        # stream below delivers — never hand-predicted, so this test isn't
        # fragile to exactly where the real tracker happens to cut.
        tracker = ThoughtStreamTracker()
        buffer = opener
        expected_cut = None
        for piece in pieces:
            buffer += piece
            expected_cut = tracker.feed(buffer)
            if expected_cut is not None:
                break
        assert expected_cut is not None, "fixture must actually trigger the detector"

        calls: list[dict] = []

        async def raw_create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _chunked_text_completion([opener, *pieces])
            return _text_completion("<|channel>thought\nokay, settled\n<channel|>Sounds good.")

        client = LlmClient(
            api_url="http://localhost:11434",
            model=_GEMMA_MODEL,
            db=db,
            max_retries=1,
            retry_delay=0.0,
        )
        monkeypatch.setattr(client.client.completions, "create", raw_create)

        response = await client.chat(
            [LlmMessage(role="user", content="what's up with the meeting")],
            agent_name="chat",
            call_link=CallLink.top("run-loop"),
        )
        await client.close()

        assert len(calls) == 2, "the cut draw + its continuation are TWO raw calls"
        assert calls[1]["prompt"] == calls[0]["prompt"] + expected_cut.continuation_suffix()

        # chat()'s own interface stays untouched: ONE LlmResponse out — the
        # CONTINUATION's, never the discarded draw's.
        assert response.content == "Sounds good."
        assert response.thinking == "okay, settled"

        with db.get_session() as session:
            rows = session.exec(select(PromptLog).order_by(col(PromptLog.id))).all()
        assert len(rows) == 2, "the cut draw and its continuation are TWO promptlog rows"
        cut_row, continuation_row = rows
        assert cut_row.discard_reason == "thinking_loop"
        assert cut_row.discard_detail == expected_cut.discard_detail()
        assert cut_row.call_outcome == CallOutcome.COMPLETED.value
        assert cut_row.thinking == expected_cut.kept_body
        assert continuation_row.discard_reason is None
        assert continuation_row.call_outcome == CallOutcome.COMPLETED.value
        assert continuation_row.thinking == "okay, settled"

    @pytest.mark.asyncio
    async def test_a_looping_continuation_hands_off_to_the_budget_cut_path(
        self, monkeypatch, db: Database
    ) -> None:
        """#2843: the digest bug — a cut landing mid-draft let a real prod
        continuation just resume the unfinished line and loop AGAIN, running
        to its own 16,000-token budget. The continuation is itself guarded
        (``RepeatStreamTracker``, whole output, content included): a second
        loop there gets no THIRD call — it's reported as if the token budget
        had been hit (``finish_reason="length"``), handing off to Penny's
        EXISTING budget-cut discard-and-reroll path at the agent level.
        """
        opener = "<|channel>thought\n"
        first_sentence = "The user is asking about todays meeting. "
        first_pieces = [first_sentence] * 20

        tracker = ThoughtStreamTracker()
        buffer = opener
        first_cut = None
        for piece in first_pieces:
            buffer += piece
            first_cut = tracker.feed(buffer)
            if first_cut is not None:
                break
        assert first_cut is not None, "fixture must trigger the FIRST cut"

        # The continuation's own output loops too — in plain CONTENT, no
        # thought markers at all (exactly the digest bug's own shape: the
        # repeat was in the resumed list item, not a new thought).
        second_sentence = "Here is the same unfinished point again. "
        second_pieces = [second_sentence] * 20
        repeat_tracker = RepeatStreamTracker()
        buffer2 = ""
        second_cut = None
        for piece in second_pieces:
            buffer2 += piece
            second_cut = repeat_tracker.feed(buffer2)
            if second_cut is not None:
                break
        assert second_cut is not None, "fixture must trigger the SECOND cut"

        calls: list[dict] = []

        async def raw_create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _chunked_text_completion([opener, *first_pieces])
            return _chunked_text_completion(second_pieces)

        client = LlmClient(
            api_url="http://localhost:11434",
            model=_GEMMA_MODEL,
            db=db,
            max_retries=1,
            retry_delay=0.0,
        )
        monkeypatch.setattr(client.client.completions, "create", raw_create)

        response = await client.chat(
            [LlmMessage(role="user", content="what's up with the meeting")],
            agent_name="chat",
            call_link=CallLink.top("run-double-loop"),
        )
        await client.close()

        assert len(calls) == 2, "a second loop gets NO third call"
        assert response.finish_reason == PennyConstants.FINISH_REASON_LENGTH
        assert response.content == second_cut.kept_text

        with db.get_session() as session:
            rows = session.exec(select(PromptLog).order_by(col(PromptLog.id))).all()
        assert len(rows) == 2
        cut_row, continuation_row = rows
        assert cut_row.discard_reason == "thinking_loop"
        assert continuation_row.call_outcome == CallOutcome.COMPLETED.value

    @pytest.mark.asyncio
    async def test_an_image_bearing_call_still_uses_chat_even_on_a_vendored_model(
        self, monkeypatch, db: Database
    ) -> None:
        """Images need the provider's own vision processor, which raw completions
        has no equivalent for (#2829) — forced back to chat regardless of model."""

        async def chat_create(**kwargs):
            return _completion("I see it")

        async def raw_create(**kwargs):
            raise AssertionError("an image-bearing call must not route raw")

        client = LlmClient(
            api_url="http://localhost:11434",
            model=_GEMMA_MODEL,
            db=db,
            max_retries=1,
            retry_delay=0.0,
        )
        monkeypatch.setattr(client.client.chat.completions, "create", chat_create)
        monkeypatch.setattr(client.client.completions, "create", raw_create)

        await client.chat(
            [LlmMessage(role="user", content="what is this", images=["c3VpdGNhc2U="])],
            call_link=CallLink.top("run-image"),
        )
        await client.close()

        with db.get_session() as session:
            row = session.exec(select(PromptLog)).one()
        assert row.transport == PromptTransport.CHAT.value

    @pytest.mark.asyncio
    async def test_a_structured_format_call_still_uses_chat_on_a_vendored_model(
        self, monkeypatch, db: Database
    ) -> None:
        """``format`` is an OpenAI-API-level field gemma's raw template has no
        construct for (#2829) — forced back to chat regardless of model."""

        async def chat_create(**kwargs):
            return _completion("{}")

        async def raw_create(**kwargs):
            raise AssertionError("a structured-format call must not route raw")

        client = LlmClient(
            api_url="http://localhost:11434",
            model=_GEMMA_MODEL,
            db=db,
            max_retries=1,
            retry_delay=0.0,
        )
        monkeypatch.setattr(client.client.chat.completions, "create", chat_create)
        monkeypatch.setattr(client.client.completions, "create", raw_create)

        await client.chat(
            [LlmMessage(role="user", content="profile please")],
            format={"type": "object"},
            call_link=CallLink.top("run-format"),
        )
        await client.close()

        with db.get_session() as session:
            row = session.exec(select(PromptLog)).one()
        assert row.transport == PromptTransport.CHAT.value

    @pytest.mark.asyncio
    async def test_disabling_the_raw_transport_reroutes_the_next_call_to_chat(
        self, monkeypatch, db: Database
    ) -> None:
        """The startup preflight's raw-faithfulness check calls this on a failure
        (#2829 build decision 3) — a named degraded mode for the rest of this
        client's life, never a silent one."""

        async def chat_create(**kwargs):
            return _completion("fell back")

        async def raw_create(**kwargs):
            raise AssertionError("the raw transport was disabled and must not be called")

        client = LlmClient(
            api_url="http://localhost:11434",
            model=_GEMMA_MODEL,
            db=db,
            max_retries=1,
            retry_delay=0.0,
        )
        monkeypatch.setattr(client.client.chat.completions, "create", chat_create)
        monkeypatch.setattr(client.client.completions, "create", raw_create)
        assert client.raw_template_enabled

        client.disable_raw_template_transport("endpoint wraps raw completions as a chat turn")
        assert not client.raw_template_enabled

        response = await client.chat(
            [LlmMessage(role="user", content="hi")], call_link=CallLink.top("run-disabled")
        )
        await client.close()

        assert response.content == "fell back"

    @pytest.mark.asyncio
    async def test_a_malformed_raw_tool_call_raises_tool_parse_error(
        self, monkeypatch, db: Database
    ) -> None:
        """A tool call that opens but does not parse is the same non-retried bucket
        a malformed PROVIDER-parsed tool call falls into on the chat path (#2829).

        #2833: the completion was already fetched before parsing it failed, so
        the row keeps that raw text as its own response instead of the usual
        no-response ``{}`` — what the model actually wrote survives for
        diagnosis. The error reaching the caller is still exactly
        ``LlmToolParseError`` with its usual ``TOOL_PARSE`` fault; only what
        gets LOGGED changed.
        """
        malformed_text = "<|tool_call>call:foo{bad"

        async def raw_create(**kwargs):
            return _text_completion(malformed_text)

        client = LlmClient(
            api_url="http://localhost:11434",
            model=_GEMMA_MODEL,
            db=db,
            max_retries=3,
            retry_delay=0.0,
        )
        monkeypatch.setattr(client.client.completions, "create", raw_create)

        with pytest.raises(LlmToolParseError) as excinfo:
            await client.chat(
                [LlmMessage(role="user", content="hi")], call_link=CallLink.top("run-malformed")
            )
        await client.close()

        assert excinfo.value.fault == LlmFault.TOOL_PARSE
        # The error names ~200 characters of context around the failure point
        # (#2833) — here the whole malformed text, since it's short.
        assert malformed_text in str(excinfo.value)

        with db.get_session() as session:
            row = session.exec(select(PromptLog)).one()
        assert row.call_outcome == "raised"
        assert row.transport == PromptTransport.RAW_TEMPLATE.value
        assert row.call_error_class == "LlmToolParseError"
        # The raw completion text — not the usual no-response "{}" — is what
        # survives in the logged row (#2833): what the model actually wrote.
        logged_choice = json.loads(row.response)["choices"][0]
        assert logged_choice["text"] == malformed_text

    @pytest.mark.asyncio
    async def test_the_faithfulness_check_carries_the_provider_pin_and_names_who_answered(
        self, monkeypatch, db: Database
    ) -> None:
        """Regression (#2831 review): the check called ``completions.create`` with
        no pin at all, so an OpenRouter run pinned to CoreWeave could still have
        this ONE check answered by any provider in the pool — usually a
        chat-wrapping one — while the real raw-template calls stayed correctly
        pinned. The pin must ride the check's own request exactly like the real
        one's (``_provider_pin_extra_body``), and the result names whichever
        upstream actually answered (read off the response, not the pin)."""
        captured: dict = {}

        async def raw_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=1),
                model_extra={"provider": "CoreWeave"},
            )

        client = LlmClient(
            api_url="https://gateway.example/api",
            model=_GEMMA_MODEL,
            db=db,
            max_retries=1,
            retry_delay=0.0,
            provider_pin=ProviderPin.pin("CoreWeave"),
        )
        monkeypatch.setattr(client.client.completions, "create", raw_create)

        result = await client.check_raw_template_faithfulness()
        await client.close()

        assert captured["extra_body"] == {
            "provider": {"order": ["CoreWeave"], "allow_fallbacks": False}
        }
        assert result.outcome is RawFaithfulnessOutcome.FAITHFUL
        assert result.provider == "CoreWeave"

    @pytest.mark.asyncio
    async def test_a_429_then_success_is_reported_faithful_not_unverified(
        self, monkeypatch, db: Database
    ) -> None:
        """Regression (#2831 second review): a run measured 71 of 1,275 samples
        hitting a single un-retried 429 here, which a naive read-as-unfaithful
        turned into every later call on that sample 404ing against the chat
        endpoint. A transient fault is retried on the client's own backoff
        (``_retry_after_openai_error``) — a draw that answers on a LATER
        attempt is FAITHFUL, not a verdict drawn from the error that preceded
        it."""
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/completions")
        response = httpx.Response(429, request=request, json={"error": {"message": "slow down"}})
        attempts = {"count": 0}

        async def raw_create(**kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise openai.RateLimitError("Error code: 429", response=response, body=None)
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1), model_extra={})

        client = LlmClient(
            api_url="https://openrouter.ai/api",
            model=_GEMMA_MODEL,
            db=db,
            max_retries=3,
            retry_delay=0.0,
            provider_pin=ProviderPin.pin("CoreWeave"),
        )
        monkeypatch.setattr(client.client.completions, "create", raw_create)

        result = await client.check_raw_template_faithfulness()
        await client.close()

        assert result.outcome is RawFaithfulnessOutcome.FAITHFUL
        assert attempts["count"] == 2
        assert client.raw_template_enabled  # nothing disables it on a FAITHFUL answer

    @pytest.mark.asyncio
    async def test_a_persistent_429_is_unverified_not_unfaithful(
        self, monkeypatch, db: Database
    ) -> None:
        """Every attempt still erroring after retries is UNVERIFIED — an open
        question, not a confirmed chat-wrap — and keeps the raw transport
        (#2831 second review): a real call surfaces the endpoint's own
        availability error honestly, rather than this check guessing on its
        behalf and routing every later call to a chat endpoint that doesn't
        even serve the model."""
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/completions")
        response = httpx.Response(429, request=request, json={"error": {"message": "slow down"}})

        async def always_429(**kwargs):
            raise openai.RateLimitError("Error code: 429", response=response, body=None)

        client = LlmClient(
            api_url="https://openrouter.ai/api",
            model=_GEMMA_MODEL,
            db=db,
            max_retries=3,
            retry_delay=0.0,
        )
        monkeypatch.setattr(client.client.completions, "create", always_429)

        result = await client.check_raw_template_faithfulness()
        await client.close()

        assert result.outcome is RawFaithfulnessOutcome.UNVERIFIED
        assert result.detail and "429" in result.detail
        assert client.raw_template_enabled  # UNVERIFIED keeps the raw transport

    @pytest.mark.asyncio
    async def test_prompt_tokens_fourteen_is_the_chat_wrap_signature_and_disables(
        self, monkeypatch, db: Database
    ) -> None:
        """A real answer whose ``prompt_tokens`` disagrees (14, a chat wrap's own
        turn-opening tokens on the bare BOS token — #2829 step 1) is UNFAITHFUL,
        the only outcome that disables the raw transport (#2831 second review)."""

        async def raw_create(**kwargs):
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=14), model_extra={})

        client = LlmClient(
            api_url="https://openrouter.ai/api",
            model=_GEMMA_MODEL,
            db=db,
            max_retries=1,
            retry_delay=0.0,
        )
        monkeypatch.setattr(client.client.completions, "create", raw_create)

        result = await client.check_raw_template_faithfulness()
        assert result.outcome is RawFaithfulnessOutcome.UNFAITHFUL

        # The preflight is what actually disables it, on exactly this outcome —
        # already covered by test_disabling_the_raw_transport_reroutes_the_next_call_to_chat.
        client.disable_raw_template_transport("chat-wraps raw completions")
        assert not client.raw_template_enabled
        await client.close()
