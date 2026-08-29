"""Tests for LlmClient error summarization.

A non-Ollama backend can return a non-JSON error (e.g. a 404 served as an HTML
page). Logging the raw body dumped thousands of characters per occurrence and
buried real signal, so ``_summarize_llm_error`` reports the HTTP status plus a
short, body-free detail instead — and that summary is what propagates through
the raised ``LlmError`` as well.
"""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from penny.constants import PennyConstants
from penny.llm.client import (
    LlmClient,
    _extract_model_ids,
    _summarize_httpx_error,
    _summarize_llm_error,
)
from penny.llm.models import (
    LlmError,
    LlmFault,
    LlmMessage,
    LlmNotFoundError,
    LlmResponse,
    LlmResponseError,
    LlmTimeoutError,
    ProviderPreference,
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
            await client.chat([{"role": "user", "content": "hi"}])

        message = str(exc_info.value)
        assert "HTTP 404" in message
        assert "<!DOCTYPE" not in message  # summarized, not the raw HTML body

        await client.close()


def _completion(content: str):
    """A minimal ChatCompletion-shaped response — what the SDK hands ``chat`` back."""

    class _Message:
        role = "assistant"
        tool_calls = None
        model_extra: dict = {}

        def __init__(self, text: str) -> None:
            self.content = text

    class _Choice:
        def __init__(self, text: str) -> None:
            self.message = _Message(text)

    class _Completion:
        model = "m"
        usage = None

        def __init__(self, text: str) -> None:
            self.choices = [_Choice(text)]

        def model_dump(self) -> dict:
            return {"choices": [{"message": {"content": content}}]}

    return _Completion(content)


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
        await client.chat([{"role": "user", "content": "hi"}])

        assert sent["extra_body"] == {"reasoning": {"enabled": True}}
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
            await client.chat([{"role": "user", "content": "hi"}])

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
            await client.chat([{"role": "user", "content": "hi"}])

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
            await client.chat([{"role": "user", "content": "hi"}])

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
            await client.chat([{"role": "user", "content": "hi"}])

        assert exc_info.value.fault == LlmFault.NO_CHOICES
        await client.close()


class TestRoutingIsAPreferenceNotAWall:
    """Which upstream serves a model is asked for, observed, and never enforced (#1996)."""

    def test_a_preference_travels_with_fallbacks_on(self) -> None:
        """Hard pinning concentrates a whole run's load on one upstream.

        Measured: `allow_fallbacks: false` put 325 rate limits on ONE endpoint at a
        concurrency the same run handled with zero unpinned. So the default is a
        preference — the pool's throughput stays available and a fallback is recorded
        rather than forbidden.
        """
        preference = ProviderPreference.prefer("Cloudflare")
        assert preference is not None
        assert preference.as_request_field() == {
            "order": ["Cloudflare"],
            "allow_fallbacks": True,
        }

    def test_no_configured_provider_is_no_preference_at_all(self) -> None:
        """A direct endpoint has no upstreams, so nothing is sent — not an empty pin."""
        assert ProviderPreference.prefer(None) is None
        assert ProviderPreference.prefer("") is None

    @pytest.mark.asyncio
    async def test_the_preference_reaches_the_request_body(self, monkeypatch) -> None:
        """A preference nothing sends is a preference that does nothing."""
        sent: dict = {}

        async def capture(**kwargs):
            sent.update(kwargs)
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1"))

        client = LlmClient(
            api_url="https://gateway.example/api",
            model="m",
            max_retries=1,
            retry_delay=0.0,
            provider_preference=ProviderPreference.prefer("Cloudflare"),
        )
        monkeypatch.setattr(client.client.chat.completions, "create", capture)

        with pytest.raises(LlmError):
            await client.chat([{"role": "user", "content": "hi"}])

        assert sent["extra_body"]["provider"] == {
            "order": ["Cloudflare"],
            "allow_fallbacks": True,
        }
        # The reasoning switch still rides the same passthrough — one is not added at the
        # cost of the other, and a run that lost it would be comparing a model to itself.
        assert sent["extra_body"]["reasoning"] == {"enabled": True}
        await client.close()

    @pytest.mark.asyncio
    async def test_a_client_with_no_preference_sends_the_body_it_always_sent(
        self, monkeypatch
    ) -> None:
        sent: dict = {}

        async def capture(**kwargs):
            sent.update(kwargs)
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1"))

        client = LlmClient(
            api_url="http://localhost:11434", model="m", max_retries=1, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", capture)

        with pytest.raises(LlmError):
            await client.chat([{"role": "user", "content": "hi"}])

        assert sent["extra_body"] == {"reasoning": {"enabled": True}}
        await client.close()

    @pytest.mark.asyncio
    async def test_the_upstream_that_answered_comes_back_on_the_response(self, monkeypatch) -> None:
        """Reproducibility is OBSERVED: the answer says who served it, preference or not."""

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
                        )()
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
            provider_preference=ProviderPreference.prefer("Cloudflare"),
        )
        monkeypatch.setattr(client.client.chat.completions, "create", answer)

        response = await client.chat([{"role": "user", "content": "hi"}])

        # It preferred Cloudflare and DeepInfra answered — a fallback, stated rather than
        # hidden behind an assumption that the pin held.
        assert response.provider == "DeepInfra"
        assert isinstance(response, LlmResponse)
        assert isinstance(response.message, LlmMessage)
        await client.close()
