"""The up-front endpoint check: what it proves, and what it says when it fails."""

from __future__ import annotations

import pytest

from penny.llm.client import LlmClient
from penny.llm.models import (
    LlmFault,
    LlmMessage,
    LlmNotFoundError,
    LlmResponse,
    LlmResponseError,
)
from penny.tests.eval.utils.endpoint_smoke import (
    SMOKE_ATTEMPTS,
    main,
    smoke,
    smoke_embedding,
)

# The verbatim shape of the refusal that motivated this check: a provider allowlist that
# excludes every provider serving the requested model. Kept whole because the point of the
# check is that this text reaches the operator unaltered.
_PROVIDER_REFUSAL = (
    "HTTP 404: No allowed providers are available for the selected model. Providers "
    "serving google/gemma-4-26b-a4b-it: deepinfra, cloudflare, google-vertex, but your "
    "account's allowed-providers setting permits only: amazon-bedrock."
)


@pytest.mark.asyncio
async def test_a_served_model_passes_the_check(monkeypatch) -> None:
    """A completion coming back is the whole contract — the tool need not be called.

    A draw that answers in prose is a working endpoint and a stochastic model; failing the
    run over it would make a gate that exists to be decisive into a flaky one.
    """
    sent = {}

    async def answer(self, messages, tools=None, **kwargs):  # bound: patched on the class
        sent["tools"] = tools
        return LlmResponse(message=LlmMessage(role="assistant", content="ready."))

    monkeypatch.setattr(LlmClient, "chat", answer)
    result = await smoke(api_url="http://localhost:11434", model="some-model", api_key="k")

    assert result.ok
    assert "some-model" in result.detail and "http://localhost:11434" in result.detail
    # Tool-bearing on purpose: a bare reachability probe passes against an endpoint that
    # cannot actually serve a completion, which is the case this exists to catch.
    assert sent["tools"], "the smoke call must carry a tool surface"


@pytest.mark.asyncio
async def test_a_refused_model_fails_carrying_the_providers_own_words(monkeypatch) -> None:
    """The provider's message IS the answer — nothing the harness infers improves on it."""

    async def refuse(self, messages, tools=None, **kwargs):  # bound: patched on the class
        raise LlmNotFoundError(_PROVIDER_REFUSAL)

    monkeypatch.setattr(LlmClient, "chat", refuse)
    result = await smoke(
        api_url="https://openrouter.ai/api", model="google/gemma-4-26b-a4b-it", api_key="k"
    )

    assert not result.ok
    assert "allowed-providers setting permits only: amazon-bedrock" in result.detail
    assert "REFUSED" in result.render()


@pytest.mark.asyncio
async def test_a_dead_embedding_backend_is_caught_before_the_run(monkeypatch) -> None:
    """Listing the model is not proof it can embed — only a real call is.

    The failure this exists for is the quiet one: a backend refused every embedding call,
    the write path stored NULL vectors and carried on, and a whole memory suite scored
    ~1.00 with no vectors in it at all. The preflight had passed, because the model WAS
    listed; it just could not embed.
    """

    async def refuse(self, text):  # bound: patched on the class
        raise LlmResponseError("HTTP 400: embeddings do not support base64 encoding_format")

    monkeypatch.setattr(LlmClient, "embed", refuse)
    result = await smoke_embedding(
        api_url="https://openrouter.ai/api", model="some/embedder", api_key="k"
    )
    assert not result.ok
    assert "base64 encoding_format" in result.detail
    assert "embedding endpoint REFUSED" in result.render()


@pytest.mark.asyncio
async def test_an_empty_vector_is_a_refusal_not_a_pass(monkeypatch) -> None:
    """A backend that answers with nothing is as dead as one that errors."""

    async def empty(self, text):  # bound: patched on the class
        return [[]]

    monkeypatch.setattr(LlmClient, "embed", empty)
    result = await smoke_embedding(api_url="http://localhost:11434", model="m", api_key="k")
    assert not result.ok
    assert "no vector" in result.detail


def test_the_cli_exit_code_is_what_stops_the_run(monkeypatch) -> None:
    """`make eval` reads the exit code, so a refusal must be non-zero and a pass zero."""
    monkeypatch.setenv("LLM_API_URL", "https://openrouter.ai/api")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_API_KEY", "k")

    async def refuse(self, messages, tools=None, **kwargs):  # bound: patched on the class
        raise LlmNotFoundError(_PROVIDER_REFUSAL)

    monkeypatch.setattr(LlmClient, "chat", refuse)
    assert main([]) == 1

    async def answer(self, messages, tools=None, **kwargs):  # bound: patched on the class
        return LlmResponse(message=LlmMessage(role="assistant", content="ready."))

    async def embed_ok(self, text):  # bound: patched on the class
        return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(LlmClient, "chat", answer)
    monkeypatch.setattr(LlmClient, "embed", embed_ok)
    assert main([]) == 0

    # A stray argument is a usage error, distinct from a refused endpoint.
    assert main(["unexpected"]) == 2


# ── One bad draw is not an unroutable model (#1996) ──


@pytest.fixture(autouse=True)
def _no_real_waiting(monkeypatch) -> None:
    """The backoff between draws is real seconds in production and none of them here."""

    async def instantly(seconds: float) -> None:
        return None

    monkeypatch.setattr("penny.tests.eval.utils.endpoint_smoke.asyncio.sleep", instantly)


@pytest.mark.asyncio
async def test_a_provider_having_a_bad_minute_gets_another_draw(monkeypatch) -> None:
    """The measured failure: a single `HTTP 502` blocked two whole runs the model was fine for.

    A 5xx is a moment, not a verdict, so the check draws again — and the second answer is
    the one that counts.
    """
    draws = 0

    async def flaky(self, messages, tools=None, **kwargs):  # bound: patched on the class
        nonlocal draws
        draws += 1
        if draws == 1:
            raise LlmResponseError("HTTP 502: Bad gateway", fault=LlmFault.SERVER_ERROR)
        return LlmResponse(message=LlmMessage(role="assistant", content="ready."))

    monkeypatch.setattr(LlmClient, "chat", flaky)
    result = await smoke(api_url="https://gateway.example/api", model="m", api_key="k")

    assert result.ok
    assert draws == 2


@pytest.mark.asyncio
async def test_an_unroutable_model_still_refuses_on_the_first_draw(monkeypatch) -> None:
    """The check exists to be DECISIVE. A 404 is not transient, so nothing is spent on it."""
    draws = 0

    async def refuse(self, messages, tools=None, **kwargs):  # bound: patched on the class
        nonlocal draws
        draws += 1
        raise LlmNotFoundError(_PROVIDER_REFUSAL)

    monkeypatch.setattr(LlmClient, "chat", refuse)
    result = await smoke(
        api_url="https://openrouter.ai/api", model="google/gemma-4-26b-a4b-it", api_key="k"
    )

    assert not result.ok
    assert draws == 1
    assert "allowed-providers setting permits only: amazon-bedrock" in result.detail


@pytest.mark.asyncio
async def test_a_transient_fault_that_never_clears_gives_up_and_says_why(monkeypatch) -> None:
    """Retrying is bounded: an endpoint down for the whole window still stops the run."""
    draws = 0

    async def always_502(self, messages, tools=None, **kwargs):  # bound: patched on the class
        nonlocal draws
        draws += 1
        raise LlmResponseError("HTTP 502: Bad gateway", fault=LlmFault.SERVER_ERROR)

    monkeypatch.setattr(LlmClient, "chat", always_502)
    result = await smoke(api_url="https://gateway.example/api", model="m", api_key="k")

    assert not result.ok
    assert draws == SMOKE_ATTEMPTS
    assert "Bad gateway" in result.detail


@pytest.mark.asyncio
async def test_a_dead_vector_backend_is_retried_the_same_way(monkeypatch) -> None:
    """One rule, both probes: the question — dead, or having a moment? — is the same."""
    draws = 0

    async def flaky(self, text):  # bound: patched on the class
        nonlocal draws
        draws += 1
        if draws == 1:
            return [[]]  # answered with nothing — the vector twin of an empty completion
        return [[0.1, 0.2]]

    monkeypatch.setattr(LlmClient, "embed", flaky)
    result = await smoke_embedding(api_url="http://localhost:11434", model="m", api_key="k")

    assert result.ok
    assert draws == 2


@pytest.mark.asyncio
async def test_the_answering_provider_is_reported_for_the_manifest(monkeypatch) -> None:
    """A run poisoned by one member of a routing pool must be diagnosable from artifacts.

    The provider that answered rides out of the check on its own parseable line, so the
    manifest records WHERE the model was served from beside WHICH model it was.
    """

    async def answer(self, messages, tools=None, **kwargs):  # bound: patched on the class
        return LlmResponse(
            message=LlmMessage(role="assistant", content="ready."), provider="cloudflare"
        )

    monkeypatch.setattr(LlmClient, "chat", answer)
    result = await smoke(api_url="https://openrouter.ai/api", model="m", api_key="k")

    assert result.provider == "cloudflare"
    assert result.render_provider() == "eval: chat provider = cloudflare"


@pytest.mark.asyncio
async def test_an_endpoint_that_names_no_provider_reports_no_line(monkeypatch) -> None:
    """A local Ollama names no upstream, and the check invents nothing to fill the gap."""

    async def answer(self, messages, tools=None, **kwargs):  # bound: patched on the class
        return LlmResponse(message=LlmMessage(role="assistant", content="ready."))

    monkeypatch.setattr(LlmClient, "chat", answer)
    result = await smoke(api_url="http://localhost:11434", model="m", api_key="k")

    assert result.provider is None
    assert result.render_provider() is None
