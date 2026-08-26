"""The up-front endpoint check: what it proves, and what it says when it fails."""

from __future__ import annotations

import pytest

from penny.llm.client import LlmClient
from penny.llm.models import LlmMessage, LlmNotFoundError, LlmResponse
from penny.tests.eval.endpoint_smoke import main, smoke

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

    monkeypatch.setattr(LlmClient, "chat", answer)
    assert main([]) == 0

    # A stray argument is a usage error, distinct from a refused endpoint.
    assert main(["unexpected"]) == 2
