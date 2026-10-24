"""Client for Ollama's ``/v1/systemone`` decision-model endpoint.

One Ollama-specific REST surface the OpenAI-compatible ``LlmClient`` does not
carry (unlike image generation, which moved off Ollama entirely onto the local
mflux server — see ``penny.llm.image_client.MfluxImageClient`` — #2792).
A decision model answers *typed choice questions* — a state, one question, a
fixed set of options each with its own description — and returns the chosen
option with its per-option probabilities.

Customers: the entry-identity ladder (``penny.database.memory.identity``), which
asks whether a new entry is the same thing as a stored one, and — later — the
collection-name ladder and the browse picker.  ``model`` unset means no client:
the caller runs in its named degraded mode, decided at the construction site
from config, never by a silent ``None`` default deep in a call chain.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel

_DEFAULT_TIMEOUT = 60.0
_QUESTION_ID = "decision"


class DecisionError(Exception):
    """A decision-model call failed — unreachable endpoint, bad status, or an
    answer without a choice.  The caller treats this as its ``unsure`` outcome
    and says so, never as a silently-settled decision."""


class DecisionAnswer(BaseModel):
    """One answered choice question: the chosen option and, when the endpoint
    reports them, the probabilities per option."""

    choice: str
    probabilities: dict[str, float] = {}


class DecisionClient:
    """Async client for Ollama's ``/v1/systemone`` choice-question endpoint."""

    def __init__(self, model: str, api_url: str, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self.model = model
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    async def decide(
        self, *, state: str, instructions: str, criteria: dict[str, str]
    ) -> DecisionAnswer:
        """Ask one ``choice`` question about ``state`` and return the chosen
        option.

        ``criteria`` maps option name → its description; the endpoint sees both
        the question's ``instructions`` and the per-option ``criteria``.  Any
        transport, status, or shape failure raises :class:`DecisionError` rather
        than a guessed answer."""
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                _QUESTION_ID: {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": criteria,
                }
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.api_url}/v1/systemone", json=payload)
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise DecisionError(f"decision-model call failed: {error}") from error
        answer = (body.get("answers") or {}).get(_QUESTION_ID) or {}
        choice = answer.get("choice")
        if not isinstance(choice, str) or not choice:
            raise DecisionError(f"decision-model answer carries no choice: {str(body)[:200]}")
        probabilities = answer.get("probabilities") or {}
        return DecisionAnswer(choice=choice, probabilities=probabilities)

    async def list_models(self) -> list[str]:
        """Model names the endpoint serves (Ollama's native ``/api/tags``) — the
        startup preflight's availability check for the decision model."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.api_url}/api/tags")
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise DecisionError(f"decision-model listing failed: {error}") from error
        return [entry.get("name", "") for entry in body.get("models", []) if entry.get("name")]


def decision_api_url(config) -> str:
    """The endpoint a decision-model call goes to: ``LLM_DECISION_API_URL`` when
    set, else the embedding endpoint's host — the same Ollama both already sit
    on.  ``config`` is a :class:`~penny.config.Config`; typed loosely so this
    module stays import-light."""
    return config.llm_decision_api_url or config.llm_embedding_api_url or config.llm_api_url
