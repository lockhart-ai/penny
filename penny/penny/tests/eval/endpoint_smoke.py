"""One call against the configured model, before a run commits to anything.

A run's endpoint config is proven 755 times or not at all: every sample builds its own
Penny and its own preflight, so a model the endpoint will not serve is discovered
concurrently, per sample, minutes deep — 755 correct hard-failures where one would have
done.  That is how an 18-minute full-suite run was spent against a model whose provider
answered every single call with 404: the harness knew from the first sample and had no
way to say so before the rest started.

So the endpoint is checked ONCE, up front, by the thing that is about to spend an hour on
it, and the run refuses to start when the check fails — carrying the PROVIDER'S OWN words,
since "no allowed providers are available for the selected model … your account's
allowed-providers setting permits only: amazon-bedrock" is the whole answer and nothing
the harness could infer would improve on it.

The check is the real ``LlmClient`` making a real tool-bearing chat call, because what a
smoke test proves has to be what the run does — a bare reachability probe would have
passed happily against exactly the endpoint that could not serve a completion.

It does NOT require the model to actually CALL the tool: a draw that answers in prose is
a working endpoint and a stochastic model, and refusing the run over it would be a flaky
gate on a check whose whole job is to be decisive.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from penny.llm.client import LlmClient
from penny.llm.models import LlmError, LlmFault, ProviderPreference

USAGE = "usage: python -m penny.tests.eval.endpoint_smoke"

# The line the Makefile reads the answering provider off, so the run's manifest can record
# WHERE its model was served from beside WHICH model it was.  Human-readable AND parseable
# on purpose: a machine-only channel would be a second thing to keep in step with what the
# operator sees, and the operator wants to see it too.
PROVIDER_LINE_PREFIX = "eval: chat provider ="

# The run's configured routing preference, forwarded by the Makefile from the roster.
LLM_PROVIDER_ENV = "LLM_PROVIDER"

# The probe's own message and tool.  Deliberately trivial: this measures whether the
# endpoint SERVES the model with a tool surface attached, not whether the model is any
# good, so the cheapest possible exchange is the right one.
SMOKE_PROMPT = "Reply with the single word: ready."
SMOKE_TOOL = {
    "type": "function",
    "function": {
        "name": "ready",
        "description": "Acknowledge that the endpoint is reachable.",
        "parameters": {"type": "object", "properties": {}},
    },
}

# The client's OWN retry budget stays at one, because retrying inside the client makes one
# attempt take three times as long and reports the burst as a single opaque failure.  The
# attempts loop is out here instead (see ``SMOKE_ATTEMPTS``), where it can read the fault
# CLASS and decide.
# The embedding probe's text. Content is irrelevant — what is being proven is that the
# vector backend ANSWERS, because its failure is the quiet kind: the write path stores a
# NULL vector and carries on, so a whole memory suite ran against a dead embedding endpoint
# (119 consecutive 400s) and still scored ~1.00. Listing the model is not enough either —
# the preflight checked that gemini-embedding-001 was listed, and it was; it just could not
# embed. Only a real call settles it.
SMOKE_EMBED_TEXT = "a harbour timetable page"

SMOKE_MAX_RETRIES = 1
SMOKE_RETRY_DELAY = 0.0
SMOKE_TIMEOUT_SECONDS = 30.0

# How many draws a TRANSIENT fault gets before the check gives up, and how long it waits
# between them (doubling, like the client's own backoff).
#
# The check refused two whole runs on ONE un-retried `HTTP 502` while the same model
# answered a direct call seconds later.  That is the right refusal for a model the endpoint
# will never serve and the wrong one for a provider having a bad minute — and the fault
# class already tells the two apart, so retrying is keyed to ``LlmFault.transient`` rather
# than to a count of everything.  A permanently unroutable model still refuses on the FIRST
# draw (a 404 is not transient), so the fast, decisive refusal this check exists for is
# unchanged; only a moment gets a second look.  Three draws with a 2s then 4s wait spends at
# most ~6 seconds of waiting against the 18-minute run it is protecting.
SMOKE_ATTEMPTS = 3
SMOKE_ATTEMPT_DELAY_SECONDS = 2.0

_LOCAL_ENDPOINT = "http://localhost:11434"
_DEFAULT_MODEL = "gpt-oss:20b"
_DEFAULT_EMBEDDING_MODEL = "embeddinggemma"
_DEFAULT_API_KEY = "not-needed"


class SmokeResult(BaseModel):
    """Whether one configured endpoint served its model, what it said, and who answered."""

    subject: str
    ok: bool
    detail: str
    # Which upstream served the probe, when the endpoint is a gateway that names one.  It
    # rides the run into the manifest, so a run poisoned by one member of a routing pool is
    # diagnosable from its artifacts instead of by running the whole thing again with a pin.
    provider: str | None = None
    # What a failed draw failed OF — the value that decides whether another draw is worth
    # making.  ``None`` on a draw that answered.
    fault: LlmFault | None = None

    @property
    def retryable(self) -> bool:
        """Whether another draw could plausibly get past this failure."""
        return self.fault is not None and self.fault.transient

    def render(self) -> str:
        """The one line the Makefile prints, in the house's eval-line voice."""
        mark = "ok" if self.ok else "REFUSED"
        return f"eval: {self.subject} endpoint {mark} — {self.detail}"

    def render_provider(self) -> str | None:
        """The line the Makefile READS the provider off, or ``None`` when none answered."""
        return f"{PROVIDER_LINE_PREFIX} {self.provider}" if self.provider else None


async def _retrying(probe: Callable[[], Awaitable[SmokeResult]]) -> SmokeResult:
    """Run a probe until it passes, a fault says another draw cannot help, or draws run out.

    ONE place decides how many looks a refusal gets, shared by both probes, because the
    question — is this endpoint unserveable, or is it having a moment? — is the same for a
    chat model and a vector backend.  The verdict is the LAST draw's, so the message the
    operator reads is the reason the check actually gave up on.
    """
    for attempt in range(SMOKE_ATTEMPTS):
        result = await probe()
        if result.ok or not result.retryable or attempt == SMOKE_ATTEMPTS - 1:
            return result
        print(
            f"eval: {result.subject} endpoint draw {attempt + 1} of {SMOKE_ATTEMPTS} "
            f"failed on a transient fault — {result.detail}",
            file=sys.stderr,
        )
        await asyncio.sleep(SMOKE_ATTEMPT_DELAY_SECONDS * 2**attempt)
    raise AssertionError("unreachable: the loop returns on its last attempt")


def _smoke_client(
    *, api_url: str, model: str, api_key: str, provider: str | None = None
) -> LlmClient:
    """The probe's client — the REAL one, with its own retries off (see SMOKE_ATTEMPTS).

    It carries the run's own routing PREFERENCE, so what the probe proves is the routing
    the samples will use — which makes the provider it reports back the answer to "did the
    preference hold?" rather than to "who serves this model in general?".
    """
    return LlmClient(
        api_url=api_url,
        model=model,
        provider_preference=ProviderPreference.prefer(provider),
        max_retries=SMOKE_MAX_RETRIES,
        retry_delay=SMOKE_RETRY_DELAY,
        api_key=api_key,
        timeout=SMOKE_TIMEOUT_SECONDS,
    )


async def smoke(
    *, api_url: str, model: str, api_key: str, provider: str | None = None
) -> SmokeResult:
    """Make tool-bearing chat calls until one answers or a fault settles it.

    Every failure is reported rather than raised, because the caller's job is to print the
    reason and stop the run, and a traceback out of a smoke test buries the provider's own
    message under the harness's.
    """
    return await _retrying(
        lambda: _smoke_once(api_url=api_url, model=model, api_key=api_key, provider=provider)
    )


async def _smoke_once(
    *, api_url: str, model: str, api_key: str, provider: str | None = None
) -> SmokeResult:
    """One tool-bearing chat call, reported with the fault class its failure carried."""
    client = _smoke_client(api_url=api_url, model=model, api_key=api_key, provider=provider)
    try:
        response = await client.chat(
            [{"role": "user", "content": SMOKE_PROMPT}], tools=[SMOKE_TOOL]
        )
        return SmokeResult(
            subject="chat",
            ok=True,
            detail=f"{model} answered at {api_url}",
            provider=response.provider,
        )
    except LlmError as error:
        return SmokeResult(
            subject="chat",
            ok=False,
            detail=f"{model} at {api_url}: {error}",
            fault=error.fault,
        )
    finally:
        await client.close()


async def smoke_embedding(*, api_url: str, model: str, api_key: str) -> SmokeResult:
    """Make real embedding calls until a usable vector comes back or a fault settles it.

    Separate from the chat probe because they are separate endpoints with separate
    credentials, and a run can have one working while the other does not — which is
    precisely the state that produced a green memory suite with no vectors in it.
    """
    return await _retrying(
        lambda: _smoke_embedding_once(api_url=api_url, model=model, api_key=api_key)
    )


async def _smoke_embedding_once(*, api_url: str, model: str, api_key: str) -> SmokeResult:
    """One real embedding call, reported with the fault class its failure carried."""
    client = _smoke_client(api_url=api_url, model=model, api_key=api_key)
    try:
        vectors = await client.embed(SMOKE_EMBED_TEXT)
        if not vectors or not vectors[0]:
            return SmokeResult(
                subject="embedding",
                ok=False,
                detail=f"{model} at {api_url} returned no vector",
                # A backend answering with nothing is the vector twin of a 200 carrying no
                # choices — the class another draw most often gets past.
                fault=LlmFault.NO_CHOICES,
            )
        return SmokeResult(
            subject="embedding",
            ok=True,
            detail=f"{model} answered at {api_url} (dim {len(vectors[0])})",
        )
    except LlmError as error:
        return SmokeResult(
            subject="embedding",
            ok=False,
            detail=f"{model} at {api_url}: {error}",
            fault=error.fault,
        )
    finally:
        await client.close()


def main(argv: list[str]) -> int:
    """Print the one-line verdict; 0 when the endpoint served the model, 1 when it did not.

    Reads the same three variables the run itself is given, so what is proven here and
    what the samples use cannot be two different configurations.
    """
    if argv:
        print(USAGE, file=sys.stderr)
        return 2

    async def both() -> list[SmokeResult]:
        return [
            await smoke(
                api_url=os.environ.get("LLM_API_URL") or _LOCAL_ENDPOINT,
                model=os.environ.get("LLM_MODEL") or _DEFAULT_MODEL,
                api_key=os.environ.get("LLM_API_KEY") or _DEFAULT_API_KEY,
                provider=os.environ.get(LLM_PROVIDER_ENV) or None,
            ),
            await smoke_embedding(
                api_url=os.environ.get("LLM_EMBEDDING_API_URL") or _LOCAL_ENDPOINT,
                model=os.environ.get("LLM_EMBEDDING_MODEL") or _DEFAULT_EMBEDDING_MODEL,
                api_key=os.environ.get("LLM_EMBEDDING_API_KEY") or _DEFAULT_API_KEY,
            ),
        ]

    # BOTH are reported before the verdict: knowing which of the two refused is the whole
    # difference between a wrong model name and a dead vector backend.
    results = asyncio.run(both())
    for result in results:
        print(result.render(), file=sys.stdout if result.ok else sys.stderr)
        provider_line = result.render_provider()
        if provider_line:
            print(provider_line)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
