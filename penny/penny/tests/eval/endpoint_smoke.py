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

from pydantic import BaseModel

from penny.llm.client import LlmClient
from penny.llm.models import LlmError

USAGE = "usage: python -m penny.tests.eval.endpoint_smoke"

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

# One attempt, not the run's retry budget: a smoke test that retries turns a fast, clear
# refusal into a slow one, and the failures it exists to catch (an unroutable model, a
# rejected key) are not transient.
SMOKE_MAX_RETRIES = 1
SMOKE_RETRY_DELAY = 0.0
SMOKE_TIMEOUT_SECONDS = 30.0

_LOCAL_ENDPOINT = "http://localhost:11434"
_DEFAULT_MODEL = "gpt-oss:20b"
_DEFAULT_API_KEY = "not-needed"


class SmokeResult(BaseModel):
    """Whether the configured endpoint served the configured model, and what it said."""

    ok: bool
    detail: str

    def render(self) -> str:
        """The one line the Makefile prints, in the house's eval-line voice."""
        mark = "eval: endpoint ok" if self.ok else "eval: endpoint REFUSED"
        return f"{mark} — {self.detail}"


async def smoke(*, api_url: str, model: str, api_key: str) -> SmokeResult:
    """Make one tool-bearing chat call and report what came back.

    Every failure is reported rather than raised, because the caller's job is to print the
    reason and stop the run, and a traceback out of a smoke test buries the provider's own
    message under the harness's.
    """
    client = LlmClient(
        api_url=api_url,
        model=model,
        max_retries=SMOKE_MAX_RETRIES,
        retry_delay=SMOKE_RETRY_DELAY,
        api_key=api_key,
        timeout=SMOKE_TIMEOUT_SECONDS,
    )
    try:
        await client.chat([{"role": "user", "content": SMOKE_PROMPT}], tools=[SMOKE_TOOL])
        return SmokeResult(ok=True, detail=f"{model} answered at {api_url}")
    except LlmError as error:
        return SmokeResult(ok=False, detail=f"{model} at {api_url}: {error}")
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
    result = asyncio.run(
        smoke(
            api_url=os.environ.get("LLM_API_URL") or _LOCAL_ENDPOINT,
            model=os.environ.get("LLM_MODEL") or _DEFAULT_MODEL,
            api_key=os.environ.get("LLM_API_KEY") or _DEFAULT_API_KEY,
        )
    )
    print(result.render(), file=sys.stderr if not result.ok else sys.stdout)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
