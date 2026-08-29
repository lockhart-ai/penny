"""Native-API performance probe — the true generation (decode) tok/s.

The contract cases report END-TO-END throughput (visible output ÷ full request
wall, which is dominated by prompt processing) because our ``LlmClient`` talks to
Ollama's OpenAI-compatible ``/v1`` endpoint, which strips the timing breakdown.
This probe hits Ollama's NATIVE ``/api/chat`` for the configured model and reports
what ``/v1`` can't: prefill (prompt-processing) tok/s, decode (generation) tok/s,
and the reasoning share of generation — the same numbers ``ollama run --verbose``
prints, captured per model so a swap can be judged on raw speed as well as
correctness.

Marked ``eval`` so it runs only under ``make eval``.  It measures; it does not
gate — it asserts only that the native endpoint answered with timings.

It is the ONE case that cannot run against a remote OpenAI-compatible provider: the
native endpoint is Ollama's, and a gateway serving ``/v1`` has no ``/api/chat`` to
answer.  Rather than fail the run with a 404 on a URL nobody meant to build, it SKIPS
and names the endpoint it wanted — a suite otherwise green should not carry one red
case that is red for being pointed somewhere it was never meant to go.
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.eval

_API_URL = os.environ.get("LLM_API_URL", "http://localhost:11434").rstrip("/")
_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:20b")

# A prompt that generates a few hundred tokens — long enough that decode reaches
# steady state (a short generation is dominated by first-token latency).
_DECODE_PROMPT = "Write about 300 words explaining, in plain language, how ocean tides work."
# A long input with a tiny output — isolates prompt-processing (prefill) speed.
_PREFILL_PREAMBLE = "Reference material:\n" + ("Tides are driven by gravity. " * 800)


async def _chat(client: httpx.AsyncClient, content: str) -> dict:
    """One non-streamed native chat call — the response carries Ollama's timings."""
    response = await client.post(
        f"{_API_URL}/api/chat",
        json={"model": _MODEL, "messages": [{"role": "user", "content": content}], "stream": False},
    )
    response.raise_for_status()
    return response.json()


def _is_native_ollama(api_url: str) -> bool:
    """Whether this endpoint serves Ollama's NATIVE API, the only one this probe can read.

    Read from the endpoint rather than from a flag the caller must remember to pass: the
    probe's own requirement is what decides, so pointing a run anywhere else skips it
    without anyone having to know it exists.
    """
    return not api_url.startswith("https://")


async def test_perf_probe() -> None:
    if not _is_native_ollama(_API_URL):
        pytest.skip(f"native-API probe needs an Ollama endpoint; {_API_URL} serves /v1 only")
    async with httpx.AsyncClient(timeout=300.0) as client:
        await _chat(client, "Hi")  # warm-up: load weights so the measured calls are steady-state
        decode = await _chat(client, _DECODE_PROMPT)
        prefill = await _chat(
            client, f"{_PREFILL_PREAMBLE}\n\nSummarize the above in one sentence."
        )

    eval_count, eval_ns = decode.get("eval_count", 0), decode.get("eval_duration", 0)
    prompt_count, prompt_ns = (
        prefill.get("prompt_eval_count", 0),
        prefill.get("prompt_eval_duration", 0),
    )
    decode_tps = eval_count / (eval_ns / 1e9) if eval_ns else 0.0
    prefill_tps = prompt_count / (prompt_ns / 1e9) if prompt_ns else 0.0
    message = decode.get("message", {})
    thinking_len = len(message.get("thinking") or "")
    content_len = len(message.get("content") or "")
    reasoning_share = thinking_len / (thinking_len + content_len or 1)
    decode_wall_s = decode.get("total_duration", 0) / 1e9
    print(
        f"\nPERF-PROBE [{_MODEL}] "
        f"decode {decode_tps:.1f} tok/s ({eval_count} tok) · "
        f"prefill {prefill_tps:.1f} tok/s ({prompt_count} tok) · "
        f"decode-call wall {decode_wall_s:.1f}s · "
        f"reasoning {reasoning_share * 100:.0f}% of generation"
    )
    assert eval_count > 0, "native /api/chat returned no eval_count — timing unavailable"
