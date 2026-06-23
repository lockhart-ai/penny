"""Fixtures for the live-model eval suite.

Construction reuses the integration-test isolation core (``running_penny``)
with a config whose model points at the real Ollama endpoint — no second
construction path, no stubs.  Each case samples N runs (the model is
stochastic) and reports a pass-rate against PERSISTED DB state, which is the
real contract.  A case gates on a ``min_pass_rate`` threshold, or — for
inherently stochastic behaviours (``min_pass_rate=None``) — just prints its X/Y
rate for inspection without failing the run.  See docs/self-improvement-loop.md.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import ceil
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel import Session, select

from penny.config import Config
from penny.constants import ChannelType
from penny.database import Database
from penny.database.memory import EntryInput, Inclusion, RecallMode
from penny.database.message_store import PromptPerf
from penny.database.models import MemoryRow, PromptLog
from penny.llm.client import LlmClient
from penny.llm.models import LlmMessage, LlmResponse, LlmToolCall, LlmToolCallFunction
from penny.penny import Penny
from penny.startup import get_restart_message
from penny.tests.conftest import TEST_SENDER, run_penny_with_server
from penny.tests.eval.fixtures import CannedPage, SynthCollection
from penny.tests.mocks.signal_server import MockSignalServer

# Samples per case.  Override with EVAL_SAMPLES=2 for a quick smoke run.
SAMPLES = int(os.environ.get("EVAL_SAMPLES", "5"))

# Embedding backfill batch size for seeded memory.
_EMBED_BATCH = 100

# A chat scorer reads persisted DB state (the pre-run collection names + the
# final reply text) and returns failure strings — empty means the sample passed.
Scorer = Callable[[Database, set[str], str], list[str]]
Seeder = Callable[[Database], None]
# A collector scorer also sees the pre-cycle snapshot and the messages the cycle
# sent the user.  ``snapshot`` is whatever the case's ``snapshot`` callback returned.
Snapshotter = Callable[[Database], object]
CollectorScorer = Callable[[Database, object, list[str]], list[str]]
# A text scorer sees only a returned string (e.g. a generated announcement) and
# returns failure strings — empty means the sample passed.
TextScorer = Callable[[str], list[str]]


@dataclass
class SampleResult:
    passed: bool
    fails: list[str]


@dataclass
class _Perf:
    """Running totals of model calls + tokens across a case's samples.

    Sourced from the real promptlog (``duration_ms`` per call + token usage
    stored in each response) — the same numbers prod records, not a harness
    stopwatch.  Printed per case so ``make eval`` shows wall time and tok/s.
    """

    calls: int = 0
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_chars: int = 0
    output_chars: int = 0

    def add(self, perf: PromptPerf) -> None:
        self.calls += perf.calls
        self.duration_ms += perf.duration_ms
        self.input_tokens += perf.input_tokens
        self.output_tokens += perf.output_tokens
        self.thinking_chars += perf.thinking_chars
        self.output_chars += perf.output_chars

    def report(self, case_id: str, samples: int) -> None:
        if not self.calls:
            return
        seconds = self.duration_ms / 1000
        # tok/s here is END-TO-END (output_tokens / full request wall, which
        # includes prompt processing) — NOT the model's raw decode rate.  For
        # true generation tok/s see the native probe in test_perf_probe.py.
        tokens_per_second = self.output_tokens / seconds if seconds else 0.0
        per_call_ms = self.duration_ms / self.calls
        # output_tokens bundles reasoning + visible; split it by the char ratio.
        share = self.thinking_chars / (self.thinking_chars + self.output_chars or 1)
        reasoning_tokens = round(self.output_tokens * share)
        print(
            f"\nPERF [{case_id}] {samples} samples · {self.calls} calls · "
            f"{seconds:.1f}s wall · {per_call_ms:.0f}ms/call · "
            f"{self.input_tokens} in / {self.output_tokens} out tok "
            f"({reasoning_tokens} reasoning, {share * 100:.0f}%) · "
            f"{tokens_per_second:.1f} end-to-end tok/s"
        )


def _real_model_config(
    make_config: Callable[..., Config], *, signal_api_url: str, db_path: str
) -> Config:
    """A test Config pointed at the real Ollama text + embedding models.

    Reads endpoint/model from the environment so the same suite runs on the
    host (localhost) and inside the penny container (host.docker.internal),
    falling back to local defaults.  ``signal_api_url`` binds to the sample's
    own mock server so samples never share a channel.
    """
    return make_config(
        signal_api_url=signal_api_url,
        llm_model=os.environ.get("LLM_MODEL", "gpt-oss:20b"),
        llm_api_url=os.environ.get("LLM_API_URL", "http://localhost:11434"),
        llm_embedding_model=os.environ.get("LLM_EMBEDDING_MODEL", "embeddinggemma"),
        db_path=db_path,
    )


def seed_user(db: Database) -> None:
    """Create the test user + register their Signal device.

    Each sample uses a fresh DB, so the ``test_user_info`` fixture (bound to one
    path) doesn't apply — seed the user explicitly after Penny builds the DB.
    """
    db.users.save_info(
        sender=TEST_SENDER,
        name="Test User",
        location="Seattle, WA",
        timezone="America/Los_Angeles",
        date_of_birth="1990-01-01",
    )
    db.devices.register(ChannelType.SIGNAL, TEST_SENDER, "Test Signal", is_default=True)


def seed_collection(
    db: Database,
    synth: SynthCollection,
    *,
    extraction_prompt: str | None = None,
    intent: str | None = None,
    interval: int | None = None,
    published: bool = False,
) -> None:
    """Create a synthetic collection + its entries (key = text before ' — ')."""
    db.memories.create_collection(
        synth.name,
        synth.description,
        Inclusion(synth.inclusion),
        RecallMode.RELEVANT,
        extraction_prompt=extraction_prompt,
        collector_interval_seconds=interval,
        intent=intent,
        published=published,
    )
    db.memory(synth.name).write(
        [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in synth.entries],
        author="user",
    )


def collection_names(db: Database) -> set[str]:
    """Every memory name currently in the DB — the pre-run snapshot for scorers."""
    return {memory.name for memory in db.memories.list_all()}


def new_collections(db: Database, before: set[str]) -> list[MemoryRow]:
    """Collections that didn't exist before the run — what the model created."""
    return [memory for memory in db.memories.list_all() if memory.name not in before]


def collection_entries(db: Database, name: str) -> dict[str, str]:
    """``{key: content}`` for every keyed entry in a collection — a snapshot a
    collector scorer compares before/after a cycle to detect writes/edits/deletes."""
    memory = db.memory(name)
    rows = memory.read_all() if memory is not None else []
    return {entry.key: entry.content for entry in rows if entry.key is not None}


def tool_was_called(db: Database, tool_name: str) -> bool:
    """Did the model actually invoke ``tool_name`` this run?

    Scans the persisted promptlog responses for a matching tool call — the real
    record of what the model did, not a harness-side spy.
    """
    return any(
        any(call.get("function", {}).get("name") == tool_name for call in _response_tool_calls(row))
        for row in db.messages.recent_prompts(limit=200)
    )


_NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)


def looks_numbered(text: str) -> bool:
    """True when ``text`` reads as a numbered list (≥2 lines like ``1.`` / ``2)``).

    Used by format contracts: a prompt the model follows reliably is a numbered
    instruction/tool-call recipe, not flowing prose.
    """
    return len(_NUMBERED_LINE.findall(text)) >= 2


def _response_tool_calls(prompt_log) -> list[dict]:
    response = json.loads(prompt_log.response) if prompt_log.response else {}
    choices = response.get("choices") or []
    if not choices:
        return []
    return choices[0].get("message", {}).get("tool_calls") or []


# A browse-less query returns this so a case can still exercise the graceful
# "nothing found" path; matched queries return their CannedPage text instead.
_NO_RESULTS_PAGE = (
    "Title: No results\nNo relevant results were found for this query. "
    "Try a different source or reword the query."
)


def install_browse(penny: Penny, pages: list[CannedPage]) -> None:
    """Replace the generic browse mock with query-aware canned pages.

    ``run_penny_with_server`` wires a single fixed ``"Mock search results"``
    string onto every browse call, which only lets a case check *whether* the
    model browsed.  A real tool-reasoning contract needs to score the model's
    *subsequent* call — did it extract the right fact/URL and chain to the
    correct next tool?  So a case seeds realistic pages, each keyed by a
    ``match`` substring.  A query the model issues becomes a URL (search →
    ``SEARCH_URL`` + ``quote(query)``; direct read → the URL itself), so a
    case-token substring matches both shapes, and a refined follow-up query
    maps to a different page — supporting multi-hop chains.  Installed on BOTH
    agents (chat + collector) since the generic mock sits on both.
    """

    async def request_fn(method: str, params: dict) -> tuple[str, str | None]:
        url = params.get("url", "").lower()
        for page in pages:
            if page.match.lower() in url:
                return page.text, page.image
        return _NO_RESULTS_PAGE, None

    def provider() -> tuple[Callable, MagicMock]:
        return request_fn, MagicMock(check_domain=AsyncMock())

    penny.chat_agent._browse_provider = provider
    penny.collector._browse_provider = provider


async def _embed_seeds(penny: Penny) -> None:
    """Vectorize seeded memory so stage-1/2 recall behaves like prod.

    Penny's startup backfill ran on the empty DB before we seeded; re-run it so
    seeded descriptions/entries get embeddings the recall path can match.
    """
    if penny.embedding_model_client is None:
        return
    await penny._backfill_memory_embeddings(_EMBED_BATCH)
    await penny._backfill_description_embeddings(_EMBED_BATCH)


def _assert_threshold(
    case_id: str, results: list[SampleResult], min_pass_rate: float | None
) -> None:
    """Print the case's X/Y pass rate, and — unless report-only — gate on it.

    ``min_pass_rate=None`` is report-only: the X/Y line and any per-sample
    failures print for insight, but the case never fails the run.  Use it for
    inherently stochastic behaviours we want to *observe* rather than gate (the
    self-correction cases — the model can't clear every cross-run repeat, and a
    flaky red adds no signal beyond the printed rate).
    """
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    failures = "\n".join(
        f"  [{i + 1}] {'; '.join(result.fails)}"
        for i, result in enumerate(results)
        if not result.passed
    )
    if min_pass_rate is None:
        print(f"\nRESULT [{case_id}] {passed}/{total} passed (report-only)")
        if failures:
            print(failures)
        return
    need = ceil(min_pass_rate * total)
    print(f"\nRESULT [{case_id}] {passed}/{total} passed (need >={need}, rate {min_pass_rate})")
    if passed < need:
        pytest.fail(f"{case_id}: {passed}/{total} passed (need >={need}):\n{failures}")


def _dump_thinking(db: Database, case_id: str, sample_index: int, *, failed: bool) -> None:
    """Print every LLM call's thinking + tool calls for one sample.

    Auto-dumps for any FAILED sample: the reason a prompt change didn't work
    almost always lives in the model's thinking, so an iteration loop must always
    surface it (pytest shows captured stdout for failed tests automatically, so
    these land in the failure report without needing ``-s``).  Set
    ``EVAL_DUMP_THINKING=1`` to additionally dump passing samples for full
    visibility.  Reads the ephemeral per-sample promptlog before the DB is
    discarded — the only place the model's reasoning survives (the eval DB is in
    a --rm container).
    """
    if not failed and not os.environ.get("EVAL_DUMP_THINKING"):
        return
    with Session(db.engine) as session:
        rows = session.exec(select(PromptLog).order_by(PromptLog.timestamp.asc())).all()
    print(f"\n===== THINKING [{case_id} #{sample_index}] — {len(rows)} LLM call(s) =====")
    for index, row in enumerate(rows, start=1):
        label = row.agent_name or row.prompt_type or "?"
        if row.thinking:
            print(f"[{index}:{label}] THINKING: {row.thinking.strip()}")
        for call in _response_tool_calls(row):
            function = call.get("function", {})
            print(f"[{index}:{label}] TOOL: {function.get('name')}({function.get('arguments')})")
    print("===== END THINKING =====\n")


# A chat-eval runner: (case_id, message, scorer, optional seeder) -> asserts threshold.
ChatEval = Callable[..., Awaitable[None]]


@pytest.fixture
def chat_eval(make_config: Callable[..., Config], tmp_path) -> ChatEval:
    """Drive the real chat flow N times for one user message and score each run.

    Each sample is fully hermetic — its own mock Signal server, DB, and
    real-model Penny: seed user (+ any case seed), embed the seeds, push the
    message, wait for the reply, then score persisted state.  A per-sample
    server is essential: a shared one leaks a prior sample's shut-down channel,
    which then errors on the next sample's broadcast.  A timeout counts as a
    failed sample, not a crash.
    """

    async def _run(
        *,
        case_id: str,
        message: str,
        score: Scorer,
        seed: Seeder | None = None,
        browse: list[CannedPage] | None = None,
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
        timeout: float = 120.0,
    ) -> None:
        results: list[SampleResult] = []
        perf = _Perf()
        for sample_index in range(samples):
            server = MockSignalServer()
            await server.start()
            try:
                config = _real_model_config(
                    make_config,
                    signal_api_url=f"http://localhost:{server.port}",
                    db_path=str(tmp_path / f"{case_id}-{sample_index}.db"),
                )
                async with run_penny_with_server(config, server) as penny:
                    seed_user(penny.db)
                    if seed is not None:
                        seed(penny.db)
                    await _embed_seeds(penny)
                    if browse is not None:
                        install_browse(penny, browse)
                    before = collection_names(penny.db)
                    try:
                        await server.push_message(sender=TEST_SENDER, content=message)
                        response = await server.wait_for_message(timeout=timeout)
                        reply = str(response.get("message", ""))
                        fails = score(penny.db, before, reply)
                        results.append(SampleResult(not fails, fails))
                    except TimeoutError:
                        results.append(SampleResult(False, ["no reply within timeout"]))
                    _dump_thinking(penny.db, case_id, sample_index, failed=not results[-1].passed)
                    perf.add(penny.db.messages.prompt_perf())
            finally:
                await server.stop()
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate)

    return _run


# A collector-eval runner: (case_id, collection, seed, score, snapshot) -> asserts.
CollectorEval = Callable[..., Awaitable[None]]


@pytest.fixture
def collector_eval(make_config: Callable[..., Config], tmp_path) -> CollectorEval:
    """Drive a real collector cycle (``run_for``) N times for one collection.

    Each sample is hermetic.  Seeds run first (the collection under test + any
    input logs/entries), embeddings backfill, then ``run_for`` executes the real
    cycle against the real model.  The scorer reads persisted state, the pre-cycle
    snapshot, and any messages the cycle sent the user (captured off the server).
    """

    async def _run(
        *,
        case_id: str,
        collection: str,
        seed: Seeder,
        score: CollectorScorer,
        snapshot: Snapshotter | None = None,
        browse: list[CannedPage] | None = None,
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
    ) -> None:
        results: list[SampleResult] = []
        perf = _Perf()
        for sample_index in range(samples):
            server = MockSignalServer()
            await server.start()
            try:
                config = _real_model_config(
                    make_config,
                    signal_api_url=f"http://localhost:{server.port}",
                    db_path=str(tmp_path / f"{case_id}-{sample_index}.db"),
                )
                async with run_penny_with_server(config, server) as penny:
                    seed_user(penny.db)
                    seed(penny.db)
                    await _embed_seeds(penny)
                    if browse is not None:
                        install_browse(penny, browse)
                    before = snapshot(penny.db) if snapshot is not None else None
                    sent_before = len(server.outgoing_messages)
                    await penny.collector.run_for(collection)
                    # A collector cycle ENQUEUES sends (send_queue) — the drainer
                    # that would deliver them to the channel is a separate schedule
                    # that doesn't run inside run_for.  So read sends off the queue,
                    # plus anything the drainer happened to deliver to the server.
                    sent = [item.content for item in penny.db.send_queue.pending_items()] + [
                        str(message.get("message", ""))
                        for message in server.outgoing_messages[sent_before:]
                    ]
                    fails = score(penny.db, before, sent)
                    results.append(SampleResult(not fails, fails))
                    _dump_thinking(penny.db, case_id, sample_index, failed=bool(fails))
                    perf.add(penny.db.messages.prompt_perf())
            finally:
                await server.stop()
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate)

    return _run


class _InjectingClient(LlmClient):
    """Base for the eval injectors that wrap a real ``LlmClient`` to force ONE bad
    response deterministically, then delegate every other call to the real model.

    Subclasses ``LlmClient`` (so it's assignable to ``collector._model_client``)
    but deliberately skips its ``__init__`` — it owns no real connection, only the
    wrapped client.  Holds ``bail_injected`` (a declared attribute, so callers read
    ``wrapper.bail_injected`` directly — no ``getattr`` probing); ``chat`` is
    overridden by subclasses and every other attribute (e.g. ``model``) forwards to
    the real client."""

    def __init__(self, real: LlmClient) -> None:
        self._real = real
        self.bail_injected = False

    async def chat(self, messages, tools=None, *args, **kwargs):
        raise NotImplementedError

    def __getattr__(self, name):
        return getattr(self._real, name)


class _InjectTextBail(_InjectingClient):
    """Injects ONE plain-text response right after the model's first tool call.

    This reproduces — deterministically, against the live model — a collector
    that narrates "Done." (or any prose) instead of continuing with / closing
    via a tool call.  The stochastic ~25% slip can't be reliably reproduced by
    seeding alone, so we force it once and let the production text-step nudge
    drive the recovery on the real model.  ``bail_injected`` records that the
    scenario actually fired (else the contract test would be vacuous).
    """

    def __init__(self, real, bail_text: str) -> None:
        super().__init__(real)
        self._bail_text = bail_text
        self._saw_tool = False

    async def chat(self, messages, tools=None, *args, **kwargs):
        if self._saw_tool and not self.bail_injected:
            self.bail_injected = True
            return LlmResponse(message=LlmMessage(role="assistant", content=self._bail_text))
        response = await self._real.chat(messages, *args, tools=tools, **kwargs)
        if response.has_tool_calls:
            self._saw_tool = True
        return response


# A nudge-eval runner: (collection, seed, bail_text) -> asserts recovery.
NudgeEval = Callable[..., Awaitable[None]]


@pytest.fixture
def nudge_eval(make_config: Callable[..., Config], tmp_path) -> NudgeEval:
    """Contract test for the collector text-step nudge.

    Drives a real collector cycle but forces one plain-text "bail" right after
    the model's first tool call (via ``_InjectTextBail``).  Without the nudge the
    loop would treat that text as a final answer and end the cycle with no
    ``done()`` — marked failed, cursor uncommitted.  With it, the model is nudged
    back to a tool call and the cycle must recover to a successful close.  Each
    sample asserts the bail actually fired AND the cycle recovered (``run_for``
    returned success); an optional ``score`` adds case-specific checks.
    """

    async def _run(
        *,
        case_id: str,
        collection: str,
        seed: Seeder,
        bail_text: str,
        score: CollectorScorer | None = None,
        snapshot: Snapshotter | None = None,
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
    ) -> None:
        results: list[SampleResult] = []
        perf = _Perf()
        for sample_index in range(samples):
            server = MockSignalServer()
            await server.start()
            try:
                config = _real_model_config(
                    make_config,
                    signal_api_url=f"http://localhost:{server.port}",
                    db_path=str(tmp_path / f"{case_id}-{sample_index}.db"),
                )
                async with run_penny_with_server(config, server) as penny:
                    seed_user(penny.db)
                    seed(penny.db)
                    await _embed_seeds(penny)
                    before = snapshot(penny.db) if snapshot is not None else None
                    sent_before = len(server.outgoing_messages)
                    wrapper = _InjectTextBail(penny.collector._model_client, bail_text)
                    penny.collector._model_client = wrapper
                    success, _ = await penny.collector.run_for(collection)
                    sent = [item.content for item in penny.db.send_queue.pending_items()] + [
                        str(message.get("message", ""))
                        for message in server.outgoing_messages[sent_before:]
                    ]
                    fails = list(score(penny.db, before, sent)) if score is not None else []
                    if not wrapper.bail_injected:
                        fails.append("forced text bail never fired — contract not exercised")
                    elif not success:
                        fails.append("cycle did not recover to a successful close after the nudge")
                    results.append(SampleResult(not fails, fails))
                    perf.add(penny.db.messages.prompt_perf())
            finally:
                await server.stop()
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate)

    return _run


class _InjectDoneBail(_InjectingClient):
    """Forces a ``done()`` tool call as the model's FIRST response — the first-move
    bail the premature-done guard must refuse.

    Reproduces, deterministically against the live model, a collector that opens
    with ``done(success=true, "no new matches")`` before reading anything.  Pre-fix
    that bail closes the cycle; post-fix the guard returns an error tool response
    and the real model must recover (read its inputs, then do the work).
    ``bail_injected`` records the scenario actually fired."""

    async def chat(self, messages, tools=None, *args, **kwargs):
        if not self.bail_injected:
            self.bail_injected = True
            return LlmResponse(
                message=LlmMessage(
                    role="assistant",
                    tool_calls=[
                        LlmToolCall(
                            id="bail-done",
                            function=LlmToolCallFunction(
                                name="done",
                                arguments={
                                    "success": True,
                                    "summary": "no new matches this cycle",
                                },
                            ),
                        )
                    ],
                )
            )
        return await self._real.chat(messages, *args, tools=tools, **kwargs)


class _InjectSendBail(_InjectingClient):
    """Injects ONE malformed ``send_message`` tool call right after the model's
    first real tool call.

    Reproduces a collector that emits a half-formed send (``"Hi there! ......???"``)
    mid-cycle.  Pre-fix the send gate let that shape through (the truncation regex
    missed it) and the user received junk; post-fix the gate refuses it with an
    error tool response and the model must resend a complete message.
    ``bail_injected`` records the scenario actually fired."""

    def __init__(self, real, junk: str) -> None:
        super().__init__(real)
        self._junk = junk
        self._saw_tool = False

    async def chat(self, messages, tools=None, *args, **kwargs):
        if self._saw_tool and not self.bail_injected:
            self.bail_injected = True
            return LlmResponse(
                message=LlmMessage(
                    role="assistant",
                    tool_calls=[
                        LlmToolCall(
                            id="bail-send",
                            function=LlmToolCallFunction(
                                name="send_message", arguments={"content": self._junk}
                            ),
                        )
                    ],
                )
            )
        response = await self._real.chat(messages, *args, tools=tools, **kwargs)
        if response.has_tool_calls:
            self._saw_tool = True
        return response


# A guard-recovery runner: (collection, seed, wrap_client, score) -> asserts recovery.
GuardRecoveryEval = Callable[..., Awaitable[None]]


@pytest.fixture
def guard_recovery_eval(make_config: Callable[..., Config], tmp_path) -> GuardRecoveryEval:
    """Contract test for a runtime guard that refuses a bad tool call.

    Drives a real collector cycle but forces one bad tool call via an injector
    (``wrap_client(real) -> injector`` with a ``bail_injected`` flag).  The guard
    must refuse it with an error tool response (not stop the cycle), and the live
    model must recover.  Each sample asserts the bail actually fired AND the
    case's ``score(db, sent) -> [fails]`` passed.  Mirrors ``nudge_eval`` but for
    the coherent-but-wrong tool-call path rather than the plain-text-bail path."""

    async def _run(
        *,
        case_id: str,
        collection: str,
        seed: Seeder,
        wrap_client: Callable[[object], _InjectingClient],
        score: Callable[[Database, list[str]], list[str]],
        browse: list[CannedPage] | None = None,
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
    ) -> None:
        results: list[SampleResult] = []
        perf = _Perf()
        for sample_index in range(samples):
            server = MockSignalServer()
            await server.start()
            try:
                config = _real_model_config(
                    make_config,
                    signal_api_url=f"http://localhost:{server.port}",
                    db_path=str(tmp_path / f"{case_id}-{sample_index}.db"),
                )
                async with run_penny_with_server(config, server) as penny:
                    seed_user(penny.db)
                    seed(penny.db)
                    await _embed_seeds(penny)
                    if browse is not None:
                        install_browse(penny, browse)
                    sent_before = len(server.outgoing_messages)
                    wrapper = wrap_client(penny.collector._model_client)
                    penny.collector._model_client = wrapper
                    await penny.collector.run_for(collection)
                    sent = [item.content for item in penny.db.send_queue.pending_items()] + [
                        str(message.get("message", ""))
                        for message in server.outgoing_messages[sent_before:]
                    ]
                    fails = list(score(penny.db, sent))
                    if not wrapper.bail_injected:
                        fails.append("forced bail never fired — contract not exercised")
                    results.append(SampleResult(not fails, fails))
                    _dump_thinking(penny.db, case_id, sample_index, failed=bool(fails))
                    perf.add(penny.db.messages.prompt_perf())
            finally:
                await server.stop()
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate)

    return _run


# A recall-eval runner: (seed, check) over a single deterministic pass.
RecallEval = Callable[..., Awaitable[None]]


@pytest.fixture
def recall_eval(make_config: Callable[..., Config], tmp_path) -> RecallEval:
    """Drive the REAL two-stage recall path once and check routing per message.

    Embeddings are deterministic, so this is a single pass (no sampling): build
    one real-model Penny, seed the collections, backfill embeddings, then for
    each message render ``ChatAgent._recall_section`` and run ``check`` against
    the rendered block.  ``check(recall_block, message) -> list[str]`` failures
    are aggregated and asserted to be empty.
    """

    async def _run(
        *, case_id: str, seed: Seeder, messages, check, min_pass_rate: float | None = 0.75
    ) -> None:
        server = MockSignalServer()
        await server.start()
        try:
            config = _real_model_config(
                make_config,
                signal_api_url=f"http://localhost:{server.port}",
                db_path=str(tmp_path / f"{case_id}.db"),
            )
            async with run_penny_with_server(config, server) as penny:
                seed_user(penny.db)
                seed(penny.db)
                await _embed_seeds(penny)
                limit = int(penny.config.runtime.RECALL_LIMIT)
                results: list[SampleResult] = []
                for message in messages:
                    recall_block = await penny.chat_agent._recall_section(
                        current_message=message.text,
                        conversation_history=list(message.history),
                        limit=limit,
                    )
                    fails = check(recall_block or "", message)
                    results.append(SampleResult(not fails, fails))
                _assert_threshold(case_id, results, min_pass_rate)
        finally:
            await server.stop()

    return _run


# A startup-eval runner: (case_id, commit_message, score) -> asserts threshold.
StartupEval = Callable[..., Awaitable[None]]


@pytest.fixture
def startup_eval(make_config: Callable[..., Config], tmp_path) -> StartupEval:
    """Drive the real startup-announcement prompt N times and score its text.

    ``get_restart_message`` transforms the latest commit (read from the
    ``GIT_COMMIT_MESSAGE`` env var, set at build time) into a casual one-line
    announcement — a single-shot generation prompt, no tools.  Each sample sets
    the env var to the case's commit, calls the real generator against the real
    model, and scores the returned string; the prior env value is restored.
    """

    async def _run(
        *,
        case_id: str,
        commit_message: str,
        score: TextScorer,
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
    ) -> None:
        results: list[SampleResult] = []
        perf = _Perf()
        for sample_index in range(samples):
            server = MockSignalServer()
            await server.start()
            try:
                config = _real_model_config(
                    make_config,
                    signal_api_url=f"http://localhost:{server.port}",
                    db_path=str(tmp_path / f"{case_id}-{sample_index}.db"),
                )
                async with run_penny_with_server(config, server) as penny:
                    seed_user(penny.db)
                    prior = os.environ.get("GIT_COMMIT_MESSAGE")
                    os.environ["GIT_COMMIT_MESSAGE"] = commit_message
                    try:
                        announcement = await get_restart_message(penny.model_client)
                    finally:
                        if prior is None:
                            os.environ.pop("GIT_COMMIT_MESSAGE", None)
                        else:
                            os.environ["GIT_COMMIT_MESSAGE"] = prior
                    fails = score(announcement)
                    results.append(SampleResult(not fails, fails))
                    perf.add(penny.db.messages.prompt_perf())
            finally:
                await server.stop()
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate)

    return _run
