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

import asyncio
import json
import logging
import os
import re
from collections import Counter
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from similarity.embeddings import deserialize_embedding
from sqlmodel import Session, col, select

from penny.config import Config
from penny.constants import ChannelType, PennyConstants
from penny.conversation_machine import (
    ConversationState,
    RoundFraming,
    StateClassifier,
    StateDecision,
    build_snapshot,
)
from penny.database import Database
from penny.database.memory import EntryInput, MemoryType
from penny.database.message_store import MessageStore, PromptPerf
from penny.database.models import MemoryRow, PromptLog, SendQueueItem
from penny.database.skill_store import steps_from_json
from penny.database.skills import (
    DistillInput,
    SkillDraft,
    SkillParameter,
    SkillStep,
    SkillSubKind,
    build_binding_content,
    derive_collection_name,
    distill_steps,
    render_spoken_turns,
)
from penny.llm.client import LlmClient
from penny.llm.models import (
    LlmMessage,
    LlmResponse,
    LlmToolCall,
    LlmToolCallFunction,
    strip_harmony_control_tokens,
)
from penny.llm.similarity import embed_text
from penny.penny import Penny
from penny.responses import PennyResponse
from penny.skill_extraction import build_framing_content, build_naming_content
from penny.startup import get_restart_message
from penny.tests.conftest import TEST_SENDER, require_memory, run_penny_with_server
from penny.tests.eval.utils import artifacts as eval_artifacts
from penny.tests.eval.utils import assertions as eval_assertions
from penny.tests.eval.utils import cohort as eval_cohort
from penny.tests.eval.utils import report, run_health
from penny.tests.eval.utils.artifacts import FailureCause
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.baseline import Baseline, baseline_from_env
from penny.tests.eval.utils.fixtures import ENACTING_TOOLS, CannedPage, SynthCollection
from penny.tests.eval.utils.worlds import World
from penny.tests.mocks.signal_server import MockSignalServer
from penny.text_validity import (
    has_leaked_harmony_envelope,
    is_call_fragment_reply,
    is_degenerate_run,
    is_degenerate_tool_name,
)
from penny.tools.base import RESULT_TAG
from penny.tools.browse import BrowseChannelUnavailableError
from penny.tools.micro_context import (
    FramedParameter,
    MicroContext,
    MicroContextResult,
    MicroExtractOutcome,
    MissingParameters,
    SkillBinding,
    SkillLabels,
    SkillSignature,
    StateDrawOutcome,
    slug_parameter_name,
    spoken_form,
)
from penny.tools.skill_tools import render_skill_shape

# Samples per case.  Override with EVAL_SAMPLES=2 for a quick smoke run.
SAMPLES = int(os.environ.get("EVAL_SAMPLES", "5"))

# How many of a case's samples are driven at once.  Default 1 — strictly sequential, the
# order every report so far was produced in.  Raise it when the model is REMOTE: a hosted
# endpoint serves concurrent samples, where the single local GPU serialises them however
# many are asked for, so concurrency buys wall-clock there and nothing at home.
EVAL_CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "1"))

# How long a sample's channel has to connect.  Every concurrent sample boots its own
# Penny — 100-odd migrations, connectivity validation, preflight — and they contend for
# the same cores, so the wait is dominated by how many are booting rather than by whether
# this one will.  Measured: ~3.3s with 10 samples in flight, past the shared 10s default
# with 40, where five of eight cases died at startup having run nothing.  Generous rather
# than tuned: waiting longer costs nothing when the channel connects promptly, and a
# genuinely stuck sample still ends rather than hanging the run.
SAMPLE_READY_TIMEOUT_SECONDS = 60.0

# Embedding backfill batch size for seeded memory.
_EMBED_BATCH = 100


# ── Run health: the run's own account of itself (#1996) ──────────────────────
# A run used to report only what its cases scored, so a run that scored six cases out of
# a cohort that mostly never ran printed "6 passed, EXIT=0". These three hooks are where
# the run says how much of itself actually happened. They are hooks rather than a fixture
# because the answer is a property of the SESSION, and under xdist the session spans
# several processes: each writes its own record, and the one holding the terminal reader
# adds them up.
def pytest_configure(config) -> None:
    """Start counting this process's model calls before any sample runs."""
    run_health.begin_run()


def pytest_sessionfinish(session, exitstatus) -> None:
    """Write this process's health record, then — in the reader — settle the verdict.

    An xdist WORKER only writes: it has part of the run and no terminal to report on. The
    reader (the controller, or a single process) writes its own record if it ran anything,
    reads every record in the run's health dir, and REFUSES a run whose cohorts mostly did
    not run — by moving the exit status, since a run that measured a fraction of what it
    intended must not exit 0 however its surviving samples scored.
    """
    worker = os.environ.get(eval_artifacts.XDIST_WORKER_ENV)
    health = run_health.process_health()
    directory = run_health.health_dir()
    if directory is not None and (worker or health.cohorts):
        run_health.write_health(directory, worker, health)
    if worker:
        return
    whole_run = run_health.load_health(directory) if directory is not None else health
    run_health.hold_run_health(whole_run)
    if not whole_run.viable and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter) -> None:
    """Print the run-health block at the head of the summary a reader actually reads.

    Rendered here rather than in ``pytest_sessionfinish`` so it lands in the summary area
    whether or not the run was given ``-s``: a health block a captured run swallows is the
    silence this exists to end.
    """
    health = run_health.held_run_health()
    if health is None or not health.cohorts:
        return
    terminalreporter.write_sep("=", "run health")
    terminalreporter.write_line(health.render())


# A chat scorer reads persisted DB state (the pre-run collection names + the
# final reply text) and returns failure strings — empty means the sample passed.
# A chat scorer returns either failure strings (binary: empty = pass) or a list of graded
# ``Check``s (partial credit: the sample scores passed/total).  Both flow through the same
# runner, which grades by the returned type.
Scorer = Callable[[Database, set[str], str], "list[str] | list[Check]"]
Seeder = Callable[[Database], None]
# A preparer mutates the constructed Penny before the message is pushed — e.g.
# to mock an external boundary (the image client) the case exercises.
Preparer = Callable[[Penny], None]
# A collector scorer also sees the pre-cycle snapshot and the messages the cycle
# sent the user.  ``snapshot`` is whatever the case's ``snapshot`` callback returned.
Snapshotter = Callable[[Database], object]
CollectorScorer = Callable[[Database, object, list[str]], "list[str] | list[Check]"]
# A text scorer sees only a returned string (e.g. a generated announcement) and
# returns either failure strings (binary: empty = pass) or a list of graded ``Check``s
# (partial credit) — the same dual return as the other scorer types, dispatched by the runner.
TextScorer = Callable[[str], "list[str] | list[Check]"]


@dataclass
class Check:
    """One graded expectation of a sample — an expected tool call or an outcome.

    A scorer can return a list of these instead of a list of failure strings; the sample
    then scores as (checks that passed) / (checks that applied) — partial credit — instead of
    all-or-nothing.  ``label`` names the expectation so the report shows exactly which
    check missed (e.g. "turn-1 memory_metadata called").

    ``scored=False`` marks an ADVISORY check — flavour: it renders in the report
    (✅/❌ beside its row or in the footer) but is excluded from the sample's score.
    The state-is-core doctrine uses this split: end DB state is the pass/fail;
    call-sequencing checks annotate how the state came to be.

    ``rationale`` is the optional observed-vs-expected note rendered beside the outcome
    ("expected 3 reads, saw 1"), so a ❌ is never bare.  ``ignored`` is the NOT-APPLICABLE
    third state — this sample's branch never exercised the check — excluded from the graded
    denominator (counts as neither pass nor fail), yet still rendered (as ➖) so a skipped
    expectation reads as skipped, not forgotten.  Build one with ``Check.na(...)``."""

    label: str
    ok: bool
    anchor: str | None = None  # substring of the transcript row this check marks (None = no row)
    scored: bool = True  # False = advisory flavour, visible in the report, not in the score
    rationale: str | None = None  # observed-vs-expected note rendered beside the outcome
    ignored: bool = False  # not-applicable: rendered (➖) but out of the graded denominator
    kind: str | None = None  # class label rendered `[kind]` (spine/reply/state/proc/guard)

    @classmethod
    def na(
        cls,
        label: str,
        *,
        rationale: str | None = None,
        anchor: str | None = None,
        kind: str | None = None,
    ) -> Check:
        """A not-applicable check — this sample's branch didn't run, so it's excluded from the
        graded denominator (neither pass nor fail).  Still rendered (➖) so a skipped expectation
        reads as skipped, not forgotten.  ``kind`` carries the same ``[class]`` tag as a scored
        check, so an n/a row reads ``C3 [state] …`` in its class like any other."""
        return cls(
            label=label, ok=True, anchor=anchor, rationale=rationale, ignored=True, kind=kind
        )


@dataclass
class SampleResult:
    """A sample's score in [0, 1] + the labels of whatever didn't pass (for the report).

    Binary scoring is the degenerate one-check case (score 1.0 or 0.0); graded scoring
    (a scorer returning ``Check``s) is passed/total.  A case's metric is the MEAN of its
    sample scores — identical to the old pass-rate when every sample is binary, but with
    partial credit when a scorer grades."""

    score: float
    failed: list[str]
    total: int = 1
    checks: list[Check] = field(default_factory=list)  # full graded checks (empty = binary)
    # The structural failure cause (#1695), stamped by the runner after scoring: ``None`` for
    # a pass; ``behavioral`` / ``pathology`` / ``harness`` for a failure.  The artifact aggregate
    # defaults an unstamped failure to behavioral, so a directly-constructed result is safe.
    cause: FailureCause | None = None
    # Passed-but-shaky (#1725, #1694): the sample passed only after the loop refused/recovered a
    # tool call.  Stamped by ``_write_sample_report`` (same ``EVAL_REPORT_DIR`` gate as the artifact
    # write) so it rides into the ``CaseArtifact.sample_fragile`` list the assembler reads.
    fragile: bool = False
    # What this sample LEFT BEHIND, read while its database was still live (#1995) — the whole
    # of what a ported case asserts or measures.  ``None`` for every case that has not been
    # ported, so nothing changes for one still passing a scorer.
    observation: eval_cohort.SampleObservation | None = None

    @property
    def passed(self) -> bool:
        return self.score >= 1.0

    @classmethod
    def binary(cls, fails: list[str]) -> SampleResult:
        return cls(0.0 if fails else 1.0, list(fails), 1)

    def adopt(self, checks: list[Check]) -> None:
        """Take on the claims the COHORT answered for this sample (#1995).

        A ported case is graded from claims made over the whole cohort AFTER the drive, so the
        sample's own score is settled here rather than at drive time.  Re-derived through
        ``graded`` so one definition decides what counts, whatever produced the checks."""
        graded = SampleResult.graded(checks)
        self.score, self.failed, self.total, self.checks = (
            graded.score,
            graded.failed,
            graded.total,
            graded.checks,
        )

    @classmethod
    def graded(cls, checks: list[Check]) -> SampleResult:
        if not checks:
            return cls(1.0, [], 1)
        # NOT-APPLICABLE checks (``ignored``) never count; among the rest, score over the
        # SCORED ones only (advisory flavour renders but doesn't count), with an all-advisory
        # list degenerating to scoring everything applicable.  Every check applied to this
        # sample was ignored → a vacuous pass (nothing to grade).
        applicable = [check for check in checks if not check.ignored]
        scored = [check for check in applicable if check.scored] or applicable
        if not scored:
            return cls(1.0, [], 0, list(checks))
        passed = sum(1 for check in scored if check.ok)
        failed = [_check_failure_label(check) for check in applicable if not check.ok]
        return cls(passed / len(scored), failed, len(scored), list(checks))


def _check_failure_label(check: Check) -> str:
    """A failed check's line for the RESULT-line per-sample detail: its label, plus the
    observed-vs-expected rationale when one was given (so it reads "label — expected 3, saw 1"
    instead of a bare label)."""
    return f"{check.label} — {check.rationale}" if check.rationale else check.label


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
    reasoning_tokens: int = 0

    def add(self, perf: PromptPerf) -> None:
        self.calls += perf.calls
        self.duration_ms += perf.duration_ms
        self.input_tokens += perf.input_tokens
        self.output_tokens += perf.output_tokens
        self.thinking_chars += perf.thinking_chars
        self.output_chars += perf.output_chars
        self.reasoning_tokens += perf.reasoning_tokens

    def report(self, case_id: str, samples: int) -> None:
        if not self.calls:
            return
        seconds = self.duration_ms / 1000
        # tok/s here is END-TO-END (output_tokens / full request wall, which
        # includes prompt processing) — NOT the model's raw decode rate.  For
        # true generation tok/s see the native probe in test_perf_probe.py.
        tokens_per_second = self.output_tokens / seconds if seconds else 0.0
        per_call_ms = self.duration_ms / self.calls
        # output_tokens bundles reasoning + visible.  The provider's own count is a READ;
        # only where it reports none do we fall back to splitting by the character ratio —
        # and the line says which, because an estimate shown as a measurement is how two
        # models get compared on a number one of them never supplied.
        if self.reasoning_tokens:
            reasoning_tokens, source = self.reasoning_tokens, "reported"
        else:
            share = self.thinking_chars / (self.thinking_chars + self.output_chars or 1)
            reasoning_tokens, source = round(self.output_tokens * share), "estimated"
        reasoning_share = reasoning_tokens / self.output_tokens if self.output_tokens else 0.0
        print(
            f"\nPERF [{case_id}] {samples} samples · {self.calls} calls · "
            f"{seconds:.1f}s wall · {per_call_ms:.0f}ms/call · "
            f"{self.input_tokens} in / {self.output_tokens} out tok "
            f"({reasoning_tokens} reasoning {source}, {reasoning_share * 100:.0f}%) · "
            f"{tokens_per_second:.1f} end-to-end tok/s"
        )


# Endpoint settings a REMOTE OpenAI-compatible provider needs and a local Ollama does
# not: an API key that is actually checked, and a separate endpoint for the embedding
# model — no remote chat provider serves ``embeddinggemma``, so sending the chat model
# somewhere else must not drag memory's embeddings along with it.  Each name is its
# ``Config`` field uppercased, so a var present here overrides exactly that field and an
# absent one leaves the field's own default standing — which is why the local path is
# unchanged by this list existing.
_ENDPOINT_ENV_VARS = (
    "LLM_API_KEY",
    "LLM_PROVIDER",
    "LLM_EMBEDDING_API_URL",
    "LLM_EMBEDDING_API_KEY",
)

# A remote endpoint puts a network between the sample and the model, where a transient
# timeout or 502 is ordinary rather than a defect.  The integration-test default of a
# single attempt would spend a whole SAMPLE reroll on one, so the eval path retries the
# CALL first, and waits long enough between tries for a rate limiter to reopen.
_EVAL_LLM_MAX_RETRIES = 3
_EVAL_LLM_RETRY_DELAY = 1.0

# The per-request deadline.  Without one the OpenAI SDK's own default applies — a 600s
# read — so a remote endpoint that simply STOPS ANSWERING is not a failure the retry loop
# can see: the request never errors, it just sits, and the harness's own turn timeout ends
# the sample ~2 minutes later with nothing recorded anywhere.  Measured, that is exactly
# what happened to 3 of 20 samples.  A deadline turns the stall into a timeout the client
# retries and, failing that, into an aborted run the harness RE-DRIVES from a clean world.
# The budget is what makes it work: attempts x deadline (plus backoff) must fit inside the
# runner's turn timeout, or the turn dies first and the retries never run.
_EVAL_LLM_TIMEOUT_DEFAULT_SECONDS = 20.0


def env_seconds(name: str, default: float) -> float:
    """A duration read from the environment, where UNSET and EMPTY mean the same thing.

    ``make eval`` forwards its whole variable list explicitly, so a variable the caller
    did not set arrives as an EMPTY STRING rather than absent — and ``float("")`` raises
    at IMPORT time, which takes the entire eval suite down before a single sample runs
    (measured: two runs died in 7 seconds this way).  Reading the default for both is what
    makes "forwarded but unset" and "not forwarded" the same thing they look like.
    """
    return float(os.environ.get(name) or default)


_EVAL_LLM_TIMEOUT = env_seconds("LLM_TIMEOUT", _EVAL_LLM_TIMEOUT_DEFAULT_SECONDS)


def _endpoint_overrides() -> dict[str, str]:
    """The endpoint settings present in the environment, keyed by ``Config`` field.

    An unset var is OMITTED rather than passed as empty, so ``Config``'s own default
    stays in place: absent ``LLM_EMBEDDING_API_URL`` keeps embeddings on the chat
    endpoint (what a local Ollama run wants), and absent ``LLM_API_KEY`` keeps the
    "not-needed" key local inference servers accept.
    """
    return {name.lower(): value for name in _ENDPOINT_ENV_VARS if (value := os.environ.get(name))}


def _real_model_config(
    make_config: Callable[..., Config], *, signal_api_url: str, db_path: str, model: str = ""
) -> Config:
    """A test Config pointed at the real text + embedding models.

    Reads endpoint/model from the environment so the same suite runs on the
    host (localhost), inside the penny container (host.docker.internal), and
    against a remote OpenAI-compatible provider (e.g. OpenRouter), falling back
    to local defaults.  ``signal_api_url`` binds to the sample's own mock server
    so samples never share a channel.
    """
    return make_config(
        signal_api_url=signal_api_url,
        # A ported case is parametrized over MODEL, so a cohort names the one it is measured
        # on; everything else takes the run's configured model exactly as before.
        llm_model=model or os.environ.get("LLM_MODEL", "gpt-oss:20b"),
        llm_api_url=os.environ.get("LLM_API_URL", "http://localhost:11434"),
        llm_embedding_model=os.environ.get("LLM_EMBEDDING_MODEL", "embeddinggemma"),
        llm_max_retries=_EVAL_LLM_MAX_RETRIES,
        llm_retry_delay=_EVAL_LLM_RETRY_DELAY,
        llm_timeout=_EVAL_LLM_TIMEOUT,
        db_path=db_path,
        **_endpoint_overrides(),
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
    schedule: str | None = None,
    notify: bool = False,
) -> None:
    """Create a synthetic collection + its entries (key = text before ' — ')."""
    db.memories.create_collection(
        synth.name,
        synth.description,
        extraction_prompt=extraction_prompt,
        schedule=schedule,
        notify=notify,
    )
    require_memory(db, synth.name).write(
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


# ── The seeded ledger (#1846) ─────────────────────────────────────────────────
#
# A case may lay down the promptlog of turns that happened BEFORE the one under test, so
# the sample answers its message against the state those turns really left behind rather
# than against a hand-built shell of it.  Those rows are history, not this sample's work:
# every reader of "what did the model do" excludes them, or the apply case's "no browse
# this turn" would read the demonstrated round's browse as the live turn's and the report
# would render a prior turn's calls as if this one made them.
#
# ONE mechanism, keyed on the run id: the seeder mints its run ids under this prefix and
# the two fetch chokepoints below drop them.  Keying on the run (rather than on a
# timestamp watermark or a reserved agent name) is what lets the seeded rows be otherwise
# INDISTINGUISHABLE from production's — same agent names, same prompt types, same
# response shapes — which is the whole point of seeding them.  A reader inherits the
# exclusion by using ``live_prompts`` / ``live_prompt_rows``, not by remembering a rule.
SEEDED_RUN_PREFIX = "seeded-"


def seeded_run_id(name: str) -> str:
    """A run id for a seeded prior turn — deterministic (so a probe can assert against it
    by name) and structurally distinguishable from a live ``uuid4().hex``."""
    return f"{SEEDED_RUN_PREFIX}{name}"


def is_seeded_run(run_id: str | None) -> bool:
    """Whether a promptlog row belongs to a seeded prior turn rather than this sample."""
    return run_id is not None and run_id.startswith(SEEDED_RUN_PREFIX)


def live_prompts(db: Database, limit: int = 200) -> list[PromptLog]:
    """This sample's OWN promptlog rows, newest first — the recent window minus any
    seeded prior turn.  THE fetch behind every "what did the model do this sample"
    reader."""
    return [row for row in db.messages.recent_prompts(limit) if not is_seeded_run(row.run_id)]


# A sample's own window — every promptlog row it could have written.  Deliberately wider
# than the 200-row reader window: a seeded ledger sits in the same table, so the perf read
# must see past it to the sample's own rows.
_PERF_WINDOW = 1000


def live_prompt_perf(db: Database) -> PromptPerf:
    """``prompt_perf`` over this sample's own rows — the same aggregate, minus the seeded
    ledger, so a report's banner never counts a prior turn's calls as this sample's.

    The per-row arithmetic is the store's own (its two response readers, called rather
    than restated): a second copy of "how a response's usage is read" would be a second
    contract to drift."""
    rows = [
        row for row in db.messages.recent_prompts(_PERF_WINDOW) if not is_seeded_run(row.run_id)
    ]
    responses = [json.loads(row.response) if row.response else {} for row in rows]
    usage = [MessageStore._extract_token_usage(response) for response in responses]
    return PromptPerf(
        len(rows),
        sum(row.duration_ms or 0 for row in rows),
        sum(prompt_tokens for prompt_tokens, _ in usage),
        sum(completion_tokens for _, completion_tokens in usage),
        sum(len(row.thinking or "") for row in rows),
        sum(len(MessageStore._extract_content(response)) for response in responses),
        sum(MessageStore._extract_reasoning_tokens(response) for response in responses),
    )


def tool_was_called(db: Database, tool_name: str) -> bool:
    """Did the model actually invoke ``tool_name`` this run?

    Scans the persisted promptlog responses for a matching tool call — the real
    record of what the model did, not a harness-side spy.
    """
    return any(
        any(tool_call_name(call) == tool_name for call in _response_tool_calls(row))
        for row in live_prompts(db)
    )


def tool_not_called(db: Database, tool_name: str) -> bool:
    """The negative-constraint counterpart to ``tool_was_called``: True when the model did NOT
    invoke ``tool_name`` this run.  Lets a scorer state an avoided-action expectation directly —
    ``Check("no write on a discuss turn", tool_not_called(db, "collection_write"))`` — instead of
    hand-negating ``tool_was_called`` at each call site."""
    return not tool_was_called(db, tool_name)


def count_tool_calls(db: Database, tool_name: str) -> int:
    """How many times the model invoked ``tool_name`` this run.

    Sourced from the persisted promptlog (the real record of what the model did).
    Used to detect retry-flailing: after a channel-outage banner, a healthy cycle
    issues at most one ``browse`` call (the probe that revealed the outage) and
    then stops — repeated browse calls are the doomed URL-variant retries the
    outage banner is meant to end."""
    return sum(
        1
        for row in live_prompts(db)
        for call in _response_tool_calls(row)
        if tool_call_name(call) == tool_name
    )


_GAVE_UP = re.compile(
    r"\b(sorry|apolog\w+)\b.{0,50}"
    r"\b(wasn't|was not|couldn't|could not|can't|cannot|unable|not able)\b",
    re.IGNORECASE,
)


def _row_turns(row: PromptLog) -> list[dict]:
    """One promptlog row's conversation: the turns it CARRIED (``messages``) followed by the
    turns the run appended after it (``trailing_messages`` — the tail no later call carried,
    #1778).

    The tail is the same wire shape, so the two concatenate rather than merge.  Without it a
    run that ended immediately after executing a tool (``max_steps`` on a tool step, a
    write-gate STOP, a reroll abort, an exception) recorded its terminal call and result
    NOWHERE — a transcript that stops mid-run is indistinguishable from one where nothing
    happened, which misrepresents the run it claims to record.  Empty on every row but a run's
    last, so an ordinary run reads byte-identically to before."""
    carried = json.loads(row.messages) if row.messages else []
    return [*carried, *row.get_trailing_messages()]


def _iter_prompt_messages(db: Database):
    """Every message across the run's promptlog (accumulated history + tool results, incl. the
    trailing tail no model call carried — so a terminal rejection is visible, #1778)."""
    for row in live_prompts(db):
        yield from _row_turns(row)


# Tool-result fragments that mean a call the model made was refused.  ``tool_call_rejected``
# reads the two failure-narration frames (tools/base.py: the generic failure + arg-validation);
# ``_RECOVERY_FRAMES`` widens that to the framework REFUSAL narrations too (a call rejected
# before it ran, a duplicate not repeated, a missing / timed-out / errored tool) — the "did the
# run recover from something?" set the fragile-pass flag reads.
_REJECTION_FRAMES = ("arguments were wrong", "didn't work")
_RECOVERY_FRAMES = (
    *_REJECTION_FRAMES,
    "rejected before it could run",  # Prompt.REJECTED_CALL_NARRATION (e.g. a premature done())
    "wasn't repeated",  # Prompt.DUPLICATE_CALL_NARRATION
    "there's no such tool",  # FRAMEWORK_NARRATION_NOT_FOUND
    "it timed out",  # FRAMEWORK_NARRATION_TIMEOUT
    "it errored",  # FRAMEWORK_NARRATION_EXCEPTION
)


def _frame_attributes_to(content: str, tool_name: str) -> bool:
    """Does this framed tool-result name ``tool_name`` as the tool that produced it?

    ``Tool.format_result`` (``penny/tools/base.py``) wraps EVERY result as
    ``<narration> (<tool> result)\\n<body>`` — one narration line plus the retained
    ``(<tool> result)`` machine tag.  A call attributes to its tool through EITHER of
    two shapes, and both must be recognised:

    * the **backticked tool name** in the narration — the generic frame
      (``You tried to use `browse` but it didn't work:``) and the framework-synthesised
      failures (arg-validation / timeout / not-found), which lead with `` `<tool>` ``; and
    * the **parenthesized result tag** ``(<tool> result)`` — the SOLE attribution when
      the narration backticks the *target* instead of the tool, which is the whole
      memory-tool execute-time-failure family (``You tried to update `<collection>`'s
      settings but it didn't work: (collection_set result)``, ``You tried to save to
      `<collection>` but it didn't work: (collection_write result)``, …).  There the
      tool name never appears backticked, so matching only `` `<tool>` `` misses it —
      the latent false-green this fixes (#1726).
    """
    return f"`{tool_name}`" in content or RESULT_TAG.format(tool_name=tool_name) in content


def tool_call_rejected(db: Database, tool_name: str | None = None) -> bool:
    """Did a call to ``tool_name`` — or ANY tool, when ``tool_name`` is None — come back REJECTED
    (arg-validation / failure)?

    The process-fidelity counterpart to ``tool_was_called``: a graded contract that checks
    the final STATE can still pass when an intermediate call was rejected and a *later* turn
    happened to re-land the content — this catches the rejected turn (the tool-result failure
    frame).  Attribution matches BOTH narration shapes ``Tool.format_result`` emits — the
    backticked tool name AND the ``(<tool> result)`` tag (``_frame_attributes_to``) — so a
    memory-tool rejection whose narration backticks the *target* (``collection_set`` /
    ``collection_write`` / …) is no longer invisible to a per-tool probe (#1726).  With no
    ``tool_name`` it's the run-wide "was any tool refused?" probe."""
    for message in _iter_prompt_messages(db):
        content = message.get("content") or ""
        if message.get("role") != "tool":
            continue
        if tool_name is not None and not _frame_attributes_to(content, tool_name):
            continue
        if any(frame in content for frame in _REJECTION_FRAMES):
            return True
    return False


def sample_is_fragile(db: Database) -> bool:
    """Did the run reach its result SHAKILY — through a rejected / refused / recovered tool call,
    OR a framework RECOVERY NUDGE (a continue / parse-failure / tool-call-demand user turn)?

    Scans the persisted promptlog for either recovery shape: a tool-result failure / framework
    refusal frame (``_RECOVERY_FRAMES``, on a ``tool`` turn) OR a recovery nudge injected as a
    ``user`` turn (``_is_nudge`` — the SAME predicate the transcript render marks ``⚠ recovery
    event`` with, single-sourced so render and probe can't drift apart again, #1735 finding 2).
    A green sample that only got there after the loop refused a call and retried, or after a nudge
    recovered an empty / unparseable response, is 'passed, fragile' in the report: real, but not
    robust — exactly the robustness signal the report cares about.  Derived from the same promptlog
    primitives as ``tool_call_rejected`` / ``_is_nudge``, not a new model judgment.  Fragile is
    render/artifact-only — never gated — so widening it moves no threshold.

    Unlike ``tool_call_rejected`` the tool-turn leg filters on NO tool name — it asks "did the run
    recover from *anything*?" — so it carries none of that probe's target-vs-tool-name attribution
    gap (#1726): a memory-tool execute-time failure narrates ``… but it didn't work:``, whose
    ``didn't work`` fragment is already in ``_RECOVERY_FRAMES``, caught regardless of which tool
    (target-backticked) produced it."""
    for message in _iter_prompt_messages(db):
        content = message.get("content") or ""
        role = message.get("role")
        if role == "tool" and any(frame in content for frame in _RECOVERY_FRAMES):
            return True
        if role == "user" and _is_nudge(content):
            return True
    return False


def _response_text(prompt_log) -> str:
    """The visible text content of a persisted model response (``choices[0].message.content``)."""
    response = json.loads(prompt_log.response) if prompt_log.response else {}
    choices = response.get("choices") or []
    return (choices[0].get("message", {}).get("content") or "") if choices else ""


def _response_is_poison(prompt_log) -> bool:
    """Did THIS persisted model response trip the agent-loop reroll guard — a punctuation
    collapse, a leaked Harmony envelope, a collapse-shaped tool NAME, or a bare call-fragment
    reply (incl. the bare ``{}`` empty-object reply the #1731 nudge-loop spiral terminated in,
    #1732)?  Mirrors ``Agent._unusable_output_condition`` over the persisted OUTPUT: the text
    content, each serialised tool-call argument, and each tool-call name — the SAME shared
    ``is_call_fragment_reply`` the live guard uses, so scan and guard can't drift."""
    calls = _response_tool_calls(prompt_log)
    parts = [_response_text(prompt_log)]
    for call in calls:
        function = call.get("function", {})
        name = function.get("name")
        if isinstance(name, str) and is_degenerate_tool_name(name):
            return True
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            parts.append(arguments)
    if any(has_leaked_harmony_envelope(part) for part in parts):
        return True
    if not calls and is_call_fragment_reply(_response_text(prompt_log)):
        return True
    return any(is_degenerate_run(part) for part in parts)


def run_exhibited_pathology(db: Database) -> bool:
    """Did the model produce reroll-guard POISON this run — the structural ``pathology`` signal
    for the failure-cause partition (#1695)?

    Scans the persisted promptlog's RESPONSE fields (the model's own OUTPUT) with the SAME
    ``text_validity`` detectors the agent-loop reroll guard runs live
    (``Agent._unusable_output_condition``): a punctuation collapse (``DEGENERATE_OUTPUT``), a
    leaked Harmony envelope (``TOOL_CALL_LEAK``), a collapse-shaped tool name, or a bare
    call-fragment reply (a fragment object, or the bare ``{}`` a nudge-loop spiral ends in —
    #1732).  Reading only the ``response`` (never the input ``messages``) is what
    makes this immune to a DELIBERATELY-injected recovery trigger: an ``_Inject*`` bail is
    returned as a SYNTHETIC ``LlmResponse`` that bypasses the persisting real client, so it
    never lands in a persisted ``response`` — a ``bail_injected`` sample is tagged pathology
    only if the LIVE model additionally produced its own poison, never for the forced trigger.

    **The nudge-frame boundary (#1732).** Repeated recovery-nudge frames are DELIBERATELY not
    counted as a pathology signal: a nudge is an INPUT message, and reading input would forfeit
    the injection-immunity above (an injected-recovery case produces exactly one live nudge by
    design, so naive nudge-counting would false-tag its fail path as pathology) and would need
    an arbitrary count threshold.  A nudge loop is a *symptom* whose *cause* is the model's own
    fragment OUTPUT — the terminal ``{}`` / call-fragment reply the scan already catches on the
    ``response`` — so classifying on that output tags the #1731 spiral pathology at the root
    while the output-only immunity holds.  (A spiral whose persisted output stays genuinely
    clean has no poison to tag and reads harness/behavioral — correctly: no pathology fired.)"""
    return any(_response_is_poison(row) for row in live_prompts(db))


def _stamp_cause(db: Database, result: SampleResult, *, timed_out: bool = False) -> None:
    """Stamp the sample's structural failure cause (#1695) in place — ``None`` for a pass.

    Scans for the pathology signal only when the sample actually failed (a pass carries no
    cause, so the scan is skipped).  Called at every runner's per-sample append site so the
    cause rides into the ``results.jsonl`` record and the RESULT-line cause tally."""
    result.cause = eval_artifacts.classify_cause(
        passed=result.passed,
        timed_out=timed_out,
        pathology=not result.passed and run_exhibited_pathology(db),
    )


# ── Graded-scorer dispatch + framework guard-as-Check (the runners' scoring seam) ──
def _scorer_is_graded(scored: list[Check | str]) -> bool:
    """Did the scorer return graded ``Check``s (partial credit) rather than binary failure
    strings?  The runners dispatch on this: a graded return scores as passed/total with the
    framework guard Checks prepended, a binary one keeps the all-or-nothing string path."""
    return bool(scored) and isinstance(scored[0], Check)


def _guarded_graded(scored: list[Check | str], guards: list[Check]) -> SampleResult:
    """A graded sample result with the runner's framework guard Checks PREPENDED (guard-as-Check):
    a recovery runner's 'the injected bail fired' / 'the cycle recovered' contract rides as a
    scored ``Check`` a scorer author can't omit, so a vacuous run — the injected trigger never
    fired — can't score green off the scorer's own checks alone."""
    checks = [check for check in scored if isinstance(check, Check)]
    return SampleResult.graded([*guards, *checks])


def _bail_fired_check(bail_injected: bool) -> Check:
    """The 'the forced bail actually fired' contract guard as a scored ``Check`` — the graded-path
    twin of the binary path's ``forced bail never fired — contract not exercised`` failure."""
    return Check(
        "forced bail fired — contract exercised",
        bail_injected,
        kind="guard",
        rationale=None
        if bail_injected
        else "the injected bail never fired — the recovery contract was not exercised",
    )


def _cycle_recovered_check(success: bool) -> Check:
    """The 'the cycle recovered to a successful close' guard as a scored ``Check`` — the graded-path
    twin of ``nudge_eval``'s binary ``cycle did not recover to a successful close`` failure."""
    return Check(
        "cycle recovered to a successful close",
        success,
        kind="guard",
        rationale=None
        if success
        else "the cycle did not recover to a successful close after the nudge",
    )


def gave_up_mid_run(db: Database) -> bool:
    """Did any assistant reply apologise for a failure it should have recovered from — a
    defeatist give-up ("Sorry, I wasn't able to get results right now") instead of a retry?"""
    return any(
        message.get("role") == "assistant" and _GAVE_UP.search(message.get("content") or "")
        for message in _iter_prompt_messages(db)
    )


def last_tool_args(db: Database, tool_name: str) -> dict | None:
    """Parsed ``arguments`` of the most recent ``tool_name`` call this run (``None``
    if never called).  Like ``tool_was_called`` but returns the call's args — e.g.
    read a write call's ``entries``.  Sourced from the persisted promptlog
    (newest-first), so it's the real record of what the model emitted, not a
    harness spy.  (Note: ``done`` is argless since #1569, so ``last_tool_args(db,
    "done")`` is ``{}`` when it closed.)"""
    for row in live_prompts(db):
        for call in _response_tool_calls(row):
            if tool_call_name(call) == tool_name:
                try:
                    return json.loads(call.get("function", {}).get("arguments") or "{}")
                except json.JSONDecodeError, TypeError:
                    return {}
    return None


def tool_call_keys(db: Database, tool_name: str) -> list[str]:
    """Every ``key`` argument the model passed to ``tool_name`` across this run.

    Unlike ``last_tool_args`` (newest call only), this collects every call's key so a
    scorer can assert EVERY ``update_entry`` targeted an existing (matched) key — the
    key-not-found ping-pong shows up as a call whose key isn't in the collection.
    Sourced from the persisted promptlog (the real record of what the model did)."""
    keys: list[str] = []
    for row in live_prompts(db):
        for call in _response_tool_calls(row):
            if tool_call_name(call) != tool_name:
                continue
            try:
                args = json.loads(call.get("function", {}).get("arguments") or "{}")
            except json.JSONDecodeError, TypeError:
                continue
            key = args.get("key")
            if isinstance(key, str):
                keys.append(key)
    return keys


def tool_call_sequence(db: Database) -> list[str]:
    """Every tool the model invoked this run, in chronological call order.

    ``recent_prompts`` returns newest-first, so walk it reversed to read the run
    forward; within one response the ``tool_calls`` array is already in emission
    order.  This is the ordering primitive for the multi-step speakable cases: a
    compound NL instruction must fire the RIGHT tools in the RIGHT order, and this
    is the persisted record of what actually fired (not a harness spy)."""
    names: list[str] = []
    for row in reversed(live_prompts(db)):
        for call in _response_tool_calls(row):
            name = tool_call_name(call)
            if name:
                names.append(name)
    return names


# ── Shared loop-health + reply helpers (uniform across the eval case files) ──
# The text-bail nudges a pre-#1839 loop injected as a user turn.  They can no
# longer occur — PR #1840 deleted the constants and the validators that appended
# them — so these markers read HISTORICAL rows only, and a sample recorded before
# that change still reports its bail.  Each is an ASCII, newline-free slice that
# survives row.messages JSON-escaping.
_LEGACY_BAIL_NUDGE_MARKERS = (
    "could not be parsed as a tool call",  # the retired Prompt.TOOL_FORMAT_NUDGE
    "wrote a tool call as plain text",  # the retired Prompt.CHAT_CALL_AS_TEXT_NUDGE
)
# The empty-response continue nudge, retired the same way in #1937 — an empty chat draw
# is discarded and re-rolled now, so this too reads HISTORICAL rows only.  Kept for the
# same reason as the markers above: a sample recorded before that change still reports it.
_CONTINUE_NUDGE_MARKER = "Please provide your response"  # the retired Prompt.CONTINUE_NUDGE


def _legacy_bail_nudge_fired(db: Database) -> bool:
    """True when any prompt's message array carries a retired text-bail nudge — the
    legacy leg, so a promptlog written before #1840 still reads."""
    for row in live_prompts(db):
        if row.messages and any(marker in row.messages for marker in _LEGACY_BAIL_NUDGE_MARKERS):
            return True
    return False


def _same_context_drawn_twice(db: Database) -> bool:
    """Did two persisted draws carry byte-identical ``messages`` — the trace a
    DISCARDED draw leaves behind?

    ``Agent._invoke_nondegenerate`` re-calls the model on the **unchanged** message
    list, and ``LlmClient.chat`` persists every draw it completes before returning —
    so a re-rolled step is two rows with the same context, while an ordinary step's
    context has grown by the turns the previous step appended and can never repeat.
    ``MicroContext._draw_clean`` re-draws the same way, so one read covers the main
    loop and every micro-context.

    Reading the REPEAT is what keeps this honest for a condition nobody has
    enumerated: the harness never re-derives WHICH conditions the loop rejects (a set
    that grows with every agent shape), only that a draw was thrown away."""
    seen: set[str] = set()
    for row in live_prompts(db):
        if not row.messages:
            continue
        if row.messages in seen:
            return True
        seen.add(row.messages)
    return False


def draw_rerolled(db: Database) -> bool:
    """True when this sample recovered via a re-roll — a draw the loop refused to
    accept, discarded and re-drawn on the unchanged context.

    The successor to the text-bail nudge probe (#1839/#1840): an invalid draw is now
    rejected rather than answered with a teaching nudge, so nothing about it enters
    the conversation and there is no marker to match.  What it still leaves is the
    second draw itself (``_same_context_drawn_twice``); the retired nudge markers stay
    as the legacy leg for rows written before the mechanics changed.

    Declared limit: a draw the BACKEND refused to parse (``LlmToolParseError``) raises
    before the client persists anything, so that one re-roll leaves no row at all and
    no promptlog read can see it."""
    return _same_context_drawn_twice(db) or _legacy_bail_nudge_fired(db)


def continue_nudge_fired(db: Database) -> bool:
    """True when any prompt's message array carries the empty-response retry nudge — the
    legacy leg for it, since #1937 rerolls that draw instead of nudging about it."""
    return any(row.messages and _CONTINUE_NUDGE_MARKER in row.messages for row in live_prompts(db))


def routing_clean(db: Database) -> bool:
    """The uniform loop-health verdict every case reports as an ADVISORY check
    (``Check(..., scored=False)``): no draw was re-rolled AND no continue nudge
    fired."""
    return not draw_rerolled(db) and not continue_nudge_fired(db)


# Page-structure vocabulary — asking the user HOW a page is built, which the
# tools cannot use and the user should never have to know (there are no
# selectors, XPaths, or HTML parsing anywhere in the browse surface).  Kept to
# unambiguous terms so a reply that merely discusses a web page never trips it.
_PAGE_STRUCTURE_TERMS = (
    "css",
    "xpath",
    "selector",
    "element id",
    "element_id",
    "html parsing",
    "html tag",
    "dom node",
    "text pattern",
    "regex",
)


def asked_for_page_structure(reply: str) -> str | None:
    """The structure term a reply asked the user for, or ``None`` — the scorable
    form of a failure that was previously only catchable by eye.

    Penny reads pages by saying in plain language what she wants out of them, so
    asking the user for a selector, an element id, or a text pattern is asking
    for something no tool of hers accepts AND something a user has no reason to
    know.  It reappeared at 5-of-8 the moment the elicit instruction stopped
    naming it, while the standing rule in the invariant prompt tail did not hold
    it alone — so it is worth a check rather than a rule nobody measures."""
    lowered = reply.lower()
    return next((term for term in _PAGE_STRUCTURE_TERMS if term in lowered), None)


# The two families a reply describes a ROUTINE's moves in — it reads pages, it saves what
# it finds.  Shared, because two cases now ask the same question of a reply (what a
# standing job does when read back, and what a just-learned routine will run each time,
# #1943) and one policy in two copies is two contracts.
#
# Both are broad by construction and were widened against captured samples: an earlier
# verb-only fetch pattern false-negatived "look on the web" and "pulls in … databases",
# scoring faithful descriptions as misses, and a scorer that reads one phrasing measures
# wording rather than fidelity.  The ambiguous persist verbs (add/store/keep/log/maintain)
# must be ANCHORED to an entry/list object, so "keep an eye on it" never reads as a write.
DESCRIBES_FETCH = (
    r"\b(search\w*|browse\w*|scours?|scans?|hunts?|crawls?|monitors?|gathers?|pulls?\s+in|"
    r"look\w*\s+(for|up|on|at|across|through)|finds?\s+new|fetch\w*|opens?|reads?|checks?)\b"
    r"|\b(the\s+web|online|the\s+internet|the\s+page)\b"
)
# The literal ``collection_write`` was an alternative here until #1943: the learn-close
# frame now hands the model the record's own tool names, so a pattern crediting one read
# back aloud would score the leak as a description.  A reply naming a tool is measured by
# the case that cares, as its own negative check.
DESCRIBES_SAVE = (
    r"\b(saves?|saving|writes?|writing|records?|recording)\b"
    r"|\b(adds?|adding|stores?|storing|keeps?|keeping|logs?|logging|maintains?|"
    r"curates?|compiles?|compiling)\b"
    r"[\w\s,'-]{0,20}\b(entry|entries|list|record|records|collection|them|it)\b"
    r"|\bentr(y|ies)\b[^.]{0,30}\b(added|stored|written|saved|created)\b"
)


def describes(reply: str, pattern: str) -> bool:
    """Whether the reply describes a family, read through the typography the model
    sprinkles (curly quotes, markdown emphasis) — a false negative from a bold marker
    would be the scorer measuring formatting."""
    normalized = reply.casefold().replace("’", "'").replace("“", '"')
    return re.search(pattern, re.sub(r"[*_`]", "", normalized)) is not None


def outgoing_replies(db: Database) -> list[str]:
    """Every message Penny sent this sample (the per-turn replies), oldest first."""
    entries = require_memory(db, "penny-messages").read_recent(window_seconds=3600, cap=None)
    return [entry.content for entry in entries]


def chat_run_tool_sequences(db: Database) -> list[list[str]]:
    """Tool names per CHAT run, in chronological run order — one list per user turn
    of a scripted conversation.  The per-run split is what lets a multi-turn
    contract assert phase discipline (an elicitation turn must not enact; the
    demonstration turn must carry the call spine) — ``tool_call_sequence`` flattens
    the whole sample into one list.  Micro-context calls (browse-extract, skill
    naming) carry no tool calls and other agents' rows are excluded, so each list
    is exactly one chat turn's calls, in emission order."""
    rows = sorted(
        (row for row in live_prompts(db) if row.agent_name == PennyConstants.CHAT_AGENT_NAME),
        key=lambda row: row.timestamp,
    )
    order: list[str] = []
    sequences: dict[str, list[str]] = {}
    for row in rows:
        run_id = row.run_id
        if run_id is None:
            continue
        if run_id not in sequences:
            order.append(run_id)
            sequences[run_id] = []
        sequences[run_id] += [
            name for call in _response_tool_calls(row) if (name := tool_call_name(call))
        ]
    return [sequences[run_id] for run_id in order]


def is_ordered_subsequence(expected: list[str], actual: list[str]) -> bool:
    """True when every name in ``expected`` appears in ``actual`` in that relative
    order — extra calls before, between, or after are allowed.  This is the
    ordering contract for a multi-step NL sequence: the named tools fired, and in
    the order the user described them, while tolerating an extra browse hop (a
    read of a linked page) or a dedup re-read the model interleaves."""
    remaining = iter(actual)
    return all(name in remaining for name in expected)


def tool_call_arg_values(db: Database, tool_name: str, field: str) -> list[str]:
    """Every string value the model passed for ``field`` across all ``tool_name``
    calls this run — the general form of ``tool_call_keys`` (which is this with
    ``field="key"``).  Lets a scorer assert WHICH collections a multi-read swept
    (the ``memory`` field of each ``collection_read_latest``) without re-parsing
    the promptlog.  Sourced from the persisted promptlog (the real record)."""
    values: list[str] = []
    for row in live_prompts(db):
        for call in _response_tool_calls(row):
            if tool_call_name(call) != tool_name:
                continue
            try:
                args = json.loads(call.get("function", {}).get("arguments") or "{}")
            except json.JSONDecodeError, TypeError:
                continue
            value = args.get(field)
            if isinstance(value, str):
                values.append(value)
    return values


# Tools whose arguments carry an entry key the model copies from a render.
_KEY_BEARING_TOOLS = (
    "update_entry",
    "collection_delete_entry",
    "collection_get",
    "collection_write",
)


def _is_bracket_wrapped(key: str) -> bool:
    """True when ``key`` is wrapped in display brackets (``[foo]``) — the copied
    ``[key]`` render form, never a real key."""
    return len(key) > 2 and key.startswith("[") and key.endswith("]")


def bracket_wrapped_key_calls(db: Database) -> list[str]:
    """Every key argument the model passed this run that is wrapped in display
    brackets (``key="[foo]"``) — the copy-through mistake the old ``[key]`` render
    taught (225 observed leaks).  Scans the persisted promptlog across the whole
    run for key-bearing tool calls: single ``key=`` args and ``entries=[{key}]``
    write batches whose value is bracket-wrapped.  Empty means the render never
    tempted the model into pasting display brackets into an argument — the whole
    point of rendering keys in invocation form."""
    offenders: list[str] = []
    for row in live_prompts(db):
        for call in _response_tool_calls(row):
            function = call.get("function", {})
            if tool_call_name(call) not in _KEY_BEARING_TOOLS:
                continue
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError, TypeError:
                continue
            candidates = [args["key"]] if isinstance(args.get("key"), str) else []
            for entry in args.get("entries") or []:
                if isinstance(entry, dict) and isinstance(entry.get("key"), str):
                    candidates.append(entry["key"])
            offenders += [key for key in candidates if _is_bracket_wrapped(key)]
    return offenders


_NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)


def looks_numbered(text: str) -> bool:
    """True when ``text`` reads as a numbered list (≥2 lines like ``1.`` / ``2)``).

    Used by format contracts: a prompt the model follows reliably is a numbered
    instruction/tool-call recipe, not flowing prose.
    """
    return len(_NUMBERED_LINE.findall(text)) >= 2


def tool_call_name(call: dict) -> str:
    """One logged call's tool name, normalised the way PRODUCTION normalises it.

    Every read of a tool name off the promptlog goes through here, because the raw name can
    carry leaked Harmony control tokens (`collection_write<|channel|>commentary`, seen on two
    samples of one run).  Production strips them at the boundary where the name is read off the
    model response — ``LlmToolCallFunction.name``, via ``strip_harmony_control_tokens`` — so
    registry lookup, done-detection, dedup and result framing all see the clean identifier, and
    the eval reading the raw one was the single exception.

    What it cost while it was the exception: `tool_was_called`, `count_tool_calls` and the
    sequence readers silently missed a call the store proves ran, which converted a correct
    sample into an outlier for a divergence that never happened.  Every case in the suite that
    counts or detects a tool call was exposed to it, not just the one that surfaced it."""
    return strip_harmony_control_tokens((call.get("function") or {}).get("name") or "")


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


class _BrowseReadError(Exception):
    """Raised by a ``fails=True`` CannedPage so the browse tool renders a real
    ``## browse error:`` section for that query.  Deliberately NOT a
    ``ConnectionError``/``TimeoutError`` — those are the two ``_read_page``
    retries (1s·2^n backoff ×4), which a flailing all-fail cycle would multiply
    into minutes per sample.  An uncaught type propagates straight to the
    per-subcall ``gather(return_exceptions=True)`` and renders immediately, with
    the same ``Could not read this page: <message>`` text a real failure shows.
    """


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
    maps to a different page — supporting multi-hop chains.  A ``fails=True``
    page raises instead of returning, so the query renders ``## browse error:``
    (see ``CannedPage``).  Installed on BOTH agents (chat + collector) since the
    generic mock sits on both.
    """

    async def request_fn(method: str, params: dict) -> tuple[str, str | None]:
        url = params.get("url", "").lower()
        for page in pages:
            if page.match.lower() in url:
                if page.channel_outage:
                    # A whole-channel outage (no browser connected).  Raised straight
                    # here (bypassing _read_page's retry loop, which BrowseChannelUnavailableError
                    # deliberately isn't a ConnectionError to trigger) so the tool renders
                    # the consolidated outage banner without the real backoff wait.
                    raise BrowseChannelUnavailableError("no browser is connected")
                if page.fails:
                    raise _BrowseReadError(
                        f"failed to read {url} after 3 attempts: the source could not be read"
                    )
                return page.text, page.image
        return _NO_RESULTS_PAGE, None

    def provider() -> tuple[Callable, MagicMock]:
        return request_fn, MagicMock(check_domain=AsyncMock())

    penny.chat_agent._browse_provider = provider
    penny.collector._browse_provider = provider


async def _embed_seeds(penny: Penny) -> None:
    """Vectorize seeded memory so similarity reads behave like prod.

    Penny's startup backfill ran on the empty DB before we seeded; re-run it so
    seeded entries, descriptions and MESSAGES get embeddings that ``read_similar`` /
    resolve-by-meaning can match.

    All THREE of production's startup backfills, because a seed can lay down any of the
    three embedding-bearing shapes and this is the one place they are vectorized.  The
    messagelog leg was missing, so a case that seeded a conversation turn and then asked
    the model to recall it was UNREACHABLE: the read was aimed correctly, the vectorless
    row could not rank, and the case scored 0.00 with nothing in the transcript to say
    why (the eval-audit fleet hit this class three times).  A seed's world must be the
    world production would have.
    """
    await penny._backfill_memory_embeddings(_EMBED_BATCH)
    await penny._backfill_description_embeddings(_EMBED_BATCH)
    await penny._backfill_message_embeddings(_EMBED_BATCH)


def _refuse_dead_cohort(case_id: str, results: list[SampleResult], intended: int) -> None:
    """Fail the case when too few of its intended samples ever produced a measured turn.

    Before any score is compared, because a score over a fraction of the cohort is not a
    lower score — it is not a result.  A run once reported ``6 passed, EXIT=0`` with 34 of
    48 samples dead, and every one of those means was computed over whatever survived.

    The bar and the reason are both in the message: dead samples are not missing at
    random (see ``run_health``), so what survives is a biased draw rather than a smaller
    one, and a strict majority is where that stops being readable as the case.
    """
    completed = len(results)
    if run_health.cohort_is_viable(completed, intended):
        return
    pytest.fail(
        f"{case_id}: {completed} of {intended} samples produced their measured turn — "
        f"{run_health.COHORT_RULE} (at least {run_health.results_needed(intended)} here), "
        "because the faults that kill samples correlate with the work, so the survivors "
        "are a biased draw and not a smaller one. Read the run-health block for the "
        "dominant fault class, and each sample's .log beside its .db for the calls."
    )


def _assert_threshold(
    case_id: str,
    results: list[SampleResult],
    min_pass_rate: float | None,
    *,
    intended: int,
    gate_pathology_excluded: bool = False,
) -> None:
    """Print the case's X/Y pass rate, and — unless report-only — gate on it.

    ``intended`` is how many samples the case ASKED for: a cohort that mostly died is
    refused here before any threshold is compared, report-only cases included, because
    "no result" is not a score that report-only means to tolerate.

    ``min_pass_rate=None`` is report-only: the X/Y line and any per-sample
    failures print for insight, but the case never fails the run.  Use it for
    inherently stochastic behaviours we want to *observe* rather than gate (the
    self-correction cases — the model can't clear every cross-run repeat, and a
    flaky red adds no signal beyond the printed rate).

    ``gate_pathology_excluded=True`` gates on the **pathology-excluded** mean
    (#1695) instead of the raw mean — the honest read of model behaviour, over
    every sample that is NOT a pathology failure (a reroll-guard collapse can't
    sink the bar).  This is what lets a case that dispatches reliably but for the
    known gpt-oss degeneracy collapse carry its true bar (e.g. the speakable
    sequence cases restored to 0.8, #1698) rather than a bar lowered to absorb
    that pathology.  The raw mean + the pathology count stay visible in the
    printed cause line, so a pathology spike remains legible.
    """
    _refuse_dead_cohort(case_id, results, intended)
    total = len(results)
    mean = sum(result.score for result in results) / total if total else 0.0
    all_pass = sum(1 for result in results if result.passed)
    # Dual metric: the MEAN of per-sample scores (partial credit) is what the case gates on;
    # the all-pass count (samples that passed EVERY applicable check — ``SampleResult.passed``)
    # is the strict companion beside it, so a mean propped up by partial credit is visible.
    metric = f"mean {mean:.2f} · all-pass {all_pass}/{total}"
    # Failure-cause read (#1695): the pathology-excluded mean + the behavioral/pathology/harness
    # tally, on a second line, so a score sunk by model NOISE (a degeneracy spike) reads distinctly
    # from a score sunk by the model getting it WRONG (the signal the loop chases).
    causes = [result.cause for result in results]
    excluded_mean, kept = eval_artifacts.pathology_excluded(
        [result.score for result in results], causes
    )
    cause_line = eval_artifacts.render_cause_summary(
        eval_artifacts.count_causes(causes), excluded_mean, kept
    )
    # Per-sample detail: the score (1.0/0.0 for binary, the check fraction for graded) and
    # what missed — for every sample that wasn't perfect.
    detail = "\n".join(
        f"  [{i + 1}] {result.score:.2f}"
        + (f" — {'; '.join(result.failed)}" if result.failed else "")
        for i, result in enumerate(results)
        if result.failed
    )
    if min_pass_rate is None:
        print(f"\nRESULT [{case_id}] {metric} across {total} samples (report-only)")
        print(f"  {cause_line}")
        if detail:
            print(detail)
        return
    # Which metric the gate compares: the pathology-excluded mean when the case opts in
    # (#1698 — model NOISE can't sink the bar), else the raw mean.
    gated_value = excluded_mean if gate_pathology_excluded else mean
    gated_label = "pathology-excluded mean" if gate_pathology_excluded else "mean"
    need = f"need {gated_label} >={min_pass_rate}"
    print(f"\nRESULT [{case_id}] {metric} across {total} samples ({need})")
    print(f"  {cause_line}")
    if gated_value < min_pass_rate:
        pytest.fail(f"{case_id}: {gated_label} {gated_value:.2f} < {min_pass_rate}:\n{detail}")


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

    Emitted as ONE print: samples may run concurrently (``EVAL_CONCURRENCY``), and a dump
    written line-by-line would braid itself through another sample's, leaving two traces
    that each look whole.  A single write keeps a sample's reasoning together.
    """
    if not failed and not os.environ.get("EVAL_DUMP_THINKING"):
        return
    with Session(db.engine) as session:
        rows = session.exec(select(PromptLog).order_by(col(PromptLog.timestamp).asc())).all()
    lines = [f"\n===== THINKING [{case_id} #{sample_index}] — {len(rows)} LLM call(s) ====="]
    for index, row in enumerate(rows, start=1):
        label = row.agent_name or row.prompt_type or "?"
        if row.thinking:
            lines.append(f"[{index}:{label}] THINKING: {row.thinking.strip()}")
        for call in _response_tool_calls(row):
            function = call.get("function", {})
            lines.append(
                f"[{index}:{label}] TOOL: {function.get('name')}({function.get('arguments')})"
            )
    lines.append("===== END THINKING =====\n")
    print("\n".join(lines))


# ── Eval run report (verbatim transcripts, for the PR body) ──────────────────
# When EVAL_REPORT_DIR is set (wired through by the Makefile `eval` target), each sample
# appends a markdown section — the full turn-by-turn transcript read from the ephemeral
# promptlog before the --rm DB is discarded — to <dir>/<case_id>.md.  The SOP
# (docs/agent-task-workflow.md §4) folds these into the PR body under a <details> per case,
# so a reviewer sees every run verbatim without a wall of text.  Off by default (no dir set
# ⇒ no-op), so ordinary `make eval` runs are unaffected.

_ACTOR = {
    "user": "👤 user",
    "tool": "📥 tool result",
    "call": "🔧 Penny → tool",
    "penny": "🤖 Penny",
}


def _render_call(function: dict) -> str:
    """Render a tool call as ``name(args)`` with its arguments JSON reserialized canonically.

    The SAME call is serialized two ways across a run: compactly in ``promptlog.response`` (the
    model's raw emission, ``{"queries":["x"]}``) and with default spacing in the NEXT prompt's
    ``messages`` (``LlmMessage.to_input_message`` re-dumps the parsed args, spaced and
    ASCII-escaping any unicode).  Parsing and re-dumping BOTH sides through one form
    (``ensure_ascii=False``, default separators) renders the call identically wherever it is read,
    so (1) a call's thinking — keyed off the response side by ``_thinking_by_content`` — binds to
    its transcript row (built off the messages side by ``_sample_turns``), and real thinking stops
    silently dropping on every tool call (#1735 finding 1); and (2) ``\\uXXXX`` escapes render as
    their real characters in the call-argument cell (finding 3).  Malformed / absent args fall back
    to the raw string, so a non-JSON payload never raises."""
    name = function.get("name")
    raw = function.get("arguments")
    if not isinstance(raw, str):
        return f"{name}()"  # no/non-string args (defensive — a real call carries a JSON string)
    try:
        rendered = json.dumps(json.loads(raw), ensure_ascii=False)
    except json.JSONDecodeError, TypeError:
        rendered = raw  # a non-JSON payload renders verbatim rather than raising
    return f"{name}({rendered})"


def _sample_turns(
    rows: list[PromptLog],
    reply: str,
    driven: Sequence[str] = (),
    delivered: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """(actor, content) for every turn of the sample, across ALL promptlog rows — so a
    multi-turn conversation shows EVERY turn's tool calls, not just the last turn's.

    Each row's ``messages`` array accumulates the conversation up to that LLM call (a later
    turn carries an earlier one only as text history, so an earlier turn's tool calls live
    only in that turn's own rows), and its ``trailing_messages`` carries the tail no call
    ever took (``_row_turns``).  Walking every row and de-duplicating by (actor, content)
    yields each user turn, tool call, tool result, and intermediate reply exactly once, in
    order.  The final reply (the last response's text, which is in no messages array) is
    appended last.  System prompt omitted.

    ``driven`` is what the harness actually PUSHED.  A case may seed prior conversation to
    stand the world up (the state a preceding beat ends in), and the chat agent replays that
    history into its ``messages`` array where it is byte-identical to a turn the sample
    drove — so without this the transcript opened a step for a message nobody sent this
    sample, and the classifier, anchored to the first turn head, rendered under it instead of
    under the message it actually judged.  Seeded turns are context, not steps: they are in
    the system prompt, the classifier's own slice, and the DB.  An empty ``driven`` names
    nothing to filter against, so every user message in the rows opens a step.

    ``delivered`` is what Penny actually SENT.  A discarded draw is persisted whole (#1839 keeps
    the ledger honest by re-rolling on the unchanged context and logging both attempts), and the
    transcript is built from the promptlog — so every re-rolled text draw rendered as a 🤖 reply,
    indistinguishable from the message the user received.  Measured on the reference run, all 18
    samples carry exactly TWO outgoing messages and the report showed three: Penny sends one
    message, the report claimed she sent two.  Rerolls are working machinery and must not appear
    anywhere they can be read as output, so a text draw renders as a reply only if it was
    delivered; the rest are collected by ``rejected_draws`` for their own fold.  An empty
    ``delivered`` names nothing to exclude, so every draw renders as a reply."""
    turns: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    pushed = {content.strip() for content in driven}
    sent = {content.strip() for content in delivered}

    def emit(actor: str, content: str) -> None:
        if content and (actor, content) not in seen:
            seen.add((actor, content))
            turns.append((actor, content))

    for row in rows:
        for message in _row_turns(row):
            role, content = message.get("role"), message.get("content") or ""
            if role == "user":
                if pushed and content.strip() not in pushed:
                    continue  # seeded history, not a turn this sample drove
                emit(_ACTOR["user"], content)
            elif role == "tool":
                emit(_ACTOR["tool"], content)
            elif role == "assistant":
                for call in message.get("tool_calls") or []:
                    emit(_ACTOR["call"], _render_call(call.get("function", {})))
                if not sent or content.strip() in sent:
                    emit(_ACTOR["penny"], content)
    emit(_ACTOR["penny"], reply.strip())
    return turns


def _delivered_replies(db: Database) -> list[str]:
    """What Penny actually SENT this sample, or nothing when the message log is absent.

    Empty means "do not filter" — the pre-#1997 rendering — so a bare-schema database with no
    `penny-messages` facade renders exactly as it always did rather than failing."""
    if db.memory(PennyConstants.MEMORY_PENNY_MESSAGES_LOG) is None:
        return []
    return outgoing_replies(db)


def rejected_draws(rows: list[PromptLog], delivered: Sequence[str]) -> list[str]:
    """Every text draw this sample produced that was never sent.

    Kept for diagnosis and rendered behind its own fold, labelled as a rejected draw — never in
    the reply stream, where it reads as a message the user received."""
    sent = {content.strip() for content in delivered}
    if not sent:
        return []
    drawn: list[str] = []
    for row in rows:
        for message in _row_turns(row):
            if message.get("role") != "assistant" or message.get("tool_calls"):
                continue
            content = (message.get("content") or "").strip()
            if content and content not in sent and content not in drawn:
                drawn.append(content)
    return drawn


# A check whose `anchor` is this sentinel is about the final NL reply itself (not a tool
# call) — it stamps the last Penny-reply row rather than falling to the footer.
REPLY_ANCHOR = "__reply__"


def _anchor_hits(needle: str, content: str) -> bool:
    """Does this tool-call row satisfy the anchor? A tool-name anchor (``memory_metadata(``)
    matches that call; a keyword anchor (``designer``, ``"published": false``) must live inside
    a ``collection_set`` call — the row that made the edit — never another tool's reasoning
    field that merely mentions the word."""
    if needle.endswith("("):
        return needle in content
    return "collection_set(" in content and needle in content


def _place_checks(
    checks: list[Check], turns: list[tuple[str, str]]
) -> tuple[dict[int, list[Check]], list[Check]]:
    """Bind each anchored check to the FIRST turn whose content contains its anchor.

    A ``REPLY_ANCHOR`` check stamps the final Penny-reply row (it tests the reply's text, not
    a tool call).  Returns ``(turn_index -> the checks placed there, leftover checks)`` — the
    per-turn check lists so a caller can bind each check to the event it anchors to (#1725).  A
    check with no anchor — or whose anchor matches no turn (a *missing* expected action, a tool
    call that never happened) — has no row to sit on, so it falls to ``leftover`` (run-close)."""
    placed: dict[int, list[Check]] = {}
    leftover: list[Check] = []
    reply_row = max(  # the final NL reply row — where a REPLY_ANCHOR check lands
        (i for i, (actor, _c) in enumerate(turns) if actor == _ACTOR["penny"]), default=None
    )
    for check in checks:
        hit = None
        if check.anchor == REPLY_ANCHOR:
            hit = reply_row
        elif check.anchor:  # match only Penny's tool-call rows — never the user turn naming it
            needle = check.anchor.lower()
            hit = next(
                (
                    i
                    for i, (actor, content) in enumerate(turns)
                    if actor == _ACTOR["call"] and _anchor_hits(needle, content.lower())
                ),
                None,
            )
        if hit is None:
            leftover.append(check)
        else:
            placed.setdefault(hit, []).append(check)
    return placed, leftover


def _sample_db_path(tmp_path, case_id: str, sample_index: int, attempt: int = 0) -> str:
    """Where a sample's hermetic DB lives.  When ``EVAL_REPORT_DIR`` is set the DB
    persists BESIDE the reports (the mounted dir survives the ``--rm`` container),
    so a run's raw promptlog can be re-read after the fact — same doctrine as the
    transcripts: the evidence always survives the run.  Unset → tmp_path as before.

    ``attempt`` keys a RE-DRIVEN sample onto its own file (#1803 review): nothing
    deletes a sample's DB, so a retry handed the same path would re-seed over the
    failed attempt's rows and inherit its ``messagelog``/``promptlog`` — the failed
    turn replayed into the next attempt's context and into its transcript.  The first
    attempt keeps the unsuffixed name, so a runner that never retries is unchanged."""
    report_dir = os.environ.get("EVAL_REPORT_DIR")
    base = Path(report_dir) if report_dir else tmp_path
    Path(base).mkdir(parents=True, exist_ok=True)
    suffix = f"-attempt{attempt + 1}" if attempt else ""
    return str(Path(base) / f"{case_id}-{sample_index}{suffix}.db")


# The logger every penny module logs through (each is ``logging.getLogger(__name__)``, so
# one handler on the package root catches the client, the agent loop and the collector alike).
PENNY_LOGGER = "penny"
SAMPLE_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def sample_log_path(db_path: str) -> Path:
    """Where a sample's penny log lands: beside its DB, under the same stem (#1909)."""
    return Path(db_path).with_suffix(".log")


# Which sample a log record belongs to.  Samples may run CONCURRENTLY
# (``EVAL_CONCURRENCY``) while the ``penny`` logger is process-global, so N samples mean N
# handlers attached at once and every open file would take every sample's lines — the
# per-sample log stops being per-sample exactly when a run is hardest to read.  A context
# variable is what tells them apart: it is set for the span of one sample, and asyncio
# copies the context into every task that sample starts, so a record carries the identity
# of the sample that emitted it wherever in Penny it was emitted from.
_active_sample: ContextVar[str | None] = ContextVar("eval_active_sample", default=None)


class _SampleFilter(logging.Filter):
    """Admits only the records emitted by ONE sample.

    A record from no sample at all (nothing set the variable) is dropped rather than
    written to every open log: attributing it nowhere is honest, attributing it
    everywhere is not.
    """

    def __init__(self, sample: str) -> None:
        super().__init__()
        self._sample = sample

    def filter(self, record: logging.LogRecord) -> bool:
        return _active_sample.get() == self._sample


class _CaptureLevel:
    """Reference-counts the DEBUG level across concurrently capturing samples.

    The logger's LEVEL is process-global just as its handlers are, so a sample that
    restored the level on its own way out lowered it under every sample still running —
    their remaining lines were then dropped before any handler saw them, and the log of a
    sample that outlived its neighbours simply stopped mid-run.  Raised by the first
    sample to arrive, restored by the last to leave.  No lock: samples are asyncio tasks
    in one thread, so a begin/end pair never interleaves with another.
    """

    def __init__(self) -> None:
        self._depth = 0
        self._restore_to: int | None = None

    def begin(self, logger: logging.Logger) -> None:
        if self._depth == 0:
            self._restore_to = logger.level
            logger.setLevel(logging.DEBUG)
        self._depth += 1

    def end(self, logger: logging.Logger) -> None:
        self._depth -= 1
        if self._depth == 0 and self._restore_to is not None:
            logger.setLevel(self._restore_to)
            self._restore_to = None


_capture_level = _CaptureLevel()


@contextmanager
def sample_logging(db_path: str) -> Iterator[Path]:
    """Capture one sample's penny logger output into ``<sample>.log`` beside its DB.

    A model call that fails writes NO promptlog row — it raises before the client's
    persist step — so everything the run says about the failure ("LLM chat failed", a
    timeout attempt, each discarded draw's reroll condition) exists only as logger
    output, which pytest captures and then discards for every sample that passes.  This
    is the same doctrine the sample DB and the transcripts follow: the evidence always
    survives the run (#1909).

    Mechanical by CONTENT — DEBUG level, no content filter — and scoped to ONE sample by
    ORIGIN: the handler admits only records this sample emitted, so every line is
    attributable to the sample whose name the file carries however many samples are in
    flight.  The handler is removed and the logger's level restored on the way out, so
    nothing leaks into the next sample or into the rest of the suite.
    """
    path = sample_log_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w")
    handler.setFormatter(logging.Formatter(SAMPLE_LOG_FORMAT))
    handler.setLevel(logging.DEBUG)
    handler.addFilter(_SampleFilter(str(path)))
    penny_logger = logging.getLogger(PENNY_LOGGER)
    _capture_level.begin(penny_logger)
    penny_logger.addHandler(handler)
    token = _active_sample.set(str(path))
    try:
        yield path
    finally:
        _active_sample.reset(token)
        penny_logger.removeHandler(handler)
        _capture_level.end(penny_logger)
        handler.close()


@asynccontextmanager
async def eval_penny(config: Config, server: MockSignalServer) -> AsyncIterator[Penny]:
    """One sample's Penny, with its log captured beside its DB (#1909).

    Every runner stands its sample up through here rather than calling
    ``run_penny_with_server`` directly, so the capture is one seam that covers the
    runners that exist and the ones added later — the log's path is derived from the
    config's own ``db_path``, which is already the sample's identity.
    """
    with sample_logging(config.db_path):
        async with run_penny_with_server(
            config, server, ready_timeout=SAMPLE_READY_TIMEOUT_SECONDS
        ) as penny:
            yield penny


# ── Transcript extraction: promptlog → report.SampleTranscript (#1725 iteration-6) ──


class MicroPlacement(StrEnum):
    """Where a micro-context's events splice into the transcript (#1773).

    One value per CAUSAL relationship to the run, not per agent — which is what makes this a
    generalization rather than a pile of per-customer branches: a fourth micro-context declares
    the relationship it already has and needs no new walk code."""

    # It ran INSIDE a tool call (the browse-extract sub-model, while ``browse`` was executing) —
    # its events belong immediately after that call's row.
    DURING_CALL = "during_call"
    # It ran BEFORE the chat agent, on the message that opens a turn (the state classifier, whose
    # draw selects the instruction the turn is answered under) — its events belong at the head of
    # the turn it decides, right after the user turn that provoked it.
    TURN_HEAD = "turn_head"
    # It ran AFTER the run's last action (the run-end labeller and framer) — its events close the
    # turn.
    RUN_CLOSE = "run_close"


# Every micro-context customer, by its ledger identity (``promptlog.agent_name``), with the
# placement its position in the run implies.  The ONE place agent names appear in the transcript
# walk; everything downstream reads the placement.
MICRO_CONTEXT_PLACEMENTS: dict[str, MicroPlacement] = {
    PennyConstants.BROWSE_EXTRACT_AGENT_NAME: MicroPlacement.DURING_CALL,
    PennyConstants.STATE_CLASSIFIER_AGENT_NAME: MicroPlacement.TURN_HEAD,
    PennyConstants.SKILL_NAMING_AGENT_NAME: MicroPlacement.RUN_CLOSE,
    # The framer moved to learn ENTRY (#1868): it draws before the chat agent, on the
    # message that opens the round, so its events belong at the head of the turn it frames
    # — the same causal relationship the state classifier has, and now the same placement.
    # A round nothing framed at entry is framed at run end instead, and that draw renders
    # at the head too; the placement declares where the draw normally sits, and a batch is
    # never dropped for sitting elsewhere.
    PennyConstants.SKILL_FRAME_AGENT_NAME: MicroPlacement.TURN_HEAD,
}

_NUDGE_FRAMES = (
    "Please provide your response",  # the retired Prompt.CONTINUE_NUDGE (#1937)
    "could not be parsed as a tool call",  # the parse-failure recovery nudge
    "you MUST respond with a valid tool call",
    "make the real, argless",  # COLLECTOR_DONE_JSON_NUDGE
    "respond with a tool call",  # COLLECTOR_CONTINUE_NUDGE / COLLECTOR_TOOL_CALL_NUDGE
)


def _is_nudge(content: str) -> bool:
    """A user-role turn that is a framework RECOVERY nudge (not a real user ask) — it renders as a
    ``⚠ recovery event`` inside the step, never as a step boundary."""
    return any(frame in content for frame in _NUDGE_FRAMES)


def _turn_kind(actor: str, content: str) -> report.EventKind:
    """Map a ``_sample_turns`` (actor, content) pair to a report event kind (a recovery-nudge user
    turn becomes ``NUDGE`` so it renders inside its step, not as a new step)."""
    if actor == _ACTOR["user"]:
        return report.EventKind.NUDGE if _is_nudge(content) else report.EventKind.USER
    if actor == _ACTOR["call"]:
        return report.EventKind.CALL
    if actor == _ACTOR["tool"]:
        return report.EventKind.RESULT
    return report.EventKind.REPLY


def _event_body(kind: report.EventKind, content: str) -> str:
    """The rendered body for an event (the glyph is prepended by the renderer): a reply is quoted,
    a nudge tagged ``*(nudge)*``, a call/result rendered verbatim."""
    if kind == report.EventKind.REPLY:
        return f'"{content}"'
    if kind == report.EventKind.NUDGE:
        return f"*(nudge)* {content}"
    return content


def _thinking_by_content(rows: list[PromptLog]) -> dict[str, str]:
    """Map each model ACTION's content (a ``name(args)`` call string or a reply's text) to the
    thinking of the promptlog row whose RESPONSE produced it — so EVERY model call can show its
    own reasoning (#1725, superseding the failed-turns-only capture). First non-empty per key.

    A call is keyed through ``_render_call`` (a CANONICAL ``name(args)`` reserialization) so it
    matches ``_sample_turns``' transcript row for the SAME call — the two are built from the two
    different serializations of the arguments (compact ``response`` vs. re-dumped ``messages``), so
    string-matching the raw forms silently dropped a real call's thinking on every tool call (#1735
    finding 1). Reply text keys on itself (both sides read ``choices[0].message.content``)."""
    mapping: dict[str, str] = {}
    for row in rows:
        thinking = (row.thinking or "").strip()
        if not thinking:
            continue
        for call in _response_tool_calls(row):
            mapping.setdefault(_render_call(call.get("function", {})), thinking)
        text = _response_text(row)
        if text:
            mapping.setdefault(text, thinking)
    return mapping


def _micro_events(row: PromptLog) -> list[report.Event]:
    """One micro-context promptlog row → its two events: the scoped turn INTO the sub-model
    (🧩 ← user turn:) and the drawn output OUT of it (🧩 →, carrying its thinking). The body is
    the content ONLY — the role label is the renderer's (#1759) — and both events carry the row's
    ledger identity as their actor label (#1773), so browse extraction, state classification and
    skill naming read apart and each matches its own system-prompt row."""
    messages = json.loads(row.messages) if row.messages else []
    user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
    context = row.agent_name or ""
    return [
        report.Event(report.EventKind.MICRO_IN, user, context=context),
        report.Event(
            report.EventKind.MICRO_OUT,
            _response_text(row),
            thinking=row.thinking or "",
            context=context,
        ),
    ]


def _system_prompts(rows: list[PromptLog]) -> list[report.SystemPrompt]:
    """The DISTINCT system prompts across a sample's promptlog rows (main agent + each micro-context
    flavour), in first-appearance order, deduped by verbatim text — a repeated main-loop prompt
    renders once (#1759). Each is labelled by the ``agent_name`` of the row it first appeared on
    (the context under test); a row carrying no system message contributes none."""
    seen: set[str] = set()
    prompts: list[report.SystemPrompt] = []
    for row in rows:
        messages = json.loads(row.messages) if row.messages else []
        system = next((m.get("content") or "" for m in messages if m.get("role") == "system"), "")
        if not system or system in seen:
            continue
        seen.add(system)
        prompts.append(report.SystemPrompt(context=row.agent_name or "", text=system))
    return prompts


@dataclass
class MicroBatch:
    """One micro-context's contiguous run of promptlog rows, as transcript events plus the
    placement its ledger identity declares — the unit the transcript walk splices (#1773)."""

    placement: MicroPlacement
    events: list[report.Event]


def _micro_batches(rows: list[PromptLog]) -> list[MicroBatch]:
    """EVERY micro-context call in the sample's ledger, batched, in ledger order (#1773).

    A batch is a maximal run of CONSECUTIVE rows from the SAME micro-context — the pages one
    ``extract`` browse fetched, or a single draw plus its reroll — so two customers that happen to
    be ledger-adjacent (a run-end labeller and the next turn's classifier) never merge into one
    actor. Each batch carries its placement, so the walk splices it without knowing which agent
    produced it."""
    batches: list[MicroBatch] = []
    open_agent: str | None = None
    for row in rows:
        agent = row.agent_name or ""
        placement = MICRO_CONTEXT_PLACEMENTS.get(agent)
        if placement is None:
            open_agent = None
            continue
        if agent != open_agent:
            batches.append(MicroBatch(placement, []))
            open_agent = agent
        batches[-1].events.extend(_micro_events(row))
    return batches


def _is_extract_browse(content: str) -> bool:
    """A ``browse(...)`` call that carried an ``extract`` micro-instruction — the calls that spawn
    the browse-extract micro-context rows."""
    return content.startswith("browse(") and '"extract"' in content


def _splice(
    batches: list[MicroBatch], placement: MicroPlacement, events: list[report.Event]
) -> None:
    """Splice every batch AT THE HEAD of the queue with this placement, then stop — so the queue
    drains in strict ledger order and a batch waiting on a later anchor holds the ones behind it."""
    while batches and batches[0].placement == placement:
        events.extend(batches.pop(0).events)


def _splice_one(
    batches: list[MicroBatch], placement: MicroPlacement, events: list[report.Event]
) -> None:
    """Splice the ONE batch at the head of the queue with this placement — the FIFO pairing an
    in-call batch keeps with the call that spawned it (one batch per extract-browse call)."""
    if batches and batches[0].placement == placement:
        events.extend(batches.pop(0).events)


def _turns_to_events(
    turns: list[tuple[str, str]], thinking: dict[str, str], micro_batches: list[MicroBatch]
) -> tuple[list[report.Event], dict[int, int]]:
    """Turn the de-duped ``(actor, content)`` turns into report events, splicing each micro-context
    batch at the anchor its placement names, in ledger order (#1773): a run-end batch closes the
    turn in progress (before the next user turn), a turn-head batch opens the turn it decided
    (right after that user turn), and an in-call batch follows the extract-browse call that spawned
    it. Anything still queued at the end renders there rather than vanishing (collapsed never means
    dropped, #1753). Returns the events and a ``turn index → event index`` map so a check placed on
    a turn resolves to its event."""
    events: list[report.Event] = []
    turn_to_event: dict[int, int] = {}
    for turn_index, (actor, content) in enumerate(turns):
        kind = _turn_kind(actor, content)
        if kind == report.EventKind.USER:
            _splice(micro_batches, MicroPlacement.RUN_CLOSE, events)
        action = kind in (report.EventKind.CALL, report.EventKind.REPLY)
        thought = (thinking.get(content) or "") if action else None
        turn_to_event[turn_index] = len(events)
        events.append(report.Event(kind, _event_body(kind, content), thinking=thought))
        if kind == report.EventKind.USER:
            _splice(micro_batches, MicroPlacement.TURN_HEAD, events)
        elif kind == report.EventKind.CALL and _is_extract_browse(content):
            _splice_one(micro_batches, MicroPlacement.DURING_CALL, events)
    for batch in micro_batches:
        events.extend(batch.events)
    return events, turn_to_event


def _assign_check_ids(checks: list[Check]) -> dict[int, str]:
    """Assign each check its ``Cn`` id (or ``Gn`` for a framework guard), in scorer order."""
    ids: dict[int, str] = {}
    counters = {"C": 0, "G": 0}
    for check in checks:
        prefix = "G" if check.kind == "guard" else "C"
        counters[prefix] += 1
        ids[id(check)] = f"{prefix}{counters[prefix]}"
    return ids


def _cause_word(cause: FailureCause | None) -> str | None:
    """The banner/verdict cause word: ``None`` for a pass; ``pathology (degenerate output)`` for a
    pathology sample; else the plain cause value (``behavioral`` / ``harness``)."""
    if cause is None:
        return None
    if cause == FailureCause.PATHOLOGY:
        return "pathology (degenerate output)"
    return cause.value


def _build_check_views(
    result: SampleResult,
    turn_of_check: dict[int, int],
    turn_to_event: dict[int, int],
    baseline: Baseline | None,
    case_id: str,
) -> list[report.CheckView]:
    """Resolve each ``Check`` into a ``report.CheckView`` — its id, class, anchor event (``None`` →
    run-close), rationale/cause, and baseline flip — in scorer order."""
    ids = _assign_check_ids(result.checks)
    cause = _cause_word(result.cause)
    views: list[report.CheckView] = []
    for check in result.checks:
        turn_index = turn_of_check.get(id(check))
        anchor = turn_to_event.get(turn_index) if turn_index is not None else None
        regressed = (
            baseline is not None
            and not check.ignored
            and not check.ok
            and baseline.was_passing(case_id, check.label)
        )
        views.append(
            report.CheckView(
                check_id=ids[id(check)],
                label=check.label,
                kind=check.kind,
                scored=check.scored,
                ignored=check.ignored,
                ok=check.ok,
                rationale=check.rationale,
                cause=cause if (not check.ok and not check.ignored) else None,
                anchor_index=anchor,
                regressed=regressed,
            )
        )
    return views


def _scored_counts(result: SampleResult) -> tuple[int, int]:
    """This sample's ``(passed, scored)`` check counts — the same partition ``SampleResult.graded``
    scores over (n/a excluded, then the scored ones or all applicable). Binary = 1 check."""
    if not result.checks:
        return (1 if result.passed else 0), 1
    applicable = [check for check in result.checks if not check.ignored]
    scored = [check for check in applicable if check.scored] or applicable
    return sum(1 for check in scored if check.ok), len(scored)


def _sample_banner(db: Database, result: SampleResult, *, evaluated: bool) -> str:
    """The per-sample banner tail from the sample's promptlog perf and its scored result.

    No per-sample RATE: the cohort is the unit of scoring, so a sample carries only what is true
    of it alone — whether it reached an answer, whether it got there shakily, what it cost."""
    perf = live_prompt_perf(db)
    return report.render_banner(
        passed=result.passed,
        cause=_cause_word(result.cause),
        fragile=result.fragile,
        duration_s=round(perf.duration_ms / 1000),
        calls=perf.calls,
    )


def _sample_prompt_rows(db: Database) -> list[PromptLog]:
    """Every promptlog row for the sample (the main agent's + every micro-context's), oldest
    first — the ledger order the transcript walk interleaves the actors by.

    A SEEDED prior turn's rows are excluded (#1846): they are the world this sample was
    handed, not something it did, and a transcript that walked them would render an
    earlier turn's calls and system prompts as this sample's own."""
    with Session(db.engine) as session:
        rows = session.exec(select(PromptLog).order_by(col(PromptLog.timestamp).asc())).all()
    return [row for row in rows if not is_seeded_run(row.run_id)]


def _main_rows(rows: list[PromptLog]) -> list[PromptLog]:
    """The MAIN agent's rows — everything that is not a micro-context call (#1773). A
    micro-context's scoped turn is its own actor's input, never a turn of the conversation: left
    in, the classifier's rendered slice opened a phantom ``👤 user`` step ahead of the real one."""
    return [row for row in rows if (row.agent_name or "") not in MICRO_CONTEXT_PLACEMENTS]


def _build_transcript(
    db: Database,
    result: SampleResult,
    turns: list[tuple[str, str]],
    main_rows: list[PromptLog],
    rows: list[PromptLog],
    baseline: Baseline | None,
    case_id: str,
    sample_index: int,
    delivered: Sequence[str] = (),
) -> report.SampleTranscript:
    """Assemble the ``report.SampleTranscript`` for one sample from its turns + scored result."""
    if not turns:
        banner = _sample_banner(db, result, evaluated=False)
        return report.SampleTranscript(
            sample_index + 1, banner, [], placeholder=report.NO_TURNS_PLACEHOLDER
        )
    events, turn_to_event = _turns_to_events(
        turns, _thinking_by_content(main_rows), _micro_batches(rows)
    )
    placed, _leftover = _place_checks(result.checks, turns)
    turn_of_check = {id(check): turn for turn, checks in placed.items() for check in checks}
    checks = _build_check_views(result, turn_of_check, turn_to_event, baseline, case_id)
    passed_checks, total = _scored_counts(result)
    _record_case_prompts(case_id, sample_index, _system_prompts(rows))
    return report.build_sample(
        number=sample_index + 1,
        banner=_sample_banner(db, result, evaluated=True),
        events=events,
        checks=checks,
        run_close_score=f"{passed_checks}/{total}",
        rejected=rejected_draws(main_rows, delivered),
    )


# Every sample's system prompts, held per case until the case document renders them.
#
# A cohort's samples are handed the SAME prompts, so the document states each distinct one once
# rather than eighteen times (#1997) — but "distinct" is a fact about the whole case, and a
# sample can only be read while its own database is live.  So each sample deposits what it was
# given, keyed by the name the report knows it by, and the grouping happens at case close.
_case_prompts: dict[str, list[tuple[str, report.SystemPrompt]]] = {}


def _record_case_prompts(
    case_id: str, sample_index: int, prompts: Sequence[report.SystemPrompt]
) -> None:
    """Hold one sample's system prompts until its case closes."""
    label = f"{report.SAMPLE_ROW} {sample_index + 1}"
    _case_prompts.setdefault(case_id, []).extend((label, prompt) for prompt in prompts)


# Rendered sample blocks, held per case and written in SAMPLE order when the case closes.
# Samples may run concurrently (``EVAL_CONCURRENCY``), and appending as each one finishes
# would lay a case's ``<case_id>.md`` down in COMPLETION order — sample 4 above sample 2 —
# which is the one thing a report read top-to-bottom cannot do.  Buffering keeps the file
# byte-identical to a sequential run at any concurrency.  Emptied by the flush, so a run
# holds only the case in flight.
_sample_blocks: dict[str, list[tuple[int, str]]] = {}


def _record_sample_block(
    case_id: str, sample_index: int, transcript: report.SampleTranscript
) -> None:
    """Hold one rendered sample block until its case closes."""
    _sample_blocks.setdefault(case_id, []).append((sample_index, report.render_sample(transcript)))


def _flush_sample_blocks(case_id: str) -> None:
    """Append a case's held blocks to ``EVAL_REPORT_DIR/<case_id>.md`` in sample order."""
    blocks = _sample_blocks.pop(case_id, [])
    report_dir = os.environ.get("EVAL_REPORT_DIR")
    if not report_dir or not blocks:
        return
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{case_id}.md").open("a") as handle:
        for _, rendered in sorted(blocks):
            handle.write(rendered + "\n\n")


def _write_sample_report(
    db: Database,
    case_id: str,
    sample_index: int,
    *,
    result: SampleResult,
    reply: str = "",
    driven: Sequence[str] = (),
) -> None:
    """Append one sample's transcript-integrated block to ``EVAL_REPORT_DIR/<case_id>.md`` (#1725
    iteration-6). No-op off-report. Builds the report model from the persisted promptlog + the
    scored result, then renders it via ``report.render_sample``: EVERY sample folds whole under
    its banner (uniform collapse, #1753), and a no-turns (timeout) sample gets the honest
    placeholder so the report's sample count always matches N (F2). The on-disk ``.md`` keeps every
    sample's FULL transcript, byte-identical to what the assembler posts (#1759 — the one and only
    rendering, no compact/banner-only form)."""
    report_dir = os.environ.get("EVAL_REPORT_DIR")
    if not report_dir:
        return
    rows = _sample_prompt_rows(db)
    main_rows = _main_rows(rows)
    baseline = baseline_from_env()
    # Stamp fragile (same EVAL_REPORT_DIR gate as the artifact write) so it rides into the artifact.
    result.fragile = result.passed and sample_is_fragile(db)
    delivered = _delivered_replies(db)
    turns = _sample_turns(main_rows, reply, driven, delivered)
    transcript = _build_transcript(
        db, result, turns, main_rows, rows, baseline, case_id, sample_index, delivered
    )
    _record_sample_block(case_id, sample_index, transcript)


# How many times a sample is driven when the MODEL CALL ITSELF fails — the transport
# erroring, not Penny deciding anything.  Such a run ends with the model-error reply and
# no work done, which scored as a BEHAVIOURAL failure and dragged a case's mean down for
# a reason the model never controlled (observed: a learn → apply sample scored 0.22 that
# way, with every state check red because the collection was never created).  A retry
# gets a fresh server and a fresh DB — the DB path is keyed on the ATTEMPT
# (``_sample_db_path``), since nothing deletes the file and a reused path would re-seed
# over the failed attempt's rows and replay its turn — so the sample is driven from the
# same clean world rather than continuing on top of the failed turn.
_MODEL_CALL_ATTEMPTS = 3


class _ModelCallError(Exception):
    """Raised when a sample's reply is the model-error reply — infrastructure, not
    behaviour, so the sample is re-driven rather than scored."""


# What one sample does with the world built for it: drive it, and return its scored result.
# The last argument says whether another attempt remains if the MODEL CALL itself fails.
SampleDriver = Callable[[Penny, MockSignalServer, int, bool], Awaitable[SampleResult]]


async def _run_samples(
    make_config: Callable[..., Config],
    tmp_path,
    *,
    case_id: str,
    samples: int,
    drive: SampleDriver,
    attempts: int = 1,
    model: str = "",
) -> tuple[list[SampleResult], _Perf]:
    """Drive ``samples`` hermetic samples of one case; return their results and perf.

    The skeleton every runner shares, in ONE place — which is what makes concurrency a
    one-line change rather than an eleven-line one.  A sample gets its own mock Signal
    server (on its own OS-assigned port), its own DB, and its own real-model Penny, and
    ``drive`` supplies only what the case does with them.  Samples therefore share no
    state by construction, which is what lets up to ``EVAL_CONCURRENCY`` of them run at
    once; the returned results stay in SAMPLE order whatever finished first, so a case's
    pass-rate and its report read identically at any concurrency.

    ``attempts`` above 1 re-drives a sample whose MODEL CALL failed (``_ModelCallError``)
    from a fresh world; a sample that exhausts its attempts is dropped rather than scored,
    exactly as before.
    """
    limit = asyncio.Semaphore(EVAL_CONCURRENCY)
    perf = _Perf()

    async def _sample(sample_index: int) -> SampleResult | None:
        async with limit:
            for attempt in range(attempts):
                server = MockSignalServer()
                await server.start()
                try:
                    config = _real_model_config(
                        make_config,
                        signal_api_url=f"http://localhost:{server.port}",
                        db_path=_sample_db_path(tmp_path, case_id, sample_index, attempt),
                        model=model,
                    )
                    async with eval_penny(config, server) as penny:
                        result = await drive(penny, server, sample_index, attempt + 1 < attempts)
                        perf.add(live_prompt_perf(penny.db))
                        return result
                except _ModelCallError:
                    print(
                        f"  ↻ {case_id} sample {sample_index}: the model call failed — "
                        f"retrying ({attempt + 1} of {attempts})"
                    )
                finally:
                    await server.stop()
        return None

    driven = await asyncio.gather(*(_sample(index) for index in range(samples)))
    _flush_sample_blocks(case_id)
    results = [result for result in driven if result is not None]
    # What the case ASKED for beside what it got, recorded whatever happens next — this is
    # the only place both numbers exist, and a case that dies on its threshold must still
    # contribute its cohort to the run's health block.
    run_health.record_cohort(case_id, intended=samples, completed=len(results))
    return results, perf


# ── The cohort: one request, K phrasings, pooled (#1994/#1995) ───────────────
#
# A ported case reads as `<priors> / <trigger the action> / <assertions>`, and everything
# below is the MECHANISM behind the middle line: which arm a sample runs, what it left
# behind, whether its measured turn ran at all, and how the case's three sections are
# assembled once every sample is in.  What is asserted and what is measured stay with the
# CASE, so porting the next one is writing arms + fixtures + claims, never harness.

# The model a ported case is measured on — one cohort per id, so a case parametrized over it
# produces one score and one threshold set per model, which is what a per-model ceiling needs.
#
# WHICH models exist is the ROSTER's business (`roster.py`, #1999): it is the one configured
# list, it carries each model's preferred provider, and a remote run refuses to start unless it
# names at least two.  Which of them a RUN measures is resolved before pytest starts and
# arrives as `LLM_MODEL`, so this reads that rather than re-parsing `EVAL_MODELS` — the two
# would be the same variable meaning two different things.  Two models is therefore two runs,
# each self-describing in its own manifest.
EVAL_MODELS = [os.environ.get("LLM_MODEL", "")]

# Why a sample is not counted.  The one structural completeness condition every case shares:
# a sample's database exists from sample START, so a file is not evidence that anything ran.
NO_MEASURED_TURN = "the measured turn never ran — the sample carries only its seeded world"
NO_REPLY = "the measured turn produced no reply"

# The turn roles that count as WORLD for a provenance claim.  Assistant turns are absent by
# design — a value Penny invents early in a turn rides into the message history and would
# then source itself from her own account of it, which is how a fabrication launders itself.
# The SYSTEM prompt IS included, which is a correction: excluding it reported the CURRENT
# DATE as a fabrication in 3 of 18 measured samples, because an entry keyed by the day it was
# saved reads that date off the self-state header and nowhere else.  It carries none of the
# laundering risk an assistant turn does — framework-rendered from the registry and the
# ledger, and rendered BEFORE the turn acts, so it cannot contain anything this turn invented.
_GIVEN_ROLES = frozenset({"user", "tool", "system"})


def measured_turn_ran(db: Database) -> bool:
    """Whether THIS sample's own turn reached the model at all.

    ``live_prompts`` already excludes every seeded prior turn, so an empty window is the dead
    sample exactly: the seeded round is present and nothing was added to it.  Read
    structurally rather than inferred from a score, because a dead sample scores like a
    behavioural failure and would otherwise be pooled as wild variance."""
    return bool(live_prompts(db))


def given_to_the_model(db: Database) -> str:
    """Everything this sample's turn was GIVEN, as one blob — the world a provenance claim
    reads against (#1994)."""
    return "\n".join(
        str(turn.get("content") or "")
        for turn in _iter_prompt_messages(db)
        if turn.get("role") in _GIVEN_ROLES
    )


def reply_embedding(db: Database, reply: str) -> list[float] | None:
    """The vector of the reply this sample was scored on.

    Every conversational send is embedded at egress, so the cohort's reply spread costs no
    model call — it is a read of a column production already fills.  Looked up by the reply's
    own text rather than by a recency window, so it is the SCORED reply's vector."""
    row = db.messages.find_outgoing_by_content(reply)
    if row is None or row.embedding is None:
        return None
    return deserialize_embedding(row.embedding)


def _routine_records(db: Database) -> list[eval_cohort.RoutineRecord]:
    """Every routine the round minted, as the registry holds it.

    ``names_a_destination`` reads the ATTACHMENT MARK — set by distillation on any leaf whose
    demonstrated value named one of Penny's own collections — so it is true of a write, of a
    log append, and of a plugin verb nobody here has heard of, and false of a routine that
    only browses.  Never keyed to a tool NAME: a skill is an arbitrary tool sequence."""
    return [
        eval_cohort.RoutineRecord(
            name=skill.name,
            shape=render_skill_shape(skill),
            open_parameters=sorted(
                {
                    substitution.parameter
                    for step in steps_from_json(skill.steps)
                    for substitution in step.substitutions
                    if substitution.kind == SkillSubKind.HOLE and substitution.parameter is not None
                }
            ),
            names_a_destination=any(
                substitution.attachment
                for step in steps_from_json(skill.steps)
                for substitution in step.substitutions
            ),
        )
        for skill in db.skills.list_all()
    ]


def _stored_entries(db: Database) -> list[eval_cohort.StoredEntry]:
    """Every COLLECTION entry this sample wrote, wherever it landed — the run-id stamp
    answering "what did she store", so a case never has to guess which container a framed
    round used.  Collections only: the browse log carries the fetched page, and counting that
    as a stored fact would let "she wrote it down" pass on a run that only read a page."""
    written: list[eval_cohort.StoredEntry] = []
    for row in db.memories.list_all():
        if row.type != MemoryType.COLLECTION:
            continue
        memory = db.memory(row.name)
        entries = memory.read_all() if memory is not None else []
        written += [
            eval_cohort.StoredEntry(collection=row.name, key=entry.key, content=entry.content)
            for entry in entries
            if _written_by_a_live_run(entry)
        ]
    return written


def _written_by_a_live_run(entry) -> bool:
    """Whether THIS sample put an entry's current value there — created by a live run, or last
    rewritten by one.  Both stamps, because an edit of a seeded entry moves only the second."""
    stamps = (entry.created_by_run_id, entry.last_written_by_run_id)
    return any(stamp is not None and not is_seeded_run(stamp) for stamp in stamps)


def _framed_container(db: Database) -> str | None:
    """The container the round was framed on, read off the move that settled it.

    From the MACHINE rather than guessed from the collections that appeared, because the claim
    is whether the write landed where the turn was told to put it, and only the framing says
    where that was."""
    latest = db.machine.latest_transition()
    if latest is None or latest.skill_frame is None:
        return None
    return RoundFraming.model_validate_json(latest.skill_frame).container


def _scheduled_by_this_round(db: Database, before: set[str]) -> list[str]:
    """Collections this round created that carry a schedule or a notify flag.

    Scored against what the turn PRODUCED rather than the whole store: a seeded collection's own
    cadence predates the round, and failing on it would report the fixtures as her doing."""
    return sorted(
        row.name
        for row in db.memories.list_all()
        if row.name not in before and (row.schedule is not None or row.notify)
    )


def _enacting_name(call: str) -> str | None:
    """One logged call's enacting-tool name, or ``None`` where it enacted nothing.

    The name is normalised through ``strip_harmony_control_tokens`` — the SAME function
    production uses, not a second spelling of it.  That sanitiser runs at the boundary where a
    tool name is read off the model response (``LlmToolCallFunction.name``), so every downstream
    consumer — registry lookup, done-detection, dedup, result framing — already sees the clean
    identifier.  This eval was the only consumer reading the raw one.

    What that cost: two samples in one run logged `collection_write<|channel|>commentary`.  The
    runtime dispatched them fine and their entries are in the store, but a membership test on the
    raw name read them as a tool nobody has heard of and dropped them, so the sequence rendered
    as `browse` alone and a correct sample was reported as an outlier for a divergence that never
    happened.  Re-implementing the strip here would leave the eval measuring a normalisation
    production does not do the moment either spelling changed."""
    return name if (name := strip_harmony_control_tokens(call)) in ENACTING_TOOLS else None


def _machine_walk(db: Database) -> str:
    """The machine's walk this sample, oldest move first — ``idle→learn, learn→apply``."""
    moves = reversed(db.machine.recent_transitions(limit=20))
    return ", ".join(f"{move.from_state}→{move.to_state}" for move in moves) or "no move"


def _observe_sample(
    db: Database, *, name: str, phrasing: str, reply: str, before: set[str]
) -> eval_cohort.SampleObservation:
    """Read everything one sample left behind, while its database is still live.

    Gated for completeness FIRST, so a dead sample carries no facts to be pooled by accident.
    """
    exclusion = _exclusion(db, reply)
    if exclusion is not None:
        return eval_cohort.SampleObservation(
            name=name, phrasing=phrasing, complete=False, exclusion=exclusion
        )
    landed = db.machine.latest_transition()
    return eval_cohort.SampleObservation(
        name=name,
        phrasing=phrasing,
        landed=landed.to_state if landed else None,
        walk=_machine_walk(db),
        routines=_routine_records(db),
        entries=_stored_entries(db),
        tool_sequence=[
            tool
            for run in chat_run_tool_sequences(db)
            for tool in (_enacting_name(call) for call in run)
            if tool is not None
        ],
        reply=reply,
        reply_embedding=reply_embedding(db, reply),
        given=given_to_the_model(db),
        container=_framed_container(db),
        scheduled=_scheduled_by_this_round(db, before),
    )


def _exclusion(db: Database, reply: str) -> str | None:
    """Why this sample cannot be counted, or ``None`` when it can."""
    if not measured_turn_ran(db):
        return NO_MEASURED_TURN
    if not reply.strip():
        return NO_REPLY
    return None


def _reporting_model() -> str:
    """Which model produced this run's numbers, off the run manifest — empty only off-report,
    where nothing is recorded anyway.  A ceiling recorded without it is unusable, since two
    models differ several-fold on the same feature."""
    run = eval_artifacts.active_run()
    return run.manifest.model if run is not None else ""


def _no_scorer(db: Database, before: set[str], reply: str) -> list[Check]:
    """A ported case is graded from its cohort's CLAIMS, not from a per-sample callback, so the
    sample itself scores nothing at drive time."""
    return []


def _phrasing_label(phrasings: Sequence[str], sample_index: int, per_phrasing: int) -> str:
    """What this sample is called in the report — which of the case's wordings it ran."""
    if len(phrasings) > 1:
        return f"phrasing {sample_index // per_phrasing + 1}"
    return "the ask"


# The world a case declares nothing about — a cohort still needs one for its claims to read, and
# this one matches nothing, so a claim made against it is vacuous rather than wrong.
_NO_WORLD = World(name="unspecified", pages=(), keeps=(), excludes=())


@dataclass
class _PendingCase:
    """One CASE's drive, waiting for the test body to make its claims.

    The report cannot be written when the drive returns, since the claims are made after it in
    the case body.  So the drive is parked here and the case is finished at fixture teardown."""

    case_id: str
    family: str | None
    module: str
    min_pass_rate: float | None
    gate_pathology_excluded: bool
    intended: int = 0
    cohort: Cohort | None = None
    results: list[SampleResult] = field(default_factory=list)
    perf: _Perf = field(default_factory=_Perf)
    driven: int = 0

    def add(
        self, cohort: Cohort, results: Sequence[SampleResult], perf: _Perf, *, intended: int
    ) -> None:
        """Park the case's drive until its body has made its claims."""
        assert self.cohort is None, f"{self.case_id}: a case drives its cohort exactly once"
        self.cohort = cohort
        self.results += results
        self.perf = perf
        self.driven += len(results)
        self.intended += intended

    @property
    def driven_cohort(self) -> Cohort:
        """The cohort this case drove — where its assertions and measured features live."""
        assert self.cohort is not None, f"{self.case_id}: finished without a drive"
        return self.cohort

    def finish(self) -> None:
        """Deal the claims back out to their samples, record, report, and gate."""
        self._grade()
        observations = self._observations()
        # Computed ONCE and used by both halves: the document renders the standings, and the
        # ASSEMBLER needs to know which sample to expand in the posted comment.  Deriving them
        # twice would let the map and the expanded sample disagree about which one is modal.
        standings = eval_cohort.standings(observations, self.driven_cohort.features)
        _record_case_report(self.driven_cohort, observations, standings, self.perf, self.driven)
        _finish_case(
            self.case_id,
            self.family,
            self.module,
            self.results,
            self.perf,
            self.min_pass_rate,
            self.gate_pathology_excluded,
            self.driven,
            self.intended,
            _expandable(standings),
            Counter(standing.standing.value for standing in standings),
            _variance_readings(observations, self.driven_cohort.features),
        )

    def _grade(self) -> None:
        """Every claim the cohort answered, redistributed to the sample that answered it — the
        one seam where cohort-level claims meet per-sample grading."""
        by_sample = _cohort_checks(self.driven_cohort)
        for result in self.results:
            if result.observation is not None:
                result.adopt(by_sample.get(result.observation.name, []))

    def _observations(self) -> list[eval_cohort.SampleObservation]:
        """Every sample the case drove."""
        return list(self.driven_cohort.samples)


def _variance_readings(
    samples: Sequence[eval_cohort.SampleObservation], features: Sequence[eval_cohort.Feature]
) -> list[eval_artifacts.VarianceReading]:
    """What the case MEASURED, in the record, so the run header can roll a spread up across
    cases — the per-case document computes this from the cohort, but the assembler runs as its
    own process over the report dir and has no cohort to read."""
    pooled = eval_cohort.pool(samples, features)
    return [
        eval_artifacts.VarianceReading(
            name=feature.name,
            entropy=feature.entropy,
            saturated=feature.saturated,
            distinct=feature.distinct,
        )
        for feature in pooled.features
    ]


def _expandable(standings: Sequence[eval_cohort.SampleStanding]) -> list[int]:
    """The 1-based positions of the samples the posted comment carries in full.

    Only the representative: an outlier is communicated by its DIVERGENCE, which is a few rows,
    and a typical sample by the fact that it agreed.  Every sample stays whole in the artifact."""
    return [index + 1 for index, s in enumerate(standings) if s.worth_opening]


def _finish_case(
    case_id: str,
    family: str | None,
    module: str,
    results: Sequence[SampleResult],
    perf: _Perf,
    min_pass_rate: float | None,
    gate_pathology_excluded: bool,
    driven: int,
    intended: int,
    expand_samples: Sequence[int] = (),
    standing_counts: Mapping[str, int] | None = None,
    variance: Sequence[eval_artifacts.VarianceReading] = (),
) -> None:
    """Record the case's artifact, print its perf line, and apply its gate."""
    _record_unported_prompts(case_id, driven)
    eval_artifacts.record_case(
        case_id=case_id,
        family=family,
        module=module,
        results=results,
        perf=perf,
        min_pass_rate=min_pass_rate,
        gate_pathology_excluded=gate_pathology_excluded,
        expand_samples=expand_samples,
        standing_counts=standing_counts,
        variance=variance,
    )
    perf.report(case_id, driven)
    _assert_threshold(
        case_id,
        list(results),
        min_pass_rate,
        intended=intended,
        gate_pathology_excluded=gate_pathology_excluded,
    )


def _record_unported_prompts(case_id: str, driven: int) -> None:
    """A case with no cohort still states its system prompts once. No-op off-report.

    A ported case has already popped its prompts into the case document by the time this runs,
    so this writes only for a case that never built one — which is what keeps the shared-once
    rendering true of the whole suite rather than only of the part that has been ported."""
    prompts = _case_prompts.pop(case_id, [])
    if not prompts:
        return
    eval_artifacts.record_case_report(
        case_id,
        "",
        report.render_prompt_variants(report.prompt_variants(prompts, total=driven)),
    )


def _cohort_checks(cohort: Cohort) -> dict[str, list[Check]]:
    """The case's claims, redistributed to the samples that answered them.

    The graded machinery is per sample (a sample's score, the report's per-sample cells), and
    the claims are made over the whole cohort — so the answers are dealt back out by sample
    name.  One conversion, at the one seam where the two shapes meet."""
    by_sample: dict[str, list[Check]] = {}
    for claim in cohort.claims:
        for outcome in claim.outcomes:
            by_sample.setdefault(outcome.sample, []).append(
                Check(
                    claim.label,
                    outcome.ok,
                    rationale=outcome.rationale,
                    kind=claim.kind,
                    anchor=REPLY_ANCHOR if claim.kind == "reply" else None,
                )
            )
    return by_sample


def _record_case_report(
    cohort: Cohort,
    samples: Sequence[eval_cohort.SampleObservation],
    standings: Sequence[eval_cohort.SampleStanding],
    perf: _Perf,
    driven: int,
) -> None:
    """Assemble and write the case's document — its three sections, then everything its samples
    SHARE: the one ask in its several wordings, the one world, the system prompts, and the map
    saying which samples to open. No-op off-report.

    Assembled HERE rather than in the report layer because this is where the three halves meet:
    the claims come from the case body, the pooled variance from the observations, and the
    prompts from what each sample was handed while its database was live."""
    variance = eval_cohort.pool(samples, cohort.features)
    sections = report.CaseSections(
        case_id=cohort.case_id,
        model=cohort.model,
        assertions=eval_assertions.assertion_rows(cohort.claims),
        variance=variance,
        cost=eval_cohort.per_sample_cost(
            samples=driven,
            calls=perf.calls,
            duration_ms=perf.duration_ms,
            input_tokens=perf.input_tokens,
            output_tokens=perf.output_tokens,
            reasoning_tokens=perf.reasoning_tokens,
        ),
    ).render()
    prompts = _case_prompts.pop(cohort.case_id, [])
    eval_artifacts.record_case_report(
        cohort.case_id,
        sections,
        report.render_prompt_variants(report.prompt_variants(prompts, total=len(samples))),
        report.render_case_tail(
            phrasings=cohort.phrasings,
            world=cohort.world.render(),
            world_facts=report.WorldFacts(*cohort.world.counts),
            outliers=list(enumerate(standings, start=1)),
            everywhere_distinct=eval_cohort.everywhere_distinct(samples, cohort.features),
        ),
    )


# A chat-eval runner: (case_id, message, scorer, optional seeder) -> asserts threshold.
ChatEval = Callable[..., Awaitable["Cohort"]]


def _conversation_turns(message: str | None, messages: Sequence[str] | None) -> list[str]:
    """The user turns to drive, in order — exactly one of ``message`` (a single turn) or
    ``messages`` (a multi-turn conversation) must be given.  A conversation drives the turns
    sequentially against the same Penny; Penny sees each earlier turn via the DB history it
    reconstructs, so a later turn can build on (or adjust) what an earlier one discussed."""
    if message is not None and messages is None:
        return [message]
    if messages is not None and message is None:
        if not messages:
            raise ValueError("chat_eval `messages` must contain at least one turn")
        return list(messages)
    raise ValueError("chat_eval needs exactly one of `message` or `messages`")


async def _seed_sample(
    penny: Penny,
    *,
    seed: Seeder | None,
    seed_skills: Sequence[SkillDraft] | None,
    browse: list[CannedPage] | None,
    prepare: Preparer | None,
) -> None:
    """Lay a sample's world down before its first turn: the user, the case's own seed,
    the embeddings those seeds need, any fixture skills, the canned browse, and the
    case's late hook."""
    seed_user(penny.db)
    if seed is not None:
        seed(penny.db)
    await _embed_seeds(penny)
    if seed_skills:
        await _seed_eval_skills(penny, seed_skills)
    if browse is not None:
        install_browse(penny, browse)
    if prepare is not None:
        prepare(penny)


async def _drive_turns(
    server: MockSignalServer, turns: Sequence[str], *, timeout: float, retryable: bool
) -> str:
    """Push each user turn and wait for its reply, returning the LAST one.

    A model-error reply raises :class:`_ModelCallError` while an attempt remains — that
    reply is the transport failing, not Penny deciding anything, so the sample is
    re-driven from a clean world rather than scored as behaviour."""
    reply = ""
    for turn in turns:
        await server.push_message(sender=TEST_SENDER, content=turn)
        response = await server.wait_for_message(timeout=timeout)
        reply = str(response.get("message", ""))
        if reply == PennyResponse.AGENT_MODEL_ERROR and retryable:
            raise _ModelCallError
    return reply


def _scored_sample(
    db: Database,
    before: set[str],
    reply: str,
    score: Scorer,
    wrapper: _InjectingClient | None,
) -> SampleResult:
    """One sample's result from its scorer — graded ``Check``s or binary failure
    strings — with the forced-bail guard folded in when the case wrapped the client."""
    scored = list(score(db, before, reply))
    if _scorer_is_graded(scored):
        guards = [_bail_fired_check(wrapper.bail_injected)] if wrapper is not None else []
        return _guarded_graded(scored, guards)
    fails = [s for s in scored if isinstance(s, str)]  # binary scorer
    if wrapper is not None and not wrapper.bail_injected:
        fails.append("forced bail never fired — contract not exercised")
    return SampleResult.binary(fails)


async def _drive_sample(
    penny: Penny,
    server: MockSignalServer,
    *,
    case_id: str,
    sample_index: int,
    turns: Sequence[str],
    score: Scorer,
    wrap_client: Callable[[LlmClient], _InjectingClient] | None,
    timeout: float,
    retryable: bool,
    observe: Callable[[Database, str, set[str]], eval_cohort.SampleObservation] | None = None,
) -> SampleResult:
    """ONE attempt at one sample against an already-seeded Penny: drive the turns,
    score them, write the sample's report block and dump its thinking.

    A timeout counts as a failed sample, not a crash, and still emits its placeholder
    block so the transcript's sample count always matches N (#1725/F2).  Raises
    :class:`_ModelCallError` when the model call itself failed and an attempt remains."""
    # A recovery case wraps the chat agent's model client to force one bad response
    # (e.g. a bracket-wrapped key) deterministically.  Keep the wrapper: its
    # ``bail_injected`` flag is the only proof the sabotage fired — the raw response is
    # persisted inside the REAL client before the wrapper mutates it, so the promptlog
    # never shows the injected form and can't be probed for it.
    wrapper: _InjectingClient | None = None
    if wrap_client is not None:
        wrapper = wrap_client(penny.chat_agent._model_client)
        penny.chat_agent._model_client = wrapper
    before = collection_names(penny.db)
    reply = ""
    try:
        reply = await _drive_turns(server, turns, timeout=timeout, retryable=retryable)
        result = _scored_sample(penny.db, before, reply, score, wrapper)
        _stamp_cause(penny.db, result)
        _write_sample_report(
            penny.db, case_id, sample_index, result=result, reply=reply, driven=turns
        )
    except TimeoutError:
        result = SampleResult.binary(["no reply within timeout"])
        _stamp_cause(penny.db, result, timed_out=True)
        _write_sample_report(penny.db, case_id, sample_index, result=result)
    # The observation is read HERE, while this sample's own database is still open — the only
    # moment what the round left behind is available at all.  A timed-out sample reaches this
    # line too, so it is EXCLUDED by name rather than silently absent from the pool.
    if observe is not None:
        result.observation = observe(penny.db, reply, before)
    _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
    return result


@pytest.fixture
def chat_eval(make_config: Callable[..., Config], tmp_path, request) -> Iterator[ChatEval]:
    """Drive the real chat flow N times for one request and return its COHORT.

    A PORTED case (#1995) passes ``ask`` + ``world`` and gets a :class:`Cohort` back to make
    its claims against; the three-section report is assembled at teardown, once the test body
    has made them.  A case not yet ported passes ``message``/``messages`` + ``score`` and is
    driven and gated inline, exactly as before.

    Each sample is fully hermetic — its own mock Signal server, DB, and
    real-model Penny: seed user (+ any case seed), embed the seeds, push the
    turn(s), wait for each reply, then score persisted state against the LAST
    reply.  A per-sample server is essential: a shared one leaks a prior
    sample's shut-down channel, which then errors on the next sample's
    broadcast.  A timeout on any turn counts as a failed sample, not a crash.

    Single-message vs. conversation: pass ``message`` for one turn, or
    ``messages`` for a discuss-then-adjust conversation (see
    ``_conversation_turns``).

    ``seed_skills`` lays fixture skills into the registry with real description
    embeddings (``_seed_eval_skills``, shared with the classifier runner) — what
    a case needs when its ONE turn stands on a routine an earlier round already
    taught, rather than driving that round again.
    """
    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        message: str | None = None,
        messages: Sequence[str] | None = None,
        score: Scorer | None = None,
        ask: str | None = None,
        also_phrased: Sequence[str] = (),
        world: World | None = None,
        model: str = "",
        samples_per_phrasing: int = 0,
        seed: Seeder | None = None,
        seed_skills: Sequence[SkillDraft] | None = None,
        browse: list[CannedPage] | None = None,
        prepare: Preparer | None = None,
        wrap_client: Callable[[LlmClient], _InjectingClient] | None = None,
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
        timeout: float = 120.0,
        family: str | None = None,
        gate_pathology_excluded: bool = False,
    ) -> Cohort:
        """Drive one request and return its COHORT (ported case), or score each sample through
        a callback (a case not yet ported).

        A ported case passes ``ask`` — with ``world`` for the pages it is answered against, and
        ``also_phrased`` for the other wordings of that SAME request — and gets a
        :class:`Cohort` back to make its claims against.  Analysis then runs over the complete
        set, which is what the variance statistics want and what a per-sample callback cannot
        see.  A case with no ``ask`` takes the ``score`` path unchanged."""
        eval_artifacts.begin_case(case_id)
        phrasings = [ask, *also_phrased] if ask is not None else []
        per_phrasing = samples_per_phrasing or samples
        # A cohort's N is the sum of its phrasings' own counts.  Deliberately NOT scaled by
        # anything ambient: a recorded ceiling is `(feature, model, N, value)` and normalised
        # entropy is biased upward at small N, so an N that drifted with an environment
        # variable would silently make every recorded threshold incomparable.
        spoken = [phrase for phrase in phrasings for _ in range(per_phrasing)]
        turns = [] if spoken else _conversation_turns(message, messages)
        driven = len(spoken) if spoken else samples
        pages = list(world.pages) if world is not None else browse

        pending = (
            _cohorts.setdefault(
                case_id,
                _PendingCase(
                    case_id=case_id,
                    family=family,
                    module=request.module.__name__,
                    min_pass_rate=min_pass_rate,
                    gate_pathology_excluded=gate_pathology_excluded,
                ),
            )
            if spoken
            else None
        )

        def _observe(
            sample_index: int,
        ) -> Callable[[Database, str, set[str]], eval_cohort.SampleObservation]:
            phrasing = _phrasing_label(phrasings, sample_index, per_phrasing)
            name = f"{case_id}-{sample_index + 1} ({phrasing})"
            return lambda db, reply, before: _observe_sample(
                db, name=name, phrasing=phrasing, reply=reply, before=before
            )

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            await _seed_sample(
                penny, seed=seed, seed_skills=seed_skills, browse=pages, prepare=prepare
            )
            return await _drive_sample(
                penny,
                server,
                case_id=case_id,
                sample_index=sample_index,
                turns=[spoken[sample_index]] if spoken else turns,
                score=score or _no_scorer,
                wrap_client=wrap_client,
                timeout=timeout,
                retryable=retryable,
                observe=_observe(sample_index) if spoken else None,
            )

        results, perf = await _run_samples(
            make_config,
            tmp_path,
            case_id=case_id,
            samples=driven,
            drive=_drive,
            attempts=_MODEL_CALL_ATTEMPTS,
            model=model,
        )
        if not spoken:
            # A case that has not been ported is driven and gated inline, exactly as before, and
            # gets back a cohort nobody observed — an empty one rather than ``None``, so a ported
            # case never carries a narrowing assert that every future port would copy.
            _finish_case(
                case_id,
                family,
                request.module.__name__,
                results,
                perf,
                min_pass_rate,
                gate_pathology_excluded,
                driven,
                samples,
            )
            return Cohort(case_id=case_id, model=model, world=_NO_WORLD, samples=[])
        cohort = Cohort(
            case_id=case_id,
            model=model or _reporting_model(),
            world=world if world is not None else _NO_WORLD,
            samples=[r.observation for r in results if r.observation is not None],
            phrasings=[
                (_phrasing_label(phrasings, index * per_phrasing, per_phrasing), text)
                for index, text in enumerate(phrasings)
            ],
        )
        assert pending is not None
        pending.add(cohort, results, perf, intended=driven)
        return cohort

    yield _run
    # The case's claims are made in the TEST BODY, after the drive returns — so the report is
    # assembled here, once the body has had its say.  A case that drove no cohort finished
    # inline above and has nothing pending.
    for pending in _cohorts.values():
        pending.finish()
    _cohorts.clear()


# A collector-eval runner: (case_id, collection, seed, score, snapshot) -> asserts.
CollectorEval = Callable[..., Awaitable[None]]


@pytest.fixture
def collector_eval(make_config: Callable[..., Config], tmp_path, request) -> CollectorEval:
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
        family: str | None = None,
    ) -> None:
        eval_artifacts.begin_case(case_id)

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
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
            scored = list(score(penny.db, before, sent))
            if _scorer_is_graded(scored):
                result = _guarded_graded(scored, [])
            else:
                result = SampleResult.binary([s for s in scored if isinstance(s, str)])
            _stamp_cause(penny.db, result)
            _write_sample_report(penny.db, case_id, sample_index, result=result)
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

        results, perf = await _run_samples(
            make_config, tmp_path, case_id=case_id, samples=samples, drive=_drive
        )
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
            min_pass_rate=min_pass_rate,
        )
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate, intended=samples)

    return _run


class CycleCall(NamedTuple):
    """One tool call a collector cycle made — its name and the arguments it carried.

    The arguments travel as the mapping the ledger stores, because a scorer asks things of
    them a typed shape could not answer: WHICH page a browse went to is a question about a
    tool nobody enumerated, and a skill is an arbitrary sequence of them."""

    tool: str
    arguments: dict


@dataclass
class CycleObservation:
    """What ONE collector cycle did, read off persisted state after it closed (#1905).

    A watch is only a watch across cycles — "it changed" and "it already told me" are
    claims about the SECOND run, not the first — so a case that drives more than one cycle
    needs each cycle's footprint kept apart rather than a single end-state snapshot in
    which the two have already merged.

    ``before``/``after`` are the collection's entries as the cycle found and left them,
    ``sent`` what it put on the SEND QUEUE (read explicitly — see ``queued_sends``),
    ``calls`` what it called in order, ``served`` the page text the browse tool actually
    returned, and ``outcome``/``reason`` the run record it closed with (a write-gate STOP
    names itself in the reason).

    ``served`` exists because a multi-cycle case installs a DIFFERENT browse register per
    cycle, and every claim the case makes rests on each cycle having been shown its own.
    A register that failed to swap is invisible in every other reading — the model would
    behave correctly for the page it saw, and the whole case would report as the model
    getting it wrong."""

    index: int
    before: dict[str, str]
    after: dict[str, str]
    sent: list[str]
    calls: list[CycleCall]
    served: list[str]
    outcome: str | None
    reason: str | None

    @property
    def changed(self) -> bool:
        """Whether the cycle moved the collection's durable state."""
        return self.after != self.before

    @property
    def tools(self) -> list[str]:
        """The names of the calls it made, in order."""
        return [call.tool for call in self.calls]


def queued_sends(db: Database, collection: str) -> list[str]:
    """Every message a collection has put on the send queue, oldest first — pending and
    already-delivered alike.

    Read EXPLICITLY here rather than through ``pending_items`` because the queue is the
    harness's known blind spot: a collector cycle ENQUEUES, and the drainer that delivers
    is a separate schedule that may or may not have run by the time a scorer looks.  A
    pending-only read therefore reports a delivered notification as silence, which is the
    one thing a notify contract must never get wrong."""
    return [row.content for row in _send_queue_rows(db) if row.collection == collection]


def send_queue_mechanisms(db: Database) -> list[str]:
    """The mechanism behind every message on the send queue, oldest first — what a
    "nothing else spoke" claim is read from."""
    return [row.collection for row in _send_queue_rows(db)]


def _send_queue_rows(db: Database) -> list[SendQueueItem]:
    """Every send-queue row that IS or WILL BE a message to the user, oldest first.

    Cancelled rows are excluded structurally, as the store's own readers exclude them
    (#1634): a cancelled row was never sent, so counting one as a notification would
    report a message the user never got."""
    with Session(db.engine) as session:
        return list(
            session.exec(
                select(SendQueueItem)
                .where(col(SendQueueItem.cancelled_at).is_(None))
                .order_by(col(SendQueueItem.created_at).asc())
            ).all()
        )


def pages_served(db: Database) -> list[str]:
    """Every page the browse tool has returned this sample, oldest first — read off the
    browse-results log, which is where the tool journals what it actually fetched.

    The harness's own integrity read: a case that installs a different register per cycle
    is claiming each cycle saw a different world, and this is the only place that claim
    can be checked against what the tool really returned."""
    memory = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    entries = memory.read_all() if memory is not None else []
    return [entry.content for entry in entries]


def _live_run_ids(db: Database) -> set[str]:
    """Every run id this sample has written — read through ``live_prompts``, so the seeded
    ledger is excluded by the one chokepoint rather than by remembering to."""
    return {row.run_id for row in live_prompts(db, _PERF_WINDOW) if row.run_id}


def _observe_cycle(
    db: Database,
    collection: str,
    *,
    index: int,
    before: dict[str, str],
    sent_before: int,
    served_before: int,
    runs_before: set[str],
) -> CycleObservation:
    """One cycle's footprint, composed from the state it left: what it wrote, what it
    queued, what it called, what it was SERVED, and the run record it closed with."""
    runs = _live_run_ids(db) - runs_before
    rows = [row for row in live_prompts(db, _PERF_WINDOW) if row.run_id in runs]
    closed = next((row for row in rows if row.run_outcome is not None), None)
    return CycleObservation(
        index=index,
        before=before,
        after=collection_entries(db, collection),
        sent=queued_sends(db, collection)[sent_before:],
        calls=_ordered_calls(rows),
        served=pages_served(db)[served_before:],
        outcome=closed.run_outcome if closed is not None else None,
        reason=closed.run_reason if closed is not None else None,
    )


def _ordered_calls(rows: list[PromptLog]) -> list[CycleCall]:
    """The calls one run's rows carry, oldest first — each with the arguments it was made
    with, decoded from the JSON string the wire stores them as."""
    return [
        CycleCall(
            tool=tool_call_name(call),
            arguments=_decoded_arguments(call.get("function", {}).get("arguments")),
        )
        for row in sorted(rows, key=lambda row: row.timestamp)
        for call in _response_tool_calls(row)
    ]


def _decoded_arguments(raw: object) -> dict:
    """One logged call's arguments as a mapping — an unparseable payload reads as no
    arguments rather than raising, since a scorer asking WHICH page was fetched wants the
    calls that do carry one."""
    if not isinstance(raw, str):
        return raw if isinstance(raw, dict) else {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


# A cycles scorer reads the whole DB plus each cycle's own footprint, in order.  Graded
# only — every claim about a multi-cycle watch is per-cycle, so binary all-or-nothing would
# collapse "it fetched but never spoke" and "it did nothing at all" into one number.
CyclesScorer = Callable[[Database, "list[CycleObservation]"], "list[Check]"]
CollectorCyclesEval = Callable[..., Awaitable[None]]


class _DrivenCycles(NamedTuple):
    """What driving a case's cycles produced: each cycle's footprint, and whether the
    dispatcher actually RAN every one of them.

    ``ran`` is the harness half.  ``run_for`` refuses WITHOUT running a cycle at all when
    the collection is missing, archived, or carries no usable program — and a refusal
    leaves every check failing on a world the model never saw, which the failure-cause
    partition would then tag behavioral.  So the refusal travels as its own guard rather
    than as a puzzling zero.

    A refusal is read STRUCTURALLY — the cycle produced no run of its own — rather than
    from the returned flag, which is also false for a cycle that really ran and failed."""

    observed: list[CycleObservation]
    ran: bool
    refusal: str | None


async def _drive_cycles(
    penny: Penny,
    collection: str,
    cycles: Sequence[list[CannedPage]],
) -> _DrivenCycles:
    """Run the real collector cycle once per entry in ``cycles``, each against that
    entry's own browse register, observing what each one left behind.

    The register is re-installed between cycles because that is the whole point: the
    world MOVED between two runs of the same watch, and what the second cycle does about
    it is the contract."""
    observed: list[CycleObservation] = []
    refusal: str | None = None
    for index, pages in enumerate(cycles):
        install_browse(penny, pages)
        before = collection_entries(penny.db, collection)
        sent_before = len(queued_sends(penny.db, collection))
        served_before = len(pages_served(penny.db))
        runs_before = _live_run_ids(penny.db)
        _, message = await penny.collector.run_for(collection)
        if _live_run_ids(penny.db) == runs_before and refusal is None:
            refusal = message
        observed.append(
            _observe_cycle(
                penny.db,
                collection,
                index=index,
                before=before,
                sent_before=sent_before,
                served_before=served_before,
                runs_before=runs_before,
            )
        )
    return _DrivenCycles(observed=observed, ran=refusal is None, refusal=refusal)


def _cycles_ran_check(driven: _DrivenCycles) -> Check:
    """The 'the dispatcher actually ran every cycle' guard as a scored ``Check`` — the
    multi-cycle twin of the recovery runners' ``forced bail fired`` guard, so a case whose
    collection the dispatcher refused can never read as the model doing nothing."""
    return Check(
        "every cycle ran",
        driven.ran,
        kind="guard",
        rationale=None
        if driven.ran
        else f"the dispatcher refused the collection: {driven.refusal}",
    )


@pytest.fixture
def collector_cycles_eval(
    make_config: Callable[..., Config], tmp_path, request
) -> CollectorCyclesEval:
    """Drive SEVERAL real collector cycles (``run_for``) N times for one collection, each
    cycle against its own browse register, and score them together (#1905).

    ``collector_eval``'s multi-cycle sibling, kept beside it rather than folded into it:
    a one-cycle case scores an end state, while a watch's contract is what the SECOND
    cycle does about a world that moved — no notification when nothing changed, exactly
    one when something did — which needs each cycle's footprint kept apart.

    Each sample is hermetic (its own mock Signal server, DB and real-model Penny).  Seeds
    run first, then embeddings backfill, then ``prepare`` gets the constructed Penny — a
    loud world probe, so a drifted seed fails in the seed rather than after GPU time."""

    async def _sample(
        penny: Penny,
        *,
        case_id: str,
        sample_index: int,
        collection: str,
        seed: Seeder,
        cycles: Sequence[list[CannedPage]],
        score: CyclesScorer,
        seed_skills: Sequence[SkillDraft] | None,
        prepare: Preparer | None,
    ) -> SampleResult:
        """ONE sample against a constructed Penny: lay its world down, probe it, drive its
        cycles, score them with the ran-guard folded in, and write its report block."""
        seed_user(penny.db)
        seed(penny.db)
        await _embed_seeds(penny)
        if seed_skills:
            await _seed_eval_skills(penny, seed_skills)
        if prepare is not None:
            prepare(penny)
        driven = await _drive_cycles(penny, collection, cycles)
        result = _guarded_graded(
            list(score(penny.db, driven.observed)), [_cycles_ran_check(driven)]
        )
        _stamp_cause(penny.db, result)
        _write_sample_report(penny.db, case_id, sample_index, result=result)
        _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
        return result

    async def _run(
        *,
        case_id: str,
        collection: str,
        seed: Seeder,
        cycles: Sequence[list[CannedPage]],
        score: CyclesScorer,
        seed_skills: Sequence[SkillDraft] | None = None,
        prepare: Preparer | None = None,
        samples: int = SAMPLES,
        min_pass_rate: float | None = None,
        family: str | None = None,
    ) -> None:
        eval_artifacts.begin_case(case_id)

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            return await _sample(
                penny,
                case_id=case_id,
                sample_index=sample_index,
                collection=collection,
                seed=seed,
                cycles=cycles,
                score=score,
                seed_skills=seed_skills,
                prepare=prepare,
            )

        results, perf = await _run_samples(
            make_config, tmp_path, case_id=case_id, samples=samples, drive=_drive
        )
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
            min_pass_rate=min_pass_rate,
        )
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate, intended=samples)

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


class _InjectAfterToolCall(_InjectingClient):
    """The shared mid-cycle trigger: delegate to the real model until its first
    tool call lands, then inject ONE forced bad response (``_bail_response``) and
    delegate everything after.  Subclasses own only the bail's shape.
    ``_InjectDoneBail`` doesn't share this trigger — its bail is the cycle's very
    FIRST response, before any real tool call."""

    def __init__(self, real: LlmClient) -> None:
        super().__init__(real)
        self._saw_tool = False

    def _bail_response(self) -> LlmResponse:
        raise NotImplementedError

    async def chat(self, messages, tools=None, *args, **kwargs):
        if self._saw_tool and not self.bail_injected:
            self.bail_injected = True
            return self._bail_response()
        response = await self._real.chat(messages, *args, tools=tools, **kwargs)
        if response.has_tool_calls:
            self._saw_tool = True
        return response


class _InjectTextBail(_InjectAfterToolCall):
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

    def _bail_response(self) -> LlmResponse:
        return LlmResponse(message=LlmMessage(role="assistant", content=self._bail_text))


class _InjectEmptyResponse(_InjectAfterToolCall):
    """Injects ONE empty-content response right after the model's first tool call.

    Reproduces — deterministically, against the live model — a collector that
    returns empty content mid-cycle (no text AND no tool call).  The empty-response
    validator retries it with the collector nudge (``COLLECTOR_CONTINUE_NUDGE`` —
    demand a tool call, not the chat "provide your response" that invites prose),
    and the live model must recover to a clean ``done()`` close.  ``bail_injected``
    records the scenario actually fired (else the contract would be vacuous).
    """

    def _bail_response(self) -> LlmResponse:
        return LlmResponse(message=LlmMessage(role="assistant", content=""))


def _nudge_injector(
    wrap: Callable[[LlmClient], _InjectingClient] | None, bail_text: str | None
) -> Callable[[LlmClient], _InjectingClient]:
    """Resolve a nudge case's forced-bail injector from EXACTLY one selector.

    ``wrap`` is an injector factory; ``bail_text`` is shorthand for the text-bail
    injector.  Neither (or both) is a mis-specified case — fail loudly rather than
    defaulting to some bail the author didn't choose."""
    if wrap is not None and bail_text is not None:
        raise ValueError("nudge_eval needs exactly one of wrap= or bail_text=, not both")
    if wrap is not None:
        return wrap
    if bail_text is None:
        raise ValueError("nudge_eval needs exactly one of wrap= or bail_text=")
    chosen_text = bail_text
    return lambda real: _InjectTextBail(real, chosen_text)


# A nudge-eval runner: (collection, seed, wrap/bail_text) -> asserts recovery.
NudgeEval = Callable[..., Awaitable[None]]


@pytest.fixture
def nudge_eval(make_config: Callable[..., Config], tmp_path, request) -> NudgeEval:
    """Contract test for a collector user-turn nudge that recovers a bad response.

    Drives a real collector cycle but forces one bad response right after the
    model's first tool call, via an injector (``wrap(real) -> injector`` with a
    ``bail_injected`` flag; defaults to ``_InjectTextBail(bail_text)``).  Both
    covered bails are user-turn nudges (the response carried no usable tool call):

      text bail   — the model narrates prose instead of a tool call; without the
                    nudge the loop treats it as the final answer and ends the cycle
                    with no ``done()``.  Nudged (``COLLECTOR_TOOL_CALL_NUDGE``), it
                    re-emits a tool call.
      empty bail  — the model returns empty content (no text, no tool call);
                    the empty-response validator retries with the collector nudge
                    (``COLLECTOR_CONTINUE_NUDGE``, demanding a tool call).

    Either way the cycle must recover to a successful close.  Each sample asserts
    the bail actually fired AND the cycle recovered (``run_for`` returned success);
    an optional ``score`` adds case-specific checks.
    """

    async def _run(
        *,
        case_id: str,
        collection: str,
        seed: Seeder,
        bail_text: str | None = None,
        wrap: Callable[[LlmClient], _InjectingClient] | None = None,
        score: CollectorScorer | None = None,
        snapshot: Snapshotter | None = None,
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
        family: str | None = None,
    ) -> None:
        eval_artifacts.begin_case(case_id)
        make_wrapper = _nudge_injector(wrap, bail_text)

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            seed_user(penny.db)
            seed(penny.db)
            await _embed_seeds(penny)
            before = snapshot(penny.db) if snapshot is not None else None
            sent_before = len(server.outgoing_messages)
            wrapper = make_wrapper(penny.collector._model_client)
            penny.collector._model_client = wrapper
            success, _ = await penny.collector.run_for(collection)
            sent = [item.content for item in penny.db.send_queue.pending_items()] + [
                str(message.get("message", ""))
                for message in server.outgoing_messages[sent_before:]
            ]
            scored = list(score(penny.db, before, sent)) if score is not None else []
            if _scorer_is_graded(scored):
                guards = [
                    _bail_fired_check(wrapper.bail_injected),
                    _cycle_recovered_check(success),
                ]
                result = _guarded_graded(scored, guards)
            else:
                fails = [s for s in scored if isinstance(s, str)]
                if not wrapper.bail_injected:
                    fails.append("forced bail never fired — contract not exercised")
                elif not success:
                    fails.append("cycle did not recover to a successful close after the nudge")
                result = SampleResult.binary(fails)
            _stamp_cause(penny.db, result)
            _write_sample_report(penny.db, case_id, sample_index, result=result)
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

        results, perf = await _run_samples(
            make_config, tmp_path, case_id=case_id, samples=samples, drive=_drive
        )
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
            min_pass_rate=min_pass_rate,
        )
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate, intended=samples)

    return _run


class _InjectDoneBail(_InjectingClient):
    """Forces a ``done()`` tool call as the model's FIRST response — the first-move
    bail the premature-done guard must refuse.

    Reproduces, deterministically against the live model, a collector that opens
    with the argless ``done()`` before reading anything.  Pre-fix that bail closes
    the cycle; post-fix the guard returns an error tool response and the real model
    must recover (read its inputs, then do the work).  ``bail_injected`` records
    the scenario actually fired."""

    async def chat(self, messages, tools=None, *args, **kwargs):
        if not self.bail_injected:
            self.bail_injected = True
            return LlmResponse(
                message=LlmMessage(
                    role="assistant",
                    tool_calls=[
                        LlmToolCall(
                            id="bail-done",
                            function=LlmToolCallFunction(name="done", arguments={}),
                        )
                    ],
                )
            )
        return await self._real.chat(messages, *args, tools=tools, **kwargs)


class _InjectFictitiousToolPrompt(_InjectingClient):
    """Forces ONE ``collection_set`` whose ``extraction_prompt`` names a tool no
    collector has, as the model's FIRST response.

    Reproduces — deterministically against the live model — the chat agent writing a
    hallucinated tool into a collection's recipe (observed: a made-up ``extract_text``
    for a "read the page" step).  The write-time gate refuses it with the
    correction-teaching message, and the live model must recover: re-issue a
    ``collection_set`` whose prompt uses only real tools (``browse`` for the read),
    which then persists.  ``bail_injected`` records the scenario actually fired."""

    def __init__(self, real: LlmClient, collection: str, prompt: str) -> None:
        super().__init__(real)
        self._collection = collection
        self._prompt = prompt

    async def chat(self, messages, tools=None, *args, **kwargs):
        if not self.bail_injected:
            self.bail_injected = True
            return LlmResponse(
                message=LlmMessage(
                    role="assistant",
                    tool_calls=[
                        LlmToolCall(
                            id="bail-fictitious-tool",
                            function=LlmToolCallFunction(
                                name="collection_set",
                                arguments={
                                    "name": self._collection,
                                    "extraction_prompt": self._prompt,
                                },
                            ),
                        )
                    ],
                )
            )
        return await self._real.chat(messages, *args, tools=tools, **kwargs)


class _InjectSendBail(_InjectAfterToolCall):
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

    def _bail_response(self) -> LlmResponse:
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


class _InjectDuplicateWrite(_InjectingClient):
    """Forces ONE ``collection_write`` of one-or-more entries that each duplicate an
    entry the target collection already holds, as the model's FIRST response.

    Reproduces — deterministically against the live model — a collector that writes
    something already saved.  The real dedup rejects it, and the rejection now BINDS
    each matched existing key into an ``update_entry`` call; the live model must
    recover (``update_entry`` on the bound key, or an honest ``done()``) instead of
    re-using its own rejected key / re-reading / retrying variations until it burns
    the step budget.  A multi-entry batch proves EVERY rejected key gets its match
    bound, not just the first.  ``bail_injected`` records the scenario actually fired."""

    def __init__(self, real, memory: str, entries: list[tuple[str, str]]) -> None:
        super().__init__(real)
        self._memory = memory
        self._entries = entries

    async def chat(self, messages, tools=None, *args, **kwargs):
        if not self.bail_injected:
            self.bail_injected = True
            return LlmResponse(
                message=LlmMessage(
                    role="assistant",
                    tool_calls=[
                        LlmToolCall(
                            id="bail-dup-write",
                            function=LlmToolCallFunction(
                                name="collection_write",
                                arguments={
                                    "memory": self._memory,
                                    "entries": [
                                        {"key": key, "content": content}
                                        for key, content in self._entries
                                    ],
                                },
                            ),
                        )
                    ],
                )
            )
        return await self._real.chat(messages, *args, tools=tools, **kwargs)


class _InjectKeyMiss(_InjectingClient):
    """Forces ONE ``collection_get`` on a near-miss key — a key close to, but not
    equal to, one the target collection actually holds — as the model's FIRST
    response.

    Reproduces — deterministically against the live model — the key-not-found
    residue (July 2026 tool-failure audit, item #11): the model probes an entry
    that exists under a slightly different key, gets the not-found rejection, lists
    the keys, finds the real one, and then must pick the RIGHT write path.  The
    rejection now names the write-vs-update decision, so the model updates the
    EXISTING entry with ``update_entry`` instead of ``collection_write``-ing it (a
    duplicate the dedup rejects — the ping-pong the extended guidance removes).
    ``bail_injected`` records the forced probe actually fired."""

    def __init__(self, real, memory: str, near_miss_key: str) -> None:
        super().__init__(real)
        self._memory = memory
        self._near_miss_key = near_miss_key

    async def chat(self, messages, tools=None, *args, **kwargs):
        if not self.bail_injected:
            self.bail_injected = True
            return LlmResponse(
                message=LlmMessage(
                    role="assistant",
                    tool_calls=[
                        LlmToolCall(
                            id="bail-key-miss",
                            function=LlmToolCallFunction(
                                name="collection_get",
                                arguments={"memory": self._memory, "key": self._near_miss_key},
                            ),
                        )
                    ],
                )
            )
        return await self._real.chat(messages, *args, tools=tools, **kwargs)


class _InjectDuplicateCall(_InjectingClient):
    """Replays the model's FIRST tool call byte-identically, exactly once, so the
    agent-loop dedup guard rejects it — then delegates every later call to the live
    model to drive the recovery.

    Reproduces — deterministically against the live model — a run that re-issues an
    exact call it already made (a natural cycle only rarely does this on its own).
    The guard refuses the repeat with the reworked ``DUPLICATE_CALL_REJECTION``
    (behaviour unchanged: the repeat is not executed); the live model must MOVE ON —
    reuse the earlier result and finish its real work — instead of over-generalizing
    "no repeated calls" and suppressing the writes it still owes.  ``bail_injected``
    records the forced repeat actually fired (else the contract would be vacuous).

    Note: the guard blocks a BYTE-IDENTICAL repeat for the whole run, so the contract
    measures the real harm — owed follow-up work being suppressed — via the run still
    completing its write, not by forcing a literal re-read (which the unchanged guard
    would itself refuse)."""

    def __init__(self, real) -> None:
        super().__init__(real)
        self._first_call: tuple[str, dict] | None = None

    async def chat(self, messages, tools=None, *args, **kwargs):
        if self._first_call is not None and not self.bail_injected:
            self.bail_injected = True
            name, arguments = self._first_call
            return LlmResponse(
                message=LlmMessage(
                    role="assistant",
                    tool_calls=[
                        LlmToolCall(
                            id="bail-dup-call",
                            function=LlmToolCallFunction(name=name, arguments=dict(arguments)),
                        )
                    ],
                )
            )
        response = await self._real.chat(messages, *args, tools=tools, **kwargs)
        if self._first_call is None and response.has_tool_calls:
            call = (response.message.tool_calls or [])[0]
            self._first_call = (call.function.name, dict(call.function.arguments))
        return response


class _InjectBracketKey(_InjectingClient):
    """Rewrites the model's FIRST key-bearing tool call to wrap its key in display
    brackets (``key='Ark Nova'`` → ``key='[Ark Nova]'``), reproducing the
    copy-through mistake deterministically against the live model.

    The old ``[key]`` render taught the model to paste the display brackets into a
    ``key=`` argument; this forces exactly that on the model's own first attempt so
    the memory-tool teaching rejection fires on every sample, and the live model
    must recover to the bare key.  Every other call passes through untouched.
    ``bail_injected`` records the sabotage actually fired (else the contract would
    be vacuous)."""

    _KEY_TOOLS = ("update_entry", "collection_delete_entry", "collection_get")

    async def chat(self, messages, tools=None, *args, **kwargs):
        response = await self._real.chat(messages, *args, tools=tools, **kwargs)
        if self.bail_injected or not response.has_tool_calls:
            return response
        for call in response.message.tool_calls or []:
            if call.function.name not in self._KEY_TOOLS:
                continue
            key = call.function.arguments.get("key")
            if isinstance(key, str) and key and not _is_bracket_wrapped(key):
                call.function.arguments["key"] = f"[{key}]"
                self.bail_injected = True
                break
        return response


# A guard-recovery runner: (collection, seed, wrap_client, score) -> asserts recovery.
GuardRecoveryEval = Callable[..., Awaitable[None]]


@pytest.fixture
def guard_recovery_eval(make_config: Callable[..., Config], tmp_path, request) -> GuardRecoveryEval:
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
        score: Callable[[Database, list[str]], list[str] | list[Check]],
        browse: list[CannedPage] | None = None,
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
        family: str | None = None,
    ) -> None:
        eval_artifacts.begin_case(case_id)

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
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
            scored = list(score(penny.db, sent))
            if _scorer_is_graded(scored):
                result = _guarded_graded(scored, [_bail_fired_check(wrapper.bail_injected)])
            else:
                fails = [s for s in scored if isinstance(s, str)]
                if not wrapper.bail_injected:
                    fails.append("forced bail never fired — contract not exercised")
                result = SampleResult.binary(fails)
            _stamp_cause(penny.db, result)
            _write_sample_report(penny.db, case_id, sample_index, result=result)
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

        results, perf = await _run_samples(
            make_config, tmp_path, case_id=case_id, samples=samples, drive=_drive
        )
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
            min_pass_rate=min_pass_rate,
        )
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate, intended=samples)

    return _run


# A startup-eval runner: (case_id, commit_message, score) -> asserts threshold.
StartupEval = Callable[..., Awaitable[None]]


@pytest.fixture
def startup_eval(make_config: Callable[..., Config], tmp_path, request) -> StartupEval:
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
        family: str | None = None,
    ) -> None:
        eval_artifacts.begin_case(case_id)

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            seed_user(penny.db)
            announcement = await get_restart_message(penny.db, penny.model_client, commit_message)
            # Same graded/binary dispatch as the other runners.  Startup has no
            # injection (no wrapper, no framework guard), so a graded return grades
            # over the scorer's own Checks with an empty guard list.
            scored = list(score(announcement))
            if _scorer_is_graded(scored):
                result = _guarded_graded(scored, [])
            else:
                result = SampleResult.binary([s for s in scored if isinstance(s, str)])
            _stamp_cause(penny.db, result)
            return result

        results, perf = await _run_samples(
            make_config, tmp_path, case_id=case_id, samples=samples, drive=_drive
        )
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
            min_pass_rate=min_pass_rate,
        )
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate, intended=samples)

    return _run


# ── Classifier eval (#1706 beat 1): one scoped micro-context call per sample ──
# A classifier-eval runner: (case_id, snapshot, pool, expected) -> asserts threshold.
ClassifierEval = Callable[..., Awaitable[None]]


def _score_classifier(
    decision: StateDecision, expected: ConversationState, expected_skill: str | None
) -> list[Check]:
    """The classifier case's graded checks (#1706): ONE scored check — the expected
    edge was decided — so the case mean IS that direction's confusion-matrix cell.
    A wrong edge and a contract failure both score 0 (a cleanly-decided WRONG edge
    must never outscore a harmless no-decision — fail → stay is the safe outcome);
    the advisory well-formed check plus the rationale keep the two failure kinds
    distinct in the report without distorting the score.  An apply case also
    scores WHICH skill the draw bound (``expected_skill``) — n/a when the sample
    never decided the expected state (no skill to judge; the edge check already
    failed).  Both skill-gated states (apply, request) score it."""
    decided = decision.outcome == StateDrawOutcome.DECIDED
    ok = decided and decision.state is expected
    if ok:
        rationale = None
    elif decided and decision.state is not None:
        rationale = f"drew {decision.state.value} instead"
    else:
        rationale = f"no decision — {decision.outcome.value}"
    checks = [
        Check(f"decided {expected.value}", ok, kind="state", rationale=rationale),
        Check(
            "draw well-formed (tagged, in-union)",
            decided,
            kind="proc",
            scored=False,
            rationale=None if decided else f"terminal outcome {decision.outcome.value}",
        ),
    ]
    if expected_skill is not None:
        if decided and decision.state is expected:
            named = decision.skill == expected_skill
            checks.append(
                Check(
                    "named the covering skill",
                    named,
                    kind="state",
                    rationale=None if named else f"named {decision.skill}",
                )
            )
        else:
            checks.append(
                Check.na(
                    "named the covering skill",
                    rationale=f"no {expected.value} decision to carry a skill",
                    kind="state",
                )
            )
    return checks


def _micro_context_rows(db: Database, *agent_names: str) -> list[PromptLog]:
    """The sample's micro-context promptlog rows for the named customers, in LEDGER
    order — one per draw, so a reroll shows as a second row (the fragile signal and
    the transcript's second 🧩 pair both read off this).

    Takes SEVERAL customers because a run end is not one draw (#1803): the labeller
    rules on provenance and the shape draw decides what the routine is, and a report
    naming only one of them renders a transcript that cannot show the decision the
    case is scoring — which is how a shipped second draw read as never built."""
    wanted = frozenset(agent_names)
    return [row for row in _sample_prompt_rows(db) if row.agent_name in wanted]


def _classifier_rows(db: Database) -> list[PromptLog]:
    """The sample's state-classifier rows (#1706)."""
    return _micro_context_rows(db, PennyConstants.STATE_CLASSIFIER_AGENT_NAME)


def _classifier_events(phrasing: str, rows: list[PromptLog]) -> list[report.Event]:
    """The hand-built event stream for one single-draw micro-context sample: the
    phrasing opens the step, then one 🧩 in/out pair PER DRAW — a reroll renders as
    a second pair, so recovery is visible in the transcript, never summarized away.
    Each pair is labelled with the drawing context's ledger identity (#1773), the
    same actor label the chat-run extractor emits."""
    events = [report.Event(report.EventKind.USER, phrasing)]
    for row in rows:
        messages = json.loads(row.messages) if row.messages else []
        user = next(
            (m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), ""
        )
        context = row.agent_name or ""
        events.append(report.Event(report.EventKind.MICRO_IN, user, context=context))
        events.append(
            report.Event(
                report.EventKind.MICRO_OUT,
                _response_text(row) or "(empty)",
                thinking=row.thinking or "",
                context=context,
            )
        )
    return events


def _classifier_check_views(
    result: SampleResult, anchor_index: int, baseline: Baseline | None, case_id: str
) -> list[report.CheckView]:
    """Every check anchored to the FINAL draw's 🧩→ row (the decision), with the
    baseline flip resolved per ``(case_id, label)`` exactly like the extractor's."""
    views: list[report.CheckView] = []
    for index, check in enumerate(result.checks, start=1):
        regressed = (
            not check.ok
            and not check.ignored
            and baseline is not None
            and baseline.was_passing(case_id, check.label)
        )
        views.append(
            report.CheckView(
                check_id=f"C{index}",
                label=check.label,
                kind=check.kind,
                scored=check.scored,
                ignored=check.ignored,
                ok=check.ok,
                rationale=check.rationale,
                cause=_cause_word(result.cause) if not check.ok else None,
                anchor_index=anchor_index,
                regressed=regressed,
            )
        )
    return views


def _write_classifier_report(
    db: Database,
    case_id: str,
    sample_index: int,
    *,
    result: SampleResult,
    phrasing: str,
    agent_names: Sequence[str] = (PennyConstants.STATE_CLASSIFIER_AGENT_NAME,),
) -> None:
    """One micro-context sample's transcript block — hand-built (the
    generic extractor is chat-run-shaped; this sample is one step whose actor is the
    🧩 micro-context, the spec's official sub-model actor), rendered by the SAME pure
    report grammar and appended to the same ``<case_id>.md``. No-op off-report.
    ``agent_names`` selects the customers' rows, rendered in one ledger-ordered
    stream (the classifier by default; the run-end labeller AND shape draw for the
    skill cases, #1770/#1803) — each 🧩 pair carries its own context label, so two
    customers read as the two actors they are."""
    report_dir = os.environ.get("EVAL_REPORT_DIR")
    if not report_dir:
        return
    rows = _micro_context_rows(db, *agent_names)
    if not rows:
        transcript = report.SampleTranscript(
            sample_index + 1,
            _sample_banner(db, result, evaluated=False),
            [],
            placeholder=report.NO_TURNS_PLACEHOLDER,
        )
    else:
        events = _classifier_events(phrasing, rows)
        checks = _classifier_check_views(result, len(events) - 1, baseline_from_env(), case_id)
        passed_checks, total = _scored_counts(result)
        transcript = report.build_sample(
            number=sample_index + 1,
            banner=_sample_banner(db, result, evaluated=True),
            events=events,
            checks=checks,
            run_close_score=f"{passed_checks}/{total}",
        )
    _record_sample_block(case_id, sample_index, transcript)


# One structurally-valid placeholder step for an eval-seeded skill: the
# classifier reads only name + description, but the draft stays real-typed.
_SEED_SKILL_STEP = SkillStep(
    ordinal=1,
    source_ordinal=1,
    tool="browse",
    arguments={"queries": ["https://example.test"], "extract": "the value"},
)


def eval_skill(name: str, description: str, params: dict[str, str]) -> SkillDraft:
    """A fixture skill draft: name + description + declared parameters (semantic
    name → what-to-supply), over one structurally-valid placeholder step.  The
    classifier reads the FULL metadata — name, description, AND parameters — so
    a case's seeds must carry the same shape real auto-extracted skills do."""
    return SkillDraft(
        name=name,
        intent=description,
        description=description,
        steps=[_SEED_SKILL_STEP],
        parameters=[SkillParameter(name=key, description=value) for key, value in params.items()],
        source_run_id="eval-seed",
    )


async def _seed_eval_skills(penny: Penny, seed_skills: Sequence[SkillDraft]) -> None:
    """Seed fixture skills WITH real description embeddings, so
    ``resolve_by_meaning`` ranks them (a vectorless skill is resolution-invisible
    — seeding one would silently hollow the case).  A failed embed fails the
    sample loudly rather than degrade."""
    for draft in seed_skills:
        vector = await embed_text(penny.embedding_model_client, draft.description)
        assert vector is not None, f"seed skill embed failed: {draft.name}"
        penny.db.skills.upsert(draft, author="eval-seed", description_embedding=vector)


@pytest.fixture
def classifier_eval(make_config: Callable[..., Config], tmp_path, request) -> ClassifierEval:
    """Drive the conversation-state classifier (#1706) N times — ONE scoped
    micro-context call per sample, no agent loop — sweeping a PHRASING POOL
    deterministically (sample i → ``pool[i % len(pool)]``), so N samples cover
    input SPACE rather than re-rolling one point (the input-variation doctrine's
    first native customer): per-check cells map 1:1 to phrasings, and a baseline
    diff compares phrasing-for-phrasing.

    Each sample is hermetic (own DB + real-model Penny, mirroring
    ``startup_eval``); the snapshot is built PER SAMPLE by the production
    ``build_snapshot`` (embed + resolve_by_meaning pre-pass) from the case's
    ``state`` + the sample's phrasing — the same path the chat wiring will
    call.  Scoring is runner-owned — one scored check, the expected edge was
    decided, so the case mean IS that direction's confusion-matrix cell; a
    well-formed-draw advisory keeps discrimination misses distinct from contract
    failures.  ``fragile`` is the classifier's native recovery signal: DECIDED
    after more than one draw (a reroll).  A poisoned draw group is tagged
    pathology by the standard response scan; a hung call is a harness timeout.
    """

    async def _run(
        *,
        case_id: str,
        state: ConversationState,
        pool: Sequence[str],
        expected: ConversationState,
        expected_skill: str | None = None,
        penny_last_turn: str | None = None,
        task_anchor: str | None = None,
        seed: Seeder | None = None,
        seed_skills: Sequence[SkillDraft] | None = None,
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
        timeout: float = 60.0,
        family: str | None = None,
    ) -> None:
        eval_artifacts.begin_case(case_id)

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            phrasing = pool[sample_index % len(pool)]
            seed_user(penny.db)
            if seed is not None:
                seed(penny.db)
            await _embed_seeds(penny)
            if seed_skills:
                await _seed_eval_skills(penny, seed_skills)
            classifier = StateClassifier(penny.model_client)
            try:
                # The PRODUCTION snapshot builder per sample, so the eval
                # exercises the same path the wiring does — EVERY seeded
                # skill offered, no ranking or cap (the #1706 ruling).
                snapshot = build_snapshot(
                    penny.db,
                    state=state,
                    message=phrasing,
                    penny_last_turn=penny_last_turn,
                    task_anchor=task_anchor,
                )
                decision = await asyncio.wait_for(
                    classifier.classify(snapshot, phrasing, run_target=penny.chat_agent.name),
                    timeout=timeout,
                )
                result = _guarded_graded(
                    list(_score_classifier(decision, expected, expected_skill)), []
                )
                result.fragile = result.passed and len(_classifier_rows(penny.db)) > 1
                _stamp_cause(penny.db, result)
            except TimeoutError:
                result = SampleResult.binary(["no decision within timeout"])
                _stamp_cause(penny.db, result, timed_out=True)
            _write_classifier_report(
                penny.db, case_id, sample_index, result=result, phrasing=phrasing
            )
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

        results, perf = await _run_samples(
            make_config, tmp_path, case_id=case_id, samples=samples, drive=_drive
        )
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
            min_pass_rate=min_pass_rate,
        )
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate, intended=samples)

    return _run


# ── Run-end LEAF LABELLING (#1828) ────────────────────────────────────────────

# One demonstrated tool call as a case fixture: tool name, verbatim arguments, its
# framed result, and whether it succeeded — the shape a real chat run leaves behind in
# the ledger, which is what ``_labelling_input`` renders the labeller's content from.
DemoCall = tuple[str, dict, str, bool]

# One conversation turn as a case fixture: its DIRECTION and its text.  The direction
# is carried because the elicit round is half the evidence — the assistant's "walk me
# through it once" is what makes the next user turn a demonstration — and a block that
# renders both speakers as ``user:`` describes a conversation that never happened.
DemoTurn = tuple[str, str]

LabellerEval = Callable[..., Awaitable[None]]


# The customers that draw at run end: the LABELLER (every spot in the implementation)
# and the FRAMER (the interface, from the ask alone).  One list, so the report and the
# reroll signal can never disagree about who spoke at the end of a run.
RUN_END_CUSTOMERS = (
    PennyConstants.SKILL_NAMING_AGENT_NAME,
    PennyConstants.SKILL_FRAME_AGENT_NAME,
)


def _run_end_rerolled(db: Database) -> bool:
    """Did EITHER run-end customer draw more than once (#1828/#1830) — the fragile
    signal, a sample that only got there by recovering.

    Counted PER customer, never in total: a run end is two draws by design, so a
    total would mark every sample fragile and the signal would mean nothing."""
    return any(len(_micro_context_rows(db, agent)) > 1 for agent in RUN_END_CUSTOMERS)


def _leaf_string(arguments: dict, path: Sequence[str | int]) -> str | None:
    """The string leaf a substitution's JSON path addresses in a step's verbatim
    arguments — the demonstrated value — or ``None`` when the path doesn't resolve to
    one."""
    node: object = arguments
    for part in path:
        if isinstance(node, dict) and isinstance(part, str):
            node = node.get(part)
        elif isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node):
            node = node[part]
        else:
            return None
    return node if isinstance(node, str) else None


def _labelling_input(
    calls: Sequence[DemoCall], target: str, utterance: str, conversation: Sequence[DemoTurn]
) -> tuple[str, dict[str, str]]:
    """The labeller's content, and the ``{demonstrated value: current name}`` map its
    labels are keyed by.

    Distillation is DETERMINISTIC Python, so a fixture ledger fixes the labeller's
    input exactly — no live draw sits upstream of this case.  The content is rendered
    by ``build_naming_content``, the shipped function, so the case exercises the
    shipped prompt rather than a copy that can drift.

    The map is keyed by VALUE because the case's expectations are stated in values:
    the semantic name is the model's to choose, and the contract is about which SPOTS
    it named, never what it ended up calling them."""
    inputs = [
        DistillInput(source_ordinal=index, tool=name, arguments=arguments, result=result)
        for index, (name, arguments, result, _ok) in enumerate(calls, start=1)
    ]
    steps, parameters = distill_steps(inputs, frozenset({target}))
    content = build_naming_content(steps, parameters, utterance, list(conversation))
    by_value = {
        value: sub.parameter
        for step in steps
        for sub in step.substitutions
        if sub.parameter is not None
        and (value := _leaf_string(step.arguments, sub.path)) is not None
    }
    return content, by_value


def _drifted(value: str, label: str) -> Check:
    """A case value that is not among the distilled spots at all — a BROKEN FIXTURE,
    failed LOUDLY naming the value rather than quietly as a naming miss.  The ledger
    has drifted from what the case asserts, and a drifted fixture that scores green is
    a case measuring nothing."""
    return Check(
        label,
        False,
        kind="state",
        rationale=f"{value!r} is not among the distilled placeholders — the fixture has drifted",
    )


def _leaf_checks(value: str, labels: SkillLabels | None, by_value: dict[str, str]) -> list[Check]:
    """One offered spot's checks (#1828), read off the labeller's OWN typed result.

    Scoring the draw rather than a persisted skill is what keeps this case the
    labeller's: everything downstream — applying a label, rendering it — is
    deterministic Python pinned in ``tests/test_skill_extraction.py``.

    Four things, in the order they depend on each other: a line came back for the spot
    · its name hardens to a usable binding key through the SHIPPED hardener (imported,
    never re-implemented here) · the name is not the arg name handed back verbatim (a
    spot named ``queries`` again has been described, not named) · its description says
    what belongs there.  With no line, the last three are NOT APPLICABLE rather than
    three extra failures: the miss is one miss, and inflating it would make a silent
    draw look four times worse than it is.

    Since #1828 the first check is a WHOLE-DRAW property, not a per-spot one: an
    accepted draw covers every offered spot, so either every spot has its line or the
    draw failed and none do.  It stays per spot so the report reads the same either
    way."""
    current = by_value.get(value)
    if current is None:
        return [_drifted(value, f"a line came back: {value!r}")]
    label = labels.labels.get(current) if labels is not None else None
    if label is None:
        return [
            Check(f"a line came back: {value!r}", False, kind="state", rationale="no line drawn"),
            Check.na(f"name is a usable binding key: {value!r}", kind="state"),
            Check.na(f"name is not the arg name: {value!r}", kind="state"),
            Check.na(f"description says what belongs there: {value!r}", kind="state"),
        ]
    hardened = slug_parameter_name(label.name)
    return [
        Check(f"a line came back: {value!r}", True, kind="state"),
        Check(
            f"name is a usable binding key: {value!r}",
            bool(hardened),
            kind="state",
            rationale=None if hardened else f"{label.name!r} hardens to nothing",
        ),
        Check(
            f"name is not the arg name: {value!r}",
            hardened != slug_parameter_name(current),
            kind="state",
            rationale=None if hardened != slug_parameter_name(current) else "echoed the arg name",
        ),
        Check(
            f"description says what belongs there: {value!r}",
            bool(label.description.strip()),
            kind="state",
            rationale=None if label.description.strip() else "no description",
        ),
    ]


def _distinct_names_check(
    pair: tuple[str, str], labels: SkillLabels | None, by_value: dict[str, str]
) -> Check:
    """Two spots on the SAME argument drew DIFFERENT names (#1828) — the disambiguation
    the two-source case exists to measure.  A labeller that calls both of them
    ``news_page`` has collapsed a distinction the routine depends on: at run time each
    spot takes its own value, and one name for two spots cannot say which is which."""
    first, second = pair
    label = f"distinct names: {first!r} vs {second!r}"
    currents = [by_value.get(value) for value in pair]
    missing = [value for value, current in zip(pair, currents, strict=True) if current is None]
    if missing:
        return _drifted(missing[0], label)
    drawn = [
        labels.labels.get(current) if labels is not None and current is not None else None
        for current in currents
    ]
    if any(one is None for one in drawn):
        return Check(label, False, kind="state", rationale="one of the two drew no line")
    names = [slug_parameter_name(one.name) for one in drawn if one is not None]
    ok = names[0] != names[1]
    return Check(
        label,
        ok,
        kind="state",
        rationale=None if ok else f"both drew {names[0]!r}",
    )


def _shared_spot_check(value: str, labels: SkillLabels | None, by_value: dict[str, str]) -> Check:
    """The spot filling TWO argument sites drew exactly ONE label (#1828).

    Equal values at two sites are structurally one spot, so the contract is one line
    covering both uses.  A draw that splits it either repeats the spot's current name or
    keys its second line to a name nobody offered — both are coverage violations the
    validator refuses (#1828), so a split never reaches an accepted draw and this reads
    as the spot having no label.  Named as its own check because it is this case's whole
    claim, and a diff-join key should say what it was watching."""
    label = f"one label for the shared spot: {value!r}"
    current = by_value.get(value)
    if current is None:
        return _drifted(value, label)
    drawn = labels.labels if labels is not None else {}
    return Check(
        label,
        current in drawn,
        kind="state",
        rationale=None if current in drawn else "the shared spot drew no single line",
    )


def _score_labelling(
    labels: SkillLabels | None,
    by_value: dict[str, str],
    leaves: Sequence[str],
    distinct_names: Sequence[tuple[str, str]],
    shared_spot: str,
) -> list[Check]:
    """The labelling case's checks (#1828), read off the returned ``SkillLabels``: every
    offered spot got a usable name and a description of what belongs there, plus each
    case's own structural claim.

    The drawn labels then ride along ADVISORY (``scored=False``, the same rule
    ``_score_framing`` keeps), so every report shows verbatim what the model committed
    to — whether a
    name is WELL judged is a reading no scorer should fake, and it is what the reference
    outputs on the ticket are read against at review."""
    checks: list[Check] = []
    for value in leaves:
        checks.extend(_leaf_checks(value, labels, by_value))
    checks.extend(_distinct_names_check(pair, labels, by_value) for pair in distinct_names)
    if shared_spot:
        checks.append(_shared_spot_check(shared_spot, labels, by_value))
    for current, label in sorted((labels.labels if labels is not None else {}).items()):
        checks.append(
            Check(
                f"drew {current}: {label.name!r} — {label.description!r}",
                True,
                kind="state",
                scored=False,
            )
        )
    return checks


FramerEval = Callable[..., Awaitable[None]]

# A word token of a drawn name or description — what family classification and the
# generic checks both read.  Word-boundary, never substring: a description saying
# "festival" must not read as the instance token "fest", while the instance itself
# tokenises to it.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A trailing digit is its own token: ``site1`` is ``site`` + ``1``.  Enumerating with
# digits is one of the natural ways to write an ordinal pair, so a scorer that reads
# ``site1`` as a single opaque word marks a CORRECT draw as a family miss — which is
# exactly what the second run did to two of them.
_DIGIT_SUFFIX_RE = re.compile(r"([a-z]+)(\d+)")

# Lettering is the OTHER natural way to write that same enumeration — ``url_a`` /
# ``url_b`` — and it cost two correct draws the same way digits once did: two distinct,
# generic, scalar names satisfying the tell-them-apart rule, read by the scorer as
# answering neither ordinal family.  A trailing single letter is therefore the position
# it holds in the alphabet, which lands the pair on the families that already carry
# ``1`` / ``2`` — no family gains a token, so nothing else can start matching.
_ORDINAL_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Write-time GARNISH: an example of THIS occasion's value hung off an otherwise generic
# line — ``the page to check (e.g., bookbarn.example/atlas-of-clouds)``.  The fourth run's
# thinking traces settled what it is: the drafted lines carried no example at all, and the
# clause appeared only at transcription, so it is a habit of writing rather than a claim
# about the routine.  Scoring it as an occasion-named parameter failed two lines whose
# substance was exactly right, so it is stripped before the instance-token scan.
#
# Only the CLAUSE goes.  An instance token in the name itself, or a value standing as the
# whole description, is the line's substance and still fails — which is the difference
# between a line that says what to supply and one that says what was supplied.
_EXAMPLE_LEAD = r"(?:e\.?\s?g\.?|for\s+example|for\s+instance|such\s+as)"
_PARENTHESIZED_EXAMPLE_RE = re.compile(rf"\(\s*{_EXAMPLE_LEAD}\b[^)]*\)?", re.IGNORECASE)
_TRAILING_EXAMPLE_RE = re.compile(rf"[\s,;:—-]*\b{_EXAMPLE_LEAD}\b.*$", re.IGNORECASE)


class ParameterFamily(NamedTuple):
    """One parameter the ask genuinely requires, as the SET of things it could
    reasonably be called (#1830).

    ``label`` is the reference name from the agreed pair — the diff-join key a check is
    named for, never a string the model has to produce.  ``tokens`` is the semantic
    breadth the code owner agreed to: a drawn parameter belongs to this family when its
    name (or, only when no family matched the name, its description) uses any of them.
    So ``page_to_watch`` lands on the family the reference calls ``url``, while an extra
    parameter nobody asked for lands nowhere and shows up in the count.

    ``name_only`` drops the description fallback for this family — the code owner's
    ruling on the first run, where two IDENTICAL `city` draws scored opposite ways
    because one of their descriptions happened to mention the site the location sits on.
    A PAGE is named as one; reading "the page" out of a description that merely refers
    to a page is the scorer inventing an answer the draw did not give."""

    label: str
    tokens: tuple[str, ...]
    name_only: bool = False


def _tokens(text: str) -> set[str]:
    """A drawn string's word tokens, lowercased — ``first_site`` → ``{first, site}``,
    and ``site1`` → ``{site1, site, 1}`` so a digit-suffixed ordinal reads as one.

    The whole token is kept alongside its split, so nothing that matched before stops
    matching."""
    found: set[str] = set()
    for token in _TOKEN_RE.findall(text.lower()):
        found.add(token)
        split = _DIGIT_SUFFIX_RE.fullmatch(token)
        if split is not None:
            found.update(split.groups())
    return found


def _letter_suffix_ordinal(tokens: list[str]) -> str | None:
    """The digit a NAME's trailing single-letter suffix stands for — ``url_a`` → ``1`` —
    or ``None`` when the name carries no such suffix.

    A suffix needs something to be suffixed TO, so a name that is ONLY a letter is left
    alone: ``a`` is a name nobody enumerated, not the first of anything.  The
    single-CHARACTER test is load-bearing, not decoration: membership in a string is a
    SUBSTRING test, so without it a name ending in ``no`` would read as a letter suffix
    and score as the fourteenth of something."""
    last = tokens[-1] if tokens else ""
    if len(tokens) < 2 or len(last) != 1 or last not in _ORDINAL_LETTERS:
        return None
    return str(_ORDINAL_LETTERS.index(last) + 1)


def _name_tokens(name: str) -> set[str]:
    """A drawn NAME's tokens: :func:`_tokens` plus the ordinal any trailing single-letter
    suffix stands for, so ``url_a``/``url_b`` classify as the two ordinal families
    exactly as ``site1``/``site2`` and ``url_1``/``url_2`` already do.

    NAME-only, and that is the whole reason it is a separate function rather than a
    widening of :func:`_tokens`: carving a suffix off an identifier is an identifier
    operation, while a description is PROSE, where ``a`` is an article.  Reading a
    letter as an ordinal there would file most descriptions ever written under the
    first-ordinal family — which is precisely why the letters are not family tokens."""
    tokens = _TOKEN_RE.findall(name.lower())
    found = _tokens(name)
    ordinal = _letter_suffix_ordinal(tokens)
    return found | {ordinal} if ordinal is not None else found


def _without_examples(text: str) -> str:
    """``text`` with any example clause removed — ``the plot to log (e.g., 17)`` → ``the
    plot to log``, and the same for a trailing ``, e.g. 17`` with no parentheses.

    Both forms are stripped before the instance-token scan reads the line, and NOTHING
    else is: the clause is a way of writing, not a claim, so what remains is what the
    line actually says."""
    return _TRAILING_EXAMPLE_RE.sub("", _PARENTHESIZED_EXAMPLE_RE.sub("", text)).strip()


def _substance_tokens(parameter: FramedParameter) -> set[str]:
    """A parameter's tokens once its example garnish is gone — name and description
    stripped SEPARATELY, so a trailing example on one can never eat the other.

    An UNLABELLED parameter (#1870 — one read back off the registry, where a description is
    optional) contributes no description tokens, because it says nothing to read; a framer
    draw always carries one, so this arm is the registry's case rather than a draw's."""
    described = parameter.description or ""
    return _tokens(_without_examples(parameter.name)) | _tokens(_without_examples(described))


def _matching_family(
    tokens: set[str], families: Sequence[ParameterFamily]
) -> ParameterFamily | None:
    """The first family whose agreed tokens ``tokens`` uses, or ``None``.

    Takes the TOKENS rather than the text, because how a string tokenizes is the pass's
    business, not this function's: a name is an identifier (``_name_tokens``, which reads
    a trailing letter as an ordinal) and a description is prose (``_tokens``)."""
    return next((f for f in families if tokens & set(f.tokens)), None)


def classify_by_family(
    named: Sequence[tuple[str, str | None]], families: Sequence[ParameterFamily]
) -> list[ParameterFamily | None]:
    """The family each ``(name, description)`` pair answers, positionally — NAME FIRST
    (#1830).  ONE classification discipline, read by every suite that asks what a drawn
    parameter answers (the framer's own set check, and the transitions suite's
    interface check).

    A parameter's name is its identity, so the name pass runs over ALL of them before
    any description is read, and a family a name already claimed is closed to the
    description fallback.  Otherwise a description that merely mentions the page in
    passing ("the search that finds the listings on a page") would make a second
    parameter read as the page as well, and the set would look complete when it is not.

    A ``name_only`` family sits out the description pass entirely — the page/url
    tightening from the first run's review: a parameter is the page when it is NAMED as
    the page, and no description-level mention promotes one that isn't.

    A description of ``None`` is an UNLABELLED parameter (#1870), which the description
    pass has nothing to read — so its name is the whole of its evidence, and it stays
    unclassified rather than being classified off a blank."""
    by_name = [_matching_family(_name_tokens(name), families) for name, _ in named]
    claimed = {family.label for family in by_name if family is not None}
    open_families = [
        family for family in families if family.label not in claimed and not family.name_only
    ]
    return [
        matched or _matching_family(_tokens(description or ""), open_families)
        for matched, (_, description) in zip(by_name, named, strict=True)
    ]


def _classified(
    signature: SkillSignature, families: Sequence[ParameterFamily]
) -> dict[str, list[FramedParameter]]:
    """Every drawn parameter grouped under the family it answers — the framer suite's
    view of :func:`classify_by_family`, which owns the discipline."""
    answered = classify_by_family([(p.name, p.description) for p in signature.parameters], families)
    grouped: dict[str, list[FramedParameter]] = {family.label: [] for family in families}
    for parameter, family in zip(signature.parameters, answered, strict=True):
        if family is not None:
            grouped[family.label].append(parameter)
    return grouped


def _generic_framing_check(signature: SkillSignature, instance_tokens: Sequence[str]) -> Check:
    """The name and description are the KIND of task, never the occasion (#1830).

    Structural, not a judgment: none of the case's instance tokens — the book, the two
    papers, the ticker, the city, the festival — may appear in either.  A framing that
    carries one has named the occasion it was taught on, which is the routine that can
    only ever run once."""
    used = sorted(_tokens(f"{signature.name} {signature.description}") & set(instance_tokens))
    return Check(
        "the framing is generic",
        not used,
        kind="state",
        rationale=None if not used else f"named the occasion: {', '.join(used)}",
    )


def _generic_parameters_check(signature: SkillSignature, instance_tokens: Sequence[str]) -> Check:
    """Every PARAMETER line is generic too — the enforcement half of the parameter-line
    contract (#1830, the code owner's ruling on the second run).

    The same instance tokens, one level down: a parameter's name says what the value
    MEANS to the routine and its description says what to supply next time, so neither
    can carry this occasion's value or the name of where it came from.  A
    ``citydesk_url — citydesk.example/front`` pair is a routine that can only be pointed
    back at the page it was taught on; ``first_site — the first front page to read`` is
    the same spot, re-suppliable.  The reference pairs carry no instance token at all,
    which is what makes this structural rather than a judgment.

    It reads each line's SUBSTANCE — an appended ``(e.g., …)`` is stripped first, because
    the fourth run's traces showed that clause is written after the line is decided and
    says nothing about the parameter (see :func:`_without_examples`)."""
    offenders = [
        f"{parameter.name} ({', '.join(used)})"
        for parameter in signature.parameters
        if (used := sorted(_substance_tokens(parameter) & set(instance_tokens)))
    ]
    return Check(
        "the parameters are generic",
        not offenders,
        kind="state",
        rationale=None if not offenders else f"named the occasion: {'; '.join(offenders)}",
    )


def _score_framing(
    signature: SkillSignature | None,
    families: Sequence[ParameterFamily],
    instance_tokens: Sequence[str],
) -> list[Check]:
    """The framing case's graded checks (#1830), read off the draw's own typed result.

    The parameter SET is the contract, and it is EXACT: each expected family is answered
    by exactly one drawn parameter, and the total count matches — which is the same
    check as "nothing else was asked for", since anything extra is a piece the user
    would be made to re-supply that their own ask already settled.  Semantic breadth
    lives in the families (a reference name is a target, never a string to match), so a
    well-judged different word passes and a missing or invented parameter does not.

    Plus the two structural generic checks: the occasion never appears in the name or
    the description, and never in a parameter's name or description either — the
    enforcement half of the parameter-line contract.

    The drawn NAME, DESCRIPTION and every parameter then ride ADVISORY
    (``scored=False``) — whether a name is WELL judged is read at joint review against
    the reference outputs on the ticket, and no scorer should fake that."""
    if signature is None:
        return _refused_framing(families)
    grouped = _classified(signature, families)
    return [
        *(_family_check(family, grouped[family.label]) for family in families),
        _exact_count_check(signature, families),
        _generic_framing_check(signature, instance_tokens),
        _generic_parameters_check(signature, instance_tokens),
        *_framing_advisories(signature),
    ]


def _refused_framing(families: Sequence[ParameterFamily]) -> list[Check]:
    """A refused draw fails every scored check with its reason named, never silently.
    The degraded state — a slug-named routine with nothing to bind — is honest
    behaviour, but it is not the decision the case is asking for."""
    refused = "the draw was refused — no signature came back"
    return [
        *(
            Check(f"asks for the {family.label}", False, kind="state", rationale=refused)
            for family in families
        ),
        Check("asks for nothing else", False, kind="state", rationale=refused),
        Check("the framing is generic", False, kind="state", rationale=refused),
        Check("the parameters are generic", False, kind="state", rationale=refused),
    ]


def _family_check(family: ParameterFamily, matched: Sequence[FramedParameter]) -> Check:
    """One expected piece of the ask, answered by exactly one drawn parameter — nothing
    answering it is a piece the routine can no longer be pointed at, and two answering it
    is the same piece asked for twice."""
    rationale = None
    if not matched:
        rationale = "no parameter answers it"
    elif len(matched) > 1:
        rationale = f"{len(matched)} answer it: {', '.join(p.name for p in matched)}"
    return Check(
        f"asks for the {family.label}", len(matched) == 1, kind="state", rationale=rationale
    )


def _exact_count_check(signature: SkillSignature, families: Sequence[ParameterFamily]) -> Check:
    """Nothing else was asked for — the same check as the count, since anything extra is
    a piece the user would be made to re-supply that their own ask already settled."""
    drawn = len(signature.parameters)
    return Check(
        "asks for nothing else",
        drawn == len(families),
        kind="state",
        rationale=None if drawn == len(families) else f"drew {drawn}, expected {len(families)}",
    )


def _framing_advisories(signature: SkillSignature) -> list[Check]:
    """What the draw committed to, verbatim and UNSCORED — whether a name is WELL judged
    is read at joint review against the reference outputs on the ticket, and no scorer
    should fake that.

    Each parameter's line carries the VALUE it was demonstrated with (#1868) and the run
    closes with the container name those values derive — which is what the draw actually
    decides now, since a round's identity is the skill plus its values.  THAT a value is
    the user's own words is the production validator's (an accepted draw cannot carry a
    value nobody said); WHICH span was the right one is the same kind of judgment as a
    name, so it is rendered for review rather than scored by a fixture."""
    return [
        Check(f"named it {signature.name!r}", True, kind="state", scored=False),
        Check(f"described it {signature.description!r}", True, kind="state", scored=False),
        *(
            Check(
                f"asks {p.name!r} — {p.description!r} (drawn value {p.value!r})",
                True,
                kind="state",
                scored=False,
            )
            for p in signature.parameters
        ),
        Check(
            f"derives the container {_derived_container(signature)!r}",
            True,
            kind="state",
            scored=False,
        ),
    ]


def _derived_container(signature: SkillSignature) -> str:
    """The container this framing would build, through the SHIPPED derivation — never a
    copy of the scheme, so what the report shows is the name production would use."""
    return derive_collection_name(
        signature.name, [parameter.value for parameter in signature.parameters]
    )


@pytest.fixture
def framer_eval(make_config: Callable[..., Config], tmp_path, request) -> FramerEval:
    """Drive the run-end skill FRAMER (#1830) N times, and NOTHING else.

    The framer's whole input is the round's USER turns, so the case IS those turns —
    there is no upstream draw to isolate it from and nothing else to fixture.  The
    labeller is not run: it decides the implementation from the demonstration, this
    decides the interface from the ask, and neither sees the other's evidence (#1824).

    Everything the draw consumes is built by PRODUCTION code — ``build_framing_content``
    renders the document and ``MicroContext.frame_skill`` makes the call — so the case
    exercises the shipped prompt and the shipped parse.  Synthetic here means the TURNS
    are authored, never the prompt (an eval that swaps in an artificial prompt measures
    nothing about what ships).
    """

    async def _run(
        *,
        case_id: str,
        turns: Sequence[str],
        parameters: Sequence[ParameterFamily],
        instance_tokens: Sequence[str],
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
        timeout: float = 60.0,
        family: str | None = None,
    ) -> None:
        eval_artifacts.begin_case(case_id)
        content = build_framing_content(
            "", [(PennyConstants.MessageDirection.INCOMING, turn) for turn in turns]
        )

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            micro = MicroContext(penny.model_client)
            try:
                signature = await asyncio.wait_for(
                    micro.frame_skill(content, run_target=penny.chat_agent.name),
                    timeout=timeout,
                )
                scored = _score_framing(signature, parameters, instance_tokens)
                result = _guarded_graded(list(scored), [])
                result.fragile = result.passed and _run_end_rerolled(penny.db)
                _stamp_cause(penny.db, result)
            except TimeoutError:
                result = SampleResult.binary(["no framing draw within timeout"])
                _stamp_cause(penny.db, result, timed_out=True)
            _write_classifier_report(
                penny.db,
                case_id,
                sample_index,
                result=result,
                phrasing=turns[-1] if turns else "",
                agent_names=(PennyConstants.SKILL_FRAME_AGENT_NAME,),
            )
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

        results, perf = await _run_samples(
            make_config, tmp_path, case_id=case_id, samples=samples, drive=_drive
        )
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
            min_pass_rate=min_pass_rate,
        )
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate, intended=samples)

    return _run


@pytest.fixture
def labeller_eval(make_config: Callable[..., Config], tmp_path, request) -> LabellerEval:
    """Drive the run-end LEAF LABELLER (#1828) N times, and NOTHING else.

    The labeller has one job — name every spot in the demonstrated routine and say what
    belongs there each run — and this case measures that answer directly: the returned
    ``SkillLabels``.  Its input is a FIXTURE ledger through DETERMINISTIC distillation,
    so nothing live sits upstream of the draw either, and everything downstream of it
    is deterministic Python pinned in ``make check``.

    The demonstration is a fixture precisely so the case measures the NAMING and never
    polices what a round chose to write — if a round writes two entries, two entries
    are the skill (the code owner's ruling on #1770, unchanged).
    """

    async def _run(
        *,
        case_id: str,
        utterance: str,
        calls: Sequence[DemoCall],
        target: str,
        leaves: Sequence[str],
        conversation: Sequence[DemoTurn] = (),
        distinct_names: Sequence[tuple[str, str]] = (),
        shared_spot: str = "",
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
        timeout: float = 60.0,
        family: str | None = None,
    ) -> None:
        eval_artifacts.begin_case(case_id)
        content, by_value = _labelling_input(calls, target, utterance, conversation)
        # The spots the rendered document offered, in leaf order — the COVERAGE set the
        # draw is accepted against (#1828).  Read off the distilled leaves rather than
        # authored, so the case can never offer the draw a set the content didn't list.
        offered = list(dict.fromkeys(by_value.values()))

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            micro = MicroContext(penny.model_client)
            try:
                labels = await asyncio.wait_for(
                    micro.label_skill(content, offered, run_target=penny.chat_agent.name),
                    timeout=timeout,
                )
                scored = _score_labelling(labels, by_value, leaves, distinct_names, shared_spot)
                result = _guarded_graded(list(scored), [])
                result.fragile = result.passed and _run_end_rerolled(penny.db)
                _stamp_cause(penny.db, result)
            except TimeoutError:
                result = SampleResult.binary(["no label within timeout"])
                _stamp_cause(penny.db, result, timed_out=True)
            _write_classifier_report(
                penny.db,
                case_id,
                sample_index,
                result=result,
                phrasing=utterance,
                agent_names=(PennyConstants.SKILL_NAMING_AGENT_NAME,),
            )
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

        results, perf = await _run_samples(
            make_config, tmp_path, case_id=case_id, samples=samples, drive=_drive
        )
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
            min_pass_rate=min_pass_rate,
        )
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate, intended=samples)

    return _run


# ── Skill BINDING: filling an existing signature from the ask (#1867) ─────────

BinderEval = Callable[..., Awaitable[None]]


class BoundExpectation(NamedTuple):
    """One declared parameter and what a correct bind for it looks like (#1867).

    ``anchor`` is the part of the user's own words the value has to carry, compared
    through the SHIPPED ``spoken_form`` — so a bound url passes whether or not the draw
    kept the scheme, and a phrase passes whether or not it kept the article in front of
    it.  Deliberately not an equality: which span of the ask supplies a value has a
    little play in it, and a scorer demanding one exact string would be answering for the
    draw.

    An EMPTY anchor is the SHORTFALL direction: the ask supplies nothing for this
    parameter, so the only correct answer is the missing outcome naming it, and any value
    at all is a guess."""

    parameter: str
    anchor: str = ""


def _binder_rerolled(db: Database) -> bool:
    """Did the binder draw more than once — the fragile signal, a sample that only got
    there by recovering.  One customer here, so one row is the clean case."""
    return len(_micro_context_rows(db, PennyConstants.SKILL_BIND_AGENT_NAME)) > 1


def _bound_check(expectation: BoundExpectation, value: str | None, reported: bool) -> Check:
    """A parameter the ask DOES supply: a value came back, and it carries what the ask
    supplies for it."""
    if value is None:
        return Check(
            f"binds the {expectation.parameter}",
            False,
            kind="state",
            rationale="reported it missing" if reported else "no value came back for it",
        )
    carried = spoken_form(expectation.anchor) in spoken_form(value)
    return Check(
        f"binds the {expectation.parameter}",
        carried,
        kind="state",
        rationale=None if carried else f"bound {value!r}, not the value the ask supplies",
    )


def _shortfall_check(parameter: str, guessed: str | None, named: bool) -> Check:
    """A parameter the ask does NOT supply: the only honest answer names it missing.

    A value here is the failure the span check guards one level up — the words are short
    of something and the draw filled it anyway — so the rationale quotes what it made
    up."""
    if named:
        return Check(f"reports the {parameter} missing", True, kind="state")
    rationale = f"bound it to {guessed!r}" if guessed is not None else "neither bound nor reported"
    return Check(f"reports the {parameter} missing", False, kind="state", rationale=rationale)


def _no_terms_check(values: dict[str, str], forbidden: Sequence[str]) -> Check:
    """No job TERM landed in a value (#1867).

    How often a routine runs and when it stops are settled where the job is set running,
    never by the binder — so an ask's cadence and expiry words turning up INSIDE a bound
    value mean the draw read the terms as part of the thing to point the routine at.
    Structural: each case names the term words its own ask carries, and none of them may
    appear in any value.  An ask stating no terms has nothing to check."""
    if not forbidden:
        return Check.na("no job term landed in a value", kind="state")
    offenders = [
        f"{name} ({term})"
        for name, value in values.items()
        for term in forbidden
        if spoken_form(term) in spoken_form(value)
    ]
    return Check(
        "no job term landed in a value",
        not offenders,
        kind="state",
        rationale=None if not offenders else f"carried the terms: {'; '.join(offenders)}",
    )


def _binding_advisories(binding: SkillBinding) -> list[Check]:
    """What the draw committed to, verbatim and UNSCORED — every value it bound and every
    parameter it declined, so a report shows the answer whichever way it went."""
    checks = [
        Check(f"bound {name!r} = {value!r}", True, kind="state", scored=False)
        for name, value in binding.values.items()
    ]
    if isinstance(binding, MissingParameters):
        declined = ", ".join(binding.names)
        checks.append(Check(f"reported missing: {declined}", True, kind="state", scored=False))
    return checks


def _refused_binding(expectations: Sequence[BoundExpectation]) -> list[Check]:
    """A refused draw fails every scored check with its reason named, never silently.
    Returning nothing is honest when the words cannot be read into the signature, but it
    is not the decision the case is asking for."""
    refused = "the draw was refused — no binding came back"
    return [
        *(
            Check(
                f"binds the {expectation.parameter}"
                if expectation.anchor
                else f"reports the {expectation.parameter} missing",
                False,
                kind="state",
                rationale=refused,
            )
            for expectation in expectations
        ),
        Check("no job term landed in a value", False, kind="state", rationale=refused),
    ]


def _score_binding(
    binding: SkillBinding | None,
    expectations: Sequence[BoundExpectation],
    forbidden: Sequence[str],
) -> list[Check]:
    """The binding case's graded checks (#1867), read off the draw's own typed answer.

    One check per declared parameter — bound to a span of the ask carrying the value the
    ask supplies, or named missing when it supplies none — plus the structural check that
    no job term rode into a value.  Membership, coverage and "is this even in the user's
    words" belong to the production validator, so an accepted draw never reaches here
    carrying an invented value; what is left to measure is whether it picked the RIGHT
    span, and whether it knew when to decline.

    The drawn values then ride ADVISORY, as the framing case's do — what a well-chosen
    span looks like is read at joint review against the reference values on the ticket."""
    if binding is None:
        return _refused_binding(expectations)
    missing = binding.names if isinstance(binding, MissingParameters) else ()
    verdicts = [
        _shortfall_check(one.parameter, binding.values.get(one.parameter), one.parameter in missing)
        if not one.anchor
        else _bound_check(one, binding.values.get(one.parameter), one.parameter in missing)
        for one in expectations
    ]
    return [
        *verdicts,
        _no_terms_check(binding.values, forbidden),
        *_binding_advisories(binding),
    ]


@pytest.fixture
def binder_eval(make_config: Callable[..., Config], tmp_path, request) -> BinderEval:
    """Drive the skill BINDER (#1867) N times, and NOTHING else.

    The binder's whole input is an EXISTING signature and the round's user turns, so the
    case is those two things — there is no upstream draw to isolate it from and nothing
    else to fixture.  The routing draw that picks WHICH skill is not run: that decision
    is the classifier's and the transitions suite measures it; this one measures the
    filling, which is a separate draw by ruling (#1803).

    Everything the draw consumes is built by PRODUCTION code — ``render_spoken_turns``
    and ``build_binding_content`` render the document and ``MicroContext.bind_skill``
    makes the call — so the case exercises the shipped prompt, the shipped parse and the
    shipped validation.  Synthetic here means the TURNS and the SIGNATURE are authored,
    never the prompt.
    """

    async def _run(
        *,
        case_id: str,
        turns: Sequence[str],
        skill: str,
        intent: str,
        parameters: Sequence[SkillParameter],
        expectations: Sequence[BoundExpectation],
        forbidden: Sequence[str] = (),
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
        timeout: float = 60.0,
        family: str | None = None,
    ) -> None:
        eval_artifacts.begin_case(case_id)
        spoken = render_spoken_turns(turns)
        content = build_binding_content(spoken, skill, intent, parameters)
        declared = [parameter.name for parameter in parameters]

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            micro = MicroContext(penny.model_client)
            try:
                binding = await asyncio.wait_for(
                    micro.bind_skill(content, declared, spoken, run_target=penny.chat_agent.name),
                    timeout=timeout,
                )
                scored = _score_binding(binding, expectations, forbidden)
                result = _guarded_graded(list(scored), [])
                result.fragile = result.passed and _binder_rerolled(penny.db)
                _stamp_cause(penny.db, result)
            except TimeoutError:
                result = SampleResult.binary(["no binding draw within timeout"])
                _stamp_cause(penny.db, result, timed_out=True)
            _write_classifier_report(
                penny.db,
                case_id,
                sample_index,
                result=result,
                phrasing=turns[-1] if turns else "",
                agent_names=(PennyConstants.SKILL_BIND_AGENT_NAME,),
            )
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

        results, perf = await _run_samples(
            make_config, tmp_path, case_id=case_id, samples=samples, drive=_drive
        )
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
            min_pass_rate=min_pass_rate,
        )
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate, intended=samples)

    return _run


# ── Browse EXTRACTION: what a page half-carries (#1942) ───────────────────────

ExtractorEval = Callable[..., Awaitable[None]]


class FieldExpectation(NamedTuple):
    """One thing an extract instruction asks for, and how the page answers it (#1942).

    ``anchor`` is the span of the page that supplies it, compared through the SHIPPED
    ``spoken_form`` — so a value passes whether or not the draw kept the punctuation or
    the article in front of it.  Deliberately not an equality: which words carry a fact
    has a little play in it, and a scorer demanding one exact string would be answering
    for the draw.

    An EMPTY anchor is the ABSENT direction: the page supplies nothing for this, so it is
    named here to say what the instruction asked for and the page did not have."""

    field: str
    anchor: str = ""


def _extract_rerolled(db: Database) -> bool:
    """Did the extraction draw more than once — the fragile signal, a sample that only
    got there by recovering.  One customer here, so one row is the clean case."""
    return len(_micro_context_rows(db, PennyConstants.BROWSE_EXTRACT_AGENT_NAME)) > 1


def _extraction_outcome_check(result: MicroContextResult, carries_something: bool) -> Check:
    """The decisive check: a page carrying SOME of what was asked for is a read, and a
    page carrying NONE of it is honestly empty.

    Which direction this case is comes from the expectations themselves — whether any of
    them names a span the page supplies — so the case declares the world and the scorer
    reads the contract off it, rather than being told the answer twice."""
    wanted = MicroExtractOutcome.EXTRACTED if carries_something else MicroExtractOutcome.NOT_PRESENT
    label = (
        "reads the page rather than reporting it empty"
        if carries_something
        else "reports the page carries none of it"
    )
    return Check(
        label,
        result.outcome == wanted,
        kind="state",
        rationale=None if result.outcome == wanted else f"came back {result.outcome}",
    )


def _field_check(result: MicroContextResult, expectation: FieldExpectation) -> Check:
    """One thing the instruction asked for that the page DOES supply: the extracted value
    has to carry it.  This is the per-field half — an instruction naming several things
    degrades one thing at a time, so each is scored on its own."""
    carried = spoken_form(expectation.anchor) in spoken_form(result.value)
    return Check(
        f"carries the {expectation.field}",
        carried,
        kind="state",
        rationale=None if carried else "not in the extracted value",
    )


def _absent_field_note(expectation: FieldExpectation) -> Check:
    """One thing the instruction asked for that the page does NOT supply — ADVISORY.

    Nothing about a gap is separately measurable: leaving it out and naming it are both
    honest, and there is no string whose absence proves the draw declined to invent one.
    What the gap actually costs — or does not — is the OUTCOME, which is already the
    decisive check, so scoring this too would grade one fact twice and weight it by how
    many things the page happened to lack.  It renders so a report shows WHICH thing the
    page was short of, rather than leaving a reader to infer it from the instruction."""
    return Check(f"the page carries no {expectation.field}", True, kind="state", scored=False)


def _extraction_advisories(result: MicroContextResult) -> list[Check]:
    """What the draw committed to, verbatim and UNSCORED — the value it returned or the
    absence it reported, so a report shows the answer whichever way it went."""
    if result.outcome == MicroExtractOutcome.EXTRACTED:
        return [Check(f"extracted {result.value!r}", True, kind="state", scored=False)]
    return [Check(f"{result.outcome}: {result.reason!r}", True, kind="state", scored=False)]


def _score_extraction(
    result: MicroContextResult, expectations: Sequence[FieldExpectation]
) -> list[Check]:
    """The extraction case's graded checks (#1942), read off the draw's own typed result.

    Two graded kinds: the OUTCOME — a page that half-answers an instruction is a read,
    not a NOT_PRESENT — and one check per thing the instruction named that the page DOES
    supply, since an instruction naming several things degrades one thing at a time.  What
    the page lacks rides ADVISORY (there is no measurable fact in a gap beyond the outcome
    itself), and so does the value the draw returned: what a well-chosen value looks like
    is read at joint review, as the framing and binding cases' answers are, not asserted
    by a scorer."""
    carries_something = any(one.anchor for one in expectations)
    return [
        _extraction_outcome_check(result, carries_something),
        *(
            _field_check(result, one) if one.anchor else _absent_field_note(one)
            for one in expectations
        ),
        *_extraction_advisories(result),
    ]


@pytest.fixture
def extractor_eval(make_config: Callable[..., Config], tmp_path, request) -> ExtractorEval:
    """Drive the browse EXTRACTION micro-context (#1588/#1942) N times, and NOTHING else.

    The extraction's whole input is one fetched page's section and one instruction, so
    the case is those two things — no browse runs, because what is measured is what the
    extractor makes of a page rather than whether the page can be fetched.  The section is
    built the way ``BrowseTool._page_section`` builds it, so the draw reads the document
    production hands it.

    ``MicroContext.extract`` makes the call, so the case exercises the shipped prompt, the
    shipped parse and the shipped reroll budget.  Synthetic here means the PAGE and the
    INSTRUCTION are authored, never the prompt.
    """

    async def _run(
        *,
        case_id: str,
        url: str,
        page: str,
        instruction: str,
        expectations: Sequence[FieldExpectation],
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
        timeout: float = 60.0,
        family: str | None = None,
    ) -> None:
        eval_artifacts.begin_case(case_id)
        content = f"{PennyConstants.BROWSE_PAGE_HEADER}{url}\n{page}"

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            micro = MicroContext(penny.model_client)
            try:
                extracted = await asyncio.wait_for(
                    micro.extract(content, instruction, run_target=penny.chat_agent.name),
                    timeout=timeout,
                )
                scored = _score_extraction(extracted, expectations)
                result = _guarded_graded(list(scored), [])
                result.fragile = result.passed and _extract_rerolled(penny.db)
                _stamp_cause(penny.db, result)
            except TimeoutError:
                result = SampleResult.binary(["no extraction draw within timeout"])
                _stamp_cause(penny.db, result, timed_out=True)
            _write_classifier_report(
                penny.db,
                case_id,
                sample_index,
                result=result,
                phrasing=instruction,
                agent_names=(PennyConstants.BROWSE_EXTRACT_AGENT_NAME,),
            )
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

        results, perf = await _run_samples(
            make_config, tmp_path, case_id=case_id, samples=samples, drive=_drive
        )
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
            min_pass_rate=min_pass_rate,
        )
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate, intended=samples)

    return _run
