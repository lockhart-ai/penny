"""Fixtures for the live-model eval suite.

Construction reuses the integration-test isolation core (``running_penny``)
with a config whose model points at the real Ollama endpoint — no second
construction path, no stubs.  Each case drives one cohort (the model is
stochastic): K wordings of one ask, sampled N times each, read off PERSISTED DB
state.  The case body makes its claims against the cohort and names the features
it measures; the claims are counted and reported, never gated.  What fails a run
is run health — a cohort that mostly never ran.  See docs/eval-case-design.md.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
from collections import Counter
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterator,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from similarity.embeddings import deserialize_embedding
from sqlmodel import Session, col, select

from penny.agents.collector import Collector
from penny.config import Config
from penny.constants import ChannelType, MutationEntityType, PageReading, PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput, MemoryType
from penny.database.memory.store import snapshot_fields
from penny.database.message_store import MessageStore, PromptPerf
from penny.database.models import Media, MemoryRow, MutationEvent, PromptLog, SendQueueItem
from penny.database.mutation_store import mutation_detail
from penny.datetime_utils import user_timezone_name
from penny.llm.call_link import CallLink
from penny.llm.models import CallOutcome, LlmFault, strip_harmony_control_tokens
from penny.notification import NotificationInput, render_notification
from penny.penny import Penny
from penny.preflight import PreflightError
from penny.responses import PennyResponse
from penny.tests.conftest import (
    TEST_SENDER,
    require_memory,
    run_penny_with_server,
    stop_task,
    wait_until,
)
from penny.tests.eval.utils import artifacts as eval_artifacts
from penny.tests.eval.utils import assertions as eval_assertions
from penny.tests.eval.utils import catalogue as eval_catalogue
from penny.tests.eval.utils import cohort as eval_cohort
from penny.tests.eval.utils import report, run_health
from penny.tests.eval.utils.artifacts import FailureCause
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.baseline import Baseline, baseline_from_env
from penny.tests.eval.utils.budget_hits import BudgetHits, tally_budget_hits
from penny.tests.eval.utils.catalogue import Area, Layer
from penny.tests.eval.utils.clock import SAMPLE_TIMEZONE, PinnedClock
from penny.tests.eval.utils.fixtures import CannedPage, SynthCollection
from penny.tests.eval.utils.given import read_given
from penny.tests.eval.utils.mailbox import CannedEmail, CannedMailbox
from penny.tests.eval.utils.worlds import World
from penny.tests.mocks.signal_server import MockSignalServer
from penny.text_validity import (
    has_leaked_harmony_envelope,
    is_call_fragment_reply,
    is_degenerate_run,
    is_degenerate_tool_name,
)
from penny.tools.micro_context import (
    MicroContext,
    MicroContextResult,
    MicroExtractOutcome,
)

# A new eval DRIVER (a ``*_eval`` pytest fixture) lives in its own module under
# ``drivers/``, never appended here (#2395) — two case PRs that each add a driver no
# longer land on the same lines.  Loaded as a plugin from ``penny/tests/conftest.py``
# (the suite's actual root — ``pytest_plugins`` is only honoured in an INITIAL
# conftest, and this one isn't always that: a run scoped under ``penny/tests/eval/``
# makes it initial, but the real suite root (``penny/tests/``) never does), so
# dropping a module into ``drivers/`` is the whole registration — nothing here
# names it, and nothing here declares ``pytest_plugins`` either.

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

# How long ONE drive of a sample may take — standing its world up, its turns or cycles, its
# teardown — before the harness stops it and excludes it by name (#2168).  The waits inside a
# drive carry budgets of their own (the readiness wait above, each turn's reply timeout of up to
# 300s, the model client's per-call deadline), so this is the backstop for a wait that has
# none: a run once sat twenty minutes on one sample whose teardown was waiting on a task that
# had absorbed its cancel, and wrote nothing until it was killed by hand.  Generous rather than
# tuned: a healthy drive, a slow local model's several turns or collector cycles included,
# finishes well inside it, and a cut here costs a sample that could have been measured.
SAMPLE_WALL_CLOCK_SECONDS = 1800.0

# How long a single-call drive (classifier, framer, namer, binder, extractor) waits for its one
# model call.  The client's per-request deadline does not bound it: a gateway that sends
# keep-alive bytes while the upstream reasons never trips a read timeout, so the call simply
# runs as long as the model thinks.  Measured on the first whole-suite baseline: a seven-spot
# namer draw averaged 38s on one roster model, and at a 60s wait 19 samples in 8 cases were cut
# before they answered — 8 of 15 in one case — while the other model lost none.  A cut sample
# measures the wait, not the model, so the wait is generous rather than tuned; the slowness
# itself is read off the case's timings.
SINGLE_CALL_WAIT_SECONDS = 180.0

# How many times one sample is stood up before a failing preflight voids it (#2230), and how
# long it waits before the first re-boot (doubling each time after).  A sample's preflight
# probes the endpoint ONCE, so one timed-out request used to lose the sample and thin the
# run — measured on four consecutive runs, one sample each, at a random index, with no fault
# on any model call beside it.  Only a TRANSIENT failure earns another boot (see
# :func:`preflight_was_transient`); a verdict voids at once.  Small on purpose: a provider
# that cannot answer a model listing three times in a few seconds is not having a moment, and
# the sample is then voided by name exactly as before.
SAMPLE_BOOT_ATTEMPTS = 3
SAMPLE_BOOT_RETRY_SECONDS = 2.0

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


# A seeder lays a sample's world down in its database before the measured turn.
Seeder = Callable[[Database], None]
# A preparer mutates the constructed Penny before the message is pushed — e.g.
# to mock an external boundary (the image client) the case exercises.
Preparer = Callable[[Penny], None]


@dataclass
class Check:
    """One claim's answer for ONE sample — the per-sample form of a cohort claim.

    A case makes its claims over the whole cohort; ``_cohort_checks`` deals each claim's
    answers back out by sample, so a sample's score, the report's per-sample cells and the
    ``results.jsonl`` record read one object.  ``label`` is the claim's own sentence.

    ``rationale`` is the observed-vs-expected note rendered beside the outcome
    (``changed ['<key>']``), so a ❌ is never bare.

    ``scored`` and ``ignored`` are the advisory and not-applicable states the report and
    artifact layers render.  Nothing in the harness sets either: a claim is strictly true or
    false of a sample, and every claim counts."""

    label: str
    ok: bool
    anchor: str | None = None  # substring of the transcript row this check marks (None = no row)
    scored: bool = True  # False = advisory, visible in the report, not in the score
    rationale: str | None = None  # observed-vs-expected note rendered beside the outcome
    ignored: bool = False  # not-applicable: rendered (➖) but out of the graded denominator
    kind: str | None = None  # class label rendered `[kind]` (spine/reply/state/proc/guard)


@dataclass
class SampleResult:
    """A sample's score in [0, 1] + the labels of whatever didn't pass (for the report).

    The score is claims-passed over claims-answered for this sample, settled when the cohort's
    claims are dealt back out at case close (``adopt``).  A sample that timed out or was
    excluded scores 0.0 and carries its reason where a failure's labels go."""

    score: float
    failed: list[str]
    total: int = 1
    checks: list[Check] = field(default_factory=list)  # this sample's answered claims
    # The structural failure cause (#1695), settled wherever the score is: ``None`` for a pass;
    # ``behavioral`` / ``pathology`` / ``harness`` for a failure.  The artifact aggregate defaults
    # an unstamped failure to behavioral, so a directly-constructed result is safe.
    cause: FailureCause | None = None
    # The structural facts the cause and the fragile flag are READ from, observed while the
    # sample's database was live (#2125, #2127).  Carried rather than folded straight into the
    # verdict because a sample's score is not settled at drive time — its cohort's claims
    # come back at case close, long after the database is gone — so both derivations have to run
    # again there, and a derivation needs its inputs.
    timed_out: bool = False
    pathology: bool = False
    rerolled: bool = False
    # Why the cohort could not pool this sample, dealt out with the claims at the same seam
    # (#2125).  ``None`` for a sample that was pooled.
    excluded: str | None = None
    # Passed-but-shaky (#1725, #1694): the sample reached its result only after the loop
    # refused/recovered a draw.  DERIVED from the settled score (#2127), the way the cause is —
    # stamped instead from the drive-time ``passed``, which a sample earns VACUOUSLY
    # before it has answered a claim, it degenerated into "the run rerolled" and marked
    # claim-FAILING samples fragile.  It renders on the sample's own banner (the flag, and the
    # legend note that flag pulls in) and rides into ``CaseArtifact.sample_fragile``.
    fragile: bool = False
    # What this sample LEFT BEHIND, read while its database was still live (#1995) — the whole
    # of what a case asserts or measures.  ``None`` only for a result built outside a drive.
    observation: eval_cohort.SampleObservation | None = None

    @property
    def passed(self) -> bool:
        return self.score >= 1.0

    @property
    def aborted_on_model_error(self) -> bool:
        """Whether this sample's turn ended on a model error (#2252) — a straight read of its
        observation, carried for the record: unlike the cause/fragile flags, nothing about it
        changes when the cohort's claims settle the score, since it is a fact about how the
        turn ended rather than about what it is scored against."""
        return self.observation is not None and self.observation.aborted_on_model_error

    @classmethod
    def binary(cls, fails: list[str]) -> SampleResult:
        return cls(0.0 if fails else 1.0, list(fails), 1)

    def adopt(self, checks: list[Check]) -> None:
        """Take on the claims the COHORT answered for this sample (#1995).

        A case is graded from claims made over the whole cohort AFTER the drive, so the
        sample's own score is settled here rather than at drive time.  Re-derived through
        ``graded`` so one definition decides what counts, whatever produced the checks."""
        graded = SampleResult.graded(checks)
        self.score, self.failed, self.total, self.checks = (
            graded.score,
            graded.failed,
            graded.total,
            graded.checks,
        )
        self._settle()

    def exclude(self, reason: str) -> None:
        """The cohort could not pool this sample, so record it as the LOSS it is (#2125).

        An excluded sample answers no claim (``Cohort._answer`` skips it), and an empty answer
        set grades a vacuous 1.0 — so the record reported a sample the case document names as
        lost as a passing one, and a later run diffed against it saw no regression.  It scores
        nothing, carries its reason where a failure's labels go, and reads as the infrastructure
        fault it is rather than as the model getting anything wrong."""
        self.excluded = reason
        self.score, self.failed, self.total, self.checks = 0.0, [reason], 1, []
        self._settle()

    def observe_faults(self, *, timed_out: bool, pathology: bool) -> None:
        """The structural facts this sample's CAUSE is read from, observed while its database
        was live — and the cause settled from them."""
        self.timed_out, self.pathology = timed_out, pathology
        self._settle_cause()

    def observe_rerolled(self, *, rerolled: bool) -> None:
        """The structural fact this sample's FRAGILE flag is read from — did it reach its result
        only after a draw was refused, rejected or nudged — observed while its database was live,
        and the flag settled from it."""
        self.rerolled = rerolled
        self._settle_fragile()

    def _settle(self) -> None:
        """Re-derive everything the SCORE decides — called wherever the score itself settles.

        Each derivation also runs where its own fact is observed; this is the seam that runs
        BOTH, because a new score moves the answer to both questions at once (#2125, #2127)."""
        self._settle_cause()
        self._settle_fragile()

    def _settle_cause(self) -> None:
        """Why this sample failed, derived from the carried facts (#2125).

        One derivation, applied at drive time and again when a cohort's claims arrive, so a
        re-scored sample can never keep the cause its old score earned."""
        self.cause = eval_artifacts.classify_cause(
            passed=self.passed,
            # An exclusion is the same KIND of fact a timeout is: the sample produced nothing
            # to judge, for a reason the model never touched.
            timed_out=self.timed_out or self.excluded is not None,
            pathology=self.pathology,
        )

    def _settle_fragile(self) -> None:
        """Whether this sample is a PASSED-but-shaky one, derived the same way (#2127).

        A sample the claims failed is never fragile: it is a failure, and its cause says which
        kind.  Stamped instead from the drive-time ``passed`` — which a sample earns
        vacuously, before it has answered a claim — the flag said only that the run rerolled."""
        self.fragile = self.passed and self.rerolled

    @classmethod
    def graded(cls, checks: list[Check]) -> SampleResult:
        """The score a sample's answered claims come to: passed over answered.

        No checks at all is a vacuous 1.0 — what a sample holds at drive time, before its
        cohort's claims arrive."""
        if not checks:
            return cls(1.0, [], 1)
        # NOT-APPLICABLE checks (``ignored``) never count; among the rest, score over the
        # SCORED ones only (an advisory check renders but doesn't count), with an all-advisory
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
    stopwatch.  Printed per case so ``make eval-remote`` shows wall time and tok/s.
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

    ``make eval-remote`` forwards its whole variable list explicitly, so a variable the caller
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

    The sample's CLOCK is set here too, on the one seam every runner builds its config
    through: a ``PinnedClock`` that starts at the suite's declared instant, so what a
    sample is told the time is — and everything it records and counts from — does not
    depend on when the run is made (``utils/clock.py``, #2207).
    """
    return make_config(
        signal_api_url=signal_api_url,
        # A case is parametrized over MODEL, so a cohort names the one it is measured on;
        # with none named, the run's configured model.
        llm_model=model or os.environ.get("LLM_MODEL", "gpt-oss:20b"),
        llm_api_url=os.environ.get("LLM_API_URL", "http://localhost:11434"),
        llm_embedding_model=os.environ.get("LLM_EMBEDDING_MODEL", "embeddinggemma"),
        llm_max_retries=_EVAL_LLM_MAX_RETRIES,
        llm_retry_delay=_EVAL_LLM_RETRY_DELAY,
        llm_timeout=_EVAL_LLM_TIMEOUT,
        db_path=db_path,
        clock=PinnedClock(),
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
        timezone=SAMPLE_TIMEZONE,
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
    """Create a synthetic collection + its entries, keyed as ``SynthCollection.keyed`` says.

    The key derivation is the collection's own, not this function's, because the world render
    shows these same rows to say what the sample was answering against — a report keying an
    entry differently from the row this wrote describes a store nobody seeded.

    THE one place a mechanism is laid down for a sample, so the run its creation cites is
    decided here rather than remembered at each call — the registry create and the entry write
    both cite the collection's own seeded run (#2129).  A birth is itself a mutation event, and
    ``is_seeded_run(None)`` is false, so an unstamped creation reads as a LIVE run's work: every
    claim that this turn created or changed no mechanism failed on every sample, naming a
    collection the turn never touched.  The run is named for the COLLECTION rather than for the
    world or the case, because this is the only anchor available at a seam a case file calls
    directly — and a parameter for it would be the thing a caller can forget."""
    seeded_by = seeded_run_id(synth.name)
    db.memories.create_collection(
        synth.name,
        synth.description,
        extraction_prompt=extraction_prompt,
        schedule=schedule,
        notify=notify,
        created_by_run_id=seeded_by,
    )
    require_memory(db, synth.name).write(
        [EntryInput(key=key, content=content) for key, content in synth.keyed],
        author="user",
        run_id=seeded_by,
    )


def seed_world_stores(db: Database, world: World | None) -> None:
    """Lay down the entries a world says are ALREADY in the store when the turn begins.

    The world is what a sample is GIVEN — the driver already serves its ``pages`` as the
    browse register — so its ``stores`` are seeded from the same declaration the report
    renders. That is what closes the drift the report otherwise carries: a world cannot claim
    a ground the sample never had, because the claim IS the seed (#2108).

    Through ``seed_collection`` rather than the store directly, so a declared store's creation
    cites a seeded run without this loop having to know that it must (#2129)."""
    for held in world.stores if world is not None else ():
        seed_collection(db, held)


# How far back ONE mechanism's mutation history is read when asking whether THIS sample
# touched it.  Generous rather than tight: a seeded job carries two events (created by its
# round, re-rendered by its apply turn), and reading well past them is what makes "this turn
# changed nothing here" a read rather than a guess about where the window fell.
#
# Declared HERE because two readers ask the same question of the same read — the observation
# below and the transition suite's own jobs-untouched check — and two spellings of one window
# are two answers waiting to disagree about a world with a longer history than either expected.
MUTATION_HISTORY_WINDOW = 20


def collection_names(db: Database) -> set[str]:
    """Every memory name currently in the DB — the pre-run snapshot an observer reads."""
    return {memory.name for memory in db.memories.list_all()}


def collection_entries(db: Database, name: str) -> dict[str, str]:
    """``{key: content}`` for every keyed entry in a collection — the snapshot a cycle's
    footprint is read against, before and after, to see writes, edits and deletes."""
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


def live_budget_hits(db: Database) -> BudgetHits:
    """``tally_budget_hits`` over this sample's own rows — the same window
    ``live_prompt_perf`` reads, so a cut counted here is never a seeded prior turn's."""
    return tally_budget_hits(live_prompts(db, _PERF_WINDOW))


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


# Tool-result fragments that mean a call the model made was refused: the two
# failure-narration frames (tools/base.py: the generic failure + arg-validation) and the
# framework REFUSAL narrations (a call rejected before it ran, a duplicate not repeated, a
# missing / timed-out / errored tool) — the "did the run recover from something?" set the
# fragile-pass flag reads.
_RECOVERY_FRAMES = (
    "arguments were wrong",  # FRAMEWORK_NARRATION_INVALID_ARGS
    "didn't work",  # the generic failure frame
    "rejected before it could run",  # Prompt.REJECTED_CALL_NARRATION (e.g. a premature done())
    "wasn't repeated",  # Prompt.DUPLICATE_CALL_NARRATION
    "there's no such tool",  # FRAMEWORK_NARRATION_NOT_FOUND
    "it timed out",  # FRAMEWORK_NARRATION_TIMEOUT
    "it errored",  # FRAMEWORK_NARRATION_EXCEPTION
)


def sample_is_fragile(db: Database) -> bool:
    """Did the run reach its result SHAKILY — through a rejected / refused / recovered tool call,
    OR a framework RECOVERY NUDGE (a continue / parse-failure / tool-call-demand user turn)?

    Scans the persisted promptlog for either recovery shape: a tool-result failure / framework
    refusal frame (``_RECOVERY_FRAMES``, on a ``tool`` turn) OR a recovery nudge injected as a
    ``user`` turn (``_is_nudge`` — the SAME predicate the transcript render marks ``⚠ recovery
    event`` with, single-sourced so render and probe can't drift apart again, #1735 finding 2).
    A green sample that only got there after the loop refused a call and retried, or after a nudge
    recovered an empty / unparseable response, is 'passed, fragile' in the report: real, but not
    robust — exactly the robustness signal the report cares about.  Derived from the promptlog,
    not a new model judgment.  Fragile is render/artifact-only — never gated.

    The tool-turn leg filters on NO tool name — it asks "did the run recover from *anything*?"
    — so it is true of a failure whichever tool produced it: a memory-tool execute-time failure
    narrates ``… but it didn't work:`` with the TARGET backticked rather than the tool, and the
    ``didn't work`` fragment catches it all the same (#1726)."""
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
    #1732).  It reads only the ``response`` — what the model itself produced — and never the
    input ``messages``, so text the model was HANDED (a page quoting a collapse, a framework
    nudge) cannot tag a sample.

    **The nudge-frame boundary (#1732).** Repeated recovery-nudge frames are DELIBERATELY not
    counted as a pathology signal: a nudge is an INPUT message, and counting input would need
    an arbitrary count threshold.  A nudge loop is a *symptom* whose *cause* is the model's own
    fragment OUTPUT — the terminal ``{}`` / call-fragment reply the scan already catches on the
    ``response`` — so classifying on that output tags the #1731 spiral pathology at the root.
    (A spiral whose persisted output stays genuinely clean has no poison to tag and reads
    harness/behavioral — correctly: no pathology fired.)"""
    return any(_response_is_poison(row) for row in live_prompts(db))


def _stamp_cause(db: Database, result: SampleResult, *, timed_out: bool = False) -> None:
    """Stamp the sample's structural failure cause (#1695) in place — ``None`` for a pass.

    Called at every runner's per-sample append site so the cause rides into the
    ``results.jsonl`` record and the RESULT-line cause tally.

    The poison scan runs whatever the sample scored HERE, because what it scored here is
    nothing: a sample is graded from its cohort's claims at case close, long after this
    database is gone, so a scan skipped on a drive-time pass would leave every later-failing
    sample unclassifiable (#2125)."""
    result.observe_faults(timed_out=timed_out, pathology=run_exhibited_pathology(db))


# ── Reading a re-rolled draw off the promptlog ───────────────────────────────
# The text-bail nudges a pre-#1839 loop injected as a user turn.  They can no
# longer occur — PR #1840 deleted the constants and the validators that appended
# them — so these markers read HISTORICAL rows only, and a sample recorded before
# that change still reports its bail.  Each is an ASCII, newline-free slice that
# survives row.messages JSON-escaping.
_LEGACY_BAIL_NUDGE_MARKERS = (
    "could not be parsed as a tool call",  # the retired Prompt.TOOL_FORMAT_NUDGE
    "wrote a tool call as plain text",  # the retired Prompt.CHAT_CALL_AS_TEXT_NUDGE
)


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
    A response validator's ``Retry`` and ``MicroContext._draw_clean`` re-draw the same
    way, so one read covers the main loop's re-rolls, its validator retries, and every
    micro-context.

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


def outgoing_replies(db: Database) -> list[str]:
    """Every message Penny sent this sample (the per-turn replies), oldest first."""
    entries = require_memory(db, "penny-messages").read_recent(window_seconds=3600, cap=None)
    return [entry.content for entry in entries]


def chat_run_tool_sequences(
    db: Database, agent_name: str = PennyConstants.CHAT_AGENT_NAME
) -> list[list[str]]:
    """Tool names per run of ``agent_name``, in chronological run order — one list per user
    turn of a scripted conversation.  The per-run split is what lets a multi-turn
    contract assert phase discipline (an elicitation turn must not enact; the
    demonstration turn must carry the call spine) — ``tool_call_sequence`` flattens
    the whole sample into one list.  Micro-context calls (browse-extract, skill
    naming) carry no tool calls and other agents' rows are excluded, so each list
    is exactly one turn's calls, in emission order.

    The agent is a PARAMETER because the filter is otherwise a silent blindfold on every
    non-chat reader: a collector cycle's calls are logged under the collector's name, so a
    fixture reusing the chat default reads an empty sequence on every sample and its tool
    feature pools to a serene 0.000 — the same number a cohort in perfect agreement prints.
    :attr:`cohort.VarianceFeature.blind` is the backstop that makes such a reading loud; this
    is the fix that stops it happening."""
    rows = sorted(
        (row for row in live_prompts(db) if row.agent_name == agent_name),
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


def tool_call_name(call: dict) -> str:
    """One logged call's tool name, normalised the way PRODUCTION normalises it.

    Every read of a tool name off the promptlog goes through here, because the raw name can
    carry leaked Harmony control tokens (`collection_write<|channel|>commentary`, seen on two
    samples of one run).  Production strips them at the boundary where the name is read off the
    model response — ``LlmToolCallFunction.name``, via ``strip_harmony_control_tokens`` — so
    registry lookup, done-detection, dedup and result framing all see the clean identifier, and
    the eval reading the raw one was the single exception.

    What it cost while it was the exception: the sequence readers silently missed a call the
    store proves ran, which converted a correct sample into an outlier for a divergence that
    never happened.  Every case that measures a tool sequence was exposed to it, not just the
    one that surfaced it."""
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

    async def request_fn(method: str, params: dict) -> tuple[str, str | None, PageReading]:
        url = params.get("url", "").lower()
        for page in pages:
            if page.match.lower() in url:
                if page.fails:
                    raise _BrowseReadError(
                        f"failed to read {url} after 3 attempts: the source could not be read"
                    )
                return page.text, page.image, page.reading
        return _NO_RESULTS_PAGE, None, PageReading.ARTICLE

    def provider() -> tuple[Callable, MagicMock]:
        return request_fn, MagicMock(check_domain=AsyncMock())

    penny.chat_agent._browse_provider = provider
    penny.collector._browse_provider = provider


def install_mailbox(penny: Penny, emails: Sequence[CannedEmail], *, fails: bool = False) -> None:
    """Put a canned mailbox behind the chat agent's email tools.

    The email tools are config-gated: ``ChatAgent`` registers them only when a mailbox is
    configured, so installing the builder is what puts them on the surface at all.  The
    builder is PRODUCTION'S OWN Fastmail one (``Penny._fastmail_tools_builder``) wrapped
    around a ``CannedMailbox``, so the tools, their descriptions and the per-turn summarising
    read are the shipped ones and only the boundary behind them is canned.  Fastmail's
    surface — search and read — because a canned mailbox is never changed by a turn.
    ``fails=True`` (from ``World.mailbox_fails``) makes that boundary unreachable instead: the
    tools still install (so a world that means to fail is never scored on a turn that was
    never offered them), but every search and read raises."""
    mailbox = CannedMailbox(emails, fails=fails)
    penny.chat_agent._email_tools_builder = penny._fastmail_tools_builder(mailbox)


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


def _report_result(case_id: str, results: list[SampleResult], *, intended: int) -> None:
    """Refuse a dead cohort, then print the case's RESULT line.  Nothing here gates on a score.

    ``intended`` is how many samples the case ASKED for: a cohort that mostly died is
    refused before anything is printed, because "no result" is not a low score.

    The line carries two numbers over the per-sample scores (claims passed of claims
    answered): their MEAN, and the all-pass count beside it (samples every claim was true
    of), so a mean propped up by partial credit is visible.  A second line carries the
    failure-cause read (#1695): the pathology-excluded mean and the
    behavioral/pathology/harness tally, so a score sunk by model NOISE (a degeneracy spike)
    reads distinctly from one sunk by the model getting it WRONG.  Then, per sample that
    missed something, its score and what missed.
    """
    _refuse_dead_cohort(case_id, results, intended)
    total = len(results)
    mean = sum(result.score for result in results) / total if total else 0.0
    all_pass = sum(1 for result in results if result.passed)
    causes = eval_artifacts.sample_causes(results)
    excluded_mean, kept = eval_artifacts.pathology_excluded(
        [result.score for result in results], causes
    )
    cause_line = eval_artifacts.render_cause_summary(
        eval_artifacts.count_causes(causes), excluded_mean, kept
    )
    metric = f"mean {mean:.2f} · all-pass {all_pass}/{total}"
    print(f"\nRESULT [{case_id}] {metric} across {total} samples")
    print(f"  {cause_line}")
    for index, result in enumerate(results):
        if result.failed:
            print(f"  [{index + 1}] {result.score:.2f} — {'; '.join(result.failed)}")


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
    lines = [
        f"\n===== THINKING [{case_id} #{sample_number(sample_index)}] "
        f"— {len(rows)} LLM call(s) ====="
    ]
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
# When EVAL_REPORT_DIR is set (wired through by the Makefile `eval-remote` target), each sample
# appends a markdown section — the full turn-by-turn transcript read from the ephemeral
# promptlog before the --rm DB is discarded — to <dir>/<case_id>.md.  The SOP
# (docs/agent-task-workflow.md §4) folds these into the PR body under a <details> per case,
# so a reviewer sees every run verbatim without a wall of text.  Off by default (no dir set
# ⇒ no-op), so ordinary `make eval-remote` runs are unaffected.

_ACTOR = {
    "user": "👤 user",
    "tool": "📥 tool result",
    "call": "🔧 Penny → tool",
    "penny": "🤖 Penny",
    # An agent text turn that reached NOBODY.  A collector cycle narrates to itself and
    # delivers nothing — what a user receives is a send-queue row the framework enters after
    # the cycle — so rendering that narration with the reply mark said the opposite of the
    # truth on the one question a reader of that fold is asking.
    "aside": "🗒 Penny (to herself)",
    # A CYCLE BOUNDARY, not an actor (#2375): the marker turn the end-to-end walk emits where
    # one collector run ends and the next begins, so a cycle renders as a step of its own
    # instead of folding into the user turn that commissioned it.  Its content is the cycle's
    # short header description.
    "cycle": "⏱ cycle",
}

# The collector's ledger identity — ``Collector.name``, the ``promptlog.agent_name`` every
# cycle's main-loop rows carry.  Named once because the transcript walk reads cycle boundaries
# off the ledger (each run_id of this agent is one cycle), not off where the rows sit.
COLLECTOR_AGENT_NAME = Collector.name


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


def _cycle_descriptions(rows: list[PromptLog]) -> dict[str, str]:
    """One short step description per collector run, read off the runs' own records (#2375):
    the tool name of every model draw the run made, in ledger order — ``browse,
    collection_write``.  Keyed by ``run_id``, so a cycle's step opens naming what THAT cycle
    did rather than where its events happen to sit, and an identical draw made twice within
    one run reads once — the same de-duplication the transcript walk applies to the call
    itself (``_render_call``'s canonical form on both sides)."""
    drawn: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for row in rows:
        if (row.agent_name or "") != COLLECTOR_AGENT_NAME or not row.run_id:
            continue
        run_id = row.run_id
        seen.setdefault(run_id, set())
        for call in _response_tool_calls(row):
            rendered = _render_call(call.get("function", {}))
            if rendered in seen[run_id]:
                continue
            seen[run_id].add(rendered)
            name = call.get("function", {}).get("name") or ""
            if name:
                drawn.setdefault(run_id, []).append(name)
    return {run_id: ", ".join(names) for run_id, names in drawn.items()}


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

    A COLLECTOR run is a CYCLE, not a turn of the conversation (#2375): where the walk crosses
    into a collector run_id it emits a cycle-boundary marker (``_ACTOR["cycle"]``) carrying
    that run's short description (``_cycle_descriptions``), so an end-to-end sample's cycles
    render as steps of their own, after the turns that commissioned them.  The chat's own
    final reply is emitted just BEFORE the first cycle — it is the last chat step's closing
    word, never a tail of the job's cycles.  Only a sample that DROVE a conversation splits
    its cycles this way: a collector-only sample IS one cycle, and renders as the single
    no-user-turn step it always was.

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
    pending_reply = reply.strip()
    cycle_steps = _cycle_descriptions(rows) if pushed else {}
    open_cycle: str | None = None

    def emit(actor: str, content: str) -> None:
        if content and (actor, content) not in seen:
            seen.add((actor, content))
            turns.append((actor, content))

    for row in rows:
        run_id = row.run_id or ""
        if run_id in cycle_steps:
            if pending_reply:
                emit(_ACTOR["penny"], pending_reply)
                pending_reply = ""
            if run_id != open_cycle:
                open_cycle = run_id
                # A cycle marker bypasses the de-dup: two cycles may describe themselves alike
                # and both boundaries are real.
                turns.append((_ACTOR["cycle"], cycle_steps[run_id]))
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
                # A text turn is a DELIVERED reply only if the sample actually sent it.  With
                # nothing delivered there is nothing it could be, so it renders as an aside
                # rather than borrowing the reply's mark; with a delivered set, a draw outside
                # it is a re-roll and is collected by ``rejected_draws`` instead.
                if content.strip() in sent:
                    emit(_ACTOR["penny"], content)
                elif not sent:
                    emit(_ACTOR["aside"], content)
    emit(_ACTOR["penny"], pending_reply)
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
    """Does this tool-call row satisfy the anchor? A tool-name anchor (``memory_get(``)
    matches that call; a keyword anchor (``designer``, ``"published": false``) must live inside
    a ``collection_configure`` call — the row that made the edit — never another tool's reasoning
    field that merely mentions the word."""
    if needle.endswith("("):
        return needle in content
    return "collection_configure(" in content and needle in content


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


def sample_number(sample_index: int) -> int:
    """What a sample is CALLED — the number every surface a reader meets it on renders.

    One number per sample, defined here and read everywhere: the report's sample banner, the
    cohort observation's name, the thinking dump's header, and the artifact files the reader
    opens next.  It is 1-based because that is what a reader counts, and the sample INDEX
    stays 0-based because that is what a range produces.

    Declared as its own function because the two spellings cost a real diagnosis (#2076): the
    report called a sample ``-11`` while its evidence sat in ``-10.db``, so a reader following
    the report's own name landed on the NEXT sample's database — one that had run cleanly —
    and read a correct exclusion as a harness defect.  A name that resolves to the wrong file
    is worse than no name: every fact it hands back is about something else."""
    return sample_index + 1


def _sample_db_path(tmp_path, case_id: str, sample_index: int, attempt: int = 0) -> str:
    """Where a sample's hermetic DB lives.  When ``EVAL_REPORT_DIR`` is set the DB
    persists BESIDE the reports (the mounted dir survives the ``--rm`` container),
    so a run's raw promptlog can be re-read after the fact — same doctrine as the
    transcripts: the evidence always survives the run.  Unset → tmp_path as before.

    Named by :func:`sample_number`, so the file a report's sample name points at is that
    sample's own (#2076).  Its log lands beside it under the same stem.

    ``attempt`` keys every STAND-UP after the first onto its own file — a re-driven sample
    (#1803 review) or a re-booted one (#2230): nothing deletes a sample's DB, so a retry
    handed the same path would re-seed over the failed attempt's rows and inherit its
    ``messagelog``/``promptlog`` — the failed turn replayed into the next attempt's context
    and into its transcript — and a re-boot would open a database the failed boot had
    already built, where a first boot opens none.  The first attempt keeps the unsuffixed
    name, so a sample that never retries is unchanged."""
    report_dir = os.environ.get("EVAL_REPORT_DIR")
    base = Path(report_dir) if report_dir else tmp_path
    Path(base).mkdir(parents=True, exist_ok=True)
    suffix = f"-attempt{attempt + 1}" if attempt else ""
    return str(Path(base) / f"{case_id}-{sample_number(sample_index)}{suffix}.db")


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
    # The notify composer closes a collector CYCLE rather than a chat turn (#1911) — it draws
    # after the cycle's last action.
    PennyConstants.NOTIFY_COMPOSE_AGENT_NAME: MicroPlacement.RUN_CLOSE,
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
    turn becomes ``NUDGE`` so it renders inside its step, not as a new step; a cycle boundary
    becomes ``CYCLE`` so it OPENS a step, #2375)."""
    if actor == _ACTOR["cycle"]:
        return report.EventKind.CYCLE
    if actor == _ACTOR["user"]:
        return report.EventKind.NUDGE if _is_nudge(content) else report.EventKind.USER
    if actor == _ACTOR["call"]:
        return report.EventKind.CALL
    if actor == _ACTOR["tool"]:
        return report.EventKind.RESULT
    if actor == _ACTOR["aside"]:
        return report.EventKind.ASIDE
    return report.EventKind.REPLY


def _event_body(kind: report.EventKind, content: str) -> str:
    """The rendered body for an event (the glyph is prepended by the renderer): a reply is quoted,
    a nudge tagged ``*(nudge)*``, a call/result rendered verbatim."""
    if kind in (report.EventKind.REPLY, report.EventKind.ASIDE):
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
    it. A cycle boundary closes the turn in progress the same way (#2375) — the cycle that is
    ending is the one whose composer draw renders inside it. Anything still queued at the end
    renders there rather than vanishing (collapsed never means dropped, #1753). Returns the
    events and a ``turn index → event index`` map so a check placed on a turn resolves to its
    event."""
    events: list[report.Event] = []
    turn_to_event: dict[int, int] = {}
    for turn_index, (actor, content) in enumerate(turns):
        kind = _turn_kind(actor, content)
        if kind in (report.EventKind.USER, report.EventKind.CYCLE):
            _splice(micro_batches, MicroPlacement.RUN_CLOSE, events)
        action = kind in (
            report.EventKind.CALL,
            report.EventKind.REPLY,
            report.EventKind.ASIDE,
        )
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
    """Assign each check its ``Cn`` id (or ``Gn`` for a framework guard), in order."""
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
    run-close), rationale/cause, and baseline flip — in order."""
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


def _sample_banner(cost: PromptPerf, result: SampleResult) -> str:
    """The per-sample banner tail from what the sample COST and the verdict its case settled on.

    The ONE place a banner is made, and it is made at the flush rather than when the sample's
    database closes: a sample has no verdict then — its cohort's claims are answered at
    case close — so a banner built at drive time read ``✅ pass`` above a sample the claims
    failed or the pooler excluded (#2127).

    No per-sample RATE: the cohort is the unit of scoring, so a sample carries only what is true
    of it alone — whether it reached an answer, whether it got there shakily, what it cost."""
    return report.render_banner(
        passed=result.passed,
        cause=_cause_word(result.cause),
        fragile=result.fragile,
        aborted=result.aborted_on_model_error,
        duration_s=round(cost.duration_ms / 1000),
        calls=cost.calls,
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


# A transcript assembled before its case has settled a verdict carries NO banner — one is built
# at the flush, and rendering an unbannered transcript is refused rather than left blank (#2127).
_UNSETTLED_BANNER = None


def _build_transcript(
    result: SampleResult,
    turns: list[tuple[str, str]],
    main_rows: list[PromptLog],
    rows: list[PromptLog],
    baseline: Baseline | None,
    case_id: str,
    sample_index: int,
    delivered: Sequence[str] = (),
) -> report.SampleTranscript:
    """Assemble the ``report.SampleTranscript`` for one sample from its turns + scored result.

    Everything here is read while the sample's database is live EXCEPT the banner, which states
    a verdict the case may not have settled yet — that one line is built at the flush."""
    if not turns:
        return report.SampleTranscript(
            sample_number(sample_index),
            _UNSETTLED_BANNER,
            [],
            placeholder=report.NO_TURNS_PLACEHOLDER,
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
        number=sample_number(sample_index),
        banner=_UNSETTLED_BANNER,
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
    label = f"{report.SAMPLE_ROW} {sample_number(sample_index)}"
    _case_prompts.setdefault(case_id, []).extend((label, prompt) for prompt in prompts)


@dataclass
class _HeldSample:
    """One sample's block, held from the moment its database closes until its case's scores
    settle (#2127).

    The transcript is assembled while the sample's promptlog is still readable — that is the
    only moment it exists — but the block's BANNER states the sample's VERDICT, and a sample
    has none then: its cohort's claims are answered at case close.  So the block is
    RENDERED at the flush, its banner built there from the score the case settled on and the
    cost read while the database was live."""

    index: int
    transcript: report.SampleTranscript
    result: SampleResult
    cost: PromptPerf

    def render(self) -> str:
        """The block, bannered with the verdict its case settled on."""
        self.transcript.banner = _sample_banner(self.cost, self.result)
        return report.render_sample(self.transcript)


# Sample blocks, held per case and written in SAMPLE order when the case's scores settle.
# Samples may run concurrently (``EVAL_CONCURRENCY``), and appending as each one finishes
# would lay a case's ``<case_id>.md`` down in COMPLETION order — sample 4 above sample 2 —
# which is the one thing a report read top-to-bottom cannot do.  Buffering keeps the file
# byte-identical to a sequential run at any concurrency.  Emptied by the flush, so a run
# holds only the case in flight.
_sample_blocks: dict[str, list[_HeldSample]] = {}


def _record_sample_block(
    case_id: str,
    sample_index: int,
    transcript: report.SampleTranscript,
    result: SampleResult,
    cost: PromptPerf,
) -> None:
    """Hold one sample block until its case's scores settle."""
    _sample_blocks.setdefault(case_id, []).append(
        _HeldSample(sample_index, transcript, result, cost)
    )


def _flush_sample_blocks(case_id: str) -> None:
    """Render a case's held blocks and append them to ``EVAL_REPORT_DIR/<case_id>.md`` in sample
    order — called wherever the case's scores settle, which is the earliest moment a block can
    state a verdict.  It POPS, so the first flush a path reaches is the one that renders."""
    blocks = _sample_blocks.pop(case_id, [])
    report_dir = os.environ.get("EVAL_REPORT_DIR")
    if not report_dir or not blocks:
        return
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{case_id}.md").open("a") as handle:
        for held in sorted(blocks, key=lambda block: block.index):
            handle.write(held.render() + "\n\n")


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
    # Observe the reroll (same EVAL_REPORT_DIR gate as the artifact write) so the fragile flag
    # it settles rides into the artifact.
    result.observe_rerolled(rerolled=sample_is_fragile(db))
    delivered = _delivered_replies(db)
    turns = _sample_turns(main_rows, reply, driven, delivered)
    transcript = _build_transcript(
        result, turns, main_rows, rows, baseline, case_id, sample_index, delivered
    )
    _record_sample_block(case_id, sample_index, transcript, result, live_prompt_perf(db))


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


class _Drive(NamedTuple):
    """What driving one case's samples produced — the three ways a sample can end.

    ``results`` is the samples that RAN and were scored; ``voided`` is the named stand-ins
    for the samples the rig never got started (#2070); a sample whose model call failed on
    every attempt is in neither, exactly as it always was.  Three values rather than two
    because the report has to account for all of them: a case whose cohort listed only the
    samples that ran reported its losses as a smaller cohort instead of as losses.
    """

    results: list[SampleResult]
    perf: _Perf
    voided: list[eval_cohort.SampleObservation]


def _never_started(
    case_id: str, sample_index: int, error: Exception
) -> eval_cohort.SampleObservation:
    """The named void that stands in for a sample the rig never got started (#2070).

    Recorded, then voided — never swallowed.  The sample is gone either way; what this
    decides is whether the run can SEE that it is gone, and the surface that answers
    "can this run be believed" is where an unaccounted-for sample does the most damage:
    the pooled/excluded/driven counts have to add up, and the health block's cohort has
    to be recorded at all.

    The reason carries the exception CLASS and not its message, because the class is what
    GROUPS — two samples killed by the same thing must land in one bucket for the report's
    dominant-class line to name it, and a message carrying a URL or a timeout figure
    differs per sample.  The message is printed beside the sample id where it happens.
    """
    print(
        f"  ✗ {case_id} sample {sample_number(sample_index)}: voided before it could be "
        f"measured — {type(error).__name__}: {error}"
    )
    return eval_cohort.SampleObservation(
        name=f"{case_id}-{sample_number(sample_index)}",
        phrasing=NEVER_SPOKEN,
        complete=False,
        exclusion=SAMPLE_NEVER_STARTED.format(fault=type(error).__name__),
    )


def preflight_was_transient(error: PreflightError) -> bool:
    """Was every hard failure in this preflight a MOMENT rather than a verdict (#2230)?

    Read off the fault each failed check carries — the class the client put on the error it
    raised — and judged by ``LlmFault.transient``, the one definition the endpoint smoke check
    retries on too: a timeout, a dropped or reset connection, a 5xx, a rate limit.  A check
    that failed on a VERDICT carries no fault at all (a model the endpoint does not list), and
    a refused request carries a non-transient one (credentials refused, any other 4xx), so
    either voids the sample at once.  One verdict among transient failures is still a verdict:
    another boot would meet it again."""
    failures = error.report.failures
    return bool(failures) and all(
        failure.fault is not None and failure.fault.transient for failure in failures
    )


def _boot_retry_delay(retry: int) -> float:
    """How long to wait before the ``retry``-th re-boot (from 0): short, doubling, bounded by
    :data:`SAMPLE_BOOT_ATTEMPTS`."""
    return SAMPLE_BOOT_RETRY_SECONDS * 2**retry


class SampleOverranError(Exception):
    """One drive of a sample ran past :data:`SAMPLE_WALL_CLOCK_SECONDS` and was stopped."""


def _overran(case_id: str, sample_index: int) -> eval_cohort.SampleObservation:
    """The named void that stands in for a sample stopped at its wall-clock bound (#2168).

    The same kind of void a sample that never started is, because it is the same kind of
    loss: the rig, not the model, is why nothing was measured.  Its reason names the bound
    rather than a fault class — what ended the sample is the harness's own clock, and what it
    was stuck on is in its log beside its database."""
    exclusion = SAMPLE_NEVER_FINISHED.format(bound=SAMPLE_WALL_CLOCK_SECONDS)
    print(f"  ✗ {case_id} sample {sample_number(sample_index)}: voided — {exclusion}")
    return eval_cohort.SampleObservation(
        name=f"{case_id}-{sample_number(sample_index)}",
        phrasing=NEVER_FINISHED,
        complete=False,
        exclusion=exclusion,
    )


async def _within_wall_clock(drive: Coroutine[Any, Any, SampleResult]) -> SampleResult:
    """Run one drive of a sample under its wall-clock bound, and stop it for certain past it.

    The drive runs as a task of its own so the bound is a WAIT on it rather than a cancel
    scope around it: a scope expires by cancelling, and a cancel is exactly what a stuck
    sample has been seen to absorb.  :func:`stop_task` re-delivers it until the drive is
    down, which runs the drive's own teardown — its Penny, its channel — before the sample
    is voided.  A caller cancelled mid-wait stops the drive the same way on its way out."""
    task = asyncio.create_task(drive)
    try:
        await asyncio.wait({task}, timeout=SAMPLE_WALL_CLOCK_SECONDS)
    finally:
        overran = not task.done()
        await stop_task(task)
    if overran:
        raise SampleOverranError
    return task.result()


async def _run_samples(
    make_config: Callable[..., Config],
    tmp_path,
    *,
    case_id: str,
    samples: int,
    drive: SampleDriver,
    attempts: int = 1,
    model: str = "",
) -> _Drive:
    """Drive ``samples`` hermetic samples of one case; return what each of them produced.

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

    **Any OTHER failure voids that ONE sample and nothing else (#2070).**  Standing a
    sample up runs a preflight against the live endpoint and waits on a channel, so it can
    fail for reasons the model never touched — and the failure used to escape ``gather``,
    which took the other fourteen samples' work with it AND skipped ``record_cohort``, so
    the case died in a way the viability gate could not see.  It is classified into a
    named exclusion and recorded BEFORE the sample is voided: this is not a broad except
    that swallows, it is one that files.

    **Every drive is bounded (#2168).**  A sample that never RETURNS used to hold the whole
    ``gather`` with it — no cohort recorded, no result written, the run silent until killed by
    hand.  Each attempt runs under :data:`SAMPLE_WALL_CLOCK_SECONDS`; one that runs past it is
    stopped and voided by name like a sample that never started, so the case still closes.

    **A boot whose preflight failed TRANSIENTLY is stood up again (#2230)** — torn down, then
    booted from a fresh world exactly like a first boot, after a short backoff, up to
    :data:`SAMPLE_BOOT_ATTEMPTS` boots.  Only the last boot's failure voids the sample; a
    preflight failure that is a verdict (:func:`preflight_was_transient`) voids it at once.
    Every re-boot is counted on the case's cohort record, so the run's health block shows it.
    """
    limit = asyncio.Semaphore(EVAL_CONCURRENCY)
    perf = _Perf()
    retried_boots = 0

    async def _attempt(sample_index: int, stand_up: int, retryable: bool) -> SampleResult:
        """ONE stand-up of one sample: its own channel, its own database, its own Penny."""
        server = MockSignalServer()
        await server.start()
        try:
            config = _real_model_config(
                make_config,
                signal_api_url=f"http://localhost:{server.port}",
                db_path=_sample_db_path(tmp_path, case_id, sample_index, stand_up),
                model=model,
            )
            async with eval_penny(config, server) as penny:
                result = await drive(penny, server, sample_index, retryable)
                perf.add(live_prompt_perf(penny.db))
                run_health.record_budget_hits(live_budget_hits(penny.db))
                return result
        finally:
            await server.stop()

    async def _booted(sample_index: int, stand_ups: Iterator[int], retryable: bool) -> SampleResult:
        """One attempt at one sample, booted again while its preflight fails transiently.

        The failed boot's own teardown has already run by the time its error arrives here
        (its Penny stopped and shut down, its channel closed), and the next boot is a whole
        new stand-up — a new channel, a new database, a new Penny — so a re-booted sample is
        as hermetic as one that booted first time."""
        nonlocal retried_boots
        for retry in range(SAMPLE_BOOT_ATTEMPTS - 1):
            try:
                return await _attempt(sample_index, next(stand_ups), retryable)
            except PreflightError as error:
                if not preflight_was_transient(error):
                    raise
                retried_boots += 1
                print(
                    f"  ↻ {case_id} sample {sample_number(sample_index)}: its preflight failed "
                    f"on a transient endpoint fault — booting again "
                    f"({retry + 1} of {SAMPLE_BOOT_ATTEMPTS - 1}): {error}"
                )
                await asyncio.sleep(_boot_retry_delay(retry))
        return await _attempt(sample_index, next(stand_ups), retryable)

    async def _sample(sample_index: int) -> SampleResult | eval_cohort.SampleObservation | None:
        """This sample's scored result, the named void that replaces it, or ``None`` when
        its model call failed on every attempt."""
        async with limit:
            stand_ups = itertools.count()
            for attempt in range(attempts):
                try:
                    return await _within_wall_clock(
                        _booted(sample_index, stand_ups, attempt + 1 < attempts)
                    )
                except _ModelCallError:
                    print(
                        f"  ↻ {case_id} sample {sample_number(sample_index)}: "
                        f"the model call failed — "
                        f"retrying ({attempt + 1} of {attempts})"
                    )
                except SampleOverranError:
                    return _overran(case_id, sample_index)
                # BROAD ON PURPOSE, and it files rather than hides: an enumerated list here
                # would cover the preflight timeout that motivated it and miss the next
                # stand-up fault nobody has seen, which is the failure this whole path is
                # about.  The class is recorded on the sample's exclusion, the message is
                # printed beside it, and the cohort comes back one short — so a run can SEE it.
                except Exception as error:
                    return _never_started(case_id, sample_index, error)
        return None

    driven = await asyncio.gather(*(_sample(index) for index in range(samples)))
    results = [outcome for outcome in driven if isinstance(outcome, SampleResult)]
    voided = [outcome for outcome in driven if isinstance(outcome, eval_cohort.SampleObservation)]
    # What the case ASKED for beside what it got, recorded whatever happens next — this is
    # the only place both numbers exist, and a case that dies on its threshold must still
    # contribute its cohort to the run's health block.  ``excluded`` is counted beside
    # ``completed`` rather than folded into it (#2526): a sample that ran and was excluded
    # DID complete its drive — the dead-sample majority bar above stays about samples that
    # never produced a measured turn at all — but it answered no claim either, so the
    # viability verdict reads it off ``scored`` (``completed - excluded``), not ``completed``.
    excluded = sum(1 for result in results if result.excluded is not None)
    run_health.record_cohort(
        case_id,
        intended=samples,
        completed=len(results),
        excluded=excluded,
        retried_boots=retried_boots,
    )
    return _Drive(results=results, perf=perf, voided=voided)


# ── The cohort: one request, K phrasings, pooled (#1994/#1995) ───────────────
#
# A case reads as `<priors> / <trigger the action> / <assertions>`, and everything
# below is the MECHANISM behind the middle line: which arm a sample runs, what it left
# behind, whether its measured turn ran at all, and how the case's three sections are
# assembled once every sample is in.  What is asserted and what is measured stay with the
# CASE, so writing the next one is writing arms + fixtures + claims, never harness.

# The model a case is measured on — one cohort per id, so a case parametrized over it
# produces one score and one threshold set per model, which is what a per-model ceiling needs.
#
# WHICH models exist is the ROSTER's business (`roster.py`, #1999/#2222): it is the one
# configured list, it carries each model's pinned provider, and a remote run refuses to start
# unless it names at least one.  Which of them a RUN measures is resolved before pytest starts
# and arrives as `LLM_MODEL`, so this reads that rather than re-parsing `EVAL_MODELS` — the two
# would be the same variable meaning two different things.  A roster entry is therefore its own
# run, each self-describing in its own manifest.
EVAL_MODELS = [os.environ.get("LLM_MODEL", "")]

# Why a sample is not counted.  The one structural completeness condition every case shares:
# a sample's database exists from sample START, so a file is not evidence that anything ran.
NO_MEASURED_TURN = "the measured turn never ran — the sample carries only its seeded world"
NO_REPLY = "the measured turn produced no reply"
# A naive chat turn (#2455) opens directly with the chat agent — there is no state
# classifier ahead of it anymore, so there is nothing for it to run "undecided" on.  A
# chat sample whose own rows carry no chat-agent call at all is one where the TURN never
# reached its own agent — a micro-context (browse's extraction, say) ran and logged its
# own rows, but whatever came after never got as far as the chat agent's own draw — and
# the user was sent Penny's canned model-error reply.
#
# Before #2355 part A, this was ALSO how a chat call that raised at the endpoint showed
# up: the client wrote a row only on a usable response, so a raised call wrote none and
# "no chat-agent row" was the only signal of it.  That is no longer true — every call now
# writes its one row however it ends (``CallOutcome``), so a chat agent that raised is
# NOT this case: it carries a chat-agent row with ``call_outcome="raised"``, and
# ``_chat_drew`` reads it as drawn.  A provider-side refusal (429/5xx) in that row is
# caught separately, by ``_provider_refused``/``PROVIDER_REFUSED`` below; any OTHER
# raised shape (a connection error, a model-not-found, Penny's own malformed/oversized
# request) is left to be scored — the reply IS Penny's canned apology, and that is
# exactly what she did.
NO_CHAT_DRAW = "the chat model's call failed at the endpoint, so the reply is Penny's canned error"
NO_DRAW = "the draw never returned a usable answer — it failed whole after its rerolls"
NO_CYCLE = "the dispatcher refused a cycle, so it never ran against the world the case built"
# A call the PROVIDER refused outright — rate limited (429) or a server error (5xx) —
# after the client's own retries ran out (#2526).  Keyed to the two recorded fields the
# client classifies the failure by (``call_outcome``/``call_error_fault``), never to the
# delivered text: the canned apology IS a reply, so nothing about its wording tells a
# provider's refusal apart from the model genuinely declining.  Deliberately narrow — a
# 4xx (Penny's own malformed or oversized request), a connection error, a timeout, a
# tool-parse failure and a "no choices" 200 are all real, scorable behaviour, not an
# infrastructure fault, however much they resemble one.
PROVIDER_REFUSED = "the model call was refused by the provider (rate limited or a server error)"
# The sample the rig never got STARTED (#2070).  Its siblings above are read off a live
# database, because the sample ran and left one; this one is written by ``_run_samples``
# when standing the world up raised — a preflight against an unreachable endpoint, a
# readiness wait running out, a port that would not bind, a seed that failed.  It is a
# HARNESS fault and never a provider one: the catch site covers every one of those causes,
# so naming it after the endpoint would key it to the failure that happened to motivate it.
# The run's provider tally stays a read of CHAT ATTEMPTS at one chokepoint, and a synthetic
# entry there would either double-count a call the client already logged or add a fault
# with no attempt behind it; what run health sees of this is structural instead — a cohort
# short of its intended samples, which is what the viability verdict rules on.
SAMPLE_NEVER_STARTED = "the sample never started — {fault} while standing its world up"

# The arm a sample that never started ran: none.  A real label rather than an empty string,
# because it renders beside the sample's id wherever a wording would.
NEVER_SPOKEN = "never started"

# The sample the harness STOPPED at its wall-clock bound (#2168) — the never-started sample's
# sibling, voided the same way: it may have spoken, but nothing of it came back to be scored,
# and the reason says so by naming the bound rather than a fault nobody raised.  Its label
# stands where a wording would, because the arm it ran is the driver's to know, not this one's.
SAMPLE_NEVER_FINISHED = "the sample never finished — stopped at its {bound:g}s wall-clock bound"
NEVER_FINISHED = "never finished"

# WHICH turns count as world for a provenance claim, and which part of each was stated rather
# than wrapped around it, is ``utils/given.py``'s — the roles, the frame a result is narrated
# in, the entry render, the timestamps.  What stays HERE is the one thing only a fixture
# knows: which rows are this sample's own.


# How a chat case reads one sample: its live database, the reply it produced, the
# collections that existed before it, and the entries the store held before it.
Observer = Callable[
    [Database, str, set[str], list[eval_cohort.StoredEntry]],
    eval_cohort.SampleObservation,
]


def measured_turn_ran(db: Database) -> bool:
    """Whether THIS sample's own turn reached the model at all.

    ``live_prompts`` already excludes every seeded prior turn, so an empty window is the dead
    sample exactly: the seeded round is present and nothing was added to it.  Read
    structurally rather than inferred from a score, because a dead sample scores like a
    behavioural failure and would otherwise be pooled as wild variance."""
    return bool(live_prompts(db))


def given_to_the_model(db: Database) -> eval_cohort.Given:
    """Everything this sample's turn was GIVEN — the world a provenance claim reads against
    (#1994), split into what was stated and what the framework wrapped it in (#2203).

    THE ONE reader, for a chat turn and for a single-call micro-context draw alike, and it
    reads the PROMPTLOG rather than rebuilding the input at the call site — so the haystack is
    what the model was handed and not a second copy of part of it.

    A haystack assembled from a runner's own arguments omits the contract the draw was framed
    by, and the words that contract taught it then read as inventions (#2078).  Measured: a
    partly-present extraction answered a line per story ending ``NOT_PRESENT``, the tag's tail
    glued across the line break to the next story's first word — the name phrase
    ``PRESENT City``, whose ``PRESENT`` the page does not carry — and ``nothing invented``
    failed on 2 of 15 CORRECT draws.  ``NOT_PRESENT:`` is in the extraction contract, so
    reading what was really given passes it by construction, and passes the classifier's state
    names, the binder's outcome names and every contract token nobody has written yet with it —
    there is no list of tags to keep anywhere.

    The contract stays in the text for exactly that reason, and is scaffolding for the reason
    #2203 measured: a NAME is sourced by anything the round was handed, a NUMBER only by what
    was stated to it.  ``read_given`` draws that line where the turns still have their roles."""
    return read_given(_iter_prompt_messages(db))


def reply_embedding(db: Database, reply: str) -> list[float] | None:
    """The vector of the reply this sample was scored on.

    Every conversational send is embedded at egress, so the cohort's reply spread costs no
    model call — it is a read of a column production already fills.  Looked up by the reply's
    own text rather than by a recency window, so it is the SCORED reply's vector."""
    row = db.messages.find_outgoing_by_content(reply)
    if row is None or row.embedding is None:
        return None
    return deserialize_embedding(row.embedding)


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


def _held_entries(db: Database) -> list[eval_cohort.StoredEntry]:
    """Every entry the store HOLDS at the end of the sample, whoever wrote it.

    The counterpart to ``_stored_entries``, which reads only what this round WROTE — and
    the same ``StoredEntry`` shape, because it is the same three facts about the same rows.
    A claim about what a round left ALONE cannot be answered from a list of its writes:
    there, an entry it never touched and an entry it deleted are both simply absent.
    Collections only, for the same reason ``_stored_entries`` is: a log is keyless and
    append-only, so it holds nothing a round could have left in a different state."""
    held: list[eval_cohort.StoredEntry] = []
    for row in db.memories.list_all():
        if row.type != MemoryType.COLLECTION:
            continue
        memory = db.memory(row.name)
        entries = memory.read_all() if memory is not None else []
        held += [
            eval_cohort.StoredEntry(collection=row.name, key=entry.key, content=entry.content)
            for entry in entries
        ]
    return held


def stored_images(db: Database) -> list[eval_cohort.StoredImage]:
    """Every image the media store holds, oldest first — its description and its source page.

    Read off the ``media`` table directly: the store has no list read of its own, because
    production only ever selects one image to attach.  Ordered by ``created_at``, the store's
    own clock, so a turn that made two pictures reads them in the order it made them."""
    with Session(db.engine) as session:
        rows = session.exec(select(Media).order_by(col(Media.created_at).asc())).all()
    return [
        eval_cohort.StoredImage(description=row.title or "", source_url=row.source_url)
        for row in rows
    ]


def _mechanism_records(db: Database, before: set[str]) -> list[eval_cohort.MechanismRecord]:
    """Every mechanism the registry holds when the sample ends, ARCHIVED ONES INCLUDED.

    Archived rows are the point: a bail retires the container its round built by archiving it,
    and a reader that dropped retired rows would report the one thing the cleanup did as the
    container having vanished.

    ``touched_this_run`` is read off the mutation ledger — a live turn's mutation cites a live
    run and every event a seeded world wrote cites a seeded one — so it is a read rather than a
    diff against a remembered before-state, and it names what the STORE says it changed rather
    than any field this file enumerates.

    ``moved_this_run`` is the same ledger read against the row NOW: the fields whose value
    differs from what the sample's first edit found there, so a value a call merely restated
    is not reported as a change the row carries.

    The three configuration values beside it are the row's own, copied verbatim: the ledger says
    what moved and never where it landed, so a claim naming a value has to read the row.

    ``expires`` beside them is whether the row carries an end condition at all — a TERM a turn
    that stood a job up committed to, read as a named column for the same reason."""
    return [
        eval_cohort.MechanismRecord(
            name=row.name,
            archived=row.archived,
            notifies=row.notify,
            schedule=row.schedule,
            program=row.extraction_prompt,
            born_this_run=row.name not in before,
            touched_this_run=_touched_this_run(_live_events(db, row.name)),
            moved_this_run=_moved_this_run(_live_events(db, row.name), row),
            expires=row.expires_at is not None,
            expires_at=row.expires_at,
            max_runs=row.max_runs,
            created_at=row.created_at,
        )
        for row in db.memories.list_all()
        if row.type == MemoryType.COLLECTION
    ]


def _live_events(db: Database, name: str) -> list[MutationEvent]:
    """This sample's own mutation events on one registry row, newest first.

    Seeded events are excluded the way every other "what did THIS sample do" reader excludes
    them — by the run id — so a world's own history never reads as the turn's work."""
    return [
        event
        for event in db.mutations.history(
            name, MUTATION_HISTORY_WINDOW, entity_type=MutationEntityType.COLLECTION
        )
        if not is_seeded_run(event.run_id)
    ]


def _touched_this_run(events: list[MutationEvent]) -> list[str]:
    """What this sample's own runs did to one registry row, in the store's own vocabulary.

    An UPDATE names the fields it reported changing; every other action names ITSELF, because
    the store deliberately names those by action rather than as a field edit — archiving carries
    no changed field of its own, and a creation is not an edit to anything.  So a row this turn
    archived reads ``['archived']`` and a row whose notify it flipped reads ``['notify']``."""
    touched: list[str] = []
    for event in events:
        detail = mutation_detail(event)
        moved = detail.changed_fields if detail is not None else []
        touched += moved or [event.action]
    return sorted(set(touched))


def _moved_this_run(events: list[MutationEvent], row: MemoryRow) -> list[str]:
    """Every field whose value the row holds NOW differs from what it held before this sample's
    first edit to it — the end state, never the calls.

    Each event records the before-value of every field it moved (#1946), so the OLDEST prior per
    field is what the sample found there, and ``snapshot_fields`` renders the row now in the
    same form a prior was recorded in.  Keyed to whichever fields the ledger names, so a field
    the store learns to change tomorrow is compared for free.  An event carrying no prior at all
    — a creation — names its own action, since there was no row for it to have changed, and a
    field an update names without recording its prior counts as moved, since nothing says it
    came back to where it was."""
    found: dict[str, str | None] = {}
    moved: set[str] = set()
    for event in reversed(events):
        detail = mutation_detail(event)
        priors = detail.priors if detail is not None else []
        if not priors:
            moved.add(event.action)
        for prior in priors:
            found.setdefault(prior.field, prior.value)
        named = detail.changed_fields if detail is not None else []
        moved |= set(named) - {prior.field for prior in priors}
    now = snapshot_fields(row)
    moved |= {field for field, value in found.items() if now.get(field) != value}
    return sorted(moved)


def _written_by_a_live_run(entry) -> bool:
    """Whether THIS sample put an entry's current value there — created by a live run, or last
    rewritten by one.  Both stamps, because an edit of a seeded entry moves only the second."""
    stamps = (entry.created_by_run_id, entry.last_written_by_run_id)
    return any(stamp is not None and not is_seeded_run(stamp) for stamp in stamps)


def _framed_container(db: Database) -> str | None:
    """The container a round was framed on — always ``None`` on this branch (#2455 naive
    baseline): there is no conversation machine and no round framing, so nothing ever
    stamps ``state_transition.skill_frame``.  Kept as a function (not inlined at the call
    site) so the field it feeds stays named the same on ``SampleObservation``."""
    return None


def _scheduled_by_this_round(db: Database, before: set[str]) -> list[str]:
    """Collections this round created that carry a schedule or a notify flag.

    Scored against what the turn PRODUCED rather than the whole store: a seeded collection's own
    cadence predates the round, and failing on it would report the fixtures as her doing."""
    return sorted(
        row.name
        for row in db.memories.list_all()
        if row.name not in before and (row.schedule is not None or row.notify)
    )


def _chat_tool_sequence(db: Database) -> list[str]:
    """Every call this sample's CHAT turns made, in emission order.

    Keyed to NO name list.  Which tool names a turn issued is a structural fact in the prompt
    log; which of them mean the behaviour happened is the CASE's claim, made against the world
    the turn left.  A membership test here is the #1783 shape: it answers for the names someone
    enumerated and reads ``no call`` for every other shape — an archive, a dispatch, a mute, a
    log read, and every tool a plugin contributes — whatever the turn actually did.

    Names arrive already normalised: ``chat_run_tool_sequences`` reads each one through
    ``tool_call_name``, which is production's own ``strip_harmony_control_tokens``, so a leaked
    Harmony control token (`collection_write<|channel|>commentary`, seen on two samples of one
    run) cannot hide a call whose entries are in the store."""
    return [tool for run in chat_run_tool_sequences(db) for tool in run]


def _machine_walk(db: Database) -> str:
    """The machine's walk this sample, oldest move first (#2538 naive baseline): there
    is no conversation machine any more, so no sample ever moves one — ``db`` stays the
    parameter every caller already passes, so a future walk-shaped reader needs no
    signature change to read something real again."""
    del db
    return "no move"


def _last_scripted_turn_at(db: Database) -> datetime | None:
    """The moment this sample's own script turned — the last message ``TEST_SENDER`` sent,
    on the pinned clock every other read in the sample shares (``MessageStore.log_message``
    stamps ``timestamp`` from the store's own ``Clock``, the same one a job's schedule and
    ``expires_at`` resolve against).

    The naive branch's stand-in for ``db.machine.latest_transition().created_at`` (#2455):
    there is no conversation machine any more, so no transition row ever lands, and reading
    one here answered ``None`` on every sample — the end-claim's rationale is literally "the
    sample recorded no turn to read the ask's end against," on every sample, pass or fail.
    ``get_user_messages`` is oldest-first and a seeded prior conversation (``seed_known_
    routine``) logs its own ask BEFORE any live turn runs, so the most recent entry is always
    this sample's own last scripted turn, never the seed's."""
    messages = db.messages.get_user_messages(TEST_SENDER, limit=1)
    return messages[-1].timestamp if messages else None


def _observe_sample(
    db: Database,
    *,
    name: str,
    phrasing: str,
    arm: int,
    reply: str,
    before: set[str],
    held_before: list[eval_cohort.StoredEntry],
) -> eval_cohort.SampleObservation:
    """Read everything one CHAT sample left behind, while its database is still live.

    Gated for completeness FIRST, so a dead sample carries no facts to be pooled by accident.

    Every read below is a CHAT read — the machine's walk, the registry, the collections, the
    outgoing message the reply was embedded on, and a tool sequence filtered to the chat
    agent's own rows.  A fixture driving another shape writes its OWN observer rather than
    calling this one: reused elsewhere it returns a row that is structurally fine and
    substantively empty (#2017).
    """
    exclusion = _exclusion(db, reply)
    if exclusion is not None:
        return eval_cohort.SampleObservation(
            name=name, phrasing=phrasing, arm=arm, complete=False, exclusion=exclusion
        )
    return eval_cohort.SampleObservation(
        name=name,
        phrasing=phrasing,
        arm=arm,
        turn_at=_last_scripted_turn_at(db),
        timezone=user_timezone_name(db),
        walk=_machine_walk(db),
        entries=_stored_entries(db),
        held=_held_entries(db),
        held_before=held_before,
        mechanisms=_mechanism_records(db, before),
        muted=db.users.is_muted(TEST_SENDER),
        images=stored_images(db),
        tool_sequence=_chat_tool_sequence(db),
        reply=reply,
        reply_embedding=reply_embedding(db, reply),
        aborted_on_model_error=_aborted_on_model_error(reply),
        given=given_to_the_model(db),
        container=_framed_container(db),
        scheduled=_scheduled_by_this_round(db, before),
    )


def _aborted_on_model_error(reply: str) -> bool:
    """Whether the delivered reply IS the framework's own model-error apology (#2252) — read by
    IDENTITY against the constant the channel sends verbatim on an aborted chat turn
    (``PennyResponse.AGENT_MODEL_ERROR``), never a copied string or a substring match.  A chat
    turn's abort is otherwise unrecorded (no ``run_outcome``, and ``ControllerResponse.abort`` is
    read only by the collector), so the delivered text is the one place left to read it: the
    channel does ``response.answer.strip()`` with nothing else, so the sent reply is byte-equal
    to the constant exactly when the turn ended there rather than on a reply the model composed.
    A real reply that merely QUOTES the apology's words is a reply, not an abort."""
    return reply == PennyResponse.AGENT_MODEL_ERROR


# The completeness gate is decided per FIXTURE, below.  ``measured_turn_ran`` is the half every
# shape shares — a sample's database exists from sample START, so a file is not a result — and
# each gate adds whatever "this shape produced nothing" means for the shape it drives.
def _exclusion(db: Database, reply: str) -> str | None:
    """Why a CHAT sample cannot be counted, or ``None`` when it can.

    Chat's own second condition is an empty reply: the turn is a conversation and a turn that
    said nothing produced no behaviour to read.  It is stated HERE rather than shared, because
    it is false of every other shape — a micro-context sample has no reply at all, and a gate
    that voided one for that would void the entire cohort and refuse the run (#2017).

    A naive chat turn (#2455) opens directly with the chat agent, so a sample whose turn ran
    but whose own rows carry no chat-agent draw never reached its own agent at all (see
    ``NO_CHAT_DRAW``'s own comment for what that is, now that #2355 logs a row on every
    outcome).  A chat-agent call the PROVIDER refused outright (``PROVIDER_REFUSED``) is
    checked separately and BEFORE the reply is read, since that call's row is there and the
    delivered reply is Penny's canned apology — a real, non-empty reply that would otherwise
    read as the model's own failure to answer.
    """
    if not measured_turn_ran(db):
        return NO_MEASURED_TURN
    if not _chat_drew(db):
        return NO_CHAT_DRAW
    if _provider_refused(db):
        return PROVIDER_REFUSED
    if not reply.strip():
        return NO_REPLY
    return None


def _chat_drew(db: Database) -> bool:
    """Whether the measured turn's chat agent logged a draw of its own — read off every
    sample row directly, since a naive chat turn (#2455) opens with the chat agent and has
    no earlier classifier row to read a "since" boundary from."""
    return any(row.agent_name == PennyConstants.CHAT_AGENT_NAME for row in _sample_prompt_rows(db))


def _provider_refused(db: Database) -> bool:
    """Whether this sample's measured window carries a call the provider refused outright
    (#2526) — read off the two fields the client classifies the failure by, never off any
    delivered text.

    ``call_outcome == "raised"`` is every call that never got a usable answer (a connection
    error, a model-not-found, Penny's own malformed/oversized request, a provider's refusal —
    see ``CallOutcome``); ``call_error_fault`` narrows that to the two shapes that are the
    PROVIDER's own doing rather than Penny's (``fault_for_status`` in ``penny.llm.models`` —
    429 is ``RATE_LIMITED``, 5xx is ``SERVER_ERROR``).  Checked over every row in the sample's
    own window, main agent or micro-context alike, because any one of them poisons the
    sample's measured turn the same way."""
    refusing_faults = {LlmFault.RATE_LIMITED.value, LlmFault.SERVER_ERROR.value}
    return any(
        row.call_outcome == CallOutcome.RAISED and row.call_error_fault in refusing_faults
        for row in _sample_prompt_rows(db)
    )


def _draw_exclusion(db: Database, answer: str) -> str | None:
    """Why a single-call sample cannot be counted, or ``None`` when it can.

    A micro-context leaves no reply and writes no store, so the only thing that makes it
    countable is that the draw reached the model and came back with a usable answer.  The
    runner passes the parsed result's own marker as ``answer`` — empty exactly when the draw
    failed whole after its rerolls."""
    if not measured_turn_ran(db):
        return NO_MEASURED_TURN
    if not answer.strip():
        return NO_DRAW
    return None


def _reporting_model() -> str:
    """Which model produced this run's numbers, off the run manifest — empty only off-report,
    where nothing is recorded anyway.  A ceiling recorded without it is unusable, since two
    models differ several-fold on the same feature."""
    run = eval_artifacts.active_run()
    return run.manifest.model if run is not None else ""


def _driven_cohort(
    case_id: str,
    model: str,
    results: Sequence[SampleResult],
    voided: Sequence[eval_cohort.SampleObservation],
    arms: Sequence[eval_cohort.Arm],
) -> Cohort:
    """The cohort a case makes its claims against — every sample the case drove.

    Both halves, always: the samples that produced an observation, then the named void for
    each sample that never started (#2070).  A cohort listing only what ran reports a lost
    sample as a smaller cohort rather than as a loss, and the report's harness accounting
    (``pooled + excluded = driven``) is the one surface that says whether a run can be
    believed — a sample it cannot account for is how infrastructure failure gets read as
    behaviour.  The voids come LAST because they wrote no transcript block, so the samples
    that did keep the positions the assembler expands them by.
    """
    return Cohort(
        case_id=case_id,
        model=model or _reporting_model(),
        samples=[r.observation for r in results if r.observation is not None] + list(voided),
        arms=arms,
    )


def _phrasing_label(phrasings: Sequence[str], sample_index: int, per_phrasing: int) -> str:
    """What this sample is called in the report — which of the case's wordings it ran."""
    if len(phrasings) > 1:
        return f"phrasing {sample_index // per_phrasing + 1}"
    return "the ask"


class _Arms(NamedTuple):
    """A cohort's ARMS and how many samples each one runs (#1994/#2017).

    An arm is one input the behaviour is answered from, together with the WORLD it is answered
    against.  For chat that is a wording of the user's ask, and every arm shares one world.
    For a browse extraction it is a wording of the ``extract`` instruction — Penny's own words,
    written upstream at the call site — again against one page.  For a collector there is no
    user turn: what varies is the ``extract`` instruction its rendered program carries and the
    prose of the page that answers it, so an arm is a wording AND its own page, and the arms do
    not share a world.

    Lifted out of ``chat_eval``'s closure because nothing about it is chat: it maps a sample
    index to the arm that sample runs and nothing more.  The N is deliberately NOT scaled by
    anything ambient — a recorded ceiling is ``(feature, model, N, value)`` and normalised
    entropy is biased upward at small N, so an N that drifted with an environment variable
    would silently make every recorded threshold incomparable."""

    arms: list[eval_cohort.Arm]
    per_phrasing: int

    @property
    def spoken(self) -> list[str]:
        """The input each sample runs, by sample index."""
        return [arm.text for arm in self.arms for _ in range(self.per_phrasing)]

    @property
    def driven(self) -> int:
        return len(self.arms) * self.per_phrasing

    def index_of(self, sample_index: int) -> int:
        """Which ARM this sample runs, by index — the hard link a claim's world is read on."""
        return sample_index // self.per_phrasing

    def at(self, sample_index: int) -> eval_cohort.Arm:
        """The arm this sample runs — its input and the ground it is answered against."""
        return self.arms[self.index_of(sample_index)]

    def label(self, sample_index: int) -> str:
        return self.at(sample_index).label


def _arms(inputs: Sequence[str], worlds: Sequence[World], per_phrasing: int, samples: int) -> _Arms:
    """The arms a case drives, each sampled ``per_phrasing`` times.

    ``worlds`` is either one world for every arm (the chat and micro-context shape, where the
    ground is a property of the CASE) or one per arm (a collector, where the ground moves with
    the inputs)."""
    each = per_phrasing or samples
    ground = (
        list(worlds) if len(worlds) > 1 else [(worlds[0] if worlds else _NO_WORLD) for _ in inputs]
    )
    return _Arms(
        arms=[
            eval_cohort.Arm(
                label=_phrasing_label(list(inputs), index * each, each),
                text=text,
                world=ground[index],
            )
            for index, text in enumerate(inputs)
        ],
        per_phrasing=each,
    )


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
    behaviour: str = ""
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
        """Deal the claims back out to their samples, then write the document and the record."""
        self._grade()
        # The claims have settled every sample's score, so the blocks can state a verdict — and
        # they go down BEFORE the document's head is written over them (#2127).
        _flush_sample_blocks(self.case_id)
        observations = self._observations()
        # Computed ONCE and used by both halves: the document renders the standings, and the
        # ASSEMBLER needs to know which sample to expand in the posted comment.  Deriving them
        # twice would let the map and the expanded sample disagree about which one is modal.
        standings = eval_cohort.standings(observations, self.driven_cohort.features)
        _record_case_report(
            self.driven_cohort, observations, standings, self.perf, self.driven, self.behaviour
        )
        self._record(observations, standings)

    def _record(
        self,
        observations: Sequence[eval_cohort.SampleObservation],
        standings: Sequence[eval_cohort.SampleStanding],
    ) -> None:
        """Write the case's ``results.jsonl`` record, print its perf and RESULT lines, and
        refuse a cohort that mostly never ran."""
        eval_artifacts.record_case(
            case_id=self.case_id,
            family=self.family,
            module=self.module,
            results=self.results,
            perf=self.perf,
            expand_samples=_expandable(standings),
            standing_counts=Counter(standing.standing.value for standing in standings),
            variance=_variance_readings(observations, self.driven_cohort.features),
            shares=_share_records(observations, self.driven_cohort.shares),
            behaviour=self.behaviour,
            representative=_representative_sample(self.driven_cohort, observations, standings),
        )
        self.perf.report(self.case_id, self.driven)
        _report_result(self.case_id, self.results, intended=self.intended)

    def _grade(self) -> None:
        """Every claim the cohort answered, redistributed to the sample that answered it — and
        every sample the cohort refused to pool, recorded as the loss it is.

        The one seam where cohort-level claims meet per-sample grading, and therefore the one
        place a sample's score, its failure cause and its exclusion are settled — so the
        case document and ``results.jsonl`` render one scored object rather than two (#2125)."""
        by_sample = _cohort_checks(self.driven_cohort)
        for result in self.results:
            observation = result.observation
            if observation is None:
                continue
            if observation.complete:
                result.adopt(by_sample.get(observation.name, []))
            else:
                result.exclude(eval_cohort.exclusion_reason(observation))

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
            blind=feature.blind,
        )
        for feature in pooled.features
    ]


def _share_records(
    samples: Sequence[eval_cohort.SampleObservation], shares: Sequence[eval_cohort.Share]
) -> list[eval_artifacts.ShareRecord]:
    """What the case's measured SHARES read, in the record: one value per sample the case
    drove, in the order it drove them.  Read through the same pool the document renders, so a
    sample the pool excluded carries no value here either."""
    records = []
    for spread in eval_cohort.pool(samples, [], shares).shares:
        read = {reading.sample: reading.value for reading in spread.readings}
        records.append(
            eval_artifacts.ShareRecord(
                name=spread.name,
                open_above=spread.open_above,
                blind=spread.blind,
                values=[read.get(sample.name) for sample in samples],
            )
        )
    return records


def _expandable(standings: Sequence[eval_cohort.SampleStanding]) -> list[int]:
    """The 1-based positions of the samples the posted comment carries in full.

    Only the representative: an outlier is communicated by its DIVERGENCE, which is a few rows,
    and a typical sample by the fact that it agreed.  Every sample stays whole in the artifact."""
    return [index + 1 for index, s in enumerate(standings) if s.worth_opening]


def _sample_wording(cohort: Cohort, observation: eval_cohort.SampleObservation) -> str:
    """The text this sample was driven with — the arm's own wording, verbatim.

    Empty for a cohort with no arms (a case that drove no cohort at all), or for an
    observation whose arm index the cohort cannot resolve — never a guess."""
    if not cohort.arms or not 0 <= observation.arm < len(cohort.arms):
        return ""
    return cohort.arms[observation.arm].text


def _representative_sample(
    cohort: Cohort,
    observations: Sequence[eval_cohort.SampleObservation],
    standings: Sequence[eval_cohort.SampleStanding],
) -> eval_artifacts.RepresentativeSample | None:
    """The one sample the case's own document sends a reader to open — the same sample
    ``cohort.standings`` marks MODAL (#2256), carried onto the record so the run summary's
    coverage table can name it without a cohort to re-derive it from.

    ``standings`` is aligned with ``observations`` by construction (both built from the same
    ``samples`` sequence, in the same order), so the two are zipped rather than matched by name.
    ``None`` for a cohort with nothing pooled — every sample dead — which has no representative
    to name."""
    for observation, standing in zip(observations, standings, strict=True):
        if standing.standing is eval_cohort.Standing.MODAL:
            return eval_artifacts.RepresentativeSample(
                name=observation.name,
                wording=_sample_wording(cohort, observation),
                reply=observation.representative_reply,
            )
    return None


# The one sentence a case states about itself, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  REQUIRED: a case without one is a case whose contract nobody wrote down,
# and the report would render a rate with nothing saying what was being asked.
_NO_BEHAVIOUR = (
    '{case_id}: a case must state the behaviour it checks — behaviour="In <the locus>, '
    'when <X>, Penny <does Y>.", naming the shipped agent as the locus'
)


def _stated_behaviour(case_id: str, behaviour: str) -> str:
    """The case's own sentence, or a refusal naming what is missing."""
    if not behaviour.strip():
        raise ValueError(_NO_BEHAVIOUR.format(case_id=case_id))
    return behaviour.strip()


class _Unstated:
    """A parameter nobody passed.

    ``behaviour`` can use the empty string for this because a case always has something to
    say; ``min_pass_rate`` cannot, because ``None`` is the VALUE every case states."""

    def __repr__(self) -> str:
        return "<unstated>"


UNSTATED = _Unstated()
# What a driver takes for ``min_pass_rate``: the report-only ``None``, or nothing at all.  A
# number is admitted by the type so that passing one is refused by name rather than by a
# type checker nobody reads at run time.
PassRate = float | None | _Unstated

# A case's claims are counted and reported, never gated (docs/eval-case-design.md §8), and
# every case says so in as many words: ``min_pass_rate=None``.  Stated rather than defaulted,
# so the report-only setting is a line a reviewer reads in the case.
_NO_PASS_RATE = (
    "{case_id}: a case must state min_pass_rate=None — its claims are counted and reported, "
    "never gated"
)
_PASS_RATE_FLOOR = (
    "{case_id}: min_pass_rate={given!r} — an assertion carries no floor, so there is nothing "
    "to compare it with; state min_pass_rate=None"
)


def _require_report_only(case_id: str, min_pass_rate: PassRate) -> None:
    """Refuse a case that did not state the report-only setting, or stated a floor."""
    if isinstance(min_pass_rate, _Unstated):
        raise ValueError(_NO_PASS_RATE.format(case_id=case_id))
    if min_pass_rate is not None:
        raise ValueError(_PASS_RATE_FLOOR.format(case_id=case_id, given=min_pass_rate))


# What every driver refuses a case for omitting: the input its cohort's arms are built from.
# A case is ONE ask in its wordings, sampled and pooled, and its body makes claims against the
# cohort that comes back — a driver has no other way to score one.
_NO_ASK = (
    "{case_id}: {driver} needs {what} — a case drives one ask in its wordings and makes its "
    "claims against the cohort that comes back (docs/eval-case-design.md §4)"
)


def _require_ask(case_id: str, driver: str, what: str, given: Sequence[object] | None) -> None:
    """Refuse a case that names nothing to drive, before any sample runs."""
    if not given or (isinstance(given, str) and not given.strip()):
        raise ValueError(_NO_ASK.format(case_id=case_id, driver=driver, what=what))


def _declare_case(
    request: pytest.FixtureRequest,
    *,
    case_id: str,
    layer: Layer,
    behaviour: str,
    area: Area | None,
    wording: str,
) -> eval_catalogue.CatalogueEntry:
    """Refuse a case that states no behaviour or declares no area, before any sample runs —
    and return the case as the catalogue lists it.

    In catalogue mode the case is recorded and the test stops here, so collecting the suite
    stands up no sample, endpoint check or database of its own."""
    entry = eval_catalogue.declare(
        case_id=case_id,
        layer=layer,
        sentence=_stated_behaviour(case_id, behaviour),
        area=area,
        wording=wording,
        node_id=request.node.nodeid.partition(_PARAMETER_SUFFIX)[0],
    )
    destination = eval_catalogue.catalogue_path()
    if destination is not None:
        eval_catalogue.record(destination, entry)
        pytest.skip(eval_catalogue.CATALOGUED)
    return entry


# How an ask of several turns is written on one line, for an arm's text and a case's first
# wording alike.
_TURN_SEPARATOR = " / "

# Where a test's node id starts naming its parameters — the model, which varies by run and is
# no part of which test a case lives in.
_PARAMETER_SUFFIX = "["

_RAN_TO_ITS_END = (
    "{node_id}: ran to its end in catalogue mode — an eval test hands its case to a driver, "
    "which records it and stops"
)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, object, object]:
    """In catalogue mode, fail an eval test that finished without reaching a driver.

    Every case is recorded and skipped at its driver, so a test body that completes drove
    nothing through one: it is missing from the catalogue, and it ran whatever it does.  The
    mode is read before the body runs, since it is the run's and not the test's."""
    collecting = eval_catalogue.catalogue_path() is not None
    result = yield
    if collecting:
        pytest.fail(_RAN_TO_ITS_END.format(node_id=item.nodeid))
    return result


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


def _run_facts() -> report.RunFacts:
    """Where this run came from, for the case's own table — empty off-report, where there is no
    run to describe and the provenance rows are omitted rather than rendered blank."""
    run = eval_artifacts.active_run()
    if run is None:
        return report.RunFacts()
    manifest = run.manifest
    return report.RunFacts(
        commit=manifest.commit[:8],
        provider=manifest.provider or "",
        embeddings=manifest.embedding_model,
        run_id=manifest.run_id,
    )


def _record_case_report(
    cohort: Cohort,
    samples: Sequence[eval_cohort.SampleObservation],
    standings: Sequence[eval_cohort.SampleStanding],
    perf: _Perf,
    driven: int,
    behaviour: str,
) -> None:
    """Assemble and write the case's document — its three sections, then everything its samples
    SHARE: the one ask in its several wordings, the one world, the system prompts, and the map
    saying which samples to open. No-op off-report.

    Assembled HERE rather than in the report layer because this is where the three halves meet:
    the claims come from the case body, the pooled variance from the observations, and the
    prompts from what each sample was handed while its database was live."""
    variance = eval_cohort.pool(samples, cohort.features, cohort.shares)
    sections = report.CaseSections(
        case_id=cohort.case_id,
        model=cohort.model,
        behaviour=behaviour,
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
        run=_run_facts(),
    ).render()
    prompts = _case_prompts.pop(cohort.case_id, [])
    eval_artifacts.record_case_report(
        cohort.case_id,
        sections,
        report.render_prompt_variants(report.prompt_variants(prompts, total=len(samples))),
        report.render_case_tail(
            arms=cohort.arms,
            outliers=list(enumerate(standings, start=1)),
            everywhere_distinct=eval_cohort.everywhere_distinct(samples, cohort.features),
        ),
    )


# A chat-eval runner: a case passes ``ask`` + ``world`` and gets a :class:`Cohort` back to
# assert against.
ChatEval = Callable[..., Awaitable["Cohort"]]


async def _seed_sample(
    penny: Penny,
    *,
    world: World | None,
    seed: Seeder | None,
    browse: list[CannedPage] | None,
    prepare: Preparer | None,
) -> None:
    """Lay a sample's world down before its first turn: the user, the world's own seeded
    store, the case's own seed, the embeddings those seeds need, the canned browse, the
    world's mailbox, and the case's late hook."""
    seed_user(penny.db)
    seed_world_stores(penny.db, world)
    if seed is not None:
        seed(penny.db)
    await _embed_seeds(penny)
    if browse is not None:
        install_browse(penny, browse)
    if world is not None and world.mailbox:
        install_mailbox(penny, world.mailbox, fails=world.mailbox_fails)
    if prepare is not None:
        prepare(penny)


def _turn_settled(penny: Penny) -> bool:
    """Whether the turn Penny was handed has finished — its reply delivered and recorded.

    Read off the scheduler's foreground flag, which the channel clears as the last act of
    handling a message: after the reply's send has come back and its record is complete."""
    return not penny.scheduler.foreground_active


async def _drive_turns(
    penny: Penny,
    server: MockSignalServer,
    turns: Sequence[str],
    *,
    timeout: float,
    retryable: bool,
) -> str:
    """Push each user turn, wait for its reply and for the turn to SETTLE, returning the
    LAST reply.

    The mock server holds a reply the moment its send ARRIVES, which is before Penny has
    heard back about it.  A sample that moved on from there tore its channel down under a
    send still in flight: the send failed, and Penny logged her delivery-failure apology
    into the sample's record as a message the user received (#2195).  So a turn is over
    when Penny has finished handling it, not when its reply was seen.

    A model-error reply raises :class:`_ModelCallError` while an attempt remains — that
    reply is the transport failing, not Penny deciding anything, so the sample is
    re-driven from a clean world rather than scored as behaviour."""
    reply = ""
    for turn in turns:
        await server.push_message(sender=TEST_SENDER, content=turn)
        response = await server.wait_for_message(timeout=timeout)
        await wait_until(lambda: _turn_settled(penny), timeout=timeout)
        reply = str(response.get("message", ""))
        if reply == PennyResponse.AGENT_MODEL_ERROR and retryable:
            raise _ModelCallError
    return reply


async def _drive_sample(
    penny: Penny,
    server: MockSignalServer,
    *,
    case_id: str,
    sample_index: int,
    ask: str,
    timeout: float,
    retryable: bool,
    observe: Observer,
) -> SampleResult:
    """ONE attempt at one sample against an already-seeded Penny: push the ask, write the
    sample's report block, read what the turn left behind and dump its thinking.

    The sample scores nothing here — its cohort's claims are answered at case close.  A
    timeout still emits its placeholder block, so the transcript's sample count always
    matches N (#1725/F2).  Raises :class:`_ModelCallError` when the model call itself failed
    and an attempt remains."""
    before = collection_names(penny.db)
    held_before = _held_entries(penny.db)
    reply = ""
    try:
        reply = await _drive_turns(penny, server, [ask], timeout=timeout, retryable=retryable)
        result = SampleResult.graded([])
        _stamp_cause(penny.db, result)
        _write_sample_report(
            penny.db, case_id, sample_index, result=result, reply=reply, driven=[ask]
        )
    except TimeoutError:
        result = SampleResult.binary(["no reply within timeout"])
        _stamp_cause(penny.db, result, timed_out=True)
        _write_sample_report(penny.db, case_id, sample_index, result=result)
    # The observation is read HERE, while this sample's own database is still open — the only
    # moment what the round left behind is available at all.  A timed-out sample reaches this
    # line too, so it is EXCLUDED by name rather than silently absent from the pool.
    result.observation = observe(penny.db, reply, before, held_before)
    _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
    return result


@pytest.fixture
def chat_eval(make_config: Callable[..., Config], tmp_path, request) -> Iterator[ChatEval]:
    """Drive the real chat flow for one ask and return its COHORT.

    A case passes ``ask`` — with ``world`` for what it is answered against, and
    ``also_phrased`` for the other wordings of that SAME request — and gets a
    :class:`Cohort` back to make its claims against.  The three-section report is assembled
    at teardown, once the test body has made them.

    Each sample is fully hermetic — its own mock Signal server, DB, and real-model Penny:
    seed the user and the world (+ any case seed), embed the seeds, push the ask, wait for
    the reply, then read what the turn left behind.  A per-sample server is essential: a
    shared one leaks a prior sample's shut-down channel, which then errors on the next
    sample's broadcast.  A timeout is a sample the pool excludes by name, not a crash.
    """
    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        behaviour: str = "",
        area: Area | None = None,
        ask: str | None = None,
        also_phrased: Sequence[str] = (),
        world: World | None = None,
        model: str = "",
        samples_per_phrasing: int = 0,
        seed: Seeder | None = None,
        prepare: Preparer | None = None,
        samples: int = SAMPLES,
        min_pass_rate: PassRate = UNSTATED,
        timeout: float = 120.0,
        family: str | None = None,
    ) -> Cohort:
        """Drive one request in its wordings and return the COHORT.

        Analysis runs over the complete set, which is what the variance statistics want and
        what no single sample can see.  A case with no ``ask`` is refused by name."""
        _require_ask(case_id, "chat_eval", "ask=<the request>", ask)
        _require_report_only(case_id, min_pass_rate)
        catalogued = _declare_case(
            request,
            case_id=case_id,
            layer=Layer.WHOLE_TURN,
            behaviour=behaviour,
            area=area,
            wording=ask or "",
        )
        eval_artifacts.begin_case(case_id)
        # Chat's convenience over the general form: one world, K wordings of one ask.
        arms = _arms(
            [ask or "", *also_phrased],
            [world] if world is not None else [],
            samples_per_phrasing,
            samples,
        )
        spoken = arms.spoken
        pending = _cohorts.setdefault(
            case_id,
            _PendingCase(
                case_id=case_id,
                family=family,
                module=request.module.__name__,
                behaviour=catalogued.sentence,
            ),
        )

        def _observe(sample_index: int) -> Observer:
            phrasing = arms.label(sample_index)
            arm = arms.index_of(sample_index)
            name = f"{case_id}-{sample_number(sample_index)} ({phrasing})"
            return lambda db, reply, before, held_before: _observe_sample(
                db,
                name=name,
                phrasing=phrasing,
                arm=arm,
                reply=reply,
                before=before,
                held_before=held_before,
            )

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            await _seed_sample(
                penny,
                world=world,
                seed=seed,
                browse=list(world.pages) if world is not None else None,
                prepare=prepare,
            )
            return await _drive_sample(
                penny,
                server,
                case_id=case_id,
                sample_index=sample_index,
                ask=spoken[sample_index],
                timeout=timeout,
                retryable=retryable,
                observe=_observe(sample_index),
            )

        results, perf, voided = await _run_samples(
            make_config,
            tmp_path,
            case_id=case_id,
            samples=arms.driven,
            drive=_drive,
            attempts=_MODEL_CALL_ATTEMPTS,
            model=model,
        )
        cohort = _driven_cohort(case_id, model, results, voided, arms.arms)
        pending.add(cohort, results, perf, intended=arms.driven)
        return cohort

    yield _run
    # The case's claims are made in the TEST BODY, after the drive returns — so the report is
    # assembled here, once the body has had its say.
    for pending in _cohorts.values():
        pending.finish()
    _cohorts.clear()


class CycleCall(NamedTuple):
    """One tool call a collector cycle made — its name and the arguments it carried.

    The arguments travel as the mapping the ledger stores, because a reader asks things of
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
    is a separate schedule that may or may not have run by the time a claim is answered.  A
    pending-only read therefore reports a delivered notification as silence, which is the
    one thing a notify contract must never get wrong."""
    return [row.content for row in _send_queue_rows(db) if row.collection == collection]


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
    arguments rather than raising, since a reader asking WHICH page was fetched wants the
    calls that do carry one."""
    if not isinstance(raw, str):
        return raw if isinstance(raw, dict) else {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


CollectorCyclesEval = Callable[..., Awaitable["Cohort"]]


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
    """The 'the dispatcher actually ran every cycle' guard as a ``Check`` on the sample's
    own block, so a sample whose collection the dispatcher refused can never read as the
    model doing nothing.  The cohort excludes that sample by name (``NO_CYCLE``)."""
    return Check(
        "every cycle ran",
        driven.ran,
        kind="guard",
        rationale=None
        if driven.ran
        else f"the dispatcher refused the collection: {driven.refusal}",
    )


class CycleArm(NamedTuple):
    """ONE arm of a collector cohort: a wording of the job's instruction, and the page that
    answers it.

    A collector has no user turn, so its natural language is the ``extract`` instruction its
    rendered program carries — written by the ``SkillSubstitution`` on that path and printed by
    ``render_skill``, so a wording reaches the model through the shipped instantiation seam
    rather than through a hand-authored program — plus the prose of the page that answers it.
    Those two are what vary across the arms.

    The FACTS do not: one url, one set of bound values, one reading, and a byte-identical datum
    line on every page.  Constant facts are what let a claim name a value — the store holds the
    price the page carries — instead of retreating to a shape that cannot tell a right reading
    from a plausible one.

    ONE register, because a case drives ONE cycle.  Which behaviour a case measures is set by
    the ENTRY CONDITION its ``seed`` lays down — an empty collection or one already holding a
    reading — not by scripting several cycles inside a single test.

    ``text`` is what makes this arm this arm, rendered verbatim in the report so a reader who
    sees "phrasing 3 diverged" can see which wording that was.  ``seed`` lays THIS arm's job
    and entry condition down, ``pages`` is the register its cycle reads, and ``world`` is those
    same pages — carried so a claim is answered against the ground its own sample actually
    read, since the prose differs per arm."""

    text: str
    seed: Seeder
    pages: list[CannedPage]
    world: World


# One cycle's terminal SHAPE, as the enactment contract reads it.  A vocabulary of three words
# rather than a record per cycle: the whole watch contract is "what did each cycle do", so the
# cycles compose into one ordered script that a claim reads and a variance feature measures —
# which is why a multi-cycle sample needs no per-cycle field on the observation.
CYCLE_WROTE = "wrote"
CYCLE_QUIET = "quiet"
CYCLE_TOLD = "told"
CYCLE_DEAD = "no run"


def _cycle_shape(cycle: CycleObservation) -> str:
    """What ONE cycle did: whether it moved the collection, and whether it spoke.

    Read off persisted state — the collection's entries as the cycle found and left them, and
    the SEND QUEUE — never off a harness flag.  A cycle that produced no run of its own is
    named as such rather than reading as a silent success."""
    if cycle.outcome is None and not cycle.calls:
        return CYCLE_DEAD
    moved = CYCLE_WROTE if cycle.changed else CYCLE_QUIET
    return f"{moved}+{CYCLE_TOLD}" if cycle.sent else moved


def cycle_script(cycles: Sequence[CycleObservation]) -> str:
    """The ordered shapes of a sample's cycles — ``wrote, quiet, wrote+told``.

    The whole watch contract in one value: a claim reads a position of it and the variance
    table measures the script entire, so a cohort where one arm stopped notifying shows up as
    a divergence rather than as a number nobody looks at."""
    return ", ".join(_cycle_shape(cycle) for cycle in cycles)


def _cycles_exclusion(db: Database, script: str) -> str | None:
    """Why a cycle-driven sample cannot be counted, or ``None`` when it can.

    A cycle the dispatcher REFUSED never ran against the world the case built, so every claim
    would be answered on a world the model never saw — the failure-cause partition would then
    tag it behavioural, which is the one thing it is not.  Checked ahead of that: a call the
    PROVIDER refused outright (``PROVIDER_REFUSED``, #2526) inside any of the sample's cycles
    is the same class of infrastructure fault as a refused cycle, whichever cycle it hit."""
    if not measured_turn_ran(db):
        return NO_MEASURED_TURN
    if _provider_refused(db):
        return PROVIDER_REFUSED
    if CYCLE_DEAD in script:
        return NO_CYCLE
    return None


def _observe_cycles(
    db: Database,
    driven: _DrivenCycles,
    *,
    name: str,
    phrasing: str,
    arm: int,
    collection: str,
) -> eval_cohort.SampleObservation:
    """Read what ONE cycle-driven sample left behind — its own observer, not the chat one.

    The tool sequence is read from the CYCLES' own calls rather than through
    ``chat_run_tool_sequences``, whose agent filter would return an empty list here on every
    sample (#2017).  The notifications come off the SEND QUEUE, because a cycle enqueues and
    the drainer is a separate schedule — a pending-only read of outgoing messages reports a
    delivered notification as silence — and they are carried as a LIST as well as joined text,
    since "told once" and "told twice" are different findings a blob cannot tell apart.

    ``held``, ``run_outcome`` and ``run_reason`` come off the LAST cycle, which for an
    arm-driven case is its only one.  They are what a collector has instead of a machine walk:
    the store's own end state, and the record fields the run closed with.  ``held`` names the
    BOUND collection on every entry, because it is the one container this job writes to and a
    claim reads the whole entry — key and content — exactly as a chat sample's does."""
    script = cycle_script(driven.observed)
    exclusion = _cycles_exclusion(db, script)
    if exclusion is not None:
        return eval_cohort.SampleObservation(
            name=name, phrasing=phrasing, arm=arm, complete=False, exclusion=exclusion
        )
    sent = [message for cycle in driven.observed for message in cycle.sent]
    last = driven.observed[-1] if driven.observed else None
    return eval_cohort.SampleObservation(
        name=name,
        phrasing=phrasing,
        arm=arm,
        walk=script,
        entries=_stored_entries(db),
        tool_sequence=[call.tool for cycle in driven.observed for call in cycle.calls],
        reply="\n".join(sent),
        given=given_to_the_model(db),
        held=[
            eval_cohort.StoredEntry(collection=collection, key=key, content=content)
            for key, content in (last.after if last is not None else {}).items()
        ],
        run_outcome=last.outcome if last is not None else None,
        run_reason=last.reason if last is not None else None,
        notifications=list(sent),
    )


@pytest.fixture
def collector_cycles_eval(
    make_config: Callable[..., Config], tmp_path, request
) -> Iterator[CollectorCyclesEval]:
    """Drive ONE real collector cycle (``run_for``) per sample and return the case's COHORT.

    The one collector driver there is.  A case passes ``arms`` — five wordings of ONE job's
    instruction, each bringing its own seed, its own page and its own world — and which
    behaviour it measures is set by the ENTRY CONDITION each arm's seed lays down (#1905).
    The cycle's footprint is read off persisted state as a ``CycleObservation``.

    Each sample is hermetic (its own mock Signal server, DB and real-model Penny).  Seeds
    run first, then embeddings backfill, then ``prepare`` gets the constructed Penny — a
    loud world probe, so a drifted seed fails in the seed rather than after GPU time."""

    _cohorts: dict[str, _PendingCase] = {}

    async def _sample(
        penny: Penny,
        *,
        case_id: str,
        sample_index: int,
        collection: str,
        arm: CycleArm,
        prepare: Preparer | None,
        observe: Callable[[Database, _DrivenCycles], eval_cohort.SampleObservation],
    ) -> SampleResult:
        """ONE sample against a constructed Penny: lay its world down, probe it, drive its
        cycle, and write its report block."""
        seed_user(penny.db)
        seed_world_stores(penny.db, arm.world)
        arm.seed(penny.db)
        await _embed_seeds(penny)
        if prepare is not None:
            prepare(penny)
        driven = await _drive_cycles(penny, collection, [arm.pages])
        # The sample is graded from its cohort's CLAIMS, made once every sample has run, so it
        # scores nothing at drive time and carries only the ran-guard.
        result = SampleResult.graded([_cycles_ran_check(driven)])
        _stamp_cause(penny.db, result)
        _write_sample_report(penny.db, case_id, sample_index, result=result)
        # Read while THIS sample's database is still open — the only moment what the cycle
        # left behind is available at all.
        result.observation = observe(penny.db, driven)
        _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
        return result

    async def _run(
        *,
        case_id: str,
        collection: str,
        behaviour: str = "",
        area: Area | None = None,
        arms: Sequence[CycleArm] = (),
        samples_per_phrasing: int = 0,
        model: str = "",
        prepare: Preparer | None = None,
        samples: int = SAMPLES,
        min_pass_rate: PassRate = UNSTATED,
        family: str | None = None,
    ) -> Cohort:
        """Drive one collector cycle per sample and return the COHORT.

        The arms do NOT share a world, which is why the world lives on the arm rather than
        on the cohort.  A case with no ``arms`` is refused by name."""
        _require_ask(case_id, "collector_cycles_eval", "arms=[CycleArm(...), …]", arms)
        _require_report_only(case_id, min_pass_rate)
        catalogued = _declare_case(
            request,
            case_id=case_id,
            layer=Layer.COLLECTOR_CYCLE,
            behaviour=behaviour,
            area=area,
            wording=arms[0].text,
        )
        eval_artifacts.begin_case(case_id)
        driving = _arms(
            [arm.text for arm in arms],
            [arm.world for arm in arms],
            samples_per_phrasing,
            samples,
        )
        pending = _cohorts.setdefault(
            case_id,
            _PendingCase(
                case_id=case_id,
                family=family,
                module=request.module.__name__,
                behaviour=catalogued.sentence,
            ),
        )

        def _observer(
            sample_index: int,
        ) -> Callable[[Database, _DrivenCycles], eval_cohort.SampleObservation]:
            phrasing = driving.label(sample_index)
            arm = driving.index_of(sample_index)
            name = f"{case_id}-{sample_number(sample_index)} ({phrasing})"
            return lambda db, ran: _observe_cycles(
                db, ran, name=name, phrasing=phrasing, arm=arm, collection=collection
            )

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            return await _sample(
                penny,
                case_id=case_id,
                sample_index=sample_index,
                collection=collection,
                arm=arms[driving.index_of(sample_index)],
                prepare=prepare,
                observe=_observer(sample_index),
            )

        results, perf, voided = await _run_samples(
            make_config,
            tmp_path,
            case_id=case_id,
            samples=driving.driven,
            drive=_drive,
            model=model,
        )
        cohort = _driven_cohort(case_id, model, results, voided, driving.arms)
        pending.add(cohort, results, perf, intended=driving.driven)
        return cohort

    yield _run
    # The case's claims are made in the TEST BODY, after the drive returns — so the report is
    # assembled here, once the body has had its say.
    for pending in _cohorts.values():
        pending.finish()
    _cohorts.clear()


# ── End-to-end eval (#2375): scripted chat turns, then the job's own real cycles ─
#
# A chat case answers ONE turn; a collector case answers ONE cycle against a job seeded
# directly.  Neither can say whether what one layer builds is what the next layer actually
# runs on.  An end-to-end case drives a whole SCRIPT of real chat turns — a cold ask, and
# whatever the routine still has to be taught or re-pointed through — and then runs the job
# those turns created through its OWN real cycles, against pages that move between them.
#
# The entry condition (what the chat phase starts from) is still SEEDED, the way every case's
# is; what is new is that the round's own turns are DRIVEN rather than seeded as having
# already happened, because the question is whether the round's output is the cycle's input.
EndToEndEval = Callable[..., Awaitable["Cohort"]]


class EndToEndArm(NamedTuple):
    """ONE arm of an end-to-end cohort: a scripted wording of the whole conversation, and the
    job's own cycles that follow it.

    ``turns`` is this wording's WHOLE script — the first turn is the whole natural ask (the
    page, what to watch, the cadence, the end, and "tell me when it changes"), and every turn
    after it is a FOLLOW-UP: driven only while no job has been created yet THIS RUN, read
    structurally off the registry before it is sent (#2494).  A Penny who stands the job up
    on the first turn is driven once; a Penny who asks first is driven through whatever
    follow-ups the script still needs — the walkthrough, then an acceptance restating the
    terms, or a plain agreement.  Each turn is written to read sensibly against whatever
    Penny actually said back to the one before it, since the arm drives her real reply rather
    than a scripted one.  ``seed``/``world`` lay the chat phase's ground down — an empty
    machine, or a job already configured and running elsewhere.  ``cycles`` is the
    created job's OWN real cycles that follow the script, each against its own browse
    register: the page as taught, then however the case asks it to move."""

    label: str
    turns: Sequence[str]
    seed: Seeder | None
    world: World
    cycles: Sequence[list[CannedPage]]


def _discover_the_new_jobs(db: Database, before: set[str]) -> list[str]:
    """Every collection the script's own turns created and scheduled — found STRUCTURALLY, off
    the registry's before/after read, never by a name the case expects.  A routine minted
    fresh names its own container, and a later variant may put the same ask's job somewhere
    else entirely (#2375), so the driver finds whatever the round built rather than look up a
    name it was told in advance."""
    return _scheduled_by_this_round(db, before)


def _no_job_created_yet(db: Database, before: set[str]) -> bool:
    """Whether the script has stood up NO job yet this run — the structural gate a follow-up
    turn is driven under (#2494).  The same registry read ``_discover_the_new_jobs`` makes at
    the script's end, called again before every turn after the first: a Penny who already
    created a job on an earlier turn gets no further turns driven at her, a Penny who hasn't
    gets the next one."""
    return not _discover_the_new_jobs(db, before)


async def _drive_end_to_end_script(
    penny: Penny,
    server: MockSignalServer,
    *,
    turns: Sequence[str],
    before: set[str],
    timeout: float,
    retryable: bool,
) -> tuple[str, int]:
    """Drive one arm's WHOLE script for real: the first turn unconditionally, then every
    further turn only while no job has been created yet this run (#2494).

    Returns the LAST reply actually driven and how many turns were sent — the count becomes
    the ``TURNS_DRIVEN`` variance feature, since a Penny who stands the job up on turn one and
    a Penny who asks first and is walked through it are both measured on the same claims, and
    only differ in how many turns it took."""
    reply = ""
    sent = 0
    for turn in turns:
        if sent and not _no_job_created_yet(penny.db, before):
            break
        reply = await _drive_turns(penny, server, [turn], timeout=timeout, retryable=retryable)
        sent += 1
    return reply, sent


async def _drive_end_to_end_cycles(
    penny: Penny, jobs: Sequence[str], cycles: Sequence[list[CannedPage]]
) -> tuple[list[eval_cohort.CycleSnapshot], list[_DrivenCycles]]:
    """Run EVERY created job's own real cycles against the arm's page registers.

    Returns the flattened snapshots a claim reads as a union (see ``CycleSnapshot``) beside
    each job's own ``_DrivenCycles`` — kept apart because the ran-guard and the walk string
    need the richer per-job record, while a claim only ever wants the flat list."""
    per_job = [await _drive_cycles(penny, job, cycles) for job in jobs]
    snapshots = [
        eval_cohort.CycleSnapshot(
            job=job,
            index=cycle.index,
            held=[
                eval_cohort.StoredEntry(collection=job, key=key, content=content)
                for key, content in cycle.after.items()
            ],
            sent=list(cycle.sent),
            outcome=cycle.outcome,
            reason=cycle.reason,
        )
        for job, driven in zip(jobs, per_job, strict=True)
        for cycle in driven.observed
    ]
    return snapshots, per_job


def _end_to_end_walk(db: Database, jobs: Sequence[str], per_job: Sequence[_DrivenCycles]) -> str:
    """The sample's whole walk: the conversation machine's moves, then every created job's
    own ordered cycle script — ``idle→elicit, elicit→learn, learn→apply, wrote, quiet,
    wrote+told`` for the ordinary one-job case that ran, each extra job's script joined
    beside it.

    Names the two ways a round can fail to leave anything for a cycle to read, structurally
    rather than by excluding the sample (#2375's review): ``no job`` when the script created
    nothing to run cycles against, ``refused`` when it created one and the dispatcher turned
    every cycle away.  Both are real behavioural failures a variant can reproduce, so the
    claims below have to be able to see them — never a harness-side void."""
    if not jobs:
        return f"{_machine_walk(db)}, no job"
    scripts = [cycle_script(driven.observed) for driven in per_job if driven.observed]
    cycles_walk = " | ".join(scripts) if scripts else "refused"
    return f"{_machine_walk(db)}, {cycles_walk}"


def _observe_end_to_end(
    db: Database,
    *,
    name: str,
    phrasing: str,
    arm: int,
    reply: str,
    before: set[str],
    jobs: list[str],
    per_job: list[_DrivenCycles],
    cycles: list[eval_cohort.CycleSnapshot],
    seeded_replies: int,
    turns_driven: int,
) -> eval_cohort.SampleObservation:
    """Everything an end-to-end sample left behind, read while its database is still live —
    the chat phase's own reads (the machine walk, the mechanisms, what it wrote, what it said)
    joined with the cycle phase's (what the created jobs' cycles held and sent), so a claim
    never has to know which phase produced which fact.

    A sample where the script built no job, or where every job it built had its cycles
    refused, is COMPLETE, not excluded (#2375's review): those are Penny's own failures —
    exactly what an end-to-end case exists to catch — and a variant that regresses either one
    has to be able to fail a claim here, not disappear from the pool.  Only a chat-phase
    model/infra error still voids the sample; it answered no turn at all to read anything
    off."""
    chat_exclusion = _exclusion(db, reply)
    if chat_exclusion is not None:
        return eval_cohort.SampleObservation(
            name=name, phrasing=phrasing, arm=arm, complete=False, exclusion=chat_exclusion
        )
    sent = [message for snapshot in cycles for message in snapshot.sent]
    # ``seeded_replies`` is how many messages the seeded world wrote into the
    # ``penny-messages`` log BEFORE the scripted turns ran; the list is oldest first, so the
    # slice drops them and the reply holds only what THIS run said (#2375).
    combined_reply = "\n".join([*outgoing_replies(db)[seeded_replies:], *sent])
    return eval_cohort.SampleObservation(
        name=name,
        phrasing=phrasing,
        arm=arm,
        turn_at=_last_scripted_turn_at(db),
        timezone=user_timezone_name(db),
        walk=_end_to_end_walk(db, jobs, per_job),
        entries=_stored_entries(db),
        held=_held_entries(db),
        mechanisms=_mechanism_records(db, before),
        tool_sequence=[
            *_chat_tool_sequence(db),
            *[call.tool for driven in per_job for cycle in driven.observed for call in cycle.calls],
        ],
        reply=combined_reply,
        aborted_on_model_error=_aborted_on_model_error(reply),
        given=given_to_the_model(db),
        notifications=sent,
        cycles=cycles,
        turns_driven=turns_driven,
    )


async def _drive_end_to_end_sample(
    penny: Penny,
    server: MockSignalServer,
    *,
    case_id: str,
    sample_index: int,
    arm: EndToEndArm,
    timeout: float,
    retryable: bool,
    observe: Callable[..., eval_cohort.SampleObservation],
) -> SampleResult:
    """ONE sample: seed the chat phase's ground, run the whole script for REAL, find every
    job it created, then run each one's own cycles for REAL against the arm's pages."""
    await _seed_sample(
        penny,
        world=arm.world,
        seed=arm.seed,
        browse=list(arm.world.pages),
        prepare=None,
    )
    # The seeded world's own sends are already in the message log before the turns run — a
    # COUNT (not a timestamp; the eval clock is pinned) so the observation reads only this
    # run's replies (#2375).
    seeded_replies = len(outgoing_replies(penny.db))
    before = collection_names(penny.db)
    reply, turns_driven = await _drive_end_to_end_script(
        penny, server, turns=arm.turns, before=before, timeout=timeout, retryable=retryable
    )
    jobs = _discover_the_new_jobs(penny.db, before)
    if jobs:
        cycles, per_job = await _drive_end_to_end_cycles(penny, jobs, arm.cycles)
    else:
        cycles, per_job = [], []
    # No drive-time guard: the cohort's claims are the only way a sample is graded, and a
    # sample that built no job or ran no cycle is COMPLETE — see ``_observe_end_to_end``.
    result = SampleResult.graded([])
    _stamp_cause(penny.db, result)
    # ``driven`` is only the turns ACTUALLY sent — a script whose first turn stood the job up
    # drove one, never the follow-ups the report would otherwise show as having been said.
    _write_sample_report(
        penny.db,
        case_id,
        sample_index,
        result=result,
        reply=reply,
        driven=list(arm.turns[:turns_driven]),
    )
    result.observation = observe(
        penny.db,
        reply=reply,
        before=before,
        jobs=jobs,
        per_job=per_job,
        snapshots=cycles,
        seeded_replies=seeded_replies,
        turns_driven=turns_driven,
    )
    _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
    return result


@pytest.fixture
def end_to_end_eval(
    make_config: Callable[..., Config], tmp_path, request
) -> Iterator[EndToEndEval]:
    """Drive one end-to-end sample per arm — a scripted whole conversation, then the job it
    created's own real cycles — and return the case's COHORT.

    A case passes ``arms`` — five scripted wordings of the same whole conversation, each
    bringing its own seed, its own chat-phase world and its own per-cycle pages.  Which jobs
    exist to run cycles against is discovered structurally after the script runs; a case
    never names the container its claims hinge on (#2375)."""
    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        behaviour: str = "",
        area: Area | None = None,
        arms: Sequence[EndToEndArm] = (),
        samples_per_phrasing: int = 0,
        model: str = "",
        samples: int = SAMPLES,
        min_pass_rate: PassRate = UNSTATED,
        family: str | None = None,
    ) -> Cohort:
        """Drive one end-to-end script per sample and return the COHORT.

        A case with no ``arms`` is refused by name, the way every other driver refuses a case
        that names nothing to drive."""
        _require_ask(case_id, "end_to_end_eval", "arms=[EndToEndArm(...), …]", arms)
        _require_report_only(case_id, min_pass_rate)
        catalogued = _declare_case(
            request,
            case_id=case_id,
            layer=Layer.WHOLE_TURN,
            behaviour=behaviour,
            area=area,
            wording=arms[0].label,
        )
        eval_artifacts.begin_case(case_id)
        driving = _arms(
            [arm.label for arm in arms],
            [arm.world for arm in arms],
            samples_per_phrasing,
            samples,
        )
        pending = _cohorts.setdefault(
            case_id,
            _PendingCase(
                case_id=case_id,
                family=family,
                module=request.module.__name__,
                behaviour=catalogued.sentence,
            ),
        )

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            phrasing = driving.label(sample_index)
            arm_index = driving.index_of(sample_index)
            name = f"{case_id}-{sample_number(sample_index)} ({phrasing})"

            # ``snapshots``, never ``cycles`` — the latter is a reserved per-sample-scorer
            # argument name the AST pin in ``test_eval_harness.py`` refuses on any driver.
            def observe(
                db: Database,
                *,
                reply: str,
                before: set[str],
                jobs: list[str],
                per_job: list[_DrivenCycles],
                snapshots: list[eval_cohort.CycleSnapshot],
                seeded_replies: int,
                turns_driven: int,
            ) -> eval_cohort.SampleObservation:
                return _observe_end_to_end(
                    db,
                    name=name,
                    phrasing=phrasing,
                    arm=arm_index,
                    reply=reply,
                    before=before,
                    jobs=jobs,
                    per_job=per_job,
                    cycles=snapshots,
                    seeded_replies=seeded_replies,
                    turns_driven=turns_driven,
                )

            return await _drive_end_to_end_sample(
                penny,
                server,
                case_id=case_id,
                sample_index=sample_index,
                arm=arms[arm_index],
                timeout=240.0,
                retryable=retryable,
                observe=observe,
            )

        results, perf, voided = await _run_samples(
            make_config,
            tmp_path,
            case_id=case_id,
            samples=driving.driven,
            drive=_drive,
            model=model,
        )
        cohort = _driven_cohort(case_id, model, results, voided, driving.arms)
        pending.add(cohort, results, perf, intended=driving.driven)
        return cohort

    yield _run
    for pending in _cohorts.values():
        pending.finish()
    _cohorts.clear()


# ── Browse EXTRACTION: what a page half-carries (#1942) ───────────────────────

ExtractorEval = Callable[..., Awaitable["Cohort"]]


class FieldExpectation(NamedTuple):
    """One thing an extract instruction asks for, and how the page answers it (#1942).

    ``anchor`` is the span of the page that supplies it, compared through the SHIPPED
    ``spoken_form`` — so a value passes whether or not the draw kept the punctuation or
    the article in front of it.  Deliberately not an equality: which words carry a fact
    has a little play in it, and a claim demanding one exact string would be answering
    for the draw.

    An EMPTY anchor is the ABSENT direction: the page supplies nothing for this, so it is
    named here to say what the instruction asked for and the page did not have."""

    field: str
    anchor: str = ""


def _micro_context_rows(db: Database, *agent_names: str) -> list[PromptLog]:
    """The sample's micro-context promptlog rows for the named customers, in LEDGER
    order — one per draw, so a reroll shows as a second row (the fragile signal and
    the transcript's second 🧩 pair both read off this).

    Takes SEVERAL customers because a run end is not always one draw: a report naming
    only one of several cooperating micro-contexts would render a transcript that
    cannot show the decision the case is scoring."""
    wanted = frozenset(agent_names)
    return [row for row in _sample_prompt_rows(db) if row.agent_name in wanted]


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
    agent_names: Sequence[str],
) -> None:
    """One micro-context sample's transcript block — hand-built (the
    generic extractor is chat-run-shaped; this sample is one step whose actor is the
    🧩 micro-context, the spec's official sub-model actor), rendered by the SAME pure
    report grammar and appended to the same ``<case_id>.md``. No-op off-report.
    ``agent_names`` selects the customers' rows, rendered in one ledger-ordered
    stream — each 🧩 pair carries its own context label, so two customers read as the
    two actors they are.

    The sample's system prompts are deposited for the case document the same way the
    chat path deposits them (#2062), off the SAME rows the transcript renders — so every
    prompt the document states belongs to an actor the reader can see acting, and a
    microcontext failure carries the state the draw was given rather than sending the
    reader to the sample database for it."""
    report_dir = os.environ.get("EVAL_REPORT_DIR")
    if not report_dir:
        return
    rows = _micro_context_rows(db, *agent_names)
    if not rows:
        transcript = report.SampleTranscript(
            sample_number(sample_index),
            _UNSETTLED_BANNER,
            [],
            placeholder=report.NO_TURNS_PLACEHOLDER,
        )
    else:
        _record_case_prompts(case_id, sample_index, _system_prompts(rows))
        events = _classifier_events(phrasing, rows)
        checks = _classifier_check_views(result, len(events) - 1, baseline_from_env(), case_id)
        passed_checks, total = _scored_counts(result)
        transcript = report.build_sample(
            number=sample_number(sample_index),
            banner=_UNSETTLED_BANNER,
            events=events,
            checks=checks,
            run_close_score=f"{passed_checks}/{total}",
        )
    _record_sample_block(case_id, sample_index, transcript, result, live_prompt_perf(db))


def _extract_rerolled(db: Database) -> bool:
    """Did the extraction draw more than once — the fragile signal, a sample that only
    got there by recovering.  One customer here, so one row is the clean case."""
    return len(_micro_context_rows(db, PennyConstants.BROWSE_EXTRACT_AGENT_NAME)) > 1


# The fields of a browse extraction's structured answer, named once.  A case asserts them
# and measures them by these names, so the two halves of #2017's framing — deterministic and
# variance — read the same value rather than two spellings of it.
EXTRACT_OUTCOME = "outcome"
EXTRACT_VALUE = "value"
EXTRACT_REASON = "reason"
# The stated shortfall (#2360) — what the instruction named that the page does not carry
# at all, read alongside an EXTRACTED value.  Empty on every outcome that isn't EXTRACTED,
# and often empty on EXTRACTED too (a page that carries everything named has nothing to
# report missing) — both are the honest read, not an absent field.
EXTRACT_MISSING = "missing"

# The outcomes that mean the draw never produced a usable answer at all.  A sample landing on
# one of these is EXCLUDED rather than scored: it is the micro-context twin of a chat turn
# that produced no reply, and grading it would report a failed contract for a contract nobody
# exercised.
_EXTRACT_FAILURES = frozenset(
    {MicroExtractOutcome.EXTRACTION_FAILED, MicroExtractOutcome.POISON_REROLL_FAILED}
)


def _extraction_output(result: MicroContextResult) -> list[eval_cohort.OutputField]:
    """One extraction draw's STRUCTURED ANSWER, field by field.

    All four fields, always — including the empty one.  Which of ``value``, ``reason`` and
    ``missing`` is populated is decided by the outcome, so carrying only the filled ones
    would make the cohort's field comparison read a different set of axes per sample and
    quietly turn "the draw answered differently" into "the draw answered a different
    question"."""
    return [
        eval_cohort.OutputField(name=EXTRACT_OUTCOME, value=result.outcome.value),
        eval_cohort.OutputField(name=EXTRACT_VALUE, value=result.value),
        eval_cohort.OutputField(name=EXTRACT_REASON, value=result.reason),
        eval_cohort.OutputField(name=EXTRACT_MISSING, value=result.missing),
    ]


def _observe_extraction(
    db: Database,
    result: MicroContextResult | None,
    *,
    name: str,
    phrasing: str,
    arm: int,
) -> eval_cohort.SampleObservation:
    """Read what ONE extraction sample answered — its own observer, not the chat one.

    A single-call context leaves no machine walk, no registry row, no collection entry and no
    outgoing message, so every read ``_observe_sample`` makes would come back empty here and
    the row would be structurally fine and substantively hollow.  What there IS to read is the
    draw's typed result, and it is read directly rather than fished back out of the ledger.

    ``given`` is the draw's own prompt, read through ``given_to_the_model`` (#2078) — the page
    and the instruction under the extraction contract, so a provenance claim compares the
    answer against the whole of what the draw was handed, ``NOT_PRESENT:`` included."""
    answer = "" if result is None or result.outcome in _EXTRACT_FAILURES else result.outcome.value
    exclusion = _draw_exclusion(db, answer)
    if result is None or exclusion is not None:
        return eval_cohort.SampleObservation(
            name=name, phrasing=phrasing, arm=arm, complete=False, exclusion=exclusion or NO_DRAW
        )
    return eval_cohort.SampleObservation(
        name=name,
        phrasing=phrasing,
        arm=arm,
        given=given_to_the_model(db),
        output=_extraction_output(result),
    )


def _extraction_world(url: str, page: str, reading: PageReading) -> World:
    """The one page an extraction case is answered against, as a :class:`World`.

    Built from the case's own ``url``/``page``/``reading`` rather than declared separately, so
    there is one source of truth for what the draw read and the report's world fold renders
    exactly the document the draw was handed.  ``keeps``/``excludes`` are empty: those token
    sets exist for claims about what a ROUND chose to store, and this shape stores nothing."""
    return World(
        name=url, pages=(CannedPage(match=url, text=page, reading=reading),), keeps=(), excludes=()
    )


@pytest.fixture
def extractor_eval(
    make_config: Callable[..., Config], tmp_path, request
) -> Iterator[ExtractorEval]:
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

    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        url: str,
        page: str,
        instruction: str,
        behaviour: str = "",
        area: Area | None = None,
        also_instructed: Sequence[str] = (),
        samples_per_phrasing: int = 0,
        model: str = "",
        samples: int = SAMPLES,
        min_pass_rate: PassRate = UNSTATED,
        timeout: float = SINGLE_CALL_WAIT_SECONDS,
        family: str | None = None,
        reading: PageReading = PageReading.ARTICLE,
    ) -> Cohort:
        """Drive one page + instruction through the extraction draw and return its COHORT.

        A case passes ``instruction`` and ``also_instructed`` — the other wordings of the SAME
        request — and gets a :class:`Cohort` back to assert against.  The arm here is the
        ``extract`` instruction, which is PENNY's own text: production writes it at the browse
        call site, so what the cohort measures is whether the answer survives how the calling
        draw happened to word the request.  The page is FIXED across the arms; it is the world.
        A case with no ``instruction`` is refused by name.

        ``reading`` (#2388) declares how the extension would have read ``page`` — ARTICLE (the
        default) or LINK_LIST — so a case presents the extractor's structural note the same
        way production does: the note rides ``MicroContext.extract``'s own ``reading`` argument,
        never guessed here from the page's shape.
        """
        _require_ask(case_id, "extractor_eval", "instruction=<the extract request>", instruction)
        _require_report_only(case_id, min_pass_rate)
        catalogued = _declare_case(
            request,
            case_id=case_id,
            layer=Layer.MICRO_CONTEXT,
            behaviour=behaviour,
            area=area,
            wording=instruction,
        )
        eval_artifacts.begin_case(case_id)
        # One page, K wordings of the instruction — the same shape chat has, and the world is
        # a property of the CASE rather than of the arm.
        arms = _arms(
            [instruction, *also_instructed],
            [_extraction_world(url, page, reading)],
            samples_per_phrasing,
            samples,
        )
        spoken = arms.spoken
        content = f"{PennyConstants.BROWSE_PAGE_HEADER}{url}\n{page}"
        pending = _cohorts.setdefault(
            case_id,
            _PendingCase(
                case_id=case_id,
                family=family,
                module=request.module.__name__,
                behaviour=catalogued.sentence,
            ),
        )

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            asked = spoken[sample_index]
            micro = MicroContext(penny.model_client)
            extracted: MicroContextResult | None = None
            try:
                extracted = await asyncio.wait_for(
                    micro.extract(
                        content,
                        asked,
                        call_link=CallLink.new_top(),
                        run_target=penny.chat_agent.name,
                        reading=reading,
                    ),
                    timeout=timeout,
                )
                # The case is graded from its cohort's CLAIMS, made after every sample has
                # run, so the sample itself scores nothing at drive time.
                result = SampleResult.graded([])
                result.observe_rerolled(rerolled=_extract_rerolled(penny.db))
                _stamp_cause(penny.db, result)
            except TimeoutError:
                result = SampleResult.binary(["no extraction draw within timeout"])
                _stamp_cause(penny.db, result, timed_out=True)
            phrasing = arms.label(sample_index)
            result.observation = _observe_extraction(
                penny.db,
                extracted,
                name=f"{case_id}-{sample_number(sample_index)} ({phrasing})",
                phrasing=phrasing,
                arm=arms.index_of(sample_index),
            )
            _write_classifier_report(
                penny.db,
                case_id,
                sample_index,
                result=result,
                phrasing=asked,
                agent_names=(PennyConstants.BROWSE_EXTRACT_AGENT_NAME,),
            )
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

        results, perf, voided = await _run_samples(
            make_config, tmp_path, case_id=case_id, samples=arms.driven, drive=_drive, model=model
        )
        cohort = _driven_cohort(case_id, model, results, voided, arms.arms)
        pending.add(cohort, results, perf, intended=arms.driven)
        return cohort

    yield _run
    # The case's claims are made in the TEST BODY, after the drive returns — so the report is
    # assembled here, once the body has had its say.
    for pending in _cohorts.values():
        pending.finish()
    _cohorts.clear()


# ── The notify COMPOSER (#1911/#2363): a driver of its own ──────────────────────────────
def _notify_rerolled(db: Database) -> bool:
    """Did the notify draw more than once — the fragile signal, a sample that only got there
    by recovering.  One customer here, so one row is the clean case."""
    return len(_micro_context_rows(db, PennyConstants.NOTIFY_COMPOSE_AGENT_NAME)) > 1


def _observe_notify(
    db: Database, message: str | None, *, name: str, phrasing: str, arm: int
) -> eval_cohort.SampleObservation:
    """Read what ONE notify-compose sample answered — its own observer, not the chat one.

    A single-call context leaves no machine walk, no registry row, no collection entry and no
    outgoing message of its own — the send/queue gate is Python and runs nowhere near this
    draw.  What there IS to read is the drawn message itself, carried as ``reply`` rather than
    a structured ``output`` field: unlike extraction's typed ``MicroContextResult``, this
    customer's whole contract IS the message a user would receive, so the two SHARED reply
    claims (``assert_the_reply_answers_the_ask``, ``assert_every_value_in_the_reply_is_sourced``)
    read it directly rather than needing a composer-specific claim.

    ``given`` is the draw's own prompt, read through ``given_to_the_model`` (#2078) — the
    same contract every other single-call micro-context's provenance claim reads."""
    exclusion = _draw_exclusion(db, message or "")
    if message is None or exclusion is not None:
        return eval_cohort.SampleObservation(
            name=name, phrasing=phrasing, arm=arm, complete=False, exclusion=exclusion or NO_DRAW
        )
    return eval_cohort.SampleObservation(
        name=name, phrasing=phrasing, arm=arm, given=given_to_the_model(db), reply=message
    )


NotifyComposerEval = Callable[..., Awaitable["Cohort"]]


@pytest.fixture
def notify_composer_eval(
    make_config: Callable[..., Config], tmp_path, request
) -> Iterator[NotifyComposerEval]:
    """Drive the notify COMPOSER micro-context (#1911/#2363) N times, and NOTHING else.

    The composer's whole input is ONE framework-assembled document (``penny.notification``):
    what a cycle wrote, how it got there, and the past messages Python already looked up — no
    cycle runs, because what is measured is what the composer makes of a document rather than
    whether a cycle can produce one.  ``base`` carries every fact (the findings, the calls, the
    related messages) FIXED across the case's arms; ``description``/``also_described`` carry
    the one piece of upstream free text the document holds — the collection's own description
    — in five phrasings, the same role ``extractor_eval``'s ``instruction``/``also_instructed``
    play for the browse extractor.

    ``MicroContext.compose_notification`` makes the call, so the case exercises the shipped
    prompt, the shipped parse and the shipped reroll budget.  Synthetic here means the
    DOCUMENT's facts and the DESCRIPTION wordings are authored, never the prompt."""

    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        base: NotificationInput,
        description: str,
        world: World,
        behaviour: str = "",
        area: Area | None = None,
        also_described: Sequence[str] = (),
        samples_per_phrasing: int = 0,
        model: str = "",
        samples: int = SAMPLES,
        min_pass_rate: PassRate = UNSTATED,
        timeout: float = SINGLE_CALL_WAIT_SECONDS,
        family: str | None = None,
    ) -> Cohort:
        """Drive one document through the notify-compose draw, K phrasings of its own
        description, and return its COHORT.  A case with no ``description`` is refused by
        name."""
        _require_ask(
            case_id,
            "notify_composer_eval",
            "description=<the collection's own words>",
            description,
        )
        _require_report_only(case_id, min_pass_rate)
        catalogued = _declare_case(
            request,
            case_id=case_id,
            layer=Layer.MICRO_CONTEXT,
            behaviour=behaviour,
            area=area,
            wording=description,
        )
        eval_artifacts.begin_case(case_id)
        # One document's fixed facts, K wordings of its own description — the same shape
        # the extractor has with a fixed page and K wordings of the instruction.
        documents = [
            base.model_copy(update={"description": text}) for text in (description, *also_described)
        ]
        rendered = [render_notification(document) for document in documents]
        arms = _arms(rendered, [world], samples_per_phrasing, samples)
        spoken = arms.spoken
        pending = _cohorts.setdefault(
            case_id,
            _PendingCase(
                case_id=case_id,
                family=family,
                module=request.module.__name__,
                behaviour=catalogued.sentence,
            ),
        )

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            content = spoken[sample_index]
            micro = MicroContext(penny.model_client)
            message: str | None = None
            try:
                message = await asyncio.wait_for(
                    micro.compose_notification(
                        content, call_link=CallLink.new_top(), run_target=penny.chat_agent.name
                    ),
                    timeout=timeout,
                )
                # The case is graded from its cohort's CLAIMS, made after every sample has
                # run, so the sample itself scores nothing at drive time.
                result = SampleResult.graded([])
                result.observe_rerolled(rerolled=_notify_rerolled(penny.db))
                _stamp_cause(penny.db, result)
            except TimeoutError:
                result = SampleResult.binary(["no notify draw within timeout"])
                _stamp_cause(penny.db, result, timed_out=True)
            phrasing = arms.label(sample_index)
            result.observation = _observe_notify(
                penny.db,
                message,
                name=f"{case_id}-{sample_number(sample_index)} ({phrasing})",
                phrasing=phrasing,
                arm=arms.index_of(sample_index),
            )
            _write_classifier_report(
                penny.db,
                case_id,
                sample_index,
                result=result,
                phrasing=content,
                agent_names=(PennyConstants.NOTIFY_COMPOSE_AGENT_NAME,),
            )
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

        results, perf, voided = await _run_samples(
            make_config, tmp_path, case_id=case_id, samples=arms.driven, drive=_drive, model=model
        )
        cohort = _driven_cohort(case_id, model, results, voided, arms.arms)
        pending.add(cohort, results, perf, intended=arms.driven)
        return cohort

    yield _run
    for pending in _cohorts.values():
        pending.finish()
    _cohorts.clear()
