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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel import Session, select

from penny.config import Config
from penny.constants import ChannelType, PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.message_store import PromptPerf
from penny.database.models import MemoryRow, PromptLog
from penny.llm.client import LlmClient
from penny.llm.models import LlmMessage, LlmResponse, LlmToolCall, LlmToolCallFunction
from penny.penny import Penny
from penny.startup import get_restart_message
from penny.tests.conftest import TEST_SENDER, run_penny_with_server
from penny.tests.eval import artifacts as eval_artifacts
from penny.tests.eval.artifacts import FailureCause
from penny.tests.eval.baseline import Baseline, baseline_from_env
from penny.tests.eval.fixtures import CannedPage, SynthCollection
from penny.tests.mocks.signal_server import MockSignalServer
from penny.text_validity import (
    has_leaked_harmony_envelope,
    is_call_fragment_reply,
    is_degenerate_run,
    is_degenerate_tool_name,
)
from penny.tools.base import RESULT_TAG
from penny.tools.browse import BrowseChannelUnavailableError

# Samples per case.  Override with EVAL_SAMPLES=2 for a quick smoke run.
SAMPLES = int(os.environ.get("EVAL_SAMPLES", "5"))

# Embedding backfill batch size for seeded memory.
_EMBED_BATCH = 100

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

    @classmethod
    def na(cls, label: str, *, rationale: str | None = None, anchor: str | None = None) -> Check:
        """A not-applicable check — this sample's branch didn't run, so it's excluded from the
        graded denominator (neither pass nor fail).  Still rendered (➖) so a skipped expectation
        reads as skipped, not forgotten."""
        return cls(label=label, ok=True, anchor=anchor, rationale=rationale, ignored=True)


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

    @property
    def passed(self) -> bool:
        return self.score >= 1.0

    @classmethod
    def binary(cls, fails: list[str]) -> SampleResult:
        return cls(0.0 if fails else 1.0, list(fails), 1)

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
    interval: int | None = None,
    notify: bool = False,
) -> None:
    """Create a synthetic collection + its entries (key = text before ' — ')."""
    db.memories.create_collection(
        synth.name,
        synth.description,
        extraction_prompt=extraction_prompt,
        collector_interval_seconds=interval,
        notify=notify,
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
        for row in db.messages.recent_prompts(limit=200)
        for call in _response_tool_calls(row)
        if call.get("function", {}).get("name") == tool_name
    )


_GAVE_UP = re.compile(
    r"\b(sorry|apolog\w+)\b.{0,50}"
    r"\b(wasn't|was not|couldn't|could not|can't|cannot|unable|not able)\b",
    re.IGNORECASE,
)


def _iter_prompt_messages(db: Database):
    """Every message across the run's promptlog (accumulated history + tool results)."""
    for row in db.messages.recent_prompts(limit=200):
        yield from (json.loads(row.messages) if row.messages else [])


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
    """Did the run reach its result SHAKILY — through a rejected / refused / recovered tool call?

    Scans the persisted promptlog for any recovery frame (a tool-result failure, a framework
    refusal narration — the ``_RECOVERY_FRAMES`` set).  A green sample that only got there after
    the loop refused a call and it retried is 'passed, fragile' in the report: real, but not
    robust.  Derived from the same promptlog primitives as ``tool_call_rejected``, not a new
    model judgment.

    Unlike ``tool_call_rejected`` this filters on NO tool name — it asks "did the run recover
    from *anything*?" — so it carries none of that probe's target-vs-tool-name attribution gap
    (#1726): a memory-tool execute-time failure narrates ``… but it didn't work:``, whose
    ``didn't work`` fragment is already in ``_RECOVERY_FRAMES``, so it is caught regardless of
    which tool (target-backticked) produced it.  The audit found no frame-set gap here."""
    for message in _iter_prompt_messages(db):
        if message.get("role") != "tool":
            continue
        content = message.get("content") or ""
        if any(frame in content for frame in _RECOVERY_FRAMES):
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
    reply?  Mirrors ``Agent._unusable_output_condition`` over the persisted OUTPUT: the text
    content, each serialised tool-call argument, and each tool-call name."""
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
    call-fragment reply.  Reading only the ``response`` (never the input ``messages``) is what
    makes this immune to a DELIBERATELY-injected recovery trigger: an ``_Inject*`` bail is
    returned as a SYNTHETIC ``LlmResponse`` that bypasses the persisting real client, so it
    never lands in a persisted ``response`` — a ``bail_injected`` sample is tagged pathology
    only if the LIVE model additionally produced its own poison, never for the forced trigger."""
    return any(_response_is_poison(row) for row in db.messages.recent_prompts(limit=200))


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
    for row in db.messages.recent_prompts(limit=200):
        for call in _response_tool_calls(row):
            if call.get("function", {}).get("name") == tool_name:
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
    for row in db.messages.recent_prompts(limit=200):
        for call in _response_tool_calls(row):
            if call.get("function", {}).get("name") != tool_name:
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
    for row in reversed(db.messages.recent_prompts(limit=200)):
        for call in _response_tool_calls(row):
            name = call.get("function", {}).get("name")
            if isinstance(name, str):
                names.append(name)
    return names


# ── Shared loop-health + reply helpers (uniform across the eval case files) ──
# The chat loop's text-bail nudges (injected as a user turn when the model emits
# prose OR a call-shaped JSON blob instead of a real tool call) — their presence
# means the routing slipped, even if recovery then succeeded.  Loop-health
# visibility, not a behavior score.  TWO distinct markers cover the bail family
# (an earlier single marker silently missed one and false-greened a spiral);
# each is an ASCII, newline-free slice that survives row.messages JSON-escaping.
_BAIL_NUDGE_MARKERS = (
    "could not be parsed as a tool call",  # Prompt.TOOL_FORMAT_NUDGE
    "wrote a tool call as plain text",  # Prompt.CHAT_CALL_AS_TEXT_NUDGE
)
_CONTINUE_NUDGE_MARKER = "Please provide your response"  # Prompt.CONTINUE_NUDGE


def bail_nudge_fired(db: Database) -> bool:
    """True when any prompt's message array carries an injected text-bail nudge."""
    for row in db.messages.recent_prompts(limit=200):
        if row.messages and any(marker in row.messages for marker in _BAIL_NUDGE_MARKERS):
            return True
    return False


def continue_nudge_fired(db: Database) -> bool:
    """True when any prompt's message array carries the empty-response retry nudge."""
    for row in db.messages.recent_prompts(limit=200):
        if row.messages and _CONTINUE_NUDGE_MARKER in row.messages:
            return True
    return False


def routing_clean(db: Database) -> bool:
    """The uniform loop-health verdict every case reports as an ADVISORY check
    (``Check(..., scored=False)``): no bail nudge AND no continue nudge fired."""
    return not bail_nudge_fired(db) and not continue_nudge_fired(db)


def outgoing_replies(db: Database) -> list[str]:
    """Every message Penny sent this sample (the per-turn replies), oldest first."""
    entries = db.memory("penny-messages").read_recent(window_seconds=3600, cap=None)
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
        (
            row
            for row in db.messages.recent_prompts(limit=200)
            if row.agent_name == PennyConstants.CHAT_AGENT_NAME
        ),
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
            name
            for call in _response_tool_calls(row)
            if isinstance(name := call.get("function", {}).get("name"), str)
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
    for row in db.messages.recent_prompts(limit=200):
        for call in _response_tool_calls(row):
            if call.get("function", {}).get("name") != tool_name:
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
    for row in db.messages.recent_prompts(limit=200):
        for call in _response_tool_calls(row):
            function = call.get("function", {})
            if function.get("name") not in _KEY_BEARING_TOOLS:
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
    seeded descriptions/entries get embeddings that ``read_similar`` /
    resolve-by-meaning can match.
    """
    await penny._backfill_memory_embeddings(_EMBED_BATCH)
    await penny._backfill_description_embeddings(_EMBED_BATCH)


def _assert_threshold(
    case_id: str,
    results: list[SampleResult],
    min_pass_rate: float | None,
    *,
    gate_pathology_excluded: bool = False,
) -> None:
    """Print the case's X/Y pass rate, and — unless report-only — gate on it.

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


def _report_cell(text: str, limit: int = 1500) -> str:
    """One markdown table cell: escape pipes, newlines → <br>, truncate a long result."""
    cell = str(text).strip().replace("|", "\\|").replace("\n", "<br>")
    return cell if len(cell) <= limit else cell[:limit] + " …[truncated]"


def _sample_turns(rows: list[PromptLog], reply: str) -> list[tuple[str, str]]:
    """(actor, content) for every turn of the sample, across ALL promptlog rows — so a
    multi-turn conversation shows EVERY turn's tool calls, not just the last turn's.

    Each row's ``messages`` array accumulates the conversation up to that LLM call (a later
    turn carries an earlier one only as text history, so an earlier turn's tool calls live
    only in that turn's own rows).  Walking every row and de-duplicating by (actor, content)
    yields each user turn, tool call, tool result, and intermediate reply exactly once, in
    order.  The final reply (the last response's text, which is in no messages array) is
    appended last.  System prompt omitted."""
    turns: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def emit(actor: str, content: str) -> None:
        if content and (actor, content) not in seen:
            seen.add((actor, content))
            turns.append((actor, content))

    for row in rows:
        for message in json.loads(row.messages) if row.messages else []:
            role, content = message.get("role"), message.get("content") or ""
            if role == "user":
                emit(_ACTOR["user"], content)
            elif role == "tool":
                emit(_ACTOR["tool"], content)
            elif role == "assistant":
                for call in message.get("tool_calls") or []:
                    function = call.get("function", {})
                    emit(_ACTOR["call"], f"{function.get('name')}({function.get('arguments')})")
                emit(_ACTOR["penny"], content)
    emit(_ACTOR["penny"], reply.strip())
    return turns


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


# The REGRESSED suffix on a failing check that was green in the prior run (#1693) — a
# flip, distinct from a check that was already red.  Only ever appended to a ❌.
_REGRESSED_MARK = "❌ 🔻 REGRESSED"


def _check_mark(check: Check, *, regressed: bool = False) -> str:
    """The report marker for a check: ➖ when not-applicable (``ignored``), ✅ when passed,
    else ❌ — or ``❌ 🔻 REGRESSED`` when this failing check was fully green in the baseline
    run (``regressed``, #1693)."""
    if check.ignored:
        return "➖"
    if check.ok:
        return "✅"
    return _REGRESSED_MARK if regressed else "❌"


def _regressed_ids(checks: list[Check], baseline: Baseline | None, case_id: str) -> set[int]:
    """The ``id()`` of every check that FAILED this sample but was fully green in the baseline
    run — a flip (#1693).  Empty with no baseline, so a first run marks failures as plain ❌."""
    if baseline is None:
        return set()
    return {
        id(check)
        for check in checks
        if not check.ignored and not check.ok and baseline.was_passing(case_id, check.label)
    }


def _place_checks(
    checks: list[Check], turns: list[tuple[str, str]]
) -> tuple[dict[int, list[Check]], list[Check]]:
    """Bind each anchored check to the FIRST turn whose content contains its anchor.

    A ``REPLY_ANCHOR`` check stamps the final Penny-reply row (it tests the reply's text, not
    a tool call).  Returns ``(turn_index -> the checks placed there, leftover checks)`` — the
    per-turn check lists (not pre-rendered marks) so a caller can both stamp the row (via
    ``_row_mark``) and decide whether that turn needs a thinking block (#1693).  A check with no
    anchor — or whose anchor matches no turn (a *missing* expected action, a tool call that never
    happened) — has no row to sit on, so it falls to ``leftover`` (a footer)."""
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


def _row_mark(placed_checks: list[Check], regressed_ids: set[int]) -> str:
    """The stamp appended to a turn's actor cell — the placed checks' marks concatenated (with
    a leading space), or ``""`` for an unmarked row.  A regressed failing check renders its
    REGRESSED form (#1693)."""
    if not placed_checks:
        return ""
    return " " + "".join(_check_mark(c, regressed=id(c) in regressed_ids) for c in placed_checks)


def _sample_db_path(tmp_path, case_id: str, sample_index: int) -> str:
    """Where a sample's hermetic DB lives.  When ``EVAL_REPORT_DIR`` is set the DB
    persists BESIDE the reports (the mounted dir survives the ``--rm`` container),
    so a run's raw promptlog can be re-read after the fact — same doctrine as the
    transcripts: the evidence always survives the run.  Unset → tmp_path as before."""
    report_dir = os.environ.get("EVAL_REPORT_DIR")
    base = Path(report_dir) if report_dir else tmp_path
    Path(base).mkdir(parents=True, exist_ok=True)
    return str(Path(base) / f"{case_id}-{sample_index}.db")


def _sample_verdict(result: SampleResult, *, fragile: bool) -> str:
    """The sample's header verdict.  Binary → ``✅ PASS`` / ``❌ FAIL``; graded → ``N/M checks``
    (M = the applicable scored checks — advisory + not-applicable excluded) with a ``· K n/a``
    tail when any check was not-applicable.  A passed-but-shaky sample (a rejected/recovered tool
    call in the run) gets a ``· fragile`` tail, so green-via-recovery reads distinctly from clean
    green."""
    if not result.checks:
        core = "✅ PASS" if result.passed else "❌ FAIL"
    else:
        passed_checks = round(result.score * result.total)
        core = f"{'✅' if result.passed else '❌'} {passed_checks}/{result.total} checks"
        na = sum(1 for check in result.checks if check.ignored)
        if na:
            core += f" · {na} n/a"
    return f"{core} · fragile" if fragile else core


def _legend_entry(
    check: Check, *, regressed: bool = False, baseline_run_id: str | None = None
) -> str:
    """One legend line for a check: its marker, label, and the observed-vs-expected rationale
    when one was given.  A regressed check (#1693) renders its REGRESSED marker and names the
    prior run it was green in, so the flip is self-explaining."""
    tail = f" — {check.rationale}" if check.rationale else ""
    note = f" (was passing in `{baseline_run_id}`)" if regressed and baseline_run_id else ""
    return f"{_check_mark(check, regressed=regressed)} {check.label}{tail}{note}"


def _report_legend(
    result: SampleResult,
    leftover: list[Check],
    regressed_ids: set[int],
    baseline: Baseline | None,
    case_id: str,
) -> str:
    """The checks legend below the transcript table.  Names every check that needs words — one
    with no row (a whole-run / missing-action check), a failed one, a not-applicable one, or any
    carrying a rationale — as ``<mark> <label> — <rationale>``.  A placed passing check with no
    rationale is a row-mark only, so a clean pass stays uncluttered.  Binary samples fall back to
    their failure strings."""
    if not result.checks:
        return f"**Failed:** {'; '.join(result.failed)}" if result.failed else ""
    leftover_ids = {id(check) for check in leftover}
    spelled = [
        check
        for check in result.checks
        if id(check) in leftover_ids or check.ignored or not check.ok or check.rationale
    ]
    if not spelled:
        return ""
    baseline_run_id = baseline.run_id_for(case_id) if baseline is not None else None
    entries = [
        _legend_entry(check, regressed=id(check) in regressed_ids, baseline_run_id=baseline_run_id)
        for check in spelled
    ]
    return f"_checks: {' · '.join(entries)}_"


def _thinking_by_call(rows: list[PromptLog]) -> dict[str, str]:
    """Map each tool-call turn's content (``name(args)``) to the thinking of the promptlog row
    whose RESPONSE emitted it — so a failed turn can show the model's own reasoning (#1693).

    The tool-call content string is byte-identical to what ``_sample_turns`` renders for that
    call (it appears verbatim in the next row's ``messages`` history), so a turn matches its
    producing row's thinking.  First non-empty thinking per content wins (turns de-dupe the same
    way); rows with no thinking or no tool call contribute nothing."""
    mapping: dict[str, str] = {}
    for row in rows:
        thinking = (row.thinking or "").strip()
        if not thinking:
            continue
        for call in _response_tool_calls(row):
            function = call.get("function", {})
            content = f"{function.get('name')}({function.get('arguments')})"
            mapping.setdefault(content, thinking)
    return mapping


def _thinking_block(
    turn_number: int, content: str, failing: list[Check], regressed_ids: set[int], thinking: str
) -> list[str]:
    """A collapsed ``<details>`` carrying the model's thinking at ONE failed/regressed tool-call
    turn (#1693).  The summary names the turn, its tool, and the failing checks' marks; the body
    is the verbatim thinking as a blockquote.  Passing turns never reach here, so the comment
    doesn't bloat."""
    tool = content.split("(", 1)[0]
    marks = " ".join(_check_mark(check, regressed=id(check) in regressed_ids) for check in failing)
    quoted = "\n".join(f"> {line}" if line else ">" for line in thinking.splitlines())
    return [
        "",
        f"<details><summary>💭 thinking · turn {turn_number} ({tool}) — {marks}</summary>",
        "",
        quoted,
        "",
        "</details>",
    ]


def _write_sample_report(
    db: Database, case_id: str, sample_index: int, *, result: SampleResult, reply: str = ""
) -> None:
    """Append one sample's verbatim transcript to ``EVAL_REPORT_DIR/<case_id>.md``.

    No-op unless ``EVAL_REPORT_DIR`` is set.  ``reply`` is the chat agent's final text
    (appended as the last turn); a collector run passes none — its ``done()`` / sends are
    already tool-call turns in the transcript.  The header shows PASS/FAIL for a binary
    sample, or ``N/M checks`` for a graded one, so the partial credit is visible."""
    report_dir = os.environ.get("EVAL_REPORT_DIR")
    if not report_dir:
        return
    with Session(db.engine) as session:
        rows = list(session.exec(select(PromptLog).order_by(PromptLog.timestamp.asc())).all())
    baseline = baseline_from_env()  # a prior run's results.jsonl → REGRESSED marks (#1693)
    regressed_ids = _regressed_ids(result.checks, baseline, case_id)
    fragile = result.passed and sample_is_fragile(db)
    verdict = _sample_verdict(result, fragile=fragile)
    lines = [
        f"#### sample {sample_index + 1} — {verdict}",
        "",
        "| # | Actor | Content |",
        "|---|---|---|",
    ]
    turns = _sample_turns(rows, reply)
    placed, leftover = _place_checks(result.checks, turns)  # checks onto the row each one tests
    for index, (actor, content) in enumerate(turns, start=1):
        lines.append(
            f"| {index} | {actor}{_row_mark(placed.get(index - 1, []), regressed_ids)} "
            f"| {_report_cell(content)} |"
        )
    legend = _report_legend(result, leftover, regressed_ids, baseline, case_id)
    if legend:
        lines += ["", legend]
    lines += _thinking_sections(turns, placed, regressed_ids, _thinking_by_call(rows))
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{case_id}.md").open("a") as handle:
        handle.write("\n".join(lines) + "\n\n")


def _thinking_sections(
    turns: list[tuple[str, str]],
    placed: dict[int, list[Check]],
    regressed_ids: set[int],
    thinking: dict[str, str],
) -> list[str]:
    """The collapsed thinking ``<details>`` blocks, one per FAILED/REGRESSED tool-call turn that
    has captured thinking (#1693).  Passing turns are omitted (no bloat); a failed turn with no
    thinking recorded contributes nothing.  Rendered after the checks legend, in turn order."""
    blocks: list[str] = []
    for index, (actor, content) in enumerate(turns, start=1):
        if actor != _ACTOR["call"]:
            continue
        failing = [c for c in placed.get(index - 1, []) if not c.ignored and not c.ok]
        note = thinking.get(content)
        if failing and note:
            blocks += _thinking_block(index, content, failing, regressed_ids, note)
    return blocks


# A chat-eval runner: (case_id, message, scorer, optional seeder) -> asserts threshold.
ChatEval = Callable[..., Awaitable[None]]


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


@pytest.fixture
def chat_eval(make_config: Callable[..., Config], tmp_path, request) -> ChatEval:
    """Drive the real chat flow N times for one user message (or a multi-turn
    conversation) and score each run.

    Each sample is fully hermetic — its own mock Signal server, DB, and
    real-model Penny: seed user (+ any case seed), embed the seeds, push the
    turn(s), wait for each reply, then score persisted state against the LAST
    reply.  A per-sample server is essential: a shared one leaks a prior
    sample's shut-down channel, which then errors on the next sample's
    broadcast.  A timeout on any turn counts as a failed sample, not a crash.

    Single-message vs. conversation: pass ``message`` for one turn, or
    ``messages`` for a discuss-then-adjust conversation (see
    ``_conversation_turns``).
    """

    async def _run(
        *,
        case_id: str,
        message: str | None = None,
        messages: Sequence[str] | None = None,
        score: Scorer,
        seed: Seeder | None = None,
        browse: list[CannedPage] | None = None,
        prepare: Preparer | None = None,
        wrap_client: Callable[[LlmClient], _InjectingClient] | None = None,
        samples: int = SAMPLES,
        min_pass_rate: float | None = 0.75,
        timeout: float = 120.0,
        family: str | None = None,
        gate_pathology_excluded: bool = False,
    ) -> None:
        eval_artifacts.begin_case(case_id)
        turns = _conversation_turns(message, messages)
        results: list[SampleResult] = []
        perf = _Perf()
        for sample_index in range(samples):
            server = MockSignalServer()
            await server.start()
            try:
                config = _real_model_config(
                    make_config,
                    signal_api_url=f"http://localhost:{server.port}",
                    db_path=_sample_db_path(tmp_path, case_id, sample_index),
                )
                async with run_penny_with_server(config, server) as penny:
                    seed_user(penny.db)
                    if seed is not None:
                        seed(penny.db)
                    await _embed_seeds(penny)
                    if browse is not None:
                        install_browse(penny, browse)
                    if prepare is not None:
                        prepare(penny)
                    # A recovery case wraps the chat agent's model client to force
                    # one bad response (e.g. a bracket-wrapped key) deterministically.
                    # Keep the wrapper: its ``bail_injected`` flag is the only proof
                    # the sabotage fired — the raw response is persisted inside the
                    # REAL client before the wrapper mutates it, so the promptlog
                    # never shows the injected form and can't be probed for it.
                    wrapper: _InjectingClient | None = None
                    if wrap_client is not None:
                        wrapper = wrap_client(penny.chat_agent._model_client)
                        penny.chat_agent._model_client = wrapper
                    before = collection_names(penny.db)
                    try:
                        reply = ""
                        for turn in turns:
                            await server.push_message(sender=TEST_SENDER, content=turn)
                            response = await server.wait_for_message(timeout=timeout)
                            reply = str(response.get("message", ""))
                        scored = list(score(penny.db, before, reply))
                        if _scorer_is_graded(scored):
                            guards: list[Check] = []
                            if wrapper is not None:
                                guards = [_bail_fired_check(wrapper.bail_injected)]
                            result = _guarded_graded(scored, guards)
                        else:
                            fails = [s for s in scored if isinstance(s, str)]  # binary scorer
                            if wrapper is not None and not wrapper.bail_injected:
                                fails.append("forced bail never fired — contract not exercised")
                            result = SampleResult.binary(fails)
                        results.append(result)
                        _stamp_cause(penny.db, result)
                        _write_sample_report(
                            penny.db, case_id, sample_index, result=result, reply=reply
                        )
                    except TimeoutError:
                        timed_out = SampleResult.binary(["no reply within timeout"])
                        _stamp_cause(penny.db, timed_out, timed_out=True)
                        results.append(timed_out)
                    _dump_thinking(penny.db, case_id, sample_index, failed=not results[-1].passed)
                    perf.add(penny.db.messages.prompt_perf())
            finally:
                await server.stop()
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
        )
        perf.report(case_id, samples)
        _assert_threshold(
            case_id, results, min_pass_rate, gate_pathology_excluded=gate_pathology_excluded
        )

    return _run


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
        results: list[SampleResult] = []
        perf = _Perf()
        for sample_index in range(samples):
            server = MockSignalServer()
            await server.start()
            try:
                config = _real_model_config(
                    make_config,
                    signal_api_url=f"http://localhost:{server.port}",
                    db_path=_sample_db_path(tmp_path, case_id, sample_index),
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
                    scored = list(score(penny.db, before, sent))
                    if _scorer_is_graded(scored):
                        result = _guarded_graded(scored, [])
                    else:
                        result = SampleResult.binary([s for s in scored if isinstance(s, str)])
                    results.append(result)
                    _stamp_cause(penny.db, result)
                    _write_sample_report(penny.db, case_id, sample_index, result=result)
                    _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
                    perf.add(penny.db.messages.prompt_perf())
            finally:
                await server.stop()
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
        )
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
        results: list[SampleResult] = []
        perf = _Perf()
        for sample_index in range(samples):
            server = MockSignalServer()
            await server.start()
            try:
                config = _real_model_config(
                    make_config,
                    signal_api_url=f"http://localhost:{server.port}",
                    db_path=_sample_db_path(tmp_path, case_id, sample_index),
                )
                async with run_penny_with_server(config, server) as penny:
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
                            fails.append(
                                "cycle did not recover to a successful close after the nudge"
                            )
                        result = SampleResult.binary(fails)
                    results.append(result)
                    _stamp_cause(penny.db, result)
                    _write_sample_report(penny.db, case_id, sample_index, result=result)
                    _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
                    perf.add(penny.db.messages.prompt_perf())
            finally:
                await server.stop()
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
        )
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate)

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
        results: list[SampleResult] = []
        perf = _Perf()
        for sample_index in range(samples):
            server = MockSignalServer()
            await server.start()
            try:
                config = _real_model_config(
                    make_config,
                    signal_api_url=f"http://localhost:{server.port}",
                    db_path=_sample_db_path(tmp_path, case_id, sample_index),
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
                    scored = list(score(penny.db, sent))
                    if _scorer_is_graded(scored):
                        result = _guarded_graded(scored, [_bail_fired_check(wrapper.bail_injected)])
                    else:
                        fails = [s for s in scored if isinstance(s, str)]
                        if not wrapper.bail_injected:
                            fails.append("forced bail never fired — contract not exercised")
                        result = SampleResult.binary(fails)
                    results.append(result)
                    _stamp_cause(penny.db, result)
                    _write_sample_report(penny.db, case_id, sample_index, result=result)
                    _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
                    perf.add(penny.db.messages.prompt_perf())
            finally:
                await server.stop()
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
        )
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate)

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
        results: list[SampleResult] = []
        perf = _Perf()
        for sample_index in range(samples):
            server = MockSignalServer()
            await server.start()
            try:
                config = _real_model_config(
                    make_config,
                    signal_api_url=f"http://localhost:{server.port}",
                    db_path=_sample_db_path(tmp_path, case_id, sample_index),
                )
                async with run_penny_with_server(config, server) as penny:
                    seed_user(penny.db)
                    prior = os.environ.get("GIT_COMMIT_MESSAGE")
                    os.environ["GIT_COMMIT_MESSAGE"] = commit_message
                    try:
                        announcement = await get_restart_message(penny.db, penny.model_client)
                    finally:
                        if prior is None:
                            os.environ.pop("GIT_COMMIT_MESSAGE", None)
                        else:
                            os.environ["GIT_COMMIT_MESSAGE"] = prior
                    # Same graded/binary dispatch as the other runners.  Startup has no
                    # injection (no wrapper, no framework guard), so a graded return grades
                    # over the scorer's own Checks with an empty guard list.
                    scored = list(score(announcement))
                    if _scorer_is_graded(scored):
                        result = _guarded_graded(scored, [])
                    else:
                        result = SampleResult.binary([s for s in scored if isinstance(s, str)])
                    results.append(result)
                    _stamp_cause(penny.db, result)
                    perf.add(penny.db.messages.prompt_perf())
            finally:
                await server.stop()
        eval_artifacts.record_case(
            case_id=case_id,
            family=family,
            module=request.module.__name__,
            results=results,
            perf=perf,
        )
        perf.report(case_id, samples)
        _assert_threshold(case_id, results, min_pass_rate)

    return _run
