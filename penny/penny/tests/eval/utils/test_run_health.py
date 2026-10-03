"""What a run says about itself — the cohort bar, the fault tally, and the block (#1996).

Every case here is one of the four silences the ticket names, driven deterministically:
no live model, no GPU, no network. The tally is proven against records the REAL
``LlmClient`` emits, because a handler that reads fields nothing writes is a tally that
reports a healthy run forever.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from penny.database.message_store import PromptPerf
from penny.llm.client import LlmClient
from penny.llm.models import LlmFault, LlmResponseError, LlmTimeoutError
from penny.tests import conftest as root_conftest
from penny.tests.eval import conftest as eval_conftest
from penny.tests.eval.utils import cohort as eval_cohort
from penny.tests.eval.utils import report, run_health
from penny.tests.eval.utils.artifacts import worker_filename
from penny.tests.eval.utils.catalogue import CATALOGUE_ENV, Area, CatalogueEntry, Layer, load
from penny.tests.eval.utils.run_health import (
    CohortRecord,
    ProviderTally,
    RunHealth,
    cohort_is_viable,
    results_needed,
)


class _NoChoices:
    """A gateway answering 200 with an error payload where a completion belongs.

    The exact shape that killed 34 of 48 samples: the SDK parses it into a model whose
    ``choices`` is None, and ``model_extra`` carries both the provider's reason and — for
    a routing gateway — the name of the upstream that produced it.
    """

    choices = None
    model_extra = {
        "provider": "some-upstream",
        "error": {"message": "upstream provider returned an error"},
    }


def _rate_limited() -> openai.APIStatusError:
    """A real 429 the way the SDK raises it."""
    request = httpx.Request("POST", "https://gateway.example/api/v1/chat/completions")
    response = httpx.Response(429, request=request, json={"error": {"message": "slow down"}})
    return openai.RateLimitError("Error code: 429", response=response, body=None)


class TestTheCohortBar:
    """More than half of a case's intended samples must have run — and the bar SAYS so."""

    @pytest.mark.parametrize(
        ("completed", "intended", "viable"),
        [
            (5, 5, True),
            (3, 5, True),  # the bar itself: a strict majority clears it
            (2, 5, False),  # one short is one too few
            (0, 5, False),
            (2, 4, False),  # exactly half is NOT a majority
            (3, 4, True),
            (1, 1, True),
            (0, 1, False),
            (0, 0, True),  # a run that intended nothing is vacuously fine
        ],
    )
    def test_the_rule_is_a_strict_majority(
        self, completed: int, intended: int, viable: bool
    ) -> None:
        assert cohort_is_viable(completed, intended) is viable

    def test_the_bar_states_the_count_it_needs(self) -> None:
        """The refusal names a number, so nobody has to re-derive the rule from prose."""
        assert results_needed(5) == 3
        assert results_needed(4) == 3
        assert results_needed(1) == 1


class TestTheFaultTallyReadsWhatTheClientWrites:
    """The tally counts VALUES off the attempt record — never a phrase it recognised."""

    @pytest.mark.asyncio
    async def test_an_empty_choices_burst_is_counted_and_attributed(self, monkeypatch) -> None:
        """188 identical no-choices responses read as one class with a count and a culprit.

        Both halves are the point: the CLASS is what makes a run's dominant failure legible,
        and the PROVIDER is what separates "this model is broken" from "one member of the
        routing pool is" — which previously cost a whole second run with a pin to establish.
        """
        tally = run_health.FaultTally()
        logging.getLogger(run_health.PENNY_LOGGER).addHandler(tally)

        async def answer_without_a_completion(**kwargs):
            return _NoChoices()

        client = LlmClient(
            api_url="https://gateway.example/api", model="m", max_retries=3, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", answer_without_a_completion)
        try:
            with pytest.raises(LlmResponseError):
                await client.chat([{"role": "user", "content": "hi"}])
        finally:
            await client.close()
            logging.getLogger(run_health.PENNY_LOGGER).removeHandler(tally)

        assert tally.calls == 3
        assert tally.faults == {LlmFault.NO_CHOICES: 3}
        assert tally.providers["some-upstream"].faults == {LlmFault.NO_CHOICES: 3}

    @pytest.mark.asyncio
    async def test_a_rate_limited_burst_is_its_own_class(self, monkeypatch) -> None:
        """325 429s and 325 empty responses are two different runs, and now read as two."""
        tally = run_health.FaultTally()
        logging.getLogger(run_health.PENNY_LOGGER).addHandler(tally)

        async def refuse(**kwargs):
            raise _rate_limited()

        client = LlmClient(
            api_url="https://gateway.example/api", model="m", max_retries=2, retry_delay=0.0
        )
        monkeypatch.setattr(client.client.chat.completions, "create", refuse)
        try:
            with pytest.raises(LlmResponseError):
                await client.chat([{"role": "user", "content": "hi"}])
        finally:
            await client.close()
            logging.getLogger(run_health.PENNY_LOGGER).removeHandler(tally)

        assert tally.faults == {LlmFault.RATE_LIMITED: 2}
        # A failure carries no completion, so no upstream names itself — said plainly
        # rather than guessed at.
        assert set(tally.providers) == {"unreported"}

    def test_a_record_that_is_not_an_attempt_is_not_counted(self) -> None:
        """Every penny module logs through this logger; only attempts carry the fields."""
        tally = run_health.FaultTally()
        logger = logging.getLogger(run_health.PENNY_LOGGER)
        logger.addHandler(tally)
        try:
            logger.warning("something else entirely happened")
        finally:
            logger.removeHandler(tally)

        assert tally.calls == 0


_HEALTHY = RunHealth(
    cohorts=[
        CohortRecord(case_id="chat-reply", intended=5, completed=5),
        CohortRecord(case_id="chat-browse", intended=5, completed=5),
    ],
    calls=612,
    providers={"cloudflare": ProviderTally(calls=612)},
)

_DEGRADED = RunHealth(
    cohorts=[
        CohortRecord(case_id="chat-reply", intended=5, completed=1),
        CohortRecord(case_id="chat-browse", intended=5, completed=4, retried_boots=2),
    ],
    calls=402,
    faults={LlmFault.NO_CHOICES: 188, LlmFault.RATE_LIMITED: 3},
    providers={
        "google-vertex": ProviderTally(
            calls=210, faults={LlmFault.NO_CHOICES: 188, LlmFault.RATE_LIMITED: 3}
        ),
        "cloudflare": ProviderTally(calls=192),
    },
)


class TestTheBlockARunPrintsAboutItself:
    """Whole-render, because what a reader sees IS the deliverable here."""

    def test_a_healthy_run_says_so_in_full(self) -> None:
        assert _HEALTHY.render() == (
            "samples: 10 of 10 completed · 0 dead "
            "(completed = the sample's measured turn ran and was scored)\n"
            "cases: 2 of 2 readable — a case must complete more than half the samples "
            "it intended\n"
            "model calls: 612 attempts · 0 faulted\n"
            "providers:\n"
            "  cloudflare: 612 calls — no faults\n"
            "verdict: this run is a result."
        )

    def test_a_degraded_run_names_the_dead_cases_and_blames_the_dominant_class(self) -> None:
        assert _DEGRADED.render() == (
            "samples: 5 of 10 completed · 5 dead "
            "(completed = the sample's measured turn ran and was scored)\n"
            "  2 boot(s) retried after a transient preflight fault — chat-browse 2\n"
            "cases: 1 of 2 readable — a case must complete more than half the samples "
            "it intended\n"
            "  not readable: chat-reply 1/5\n"
            "model calls: 402 attempts · 191 faulted\n"
            "  no choices 188 · 429 3\n"
            "providers:\n"
            "  google-vertex: 210 calls — no choices 188 · 429 3\n"
            "  cloudflare: 192 calls — no faults\n"
            "REFUSED: 1 case(s) scored a fraction of their intended cohort — mostly "
            "no choices (188). A result computed from a fraction of the cohort is not a result."
        )

    def test_the_verdict_is_what_moves_the_exit_status(self) -> None:
        """A run reporting `6 passed` over a mostly-dead cohort must not be viable."""
        assert _HEALTHY.viable
        assert not _DEGRADED.viable
        assert _DEGRADED.dominant_fault == (LlmFault.NO_CHOICES, 188)
        assert _HEALTHY.dominant_fault is None

    def test_an_endpoint_that_names_no_upstream_says_that_rather_than_nothing(self) -> None:
        """A direct endpoint reports no provider, and the block must not read as a gap."""
        bare = RunHealth(cohorts=[CohortRecord(case_id="c", intended=1, completed=1)], calls=4)
        assert "providers: none reported by this endpoint" in bare.render()


class TestMergingWhatSeveralProcessesSaw:
    """Under xdist the cases are spread across processes; the run is their sum.

    Health is not its own filing scheme — it rides the per-worker JSONL convention the
    per-case results already use, so these pin the shared naming rather than a second
    copy of it.
    """

    def test_worker_records_round_trip_and_add_up(self, tmp_path) -> None:
        first = RunHealth(
            cohorts=[CohortRecord(case_id="a", intended=5, completed=5)],
            calls=100,
            faults={LlmFault.NO_CHOICES: 4},
            providers={"vertex": ProviderTally(calls=100, faults={LlmFault.NO_CHOICES: 4})},
        )
        second = RunHealth(
            cohorts=[CohortRecord(case_id="b", intended=5, completed=1)],
            calls=60,
            faults={LlmFault.NO_CHOICES: 6, LlmFault.TIMEOUT: 1},
            providers={"vertex": ProviderTally(calls=60, faults={LlmFault.NO_CHOICES: 6})},
        )
        run_health.write_health(tmp_path, "gw0", first)
        run_health.write_health(tmp_path, "gw1", second)

        # The SAME naming the results records use — one convention, two customers.
        assert (tmp_path / worker_filename(run_health.HEALTH_STEM, "gw0")).exists()
        assert (tmp_path / "health-gw1.jsonl").exists()

        merged = run_health.load_health(tmp_path)

        assert [cohort.case_id for cohort in merged.cohorts] == ["a", "b"]
        assert merged.calls == 160
        assert merged.faults == {LlmFault.NO_CHOICES: 10, LlmFault.TIMEOUT: 1}
        assert merged.providers["vertex"].calls == 160
        assert merged.providers["vertex"].faults == {LlmFault.NO_CHOICES: 10}
        # One dead case in one worker is enough to refuse the whole run.
        assert not merged.viable

    def test_a_single_process_run_writes_the_plain_name(self, tmp_path) -> None:
        """No xdist, no suffix — the unsuffixed name the convention gives every record."""
        run_health.write_health(tmp_path, None, _HEALTHY)
        assert (tmp_path / worker_filename(run_health.HEALTH_STEM, None)).exists()
        assert (tmp_path / "health.jsonl").exists()
        assert run_health.load_health(tmp_path).completed == 10

    def test_an_empty_dir_reads_as_a_run_that_measured_nothing(self, tmp_path) -> None:
        assert run_health.load_health(tmp_path) == RunHealth()


# ── The sample the rig never started (#2070) ─────────────────────────────────
#
# A boot failure escaped ``asyncio.gather``, so ONE sample's preflight timing out
# discarded the other fourteen AND skipped ``record_cohort`` — the case died in the one
# way the viability gate could not see, its health row never written at all. These drive
# the real ``_run_samples`` skeleton every runner shares, with the world stood up by a
# stub: no model, no GPU, no network.
_VOIDED_SAMPLE = 7
_STAND_UP_FAULT = "the sample never started — RuntimeError while standing its world up"
_PREFLIGHT_FAULT = "the sample never started — PreflightError while standing its world up"
_EMBEDDING_MODEL = "test-embedding-model"
# Patched in for the wall-clock bound: long past a stubbed sample's whole drive, short enough
# that the stuck one costs the test a second at most.
_BOUND_SECONDS = 0.5
_OVERRAN = "the sample never finished — stopped at its 0.5s wall-clock bound"
# How long a test that must not hang may run before it fails instead.  It never fires on a
# passing run; it turns a regression into a failure rather than a silent CI stall.
_GUARD_SECONDS = 60.0


def _stubbed_world(monkeypatch, *, stands_up: bool = True) -> None:
    """Stand every sample's Penny up without a model — or refuse to, as the incident did."""

    @asynccontextmanager
    async def _penny(config, server):
        if not stands_up:
            raise RuntimeError("LLM endpoint unreachable: Request timed out")
        yield SimpleNamespace(db=None)

    monkeypatch.setattr(eval_conftest, "eval_penny", _penny)
    monkeypatch.setattr(
        eval_conftest, "live_prompt_perf", lambda _db: PromptPerf(1, 10, 2, 3, 0, 0, 0)
    )
    _isolated_health(monkeypatch)


def _boot_through_the_real_preflight(
    monkeypatch, faults: Sequence[Exception], *, embedding_listed: bool = True
) -> list[str]:
    """Stand samples up through Penny's REAL boot and preflight, the endpoint stubbed at the
    client boundary: each boot's embedding-model listing raises the next of ``faults`` and,
    once they run out, answers — listing the embedding model or not.  Returns one entry per
    boot that reached that listing, so a test can count the boots."""
    _isolated_health(monkeypatch)
    monkeypatch.setattr(eval_conftest, "SAMPLE_READY_TIMEOUT_SECONDS", 3600.0)
    monkeypatch.setattr(eval_conftest, "SAMPLE_BOOT_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(eval_conftest, "EVAL_CONCURRENCY", 1)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_EMBEDDING_MODEL", _EMBEDDING_MODEL)
    pending = iter(faults)
    boots: list[str] = []

    async def _list_models(client: LlmClient) -> list[str]:
        if client.model != _EMBEDDING_MODEL:
            return [client.model]
        boots.append(client.model)
        if (fault := next(pending, None)) is not None:
            raise fault
        return [client.model] if embedding_listed else []

    monkeypatch.setattr(LlmClient, "list_models", _list_models)
    return boots


def _isolated_health(monkeypatch) -> None:
    """A run-health tally of this test's own, and no report directory to write into."""
    monkeypatch.setattr(run_health, "_cohorts", [])
    monkeypatch.delenv("EVAL_REPORT_DIR", raising=False)


def _pooled(case_id: str, sample_index: int) -> eval_conftest.SampleResult:
    """A sample that ran and was observed — what every healthy sample hands back."""
    return eval_conftest.SampleResult(
        1.0,
        [],
        1,
        observation=eval_cohort.SampleObservation(
            name=f"{case_id}-{eval_conftest.sample_number(sample_index)}", phrasing="the ask"
        ),
    )


class TestASampleTheRigNeverStarted:
    """One sample's boot failure voids that sample, and NOTHING else."""

    @pytest.mark.asyncio
    async def test_the_others_pool_and_the_void_is_named(self, make_config, tmp_path, monkeypatch):
        """The whole chain, on the numbers the incident lost: 14 pooled, one named void."""
        _stubbed_world(monkeypatch)

        async def _drive(penny, server, sample_index, retryable):
            if sample_index == _VOIDED_SAMPLE:
                raise RuntimeError("LLM endpoint unreachable: Request timed out")
            return _pooled("boot-failure", sample_index)

        results, perf, voided = await eval_conftest._run_samples(
            make_config, tmp_path, case_id="boot-failure", samples=15, drive=_drive
        )

        assert len(results) == 14
        # The void carries the exception CLASS, which is what groups two samples killed by
        # the same thing — a message carrying a URL or a timeout figure differs per sample.
        assert [(s.name, s.complete, s.exclusion) for s in voided] == [
            ("boot-failure-8", False, _STAND_UP_FAULT)
        ]
        # The cohort the viability gate rules on: the case asked for 15 and got 14.
        assert run_health.process_health().cohorts == [
            CohortRecord(case_id="boot-failure", intended=15, completed=14)
        ]
        # A sample that never started spent nothing, so the per-sample cost stays honest.
        assert perf.calls == 14

        cohort = eval_conftest._driven_cohort("boot-failure", "test-model", results, voided, ())
        pooled = eval_cohort.pool(cohort.samples, cohort.features)
        assert (pooled.pooled, len(pooled.excluded), pooled.driven) == (14, 1, 15)

        document = report.CaseSections(case_id="boot-failure", variance=pooled).render()
        assert (
            f"#### 🔴 Excluded samples\n"
            f"\n"
            f"<details><summary>1 of 15 · dominant: {_STAND_UP_FAULT}</summary>\n"
            f"\n"
            f"Dominant failure class: **{_STAND_UP_FAULT}** (1 of 1).\n"
            f"\n"
            f"- `boot-failure-8` — {_STAND_UP_FAULT}\n"
            f"\n"
            f"</details>"
        ) in document

    @pytest.mark.asyncio
    async def test_a_run_that_lost_every_sample_this_way_is_refused(
        self, make_config, tmp_path, monkeypatch
    ):
        """A failure standing the world up — the door the incident came through — voids the
        sample the same way, and a cohort that lost all of them REFUSES the run."""
        _stubbed_world(monkeypatch, stands_up=False)

        async def _drive(penny, server, sample_index, retryable):
            raise AssertionError("the drive is unreachable when the world never stands up")

        results, _perf, voided = await eval_conftest._run_samples(
            make_config, tmp_path, case_id="dead-cohort", samples=3, drive=_drive
        )

        assert results == []
        assert [sample.exclusion for sample in voided] == [_STAND_UP_FAULT] * 3
        health = run_health.process_health()
        assert health.cohorts == [CohortRecord(case_id="dead-cohort", intended=3, completed=0)]
        assert not health.viable
        assert "REFUSED: 1 case(s) scored a fraction of their intended cohort" in health.render()

    @pytest.mark.asyncio
    async def test_a_preflight_that_times_out_once_is_booted_again_from_a_fresh_world(
        self, mock_llm, make_config, tmp_path, monkeypatch
    ):
        """The incident's door through the REAL boot path (#2168/#2230): the first sample's
        embedding endpoint times out ONCE at preflight, so its Penny's run ENDS before its
        channel connects and the readiness wait raises what ended it.  A timeout is a moment,
        not a verdict, so the sample is torn down and booted again — onto a database of its
        own, as hermetic as a first boot — and is driven like its siblings; the case loses
        nothing, and the re-boot is counted where the health block reads it.  The readiness
        budget is one no test would outlive, so a wait that ignored the dead run would be
        caught by the guard rather than passing slowly."""
        boots = _boot_through_the_real_preflight(monkeypatch, [LlmTimeoutError("timed out")])
        driven: list[str] = []

        async def _drive(penny, server, sample_index, retryable):
            driven.append(Path(penny.config.db_path).name)
            return _pooled("preflight", sample_index)

        async with asyncio.timeout(_GUARD_SECONDS):
            results, _perf, voided = await eval_conftest._run_samples(
                make_config, tmp_path, case_id="preflight", samples=3, drive=_drive
            )

        assert (len(results), voided) == (3, [])
        assert len(boots) == 4
        assert driven == ["preflight-1-attempt2.db", "preflight-2.db", "preflight-3.db"]
        assert run_health.process_health().cohorts == [
            CohortRecord(case_id="preflight", intended=3, completed=3, retried_boots=1)
        ]

    @pytest.mark.asyncio
    async def test_a_preflight_that_times_out_every_time_voids_after_the_bound(
        self, mock_llm, make_config, tmp_path, monkeypatch
    ):
        """A provider that cannot answer a model listing on any boot is not having a moment:
        the sample is booted :data:`SAMPLE_BOOT_ATTEMPTS` times and then voided by name, as a
        sample that never started, with every re-boot still on the record."""
        timeouts = [LlmTimeoutError("timed out")] * eval_conftest.SAMPLE_BOOT_ATTEMPTS
        boots = _boot_through_the_real_preflight(monkeypatch, timeouts)

        async def _drive(penny, server, sample_index, retryable):
            raise AssertionError("the drive is unreachable when no boot passes its preflight")

        async with asyncio.timeout(_GUARD_SECONDS):
            results, _perf, voided = await eval_conftest._run_samples(
                make_config, tmp_path, case_id="preflight", samples=1, drive=_drive
            )

        assert results == []
        assert len(boots) == eval_conftest.SAMPLE_BOOT_ATTEMPTS
        assert [(s.name, s.phrasing, s.complete, s.exclusion) for s in voided] == [
            ("preflight-1", eval_conftest.NEVER_SPOKEN, False, _PREFLIGHT_FAULT)
        ]
        assert run_health.process_health().cohorts == [
            CohortRecord(case_id="preflight", intended=1, completed=0, retried_boots=2)
        ]

    @pytest.mark.asyncio
    async def test_a_preflight_refused_on_its_credentials_voids_at_once(
        self, mock_llm, make_config, tmp_path, monkeypatch
    ):
        """A refused request is a verdict: another boot would meet the same refusal.  The
        embedding model is absent from the listing and the second listing answers 401, so the
        check fails carrying a client-error fault — and the sample is voided on its FIRST boot,
        with nothing retried."""
        boots = _boot_through_the_real_preflight(monkeypatch, [], embedding_listed=False)
        refusal = LlmResponseError("HTTP 401: invalid key", fault=LlmFault.CLIENT_ERROR)

        async def _refuse(client: LlmClient) -> list[str]:
            raise refusal

        monkeypatch.setattr(LlmClient, "list_embedding_models", _refuse)

        async def _drive(penny, server, sample_index, retryable):
            raise AssertionError("the drive is unreachable when the preflight refused")

        async with asyncio.timeout(_GUARD_SECONDS):
            results, _perf, voided = await eval_conftest._run_samples(
                make_config, tmp_path, case_id="preflight", samples=1, drive=_drive
            )

        assert (results, len(boots)) == ([], 1)
        assert [sample.exclusion for sample in voided] == [_PREFLIGHT_FAULT]
        assert run_health.process_health().cohorts == [
            CohortRecord(case_id="preflight", intended=1, completed=0)
        ]

    @pytest.mark.asyncio
    async def test_a_sample_that_never_returns_is_stopped_at_its_bound(
        self, make_config, tmp_path, monkeypatch
    ):
        """A drive that never returns — and that ABSORBS the first cancel sent to stop it, as
        the incident's in-flight HTTP call did — is stopped at the wall-clock bound, for
        certain, and voided by name; the other fourteen pool and the case closes (#2168)."""
        _stubbed_world(monkeypatch)
        monkeypatch.setattr(eval_conftest, "SAMPLE_WALL_CLOCK_SECONDS", _BOUND_SECONDS)
        monkeypatch.setattr(root_conftest, "TASK_STOP_POLL_SECONDS", 0.1)
        stopped: list[int] = []

        async def _drive(penny, server, sample_index, retryable):
            if sample_index == _VOIDED_SAMPLE:
                try:
                    with suppress(asyncio.CancelledError):
                        await asyncio.Event().wait()
                    await asyncio.Event().wait()
                finally:
                    stopped.append(sample_index)
            return _pooled("stuck", sample_index)

        results, perf, voided = await eval_conftest._run_samples(
            make_config, tmp_path, case_id="stuck", samples=15, drive=_drive
        )

        assert len(results) == 14
        assert [(s.name, s.phrasing, s.complete, s.exclusion) for s in voided] == [
            ("stuck-8", eval_conftest.NEVER_FINISHED, False, _OVERRAN)
        ]
        # Stopped, not abandoned: the cancel it absorbed was re-delivered until it landed.
        assert stopped == [_VOIDED_SAMPLE]
        assert run_health.process_health().cohorts == [
            CohortRecord(case_id="stuck", intended=15, completed=14)
        ]
        assert perf.calls == 14


# ── A driver refuses a case it cannot drive as a cohort ──────────────────────
#
# Through the REAL driver fixtures rather than their guards alone, so what is pinned is that
# the refusal reaches a case author: by the case's own name, and before a sample is stood up.

_SENTENCE = "In the chat agent, when asked, Penny answers."


async def _no_sample_may_start(*args, **kwargs):
    raise AssertionError("a refused case must not stand a sample up")


class TestADriverRefusesACaseItCannotDriveAsACohort:
    async def test_a_case_naming_nothing_to_drive_is_refused_by_name(
        self,
        chat_eval,
        collector_cycles_eval,
        classifier_eval,
        framer_eval,
        labeller_eval,
        binder_eval,
        extractor_eval,
        monkeypatch,
    ):
        """Every driver, handed a case with no ask, raises before a sample is stood up."""
        monkeypatch.setattr(eval_conftest, "_run_samples", _no_sample_may_start)
        stated = {"case_id": "no-ask", "behaviour": _SENTENCE, "min_pass_rate": None}
        idle = eval_conftest.ConversationState.IDLE
        refused = {
            "chat_eval": chat_eval(**stated),
            "collector_cycles_eval": collector_cycles_eval(collection="a-job", **stated),
            "classifier_eval": classifier_eval(state=idle, **stated),
            "framer_eval": framer_eval(turns=(), also_phrased=(), **stated),
            "labeller_eval": labeller_eval(
                utterance=" ", calls=(), target="a-job", also_demonstrated=(), **stated
            ),
            "binder_eval": binder_eval(
                turns=(), skill="a_routine", intent="x", parameters=(), also_phrased=(), **stated
            ),
            "extractor_eval": extractor_eval(url="u", page="p", instruction="", **stated),
        }
        for driver, drive in refused.items():
            with pytest.raises(ValueError, match=f"no-ask: {driver} needs "):
                await drive

    async def test_a_case_must_state_the_report_only_setting_and_nothing_else(
        self, chat_eval, monkeypatch
    ):
        """An unstated ``min_pass_rate`` and a stated floor are both refused by the case's name."""
        monkeypatch.setattr(eval_conftest, "_run_samples", _no_sample_may_start)
        with pytest.raises(ValueError, match="unstated: a case must state min_pass_rate=None"):
            await chat_eval(case_id="unstated", behaviour=_SENTENCE, ask="what does it cost?")
        with pytest.raises(ValueError, match="floored: min_pass_rate=0.8"):
            await chat_eval(
                case_id="floored", behaviour=_SENTENCE, ask="what does it cost?", min_pass_rate=0.8
            )

    async def test_a_case_declaring_no_area_is_refused_by_name(self, every_driver, monkeypatch):
        """Every driver, handed a case it could drive but with no area, raises before a sample
        is stood up — and so does one handed a string where a member of the closed set goes."""
        monkeypatch.setattr(eval_conftest, "_run_samples", _no_sample_may_start)
        for driver, drive in every_driver.items():
            with pytest.raises(ValueError, match=f"{driver}: a case must declare its area"):
                await drive(case_id=driver, behaviour=_SENTENCE, min_pass_rate=None)
            with pytest.raises(ValueError, match=f"{driver}: a case must declare its area"):
                await drive(case_id=driver, behaviour=_SENTENCE, min_pass_rate=None, area="memory")

    async def test_a_case_must_state_its_behaviour_and_name_only_an_edge_the_machine_has(
        self, every_driver, monkeypatch
    ):
        """A blank behaviour sentence and an edge outside ``OUT_EDGES`` are both refused by the
        case's name."""
        monkeypatch.setattr(eval_conftest, "_run_samples", _no_sample_may_start)
        drive = every_driver["chat_eval"]
        machine = {"min_pass_rate": None, "area": Area.CONVERSATION_MACHINE}
        with pytest.raises(ValueError, match="blank: a case must state the behaviour it checks"):
            await drive(case_id="blank", behaviour="  ", **machine)
        unreachable = "from request it offers apply, elicit, idle"
        with pytest.raises(ValueError, match=rf"parked: edge=\(request, request\).*{unreachable}"):
            await drive(case_id="parked", behaviour=_SENTENCE, edge=(_REQUEST, _REQUEST), **machine)
        with pytest.raises(ValueError, match=r"done: edge=\(apply, idle\).*it offers nothing"):
            await drive(case_id="done", behaviour=_SENTENCE, edge=(_APPLY, _IDLE), **machine)

    async def test_catalogue_mode_records_the_case_and_stops_before_any_sample(
        self, every_driver, monkeypatch, tmp_path, request
    ):
        """With ``EVAL_CATALOGUE`` set, every driver appends its case to that file and skips —
        before the run is resolved and before a sample is stood up — and the entry carries the
        layer of the driver it came through and the first of the case's wordings."""
        gathered = tmp_path / "entries.jsonl"
        monkeypatch.setenv(CATALOGUE_ENV, str(gathered))
        monkeypatch.setattr(eval_conftest, "_run_samples", _no_sample_may_start)
        monkeypatch.setattr(eval_conftest.eval_artifacts, "begin_case", _no_sample_may_start)
        edge = (_IDLE, _ELICIT)
        for driver, drive in every_driver.items():
            with pytest.raises(pytest.skip.Exception, match="catalogued"):
                await drive(
                    case_id=driver,
                    behaviour=f"  {_SENTENCE}  ",
                    min_pass_rate=None,
                    area=Area.MEMORY,
                    edge=edge,
                )
        assert load(gathered) == [
            CatalogueEntry(
                case_id=driver,
                area=Area.MEMORY,
                edge=edge,
                layer=layer,
                sentence=_SENTENCE,
                wording=wording,
                node_id=request.node.nodeid,
            )
            for driver, layer, wording in _CATALOGUED
        ]

    @pytest.fixture
    def every_driver(
        self,
        chat_eval,
        collector_cycles_eval,
        classifier_eval,
        framer_eval,
        labeller_eval,
        binder_eval,
        extractor_eval,
    ) -> dict[str, Callable[..., Awaitable[object]]]:
        """Each driver, already handed something it could drive — so what a test leaves out is
        the only thing a refusal can be about."""
        arm = eval_conftest.CycleArm(
            text=_INSTRUCTION, seed=lambda db: None, pages=[], world=eval_conftest._NO_WORLD
        )
        routine = {"skill": "a_routine", "intent": "x", "parameters": ()}
        return {
            "chat_eval": partial(chat_eval, ask=_ASK),
            "collector_cycles_eval": partial(collector_cycles_eval, collection="a-job", arms=[arm]),
            "classifier_eval": partial(classifier_eval, state=_IDLE, ask=_ASK),
            "framer_eval": partial(framer_eval, turns=_TURNS, also_phrased=()),
            "labeller_eval": partial(
                labeller_eval, utterance=_ASK, calls=(), target="a-job", also_demonstrated=()
            ),
            "binder_eval": partial(binder_eval, turns=_TURNS, also_phrased=(), **routine),
            "extractor_eval": partial(extractor_eval, url="u", page="p", instruction=_INSTRUCTION),
        }


# What each driver is handed above, and what catalogue mode records of it: the driver's own
# layer, and the first wording — an ask of several turns written on one line.
_ASK = "what does it cost?"
_TURNS = ("watch the listing", "tell me when it moves")
_INSTRUCTION = "the price on the page"
_IDLE = eval_conftest.ConversationState.IDLE
_ELICIT = eval_conftest.ConversationState.ELICIT
_REQUEST = eval_conftest.ConversationState.REQUEST
_APPLY = eval_conftest.ConversationState.APPLY
_CATALOGUED = (
    ("chat_eval", Layer.WHOLE_TURN, _ASK),
    ("collector_cycles_eval", Layer.COLLECTOR_CYCLE, _INSTRUCTION),
    ("classifier_eval", Layer.CLASSIFIER_DRAW, _ASK),
    ("framer_eval", Layer.MICRO_CONTEXT, "watch the listing / tell me when it moves"),
    ("labeller_eval", Layer.MICRO_CONTEXT, _ASK),
    ("binder_eval", Layer.MICRO_CONTEXT, "watch the listing / tell me when it moves"),
    ("extractor_eval", Layer.MICRO_CONTEXT, _INSTRUCTION),
)
