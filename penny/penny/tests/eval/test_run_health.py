"""What a run says about itself — the cohort bar, the fault tally, and the block (#1996).

Every case here is one of the four silences the ticket names, driven deterministically:
no live model, no GPU, no network. The tally is proven against records the REAL
``LlmClient`` emits, because a handler that reads fields nothing writes is a tally that
reports a healthy run forever.
"""

from __future__ import annotations

import logging

import httpx
import openai
import pytest

from penny.llm.client import LlmClient
from penny.llm.models import LlmFault, LlmResponseError
from penny.tests.eval import run_health
from penny.tests.eval.run_health import (
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
        CohortRecord(case_id="chat-browse", intended=5, completed=4),
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
    """Under xdist the cases are spread across processes; the run is their sum."""

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

        merged = run_health.load_health(tmp_path)

        assert [cohort.case_id for cohort in merged.cohorts] == ["a", "b"]
        assert merged.calls == 160
        assert merged.faults == {LlmFault.NO_CHOICES: 10, LlmFault.TIMEOUT: 1}
        assert merged.providers["vertex"].calls == 160
        assert merged.providers["vertex"].faults == {LlmFault.NO_CHOICES: 10}
        # One dead case in one worker is enough to refuse the whole run.
        assert not merged.viable

    def test_a_single_process_run_writes_the_plain_name(self, tmp_path) -> None:
        run_health.write_health(tmp_path, None, _HEALTHY)
        assert (tmp_path / run_health.SINGLE_PROCESS_HEALTH_FILENAME).exists()
        assert run_health.load_health(tmp_path).completed == 10

    def test_an_empty_dir_reads_as_a_run_that_measured_nothing(self, tmp_path) -> None:
        assert run_health.load_health(tmp_path) == RunHealth()
