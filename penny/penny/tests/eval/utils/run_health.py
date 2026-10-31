"""How much of the run actually ran, and what killed the rest.

A degraded run used to look exactly like a healthy one.  ``6 passed, EXIT=0`` was printed
by a run in which 34 of 48 samples never produced their measured turn — every one of them
killed by the same thing, 188 times: a gateway answering HTTP 200 with an empty
``choices`` array until the client's retries were gone.  Another run drew 325 rate limits
and said nothing.  Neither run's output named a number, so the only way to find out was to
read per-sample logs by hand afterwards, and the only way to prove which member of a
routing pool had poisoned the run was to run the whole thing again with a pin.

So the run reports on ITSELF, from values it counted rather than sentences it matched:

* **Cohort accounting.** Each case records how many samples it INTENDED and how many
  produced their measured turn.  A sample's ``.db`` exists from the moment the sample
  starts, so a file is not a result; what counts is the turn having run and been scored.
* **A fault tally.** Every model-call attempt logs its fault class and serving provider
  as structured fields (see ``penny.llm.models``); a handler on the ``penny`` logger adds
  them up.  The tally is a read of values, never a grep of prose.
* **A verdict.** A cohort that is mostly dead is not a smaller cohort — it is not a
  result, and the run says so and fails.

Why a STRICT MAJORITY is the bar
--------------------------------
The tempting reading of a half-dead cohort is "a mean over fewer samples, so noisier".
It is worse than that: dead samples are not missing at random.  The faults that kill them
correlate with the work — the long turn, the one that made the most calls, the one that
spent the most tokens is the one most likely to draw the bad provider or hit the rate
limit — so the survivors skew to the SHORT samples, and their mean measures something
other than the case.  The bar therefore has to sit where the surviving cohort can still
be read as the case rather than as a selection effect, and the honest place for that is a
strict majority of what was asked for: more than half of the intended samples must have
run.  With the standard N=5 that is 3; with N=1 it is 1.  It is a judgement, not a
derivation, which is why it is printed in the refusal and in the health block rather than
buried in a constant — and why a run that squeaks past it at 3 of 5 still says so, on the
line above the verdict.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from penny.llm.models import (
    FAULT_LOG_FIELD,
    PROVIDER_LOG_FIELD,
    UNREPORTED_PROVIDER,
    LlmFault,
)
from penny.tests.eval.utils import artifacts as eval_artifacts
from penny.tests.eval.utils.budget_hits import BudgetHits, render_budget_hits

# Health is another per-worker JSONL record riding the convention the per-case results
# already use (``artifacts.worker_filename`` / ``load_worker_lines``), so this names only
# the thing that is its own — its stem.  Under xdist the cases are spread across worker
# PROCESSES, each with its own tally, and two processes appending to one file interleave.
HEALTH_STEM = "health"

# Where those records live.  Usually the run's report dir; a LEVER-LESS run keeps no
# artifacts at all, so the Makefile points this at an ephemeral dir inside the --rm
# container instead — the run still gets its health block, which is the point.
HEALTH_DIR_ENV = "EVAL_HEALTH_DIR"

# The logger every penny module logs through — the same root the per-sample capture
# attaches to, so one handler here sees every model call any sample made.
PENNY_LOGGER = "penny"

# The bar (see the module docstring): a case must complete MORE THAN HALF the samples it
# intended.  Expressed as the comparison rather than a ratio so it needs no rounding rule
# and reads the same at every N.
COHORT_RULE = "a case must complete more than half the samples it intended"


def cohort_is_viable(completed: int, intended: int) -> bool:
    """Did enough of the intended cohort run for its mean to be about the case?

    A run that intended nothing is vacuously fine; anything else needs a strict majority.
    """
    if intended <= 0:
        return True
    return completed * 2 > intended


def results_needed(intended: int) -> int:
    """The fewest completed samples a cohort of ``intended`` can be read from."""
    return intended // 2 + 1


def _add_counts(left: dict[LlmFault, int], right: dict[LlmFault, int]) -> dict[LlmFault, int]:
    """Two fault tallies, added — a plain dict, so the model serializes as one."""
    return dict(Counter(left) + Counter(right))


def _merge_void_causes(cohorts: list[CohortRecord]) -> dict[str, int]:
    """Every voiding-exception label across these cohorts, counts summed."""
    merged: Counter[str] = Counter()
    for cohort in cohorts:
        merged.update(cohort.void_causes)
    return dict(merged)


def void_cause_label(fault_type: str, message: str) -> str:
    """One voiding exception's refusal-line label: its class, and the first line of its
    message when it has one (#2865).

    Built once, structurally, at the moment a sample is voided — never re-derived later
    by parsing the per-sample log line that already names the same exception in prose.
    The message's first line only, because a traceback or a multi-line assertion message
    would otherwise make two samples killed by the same check read as different causes.
    """
    first_line = next((line for line in message.splitlines() if line.strip()), "")
    return f"{fault_type}: {first_line}" if first_line else fault_type


def render_faults(faults: dict[LlmFault, int]) -> str:
    """A fault tally as one line, worst first — the run's and each provider's, one rule.

    Costliest first, because what a reader brings to this line is "what killed the run";
    ordering by class name instead would put the dominant cause wherever its name happens
    to fall.  Ties break on the class name, so a re-render reads identically.
    """
    ordered = sorted(faults.items(), key=lambda item: (-item[1], item[0].value))
    return " · ".join(f"{fault.value} {count}" for fault, count in ordered)


class ProviderTally(BaseModel):
    """What one serving provider did for this run: how many calls, and which faults."""

    calls: int = 0
    faults: dict[LlmFault, int] = Field(default_factory=dict)

    @property
    def fault_total(self) -> int:
        """How many of this provider's calls faulted."""
        return sum(self.faults.values())

    def render(self, name: str) -> str:
        """One provider's line in the health block."""
        return f"  {name}: {self.calls} calls — {render_faults(self.faults) or 'no faults'}"


class CohortRecord(BaseModel):
    """One case's cohort: how many samples it asked for, how many produced a turn, and how
    many of THOSE answered no claim at all.

    ``retried_boots`` counts the times one of its samples was stood up AGAIN because its
    preflight failed on a transient endpoint fault (#2230).  A retried boot that then ran
    is a completed sample like any other, so nothing else here would show it — and a run
    whose samples only start on their second try is a run worth knowing about.

    ``excluded`` (#2526) counts samples that DID produce a measured turn — ``completed``
    already includes them — but that the cohort could not pool: a provider refusal, a
    dead draw, a refused cycle, named by its own reason on the sample itself.  A run where
    most of a case's samples were excluded used to read as fully viable, because viability
    only ever asked whether the turn ran — so a rate-limited run that answered no claim at
    all could still print "this run is a result."

    ``void_causes`` (#2865) is every sample this cohort lost to an exception in world
    setup or ``prepare`` — never a model call — keyed by :func:`void_cause_label`, counted
    at the moment each one is voided.  A dead cohort's REFUSED line reads this before it
    ever reads a provider fault: a fault tally counts every model-call attempt the RUN
    made, which has nothing to say about a sample that never reached one.
    """

    case_id: str
    intended: int
    completed: int
    excluded: int = 0
    retried_boots: int = 0
    void_causes: dict[str, int] = Field(default_factory=dict)

    @property
    def dead(self) -> int:
        """Samples that never produced their measured turn."""
        return max(self.intended - self.completed, 0)

    @property
    def scored(self) -> int:
        """Samples that produced a measured turn AND answered a claim with it — the number
        the viability bar actually needs, since an excluded sample is a turn with nothing
        to read off it."""
        return max(self.completed - self.excluded, 0)

    @property
    def viable(self) -> bool:
        """Whether this case's surviving, SCORED samples can be read as the case."""
        return cohort_is_viable(self.scored, self.intended)


class RunHealth(BaseModel):
    """A run's own account of itself — merged across every process that ran part of it."""

    cohorts: list[CohortRecord] = Field(default_factory=list)
    calls: int = 0
    faults: dict[LlmFault, int] = Field(default_factory=dict)
    providers: dict[str, ProviderTally] = Field(default_factory=dict)
    # Which call kinds a token budget cut off mid-draw (#2477) — read off each sample's own
    # promptlog, never a fault: a cut draw is retried by the runtime itself, so this is a
    # MEASURE of how tight each kind's budget is running, not a cause of this run's verdict.
    budget_hits: BudgetHits = Field(default_factory=BudgetHits)

    def render(self) -> str:
        """The whole run-health block, in the order a reader needs it."""
        return "\n".join(
            [
                self._samples_line(),
                *self._retried_boot_lines(),
                *self._excluded_lines(),
                self._cases_line(),
                *self._dead_case_lines(),
                self._calls_line(),
                *self._fault_lines(),
                *self._budget_hit_lines(),
                *self._provider_lines(),
                self._verdict_line(),
            ]
        )

    # ── Derived counts ───────────────────────────────────────────────────
    @property
    def intended(self) -> int:
        """Samples the run set out to measure."""
        return sum(cohort.intended for cohort in self.cohorts)

    @property
    def completed(self) -> int:
        """Samples that produced their measured turn."""
        return sum(cohort.completed for cohort in self.cohorts)

    @property
    def dead(self) -> int:
        """Samples the run intended and never got."""
        return max(self.intended - self.completed, 0)

    @property
    def excluded(self) -> int:
        """Samples that ran but answered no claim at all (#2526) — a provider refusal, a
        dead draw, a refused cycle, summed from each cohort's own count."""
        return sum(cohort.excluded for cohort in self.cohorts)

    @property
    def dead_cohorts(self) -> list[CohortRecord]:
        """The cases whose surviving samples cannot be read as the case."""
        return [cohort for cohort in self.cohorts if not cohort.viable]

    @property
    def viable(self) -> bool:
        """A run is a result when every one of its cases is."""
        return not self.dead_cohorts

    @property
    def fault_total(self) -> int:
        """How many model-call attempts faulted across the run."""
        return sum(self.faults.values())

    @property
    def dominant_fault(self) -> tuple[LlmFault, int] | None:
        """The fault class that cost the run the most, or ``None`` when nothing failed."""
        if not self.faults:
            return None
        fault, count = max(self.faults.items(), key=lambda item: (item[1], item[0].value))
        return (fault, count)

    def merge(self, other: RunHealth) -> RunHealth:
        """Fold another process's health into this one — cases concatenate, tallies add."""
        providers = {name: tally.model_copy(deep=True) for name, tally in self.providers.items()}
        for name, tally in other.providers.items():
            merged = providers.setdefault(name, ProviderTally())
            merged.calls += tally.calls
            merged.faults = _add_counts(merged.faults, tally.faults)
        return RunHealth(
            cohorts=[*self.cohorts, *other.cohorts],
            calls=self.calls + other.calls,
            faults=_add_counts(self.faults, other.faults),
            providers=providers,
            budget_hits=self.budget_hits.merge(other.budget_hits),
        )

    # ── Block lines ──────────────────────────────────────────────────────
    def _samples_line(self) -> str:
        return (
            f"samples: {self.completed} of {self.intended} completed · {self.dead} dead "
            "(completed = the sample's measured turn ran; excluded, below, is "
            "completed but answered no claim)"
        )

    def _retried_boot_lines(self) -> list[str]:
        retried = [cohort for cohort in self.cohorts if cohort.retried_boots]
        if not retried:
            return []
        total = sum(cohort.retried_boots for cohort in retried)
        cases = " · ".join(f"{cohort.case_id} {cohort.retried_boots}" for cohort in retried)
        return [f"  {total} boot(s) retried after a transient preflight fault — {cases}"]

    def _excluded_lines(self) -> list[str]:
        """Hidden on a run that excluded nothing — the quiet case needs no line, same as
        faults and budget cuts.  A sample named here RAN (``completed`` already counts
        it) but answered no claim, so it is what separates a case's ``completed`` count
        from the ``scored`` one the viability bar below actually reads."""
        if not self.excluded:
            return []
        cases = " · ".join(
            f"{cohort.case_id} {cohort.excluded}" for cohort in self.cohorts if cohort.excluded
        )
        return [f"  {self.excluded} sample(s) ran and answered no claim (excluded) — {cases}"]

    def _cases_line(self) -> str:
        readable = len(self.cohorts) - len(self.dead_cohorts)
        return f"cases: {readable} of {len(self.cohorts)} readable — {COHORT_RULE}"

    def _dead_case_lines(self) -> list[str]:
        if not self.dead_cohorts:
            return []
        dead = " · ".join(
            f"{cohort.case_id} {cohort.scored}/{cohort.intended}"
            + (f" ({cohort.excluded} excluded)" if cohort.excluded else "")
            for cohort in self.dead_cohorts
        )
        return [f"  not readable: {dead}"]

    def _calls_line(self) -> str:
        return f"model calls: {self.calls} attempts · {self.fault_total} faulted"

    def _fault_lines(self) -> list[str]:
        if not self.faults:
            return []
        return [f"  {render_faults(self.faults)}"]

    def _budget_hit_lines(self) -> list[str]:
        """Hidden on a run nothing was cut on — the quiet case needs no line, same as faults."""
        if not self.budget_hits.total_cuts:
            return []
        return [f"  {render_budget_hits(self.budget_hits)}"]

    def _provider_lines(self) -> list[str]:
        if not self.providers:
            return ["providers: none reported by this endpoint"]
        ordered = sorted(self.providers.items(), key=lambda item: (-item[1].calls, item[0]))
        return ["providers:", *(tally.render(name) for name, tally in ordered)]

    def _verdict_line(self) -> str:
        if self.viable:
            return "verdict: this run is a result."
        return (
            f"REFUSED: {len(self.dead_cohorts)} case(s) scored a fraction of their intended "
            f"cohort{self._refusal_cause()}. A result computed from a fraction of the cohort "
            "is not a result."
        )

    def _refusal_cause(self) -> str:
        """What actually stopped the dead cohorts' samples (#2865): the exception(s) that
        voided them, the provider fault that kept the rest from completing, or both —
        never a fault tallied from calls that had nothing to do with the loss.

        A provider fault is named only when the dead cohorts' void causes do not already
        account for every sample they lost — the fault tally counts every model-call
        attempt the whole RUN made, so a fault that voided nothing here must not be
        blamed for samples a setup exception already explains.
        """
        dead = self.dead_cohorts
        void_causes = _merge_void_causes(dead)
        total_dead = sum(cohort.dead for cohort in dead)
        parts = [
            f"{label} ({count})"
            for label, count in sorted(void_causes.items(), key=lambda item: (-item[1], item[0]))
        ]
        blamed = self.dominant_fault
        if blamed and total_dead > sum(void_causes.values()):
            parts.append(f"mostly {blamed[0].value} ({blamed[1]})")
        return f" — {' · '.join(parts)}" if parts else ""


# ── The live tally: one handler, reading the fields the client stamps ────────
class FaultTally(logging.Handler):
    """Counts every chat attempt this process made, by fault class and by provider.

    A ``logging.Handler`` because the attempt record already exists and already carries
    the two values — a second, parallel channel for the same facts would be one more
    thing to keep in step.  It reads ``llm_fault`` / ``llm_provider`` off the record and
    ignores everything else, so no message text is ever matched.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.calls = 0
        self.faults: Counter[LlmFault] = Counter()
        self.providers: dict[str, ProviderTally] = {}

    def emit(self, record: logging.LogRecord) -> None:
        """Add one attempt to the tally, or ignore a record that is not one."""
        if not hasattr(record, PROVIDER_LOG_FIELD):
            return
        fault = getattr(record, FAULT_LOG_FIELD, None)
        provider = getattr(record, PROVIDER_LOG_FIELD, None) or UNREPORTED_PROVIDER
        self.calls += 1
        tally = self.providers.setdefault(provider, ProviderTally())
        tally.calls += 1
        if isinstance(fault, LlmFault):
            self.faults[fault] += 1
            tally.faults[fault] = tally.faults.get(fault, 0) + 1

    def snapshot(self, cohorts: list[CohortRecord], budget_hits: BudgetHits) -> RunHealth:
        """This process's health so far, beside the cohorts it ran and the budget hits
        its samples read off their own promptlog — not something this handler sees
        itself, since those come off a response's raw JSON rather than an attempt log."""
        return RunHealth(
            cohorts=list(cohorts),
            calls=self.calls,
            faults=dict(self.faults),
            providers={name: tally.model_copy(deep=True) for name, tally in self.providers.items()},
            budget_hits=budget_hits,
        )


# ── Process-wide state: one tally + one cohort list per pytest process ───────
_tally: FaultTally | None = None
_cohorts: list[CohortRecord] = []
# One process's running tally of every sample's own budget-hit reading (#2477), folded in
# as each sample finishes — beside ``_cohorts``, for the same reason: the run's health is
# settled once, from whatever this process accumulated over however many cases it ran.
_budget_hits = BudgetHits()
# The whole run's health, settled once at session finish and read again by the terminal
# summary.  Two hooks, one answer: computing it twice risks the two disagreeing, and the
# verdict that moved the exit status must be the verdict the block prints.
_run_health: RunHealth | None = None


def hold_run_health(health: RunHealth) -> None:
    """Keep the settled whole-run health for the terminal summary to render."""
    global _run_health
    _run_health = health


def held_run_health() -> RunHealth | None:
    """The settled whole-run health, or ``None`` before the session finished."""
    return _run_health


def begin_run() -> FaultTally:
    """Attach the tally to the ``penny`` logger for this process (idempotent)."""
    global _tally
    if _tally is None:
        _tally = FaultTally()
        logging.getLogger(PENNY_LOGGER).addHandler(_tally)
    return _tally


def record_cohort(
    case_id: str,
    *,
    intended: int,
    completed: int,
    excluded: int = 0,
    retried_boots: int = 0,
    void_causes: dict[str, int] | None = None,
) -> None:
    """Record what one case asked for, what it got, how many of those it could not pool,
    how many boots it had to retry, and what voided the samples it never got at all."""
    _cohorts.append(
        CohortRecord(
            case_id=case_id,
            intended=intended,
            completed=completed,
            excluded=excluded,
            retried_boots=retried_boots,
            void_causes=dict(void_causes or {}),
        )
    )


def record_budget_hits(hits: BudgetHits) -> None:
    """Fold one sample's budget-hit reading into this process's running total."""
    global _budget_hits
    _budget_hits = _budget_hits.merge(hits)


def process_health() -> RunHealth:
    """What THIS process saw — its own cases, its own model calls, its own budget hits."""
    return begin_run().snapshot(_cohorts, _budget_hits)


def health_dir() -> Path | None:
    """Where health records for this run live, or ``None`` when the run keeps none."""
    directory = os.environ.get(HEALTH_DIR_ENV)
    return Path(directory) if directory else None


def write_health(directory: Path, worker: str | None, health: RunHealth) -> Path:
    """Write one process's health record where the whole-run read will find it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / eval_artifacts.worker_filename(HEALTH_STEM, worker)
    path.write_text(health.model_dump_json() + "\n")
    return path


def load_health(directory: Path) -> RunHealth:
    """Every process's health record in a run dir, merged into the run's own.

    The read is the shared one, so health and results agree about what "every worker's
    file, in a stable order" means; folding is all this adds, because merging two tallies
    is the only part that is health's own.
    """
    merged = RunHealth()
    for line in eval_artifacts.load_worker_lines(directory, HEALTH_STEM):
        merged = merged.merge(RunHealth.model_validate_json(line))
    return merged
