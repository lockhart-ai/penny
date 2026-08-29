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
from penny.tests.eval import artifacts as eval_artifacts

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
    """One case's cohort: how many samples it asked for, and how many produced a turn."""

    case_id: str
    intended: int
    completed: int

    @property
    def dead(self) -> int:
        """Samples that never produced their measured turn."""
        return max(self.intended - self.completed, 0)

    @property
    def viable(self) -> bool:
        """Whether this case's surviving samples can be read as the case."""
        return cohort_is_viable(self.completed, self.intended)


class RunHealth(BaseModel):
    """A run's own account of itself — merged across every process that ran part of it."""

    cohorts: list[CohortRecord] = Field(default_factory=list)
    calls: int = 0
    faults: dict[LlmFault, int] = Field(default_factory=dict)
    providers: dict[str, ProviderTally] = Field(default_factory=dict)

    def render(self) -> str:
        """The whole run-health block, in the order a reader needs it."""
        return "\n".join(
            [
                self._samples_line(),
                self._cases_line(),
                *self._dead_case_lines(),
                self._calls_line(),
                *self._fault_lines(),
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
        )

    # ── Block lines ──────────────────────────────────────────────────────
    def _samples_line(self) -> str:
        return (
            f"samples: {self.completed} of {self.intended} completed · {self.dead} dead "
            "(completed = the sample's measured turn ran and was scored)"
        )

    def _cases_line(self) -> str:
        readable = len(self.cohorts) - len(self.dead_cohorts)
        return f"cases: {readable} of {len(self.cohorts)} readable — {COHORT_RULE}"

    def _dead_case_lines(self) -> list[str]:
        if not self.dead_cohorts:
            return []
        dead = " · ".join(
            f"{cohort.case_id} {cohort.completed}/{cohort.intended}" for cohort in self.dead_cohorts
        )
        return [f"  not readable: {dead}"]

    def _calls_line(self) -> str:
        return f"model calls: {self.calls} attempts · {self.fault_total} faulted"

    def _fault_lines(self) -> list[str]:
        if not self.faults:
            return []
        return [f"  {render_faults(self.faults)}"]

    def _provider_lines(self) -> list[str]:
        if not self.providers:
            return ["providers: none reported by this endpoint"]
        ordered = sorted(self.providers.items(), key=lambda item: (-item[1].calls, item[0]))
        return ["providers:", *(tally.render(name) for name, tally in ordered)]

    def _verdict_line(self) -> str:
        if self.viable:
            return "verdict: this run is a result."
        blamed = self.dominant_fault
        cause = f" — mostly {blamed[0].value} ({blamed[1]})" if blamed else ""
        return (
            f"REFUSED: {len(self.dead_cohorts)} case(s) scored a fraction of their intended "
            f"cohort{cause}. A result computed from a fraction of the cohort is not a result."
        )


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

    def snapshot(self, cohorts: list[CohortRecord]) -> RunHealth:
        """This process's health so far, beside the cohorts it ran."""
        return RunHealth(
            cohorts=list(cohorts),
            calls=self.calls,
            faults=dict(self.faults),
            providers={name: tally.model_copy(deep=True) for name, tally in self.providers.items()},
        )


# ── Process-wide state: one tally + one cohort list per pytest process ───────
_tally: FaultTally | None = None
_cohorts: list[CohortRecord] = []
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


def record_cohort(case_id: str, *, intended: int, completed: int) -> None:
    """Record what one case asked for and what it got."""
    _cohorts.append(CohortRecord(case_id=case_id, intended=intended, completed=completed))


def process_health() -> RunHealth:
    """What THIS process saw — its own cases and its own model calls."""
    return begin_run().snapshot(_cohorts)


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
