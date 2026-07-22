"""Per-run eval artifacts: the input manifest + per-case JSONL results.

When ``EVAL_REPORT_DIR`` is set, a ``make eval`` run emits three durable
artifacts into that directory, beside the per-case transcript reports
(``<case_id>.md``):

  ``manifest.json``  — the run's input identity: commit SHA, dirty-diff filename,
                       model + embedding model + ``EVAL_SAMPLES``, and the
                       REQUIRED one-line lever (the run's hypothesis, from
                       ``EVAL_LEVER``).
  ``dirty.diff``     — the working-tree diff, saved verbatim (omitted when clean).
  ``results.jsonl``  — one JSON record per case: run id, case id, family, mean,
                       all-pass rate, per-check outcomes, N, timings.

The manifest also renders as a markdown header atop each ``<case_id>.md`` report
(commit · model · N · lever) — the #1711 PR-comment report header.

A report run (``EVAL_REPORT_DIR`` set) with no lever (``EVAL_LEVER``) fails fast
with an actionable message: the lever is what makes a score shift attributable to
the change that caused it, so it is not optional.

The whole surface is driven from the environment by the ``make eval`` wiring, but
the pure builders take their inputs explicitly so ``make check`` can exercise them
with synthetic fixture data — no git, no model, no container.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

# ── Environment contract (forwarded by the Makefile `eval` target) ───────────
EVAL_REPORT_DIR_ENV = "EVAL_REPORT_DIR"
EVAL_LEVER_ENV = "EVAL_LEVER"
EVAL_COMMIT_ENV = "EVAL_COMMIT"
EVAL_DIRTY_DIFF_ENV = "EVAL_DIRTY_DIFF"
EVAL_SAMPLES_ENV = "EVAL_SAMPLES"
LLM_MODEL_ENV = "LLM_MODEL"
LLM_EMBEDDING_MODEL_ENV = "LLM_EMBEDDING_MODEL"

# ── Artifact filenames (all live under EVAL_REPORT_DIR) ──────────────────────
MANIFEST_FILENAME = "manifest.json"
RESULTS_FILENAME = "results.jsonl"
DIRTY_DIFF_FILENAME = "dirty.diff"

# Defaults mirror `_real_model_config` / the Makefile so a manifest records the
# same model the run actually used even if the env var is somehow absent.
UNKNOWN_COMMIT = "unknown"
DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma"
DEFAULT_SAMPLES = 5

MISSING_LEVER_MESSAGE = (
    "EVAL_REPORT_DIR is set but EVAL_LEVER is empty. Every report run must state its "
    "hypothesis in one line so a score shift is attributable to the change that caused "
    "it. Set EVAL_LEVER, e.g.\n"
    "    EVAL_LEVER='moved instruction X from skill to prompt' "
    "EVAL_REPORT_DIR=/penny/data/eval-reports make eval"
)


class MissingLeverError(RuntimeError):
    """A report run (``EVAL_REPORT_DIR`` set) declared no lever (``EVAL_LEVER``)."""


# ── Failure-cause partition (#1695): every FAILED sample is one of these ──────
class FailureCause(StrEnum):
    """Why a failed sample failed, derived STRUCTURALLY (never a model judgment).

    The partition separates the real signal the loop chases from the noise:

    - ``behavioral`` — the model simply got it wrong (the signal).
    - ``pathology``  — a known model pathology fired (a reroll-guard poison signal in
      the run's persisted output — a punctuation collapse, a leaked Harmony envelope,
      a collapse-shaped tool name, or a bare call-fragment reply). Noise, not
      comprehension; excluded from the pathology-excluded score.
    - ``harness``    — a timeout / infrastructure fault, not the model at all.
    """

    BEHAVIORAL = "behavioral"
    PATHOLOGY = "pathology"
    HARNESS = "harness"


class CauseCounts(BaseModel):
    """Per-case tally of failed samples by cause (passing samples carry no cause)."""

    behavioral: int = 0
    pathology: int = 0
    harness: int = 0


def classify_cause(*, passed: bool, timed_out: bool, pathology: bool) -> FailureCause | None:
    """The structural cause of a sample's outcome — ``None`` when it passed.

    Order is the documented rule: a passing sample has no cause; a **pathology** signal
    outranks a timeout (the poison is the root cause, a downstream timeout its symptom);
    a clean timeout is **harness**; everything else is **behavioral**. Pure — the three
    inputs are the structural facts, computed by the caller from the persisted state.
    """
    if passed:
        return None
    if pathology:
        return FailureCause.PATHOLOGY
    if timed_out:
        return FailureCause.HARNESS
    return FailureCause.BEHAVIORAL


def count_causes(causes: Sequence[FailureCause | None]) -> CauseCounts:
    """Tally a case's per-sample causes (``None`` — a pass — counts toward nothing)."""
    return CauseCounts(
        behavioral=sum(1 for cause in causes if cause == FailureCause.BEHAVIORAL),
        pathology=sum(1 for cause in causes if cause == FailureCause.PATHOLOGY),
        harness=sum(1 for cause in causes if cause == FailureCause.HARNESS),
    )


def pathology_excluded(
    scores: Sequence[float], causes: Sequence[FailureCause | None]
) -> tuple[float, int]:
    """The honest read of model behaviour: ``(mean, kept)`` over every sample that is NOT
    a pathology failure (passing + behavioral + harness). Pathology samples drop out of the
    denominator so their noise can't sink the score; an all-pathology case is ``(0.0, 0)``."""
    kept = [
        score
        for score, cause in zip(scores, causes, strict=True)
        if cause != FailureCause.PATHOLOGY
    ]
    return (sum(kept) / len(kept), len(kept)) if kept else (0.0, 0)


def render_cause_summary(counts: CauseCounts, excluded_mean: float, kept: int) -> str:
    """The one-line failure-cause read (§4 of the format spec) — the pathology-excluded mean
    beside the behavioral/pathology/harness tally. Single-sourced so the console RESULT area
    and the PR-comment renderer (#1693) render one shape."""
    return (
        f"pathology-excluded mean {excluded_mean:.2f} ({kept} samples) · "
        f"causes — behavioral {counts.behavioral} · "
        f"pathology {counts.pathology} · harness {counts.harness}"
    )


# ── Structured inputs read off a scored case (Protocols avoid a conftest cycle) ──
class GradedCheck(Protocol):
    """The subset of a ``Check`` the per-check aggregate reads.

    Beyond ``ok``, the v2 per-case check table (#1725) reads ``scored`` (advisory flavour vs.
    a scored check), ``ignored`` (the not-applicable third state → a ➖ cell), and ``rationale``
    (the observed-vs-expected note surfaced on a miss). A ``Check`` structurally satisfies all four.
    """

    label: str
    ok: bool
    scored: bool
    ignored: bool
    rationale: str | None


class ScoredSample(Protocol):
    """The subset of a ``SampleResult`` the case artifact reads.

    ``checks`` / ``passed`` are read-only members (covariant), so a ``SampleResult``
    whose ``checks`` is a ``list[Check]`` structurally satisfies the protocol.

    ``cause`` is the structurally-classified failure cause the runner stamped (``None``
    for a pass, or an unstamped sample — the aggregate defaults an unstamped *failure*
    to ``behavioral``, never inventing a pathology/harness it can't observe).
    """

    score: float
    cause: FailureCause | None
    fragile: bool

    @property
    def passed(self) -> bool: ...

    @property
    def checks(self) -> Sequence[GradedCheck]: ...


class PerfTotals(Protocol):
    """The subset of ``_Perf`` the timings block reads."""

    calls: int
    duration_ms: int
    input_tokens: int
    output_tokens: int


# ── Serialized shapes ────────────────────────────────────────────────────────
class CheckCell(StrEnum):
    """One sample's outcome for a check, in the v2 per-case check summary table (#1725).

    ``absent`` is distinct from ``na``: ``na`` is a not-applicable check the scorer *emitted*
    for the sample (its branch didn't run → a ➖ cell); ``absent`` means the scorer emitted no
    such check for that sample at all (a blank cell), so the two never conflate.
    """

    PASSED = "passed"
    FAILED = "failed"
    NA = "na"
    ABSENT = "absent"


class CheckOutcome(BaseModel):
    """One check's aggregate across a case's samples — passed / present, plus the per-sample
    outcome cells + rationales the v2 check summary table renders from the artifact alone (#1725).

    ``passed``/``total`` are unchanged (ok-count / present-count). ``scored`` is the advisory
    flag (False → the check renders but is out of the score). ``cells`` carries one outcome per
    sample (aligned with ``sample_scores``); ``rationales`` collects the distinct
    observed-vs-expected notes from the samples that failed the check.
    """

    label: str
    passed: int
    total: int
    scored: bool = True
    cells: list[CheckCell] = Field(default_factory=list)
    rationales: list[str] = Field(default_factory=list)


class CaseTimings(BaseModel):
    """Per-case model-call totals, summed over its samples (from the promptlog)."""

    calls: int
    duration_ms: int
    input_tokens: int
    output_tokens: int


class CaseArtifact(BaseModel):
    """One case's line in ``results.jsonl`` — the mechanically-diffable record."""

    run_id: str
    case_id: str
    family: str
    mean: float
    all_pass_rate: float
    pathology_excluded_mean: float
    samples: int
    sample_scores: list[float]
    sample_causes: list[FailureCause | None]
    sample_fragile: list[bool] = Field(default_factory=list)
    cause_counts: CauseCounts
    checks: list[CheckOutcome]
    timings: CaseTimings
    # The gate this case ran under (#1725): the threshold + which score it compares
    # ("mean" | "pathology-excluded mean"). Both None for a report-only case.
    min_pass_rate: float | None = None
    gate_metric: str | None = None


class RunManifest(BaseModel):
    """The run's input identity — one ``manifest.json`` per ``make eval`` run."""

    run_id: str
    created_at: str
    commit: str
    dirty: bool
    diff_file: str | None
    model: str
    embedding_model: str
    samples: int
    lever: str


# ── Pure builders (no env, no filesystem — the make check test drives these) ──
def default_family(module: str) -> str:
    """Derive a family tag from a test module's name (``test_<x>`` → ``<x>``)."""
    leaf = module.rsplit(".", 1)[-1]
    return leaf[len("test_") :] if leaf.startswith("test_") else leaf


def _check_cell(check: GradedCheck) -> CheckCell:
    """One sample's outcome for a check: ➖ when not-applicable, ✅ when passed, else ❌."""
    if check.ignored:
        return CheckCell.NA
    return CheckCell.PASSED if check.ok else CheckCell.FAILED


def aggregate_checks(results: Sequence[ScoredSample]) -> list[CheckOutcome]:
    """Fold every sample's checks into per-label totals AND per-sample outcome cells (in order).

    ``passed``/``total`` are unchanged (ok-count / present-count). ``cells`` records each sample's
    outcome (passed/failed/na, or ``absent`` when the scorer emitted no such check that sample), so
    the v2 summary table renders one row per label with a column per sample; ``rationales`` collects
    the distinct observed-vs-expected notes from the samples that failed the check."""
    order: list[str] = []
    passed: dict[str, int] = {}
    total: dict[str, int] = {}
    scored: dict[str, bool] = {}
    cells: dict[str, list[CheckCell]] = {}
    rationales: dict[str, list[str]] = {}
    sample_count = len(results)
    for index, result in enumerate(results):
        for check in result.checks:
            if check.label not in total:
                order.append(check.label)
                scored[check.label] = check.scored
                cells[check.label] = [CheckCell.ABSENT] * sample_count
                rationales[check.label] = []
            total[check.label] = total.get(check.label, 0) + 1
            passed[check.label] = passed.get(check.label, 0) + (1 if check.ok else 0)
            cells[check.label][index] = _check_cell(check)
            missed = not check.ignored and not check.ok
            if missed and check.rationale and check.rationale not in rationales[check.label]:
                rationales[check.label].append(check.rationale)
    return [
        CheckOutcome(
            label=label,
            passed=passed[label],
            total=total[label],
            scored=scored[label],
            cells=cells[label],
            rationales=rationales[label],
        )
        for label in order
    ]


def _sample_cause(sample: ScoredSample) -> FailureCause | None:
    """The per-sample cause for the record: ``None`` for a pass; the runner-stamped cause
    for a failure, defaulting an UNSTAMPED failure to ``behavioral`` (the aggregate never
    invents a pathology/harness the runner didn't observe — it only fills the honest default)."""
    if sample.passed:
        return None
    return sample.cause or FailureCause.BEHAVIORAL


def gate_metric_label(min_pass_rate: float | None, *, gate_pathology_excluded: bool) -> str | None:
    """Which score a gated case compares — ``None`` when report-only (no ``min_pass_rate``), else
    ``pathology-excluded mean`` (the honest-threshold opt-in, #1698) or plain ``mean``."""
    if min_pass_rate is None:
        return None
    return "pathology-excluded" if gate_pathology_excluded else "mean"


def build_case_artifact(
    *,
    run_id: str,
    case_id: str,
    family: str,
    results: Sequence[ScoredSample],
    timings: CaseTimings,
    min_pass_rate: float | None = None,
    gate_pathology_excluded: bool = False,
) -> CaseArtifact:
    """Aggregate a case's samples into its ``results.jsonl`` record."""
    count = len(results)
    mean = sum(result.score for result in results) / count if count else 0.0
    all_pass = sum(1 for result in results if result.passed) / count if count else 0.0
    scores = [result.score for result in results]
    causes = [_sample_cause(result) for result in results]
    excluded_mean, _kept = pathology_excluded(scores, causes)
    metric = gate_metric_label(min_pass_rate, gate_pathology_excluded=gate_pathology_excluded)
    return CaseArtifact(
        run_id=run_id,
        case_id=case_id,
        family=family,
        mean=mean,
        all_pass_rate=all_pass,
        pathology_excluded_mean=excluded_mean,
        samples=count,
        sample_scores=scores,
        sample_causes=causes,
        sample_fragile=[bool(result.fragile) for result in results],
        cause_counts=count_causes(causes),
        checks=aggregate_checks(results),
        timings=timings,
        min_pass_rate=min_pass_rate,
        gate_metric=metric,
    )


def timings_from_perf(perf: PerfTotals) -> CaseTimings:
    """Project a case's ``_Perf`` totals onto the serialized timings shape."""
    return CaseTimings(
        calls=perf.calls,
        duration_ms=perf.duration_ms,
        input_tokens=perf.input_tokens,
        output_tokens=perf.output_tokens,
    )


def build_manifest(
    *,
    commit: str,
    dirty_diff: str,
    model: str,
    embedding_model: str,
    samples: int,
    lever: str,
    now: datetime,
) -> RunManifest:
    """Assemble the run manifest from explicit inputs (``now`` fixes the run id)."""
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    short = commit[:8] if commit and commit != UNKNOWN_COMMIT else UNKNOWN_COMMIT
    return RunManifest(
        run_id=f"run-{stamp}-{short}",
        created_at=now.isoformat(),
        commit=commit or UNKNOWN_COMMIT,
        dirty=bool(dirty_diff),
        diff_file=DIRTY_DIFF_FILENAME if dirty_diff else None,
        model=model,
        embedding_model=embedding_model,
        samples=samples,
        lever=lever,
    )


def render_manifest_header(manifest: RunManifest) -> str:
    """The markdown header atop each per-case report (commit · model · N · lever)."""
    dirty = " (dirty)" if manifest.dirty else ""
    return "\n".join(
        [
            f"### {manifest.run_id}",
            "",
            f"- commit: `{manifest.commit}`{dirty}",
            f"- model: `{manifest.model}`",
            f"- N: {manifest.samples}",
            f"- lever: {manifest.lever}",
            "",
        ]
    )


# ── The run: manifest + dirty diff + the results.jsonl append point ───────────
@dataclass
class EvalRun:
    """One ``make eval`` run's report directory, resolved manifest, and diff."""

    report_dir: Path
    manifest: RunManifest
    dirty_diff: str

    def write_inputs(self) -> None:
        """Write ``manifest.json`` + the verbatim ``dirty.diff`` (idempotent)."""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        if self.dirty_diff:
            (self.report_dir / DIRTY_DIFF_FILENAME).write_text(self.dirty_diff)
        (self.report_dir / MANIFEST_FILENAME).write_text(
            self.manifest.model_dump_json(indent=2) + "\n"
        )

    def write_case_header(self, case_id: str) -> None:
        """Stamp the manifest header atop this case's report, once (before sample 1)."""
        report = self.report_dir / f"{case_id}.md"
        if report.exists():
            return
        self.report_dir.mkdir(parents=True, exist_ok=True)
        report.write_text(render_manifest_header(self.manifest) + "\n")

    def append_case(self, artifact: CaseArtifact) -> None:
        """Append one case's record as a line to ``results.jsonl``."""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        with (self.report_dir / RESULTS_FILENAME).open("a") as handle:
            handle.write(artifact.model_dump_json() + "\n")


def run_from_env(env: Mapping[str, str], *, now: datetime | None = None) -> EvalRun | None:
    """Resolve the active run from the environment, or ``None`` off-report.

    Fails fast with :class:`MissingLeverError` when a report is requested
    (``EVAL_REPORT_DIR`` set) but no lever is declared.
    """
    report_dir = env.get(EVAL_REPORT_DIR_ENV)
    if not report_dir:
        return None
    lever = (env.get(EVAL_LEVER_ENV) or "").strip()
    if not lever:
        raise MissingLeverError(MISSING_LEVER_MESSAGE)
    dirty_diff = env.get(EVAL_DIRTY_DIFF_ENV) or ""
    manifest = build_manifest(
        commit=(env.get(EVAL_COMMIT_ENV) or UNKNOWN_COMMIT).strip(),
        dirty_diff=dirty_diff,
        model=env.get(LLM_MODEL_ENV, DEFAULT_MODEL),
        embedding_model=env.get(LLM_EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL),
        samples=int(env.get(EVAL_SAMPLES_ENV, str(DEFAULT_SAMPLES))),
        lever=lever,
        now=now or datetime.now(UTC),
    )
    return EvalRun(Path(report_dir), manifest, dirty_diff)


# ── Live singleton: one manifest/run_id shared across every case in the process ──
_active_run: EvalRun | None = None
_resolved = False


def active_run() -> EvalRun | None:
    """The process-wide run, resolved once from the environment.

    Fails fast on the first call if a report was requested without a lever;
    writes the manifest + dirty diff once on the first successful resolution.
    """
    global _active_run, _resolved
    if not _resolved:
        _active_run = run_from_env(os.environ)
        _resolved = True
        if _active_run is not None:
            _active_run.write_inputs()
    return _active_run


def begin_case(case_id: str) -> None:
    """Per-case entry point: resolve the run (fail-fast) and stamp the report header.

    No-op off-report. Called at the top of each runner ``_run`` so a missing lever
    trips before any sample executes.
    """
    run = active_run()
    if run is not None:
        run.write_case_header(case_id)


def record_case(
    *,
    case_id: str,
    family: str | None,
    module: str,
    results: Sequence[ScoredSample],
    perf: PerfTotals,
    min_pass_rate: float | None = None,
    gate_pathology_excluded: bool = False,
) -> None:
    """Append the case's ``results.jsonl`` record. No-op off-report. The gate the case ran under
    (``min_pass_rate`` + which score it compares, #1725) rides into the record for the gate line."""
    run = active_run()
    if run is None:
        return
    artifact = build_case_artifact(
        run_id=run.manifest.run_id,
        case_id=case_id,
        family=family or default_family(module),
        results=results,
        timings=timings_from_perf(perf),
        min_pass_rate=min_pass_rate,
        gate_pathology_excluded=gate_pathology_excluded,
    )
    run.append_case(artifact)
