"""Regression diff: a prior run's ``results.jsonl`` → per-check pass lookup (issue #1693).

When ``EVAL_BASELINE`` names a prior run's report directory (or its ``results.jsonl``
directly), the per-sample report marks a NOW-FAILING check **REGRESSED** (``❌ 🔻``)
when that same check was fully green in the prior run — a flip, distinct from a check
that was already red.  The lookup is per-check, keyed by ``(case_id, label)`` against
the prior run's mechanically-diffable ``CaseArtifact`` records (#1692) — the record
shapes are consumed here, never changed.

Off (``EVAL_BASELINE`` unset, or the file missing/empty) → no baseline, no REGRESSED
marks, no error: a first run simply has nothing to flip against.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from penny.tests.eval.artifacts import RESULTS_FILENAME, CaseArtifact

# ── Environment contract (forwarded by the Makefile `eval` target) ───────────
EVAL_BASELINE_ENV = "EVAL_BASELINE"


@dataclass
class Baseline:
    """A prior run's per-case records — the flip reference for REGRESSED marks.

    Keyed by ``case_id`` (last occurrence wins, so a report dir reused across runs
    reads its most recent record for each case)."""

    by_case: dict[str, CaseArtifact]

    def was_passing(self, case_id: str, label: str) -> bool:
        """True when the prior run had ``label`` FULLY GREEN in ``case_id`` — passed in
        every prior sample (``passed == total``, ``total > 0``).  An absent case/label,
        or a check that was already flaky in the baseline, is not a flip."""
        artifact = self.by_case.get(case_id)
        if artifact is None:
            return False
        for check in artifact.checks:
            if check.label == label:
                return check.total > 0 and check.passed == check.total
        return False

    def run_id_for(self, case_id: str) -> str | None:
        """The prior run id this case diffs against — named in the REGRESSED note."""
        artifact = self.by_case.get(case_id)
        return artifact.run_id if artifact is not None else None


def _resolve_results_path(raw_path: str) -> Path | None:
    """Resolve ``EVAL_BASELINE`` to a ``results.jsonl`` file, or ``None`` if absent.

    Accepts either a report DIRECTORY (its ``results.jsonl`` is used) or the file
    itself.  A path that resolves to no existing file is ``None`` — the graceful
    'no baseline yet' case, not an error."""
    path = Path(raw_path)
    if path.is_dir():
        path = path / RESULTS_FILENAME
    return path if path.is_file() else None


def load_baseline(raw_path: str) -> Baseline | None:
    """Parse a prior ``results.jsonl`` into a :class:`Baseline`, or ``None`` when the
    file is missing or empty.  Each non-blank line is one :class:`CaseArtifact`."""
    path = _resolve_results_path(raw_path)
    if path is None:
        return None
    by_case: dict[str, CaseArtifact] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        artifact = CaseArtifact.model_validate_json(stripped)
        by_case[artifact.case_id] = artifact
    return Baseline(by_case) if by_case else None


def baseline_from_env(env: Mapping[str, str] | None = None) -> Baseline | None:
    """The active baseline resolved from ``EVAL_BASELINE`` — ``None`` when unset.

    Read fresh (not cached) so each report render reflects the current environment,
    mirroring how ``_write_sample_report`` reads ``EVAL_REPORT_DIR`` directly."""
    environ = os.environ if env is None else env
    raw_path = environ.get(EVAL_BASELINE_ENV)
    return load_baseline(raw_path) if raw_path else None
