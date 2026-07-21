"""Whole-render tests for the run-comment assembler (``assemble.py``).

NOT eval-marked — they drive the deterministic assembler over a SYNTHETIC report
directory (manifest + results.jsonl + per-case ``.md`` transcripts), so they run
inside ``make check``: no git, no model, no container. The assembled comment is
asserted as a WHOLE-RENDER literal (pr-review-guide §6), plus the edge cases the
issue names — missing manifest fields (an ``unknown`` commit, a dirty tree) and a
zero-failure run — and the CLI's stdout/exit-code contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from penny.tests.eval.artifacts import (
    CaseArtifact,
    CaseTimings,
    CauseCounts,
    CheckOutcome,
    FailureCause,
    RunManifest,
    build_manifest,
    render_manifest_header,
)
from penny.tests.eval.assemble import (
    USAGE,
    assemble_run_comment,
    load_manifest,
    main,
)

_NOW = datetime(2026, 7, 20, 14, 32, 0, tzinfo=UTC)
_COMMIT = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
_TIMINGS = CaseTimings(calls=12, duration_ms=180000, input_tokens=30000, output_tokens=4000)


def _write_run(
    report_dir: Path,
    manifest: RunManifest,
    artifacts: list[CaseArtifact],
    transcripts: dict[str, str],
) -> None:
    """Materialise a completed run's report dir exactly as ``make eval`` leaves it: the
    manifest, one ``results.jsonl`` line per case, and each named case's ``<case_id>.md``
    with the same manifest-header prefix ``write_case_header`` stamps (so the assembler's
    header-strip is genuinely exercised). A case absent from ``transcripts`` gets no ``.md``."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    with (report_dir / "results.jsonl").open("w") as handle:
        for artifact in artifacts:
            handle.write(artifact.model_dump_json() + "\n")
    header = render_manifest_header(manifest) + "\n"
    for case_id, body in transcripts.items():
        (report_dir / f"{case_id}.md").write_text(header + body)


_SAMPLE_FERN = (
    "#### sample 1 — ✅ 2/2 checks\n\n"
    "| # | Actor | Content |\n|---|---|---|\n"
    '| 1 | 🔧 Penny → tool ✅ | collection_write(memory="fern-care", key="winter watering") |\n\n'
    "#### sample 3 — ❌ 1/2 checks\n\n"
    "| # | Actor | Content |\n|---|---|---|\n"
    "| 1 | 🔧 Penny → tool ❌ | done() |\n"
)
_SAMPLE_CHAT = (
    "#### sample 1 — ✅ 3/3 checks\n\n"
    "| # | Actor | Content |\n|---|---|---|\n"
    "| 1 | 🤖 Penny ✅ | Your ferns want water every 7-10 days over winter. |\n"
)


def test_assemble_two_cases_whole_render(tmp_path: Path) -> None:
    """The full comment: manifest header, run totals across both cases, and each case's dual
    RESULT line + cause summary above its folded transcript — in the spec's section order."""
    manifest = build_manifest(
        commit=_COMMIT,
        dirty_diff="",
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=4,
        lever="moved the notify-suffix guidance into the collector prompt",
        now=_NOW,
    )
    fern = CaseArtifact(
        run_id=manifest.run_id,
        case_id="test_extractors.py::watch_fern_care",
        family="extractors",
        mean=0.5,
        all_pass_rate=0.5,
        pathology_excluded_mean=2 / 3,
        samples=4,
        sample_scores=[1.0, 1.0, 0.0, 0.0],
        sample_causes=[None, None, FailureCause.BEHAVIORAL, FailureCause.PATHOLOGY],
        cause_counts=CauseCounts(behavioral=1, pathology=1, harness=0),
        checks=[CheckOutcome(label="write", passed=3, total=4)],
        timings=_TIMINGS,
    )
    chat = CaseArtifact(
        run_id=manifest.run_id,
        case_id="test_chat_response.py::answer_from_memory",
        family="chat_response",
        mean=1.0,
        all_pass_rate=1.0,
        pathology_excluded_mean=1.0,
        samples=4,
        sample_scores=[1.0, 1.0, 1.0, 1.0],
        sample_causes=[None, None, None, None],
        cause_counts=CauseCounts(),
        checks=[],
        timings=_TIMINGS,
    )
    _write_run(
        tmp_path,
        manifest,
        [fern, chat],
        {fern.case_id: _SAMPLE_FERN, chat.case_id: _SAMPLE_CHAT},
    )
    assert assemble_run_comment(tmp_path) == (
        """### run-20260720T143200Z-a1b2c3d4

- commit: `a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0`
- model: `gpt-oss:20b`
- N: 4
- lever: moved the notify-suffix guidance into the collector prompt

## Run totals

mean 0.75 · all-pass 6/8
pathology-excluded mean 0.86 (7 samples) · causes — behavioral 1 · pathology 1 · harness 0

### `test_extractors.py::watch_fern_care` — extractors

**RESULT:** mean 0.50 · all-pass 2/4
pathology-excluded mean 0.67 (3 samples) · causes — behavioral 1 · pathology 1 · harness 0

<details><summary>transcripts — test_extractors.py::watch_fern_care</summary>

#### sample 1 — ✅ 2/2 checks

| # | Actor | Content |
|---|---|---|
| 1 | 🔧 Penny → tool ✅ | collection_write(memory="fern-care", key="winter watering") |

#### sample 3 — ❌ 1/2 checks

| # | Actor | Content |
|---|---|---|
| 1 | 🔧 Penny → tool ❌ | done() |

</details>

### `test_chat_response.py::answer_from_memory` — chat_response

**RESULT:** mean 1.00 · all-pass 4/4
pathology-excluded mean 1.00 (4 samples) · causes — behavioral 0 · pathology 0 · harness 0

<details><summary>transcripts — test_chat_response.py::answer_from_memory</summary>

#### sample 1 — ✅ 3/3 checks

| # | Actor | Content |
|---|---|---|
| 1 | 🤖 Penny ✅ | Your ferns want water every 7-10 days over winter. |

</details>
"""
    )


def test_zero_failure_run_whole_render(tmp_path: Path) -> None:
    """A clean run where every sample passed: all-pass = N/N, the cause tally is all zeros,
    and the pathology-excluded mean equals the mean (nothing dropped)."""
    manifest = build_manifest(
        commit="beef1234beef1234beef1234beef1234beef1234",
        dirty_diff="",
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=3,
        lever="baseline: no change",
        now=datetime(2026, 7, 20, 9, 0, 0, tzinfo=UTC),
    )
    greets = CaseArtifact(
        run_id=manifest.run_id,
        case_id="test_peripheral.py::greets",
        family="peripheral",
        mean=1.0,
        all_pass_rate=1.0,
        pathology_excluded_mean=1.0,
        samples=3,
        sample_scores=[1.0, 1.0, 1.0],
        sample_causes=[None, None, None],
        cause_counts=CauseCounts(),
        checks=[],
        timings=_TIMINGS,
    )
    body = (
        "#### sample 1 — ✅ PASS\n\n"
        "| # | Actor | Content |\n|---|---|---|\n"
        "| 1 | 🤖 Penny ✅ | Hey! |\n"
    )
    _write_run(tmp_path, manifest, [greets], {greets.case_id: body})
    assert assemble_run_comment(tmp_path) == (
        """### run-20260720T090000Z-beef1234

- commit: `beef1234beef1234beef1234beef1234beef1234`
- model: `gpt-oss:20b`
- N: 3
- lever: baseline: no change

## Run totals

mean 1.00 · all-pass 3/3
pathology-excluded mean 1.00 (3 samples) · causes — behavioral 0 · pathology 0 · harness 0

### `test_peripheral.py::greets` — peripheral

**RESULT:** mean 1.00 · all-pass 3/3
pathology-excluded mean 1.00 (3 samples) · causes — behavioral 0 · pathology 0 · harness 0

<details><summary>transcripts — test_peripheral.py::greets</summary>

#### sample 1 — ✅ PASS

| # | Actor | Content |
|---|---|---|
| 1 | 🤖 Penny ✅ | Hey! |

</details>
"""
    )


def test_dirty_unknown_commit_and_missing_transcript_whole_render(tmp_path: Path) -> None:
    """Missing manifest fields degrade legibly: an ``unknown`` commit renders as-is with the
    ``(dirty)`` flag, and a case whose ``.md`` is absent folds an honest placeholder rather
    than crashing or emitting an empty ``<details>``."""
    manifest = build_manifest(
        commit="unknown",
        dirty_diff="--- a\n+++ b\n",
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=2,
        lever="probe the collector nudge",
        now=datetime(2026, 7, 20, 11, 15, 30, tzinfo=UTC),
    )
    artifact = CaseArtifact(
        run_id=manifest.run_id,
        case_id="test_collector_honesty.py::no_confab",
        family="collector_honesty",
        mean=0.0,
        all_pass_rate=0.0,
        pathology_excluded_mean=0.0,
        samples=2,
        sample_scores=[0.0, 0.0],
        sample_causes=[FailureCause.HARNESS, FailureCause.BEHAVIORAL],
        cause_counts=CauseCounts(behavioral=1, pathology=0, harness=1),
        checks=[],
        timings=_TIMINGS,
    )
    _write_run(tmp_path, manifest, [artifact], {})  # no transcript for this case
    assert assemble_run_comment(tmp_path) == (
        """### run-20260720T111530Z-unknown

- commit: `unknown` (dirty)
- model: `gpt-oss:20b`
- N: 2
- lever: probe the collector nudge

## Run totals

mean 0.00 · all-pass 0/2
pathology-excluded mean 0.00 (2 samples) · causes — behavioral 1 · pathology 0 · harness 1

### `test_collector_honesty.py::no_confab` — collector_honesty

**RESULT:** mean 0.00 · all-pass 0/2
pathology-excluded mean 0.00 (2 samples) · causes — behavioral 1 · pathology 0 · harness 1

<details><summary>transcripts — test_collector_honesty.py::no_confab</summary>

_(no transcript recorded)_

</details>
"""
    )


def test_missing_manifest_raises_actionable(tmp_path: Path) -> None:
    """No ``manifest.json`` → a FileNotFoundError naming the fix (this isn't a completed run)."""
    with pytest.raises(FileNotFoundError) as excinfo:
        load_manifest(tmp_path)
    assert "manifest.json" in str(excinfo.value)
    assert "make eval" in str(excinfo.value)  # actionable: names how to produce it


def test_cli_writes_comment_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``main`` writes the assembled comment to stdout verbatim and exits 0 on a good dir."""
    manifest = build_manifest(
        commit=_COMMIT,
        dirty_diff="",
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=1,
        lever="ship it",
        now=_NOW,
    )
    artifact = CaseArtifact(
        run_id=manifest.run_id,
        case_id="test_peripheral.py::greets",
        family="peripheral",
        mean=1.0,
        all_pass_rate=1.0,
        pathology_excluded_mean=1.0,
        samples=1,
        sample_scores=[1.0],
        sample_causes=[None],
        cause_counts=CauseCounts(),
        checks=[],
        timings=_TIMINGS,
    )
    _write_run(tmp_path, manifest, [artifact], {artifact.case_id: "#### sample 1 — ✅ PASS\n"})
    assert main([str(tmp_path)]) == 0
    assert capsys.readouterr().out == assemble_run_comment(tmp_path)


def test_cli_reports_bad_usage_and_missing_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No arg → usage on stderr, exit 2; a dir with no manifest → the error on stderr, exit 1."""
    assert main([]) == 2
    assert capsys.readouterr().err.strip() == USAGE
    assert main([str(tmp_path / "does-not-exist")]) == 1
    assert "manifest.json" in capsys.readouterr().err
