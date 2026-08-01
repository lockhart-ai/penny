"""Plain (non-eval) tests for the run-comment splitter (``comment_split.py``, #1808).

They drive the pure text helpers + the CLI over a SYNTHETIC oversized document built from the real
fold primitives (``report.fold_sample``), so they run inside ``make check``: no model, no GPU, no
container, no artifacts. The four properties the ticket names are asserted directly — the part
count, that no ``<details>`` straddles a boundary, that part 1 carries the run header, and that
reassembling the parts is byte-identical to the input — plus the two refusals (build noise, an
unsplittable fold).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from penny.tests.eval.comment_split import (
    GITHUB_COMMENT_LIMIT,
    PART_HEADER,
    USAGE,
    build_noise_reason,
    main,
    partition_on_sample_folds,
    sample_fold_segments,
    split_run_comment,
    unsplittable_reason,
)
from penny.tests.eval.report import fold_sample

RUN_HEADER = (
    "**run-20990101T000000Z** · commit `abcdef12` · test-model · N=8 · **lever:** a synthetic run\n"
    "**RESULT:** mean 0.50 · all-pass 4/8"
)
CASE_HEADING = "### `synthetic-case` — synthetic"
FOOTER = "_artifacts (local, never committed): `/penny/eval-artifacts/run-x`_"


def _document(samples: int, body_size: int) -> str:
    """A synthetic assembled run comment: the run header, a case heading, ``samples`` folded sample
    blocks of ``body_size`` characters each, and the footer — the same section shape (and the same
    ``\\n\\n`` joins) ``assemble.assemble_run_comment`` produces."""
    blocks = [
        fold_sample(number, "✅ pass · 3/3 (1.00) · 12s · 4 calls", "x" * body_size)
        for number in range(1, samples + 1)
    ]
    return "\n\n".join([RUN_HEADER, CASE_HEADING, *blocks, FOOTER]) + "\n"


def test_segments_cut_only_on_folds_and_rejoin_exactly() -> None:
    """Every cut sits at a sample fold's opening tag, the leading segment carries the run header +
    case heading, and the segments concatenate back to the document byte for byte."""
    document = _document(samples=3, body_size=100)
    segments = sample_fold_segments(document)
    assert len(segments) == 4  # header+heading, then one per sample (the last carries the footer)
    assert segments[0].startswith(RUN_HEADER)
    assert all(segment.startswith("<details><summary>sample ") for segment in segments[1:])
    assert "".join(segments) == document


def test_split_counts_parts_and_reassembles_byte_identically() -> None:
    """An oversized document splits into the expected number of budget-sized parts, each part is a
    whole number of folds, and the bodies rejoin byte-identically to the input."""
    document = _document(samples=8, body_size=25000)
    bodies = partition_on_sample_folds(document, budget=58000)
    # header+heading (~200) then 8 × ~25K folds: two folds per part, four parts.
    assert len(bodies) == 4
    assert all(len(body) <= 58000 for body in bodies)
    assert "".join(bodies) == document


def test_no_details_straddles_a_part_boundary() -> None:
    """Every part opens and closes on whole ``<details>`` folds — the balance of opening and
    closing tags is equal within each part, so no fold is cut in half."""
    document = _document(samples=8, body_size=25000)
    for body in partition_on_sample_folds(document, budget=58000):
        assert body.count("<details>") == body.count("</details>")
        assert "<details><summary>sample " in body


def test_part_one_carries_the_run_header_and_each_part_is_headed() -> None:
    """Part 1 opens with the split header then the run header (identity · RESULT — what a reader
    looks at first); every part announces its position and that the content is verbatim."""
    document = _document(samples=8, body_size=25000)
    parts = split_run_comment(document, budget=58000)
    assert parts[0].startswith(
        f"{PART_HEADER.format(number=1, total=4)}\n\n{RUN_HEADER}\n\n{CASE_HEADING}\n\n"
    )
    assert [part.splitlines()[0] for part in parts] == [
        f"(report {number} of 4 — split for GitHub's 64K comment cap; content verbatim)"
        for number in range(1, 5)
    ]
    # Stripping each header rejoins to the original document — nothing was dropped or rewritten.
    assert "".join(part.split("\n\n", 1)[1] for part in parts) == document


def test_a_document_that_fits_is_one_unheaded_part() -> None:
    """Under the budget there is no "more" to announce, so the comment posts exactly as it did
    before the splitter existed — byte-identical, no header."""
    document = _document(samples=2, body_size=100)
    assert split_run_comment(document) == [document]


def test_build_noise_is_refused_with_an_actionable_reason() -> None:
    """A body opening with `make assemble`'s echoed recipe / build log is refused, naming the
    marker and the correct way to capture the body; a real report is clean."""
    for noise in (
        "GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
        "docker compose build penny",
        "#1 [internal] load local bake definitions",
    ):
        reason = build_noise_reason(f"{noise}\n{_document(samples=1, body_size=10)}")
        assert reason is not None
        assert "python -m penny.tests.eval.assemble" in reason
    assert build_noise_reason(_document(samples=2, body_size=100)) is None
    # Only the OPENING line is build noise — a transcript that quotes a docker command deep in a
    # sample is a report, not pollution.
    quoted = _document(samples=2, body_size=100).replace("xx", "docker compose build penny", 1)
    assert build_noise_reason(quoted) is None


def test_an_oversized_single_fold_is_refused_not_cut() -> None:
    """One sample fold over GitHub's hard cap cannot be cut without breaking its markup, so it is
    refused loudly (naming the fold) rather than published broken; a normal split is accepted."""
    document = _document(samples=1, body_size=GITHUB_COMMENT_LIMIT + 1000)
    reason = unsplittable_reason(split_run_comment(document))
    assert reason is not None
    assert "over GitHub's 65536 cap" in reason
    assert unsplittable_reason(split_run_comment(_document(samples=8, body_size=25000))) is None


def test_cli_writes_parts_and_prints_their_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI writes ``part-NN.md`` in post order, prints the names, and the written files
    reassemble (headers stripped) to the input document."""
    document = _document(samples=8, body_size=25000)
    source = tmp_path / "body.md"
    source.write_text(document)
    out_dir = tmp_path / "comment"
    assert main([str(source), str(out_dir)]) == 0
    names = capsys.readouterr().out.split()
    assert names == ["part-01.md", "part-02.md", "part-03.md", "part-04.md"]
    written = [(out_dir / name).read_text() for name in names]
    assert "".join(part.split("\n\n", 1)[1] for part in written) == document


def test_cli_clears_stale_parts_from_a_previous_post(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A re-post after a re-assemble never leaves a stale part behind to be posted alongside the
    fresh ones."""
    out_dir = tmp_path / "comment"
    out_dir.mkdir()
    (out_dir / "part-09.md").write_text("stale")
    source = tmp_path / "body.md"
    source.write_text(_document(samples=2, body_size=100))
    assert main([str(source), str(out_dir)]) == 0
    capsys.readouterr()
    assert sorted(path.name for path in out_dir.glob("part-*.md")) == ["part-01.md"]


def test_cli_refuses_bad_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing document, an empty one, and a polluted one each fail loudly (exit 1) with an
    actionable message; a bad arg count is usage (exit 2)."""
    assert main([str(tmp_path / "nope.md"), str(tmp_path)]) == 1
    assert "no such document" in capsys.readouterr().err
    empty = tmp_path / "empty.md"
    empty.write_text("\n")
    assert main([str(empty), str(tmp_path)]) == 1
    assert "is empty" in capsys.readouterr().err
    polluted = tmp_path / "polluted.md"
    polluted.write_text(f"docker compose build penny\n{_document(samples=1, body_size=10)}")
    assert main([str(polluted), str(tmp_path)]) == 1
    assert "refusing to post" in capsys.readouterr().err
    assert main([str(empty)]) == 2
    assert capsys.readouterr().err.strip() == USAGE
