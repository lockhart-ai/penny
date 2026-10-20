"""Plain (non-eval) tests for the run-comment splitter (``comment_split.py``, #1808/#2256).

They drive the pure text helpers + the CLI over a SYNTHETIC oversized document built from the real
fold primitives (``report.fold_sample``), so they run inside ``make check``: no model, no GPU, no
container, no artifacts. The four properties the ticket names are asserted directly — the part
count, that no ``<details>`` straddles a boundary, that part 1 carries the run header, and that
reassembling the parts is byte-identical to the input — plus the build-noise refusal and the
over-the-cap INVARIANT (#2135), which is raised as a renderer bug rather than refused.

The run-summary splitter (#2256, below) gets its own synthetic fixture — a bare markdown table,
with no sample-fold machinery at all — because it cuts on a different seam (a table ROW) and
its reassembly invariant is weaker: a continuation part repeats the table's head + separator, so
concatenating parts does not reproduce the document byte for byte the way a sample-fold split
does. What IS invariant: every row stays under budget and appears in exactly one part.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from penny.tests.eval.utils.comment_split import (
    GITHUB_COMMENT_LIMIT,
    OVERSIZE_CUT_MARKER,
    PART_HEADER,
    USAGE,
    OversizedPartError,
    block_segments,
    build_noise_reason,
    enforce_part_cap,
    main,
    partition_on_sample_folds,
    partition_on_table_rows,
    sample_fold_segments,
    split_run_comment,
    split_run_summary,
)
from penny.tests.eval.utils.report import fold_sample

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
        assert "python -m penny.tests.eval.utils.assemble" in reason
    assert build_noise_reason(_document(samples=2, body_size=100)) is None
    # Only the OPENING line is build noise — a transcript that quotes a docker command deep in a
    # sample is a report, not pollution.
    quoted = _document(samples=2, body_size=100).replace("xx", "docker compose build penny", 1)
    assert build_noise_reason(quoted) is None


def test_a_part_over_the_cap_is_raised_as_the_renderer_bug_it_is() -> None:
    """Since #2135 the renderer bounds every sample fold, so a part over the hard cap means a
    fold came back over budget — a renderer bug, not a caller error, and nothing the caller
    could do about it (a lower ``EVAL_SAMPLES`` changes the N a ceiling is keyed to, and a
    re-roll spends money to hide a finding).

    The message names the fold that overflowed rather than the part header this module wrote
    above it, which is what the refusal it replaced quoted."""
    document = _document(samples=1, body_size=GITHUB_COMMENT_LIMIT + 1000)
    with pytest.raises(OversizedPartError) as raised:
        enforce_part_cap(split_run_comment(document))
    assert "over GitHub's 65536 cap" in str(raised.value)
    assert "<details><summary>sample 1 — " in str(raised.value)
    assert PART_HEADER.split("{", 1)[0] not in str(raised.value)
    enforce_part_cap(split_run_comment(_document(samples=8, body_size=25000)))


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


# ── The block rescue: an over-cap part re-cut at whole-block boundaries (#2696) ──────
#
# A real run's case comment carried its bounded representative fold (44.8K) with the case's
# tail — `#### Test inputs`, `#### Outliers` — glued after it and NO fold boundary between, so
# the fold cut emitted one uncuttable 70K part and the run could not post. These drive the
# rescue over a synthetic document of that shape, built from the real fold primitives, with
# the trap the real prompt folds carry: a ``</details>``-looking line inside a fence must not
# open or close a fold for the cut to find.

_TAIL_HEADING = "#### Test inputs"
_REPRESENTATIVE_HEADING = "#### Representative sample"
_ARTIFACT_FILE = "run-20261020T035935Z-39260-c3dd5ae3/digest-set-up-in-chat-carries-every-item.md"


def _representative_document(fold_size: int, tail_size: int) -> str:
    """A case comment shaped like the run that failed (#2696): the run header, a case heading,
    the representative section (heading + one fold whose body carries a fenced prompt with
    ``</details>``-looking lines at line start, plus a nested fold), then the case tail — its
    own heading and fold. The representative fold's summary is the sample's own banner — the
    shape ``titled_fold`` renders — so the ONLY fold seam in the document is the representative
    heading, and fold + tail assemble into ONE segment the fold cut cannot split."""
    nested = "\n\n".join(
        [
            "<details><summary>System prompts — 3 contexts · 1 shared by every sample</summary>",
            "```\nYou are Penny.\n</details> — quoted markup at line start, inside the fence\n```",
            "<details><summary>a nested fold</summary>\n\ninner\n\n</details>",
            "</details>",
        ]
    )
    rep_fold = "<details><summary>❌ fail · behavioral · 231s · 11 calls</summary>\n\n" + (
        f"{nested}\n\n{'x' * fold_size}\n\n</details>"
    )
    tail_fold = (
        "<details><summary>Phrasings — 5 wordings of one ask</summary>\n\n"
        f"{'y' * tail_size}\n\n</details>"
    )
    return (
        "\n\n".join(
            [
                RUN_HEADER,
                CASE_HEADING,
                _REPRESENTATIVE_HEADING,
                rep_fold,
                _TAIL_HEADING,
                tail_fold,
                FOOTER,
            ]
        )
        + "\n"
    )


def _fold_balance(text: str) -> int:
    """The fold balance the splitter itself reads: line-START ``<details>``/``</details>``
    edges, with a fence's content ignored exactly as ``_block_state_step`` ignores it — a
    ``</details>``-looking line inside a fence must not read as an unclosed fold."""
    depth, fence = 0, False
    for line in text.splitlines():
        if fence:
            fence = not line.startswith("```")
        elif line.startswith("```"):
            fence = True
        elif line.startswith("<details>"):
            depth += 1
        elif line.startswith("</details>"):
            depth -= 1
    return depth


def test_block_segments_cut_only_on_legal_blank_lines() -> None:
    """The segments are whole blocks: every fold, fence, and heading stays in one segment, the
    fenced ``</details>``-looking line never opens or closes a fold for the cut, and the
    segments concatenate back to the text byte for byte."""
    text = _representative_document(fold_size=100, tail_size=100)
    segments = block_segments(text)
    assert "".join(segments) == text
    for segment in segments:
        assert _fold_balance(segment) == 0
        assert segment.count("```") % 2 == 0
    rep = next(segment for segment in segments if segment.startswith("<details><summary>❌ fail"))
    assert "</details> — quoted markup" in rep, "the fence's quoted markup rode inside the fold"
    assert "<details><summary>a nested fold</summary>" in rep, "so did the fold nested inside it"


def test_an_over_cap_part_is_rescued_on_whole_block_boundaries() -> None:
    """A part over the cap with no fold boundary left to pack on — a bounded representative
    fold plus a glued case tail — is re-cut at the part's own block boundaries: every part
    under the hard cap, structurally whole, and the bodies rejoin byte-identically to the
    input (nothing dropped, nothing rewritten)."""
    document = _representative_document(fold_size=45000, tail_size=24000)
    parts = split_run_comment(document)
    assert len(parts) >= 2, "the fixture genuinely outgrew one part"
    assert all(len(part) <= GITHUB_COMMENT_LIMIT for part in parts)
    assert "".join(part.split("\n\n", 1)[1] for part in parts) == document
    for part in parts:
        assert _fold_balance(part) == 0, "no fold is cut in half"
        assert part.count("```") % 2 == 0, "no fence is left open"
    assert _TAIL_HEADING in parts[-1], (
        "the tail rides after the fold it was glued to, not before it"
    )
    for part in parts:
        last = [line for line in part.splitlines() if line.strip()][-1]
        assert last != _TAIL_HEADING, "the tail heading rides with its fold, never dangles"
    assert any(part.split("\n\n", 1)[-1].startswith(_REPRESENTATIVE_HEADING) for part in parts), (
        "a heading opens the part that carries what it introduces"
    )


def test_a_table_block_over_the_budget_is_cut_between_rows_with_its_header_repeated() -> None:
    """A table block too big for a part is cut BETWEEN its rows by the run summary's row
    splitter: every part holding rows opens its table with the head + separator, every row
    appears exactly once in order, and a continuation carries the continued-rows note. A table
    that fits is never split, so no header is ever repeated for one."""
    head, sep = "| # | note |", "|---|---|"
    table = "\n".join([head, sep, *(f"| {n} | {'x' * 2400} |" for n in range(1, 31))]) + "\n"
    document = (
        "\n\n".join([RUN_HEADER, CASE_HEADING, fold_sample(1, "✅ pass", "small"), table, FOOTER])
        + "\n"
    )
    parts = split_run_comment(document)
    assert len(parts) >= 2
    assert all(len(part) <= GITHUB_COMMENT_LIMIT for part in parts)
    seen = [int(n) for part in parts for n in re.findall(r"^\| (\d+) \|", part, flags=re.MULTILINE)]
    assert seen == list(range(1, 31)), "every row exactly once, in order"
    for part in parts:
        rows = part.splitlines()
        first = next((i for i, line in enumerate(rows) if line.startswith("|")), None)
        if first is not None:
            assert rows[first] == head and rows[first + 1] == sep, (
                "a table never renders bare cells"
            )
    assert any("continued" in part for part in parts), "a continuation says which rows it holds"


def test_a_single_block_still_over_the_cap_is_cut_and_names_its_file() -> None:
    """One block with no seam left anywhere — a single giant fold — is cut on a line boundary:
    the cut part fits the cap, closes the fold it leaves open, and ends in a line naming the
    run-dir file that holds the block whole. With no artifact file to name, the part is left
    over the cap and the invariant raises, exactly as before the rescue."""
    body = "\n".join(f"transcript line {i}: {'x' * 90}" for i in range(700))
    document = "\n\n".join([RUN_HEADER, CASE_HEADING, fold_sample(1, "❌ fail", body)]) + "\n"
    parts = split_run_comment(document, artifact_file=_ARTIFACT_FILE)
    assert len(parts) >= 2
    assert all(len(part) <= GITHUB_COMMENT_LIMIT for part in parts)
    cut = parts[-1]
    assert _ARTIFACT_FILE in cut, "the cut names the file that holds the block whole"
    assert OVERSIZE_CUT_MARKER.split("{", 1)[0] in cut
    assert _fold_balance(cut) == 0, "the cut closed the fold it left open"
    assert cut.count("```") % 2 == 0
    with pytest.raises(OversizedPartError):
        enforce_part_cap(split_run_comment(document))


# ── The run-summary splitter: a table's ROW boundary (#2256) ─────────────────
#
# The run summary carries no sample folds — it's a run heading, two headline readings, and a
# markdown table, one row per case — so these drive the splitter over a SYNTHETIC table built
# from the same head/separator/row shape ``assemble.py`` renders, with no sample machinery at
# all. The real integration (splitting an actual 120-case ``render_run_summary`` output) is
# ``test_assemble.py::test_the_run_summary_splits_a_coverage_table_too_big_for_one_comment``.

_ROW_RANGE_RE = re.compile(r"_\(continued — rows (\d+)–(\d+) of (\d+)\)_")


def _bare_table(rows: int, cell_size: int) -> str:
    """A markdown table alone — head, separator, ``rows`` data rows each padded to roughly
    ``cell_size`` characters — with no heading or prose around it."""
    head, sep = "| # | case | note |", "|---|---|---|"
    body = "\n".join(f"| {n} | `case-{n}` | {'x' * cell_size} |" for n in range(1, rows + 1))
    return f"{head}\n{sep}\n{body}\n"


def test_table_rows_split_fit_the_budget_and_cover_every_row_once() -> None:
    """A table too big for one part is cut only at a ROW boundary: every part fits the budget,
    and every case id the table names appears in exactly one part, in order."""
    document = _bare_table(rows=30, cell_size=600)
    parts = partition_on_table_rows(document, budget=4000)
    assert len(parts) > 1, "the fixture is genuinely oversized for this budget"
    assert all(len(part) <= 4000 for part in parts)
    seen = [int(n) for part in parts for n in re.findall(r"`case-(\d+)`", part)]
    assert seen == list(range(1, 31)), "every row exactly once, in the order it was driven"


def test_a_continuation_part_repeats_the_header_and_states_the_rows_it_holds() -> None:
    """A part that RESUMES a table opens with the same head + separator the first part did, and
    a note whose claimed row range matches the case ids actually present in that part. The
    first part carries no such note — nothing came before it to be a continuation of."""
    document = _bare_table(rows=30, cell_size=600)
    parts = partition_on_table_rows(document, budget=4000)
    head, sep = document.splitlines()[0], document.splitlines()[1]
    assert parts[0].startswith(f"{head}\n{sep}\n")
    assert "continued" not in parts[0]
    for part in parts[1:]:
        assert part.startswith(f"{head}\n{sep}\n")
        match = _ROW_RANGE_RE.search(part)
        assert match is not None, "a continuation states which rows it holds"
        start, end, total = (int(group) for group in match.groups())
        assert total == 30
        case_ids = [int(n) for n in re.findall(r"`case-(\d+)`", part)]
        assert case_ids == list(range(start, end + 1)), "the stated range matches what's there"


def test_no_placeholder_tokens_leak_into_a_split_table() -> None:
    """The row-range note is patched in at flush time from a placeholder token — a bug there
    would leak the raw token into the posted text instead of the note."""
    document = _bare_table(rows=40, cell_size=800)
    parts = partition_on_table_rows(document, budget=3500)
    assert len(parts) > 2, "the fixture spans more than two parts, exercising several patches"
    assert all("\x00" not in part for part in parts)


def test_preamble_before_the_table_is_carried_once_and_the_table_still_splits() -> None:
    """Non-table content (the run heading, the headline readings) is plain text to the packer —
    carried once, in the first part, never repeated — and a table after it still splits on its
    own row boundaries."""
    preamble = "## Run summary\n\n### `model`\n\nsome prose"
    document = f"{preamble}\n\n{_bare_table(rows=20, cell_size=400)}"
    parts = partition_on_table_rows(document, budget=3000)
    assert parts[0].startswith(preamble)
    assert sum(part.count("## Run summary") for part in parts) == 1
    seen = [int(n) for part in parts for n in re.findall(r"`case-(\d+)`", part)]
    assert seen == list(range(1, 21))


def test_split_run_summary_heads_parts_like_a_sample_fold_split() -> None:
    """The run summary's parts are headed exactly like a sample-fold split's — same wording,
    same numbering — so a reader sees one posting convention everywhere in the suite."""
    document = _bare_table(rows=30, cell_size=600)
    parts = split_run_summary(document, budget=4000)
    total = len(parts)
    assert total > 1
    assert [part.splitlines()[0] for part in parts] == [
        f"(report {number} of {total} — split for GitHub's 64K comment cap; content verbatim)"
        for number in range(1, total + 1)
    ]


def test_a_table_that_fits_is_one_unheaded_part() -> None:
    """Under the budget there is no "more" to announce, so a small run summary posts exactly as
    it did before the row splitter existed — byte-identical, no header, no repeated columns."""
    document = _bare_table(rows=3, cell_size=20)
    assert split_run_summary(document) == [document]
