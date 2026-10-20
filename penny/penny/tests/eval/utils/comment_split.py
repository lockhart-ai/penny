r"""Posting a run comment that exceeds GitHub's 64K cap (#1808): the sample-fold splitter — and,
for the run summary, which carries no sample folds at all, the table-row splitter (#2256).

``make eval-report`` (#1757) posts a run's assembled comment (``assemble.py``) verbatim. GitHub
refuses an issue-comment body over 65,536 characters, and the recipe had no split path, so an
over-cap run could not be posted at all by the supported route::

    GraphQL: Body is too long (maximum is 65536 characters) (addComment)

That is not an edge case — one 8-sample chat beat assembles to ~290K *after* #1763's shared-prompt
hoisting, and ~172K of a 32-sample run is thinking blocks and transcript tables, all distinct. So
the answer is splitting, not more compression. This module is the split, and the rules that keep it
honest:

  * **Cut only on sample-fold boundaries.** The assembler emits every sample as one whole
    ``<details>`` fold (#1753), and a chunk that splits a fold renders as broken markup — so the
    ONLY legal seam is the blank line before a fold's opening tag (``report.SAMPLE_BLOCK_START``,
    single-sourced with the re-normalizer). Concatenating the parts reproduces the document byte
    for byte: nothing is dropped, summarised, or re-wrapped (collapsed-never-means-removed,
    #1753/#1759, applied to posting).
  * **Head each part** ``(report N of M — …; content verbatim)`` so a reader knows there is more
    and that nothing was cut out of it. A document that fits is ONE part and carries no header —
    there is no "more" to announce — so the common case posts exactly as it did before. The run
    header (identity · RESULT · gate · flips) opens the document, so it lands on part 1 by
    construction.
  * **Refuse a body that opens with build noise.** ``make assemble`` echoes its recipe and the
    whole ``docker compose build`` log to stdout, so piping IT (rather than invoking the
    containerized module directly) produces a comment opening with ``GIT_COMMIT=…`` /
    ``docker compose build penny`` / ``#1 [internal] …``. That happened: eight comments were
    posted and had to be deleted. A polluted body now fails loudly instead of publishing.

Budget is ~58K rather than the full 64K: the part header, and GitHub counting a body's length
differently from ``wc -c``, both eat into the cap.

The RUN SUMMARY (#2256) is a different document with a different legal seam: it is a run heading,
two headline readings, and a markdown table — one row per case, and on a whole-suite run (~90
cases) that table alone can pass the part budget. It carries no sample folds for the cut above to
find, so ``split_run_summary``/``partition_on_table_rows`` cut it at a table's ROW boundary
instead — always exactly one rendered line, since the cell helpers in ``report.py`` turn every
newline into ``<br>`` and every pipe into ``\|``. A part that resumes a table repeats its head +
separator and names which rows it holds, so concatenating the parts (headers/notes dropped)
reproduces every row exactly once.

A fold LARGER than one part used to end the whole post — a real run's 94K sample fold made an
entire gpt-oss report unpostable (#2135). It cannot end it any more: the renderer bounds every
sample fold to ``PART_BUDGET`` before this module sees the document. #2696 found the case that
remained: the representative fold is bounded, but the case's TAIL rides after it with no fold
boundary between, and fold + tail outgrew a comment (70,069 characters) — the invariant raised
and the run could not post at all. So a part the fold cut leaves over the hard cap is RESCUED
first: re-cut at its own WHOLE-BLOCK boundaries (``block_segments`` — the blank lines between
top-level blocks, legal only outside a fence, a fold, or a table) and packed to the budget like
any other part. A table block too big for a part is cut between ROWS (the run summary's row
splitter, head repeated on every continuation); a single block still over the cap is CUT,
ending in a line that names the run-dir file holding it whole — never a silent truncation, and
a caller with no artifact file to name still gets ``enforce_part_cap``'s raise.

Pure text in, text out — no model, no git, no network — so it is exercised by plain (non-eval)
tests in ``make check``, like ``checkpoint.py``.

CLI (invoked in-container by the Makefile's ``eval-report`` recipe, over the mounted artifact
home): ``python -m penny.tests.eval.utils.comment_split <document> <out_dir>`` writes the parts as
``part-01.md``, ``part-02.md``, … into ``<out_dir>`` and prints their file names, one per line, in
post order. Any refusal is an actionable stderr message and exit 1 — never a partial post.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from penny.tests.eval.utils.report import SAMPLE_BLOCK_START

# GitHub's hard limit on an issue-comment body. A part over this cannot be posted at all.
GITHUB_COMMENT_LIMIT = 65536
# What we actually pack to. The gap absorbs the part header and GitHub's own length accounting.
PART_BUDGET = 58000

PART_HEADER = "(report {number} of {total} — split for GitHub's 64K comment cap; content verbatim)"
# What that header OPENS with, derived from it rather than spelled a second time — what tells a
# part's own header from the content beneath it when the invariant has to name what overflowed.
_PART_HEADER_OPENS = PART_HEADER.split("{", 1)[0]

# The recipe/build lines `make assemble` echoes to stdout — the pollution a piped assemble puts at
# the TOP of the body. Matched against the OPENING line only (the structural reading of "opens
# with", and zero-false-positive: a real report opens with its run-identity line, while a browse
# transcript that happens to quote a docker command sits far below the opening).
BUILD_NOISE_MARKERS = ("docker compose", "GIT_COMMIT=", "#1 [internal]")

PART_FILENAME = "part-{number:02d}.md"
PART_GLOB = "part-*.md"

USAGE = "usage: python -m penny.tests.eval.utils.comment_split <document> <out_dir>"

_SAMPLE_FOLD_BOUNDARY = re.compile(rf"\n\n(?={SAMPLE_BLOCK_START})")


def split_run_comment(
    document: str, budget: int = PART_BUDGET, artifact_file: str | None = None
) -> list[str]:
    """The postable comment parts for ``document``, in post order (the summary function): cut on
    sample-fold boundaries, packed to ``budget``, each headed ``report N of M``.

    A part the fold cut leaves over GitHub's cap — no fold boundary left to pack on, the case
    tail glued after a fold (#2696) — is re-cut at whole-block boundaries (``_rescue_part``),
    and ``artifact_file`` (the run-dir file that holds the case's whole record) is what a
    last-resort block cut names.

    A document that fits is returned unchanged as a single unheaded part."""
    parts: list[str] = []
    for body in partition_on_sample_folds(document, budget):
        if len(body) > GITHUB_COMMENT_LIMIT:
            parts.extend(_rescue_part(body, budget, artifact_file))
        else:
            parts.append(body)
    return render_parts(parts)


# ── The cut (sample folds only) ──────────────────────────────────────────────
def sample_fold_segments(document: str) -> list[str]:
    """``document`` cut at every sample-fold boundary — the ONLY legal seam (a cut inside a
    ``<details>`` renders as broken markup).

    Each segment carries the blank line that FOLLOWS it, so ``"".join(segments)`` is the document
    byte for byte. The leading segment is everything before the first fold: the run header, and the
    first case's heading + hoisted shared prompt."""
    cuts = [match.end() for match in _SAMPLE_FOLD_BOUNDARY.finditer(document)]
    edges = [0, *cuts, len(document)]
    return [document[start:end] for start, end in pairwise(edges) if start < end]


def partition_on_sample_folds(document: str, budget: int = PART_BUDGET) -> list[str]:
    """``document`` packed greedily into parts of at most ``budget`` characters, cut only on sample
    folds. ``"".join(...)`` reproduces the document exactly.

    A single fold larger than ``budget`` becomes its own over-budget part rather than being cut —
    the seam rule outranks the budget. Since #2135 the renderer bounds every fold to that same
    budget before this ever sees one, so that case is the invariant ``enforce_part_cap`` holds
    rather than a shape to expect."""
    parts: list[str] = []
    current = ""
    for segment in sample_fold_segments(document):
        if current and len(current) + len(segment) > budget:
            parts.append(current)
            current = ""
        current += segment
    if current:
        parts.append(current)
    return parts


def render_parts(bodies: list[str]) -> list[str]:
    """Head each body ``(report N of M — …)``. A lone body is returned unheaded — there is no
    "more" to announce, so a run that fits posts byte-identically to before the splitter existed."""
    if len(bodies) <= 1:
        return list(bodies)
    total = len(bodies)
    return [
        f"{PART_HEADER.format(number=number, total=total)}\n\n{body}"
        for number, body in enumerate(bodies, start=1)
    ]


# ── The rescue: an over-cap part re-cut at whole-block boundaries (#2696) ─────
#
# The fold cut packs whole seam-segments, and a segment can be over the cap WITHOUT a fold
# left to cut on: the representative fold is bounded to the budget (#2135), but the case's
# tail — `#### Test inputs`, `#### Outliers` — is glued after it by the assembler with no
# seam between, and fold + tail outgrew a comment on a real run (a 44.8K bounded fold plus a
# 24.8K tail). ``partition_on_sample_folds`` cannot split that, and ``enforce_part_cap``
# refused it, so the run could not post at all by the supported route.
#
# The rescue re-cuts such a part at its OWN whole-block boundaries: the blank line between
# two top-level blocks, legal only when no fence, fold, or table is open. Blocks carry their
# trailing blank line, so joining them reproduces the part byte for byte — the same no-drop
# rule the fold cut holds. A table block too big for a part is cut BETWEEN ROWS by the run
# summary's row splitter, which repeats the table's head + separator on every continuation; a
# table that fits is never split, so no header is ever repeated for one.
#
# When even a single block is over the cap — no legal seam left anywhere — the block is CUT
# on a line boundary (closing whatever the cut leaves open, so the part still renders) and
# the cut part ends with a line naming the run-dir file that holds the block whole. Never a
# silent truncation: a caller with no artifact file to name gets the over-cap part back, and
# ``enforce_part_cap`` raises on it exactly as before.

_FENCE = "```"
_DETAILS_OPENS = "<details>"
_DETAILS_CLOSES = "</details>"

# The one line a last-resort cut leaves behind: what is gone, and where the whole block is.
OVERSIZE_CUT_MARKER = (
    "_Cut for GitHub's comment cap: {elided:,} characters of this block are not shown here — "
    "the whole block is in the run artifact: `{artifact}`._"
)
# Priced off the WIDEST marker the format can render (twelve elision digits, a 300-character
# artifact path), so the cut budget never leaves the marker room to overflow the cap.
_CUT_RESERVE = len(OVERSIZE_CUT_MARKER.format(elided=999_999_999_999, artifact="x" * 300)) + len(
    "\n\n"
)


def _block_state_step(line: str, depth: int, fence: bool) -> tuple[int, bool]:
    """One line's effect on the (fold depth, fence) state a block cut reads: a fence toggles,
    a line-start ``<details>``/``</details>`` pair opens and closes a fold. The FENCE is read
    first — prompt content inside a fence can carry ``</details>``-looking lines, and reading
    the fold state first miscounts the depth exactly as the real run that motivated this
    showed. Table rows never carry state: their cells render on one line, so a line starting
    ``|`` is never a fold edge."""
    if fence:
        return depth, not line.startswith(_FENCE)
    if line.startswith(_FENCE):
        return depth, True
    if line.startswith(_DETAILS_OPENS):
        return depth + 1, fence
    if line.startswith(_DETAILS_CLOSES) and depth:
        return depth - 1, fence
    return depth, fence


def block_segments(text: str) -> list[str]:
    """``text`` cut into whole top-level blocks at the blank lines between them — the seams the
    fold cut cannot use when a part carries no fold boundary of its own (#2696). A cut is legal
    only where no structural block is open: never inside a table (its rows are contiguous ``|``
    lines, so no blank line ever falls inside one), a fence, or an open ``<details>`` fold.

    Each segment carries the blank line that FOLLOWS it, so ``"".join(block_segments(text))``
    is ``text`` byte for byte."""
    lines = text.split("\n")
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line) + 1
    segments: list[str] = []
    seg_open = 0
    depth, fence = 0, False
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line and depth == 0 and not fence:
            end = index
            while end + 1 < len(lines) and not lines[end + 1]:
                end += 1
            segments.append(text[starts[seg_open] : starts[end] + len(lines[end]) + 1])
            seg_open = end + 1
            index = end + 1
            continue
        depth, fence = _block_state_step(line, depth, fence)
        index += 1
    if seg_open < len(lines):
        segments.append(text[starts[seg_open] :])
    return segments


def _pack_blocks(blocks: Sequence[str], budget: int) -> list[str]:
    """``blocks`` packed greedily into parts of at most ``budget`` characters, order kept and
    nothing dropped — the block-level twin of ``partition_on_sample_folds``. Blocks carry their
    own separators, so joining is plain concatenation. A single block over budget becomes its
    own over-budget part: the seam rule outranks the budget here exactly as it does at fold
    level."""
    parts: list[str] = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) > budget:
            parts.append(current)
            current = ""
        current += block
    if current:
        parts.append(current)
    return parts


def _is_table_block(block: str) -> bool:
    """Whether ``block`` opens with a markdown table: a row followed by its own separator."""
    return _table_start(block.split("\n"), 0)


def _is_heading_block(block: str) -> bool:
    """Whether ``block`` is nothing but a heading: every non-blank line starts with ``#``."""
    lines = [line for line in block.splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("#") for line in lines)


def _attach_headings(blocks: Sequence[str]) -> list[str]:
    """A heading block rides with the block it introduces — a part that ends on a bare
    heading dangles it over content that opens in the next comment. Boundaries stay legal:
    the cut between a heading and what follows is dropped, not moved somewhere unsafe."""
    merged: list[str] = []
    for block in blocks:
        if merged and _is_heading_block(merged[-1]):
            merged[-1] += block
        else:
            merged.append(block)
    return merged


def _split_table_block(block: str, budget: int) -> list[str]:
    """A table block over budget, cut BETWEEN ROWS by the run summary's row splitter — every
    continuation repeats the table's head + separator (and its continued-rows note), so no
    bare cells ever render. The block's trailing blank line travels with the last row part,
    so packing the row parts back into the block stream stays byte-exact."""
    body = block.rstrip("\n")
    trailing = block[len(body) :]
    row_parts = partition_on_table_rows(body, budget)
    return [
        part + ("\n\n" if index < len(row_parts) - 1 else trailing)
        for index, part in enumerate(row_parts)
    ]


def _rescue_part(part: str, budget: int, artifact_file: str | None) -> list[str]:
    """An over-cap part re-cut at whole-block boundaries (the summary function of the rescue):
    ``block_segments`` → greedy pack, with a table block over budget cut between its rows.
    Packing cannot produce an over-cap part except from a single block, and that is the
    last-resort cut — or, with no artifact file to name, the part is left for
    ``enforce_part_cap`` to raise on."""
    blocks: list[str] = []
    for block in block_segments(part):
        if len(block) > budget and _is_table_block(block):
            blocks.extend(_split_table_block(block, budget))
        else:
            blocks.append(block)
    parts = _pack_blocks(_attach_headings(blocks), budget)
    return [
        _cut_over_cap(over, artifact_file) if len(over) > GITHUB_COMMENT_LIMIT else over
        for over in parts
    ]


def _cut_over_cap(block: str, artifact_file: str | None) -> str:
    """A single block still over GitHub's cap after every legal seam was used, CUT on a line
    boundary — never silently: the cut part ends with a line naming the run-dir file that
    holds the block whole, and whatever the cut left open (a fence, a fold) is closed so the
    part still renders. A caller with no artifact file to name gets the block back unchanged,
    leaving ``enforce_part_cap`` to raise — a refusal beats content that points nowhere."""
    if artifact_file is None:
        return block
    budget = GITHUB_COMMENT_LIMIT - _CUT_RESERVE
    lines = block.split("\n")
    kept: list[str] = []
    depth, fence, size = 0, False, 0
    for line in lines:
        if kept and size + len(line) + 1 > budget:
            break
        kept.append(line)
        size += len(line) + 1
        depth, fence = _block_state_step(line, depth, fence)
    if not kept:  # the first line alone is over the cut budget — hard-cut it mid-line
        kept, depth, fence = [lines[0][:budget]], 0, False
    if fence:
        kept.append(_FENCE)
    kept.extend([_DETAILS_CLOSES] * depth)
    elided = len(block) - size
    marker = OVERSIZE_CUT_MARKER.format(elided=elided, artifact=artifact_file)
    return "\n".join(kept).rstrip("\n") + f"\n\n{marker}\n"


# ── The cut for the RUN SUMMARY: a markdown table's row boundary (#2256) ──────
#
# The run summary carries no sample folds at all — it is a run heading, two headline readings,
# and a markdown table, one row per case. On a whole-suite run (~90 cases) that table alone can
# pass the part budget, and the sample-fold cut above has nothing to find in it. A table ROW is
# this document's own legal seam: the cell helpers in ``report.py`` turn every newline into
# ``<br>`` and every pipe into ``\|``, so a row never spans two rendered lines, and a cut between
# lines can therefore never land INSIDE one. A part that resumes a table repeats its head +
# separator and names which rows it holds, so a reader opening a later part is never shown bare
# cells with no columns — and concatenating every part (headers/notes dropped) reproduces every
# row exactly once.

_TABLE_ROW_RE = re.compile(r"^\|.*\|$")
_TABLE_SEP_RE = re.compile(r"^\|[ \t:|-]+\|$")
# Stated once a part RESUMES a table mid-way — never on the table's first row in a part, which
# needs no "continued" note because nothing of it was shown before.
ROW_RANGE_NOTE = "_(continued — rows {start}–{end} of {total})_"
# The budget check runs BEFORE the note's real text is known (its END row is only settled once
# the part stops filling, at flush time — see ``_TablePacker.flush``), so the placeholder padded
# into the part for that check must be at least as long as any note this format can ever render
# — otherwise a part that fit the PLACEHOLDER could still overflow once the real (longer) note
# replaced it. Six-digit row numbers is a generous ceiling no real run approaches.
_ROW_RANGE_RESERVE = len(ROW_RANGE_NOTE.format(start=999999, end=999999, total=999999))


def _table_start(lines: Sequence[str], index: int) -> bool:
    """Whether ``lines[index]`` opens a markdown table: itself a row, followed by its own
    ``|---|---|`` separator."""
    return (
        index + 1 < len(lines)
        and bool(_TABLE_ROW_RE.match(lines[index]))
        and bool(_TABLE_SEP_RE.match(lines[index + 1]))
    )


def _table_block(lines: Sequence[str], index: int) -> tuple[str, str, list[str], int]:
    """The table opening at ``index``: its head, its separator, every data row, and the index
    of the first line past its last row."""
    head, sep = lines[index], lines[index + 1]
    end = index + 2
    rows: list[str] = []
    while end < len(lines) and _TABLE_ROW_RE.match(lines[end]):
        if _TABLE_SEP_RE.match(lines[end]):
            break
        rows.append(lines[end])
        end += 1
    return head, sep, rows, end


@dataclass
class _TablePacker:
    """Packs a document into parts of at most ``budget`` characters, cutting only at a table's
    ROW boundary (above). The state is the part under construction (``current``), which table
    (if any) is OPEN at its tail (so a row that continues it needs no repeated header), and a
    "continued" note PENDING for the table a just-opened continuation belongs to — patched in at
    flush time, once the part's actual last row for that table is known."""

    budget: int
    parts: list[str] = field(default_factory=list)
    current: list[str] = field(default_factory=list)
    open_header: tuple[str, str] | None = None
    last_row: dict[tuple[str, str], int] = field(default_factory=dict)
    pending_notes: list[tuple[str, tuple[str, str], int, int]] = field(default_factory=list)
    _tokens: int = 0

    def _joined(self, extra: Sequence[str] = ()) -> str:
        return "\n".join([*self.current, *extra])

    def flush(self) -> None:
        """Close the part under construction, patching in every pending row-range note now
        that each one's table's LAST row in this part is known."""
        if self.current:
            text = self._joined()
            for token, header, start, total in self.pending_notes:
                end = self.last_row.get(header, start)
                note = ROW_RANGE_NOTE.format(start=start, end=end, total=total)
                text = text.replace(token, note, 1)
            self.parts.append(text)
        self.current = []
        self.open_header = None
        self.last_row = {}
        self.pending_notes = []

    def _open_candidate(
        self, head: str, sep: str, row_number: int, row: str
    ) -> tuple[list[str], str | None]:
        """The lines a row needs when it must OPEN its table's header in this part — plus the
        placeholder token for its row-range note, for a CONTINUATION (``row_number`` > 1).

        Padded with trailing NULs to :data:`_ROW_RANGE_RESERVE`, so the budget check below sees
        the WORST CASE the real note can ever render to — never a short stand-in the final,
        longer text could later overflow past."""
        candidate = [head, sep]
        token = None
        if row_number > 1:
            self._tokens += 1
            token = f"\x00ROW_RANGE_{self._tokens}\x00".ljust(_ROW_RANGE_RESERVE, "\x00")
            candidate.append(token)
        candidate.append(row)
        return candidate, token

    def add_row(self, head: str, sep: str, row_number: int, row: str, total: int) -> None:
        """Pack one table row, repeating the header (plus a pending note) if it opens a part."""
        header = (head, sep)
        if self.open_header == header:
            candidate, token = [row], None
        else:
            candidate, token = self._open_candidate(head, sep, row_number, row)
        if self.current and len(self._joined(candidate)) > self.budget:
            self.flush()
            candidate, token = self._open_candidate(head, sep, row_number, row)
        self.current.extend(candidate)
        if candidate[0] == head:
            self.open_header = header
            if token:
                self.pending_notes.append((token, header, row_number, total))
        self.last_row[header] = row_number

    def add_line(self, line: str) -> None:
        """Pack one non-table line — no header to repeat, so a cut here is a plain line cut."""
        if self.current and len(self._joined([line])) > self.budget:
            self.flush()
        self.current.append(line)
        self.open_header = None


def partition_on_table_rows(document: str, budget: int = PART_BUDGET) -> list[str]:
    """``document`` packed into parts of at most ``budget`` characters, cut only at a markdown
    table's ROW boundary — the run summary's own legal seam, since it carries no sample folds.
    ``"".join`` after dropping the injected headers/notes reproduces every row exactly once."""
    lines = document.split("\n")
    packer = _TablePacker(budget=budget)
    index = 0
    while index < len(lines):
        if _table_start(lines, index):
            head, sep, rows, end = _table_block(lines, index)
            for offset, row in enumerate(rows, start=1):
                packer.add_row(head, sep, offset, row, len(rows))
            index = end
            continue
        packer.add_line(lines[index])
        index += 1
    packer.flush()
    return packer.parts


def split_run_summary(document: str, budget: int = PART_BUDGET) -> list[str]:
    """The run summary's postable parts (the summary function): cut on table-row boundaries,
    headed ``report N of M`` exactly like a sample-fold split. A document that fits is returned
    unheaded — byte-identical to before this existed."""
    return render_parts(partition_on_table_rows(document, budget))


# ── The guards (refuse loudly, never publish) ────────────────────────────────
def build_noise_reason(document: str) -> str | None:
    """The actionable reason a document is refused for opening with build noise, or ``None`` when
    it is clean. Guards the trap a manual workaround falls into: piping ``make assemble`` (which
    echoes its recipe and the whole docker build log) instead of invoking the module directly."""
    opening = next((line.strip() for line in document.splitlines() if line.strip()), "")
    for marker in BUILD_NOISE_MARKERS:
        if opening.startswith(marker):
            return (
                f"the report opens with build noise ({marker!r} on line {opening!r}) — that is "
                f"`make assemble`'s recipe/build log, not the report. Capture the body by invoking "
                f"`python -m penny.tests.eval.utils.assemble <run_dir>` in the container "
                f"directly, never "
                f"`make assemble` piped through stripping."
            )
    return None


class OversizedPartError(RuntimeError):
    """A part still over GitHub's cap after the renderer bounded every sample fold.

    A RENDERER BUG rather than a caller error, which is why it is raised where it is found
    instead of returned as a refusal: nothing the caller can do fixes it. It was a refusal once
    (#1808), and the two remedies that refusal implied were both wrong — a lower ``EVAL_SAMPLES``
    changes the N a ceiling is keyed to, and re-rolling for a smaller fold spends money to hide
    a finding. Since #2135 an oversized fold is REDUCED instead, so reaching this means a fold
    came back over its budget."""


def enforce_part_cap(parts: Sequence[str]) -> None:
    """The invariant every posted part holds: it fits GitHub's hard cap.

    It holds by construction — the assembler bounds every sample fold to ``PART_BUDGET``
    (``report.bounded_sample_folds`` / ``report.bounded_representative``, #2135), this module
    packs whole folds to that same budget, and a part the fold cut still leaves over the cap is
    re-cut at whole-block boundaries (``_rescue_part``, #2696). What the check still catches is
    a part the rescue could not fix because nothing named where the content lives: a document
    posted with no ``artifact_file`` (the CLI, the run summary) whose single over-cap block
    would have had to be cut silent."""
    for number, part in enumerate(parts, start=1):
        if len(part) > GITHUB_COMMENT_LIMIT:
            raise OversizedPartError(
                f"part {number} is {len(part)} characters — over GitHub's {GITHUB_COMMENT_LIMIT} "
                f"cap after every sample fold was bounded and the block rescue ran, so a block "
                f"came back over cap with no artifact file to name. It opens "
                f"{_part_opening(part)[:80]!r}."
            )


def _part_opening(part: str) -> str:
    """The first line of a part's own CONTENT — past the ``report N of M`` header a split part
    carries, which is this module's own line and says nothing about what overflowed."""
    lines = [line for line in part.strip().splitlines() if line.strip()]
    body = [line for line in lines if not line.startswith(_PART_HEADER_OPENS)]
    return body[0] if body else ""


# ── CLI: python -m penny.tests.eval.utils.comment_split <document> <out_dir> ────────
def write_parts(parts: list[str], out_dir: Path) -> list[str]:
    """Write ``parts`` into ``out_dir`` as ``part-01.md``… and return their file names in post
    order. Pre-existing part files are cleared first, so a re-post after a re-assemble can never
    leave a stale part behind to be posted alongside the fresh ones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(PART_GLOB):
        stale.unlink()
    names: list[str] = []
    for number, part in enumerate(parts, start=1):
        name = PART_FILENAME.format(number=number)
        (out_dir / name).write_text(part)
        names.append(name)
    return names


def main(argv: list[str]) -> int:
    """Split the document at ``argv[0]`` into ``argv[1]``, printing each part's file name. Refuses
    (exit 1, actionable stderr) an unreadable or empty document and a body opening with build
    noise. Bad args → 2. A part still over the cap is the renderer's bug, not the caller's, so
    ``enforce_part_cap`` raises rather than refusing."""
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    document_path, out_dir = Path(argv[0]), Path(argv[1])
    if not document_path.is_file():
        print(f"comment_split: no such document: {document_path}", file=sys.stderr)
        return 1
    document = document_path.read_text()
    if not document.strip():
        print(f"comment_split: {document_path} is empty — nothing to post", file=sys.stderr)
        return 1
    reason = build_noise_reason(document)
    if reason:
        print(f"comment_split: refusing to post — {reason}", file=sys.stderr)
        return 1
    parts = split_run_comment(document)
    enforce_part_cap(parts)
    for name in write_parts(parts, out_dir):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
