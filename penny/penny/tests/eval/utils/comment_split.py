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
sample fold to ``PART_BUDGET`` before this module sees the document, so a part over the hard cap
is now an INVARIANT VIOLATION rather than a case, and ``enforce_part_cap`` raises it as the
renderer bug it is instead of refusing the caller.

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


def split_run_comment(document: str, budget: int = PART_BUDGET) -> list[str]:
    """The postable comment parts for ``document``, in post order (the summary function): cut on
    sample-fold boundaries, packed to ``budget``, each headed ``report N of M``.

    A document that fits is returned unchanged as a single unheaded part."""
    return render_parts(partition_on_sample_folds(document, budget))


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
    (``report.bounded_sample_folds`` / ``report.bounded_representative``, #2135) and this module
    packs whole folds to that same budget. What the check still catches is the residue the
    budget does not price: a part is one fold plus whatever case-level material trails it before
    the next fold opens, and a case tail wide enough to spend the gap between the budget and the
    cap would land here."""
    for number, part in enumerate(parts, start=1):
        if len(part) > GITHUB_COMMENT_LIMIT:
            raise OversizedPartError(
                f"part {number} is {len(part)} characters — over GitHub's {GITHUB_COMMENT_LIMIT} "
                f"cap after every sample fold was bounded, so a fold came back over budget. It "
                f"opens {_part_opening(part)[:80]!r}."
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
