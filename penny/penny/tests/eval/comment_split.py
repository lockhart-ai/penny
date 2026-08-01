"""Posting a run comment that exceeds GitHub's 64K cap (#1808): the sample-fold splitter.

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

Pure text in, text out — no model, no git, no network — so it is exercised by plain (non-eval)
tests in ``make check``, like ``checkpoint.py``.

CLI (invoked in-container by the Makefile's ``eval-report`` recipe, over the mounted artifact
home): ``python -m penny.tests.eval.comment_split <document> <out_dir>`` writes the parts as
``part-01.md``, ``part-02.md``, … into ``<out_dir>`` and prints their file names, one per line, in
post order. Any refusal is an actionable stderr message and exit 1 — never a partial post.
"""

from __future__ import annotations

import re
import sys
from itertools import pairwise
from pathlib import Path

from penny.tests.eval.report import SAMPLE_BLOCK_START

# GitHub's hard limit on an issue-comment body. A part over this cannot be posted at all.
GITHUB_COMMENT_LIMIT = 65536
# What we actually pack to. The gap absorbs the part header and GitHub's own length accounting.
PART_BUDGET = 58000

PART_HEADER = "(report {number} of {total} — split for GitHub's 64K comment cap; content verbatim)"

# The recipe/build lines `make assemble` echoes to stdout — the pollution a piped assemble puts at
# the TOP of the body. Matched against the OPENING line only (the structural reading of "opens
# with", and zero-false-positive: a real report opens with its run-identity line, while a browse
# transcript that happens to quote a docker command sits far below the opening).
BUILD_NOISE_MARKERS = ("docker compose", "GIT_COMMIT=", "#1 [internal]")

PART_FILENAME = "part-{number:02d}.md"
PART_GLOB = "part-*.md"

USAGE = "usage: python -m penny.tests.eval.comment_split <document> <out_dir>"

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
    the seam rule outranks the budget; ``unsplittable_reason`` is what refuses it when it also
    exceeds the hard cap."""
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
                f"`python -m penny.tests.eval.assemble <run_dir>` in the container directly, never "
                f"`make assemble` piped through stripping."
            )
    return None


def unsplittable_reason(parts: list[str]) -> str | None:
    """The actionable reason a part cannot be posted — a single sample fold larger than GitHub's
    hard cap, which no legal cut can shrink — or ``None`` when every part fits."""
    for number, part in enumerate(parts, start=1):
        if len(part) > GITHUB_COMMENT_LIMIT:
            opening = part.strip().splitlines()[0] if part.strip() else ""
            return (
                f"part {number} is {len(part)} characters — over GitHub's {GITHUB_COMMENT_LIMIT} "
                f"cap and made of ONE sample fold, which cannot be cut without breaking its "
                f"markup. Shrink the sample itself (the fold opening {opening[:80]!r}) — e.g. more "
                f"shared-prompt hoisting — or run the case at a lower EVAL_SAMPLES."
            )
    return None


# ── CLI: python -m penny.tests.eval.comment_split <document> <out_dir> ────────
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
    (exit 1, actionable stderr) an unreadable or empty document, a body opening with build noise,
    and a part no legal cut can bring under GitHub's cap. Bad args → 2."""
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
    parts = split_run_comment(document)
    reason = build_noise_reason(document) or unsplittable_reason(parts)
    if reason:
        print(f"comment_split: refusing to post — {reason}", file=sys.stderr)
        return 1
    for name in write_parts(parts, out_dir):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
