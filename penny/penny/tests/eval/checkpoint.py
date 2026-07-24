"""Eval-run checkpoint markers (#1757): the ``.posted`` marker + unreviewed-run detection.

The joint-checkpoint rule — run an eval, POST its report to the PR, then STOP for joint review
before the next run — was prose-only: posting was a manual step SEPARATE from running, so skipping
it left no visible debt and nothing interrupted the next run (a real Jul 23 violation ran 4+
sequential report-runs without posting a single report). This module makes the debt STRUCTURAL
(the root *structural-state-over-model-judgment* principle, applied to the eval loop itself):

  * a completed run dir carries a ``.posted`` marker (holding the posted comment's URL) once
    ``make eval-report`` has posted its report — so re-posting is idempotent (skipped unless FORCE),
  * a run dir with a ``manifest.json`` (a completed run) but NO ``.posted`` is UNREVIEWED, and
    ``make eval`` prints a loud banner naming every such dir before it takes a GPU queue ticket.

Pure filesystem logic over the durable artifact home (the primary checkout's
``data/eval-artifacts``, #1734), driven by the Makefile's ``eval`` / ``eval-report`` recipes and
exercised by plain (non-eval) tests — no model, no git, no network. The banner WARNS, never blocks:
an intentional multi-run sweep stays possible; the debt is just undeniable. A lever-less ephemeral
run writes no ``manifest.json``, so it never appears here (by design — no artifacts to review).

CLI (invoked in-container by the Makefile, home = the mounted ``/penny/eval-artifacts``):

  ``python -m penny.tests.eval.checkpoint latest <home>``  → prints the most-recent completed run
      dir's name (exit 0), or an actionable message on stderr (exit 1) when there are none.
  ``python -m penny.tests.eval.checkpoint banner <home>``  → prints the unreviewed-run banner to
      stdout when any exist, else nothing (always exit 0 — warn, never block).
"""

from __future__ import annotations

import sys
from pathlib import Path

from penny.tests.eval.artifacts import MANIFEST_FILENAME

# The marker a posted run dir carries (its content is the posted comment's URL). Mirrored by the
# Makefile's `POSTED_MARKER` var — both must name the same file.
POSTED_MARKER = ".posted"

# ── CLI verbs ────────────────────────────────────────────────────────────────
LATEST_CMD = "latest"
BANNER_CMD = "banner"
USAGE = (
    f"usage: python -m penny.tests.eval.checkpoint {{{LATEST_CMD}|{BANNER_CMD}}} <artifact_home>"
)

# ── Banner literals (loud, multi-line; whole-render tested) ──────────────────
_BANNER_RULE = "=" * 72
_BANNER_HEAD = "⚠  {count} unreviewed eval run(s) — post and review before running again:"
_BANNER_ACTION = "→ post each: make eval-report PR=<n> [RUN=<run-dir-name>]"


def run_dirs(home: Path) -> list[Path]:
    """Every immediate subdirectory of ``home`` that holds a ``manifest.json`` — i.e. every
    COMPLETED eval run. A lever-less ephemeral run writes no manifest, so it is not a run dir
    here (by design). Name-sorted; ``run-<stamp>`` names sort chronologically."""
    if not home.is_dir():
        return []
    return sorted(
        child
        for child in home.iterdir()
        if child.is_dir() and (child / MANIFEST_FILENAME).is_file()
    )


def latest_run_dir(home: Path) -> Path | None:
    """The most-recently-modified completed run dir under ``home`` (the default ``make eval-report``
    target when no RUN is named), or ``None`` when there are no completed runs."""
    runs = run_dirs(home)
    if not runs:
        return None
    return max(runs, key=lambda run: run.stat().st_mtime)


def is_posted(run_dir: Path) -> bool:
    """Whether ``run_dir``'s report has been posted — its ``.posted`` marker exists."""
    return (run_dir / POSTED_MARKER).is_file()


def unreviewed_runs(home: Path) -> list[Path]:
    """Every completed run dir under ``home`` still missing its ``.posted`` marker (name-sorted) —
    the unreviewed-debt set the banner names."""
    return [run for run in run_dirs(home) if not is_posted(run)]


def render_banner(runs: list[Path]) -> str:
    """The loud, multi-line unreviewed-run banner naming each dir + the post command. Empty string
    when nothing is unreviewed (the recipe then prints nothing)."""
    if not runs:
        return ""
    lines = [_BANNER_RULE, _BANNER_HEAD.format(count=len(runs))]
    lines += [f"     {run.name}" for run in runs]
    lines += [_BANNER_ACTION, _BANNER_RULE]
    return "\n".join(lines)


# ── CLI: python -m penny.tests.eval.checkpoint {latest|banner} <home> ─────────
def main(argv: list[str]) -> int:
    """Dispatch the two verbs. ``latest`` prints the most-recent completed run dir's name (1 when
    none); ``banner`` prints the unreviewed banner (always 0 — warn, never block). Bad args → 2."""
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    command, home_arg = argv
    home = Path(home_arg)
    if command == LATEST_CMD:
        latest = latest_run_dir(home)
        if latest is None:
            print(f"no completed run dirs under {home}", file=sys.stderr)
            return 1
        print(latest.name)
        return 0
    if command == BANNER_CMD:
        banner = render_banner(unreviewed_runs(home))
        if banner:
            print(banner)
        return 0
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
