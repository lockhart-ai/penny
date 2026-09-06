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

The home is **shared by every worktree** (#1734 mounts the primary checkout's
``data/eval-artifacts`` into all of them), so a run in it belongs to whichever tree measured it.
A bare ``make eval-report`` therefore resolves the run to post by the **invoking tree's HEAD** —
the commit the run's manifest records — and refuses when that is not exactly one run (#2098).
"The most recent completed run" was the old default, and with a fleet running it was routinely a
sibling agent's: one agent posted another's six reports to its own PR and stamped the sibling's
run dir ``.posted`` under a comment URL on the wrong PR.

CLI (invoked in-container by the Makefile, home = the mounted ``/penny/eval-artifacts``):

  ``python -m penny.tests.eval.utils.checkpoint resolve <home> <commit>``
      → prints the ONE unposted completed run dir measured at ``<commit>`` (exit 0), or an
      actionable multi-line message on stderr (exit 1) when there are none or several. Never
      falls back to "most recent".
  ``python -m penny.tests.eval.utils.checkpoint banner <home>``
      → prints the unreviewed-run banner to stdout when any exist, else nothing
      (always exit 0 — warn, never block).
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

from penny.tests.eval.utils.artifacts import MANIFEST_FILENAME

# The marker a posted run dir carries (its content is the posted comment's URL). Mirrored by the
# Makefile's `POSTED_MARKER` var — both must name the same file.
POSTED_MARKER = ".posted"

# ── CLI verbs ────────────────────────────────────────────────────────────────
RESOLVE_CMD = "resolve"
BANNER_CMD = "banner"
USAGE = (
    f"usage: python -m penny.tests.eval.utils.checkpoint {RESOLVE_CMD} <artifact_home> <commit>\n"
    f"       python -m penny.tests.eval.utils.checkpoint {BANNER_CMD} <artifact_home>"
)

# ── Banner literals (loud, multi-line; whole-render tested) ──────────────────
_BANNER_RULE = "=" * 72
_BANNER_HEAD = "⚠  {count} unreviewed eval run(s) — post and review before running again:"
_BANNER_ACTION = "→ post each: make eval-report PR=<n> [RUN=<run-dir-name>]"

# ── Resolution-refusal literals (loud, multi-line; whole-render tested) ──────
_RUN_LINE = "     {name}"
_SHARED_HOME_NOTE = (
    "     the artifact home is SHARED by every worktree (#1734) — a run in it belongs to "
    "whichever tree measured it"
)
_NO_MATCH_HEAD = "eval-report: no unposted completed run under {home} was measured at {commit}"
_ALREADY_POSTED = (
    "     already posted at this commit: {names} — re-post one with RUN=<run-dir> FORCE=1"
)
_NO_MATCH_ACTION = "     run `make eval` on this tree, or post a named run with RUN=<run-dir>"
_AMBIGUOUS_HEAD = "eval-report: {count} unposted runs under {home} were measured at {commit}:"
_AMBIGUOUS_ACTION = "     name the one to post: make eval-report PR=<n> RUN=<run-dir>"


def run_dirs(home: Path) -> list[Path]:
    """Every immediate subdirectory of ``home`` that holds a ``manifest.json`` — i.e. every
    COMPLETED eval run. A lever-less ephemeral run writes no manifest, so it is not a run dir
    here (by design). Name-sorted; a run dir is ``run-<stamp>-<pid>`` and the stamp leads,
    so the names still sort chronologically (the pid only orders runs within one second)."""
    if not home.is_dir():
        return []
    return sorted(
        child
        for child in home.iterdir()
        if child.is_dir() and (child / MANIFEST_FILENAME).is_file()
    )


def is_posted(run_dir: Path) -> bool:
    """Whether ``run_dir``'s report has been posted — its ``.posted`` marker exists."""
    return (run_dir / POSTED_MARKER).is_file()


def unreviewed_runs(home: Path) -> list[Path]:
    """Every completed run dir under ``home`` still missing its ``.posted`` marker (name-sorted) —
    the unreviewed-debt set the banner names."""
    return [run for run in run_dirs(home) if not is_posted(run)]


class RunIdentity(BaseModel):
    """The single ``manifest.json`` field bare-RUN resolution reads: WHICH COMMIT the run measured
    (``EVAL_COMMIT``, the invoking tree's ``HEAD`` at run time). Every other manifest field is
    ignored, so a manifest written by any other version of the suite still decodes here."""

    commit: str


def measured_commit(run_dir: Path) -> str | None:
    """The commit ``run_dir``'s manifest records, or ``None`` when the manifest cannot be read or
    records no commit. Absence is ordinary — a manifest half-written by a concurrent run reads this
    way — and ``None`` never equals a commit, so an unattributable run drops out of the candidates
    instead of matching one and being posted to the wrong PR."""
    try:
        manifest = RunIdentity.model_validate_json((run_dir / MANIFEST_FILENAME).read_bytes())
    except OSError, ValidationError:
        return None
    return manifest.commit


def runs_at_commit(home: Path, commit: str) -> list[Path]:
    """Every UNPOSTED completed run under ``home`` measured at ``commit`` (name-sorted) — the
    candidate set a bare ``make eval-report`` resolves against. Posted runs are excluded so a
    re-post is always deliberate (``RUN=<run-dir> FORCE=1``)."""
    return [run for run in unreviewed_runs(home) if measured_commit(run) == commit]


def posted_runs_at_commit(home: Path, commit: str) -> list[Path]:
    """Every already-POSTED completed run under ``home`` measured at ``commit`` (name-sorted) —
    named in the no-candidate refusal, because "this tree already posted its run" and "this tree
    never ran one" are different situations with different next moves."""
    return [run for run in run_dirs(home) if is_posted(run) and measured_commit(run) == commit]


def render_banner(runs: list[Path]) -> str:
    """The loud, multi-line unreviewed-run banner naming each dir + the post command. Empty string
    when nothing is unreviewed (the recipe then prints nothing)."""
    if not runs:
        return ""
    lines = [_BANNER_RULE, _BANNER_HEAD.format(count=len(runs))]
    lines += [_RUN_LINE.format(name=run.name) for run in runs]
    lines += [_BANNER_ACTION, _BANNER_RULE]
    return "\n".join(lines)


def render_no_match(home: Path, commit: str, posted: list[Path]) -> str:
    """The refusal when NO unposted run under ``home`` was measured at ``commit``: names the home
    and the commit, says the home is shared, and names any run already posted at this commit (the
    "I already posted mine" case, whose next move is ``FORCE=1``, not another eval)."""
    lines = [_NO_MATCH_HEAD.format(home=home, commit=commit), _SHARED_HOME_NOTE]
    if posted:
        lines.append(_ALREADY_POSTED.format(names=", ".join(run.name for run in posted)))
    lines.append(_NO_MATCH_ACTION)
    return "\n".join(lines)


def render_ambiguous(home: Path, commit: str, runs: list[Path]) -> str:
    """The refusal when SEVERAL unposted runs under ``home`` were measured at ``commit``: lists
    them and asks for ``RUN=``. Never a fall back to the most recent — that is the bug (#2098)."""
    lines = [_AMBIGUOUS_HEAD.format(count=len(runs), home=home, commit=commit)]
    lines += [_RUN_LINE.format(name=run.name) for run in runs]
    lines += [_SHARED_HOME_NOTE, _AMBIGUOUS_ACTION]
    return "\n".join(lines)


def resolve(home: Path, commit: str) -> int:
    """Print the ONE unposted run under ``home`` measured at ``commit`` (exit 0), or the matching
    loud refusal on stderr (exit 1) for zero or several. This is the whole of the bare-``RUN``
    default: three outcomes, no fourth."""
    candidates = runs_at_commit(home, commit)
    if len(candidates) == 1:
        print(candidates[0].name)
        return 0
    if candidates:
        print(render_ambiguous(home, commit, candidates), file=sys.stderr)
    else:
        print(render_no_match(home, commit, posted_runs_at_commit(home, commit)), file=sys.stderr)
    return 1


# ── CLI: checkpoint resolve <home> <commit> | checkpoint banner <home> ───────
def main(argv: list[str]) -> int:
    """Dispatch the two verbs. ``resolve`` prints this tree's run dir name (1 on zero or several);
    ``banner`` prints the unreviewed banner (always 0 — warn, never block). Bad args → 2."""
    if len(argv) == 3 and argv[0] == RESOLVE_CMD:
        return resolve(Path(argv[1]), argv[2])
    if len(argv) == 2 and argv[0] == BANNER_CMD:
        banner = render_banner(unreviewed_runs(Path(argv[1])))
        if banner:
            print(banner)
        return 0
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
