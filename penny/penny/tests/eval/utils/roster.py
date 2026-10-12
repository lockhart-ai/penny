"""Which models the suite measures, and which upstream each is pinned to.

The model a run measured is configuration, in ``.env``, read through one Pydantic model —
so "what did this run actually run against?" is a read of the roster rather than shell
history, and a run served by a broken member of a routing pool is diagnosable from it:

    EVAL_MODELS=[{"model":"vendor/model-a","provider":"SomeCloud"}]

A JSON list because that is how ``PLUGINS`` is already configured, and because an env var
name cannot contain ``/`` — a per-model-suffixed variable scheme is unavailable, not just
inelegant.

**At least one entry, required.**  A remote run REFUSES to start on an empty roster — the
fail-fast shape ``LLM_EMBEDDING_MODEL`` already has: a run that quietly measures nothing is
worse than a run that does not start.  The suite's own PRs and checkpoints are measured on
the roster's first entry; a second (or further) entry stays configurable and runnable by
name for a look, and such a run gates nothing.

**The requirement binds the REMOTE profile only.**  The field that makes an entry
load-bearing is ``provider``, and a local Ollama has none — it serves one model, on one
GPU, from one process.  ``make eval`` local keeps its own single configured model exactly
as it was.
"""

from __future__ import annotations

import os
import sys

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

# The one variable, read from the primary checkout's `.env` (or the shell) by the Makefile
# and forwarded into the run.
EVAL_MODELS_ENV = "EVAL_MODELS"

# Why one: see the module docstring.  A count, never a named model.
MINIMUM_MODELS = 1

USAGE = "usage: python -m penny.tests.eval.utils.roster [<model>]"

# The lines the Makefile reads the resolution off.  Human-readable AND parseable, so the
# operator sees the same two facts the run records.
MODEL_LINE_PREFIX = "eval: model ="
PINNED_PROVIDER_LINE_PREFIX = "eval: pinned provider ="

_SHAPE_EXAMPLE = '    EVAL_MODELS=[{"model":"vendor/model-a","provider":"SomeCloud"}]'


class RosterError(RuntimeError):
    """The roster is missing, malformed, or names too few models to measure anything."""


class EvalModel(BaseModel):
    """One model the suite measures, and the upstream it is pinned to.

    ``provider`` is optional because a direct endpoint has none to name; where it IS set,
    every call the run makes is PINNED to it with fallbacks off (see
    ``ProviderPin``), so the run is served by that provider or by nobody.  Name it
    as the gateway reports it back (OpenRouter answers ``Groq``, not ``groq``), so the
    manifest's pinned-versus-answered comparison reads a match as a match.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    provider: str | None = None

    def render(self) -> str:
        """This entry as the operator wrote it — used when the roster has to be quoted."""
        served = f" via {self.provider}" if self.provider else " (no provider named)"
        return f"{self.model}{served}"


_ROSTER_ADAPTER = TypeAdapter(list[EvalModel])


def parse_roster(raw: str | None) -> list[EvalModel]:
    """The configured roster, or a refusal that says exactly what to put in ``.env``.

    Every failure is the same class carrying its own actionable message, because the
    operator's next action is identical in all three cases: open ``.env`` and write the
    variable properly.
    """
    text = (raw or "").strip()
    if not text:
        raise RosterError(
            f"{EVAL_MODELS_ENV} is not configured. A remote run measures every case on at "
            f"least {MINIMUM_MODELS} model — set it in the primary checkout's .env, e.g.\n"
            f"{_SHAPE_EXAMPLE}"
        )
    try:
        roster = _ROSTER_ADAPTER.validate_json(text)
    except ValidationError as error:
        raise RosterError(
            f"{EVAL_MODELS_ENV} is not a list of "
            '{"model": ..., "provider": ...} entries: '
            f"{error}\nExpected, e.g.\n{_SHAPE_EXAMPLE}"
        ) from error
    if len(roster) < MINIMUM_MODELS:
        raise RosterError(
            f"{EVAL_MODELS_ENV} names nothing — a remote run measures every case on at "
            f"least {MINIMUM_MODELS} model. Add an entry, e.g.\n{_SHAPE_EXAMPLE}"
        )
    return roster


def resolve(roster: list[EvalModel], requested: str | None) -> EvalModel:
    """Which entry THIS invocation runs — the first by default, or the one named.

    A requested model outside the roster is refused rather than run: an unconfigured model
    has no provider to pin and no provider to record, which is the ad-hoc pass this whole
    variable replaces.  Adding it to ``.env`` is the fix, and the refusal says so.
    """
    if requested is None:
        return roster[0]
    for entry in roster:
        if entry.model == requested:
            return entry
    listed = "\n".join(f"  - {entry.render()}" for entry in roster)
    raise RosterError(
        f"'{requested}' is not in {EVAL_MODELS_ENV}, so this run could neither pin an "
        f"upstream for it nor record which one answered. Configured:\n{listed}\n"
        f"Add it to {EVAL_MODELS_ENV} in the primary checkout's .env to run it."
    )


def render_resolution(entry: EvalModel) -> list[str]:
    """The lines the Makefile reads: always the model, the provider when one is named."""
    lines = [f"{MODEL_LINE_PREFIX} {entry.model}"]
    if entry.provider:
        lines.append(f"{PINNED_PROVIDER_LINE_PREFIX} {entry.provider}")
    return lines


def main(argv: list[str]) -> int:
    """Print this invocation's model + pinned provider; 1 when the roster refuses.

    Run by the ``eval`` recipe BEFORE anything is spent, so an under-configured suite stops
    at the same point an unserveable endpoint does rather than minutes into a run.
    """
    if len(argv) > 1:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        entry = resolve(parse_roster(os.environ.get(EVAL_MODELS_ENV)), argv[0] if argv else None)
    except RosterError as error:
        print(f"eval: {error}", file=sys.stderr)
        return 1
    for line in render_resolution(entry):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
