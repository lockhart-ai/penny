"""Which models the suite measures, and which upstream each is pinned to.

The model a run measured is configuration, in ``.env``, read through one Pydantic model —
so "what did this run actually run against?" is a read of the roster rather than shell
history, and a run served by a broken member of a routing pool is diagnosable from it:

    EVAL_MODELS=[{"model":"vendor/model-a","provider":"SomeCloud"}]

A JSON list because that is how ``PLUGINS`` is already configured, and because an env var
name cannot contain ``/`` — a per-model-suffixed variable scheme is unavailable, not just
inelegant.

**At least one entry, required.**  A run REFUSES to start on an empty roster — the
fail-fast shape ``LLM_EMBEDDING_MODEL`` already has: a run that quietly measures nothing is
worse than a run that does not start.  The suite's own PRs and checkpoints are measured on
the roster's first entry; a second (or further) entry stays configurable and runnable by
name for a look, and such a run gates nothing.
"""

from __future__ import annotations

import os
import sys

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

# The one variable, read from the primary checkout's `.env` (or the shell) by the Makefile
# and forwarded into the run.
EVAL_MODELS_ENV = "EVAL_MODELS"

# The two hand-down variables the Makefile recipes used to set for the helpers to trust.
# Neither is read as a value any more: ``LLM_MODEL`` is only ever a NAME to look up in the
# roster, and an exported ``LLM_PROVIDER`` is only ever CHECKED against the resolved entry's
# own pin — so nothing can run on a model or upstream `EVAL_MODELS` does not define.
LLM_MODEL_ENV = "LLM_MODEL"
LLM_PROVIDER_ENV = "LLM_PROVIDER"

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


def resolve_from_env(requested: str | None = None) -> EvalModel:
    """The ONE resolution every run path uses — the eval suite, ``make replay``, and the
    A/B helper (`replay.py`'s ``replay_variants``) all land here.

    The model and its pin come from `EVAL_MODELS` through :func:`resolve` and nothing
    else.  The two hand-down variables are consulted only as selectors against that
    roster, never as values to run on:

    - ``LLM_MODEL`` names WHICH roster entry this invocation runs (unset → the first);
      a name outside the roster is :func:`resolve`'s refusal, listing every entry.
    - an exported ``LLM_PROVIDER`` that does not match the resolved entry's own pin
      REFUSES the run up front — never silently ignored and never used — because a run
      sent to an upstream the roster does not define is exactly the invalid comparison
      this module exists to make impossible.  An empty value reads as unset, the same
      convention ``env_seconds`` uses for a recipe-forwarded-but-empty variable.
    """
    roster = parse_roster(os.environ.get(EVAL_MODELS_ENV))
    entry = resolve(
        roster, requested if requested is not None else (os.environ.get(LLM_MODEL_ENV) or None)
    )
    exported = os.environ.get(LLM_PROVIDER_ENV)
    if exported and exported != (entry.provider or ""):
        listed = "\n".join(f"  - {other.render()}" for other in roster)
        raise RosterError(
            f"{LLM_PROVIDER_ENV}='{exported}' does not match the resolved entry "
            f"({entry.render()}), so this run would go to an upstream the roster does "
            f"not define. Configured:\n{listed}\n"
            f"Unset {LLM_PROVIDER_ENV} and let the resolved entry's own pin be used."
        )
    return entry


def main(argv: list[str]) -> int:
    """Print this invocation's model + pinned provider; 1 when the roster refuses.

    Run by the ``eval-remote`` recipe BEFORE anything is spent, so an under-configured suite stops
    at the same point an unserveable endpoint does rather than minutes into a run.
    """
    if len(argv) > 1:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        entry = resolve_from_env(argv[0] if argv else None)
    except RosterError as error:
        print(f"eval: {error}", file=sys.stderr)
        return 1
    for line in render_resolution(entry):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
