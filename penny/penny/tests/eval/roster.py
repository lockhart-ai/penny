"""Which models the suite measures, and which upstream each prefers.

The model a run measured used to be an ad-hoc make variable and the provider could not be
expressed at all — so "what did this run actually run against?" was answered by shell
history, and a run silently served by a broken member of a routing pool looked exactly
like a run against a broken model.  Both are configuration now, in ``.env``, read through
one Pydantic model:

    EVAL_MODELS=[{"model":"vendor/model-a","provider":"SomeCloud"},
                 {"model":"vendor/model-b","provider":"OtherCloud"}]

A JSON list because that is how ``PLUGINS`` is already configured, and because an env var
name cannot contain ``/`` — a per-model-suffixed variable scheme is unavailable, not just
inelegant.

**Two models, required.**  A suite ported and tuned against ONE model bakes that model's
quirks into its fixtures, and then measures how much like that model the next model is.
So the roster must name at least two, and a remote run REFUSES to start when it does not —
the fail-fast shape ``LLM_EMBEDDING_MODEL`` already has, for the same reason: a run that
quietly measures less than it claims to is worse than a run that does not start.  The
minimum is a COUNT rather than two named models: naming them here would rot the day the
pair changes, and what actually matters is that no single model's quirks can pass for the
suite's.

**The requirement binds the REMOTE profile only.**  The field that makes an entry
load-bearing is ``provider``, and a local Ollama has none — it serves one model, on one
GPU, from one process.  Requiring two models there would mean pulling a second model onto
a box whose whole constraint is that it serves one at a time, and would buy no signal: the
comparison this exists for is between models a GATEWAY serves.  ``make eval`` local keeps
its own single configured model exactly as it was.
"""

from __future__ import annotations

import os
import sys

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

# The one variable, read from the primary checkout's `.env` (or the shell) by the Makefile
# and forwarded into the run.
EVAL_MODELS_ENV = "EVAL_MODELS"

# Why two: see the module docstring.  A count, never a pair of names.
MINIMUM_MODELS = 2

USAGE = "usage: python -m penny.tests.eval.roster [<model>]"

# The lines the Makefile reads the resolution off.  Human-readable AND parseable, so the
# operator sees the same two facts the run records.
MODEL_LINE_PREFIX = "eval: model ="
PREFERRED_PROVIDER_LINE_PREFIX = "eval: preferred provider ="

_SHAPE_EXAMPLE = (
    '    EVAL_MODELS=[{"model":"vendor/model-a","provider":"SomeCloud"},'
    '{"model":"vendor/model-b","provider":"OtherCloud"}]'
)


class RosterError(RuntimeError):
    """The roster is missing, malformed, or names too few models to measure anything."""


class EvalModel(BaseModel):
    """One model the suite measures, and the upstream it prefers to be served by.

    ``provider`` is optional because a direct endpoint has none to name; where it IS set,
    it is a PREFERENCE the run sends with every call and never a wall (see
    ``ProviderPreference``).
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
            f"least {MINIMUM_MODELS} models, so that no single model's quirks can pass for "
            "the suite's — set it in the primary checkout's .env, e.g.\n"
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
        named = ", ".join(entry.render() for entry in roster) or "nothing"
        raise RosterError(
            f"{EVAL_MODELS_ENV} names {named} — a remote run measures every case on at "
            f"least {MINIMUM_MODELS} models, because a suite tuned against one model bakes "
            "that model's quirks into its fixtures and then measures how much like it the "
            f"next model is. Add another entry, e.g.\n{_SHAPE_EXAMPLE}"
        )
    return roster


def resolve(roster: list[EvalModel], requested: str | None) -> EvalModel:
    """Which entry THIS invocation runs — the first by default, or the one named.

    A requested model outside the roster is refused rather than run: an unconfigured model
    has no provider to prefer and no provider to record, which is the ad-hoc pass this
    whole variable replaces.  Adding it to ``.env`` is the fix, and the refusal says so.
    """
    if requested is None:
        return roster[0]
    for entry in roster:
        if entry.model == requested:
            return entry
    listed = "\n".join(f"  - {entry.render()}" for entry in roster)
    raise RosterError(
        f"'{requested}' is not in {EVAL_MODELS_ENV}, so this run could neither prefer an "
        f"upstream for it nor record which one answered. Configured:\n{listed}\n"
        f"Add it to {EVAL_MODELS_ENV} in the primary checkout's .env to run it."
    )


def render_resolution(entry: EvalModel) -> list[str]:
    """The lines the Makefile reads: always the model, the provider when one is named."""
    lines = [f"{MODEL_LINE_PREFIX} {entry.model}"]
    if entry.provider:
        lines.append(f"{PREFERRED_PROVIDER_LINE_PREFIX} {entry.provider}")
    return lines


def main(argv: list[str]) -> int:
    """Print this invocation's model + preferred provider; 1 when the roster refuses.

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
