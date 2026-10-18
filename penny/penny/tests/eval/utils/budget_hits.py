"""Which call kinds got cut by their own token budget, read off a sample's own promptlog.

Every model call now carries its call kind's token budget (#2474), and a provider that cut a
draw off mid-thought says so on the raw response: ``choices[0].finish_reason == "length"``
(``PennyConstants.FINISH_REASON_LENGTH``). Nothing about that is a pass/fail check — a cut draw
is retried by the runtime itself (thinking-off for a micro-context, the last draw for a chat or
collector turn, #2475/#2476) — but a call kind that gets cut often is a budget worth revisiting,
and a run that never says so leaves that only discoverable by hand.

This reads the same rows ``live_prompt_perf`` does (one process's own promptlog, minus any
seeded prior turn), tallied by ``agent_name`` — the call kind — rather than summed across all
of them, because a budget is set PER KIND and a pooled total would hide which one is tight.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, Field

from penny.constants import PennyConstants
from penny.database.models import PromptLog

# What a row's ``agent_name`` reads as when it carries none — an untagged call (a non-chat,
# non-collector path, or an old row from before #2474) still drew against SOME budget, so it
# is counted rather than dropped, under a name that says it was never tagged.
UNTAGGED_CALL_KIND = "untagged"


class CallKindTally(BaseModel):
    """One call kind's draws this reading covers, and how many of them were cut."""

    draws: int = 0
    cuts: int = 0

    def merge(self, other: CallKindTally) -> CallKindTally:
        return CallKindTally(draws=self.draws + other.draws, cuts=self.cuts + other.cuts)


class BudgetHits(BaseModel):
    """Every call kind's own tally, pooled across however many samples fed it."""

    by_kind: dict[str, CallKindTally] = Field(default_factory=dict)

    @property
    def total_cuts(self) -> int:
        """How many draws, of any kind, were cut by their own budget."""
        return sum(tally.cuts for tally in self.by_kind.values())

    def merge(self, other: BudgetHits) -> BudgetHits:
        """Two readings, added — a plain dict per kind, so a third reading adds the same way."""
        merged = {kind: tally.model_copy() for kind, tally in self.by_kind.items()}
        for kind, tally in other.by_kind.items():
            merged[kind] = merged[kind].merge(tally) if kind in merged else tally.model_copy()
        return BudgetHits(by_kind=merged)


def tally_budget_hits(rows: Sequence[PromptLog]) -> BudgetHits:
    """One sample's own promptlog rows, tallied by call kind.

    ``rows`` is a sample's own window — the same one ``live_prompt_perf`` reads (its
    ``recent_prompts`` minus any seeded prior turn) — so a cut counted here is this sample's
    own draw, never a turn seeded ahead of it.
    """
    by_kind: dict[str, CallKindTally] = {}
    for row in rows:
        kind = row.agent_name or UNTAGGED_CALL_KIND
        tally = by_kind.setdefault(kind, CallKindTally())
        tally.draws += 1
        if _cut_by_budget(row.response):
            tally.cuts += 1
    return BudgetHits(by_kind=by_kind)


def _cut_by_budget(raw_response: str) -> bool:
    """Whether one row's raw response was cut off by its own token budget.

    Defensive on purpose: a cancelled row's response is the literal ``"{}"`` (see
    ``MessageStore.log_prompt``), and a response with no ``choices`` at all reads as NOT cut
    rather than raising — the same guard every other promptlog reader opens with."""
    response = json.loads(raw_response) if raw_response else {}
    choices = response.get("choices") or []
    if not choices:
        return False
    return choices[0].get("finish_reason") == PennyConstants.FINISH_REASON_LENGTH


def render_budget_hits(hits: BudgetHits) -> str:
    """``budget cuts: browse-extract 11/75, chat 0/412`` — worst (most cut) first.

    Every drawn kind renders, cut or not, once ANY kind has been cut (the caller gates on
    that): a kind reading ``chat 0/412`` beside one that is cut is what tells a reader the
    cut is NOT just a universally tight budget."""
    ordered = sorted(hits.by_kind.items(), key=lambda item: (-item[1].cuts, item[0]))
    body = ", ".join(f"{kind} {tally.cuts}/{tally.draws}" for kind, tally in ordered)
    return f"budget cuts: {body}"
