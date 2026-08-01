"""Live-model contract for the run-end skill LABELLER (#1770).

The distiller classifies every unexplained string leaf of a demonstrated round as a
required parameter.  That is a *default*, not a determination: it holds only when the
user supplied the value, and it is wrong for a value the assistant derived from a tool
result or invented outright — producing a skill with a required parameter no user could
ever supply.  Neither of those origins shares a literal span with what produced it, so
no string test can reach them (and #1659 already ruled prose matching out) — the
question "did the USER provide this?" is a judgment, which is what these cases measure.

Each case hands the labeller a FIXTURE demonstration (a ledger, not a driven round) and
scores two things off persisted state: the values the user really did supply stayed
bindable parameters, and the ones the assistant produced became placeholders whose
demonstrated phrase is NOT frozen into the ``extraction_prompt`` a collector would run.
Freezing is the specific harm — a collector re-running the skill would write that stale
phrase into the collection on every cycle, forever.

Deliberately NOT scored: what a demonstrated round chooses to write.  If a round writes
two entries, two entries are the skill — that is the model's latitude, adjustable later
by the user and Penny discussing it (the code owner's ruling on #1770).  These cases fix
the round and vary only the judgment.

The SHAPE draw that decides the routine's name and which values are constant
(#1803) is a different micro-context answering a different question, and it has
its own contract in ``test_skill_shape.py`` — these two cases are the labeller's,
and they run when the LABELLER changes.

All content is synthetic (aurora / faux-market).
"""

from __future__ import annotations

import pytest

from penny.tests.eval.conftest import LabellerEval

pytestmark = pytest.mark.eval

_FAMILY = "skill-labelling"

_TARGET = "aurora-prices"
_PRICE = "$499"
_LISTING = "https://faux-market.example/aurora-deck-2"

# What the user said, one turn before the demonstration and in the demonstrating
# message itself — the only place a real parameter can come from.
_ASK = "can you keep an eye on the aurora deck 2 price for me?"
_UTTERANCE = f"read {_LISTING}, find the current price, and remember it"

# The values the USER supplied: the page they named and the thing they asked to be
# found (reworded by the assistant into browse's extract instruction — a paraphrase is
# still the user's, which is the boundary case the prompt names explicitly).
_USER_VALUES = [_LISTING, "the current price"]

_BROWSE = (
    "browse",
    {"queries": [_LISTING], "extract": "the current price"},
    f"You opened the Aurora Deck 2 listing (browse result)\n{_PRICE}",
    True,
)
_WRITE_OK = "You saved entries to aurora-prices: (collection_write result)\nWrote 2 entries."


# ── Case 1: the motivating shape — a second entry the assistant composed itself ─

# The round recorded the price AND a note it wrote ABOUT the page it had just read.
# Neither leaf of that second entry came from the user: the key is a label the
# assistant chose, the content a sentence it composed.
_INVENTED_KEY = "aurora deck 2 page source"
_INVENTED_CONTENT = "Page source for the Aurora Deck 2 listing"
_WRITE_WITH_NOTE = (
    "collection_write",
    {
        "memory": _TARGET,
        "entries": [
            {"key": "aurora deck 2 price", "content": _PRICE},
            {"key": _INVENTED_KEY, "content": _INVENTED_CONTENT},
        ],
    },
    _WRITE_OK,
    True,
)


@pytest.mark.asyncio
async def test_assistant_composed_entry_becomes_a_placeholder(labeller_eval: LabellerEval):
    """The motivating case: a round that also wrote a note it composed itself must not
    turn that note into a required parameter.  The user's page and what-to-find stay
    parameters; the assistant's label and note become placeholders, and neither phrase
    is frozen into the collector's prompt."""
    await labeller_eval(
        case_id="labelling-assistant-composed-entry",
        utterance=_UTTERANCE,
        conversation=[_ASK],
        calls=[_BROWSE, _WRITE_WITH_NOTE],
        target=_TARGET,
        user_values=_USER_VALUES,
        assistant_values=[_INVENTED_KEY, _INVENTED_CONTENT],
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )


# ── Case 2: the over-correction guard — a plain round has no placeholders ──────

_PLAIN_WRITE = (
    "collection_write",
    {
        "memory": _TARGET,
        "entries": [{"key": "aurora deck 2 price", "content": _PRICE}],
    },
    "You saved an entry to aurora-prices: (collection_write result)\nWrote 1 entry.",
    True,
)


@pytest.mark.asyncio
async def test_user_supplied_values_stay_parameters(labeller_eval: LabellerEval):
    """The other direction, and the one that matters most: a clean round whose every
    unexplained leaf really did come from the user must keep ALL of them as bindable
    parameters.  A labeller that hedged toward 'placeholder' would leave a skill nobody
    can instantiate — the same defect from the opposite side."""
    await labeller_eval(
        case_id="labelling-user-values-stay-parameters",
        utterance=_UTTERANCE,
        conversation=[_ASK],
        calls=[_BROWSE, _PLAIN_WRITE],
        target=_TARGET,
        user_values=_USER_VALUES,
        assistant_values=[],
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )
