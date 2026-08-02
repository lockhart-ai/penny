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
its own contract in ``test_skill_shape.py`` — these cases are the labeller's,
and they run when the LABELLER changes.

Cases 1 and 2 are #1770's two directions.  Cases 3 and 4 (#1821) port the two shapes
the elicit → learn beat loses to this draw — the assistant's own wording of the user's
ask, and a write key assembled out of the user's materials — down to where the judgment
is actually made, as a minimal pair differing only in the write key.

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


# ── Cases 3 & 4: the beat's two failure shapes, ported down to the labeller ────
#
# The elicit → learn beat fails on the labeller, not on the round: three of its four
# failed samples lose the extract instruction and one loses the write key, and both
# losses are decided in this draw.  Scoring them HERE isolates the judgment — the beat
# runs a live chat turn whose demonstration varies sample to sample, so a verdict
# measured there is measured through whatever wording the chat model happened to use.
#
# The two cases are a MINIMAL PAIR over the same round: identical conversation, browse
# and collection, differing only in the write key — which is the one value whose right
# answer is contested.  Anything that moves one and not the other moved the key
# judgment and nothing else.
_TEACH_TURN = f"yeah go to {_LISTING}, find the price, and remember it"

# What the assistant DECIDED, in the demonstrated round:
#   - the extract instruction, which says the user's ask in the assistant's own words
#   - the collection, which their ask never named
#   - the write key, built for a filing scheme nobody asked for
_ASSISTANT_WORDED_EXTRACT = "the price shown on the product page"
_PICKED_COLLECTION = "prices"
_BUILT_KEY = "Aurora Deck 2 price"
_SLUGGED_KEY = "Aurora Deck 2"

_PICKED_WRITE_OK = (
    f"You saved an entry to {_PICKED_COLLECTION}: (collection_write result)\nWrote 1 entry."
)

_WORDED_BROWSE = (
    "browse",
    {"queries": [_LISTING], "extract": _ASSISTANT_WORDED_EXTRACT},
    f"You opened the Aurora Deck 2 listing (browse result)\n{_PRICE}",
    True,
)


def _write_keyed(key: str) -> tuple[str, dict, str, bool]:
    """The round's write, under one candidate key — the pair's only variable."""
    return (
        "collection_write",
        {"memory": _PICKED_COLLECTION, "entries": [{"key": key, "content": _PRICE}]},
        _PICKED_WRITE_OK,
        True,
    )


@pytest.mark.asyncio
async def test_the_assistants_wording_of_their_ask_stays_a_parameter(
    labeller_eval: LabellerEval,
):
    """The value the user asked for, in the assistant's words, is still the user's.

    The failing shape, ported: asked where the extract instruction "came from", the
    draw answered on AUTHORSHIP — the user said "find the price", the assistant typed
    the instruction, therefore the assistant produced it — and a placeholder is
    withheld from the shape draw, so nothing downstream can bake what to find and the
    routine asks for it again every time it runs.  Whoever typed the string is not the
    question; whose ask decided what it says is."""
    await labeller_eval(
        case_id="labelling-their-ask-in-the-assistants-words",
        utterance=_TEACH_TURN,
        calls=[_WORDED_BROWSE, _write_keyed(_BUILT_KEY)],
        target=_PICKED_COLLECTION,
        user_values=[_LISTING, _ASSISTANT_WORDED_EXTRACT],
        assistant_values=[_PICKED_COLLECTION, _BUILT_KEY],
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_a_key_assembled_from_their_own_materials_becomes_a_placeholder(
    labeller_eval: LabellerEval,
):
    """The sneaky direction, and the guard the case above needs.

    The same round files the price under the listing's own name — words the user did
    supply, for a filing decision they never made.  Ruling that theirs leaves a
    required parameter nobody can supply, which is the #1770 harm arriving by the back
    door: reusing their materials does not make a value theirs when their ask never
    named the thing it is for.  It is the minimal pair of the case above, so a lever
    that wins one by conceding the other shows up here as a loss."""
    await labeller_eval(
        case_id="labelling-key-assembled-from-their-materials",
        utterance=_TEACH_TURN,
        calls=[_WORDED_BROWSE, _write_keyed(_SLUGGED_KEY)],
        target=_PICKED_COLLECTION,
        user_values=[_LISTING, _ASSISTANT_WORDED_EXTRACT],
        assistant_values=[_PICKED_COLLECTION, _SLUGGED_KEY],
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )
