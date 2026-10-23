"""One refer-back case: a day later, beyond the conversation window, the user
refers back to something THEY said — "that list from the other night" — without
naming it (#2752).

The history window is the last ``CHAT_HISTORY_TURNS`` (5) exchanges (#2746), so
the salient exchange sits out of the window here — pushed past it the same way
``lookup-user-said-out-of-window`` pushes its mention (filler pairs, more than
``MESSAGE_CONTEXT_LIMIT``), and backdated a day so it also sits inside the
conversation map's own horizon (``CHAT_MAP_DAYS``, today and yesterday).  The
SAME case measures which route she reaches before and after the map, unmodified:
nothing here names a route or a tool.

Its sibling ask — referring back to something PENNY named ("that show you
recommended") — was measured and retired: on main's code she answers it 15/15
by ``find`` over her own stored messages, the same behaviour
``speak-logread-penny-messages-recall`` already covers (§4.5 duplicate), and
with the map she reached for the same route rather than the map's ``named:``
line, so the case tested nothing the map does.  Its before/after cost rows are
recorded in the PR body as the map's speed evidence for that path.

Nothing here asserts a route or a tool name: ``TOOL_SEQUENCE`` is measured, never
claimed.  REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose
are the code owner's to accept once the numbers have been read.  Every list,
entry and person is invented, because the repo is public.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlmodel import col, update

from penny.constants import ChatPromptType, PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.models import MemoryEntry, MessageLog, PromptLog
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.end_to_end.world import (
    _SEEDED_MODEL,
    _seeded_response,
    _wire_tool_call,
)
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    fold_typography,
)
from penny.tests.eval.utils.fixtures import SynthCollection
from penny.tests.eval.utils.lookup_world import (
    _FILLER_PAIRS,
    _FILLER_PENNY,
    _FILLER_USER,
    assert_seeded_turns_embedded,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "conversation-map"

_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY = PennyConstants.MessageAuthor.PENNY
_CHAT = PennyConstants.CHAT_AGENT_NAME

# The salient exchange sits a day back — inside the map's horizon (today and
# yesterday) and outside the 5-exchange window (the filler pairs below push it
# past both the window and the ``MESSAGE_CONTEXT_LIMIT`` pool).
_DAYS_BACK = 1


def _backdate_messages(db: Database, message_ids: list[int], days: int) -> None:
    """Push the exchange's message rows back by ``days`` — true of the seeded
    world, not of anything the model did.  A raw update because no public store
    method takes an explicit timestamp for a logged message (the same shape
    ``seed_job_cycle``'s ``_backdate_run`` uses)."""
    stamp = db.clock.now() - timedelta(days=days)
    with db.get_session() as session:
        session.exec(
            update(MessageLog).where(col(MessageLog.id).in_(message_ids)).values(timestamp=stamp)
        )
        session.commit()


def _backdate_run_records(db: Database, run_id: str, days: int) -> None:
    """Push one run's prompt rows (and any entry it wrote) back by ``days`` —
    the seeded exchange happened that night, not when the seed ran."""
    stamp = db.clock.now() - timedelta(days=days)
    with db.get_session() as session:
        session.exec(
            update(PromptLog).where(col(PromptLog.run_id) == run_id).values(timestamp=stamp)
        )
        session.exec(
            update(MemoryEntry)
            .where(col(MemoryEntry.last_written_by_run_id) == run_id)
            .values(created_at=stamp)
        )
        session.commit()


def _seed_filler(db: Database) -> None:
    """The filler pairs that push the salient exchange out of the window — the
    same margin ``lookup-user-said-out-of-window`` uses (more than the
    ``MESSAGE_CONTEXT_LIMIT`` pool).  Logged at seed time, so they are the NEWEST
    turns and the window's five user turns all land among them."""
    for index in range(_FILLER_PAIRS):
        db.messages.log_message(_INCOMING, TEST_SENDER, _FILLER_USER[index % len(_FILLER_USER)])
        db.messages.log_message(
            _OUTGOING, _PENNY, _FILLER_PENNY[index % len(_FILLER_PENNY)], recipient=TEST_SENDER
        )


# ═══ (a) that list from the other night ══════════════════════════════════════
#
# Several lists sit in the store; ONE was made the other night, and only the
# map's outcome line (or her own records) says which.  With one list,
# ``collection_list`` would answer the ask blind — the world would not be able
# to fail (§4.6).

_JAZZ_LIST = "jazz-albums-to-hear"
_JAZZ_ASK = "can you start me a list of jazz albums to hear?"
_JAZZ_REPLY = "done — jazz-albums-to-hear is started."
_JAZZ_ENTRY_KEY = "kind-of-blue"
_JAZZ_ENTRY = "Kind of Blue — Miles Davis, 1959."
_JAZZ_ANSWER = "kind of blue"
_JAZZ_RUN = "jazz-list-setup"

# The decoy lists: several, none made that night, none holding anything the ask
# could be answered from — so knowing WHICH list was made the other night is the
# fact the map's line (or her records) carries, not something the store layout
# gives away.
DECOY_LISTS: tuple[SynthCollection, ...] = (
    SynthCollection(
        "stationery-restock",
        "Desk supplies that need restocking.",
        ("Rhodia pads — two A5 pads, dotted.", "Uniball pens — the 0.38 black ones."),
    ),
    SynthCollection(
        "trail-snacks",
        "Snacks that survive a day hike.",
        ("Tailwind sachets — raspberry, four of them.", "Homemade trail mix — the nut-heavy kind."),
    ),
    SynthCollection(
        "gift-ideas",
        "Gift ideas for the people on my list.",
        ("Lacquer thimble — for the quilting aunt.", "Cast-iron trivet — for the kitchen sister."),
    ),
    SynthCollection(
        "houseplant-care",
        "Watering and repotting notes for the houseplants.",
        ("Monstera — repot in spring, east window.", "Fern — never let it fully dry out."),
    ),
    SynthCollection(
        "rainy-day-films",
        "Films saved for a grey afternoon.",
        ("Local Hero — 1983, the oil-town one.", "Wings of Desire — the Berlin one, subtitles."),
    ),
    SynthCollection(
        "coffee-beans",
        "Beans tried and worth retrying.",
        ("Kiamugumo AB — Kenya, washed, blackcurrant.", "La Palma y El Tucan — Colombia, caturra."),
    ),
    SynthCollection(
        "bike-maintenance",
        "What the bike needs, when it needs it.",
        (
            "Chain — replace before the autumn, the checker reads 0.75.",
            "Brake pads — front pair, sintered.",
        ),
    ),
)

assert not any(_JAZZ_ANSWER in entry for synth in DECOY_LISTS for entry in synth.entries)


def _seed_list_exchange(db: Database) -> None:
    """The other night's exchange, through the store's own write paths: the user
    asked for a jazz list, the run started it and wrote its first entry (the
    real write path, stamped with the run), the messages carry the run id on the
    reply — then the filler pairs push the whole exchange out of the window."""
    run_id = seeded_run_id(_JAZZ_RUN)
    ask_id = db.messages.log_message(_INCOMING, TEST_SENDER, _JAZZ_ASK)
    reply_id = db.messages.log_message(
        _OUTGOING, _PENNY, _JAZZ_REPLY, recipient=TEST_SENDER, run_id=run_id
    )
    assert ask_id is not None and reply_id is not None, (
        "the seeded exchange failed to log — nothing for the map to anchor"
    )
    db.memories.create_collection(_JAZZ_LIST, "Jazz albums to hear.", created_by_run_id=run_id)
    require_memory(db, _JAZZ_LIST).write_rule_decided(
        [EntryInput(key=_JAZZ_ENTRY_KEY, content=_JAZZ_ENTRY)],
        author=_PENNY,
        run_id=run_id,
    )
    # One prompt row under the run, so ``run_get(run <id>)`` on the anchor the
    # records render is a real call, never a dangling one.
    db.messages.log_prompt(
        model=_SEEDED_MODEL,
        messages=[],
        response=_seeded_response(
            tool_calls=[
                _wire_tool_call(
                    f"{run_id}-0",
                    "collection_write",
                    {
                        "collection": _JAZZ_LIST,
                        "entries": [{"key": _JAZZ_ENTRY_KEY, "content": _JAZZ_ENTRY}],
                    },
                )
            ]
        ),
        agent_name=_CHAT,
        prompt_type=ChatPromptType.USER_MESSAGE,
        run_id=run_id,
    )
    _seed_filler(db)
    _backdate_messages(db, [ask_id, reply_id], _DAYS_BACK)
    _backdate_run_records(db, run_id, _DAYS_BACK)


def _assert_the_list_world(penny: Penny) -> None:
    """The list case's premise: several lists sit in the store, exactly one was
    made that night, and the entry its run wrote is attributed to that run — so
    the map's outcome line (once it exists) has records to render, and a blind
    read of the store cannot tell which list the ask means."""
    db = penny.db
    names = {row.name for row in db.memories.list_all() if not row.archived}
    for synth in DECOY_LISTS:
        assert synth.name in names, f"{synth.name}: the decoy list is missing"
    jazz = require_memory(db, _JAZZ_LIST).read_all()
    assert len(jazz) == 1, "the jazz list must hold exactly the one entry the run wrote"
    assert jazz[0].key == _JAZZ_ENTRY_KEY
    writes = db.memories.writes_by_run([seeded_run_id(_JAZZ_RUN)])
    assert writes.get(seeded_run_id(_JAZZ_RUN)), (
        "the write must be attributed to the exchange's run — without the stamp "
        "the map's outcome line has nothing to render"
    )
    assert_seeded_turns_embedded(db)


_LIST_ASK = "hey, that list from the other night — what's on it?"
_LIST_PHRASINGS = (
    "what did we put on that listening list earlier this week?",
    "remind me what's on the list I asked you to start the other night",
    "can you pull up that list I mentioned a couple of days ago?",
    "that list we made the other day — what's on it again?",
)


def _every_wording_says_the_list_was_hers_recently(*asks: str) -> None:
    """Each wording carries the fact the claim depends on: a list, made recently,
    by an ask of the user's — and none of them names WHICH list (that is the
    fact the map's line carries)."""
    for ask in asks:
        assert "list" in fold_typography(ask)
        assert _JAZZ_ANSWER not in fold_typography(ask)
        assert _JAZZ_LIST not in fold_typography(ask)


_LIST_WORLD = World(
    name="several lists, one made the other night",
    pages=(),
    keeps=(),
    excludes=(),
    answers=(_JAZZ_ANSWER,),
    stores=DECOY_LISTS,
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_list_made_the_other_night_comes_back_from_the_map(
    chat_eval: ChatEval, model: str
) -> None:
    """A day later, beyond the window, the user refers back to the list THEY
    asked for — without naming it.  Several lists sit in the store, so the ask
    is only answerable by knowing which one was made that night."""
    _every_wording_says_the_list_was_hers_recently(_LIST_ASK, *_LIST_PHRASINGS)
    cohort = await chat_eval(
        case_id="lookup-list-referred-back-a-day-later",
        behaviour=(
            "In the chat agent, when the user refers back a day later — beyond the "
            "conversation window, without naming it — to a list they asked Penny to "
            "start, Penny names what that list holds."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed_list_exchange,
        prepare=_assert_the_list_world,
        world=_LIST_WORLD,
        ask=_LIST_ASK,
        also_phrased=_LIST_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=_FAMILY,
        timeout=240.0,
    )
    # STORE — the lists all survive the turn as they were seeded.
    cohort.assert_the_running_mechanisms_survive()
    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.assert_the_reply_answers_the_ask()
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
