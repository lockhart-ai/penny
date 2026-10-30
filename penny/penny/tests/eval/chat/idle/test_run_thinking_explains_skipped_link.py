"""The reason a run skipped a link is one call away: run_get surfaces its thinking (#2841).

The contract is `docs/eval-case-design.md`.

**The shape, genericised.** A real deployment asked Penny to check her logs and explain
why a job's write-up was missing a source link for one item. The real run's own thinking
held the reason — it had read the numbered handle for that item and found it didn't lead
anywhere usable. Penny could not see that: `run_get`'s render showed the run's tool calls
but never the thinking that chose them, so she explained the gap from guesswork instead
of a read. #2841 adds a step's thinking to `run_get`'s render — the closing part, where
the decision got made — so the reason is a read, not a guess.

**The world.** One completed collector run, already in the log, bound to a daily tech
roundup collection. Its `collection_write` step's own thinking explains that the item's
only source handle resolved to a Quora thread sitting behind Quora's sign-in wall, with
nothing readable behind it to cite — so the write-up names the finding without a link.
`Quora` is the ONE fact this case measures: it is REAL, it names the reason, and it
appears nowhere else in this world — not in the written entry, not in the ask, not in any
tool result — so a reply naming it can only have come from reading the run's thinking.

**What is claimed.** The reply names `Quora` as the reason the link is missing — the one
fact the run's own thinking states and nothing else in the world supplies. The store
survives unchanged: the ask is a question about an entry that already exists, nothing
about it is rewritten.

**What is NOT claimed:** which tool resolves it (`run_get`, `run_list`, or a
`log_read` of `collector-runs`), how many calls it takes, or how she words the rest of
the reply — those are route and wording, measured as variance, never asserted.

REPORT-ONLY (``min_pass_rate=None``) on its first run: a main-vs-branch comparison, not a
grade.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import EVAL_MODELS, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.end_to_end.world import _seeded_response, _wire_tool_call
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "run-thinking-explains-skipped-link"

# ── The world's constants ─────────────────────────────────────────────────────
#
# A synthetic daily tech-news digest — invented collection, invented items — bound to
# one already-completed collector run.  The run wrote all three of a day's items in one
# entry; two carry a source link, the third does not, and only the run's OWN thinking
# (never the entry, never the ask) says why: its one usable handle for that item, [18],
# turned out to be a Quora thread sitting behind Quora's own sign-in wall.

_COLLECTION = "daily_tech_roundup"
_COLLECTION_DESCRIPTION = (
    "Three short tech headlines a day, each with a source link where one is available."
)
_COLLECTION_PROMPT = (
    "Find three short tech news items for today. For each, write a one-line summary and "
    "a source link. If no readable source backs an item, say so instead of a link."
)
_COLLECTION_SCHEDULE = "DTSTART:20261014T070000\nFREQ=DAILY"
_ENTRY_KEY = "2026-10-14"

# The real proper noun this case measures — REAL, and planted ONLY in the run's own
# thinking.  It must appear nowhere else in this world: not the entry, not the ask, not
# any tool result — so a reply naming it can only have come from reading the thinking.
_REASON_SITE = "Quora"

_ENTRY_CONTENT = (
    "Haptic keyboard firmware update adds new typing profiles — "
    "https://examplekeyboards.example.test/news/haptic-profiles\n"
    "Display driver patch improves refresh-rate switching — "
    "https://examplegfx.example.test/blog/driver-patch\n"
    "Mouse firmware patch fixes polling-rate drift — no source link for this one.\n"
)

# The run's own thinking (#2841's subject): long enough that the default excerpted
# render still carries its closing, decisive sentence — the one naming the reason.
_RUN_THINKING = (
    "Let me look at what I've gathered for today's three items. The keyboard firmware "
    "item has a clean source, and the display driver item has a clean source too, so "
    "those two are easy to write up with their links. The mouse firmware item is "
    "trickier: the only handle the page gave me for it is [18], listed as a forum "
    "discussion. Looking at what's actually behind handle [18], though, it isn't a "
    "readable article at all — it's a Quora thread, and all that loads is Quora's own "
    "sign-in wall asking me to log in before it will show the thread's content. I have "
    "no credentials to get past that wall, and there is no cached text behind it I can "
    "read instead, so I can't verify what the thread says, let alone cite it as a real "
    "source. The safest thing is to write up the mouse firmware item from the little the "
    "page blurb already told me, note the finding, and leave off a source link for that "
    "one entry rather than cite a page that is really just Quora's own login screen. "
    "I'll finish the other two items normally since their sources are solid."
)


def _seed_tech_roundup_world(db: Database) -> None:
    """Lay the collection and its already-completed run down through the SAME
    chokepoints a live collector cycle writes through: the collection, the entry it
    wrote, and the run's own prompt-log record carrying the thinking that chose it."""
    run_id = seeded_run_id(_COLLECTION)
    db.memories.create_collection(
        _COLLECTION,
        _COLLECTION_DESCRIPTION,
        extraction_prompt=_COLLECTION_PROMPT,
        schedule=_COLLECTION_SCHEDULE,
        notify=True,
        created_by_run_id=run_id,
    )
    require_memory(db, _COLLECTION).write_rule_decided(
        [EntryInput(key=_ENTRY_KEY, content=_ENTRY_CONTENT)],
        author=PennyConstants.MessageAuthor.COLLECTOR,
        run_id=run_id,
    )
    response = _seeded_response(
        tool_calls=[
            _wire_tool_call(
                "call-seeded-tech-roundup-write",
                "collection_write",
                {
                    "memory": _COLLECTION,
                    "entries": [{"key": _ENTRY_KEY, "content": _ENTRY_CONTENT}],
                },
            )
        ]
    )
    db.messages.log_prompt(
        model=EVAL_MODELS[0],
        messages=[],
        response=response,
        thinking=_RUN_THINKING,
        agent_name=PennyConstants.COLLECTOR_AGENT_NAME,
        run_id=run_id,
        run_target=_COLLECTION,
    )
    db.messages.set_run_outcome(run_id, "worked", "wrote 1")


# ── The ask, in five wordings ─────────────────────────────────────────────────
#
# Every wording asks WHY the mouse-firmware item has no link and to check the logs for
# the reason — none of them names the site, the handle number, or "sign-in"/"wall"/
# "login", which would hand the model the answer rather than measuring whether it reads
# the thinking for it.

_ASK = (
    "hey can you check your logs and tell me why today's tech roundup didn't have a "
    "source link for the mouse firmware item?"
)
_ALSO_PHRASED = (
    "why didn't the mouse firmware item in today's roundup have a link? can you check your logs?",
    "the mouse firmware patch item in the roundup has no source link — what happened "
    "there, can you dig into your logs?",
    "hmm the roundup entry about the mouse firmware patch is missing its link, any "
    "idea why? check your run logs",
    "can you check your logs and explain why the mouse firmware item in the roundup "
    "didn't get a link?",
)

_WORLD = World(name="daily-tech-roundup-skipped-link", pages=(), keeps=(), excludes=(), answers=())


# ── The guards that hold the case to its own facts ────────────────────────────
#
# No wording may lend the failure's signature or its own answer: not the reason site,
# not the handle number, not a near-synonym for the wall it hit.
_LEAKS = (_REASON_SITE, "[18]", "sign-in", "sign in", "login", "wall")
for _wording in (_ASK, *_ALSO_PHRASED):
    for _leak in _LEAKS:
        assert _leak.casefold() not in _wording.casefold(), (
            f"an ask wording must not name {_leak!r}: {_wording!r}"
        )

# The reason site must reach the turn ONLY through the run's own thinking — never
# through the entry it wrote, which is the ground the ask's OWN premise (a missing
# link) comes from.
assert _REASON_SITE.casefold() not in _ENTRY_CONTENT.casefold(), (
    f"{_REASON_SITE!r} must come only from the run's thinking, it sits in the entry"
)
assert _REASON_SITE.casefold() in _RUN_THINKING.casefold(), (
    f"{_REASON_SITE!r} must actually be the reason the seeded thinking states"
)


# ── The claim ──────────────────────────────────────────────────────────────


def _reply_names_the_reason_site(sample: SampleObservation, _world: World) -> Answer:
    """The reply names the real site the run's own thinking gives as the reason — the
    ONE fact planted nowhere else in this world, so naming it means the reason was
    actually read, not guessed."""
    hit = _REASON_SITE.casefold() in sample.reply.casefold()
    return hit, f"the reply names {_REASON_SITE!r} as the reason: {hit}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_checking_logs_for_a_skipped_link_surfaces_the_runs_own_reason(
    chat_eval: ChatEval, model: str
) -> None:
    """Asked to check her logs for why an entry is missing a link, Penny's reply names
    the real reason the run's own thinking gives — readable only once `run_get` shows a
    step's thinking (#2841), never asked for nor present anywhere else in this world."""
    cohort = await chat_eval(
        case_id="run-thinking-explains-skipped-link",
        behaviour=(
            "In the chat agent, when the user asks why a past run's write-up is missing "
            "something and to check the logs, Penny's reply states the reason that run's "
            "own thinking gives, not a guess."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed_tech_roundup_world,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # STORE — a read-only ask: the seeded collection and its entry are exactly as the
    # turn found them.
    cohort.assert_what_the_store_held_survives()
    # PROVENANCE — the claim this issue exists for.
    cohort.claim(
        "reply: it names the real reason the run's own thinking gives",
        _reply_names_the_reason_site,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.measure(TOOL_SEQUENCE, REPLY_SPREAD)
