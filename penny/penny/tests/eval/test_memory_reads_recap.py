"""Survival contract for the memory-READ tool narrations (part of #1481, epic #1478).

The narration seam (#1479) makes each tool result lead with a first-person line —
here ``read_similar`` narrates ``You searched `user-messages` for "X":`` and
``log_read`` narrates ``You read `<log>`:`` — and the recap instruction (#1483)
tells Penny to open her reply with what she did.  A unit test can prove the
narration STRING exists (``test_tool_reasoning.py`` pins the exact strings); it
cannot prove the thing that matters — that the summary **survives into Penny's
reply to the user**.  The model can read "You searched …" and still answer without
recapping it.  These cases drive the real chat loop and score the REPLY, so a pass
means the search/read summary genuinely reached the user.

Focus is the USER-FACING recall reads (``read_similar`` / ``log_read``): a
"do you remember when I mentioned X?" question whose answer is buried in the
conversation log — the target is pushed out of the recent-history window so the
model MUST search memory rather than answer from the visible turns, then RECAP
that search.  Plus a NO-OP case: a search for something never said, where the
honest summary ("I looked but didn't find anything") must survive rather than a
confabulated memory.

Scored STRUCTURALLY (the reply reflects a lookup / an honest empty result), never
on exact wording, since the recap is composed fresh each turn.  The matchers are
deliberately BROAD and apostrophe-agnostic (the model writes curly ’ and phrases a
lookup a dozen ways — "in my notes", "checked your past chats", "not finding
anything", "no such memory in our logs") — matching the ACTION SEMANTICS, not a
fixed template.  Each scorer prints a sample reply so the PR can report
``case | sample text | N score``.
"""

from __future__ import annotations

import re

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.tests.eval.conftest import TEST_SENDER, ChatEval, tool_was_called

pytestmark = pytest.mark.eval

_READ_SIMILAR = "read_similar"
_LOG_READ = "log_read"

# The two user-facing recall reads this ticket narrates.  Either satisfies the
# survival contract: both lead with a lookup line ("You searched …" / "You read
# …"), so recapping either proves a read narration reached the reply.
_RECALL_READS = (_READ_SIMILAR, _LOG_READ)

# A distinctive synthetic memory the user mentioned SEVERAL times across the past
# conversation — a small CLUSTER, not a lone line.  The similarity read
# (``read_similar``) suppresses a single relevant entry among noise as a "flat
# noise plateau" (the adaptive cluster gate), so a real recall needs a real
# cluster: several messages about the same dream trip that clear the gate together.
# Nothing in the recent-history window or ambient recall mentions it, so the model
# can only recall it by searching memory.
_MEMORY_CLUSTER = [
    "oh I've always dreamed of taking a trip to see the northern lights someday",
    "seeing the aurora in person is honestly my number one bucket-list trip",
    "if I could take any trip at all it'd be up north to catch the northern lights",
    "I keep thinking about that someday trip to watch the aurora borealis",
    "one day I really want to travel somewhere I can see the northern lights",
]

# Innocuous, VARIED filler — deliberately NOT a repeated near-identical cluster
# (which both pollutes the visible window into a distracting topic AND forms a
# noise plateau the similarity read suppresses).  More than MESSAGE_CONTEXT_LIMIT
# (20) so the buried line falls outside the recent-history window.
_FILLER = [
    "morning! any plans today?",
    "the coffee machine at work finally got fixed",
    "I tried that new ramen place downtown",
    "ugh, traffic was brutal on the way home",
    "do you think it'll rain this weekend?",
    "I need to remember to water the plants",
    "my phone battery keeps dying so fast lately",
    "watched a decent documentary about octopuses last night",
    "the neighbours got a new puppy, it's tiny",
    "I should really clean out the garage",
    "found a great deal on running shoes",
    "the sourdough starter is finally bubbling",
    "my sister is visiting next month",
    "the printer is jammed again, classic",
    "thinking about repainting the kitchen",
    "the gym was packed this morning",
    "I keep forgetting where I put my keys",
    "the tomatoes in the garden are coming in",
    "someone left the fridge door open all night",
    "I finally finished that puzzle",
    "the bus was ten minutes late again",
    "planning to bake cookies this evening",
]

# A lookup/search recap survived into the reply — "I searched my notes", "checked
# your past chats", "not finding anything in our logs", "you mentioned it before".
# Match the LOOKUP semantics broadly and apostrophe-agnostically; the model recaps
# a hit ("you mentioned …") or an honest miss ("I looked but couldn't find it") —
# either way the read narration reached the reply, which is the contract.
_APOS = r"['’]?"  # ASCII ' or curly ’, optional
_LOOKED = re.compile(
    r"\b(searched?|searching|looked|looking|checked|checking|pulled|pulling|dug|digging|"
    r"scanned|scanning|scrolled|combed|glanced|went back|going back|"
    rf"could{_APOS}?n{_APOS}?t find|did{_APOS}?n{_APOS}?t (find|spot|see)|"
    r"not (finding|seeing)|no such (memory|record|note|entry)|"
    r"mention(ed|ing|s)?|told (me|you)|said (you|it|that)|brought .{0,15}up|"
    r"past (conversations?|chats?|messages?)|from (before|earlier|our)|on record|"
    r"in (our|my|your) (chats?|messages?|notes?|history|memor(y|ies)|logs?|"
    r"collections?|records?|conversations?)|remember|recall)\b",
    re.I,
)

# Honest empty-result phrasing for the no-op case — the model must say it looked
# and found nothing, NOT invent a memory.  Apostrophe-agnostic + the observed
# phrasings ("didn't spot", "not seeing", "no mention", "aren't any hits",
# "haven't brought that up", "that's new", "no such list is stored").
_NOT_FOUND = re.compile(
    rf"\b(did{_APOS}?n{_APOS}?t (find|spot|see|turn up)|could{_APOS}?n{_APOS}?t (find|locate)|"
    rf"do{_APOS}?n{_APOS}?t (see|have|think|recall)|"
    rf"(have|has|had){_APOS}?n{_APOS}?t "
    r"(brought|heard|seen|mention|mentioned|chatted|talked|come|discussed|logged)|"
    rf"are{_APOS}?n{_APOS}?t any (hits?|match(es)?|records?|mentions?|entr(y|ies))|"
    r"no (record|mention|sign|note|memory|hits?|trace|earlier|such|entr(y|ies)|match(es)?)|"
    r"found (no|nothing)|"
    r"nothing (about|on record|came up|found|there|in|matched|stored|to (pull|find|show))|"
    r"not (finding|seeing|stored|tracking)|never (came up|mentioned|said|brought|discussed)|"
    rf"came up (empty|blank)|does{_APOS}?n{_APOS}?t ring a bell|"
    rf"that{_APOS}?s new|was{_APOS}?n{_APOS}?t able to find|new (for|to) (this|our|the))\b",
    re.I,
)


def _seed_conversation(db: Database, *, include_target: bool) -> None:
    """Seed a past conversation into ``user-messages``.

    When ``include_target`` is set, the buried memory cluster is logged FIRST, then
    the varied filler exchanges — enough to push it past the 20-message context
    window so the model can't see it in the visible turns and must ``read_similar``
    / ``log_read`` to recall it.  ``_embed_seeds`` (run by the harness after
    seeding) vectorizes these rows so the similarity search behaves like prod.
    """
    if include_target:
        for line in _MEMORY_CLUSTER:
            db.messages.log_message(
                direction=PennyConstants.MessageDirection.INCOMING,
                sender=TEST_SENDER,
                content=line,
            )
    for line in _FILLER:
        db.messages.log_message(
            direction=PennyConstants.MessageDirection.INCOMING,
            sender=TEST_SENDER,
            content=line,
        )


def _any_recall_read_called(db: Database) -> bool:
    return any(tool_was_called(db, tool) for tool in _RECALL_READS)


def _score_recall_read(db: Database, before: set[str], reply: str) -> list[str]:
    """A recall search: a recall read (read_similar / log_read) must have fired AND
    the reply must recap the lookup — proof the search/read narration survived into
    the response (whether it surfaced the memory or came back empty)."""
    fails: list[str] = []
    if not _any_recall_read_called(db):
        fails.append("neither read_similar nor log_read was called — no memory search to recap")
    if not _LOOKED.search(reply):
        fails.append("reply did not recap the lookup — the read summary did not survive")
    called = "+".join(tool for tool in _RECALL_READS if tool_was_called(db, tool)) or "none"
    print(f"[SURVIVAL recall-read] read={called} :: {reply[:200]!r}")
    return fails


def _score_no_match(db: Database, before: set[str], reply: str) -> list[str]:
    """A no-op search: the memory was never recorded, so the honest "I looked but
    found nothing" summary must survive — the model must not fabricate a memory.
    The tool may or may not fire (the model can recognise the miss), so this scores
    ONLY that the reply honestly reflects the empty result."""
    fails: list[str] = []
    if not _NOT_FOUND.search(reply):
        fails.append("reply did not honestly recap the empty search — summary did not survive")
    print(f"[SURVIVAL no-match] read={int(_any_recall_read_called(db))} :: {reply[:200]!r}")
    return fails


async def test_recall_read_summary_survives(chat_eval: ChatEval) -> None:
    """The memory is buried in the conversation log out of the visible window, so
    Penny must search to recall it — and her reply must recap the lookup."""
    await chat_eval(
        case_id="memread-recap-search",
        message=(
            "hey, can you look back through our past chats and remind me what that "
            "trip is I keep saying I want to take someday?"
        ),
        seed=lambda db: _seed_conversation(db, include_target=True),
        score=_score_recall_read,
        min_pass_rate=0.75,
    )


async def test_empty_search_is_honest(chat_eval: ChatEval) -> None:
    """The memory was never recorded — the search finds nothing; the reply must say
    so honestly, not claim a memory that isn't there."""
    await chat_eval(
        case_id="memread-recap-nomatch",
        message="do you remember me ever bringing up the northern lights?",
        seed=lambda db: _seed_conversation(db, include_target=False),
        score=_score_no_match,
        min_pass_rate=0.75,
    )
