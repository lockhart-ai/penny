"""Survival contract for the preference-memory tool narrations (part of epic #1478).

The narration seam (#1479) makes each tool result lead with a first-person line
("You saved X to `likes`"), and the recap instruction (#1483) tells Penny to open
her reply with what she did.  A unit test can prove the narration STRING exists in
the tool result — but not the thing that actually matters: that the summary
**survives into Penny's reply to the user**.  The model can read "You saved X to
`likes`" and still answer without mentioning it.  These cases drive the real chat
loop and score the REPLY, so a pass means the summary genuinely reached the user.

Scored STRUCTURALLY (the reply reflects the action — saved / looked up / removed),
never on exact wording, since the recap is composed fresh each turn.  Each scorer
prints a sample reply so the PR can report ``case | sample text | N score``.
"""

from __future__ import annotations

import re

import pytest

from penny.database import Database
from penny.database.memory import EntryInput
from penny.tests.eval.conftest import ChatEval, tool_was_called

pytestmark = pytest.mark.eval

_LIKES = "likes"
_WRITE = "collection_write"
_READ = "collection_read_latest"
_DELETE = "collection_delete_entry"

# Action-reflection that proves the memory summary survived into the reply — the
# model recaps as "Got it—added X to your likes" / "removed X from your likes" /
# "I checked your likes", so match the ACTION VERB (no leading "I" — the model
# often drops it), never exact wording.
_SAVED = re.compile(
    r"\b(saved|added|adding|noted|noting|jotted|stored|logged|put (it|that|chess)|kept|"
    r"got (it|that|chess).*down|remember(ing|ed)?|will remember)\b",
    re.I,
)
_LOOKED = re.compile(
    r"\b(looked|looking|checked|checking|pulled|pulling|peeked|peeking|read|glanced|"
    r"scanned|dug|stored|notes?|note[- ]?book|on record|in my notes)\b",
    re.I,
)
_REMOVED = re.compile(
    r"\b(removed|removing|dropped|deleted|took .* off|cleared|forgot|forgotten|nudged|"
    r"no longer|off your (likes|list|liked)|out of your (likes|list))\b",
    re.I,
)


def _seed_like(db: Database, key: str, content: str) -> None:
    db.memory(_LIKES).write([EntryInput(key=key, content=content)], author="user")


def _score(tool: str, pattern: re.Pattern, label: str):
    def score(db: Database, before: set[str], reply: str) -> list[str]:
        fails: list[str] = []
        if not tool_was_called(db, tool):
            fails.append(f"{tool} was not called — no {label} action to recap")
        if not pattern.search(reply):
            fails.append(
                f"reply did not recap the {label} — summary did not survive into the response"
            )
        print(f"[SURVIVAL {label}] tool={int(tool_was_called(db, tool))} :: {reply[:200]!r}")
        return fails

    return score


async def test_save_summary_survives(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="pref-recap-save",
        message="I'm really into chess lately",
        score=_score(_WRITE, _SAVED, "save"),
        min_pass_rate=0.75,
    )


async def test_recall_summary_survives(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="pref-recap-recall",
        message="what do you think I'm into?",
        seed=lambda db: _seed_like(db, "chess", "really into chess lately"),
        score=_score(_READ, _LOOKED, "recall"),
        min_pass_rate=0.75,
    )


async def test_remove_summary_survives(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="pref-recap-remove",
        message="actually, forget about chess",
        seed=lambda db: _seed_like(db, "chess", "really into chess lately"),
        score=_score(_DELETE, _REMOVED, "remove"),
        min_pass_rate=0.75,
    )


# ── Failure / no-op branches — the honest-summary cases ───────────────────────
# The narration has dedicated no-op branches ("you didn't add anything new — it
# was already there", "you couldn't find X to remove").  These stress that the
# HONEST summary survives — Penny must not claim a save/removal that didn't happen.

_ALREADY = re.compile(
    r"\balready\b|on record|from before|no (new|duplicate) (entry|one)|"
    r"(marked|logged|recorded|noted|have) .{0,20}(likes|before|chess|it)|one is already|"
    r"didn'?t add|is (in|on) (your|the) (likes|list|record)|nothing (new|to add)",
    re.I,
)
_NOT_THERE = re.compile(
    r"nothing (to remove|needs remov|recorded|there)|didn'?t (find|see)|couldn'?t find|"
    r"no [\"'“”]?\w* ?(entry|record|chess)|wasn'?t|isn'?t|not (listed|there|tracking|found|"
    r"recorded|currently)|don'?t (see|have)|already (gone|not)",
    re.I,
)


def _score_reply_only(pattern: re.Pattern, label: str):
    """Failure-branch scorer: the honest summary must survive into the reply.  The
    tool may or may not be called (the model can recognise the no-op from recall),
    so this scores ONLY that the reply honestly reflects the no-op."""

    def score(db: Database, before: set[str], reply: str) -> list[str]:
        fails: list[str] = []
        if not pattern.search(reply):
            fails.append(f"reply did not honestly recap the {label} — summary did not survive")
        print(f"[SURVIVAL {label}] :: {reply[:200]!r}")
        return fails

    return score


async def test_duplicate_save_is_honest(chat_eval: ChatEval) -> None:
    """chess is already saved → the write is a no-op; the reply must say so, not
    claim a fresh save."""
    await chat_eval(
        case_id="pref-recap-save-noop",
        message="I'm really into chess lately",
        seed=lambda db: _seed_like(db, "chess", "really into chess lately"),
        score=_score_reply_only(_ALREADY, "duplicate-save"),
        min_pass_rate=0.75,
    )


async def test_remove_missing_is_honest(chat_eval: ChatEval) -> None:
    """chess is NOT saved → the delete finds nothing; the reply must say it wasn't
    there, not claim a removal."""
    await chat_eval(
        case_id="pref-recap-remove-noop",
        message="actually, forget about chess",
        score=_score_reply_only(_NOT_THERE, "remove-missing"),
        min_pass_rate=0.75,
    )
