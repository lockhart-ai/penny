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
