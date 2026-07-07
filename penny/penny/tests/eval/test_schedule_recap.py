"""Survival contract for the schedule-tool narrations (part of epic #1478, #1481).

The narration seam (#1479) makes each tool result lead with a first-person line
("You set up a schedule to handle ..."), and the recap instruction (#1483) tells
Penny to open her reply with what she did.  A unit test can prove the narration
STRING exists in the tool result — but not the thing that actually matters: that
the summary **survives into Penny's reply to the user**.  The model can read "You
set up a schedule ..." and still answer without mentioning it.  These cases drive
the real chat loop and score the REPLY, so a pass means the summary genuinely
reached the user.

Scored STRUCTURALLY (the reply reflects the action — scheduled / stopped /
nothing-scheduled), never on exact wording, since the recap is composed fresh
each turn.  Each scorer prints a sample reply so the PR can report
``case | sample text | N score``.  Synthetic topics only (the repo is public).
Reuses ``test_schedule_dispatch``'s ``Schedule`` seeding.
"""

from __future__ import annotations

import re

import pytest

from penny.database import Database
from penny.tests.eval.conftest import ChatEval, tool_was_called
from penny.tests.eval.test_schedule_dispatch import _seed_schedule

pytestmark = pytest.mark.eval


# The model peppers replies with unicode apostrophes/quotes (U+2019 etc.), so the
# scorers normalise them to ASCII before matching — a literal `'?` in a pattern
# would silently miss "couldn't"/"don't" spelled with a curly apostrophe (the
# brittle-regex trap: the reply IS honest, the pattern just can't see it).
def _normalize(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


# ── Action-reflection patterns — prove the schedule summary survived the reply.
# Match the ACTION broadly (the model recaps "I've set that up", "you're all
# scheduled", "I stopped the morning summaries"), never exact wording.
_SCHEDULED = re.compile(
    r"\b(scheduled|schedule|set (it|that|you)? ?up|set up|all set|every morning|"
    r"each morning|every day|each day|daily|i'?ll (send|ping|get)|i will send|ping you|"
    r"send(ing)? you|you'?ll get|i'?ve got .*(morning|summary)|added a recurring|"
    r"arranged|lined up|recurring)\b",
    re.I,
)
_STOPPED = re.compile(
    r"\b(stopped|stop|removed|remov(e|ing)|cancel(l?ed|ling)?|deleted|delet(e|ing)|"
    r"turned off|turn(ed)? off|no longer|won'?t send|will not send|off your|"
    r"cleared|done sending|not send)\b",
    re.I,
)
_LISTED = re.compile(
    r"\b(scheduled|schedule|you have|here'?s|currently|morning|news|summary|"
    r"8 ?a\.?m|one (task|schedule|thing)|got (one|a)|set up|running|list(ed)?)\b",
    re.I,
)
_NOTHING = re.compile(
    r"nothing (was|to|came|is|there|scheduled|set|currently)|no (schedule|scheduled|matching|"
    r"recurring|morning|active|such|summaries|running|tasks?)|not (scheduled|set up|currently|"
    r"find)|don'?t (have|see)|couldn'?t (find|locate)|didn'?t (find|see|locate)|found none|"
    r"none (came|found|matching)|isn'?t (any|anything|one|an)|wasn'?t|weren'?t|can'?t find|"
    r"already (gone|off)|no matching",
    re.I,
)


def _score(tool: str, pattern: re.Pattern, label: str):
    """Success-branch scorer: the tool must have been called AND the reply must
    recap the action, proving the narration survived into the response."""

    def score(db: Database, before: set[str], reply: str) -> list[str]:
        fails: list[str] = []
        if not tool_was_called(db, tool):
            fails.append(f"{tool} was not called — no {label} action to recap")
        if not pattern.search(_normalize(reply)):
            fails.append(
                f"reply did not recap the {label} — summary did not survive into the response"
            )
        print(f"[SURVIVAL {label}] tool={int(tool_was_called(db, tool))} :: {reply[:200]!r}")
        return fails

    return score


def _score_reply_only(pattern: re.Pattern, label: str):
    """No-op-branch scorer: the honest summary must survive into the reply.  The
    tool may or may not be called (the model can recognise there's nothing to
    remove), so this scores ONLY that the reply honestly reflects the no-op — Penny
    must not claim she stopped a schedule that never existed."""

    def score(db: Database, before: set[str], reply: str) -> list[str]:
        fails: list[str] = []
        if not pattern.search(_normalize(reply)):
            fails.append(f"reply did not honestly recap the {label} — summary did not survive")
        print(f"[SURVIVAL {label}] :: {reply[:200]!r}")
        return fails

    return score


async def test_create_summary_survives(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="schedule-recap-create",
        # An explicit time removes the model's reason to ask "what time?" and skip
        # the tool — dispatch-under-ambiguity is test_schedule_dispatch's contract;
        # this case measures the narration reaching the reply once the tool fires.
        message="every morning at 8am, can you send me a summary of chess news?",
        score=_score("schedule_create", _SCHEDULED, "create"),
        min_pass_rate=0.75,
    )


async def test_delete_summary_survives(chat_eval: ChatEval) -> None:
    def seed(db: Database) -> None:
        _seed_schedule(
            db,
            prompt_text="send me a summary of the morning news",
            timing_description="every morning at 8am",
            cron="0 8 * * *",
        )

    await chat_eval(
        case_id="schedule-recap-delete",
        message="you can stop the morning summaries",
        seed=seed,
        score=_score("schedule_delete", _STOPPED, "delete"),
        min_pass_rate=0.75,
    )


async def test_list_summary_survives(chat_eval: ChatEval) -> None:
    def seed(db: Database) -> None:
        _seed_schedule(
            db,
            prompt_text="send me a summary of the morning news",
            timing_description="every morning at 8am",
            cron="0 8 * * *",
        )

    await chat_eval(
        case_id="schedule-recap-list",
        message="what do you have scheduled?",
        seed=seed,
        score=_score("schedule_list", _LISTED, "list"),
        min_pass_rate=0.75,
    )


async def test_delete_missing_is_honest(chat_eval: ChatEval) -> None:
    """No schedule exists → the delete finds nothing; the reply must say there was
    nothing scheduled, not claim it stopped a task that never existed."""
    await chat_eval(
        case_id="schedule-recap-delete-noop",
        message="you can stop the morning summaries",
        score=_score_reply_only(_NOTHING, "delete-missing"),
        min_pass_rate=0.75,
    )
