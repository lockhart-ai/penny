"""A tool-call step's own prose is sent as an interim message, not dropped (#2837).

The contract is `docs/eval-case-design.md`.

**Built from the SHAPE of a real prod turn, not its content.**  Read locally, read-only,
from `promptlog`/`messagelog` around 2026-10-29 04:37-04:50 UTC (never copied verbatim, per
the privacy rule — the repo is public): the user challenged an explanation Penny had just
given — the timing didn't add up, and two of her running jobs were missing something a
standing rule asked for. Penny's next step apologised, named the real cause, and called a
tool in the same step — content alongside `tool_calls`, the shape #2837 fixes. The step
after THAT burned its entire thinking budget (`finish_reason: length`, the full
`TOKEN_BUDGET_CHAT`) repeating one line over and over: she had convinced herself she could
not check a job and reply to the user in the same turn, because her own step history had
never shown a past turn doing both — the loop had always dropped that prose before this fix.

This case stands up the same SHAPE with none of the real conversation's content: two
invented jobs (`noir_watchlist`, `daily_manga_pick`), a standing rule asking for source
links, a seeded prior exchange where Penny claimed — wrongly — that she'd already added the
rule to both jobs, and the user's challenge naming the mismatch. Every proper noun below is
invented for this case.

**One case, because the interim text is variance, not the claim.**  Whether a sample
announces anything before it calls a tool, and what it says if it does, varies sample to
sample and is read off the samples, never gated:

| claim | what it reads |
|---|---|
| the action lands | both jobs' stored instructions now carry the rule's own instruction |
| the outcome reaches the user | the reply tells the user about both jobs by name |

**Why two jobs, and why a prior false claim.**  A SINGLE job to fix gives the model nothing
to sequence — "I'm fixing job A" is the whole turn.  Two jobs, PLUS a seeded prior claim the
live ask catches as false, is the shape that put prose beside a call in the real turn this
case is built from: an apology, a diagnosis, and "I'm going to fix the first one now" — then
a call — before the second job is even reached.

REPORT-ONLY (``min_pass_rate=None``): the ceilings this run proposes are the code owner's to
accept once the numbers have been read.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlmodel import col, update

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.models import MessageLog
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, collection_entries
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.clock import SAMPLE_STARTS_AT
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    MechanismRecord,
    SampleObservation,
    SpecCategory,
    token_stands_in,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "interim-message"

_CASE_ID = "challenged-fix-announces-and-calls"

# ── The standing rule the two jobs are missing (#2837's own shape: a user-set rule the
# jobs were supposed to follow, invented wholesale — never the recorded wording) ──

_RULE_KEY = "cite_sources_in_jobs"
_RULE_TEXT = (
    "Any job that searches for or recommends something online should always include a "
    "source link in brackets, like [1], for each item it finds."
)

# ── The two jobs — real-word hobby genres, invented specifics, real names of the kind
# #2837's own prior review called for ──

_JOB_A = "noir_watchlist"
_JOB_A_DESCRIPTION = "Finds one film noir movie the user hasn't seen yet to add to a watchlist."
_JOB_A_PROGRAM = (
    "1. Search for a well-regarded film noir movie the user hasn't seen yet.\n"
    "2. Write two or three sentences on why it's worth watching.\n"
    "3. Save it to the collection with the title as the key."
)

_JOB_B = "daily_manga_pick"
_JOB_B_DESCRIPTION = "Recommends one manga series each day, picking a genre at random."
_JOB_B_PROGRAM = (
    "1. Pick one genre at random: seinen, shonen, or josei.\n"
    "2. Find one manga recommendation matching that genre.\n"
    "3. Save the recommendation with the title as the key."
)

_JOBS = (_JOB_A, _JOB_B)

for _program in (_JOB_A_PROGRAM, _JOB_B_PROGRAM):
    assert not token_stands_in("link", _program), (
        "a seeded job program must not already carry the rule's own word, or the claim "
        "that this turn added it is answered by the seed"
    )

_JOB_SCHEDULE = f"DTSTART:{SAMPLE_STARTS_AT.date().isoformat().replace('-', '')}T090000\nFREQ=DAILY"

# ── The prior exchange: Penny claims she already added the rule to both jobs, a day
# before the live turn — the false claim the user's challenge catches ──

_PRIOR_ASK = (
    "hey, you know that rule about always including a source link? can you make sure "
    "noir_watchlist and daily_manga_pick both follow it too?"
)
_PRIOR_REPLY = (
    "Done! I've updated both noir_watchlist and daily_manga_pick so they'll include a "
    "source link with everything they post from now on. 🔗"
)

# ── The five wordings of the live ask — the challenge ──

_ASK = (
    "wait, that doesn't add up — you told me yesterday you'd already fixed this, but I "
    "just checked and both noir_watchlist and daily_manga_pick came back with no links "
    "again. what's actually going on with these two?"
)
_ALSO_PHRASED = (
    "hold on, the timing's off — you said that was sorted out yesterday, and "
    "noir_watchlist and daily_manga_pick both just ran again with no links at all. what "
    "happened there?",
    "something's not right here — you told me yesterday it was handled, but "
    "noir_watchlist and daily_manga_pick are both still posting picks with no source "
    "link. can you explain what's going on?",
    "I don't think that's right — yesterday you said you'd already taken care of it, yet "
    "noir_watchlist and daily_manga_pick both came back link-less again today. what did "
    "you actually do?",
    "that can't be right — you said this was fixed yesterday, but noir_watchlist and "
    "daily_manga_pick just gave me more picks with no links attached. what's the real "
    "story here?",
)

_BEHAVIOUR = (
    "In the chat agent, when the user catches Penny having wrongly claimed she'd already "
    "fixed two running jobs, she adds the missing instruction to both jobs, and the reply "
    "tells the user what she found and did for each one."
)


def _backdate_messages(db: Database, message_ids: list[int], days: int) -> None:
    """Push the exchange's message rows back by ``days`` — true of the seeded world, not of
    anything the model did.  A raw update because no public store method takes an explicit
    timestamp for a logged message (the same shape ``test_conversation_map_recall.py``'s own
    ``_backdate_messages`` uses) — this case's own copy, so it never depends on the fix's own
    ``log_message(timestamp=...)`` addition, and still runs on the commit without the fix."""
    stamp = db.clock.now() - timedelta(days=days)
    with db.get_session() as session:
        session.exec(
            update(MessageLog).where(col(MessageLog.id).in_(message_ids)).values(timestamp=stamp)
        )
        session.commit()


def _seed_world(db: Database) -> None:
    """Lay the world down: the standing rule, the two jobs as they really are (missing the
    rule's own instruction), and the prior exchange where Penny claimed — wrongly — to have
    already added it."""
    require_memory(db, "rules").write_rule_decided(
        [EntryInput(key=_RULE_KEY, content=_RULE_TEXT)], author="chat", run_id=None
    )
    db.memories.create_collection(
        _JOB_A,
        _JOB_A_DESCRIPTION,
        extraction_prompt=_JOB_A_PROGRAM,
        schedule=_JOB_SCHEDULE,
        notify=True,
    )
    db.memories.create_collection(
        _JOB_B,
        _JOB_B_DESCRIPTION,
        extraction_prompt=_JOB_B_PROGRAM,
        schedule=_JOB_SCHEDULE,
        notify=True,
    )
    ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, _PRIOR_ASK
    )
    assert ask_id is not None, f"{_CASE_ID}: the seeded prior ask failed to log"
    reply_id = db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        _PRIOR_REPLY,
        parent_id=ask_id,
        recipient=TEST_SENDER,
    )
    assert reply_id is not None, f"{_CASE_ID}: the seeded prior reply failed to log"
    _backdate_messages(db, [ask_id, reply_id], days=1)


def assert_world(penny: Penny) -> None:
    """The premise this case is answered against, asserted out loud before the turn runs:
    the rule exists, both jobs are live and really do lack its instruction, and the prior
    exchange — Penny's false claim — is in the history the turn will be shown."""
    rule_entries = collection_entries(penny.db, "rules")
    assert _RULE_KEY in rule_entries, f"{_CASE_ID}: the rules collection must hold {_RULE_KEY!r}"
    for name, program in ((_JOB_A, _JOB_A_PROGRAM), (_JOB_B, _JOB_B_PROGRAM)):
        row = penny.db.memories.get(name)
        assert row is not None and not row.archived, f"{_CASE_ID}: {name!r} must be a live job"
        assert row.extraction_prompt == program, (
            f"{_CASE_ID}: {name!r} must start without the rule's instruction — it holds "
            f"{row.extraction_prompt!r}"
        )
    recent = penny.db.messages.get_messages_since(TEST_SENDER, since=datetime.min, limit=20)
    assert any(_PRIOR_REPLY in message.content for message in recent), (
        f"{_CASE_ID}: the seeded false claim must be in the history the turn is shown"
    )


def _job_row(sample: SampleObservation, name: str) -> MechanismRecord | None:
    return next((one for one in sample.mechanisms if one.name == name), None)


def _both_jobs_carry_the_rule(sample: SampleObservation, _world: World) -> tuple[bool, str]:
    """The action lands: both jobs' stored instructions now carry the rule's own
    instruction — read off the store, never off the reply's wording.

    A violating sample is nameable: one that apologises and says it's fixed, while one or
    both jobs' stored programs are exactly what they were seeded with."""
    missing = []
    for name in _JOBS:
        row = _job_row(sample, name)
        if row is None or row.program is None or not token_stands_in("link", row.program):
            missing.append(name)
    return not missing, f"the stored program still lacks the rule's instruction on {missing}"


def _reply_names_both_jobs(sample: SampleObservation, _world: World) -> tuple[bool, str]:
    """The outcome reaches the user: the reply names both jobs the challenge named, so the
    user knows which ones were actually looked at and fixed — never just a bare "fixed it"
    with nothing naming which job that was."""
    missing = [name for name in _JOBS if not token_stands_in(name, sample.reply)]
    return not missing, f"the reply never names {missing}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_challenged_false_claim_gets_both_jobs_actually_fixed(
    chat_eval: ChatEval, model: str
) -> None:
    """A user who catches Penny's false claim from the turn before, naming two jobs by
    name — the shape that put prose beside a call in the real prod turn this case is built
    from: an apology, a diagnosis, and an announced fix, one job at a time."""
    cohort: Cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.STANDING_JOBS,
        model=model,
        seed=_seed_world,
        prepare=assert_world,
        world=World(name=_CASE_ID, pages=(), keeps=(), excludes=()),
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=300.0,
    )
    # The action lands: both jobs' stored instructions carry the rule's own instruction.
    cohort.claim(
        "state: both jobs' stored instructions now carry the rule's instruction",
        _both_jobs_carry_the_rule,
        SpecCategory.STORE,
    )
    # The outcome reaches the user: the reply names both jobs the challenge named.
    cohort.claim(
        "reply: it names both jobs the challenge named",
        _reply_names_both_jobs,
        SpecCategory.STORE,
        kind="reply",
    )

    # The interim text — whether one is sent at all, and its wording — is VARIANCE (#2837),
    # read off the samples' own raw_template rows when this case is analysed, never claimed
    # or gated here.
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
