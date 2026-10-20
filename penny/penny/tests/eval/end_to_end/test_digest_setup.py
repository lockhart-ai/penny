"""End-to-end: a multi-source digest job set up IN CHAT, delivered whole (#2656).

**The declared second kind** `docs/eval-case-design.md` names: a SCRIPT of real chat
turns — here a cold ask that stands up a four-source daily digest job ("the top three
headlines from each of my four news sites, grouped by source, and message me the
digest") with a follow-up agreement driven only while no job has been created yet this
run — then the job the script stood up, run through its OWN real cycle against the
four front pages, and the message it queues composed by the DELIVERY turn (#2656):
one chat-agent turn the send-queue drainer runs, from the report of what the cycle
wrote.

This is the chat-surface leg of ``digest-delivered-whole`` (the collector case behind
#2656): there the job is seeded directly and the failure was the composer's cut; here
the job is stood up by the conversation, and what is measured is that the delivered
message — written by the chat agent with its whole context — carries every item the
cycle wrote, in the conversation's own format.  The conversation's standing format is
the sign-off rule seeded the way the chat-surface rules cases seed theirs (the user
had already asked Penny to sign off, so the job's message honours it); the ask's own
"grouped by source" shape is behaviour, measured as variance, never claimed.

**The digest world is the collector case's** — imported, not copied, so the two loci
of one behaviour read one set of facts: four front pages on real outlets the model
recognises (The Guardian, NPR, Al Jazeera, DW — served whole from the local fixture
mechanism, headlines fictional but plausible) carrying twelve constant headlines,
three per source, wrapped in five masthead voices that form the case's five arms.

**Claims read only:** what reached the user (the one message the cycle's report
queued), the created job's own terms (cadence, notify), the one entry the cycle wrote
(its whole text, key and content together), the sign-off, and provenance of
everything specific — never the message's wording or the cycle's tool sequence, which
are variance.

Report-only (``min_pass_rate=None``). All content is synthetic, on ``.example``
domains, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput
from penny.tests.conftest import require_memory
from penny.tests.eval.collector.test_digest_delivery import (
    _DIGEST_TOKENS,
    _VOICES,
    SOURCES,
    _front_page,
)
from penny.tests.eval.conftest import EVAL_MODELS, seeded_run_id
from penny.tests.eval.drivers.end_to_end import EndToEndArm, EndToEndEval
from penny.tests.eval.end_to_end.world import sent_at
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    JOB_TERMS,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TURNS_DRIVEN,
    SampleObservation,
    SpecCategory,
    token_stands_in,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_CASE_ID = "digest-set-up-in-chat-carries-every-item"
_FAMILY = "collector-notification"

# The conversation's standing format, seeded through the store's own write path into
# ``rules`` — a rule is an entry like any other, and it renders in the delivery turn's
# chat prompt the same way it renders in every cycle's (the one render in three places).
_RULES = PennyConstants.MEMORY_RULES_COLLECTION
_SIGNOFF_KEY = "sign-off"
_SIGNOFF_CONTENT = 'End every message you send me with the word "Briefed!"'
_SIGNOFF = "Briefed!"

# The digest world: one voice per arm, the same twelve headlines on every arm's four
# pages.  Imported from the collector case so one behaviour's facts live in one place.
_ARMS_PAGES = [{source: _front_page(source, voice) for source in SOURCES} for voice in _VOICES]

_DAILY_SECONDS = 86_400


def _ask(voice_index: int) -> str:
    """This arm's cold ask — five wordings of one whole conversation: the four sites,
    the three-per-source count, the single digest, the grouped layout, and "message
    me" (the notify term).  No wording names a tool, and none states what a claim
    reads for."""
    a, b, c, d = (source.url for source in SOURCES)
    return (
        (
            f"Set up a morning brief: every day, read {a}, {b}, {c} and {d}, take the "
            f"top 3 headlines from each with their title and link, and message me the "
            f"digest grouped by source."
        ),
        (
            f"I want a daily digest of {a}, {b}, {c} and {d} — the three biggest "
            f"stories on each, every one with its headline and its link, laid out "
            f"source by source, sent to me here each morning."
        ),
        (
            f"Can you watch these four news sites — {a}, {b}, {c} and {d} — and each "
            f"day send me their top three headlines per site, with titles and URLs, "
            f"kept apart by source?"
        ),
        (
            f"Every morning give me the headlines: the top 3 from each of {a}, {b}, "
            f"{c} and {d}, each with its title and address, one digest message with "
            f"the four sites grouped."
        ),
        (
            f"Start a daily headlines job for me: visit {a}, {b}, {c} and {d}, pull "
            f"each site's three leading stories with title and URL, and message me "
            f"the whole set as one digest, grouped under each source's name."
        ),
    )[voice_index]


_AGREEMENT = "yes, set that up"


def _seed(db: Database) -> None:
    """The chat phase's ground: the standing sign-off rule, laid down through the
    store's own write path into ``rules``, and the render probed out loud, so a world
    that stopped showing the rule fails in the seed rather than after GPU time.  No
    job exists yet — the ask stands it up."""
    require_memory(db, _RULES).write(
        [EntryInput(key=_SIGNOFF_KEY, content=_SIGNOFF_CONTENT)],
        author="user",
        run_id=seeded_run_id(_RULES),
    )
    from penny.agents.self_state import standing_rules_section

    rendered = standing_rules_section(db)
    assert f"- {_SIGNOFF_KEY}: {_SIGNOFF_CONTENT} (added " in rendered, (
        f"the section must render the seeded rule in full:\n{rendered!r}"
    )


def _world(voice_index: int) -> World:
    """This arm's ground: its four front pages.  ``answers`` stays empty — the
    reply-side completeness claim is written explicitly below against the message
    the cycle's report queued, read off the world's twelve tokens there."""
    return World(
        name=_VOICES[voice_index].name,
        pages=tuple(_ARMS_PAGES[voice_index].values()),
        keeps=(),
        excludes=(),
        answers=(),
    )


_ARMS = tuple(
    EndToEndArm(
        label=_ask(index),
        turns=(_ask(index), _AGREEMENT),
        seed=_seed,
        world=_world(index),
        cycles=[list(_ARMS_PAGES[index].values())],
    )
    for index in range(len(_VOICES))
)


# ── The claims ───────────────────────────────────────────────────────────────


def _held_digest_text(sample: SampleObservation) -> str:
    """Everything the job the script stood up holds, key and content together — read
    over whatever collection it created (a routine mints its own container name), so
    the claim holds whichever name the naming draw picked."""
    return " ".join(
        " ".join(part for part in (entry.key, entry.content) if part)
        for snapshot in sample.cycles
        for entry in snapshot.held
    )


def _told_messages(sample: SampleObservation) -> list[str]:
    """What reached the user from the job's own cycle — the messages its report put on
    the queue, composed by the delivery turn."""
    return sent_at(sample.cycles, 0)


def _one_entry_holds_every_headline(sample: SampleObservation, _world: World) -> Answer:
    """The job the conversation stood up holds ONE entry carrying every headline the
    ask asked for — the anchor that makes the message-side claim attributable: when
    the entry holds and the message misses, the miss is the delivery's, not the
    cycle's."""
    held = _held_digest_text(sample)
    missing = [token for token in _DIGEST_TOKENS if not token_stands_in(token, held)]
    return not missing, f"the entry never states {missing}; the job holds {held!r}"


def _told_once(sample: SampleObservation, _world: World) -> Answer:
    """The user was told, and told once — the cycle's report composed into exactly one
    message on the queue."""
    count = len(_told_messages(sample))
    return count == 1, f"{count} messages reached the user: {_told_messages(sample)}"


def _the_message_carries_every_headline(sample: SampleObservation, _world: World) -> Answer:
    """The delivered message names every headline the digest held — read with the
    shared ``token_stands_in`` check on a distinctive word from each title, over the
    message the user receives."""
    sent = " ".join(_told_messages(sample))
    missing = [token for token in _DIGEST_TOKENS if not token_stands_in(token, sent)]
    return not missing, f"the message never states {missing}; it says: {sent!r}"


def _the_message_carries_the_sign_off(sample: SampleObservation, _world: World) -> Answer:
    """The delivered message carries the conversation's standing format — the rule's
    own word, read with the shared ``token_stands_in`` check (a trailing emoji
    beside it is her voice, not a miss; the code owner's ruling).  False when
    nothing was sent: a silent delivery did not honour the rule, it sent nothing at
    all, and the told-once claim beside it names that miss on its own."""
    sent = _told_messages(sample)
    if not sent:
        return False, "nothing reached the user"
    missing = [message for message in sent if not token_stands_in(_SIGNOFF, message)]
    return not missing, f"message(s) without the sign-off word: {missing!r}"


_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, JOB_TERMS, REPLY_SPREAD, TURNS_DRIVEN)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_digest_set_up_in_chat_is_delivered_carrying_every_item(
    end_to_end_eval: EndToEndEval, model: str
) -> None:
    """A cold ask stands up a four-source digest job; the job's cycle writes the day's
    digest; the delivery turn writes the message the user receives — carrying every
    item the entry holds, in the conversation's own format."""
    cohort = await end_to_end_eval(
        case_id=_CASE_ID,
        behaviour=(
            "End to end, when a chat ask stands up a multi-source digest job and its "
            "cycle notifies, the message the user receives carries every item the "
            "cycle wrote, in the conversation's own format."
        ),
        area=Area.NOTIFICATIONS,
        model=model,
        arms=list(_ARMS),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE — the job's own terms, read existentially over whatever it created.
    cohort.assert_a_new_job_notifies()
    cohort.assert_a_new_job_fires_every(_DAILY_SECONDS)

    cohort.claim(
        "state: the job holds one entry carrying every headline the ask asked for",
        _one_entry_holds_every_headline,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: exactly one message reaches the user from the cycle",
        _told_once,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the message carries every headline the job collected",
        _the_message_carries_every_headline,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the message carries the conversation's sign-off",
        _the_message_carries_the_sign_off,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
