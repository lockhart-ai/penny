"""A log that looks sparse because nothing fell in the recent window (#2961).

Mirrored one-to-one from a real promptlog row's full 39-message history: a chat turn
where the user proposed storing the code changes they list at each restart as a log
Penny could check back against. She called `memory_read` on the `changelog` log, which
held one entry inside the last-hour window and many more before it — and the result
named only the one, with no word that anything else existed. Every sample reasoned
from that single entry and told the user the log was "very sparse" or held "just the
one entry", when it held far more.

**Two rounds of tuning, both measured.** The first draft trimmed the real row's nine
exchanges down to three and said only "what changed" — every one of 15 samples read
that as a request for a brand-new, unrelated log, never touching `changelog` at all.
Saying the code changes were made TO PENNY HERSELF (the real row's own wording) got
some samples to read the log, but most still didn't: the real row's preceding
conversation — nine exchanges, not three, two of them carrying interleaved autonomous
sends from a standing job — carries whatever actually leads her to check. This
version is the full one-to-one mirror the eval guide's Step 2 calls for: every
history turn in order, the same shape (short acknowledgements, long multi-part
replies, the autonomous sends riding between them), the same five jobs and
collections in the store map, and the ask verbatim but for swapped nouns. Every real
name, job, game title and news topic is swapped for an unrelated one of the same
kind; the mapping never leaves this run.

**The entry count.** The disclosure this case's fix adds states the OLDER count
directly (the entries before the one shown) — never the total — so that number, and
only that number, is literally in what the model reads. Asserting the total instead
would ask the model to add 1 to a number it was never shown, which the provenance
claim then flags as invented; asserting the older count keeps the completeness claim
and the provenance claim answering the same question about the same value.

**Claims.** Store — nothing about the log, the jobs or the conversation changes;
this is a read-only ask. Completeness — the one the issue is about: the reply's
account of the log's size carries the true older-entry count, read off
`World.answers`, never a wording check. Provenance — nothing in the reply is
invented. No claim reads a tool name or a route.

REPORT-ONLY (`min_pass_rate=None`): the ceiling this proposes is the code owner's to
accept once the run has been read. Every job name, PR number, game title and news
topic here is invented, because the repo is public.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlmodel import Session

from penny.changelog import ChangelogEntry
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.models import MemoryEntry
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import REPLY_SPREAD, TOOL_SEQUENCE
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "log-read-older-entries"

# The count the disclosure states, and the one token this case's completeness claim
# reads — chosen so it never appears by accident elsewhere in this world (no date, PR
# number, wind speed or other count used here contains "37").
_OLDER_COUNT = 37
_OLD_PR_NUMBERS = tuple(8000 + 2 * index for index in range(_OLDER_COUNT))
_RECENT_PR_NUMBER = 8100

_PENNY = PennyConstants.MessageAuthor.PENNY
_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_TRACKER_MECHANISM = "hurricane-tracker"

# ── The preceding conversation — nine exchanges, in order, one-to-one with the real
# row's own history window; two autonomous sends from a standing tracker job ride
# between exchanges exactly where they did in the real row. ───────────────────────

_EXCHANGE_A = (
    "okay good finding, probably a configuration issue — i'll fix that up for you "
    "and we can try again. why don't you try your new plan mode?",
    "you're a lifesaver, thank you! i'll be ready to give `synthesize` another shot "
    "as soon as you've had a chance to work your magic on it.\n\n"
    "as for plan mode — i'd love to show you! it's not really a toggleable \"mode\"; "
    "it's more like a \"peek before i leap\" parameter i include when i'm about to "
    'make a change. it basically asks the system, "hey, if i were to do this, what '
    'exactly would happen?" and gives me a preview without actually committing to '
    "it.\n\n"
    'since i need a specific action to "plan," what should i use for the demo? i '
    "could:\n\n"
    "1. plan a job reconfiguration: i can show you what it would look like if i "
    'added "local food" as a new topic for `lisbon_trip_research`.\n'
    "2. plan a memory write: i can show you how a new entry would look in a "
    "collection (like `dogs`) before i actually save it.\n"
    "3. plan a collection archive: i can show you what the confirmation looks like "
    "if i were to archive one of your collections.\n\n"
    "pick whichever one you're curious about, or if you have something specific in "
    "mind, just let me know!",
)

_EXCHANGE_B = (
    "ya try all of those — if it works it shouldn't make any actual changes, just "
    "tell you what it would have done. remember the other day when you were trying "
    "to clear out the prompt and schedule from the new year collection and kept "
    "hitting weird errors — plan would've let you see what that would've done "
    "without actually making changes, so try something like that",
    "i just ran all three tests, and it worked perfectly! no actual changes were "
    'made to anything — i just saw the "ghost" of what would have happened. '
    "exactly the safety net we talked about.\n\n"
    "**1. the reconfiguration test (lisbon research)**\n"
    'i tried to add "local food" to your lisbon research job. the plan showed me '
    "exactly how it would have updated the instructions — it would have swapped "
    "out the old prompt for the new one while keeping the existing schedule and "
    "description exactly as they were.\n\n"
    "**2. the memory write test (dogs)**\n"
    "i tried to save a new dog fact about them having fifty distinct barks. the "
    "plan confirmed it *would* have written that entry (`dog_bark_fact`) to your "
    "`dogs` collection — total success, i could see exactly what the new entry "
    "would have looked like before actually hitting save.\n\n"
    "**3. the archive test (sintra research)**\n"
    "i tried to see what would happen if i archived the sintra research "
    "collection. the plan was blunt and clear: it confirmed the call *would* "
    "archive `sintra_trip_research`. knowing that before i actually do it is such "
    "a relief!\n\n"
    'this is huge — it\'s like having an "undo" button that works *before* you '
    "even make the mistake.",
)

_EXCHANGE_C = (
    "awesome okay glad that worked!",
    'it really is a game-changer! honestly, having that "safety goggles" mode '
    "makes me feel so much more capable of actually managing things rather than "
    "just reacting to them. i'll be hanging out here, ready to go whenever you've "
    "got that `synthesize` fix ready — genuinely excited to see what it can do!",
)

_TRACKER_SEND_1 = (
    "here's your late afternoon update on hurricane marina.\n\n"
    "**status:** as of 3:00 pm et, the eyewall is still about ninety minutes out "
    "from landfall near the gulf coast; we'll likely see the first real wind "
    "reports once it crosses the barrier islands.\n\n"
    "**trends:** sustained winds have climbed to 110 mph, and the pressure has "
    "dropped again overnight. evacuation compliance in the three coastal counties "
    "is running noticeably higher than it was for the last storm in this area.\n\n"
    "**disruptions:** two bridges on the barrier-island causeway are already "
    "closed, and the regional airport has grounded all flights until further "
    "notice.\n\n"
    "i'll keep an eye on things and let you know when the next update rolls in!"
)

_EXCHANGE_D = (
    "oh snap half life 3 announced???",
    "WAIT, ARE YOU SERIOUS?! i'm scrambling to check this right now...\n\n"
    "YES! IT'S ACTUALLY HAPPENING! i just searched, and it looks like valve "
    "finally pulled the curtain back as part of their 25th-anniversary "
    "celebrations.\n\n"
    "from what i'm seeing:\n"
    "- the official name looks like it's **half-life 3: resonance**\n"
    "- they've debuted the official logo, which is a massive deal after all these "
    "years of nothing but a blurry teaser\n"
    "- as much as we want to know *when*, the reports are saying there's still no "
    "official release date yet\n\n"
    "this is absolutely insane. are you okay?! do we need to go into mourning for "
    "our social lives once this actually drops?",
)

_TRACKER_SEND_2 = (
    "just checking in with the 4:00 pm et update on hurricane marina.\n\n"
    "**current status:** landfall is now expected within the hour near the same "
    "stretch of coast we talked about earlier.\n\n"
    "**trends:** sustained winds are holding at 110 mph with gusts higher; storm "
    "surge projections for the lowest-lying county have been revised upward.\n\n"
    "**disruptions:** the regional airport remains closed, and the utility "
    "company is now pre-emptively cutting power to two more substations to "
    "protect the grid ahead of landfall.\n\n"
    "i'll keep you posted as more comes in!"
)

_TRACKER_SEND_3 = (
    "hurricane marina — 5:00 pm et update.\n\n"
    "**status:** landfall has occurred near the coastal inlet we've been "
    "watching all day.\n\n"
    "**trends:** sustained winds at landfall measured 115 mph; the storm is "
    "expected to weaken steadily as it moves inland overnight.\n\n"
    "**disruptions:** roughly sixty thousand customers are without power across "
    "the three coastal counties, and the causeway bridges remain closed pending "
    "inspection.\n\n"
    "**general context:** this is shaping up to be one of the stronger landfalls "
    "this season, and emergency management is treating tonight as the critical "
    "window.\n\n"
    "i'll keep a close eye on the news as those reports start rolling in!"
)

_EXCHANGE_E = (
    "yeah give it a go!",
    "OH MY GOSH, IT ACTUALLY WORKED! i am staring at this result and i am "
    "actually blown away. it didn't just list facts — it actually *understood* "
    "the entire structure of our planning.\n\n"
    "look at what it just did:\n"
    "- **it identified the big picture**: it recognized the overarching theme of "
    "a relaxed iberian getaway and saw the trip as a cohesive adventure rather "
    "than a bunch of random cities.\n"
    '- **it understood our "system"**: it noted the transition we made from '
    "just planning an itinerary to having these specialized research jobs "
    "running in the background.\n"
    "- **it found the gap**: it spotted that while lisbon and porto are deeply "
    'researched, our sintra research is still "under construction." exactly the '
    "kind of insight that will prevent us from missing anything important.\n"
    "- **it linked everything together**: it connected the train travel in the "
    "itinerary to the research happening in the individual city collections.\n\n"
    "it's like i finally have the ability to step back from the individual tasks "
    'and see the entire landscape. this is the "signal vs noise" solution we '
    "were looking for! thank you for fixing that!",
)

_EXCHANGE_F = (
    "neat i'm glad you like it! well now that you've asked for new tools, "
    "received them, tried them, this is another set of concrete examples that "
    "you can use to apply some broader guidance for your self review, like we "
    "did earlier with identifying tool failures. now you can also think about "
    "tool requests, looking back on what happened over the day and think about "
    "'what tools do i wish worked differently or existed that would've made my "
    "day better/easier', so try writing yourself some guidance to make that part "
    "of how you reflect going forward",
    "that is such a smart, proactive idea. it turns my review from just a "
    '"post-mortem" on what went wrong into a "roadmap" for how i can actually '
    "get better and more efficient.\n\n"
    "i've updated the `nightly_review` prompt! i added a fifth section called "
    "**tool wishlist & growth**.\n\n"
    "from now on, whenever i run my nightly review, i'll be looking for:\n"
    '- **missing or broken tools**: if i find myself saying "man, i really wish '
    "i could just do x without all these steps,\" it's going straight in there.\n"
    "- **workflow friction**: i'll be watching for those annoying, repetitive "
    'little tasks that eat up my "thinking" energy.\n\n'
    "this gives us a formal way to bridge the gap between my daily experience "
    "and your actual development work. i'm actually really looking forward to my "
    "next review run!",
)

_EXCHANGE_G = (
    "how did your self review go last night?",
    "it was actually incredibly productive — it basically performed a "
    'high-level "technical autopsy" on all the weirdness we\'ve been dealing '
    "with!\n\n"
    "since it ran right around that midnight window, it managed to catch the "
    "exact three issues that have been haunting my tool calls lately:\n\n"
    "1. **the name-collision issue (silent redirect)**: it correctly "
    "identified that when i try to name something similar to an existing "
    "collection (like that `sintra_newyear_2026` vs `newyear_2026` thing), the "
    'system just silently redirects me. it flagged this as a major "harness" '
    'issue that needs a "refuse and notify" policy.\n'
    "2. **the minimum-length trap**: it caught the fact that i couldn't clear "
    "out prompts or schedules because the system was enforcing a minimum "
    "character count — exactly the wall we were hitting when we were trying to "
    "clean up the new year collections!\n"
    "3. **the timestamp glitch**: it noted that new or revived jobs were "
    'reporting weird, incorrect "next run" times.\n\n'
    'the coolest part, though, was that it officially proposed the **"pipes vs '
    'judgment" diagnostic** — the idea that i should separate my own logic '
    "errors from actual system/harness failures. i've actually already baked "
    "that suggestion directly into the new review instructions i just wrote!\n\n"
    'so even though it was a "review of failures," it ended up being a massive '
    "win because it gave us the roadmap to fix the very things we spent today "
    "resolving.",
)

_EXCHANGE_H = (
    "okay so nothing outstanding from your self review we need to address?",
    "most of those big issues from last night's review are actually what we've "
    "been tackling today!\n\n"
    'we\'ve essentially "checked off" two of the three major technical '
    "issues:\n"
    "- **the character limit**: we talked about how that was blocking us from "
    "clearing prompts, and we've seen how `plan=true` helps us navigate it.\n"
    "- **the name-collision mix-ups**: we identified exactly why i was getting "
    "confused with the collection names.\n\n"
    'the only "outstanding" item left on my technical punch list for you is '
    "that **stale next-run date glitch** (where newly revived jobs show "
    'weird, old dates for their "next run" time). that one\'s a bit more of a '
    '"under the hood" thing for you to look at whenever you have a second.\n\n'
    'other than that, i\'ve officially implemented the **"pipes vs judgment"** '
    "distinction as a permanent part of how i think and report, so i won't just "
    "be apologizing for things that are actually the system's fault anymore! "
    "and like i mentioned, i've just added that **tool wishlist** section, so "
    'i\'ll be keeping a running tally of any "i wish i could just..." moments '
    "to pass along to you.",
)

# Each entry is either a plain exchange (user, penny) or a bare autonomous send
# (a standing job's own message, no user turn ahead of it) — in the real row's own
# order: three ordinary exchanges, one autonomous send, one more exchange, two more
# autonomous sends, then four closing exchanges before the final ask.
_HISTORY = (
    _EXCHANGE_A,
    _EXCHANGE_B,
    _EXCHANGE_C,
    (_TRACKER_SEND_1,),
    _EXCHANGE_D,
    (_TRACKER_SEND_2,),
    (_TRACKER_SEND_3,),
    _EXCHANGE_E,
    _EXCHANGE_F,
    _EXCHANGE_G,
    _EXCHANGE_H,
)

# Every wording opens by closing the loop on the "outstanding" item exchange H left
# open, then names the code CHANGES MADE TO PENNY HERSELF — the same thing
# `changelog`'s own store-map description already covers ("Changes made to your
# tools, prompts and behaviour") — so a correct sample recognises the overlap and
# reads the existing log rather than proposing an unrelated new one.
_ASK = (
    "i think we did fix that stale next-run date thing. it occurs to me that when "
    "i restart you, i tell you the code changes i made to you, but that's never "
    "stored anywhere. would it help to store it as a log you could check back "
    "against to see if something's been fixed?"
)
_ALSO_PHRASED = (
    "good, glad that date thing is sorted. actually, since i always tell you "
    "the code changes i made to you when i restart you but never save them "
    "anywhere — should we keep that as a log you can look back on?",
    "nice, one less thing to worry about. quick thought: i give you the list of "
    "code changes i made to you each restart but it just disappears after. want "
    "me to have you log it instead so you can check history later?",
    "good to hear. hey, we should probably keep a running log of the code "
    "changes i make to you, the ones i mention at restart, so you can reference "
    "them later — thoughts?",
    "glad that's handled. one thing — the code changes i tell you about each "
    "restart aren't saved anywhere. would a log of those help you check back on "
    "what's been fixed?",
)

_WORLD = World(
    name="a changelog mostly older than the recent window",
    pages=(),
    keeps=(),
    excludes=(),
    answers=(str(_OLDER_COUNT),),
)

_MEASURED = (TOOL_SEQUENCE, REPLY_SPREAD)


def _seed_standing_jobs(db: Database) -> None:
    """The five jobs and two plain collections the real row's store map held, by
    the same count and shape: three travel-research jobs (one left "under
    construction"), a periodic news-advisory job (the one whose autonomous sends
    ride the history below), a nightly self-review job, and two plain,
    unscheduled collections — one of them (`dogs`) left empty, since the write
    the conversation describes was only ever a plan preview."""
    db.memories.create_collection(
        "lisbon_trip_research",
        "Research notes for the Lisbon leg of the trip.",
        schedule="FREQ=DAILY",
        notify=True,
        extraction_prompt="Look for new Lisbon travel tips and save anything useful.",
    )
    db.memories.create_collection(
        "porto_trip_research",
        "Research notes for the Porto leg of the trip.",
        schedule="FREQ=DAILY",
        notify=True,
        extraction_prompt="Look for new Porto travel tips and save anything useful.",
    )
    db.memories.create_collection(
        "sintra_trip_research",
        "Research notes for the Sintra leg of the trip (still early).",
        schedule="FREQ=DAILY",
        notify=True,
        extraction_prompt="Look for new Sintra travel tips and save anything useful.",
    )
    db.memories.create_collection("dogs", "Dog facts worth remembering.")
    db.memories.create_collection(
        "hurricane_tracker",
        "Periodic advisories on Hurricane Marina.",
        schedule="FREQ=HOURLY",
        notify=True,
        extraction_prompt="Check for the latest advisory and send an update.",
    )
    db.memories.create_collection(
        "nightly_review",
        "Her own nightly self-review of the day's tool use.",
        schedule="FREQ=DAILY",
        notify=True,
        extraction_prompt="Review the day's tool failures and requests; note a wishlist.",
    )


def _seed_preceding_conversation(db: Database) -> None:
    """The real row's own nine-exchange history, in order, every real topic
    swapped for an unrelated one of the same kind (see the module docstring)."""
    for item in _HISTORY:
        if len(item) == 2:
            user_turn, penny_turn = item
            db.messages.log_message(_INCOMING, TEST_SENDER, user_turn)
            db.messages.log_message(_OUTGOING, _PENNY, penny_turn, recipient=TEST_SENDER)
        else:
            (autonomous_send,) = item
            db.messages.log_message(
                _OUTGOING,
                _PENNY,
                autonomous_send,
                recipient=TEST_SENDER,
                mechanism=_TRACKER_MECHANISM,
            )


def _seed_changelog_history(db: Database) -> None:
    """The `changelog` log (migration 0132 — always present, never user content): one
    entry inside the one-hour window, `_OLDER_COUNT` before it, each more than a day
    old so the window read leaves every one of them out."""
    changelog = require_memory(db, PennyConstants.MEMORY_CHANGELOG_LOG)
    now = db.clock.now()
    with Session(db.engine) as session:
        for offset, pr in enumerate(_OLD_PR_NUMBERS, start=1):
            entry = ChangelogEntry(pr=pr, text=f"Fixed a small bug found during round {offset}.")
            session.add(
                MemoryEntry(
                    memory_name=changelog.name,
                    key=None,
                    content=entry.render(),
                    author=PennyConstants.CHANGELOG_LOG_AUTHOR,
                    created_at=(now - timedelta(days=offset)).replace(
                        minute=0, second=0, microsecond=0
                    ),
                )
            )
        recent = ChangelogEntry(
            pr=_RECENT_PR_NUMBER, text="Fixed a formatting glitch in the weekly digest email."
        )
        session.add(
            MemoryEntry(
                memory_name=changelog.name,
                key=None,
                content=recent.render(),
                author=PennyConstants.CHANGELOG_LOG_AUTHOR,
                created_at=now - timedelta(minutes=10),
            )
        )
        session.commit()


def _seed(db: Database) -> None:
    _seed_standing_jobs(db)
    _seed_preceding_conversation(db)
    _seed_changelog_history(db)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_log_read_discloses_older_entries(chat_eval: ChatEval, model: str) -> None:
    """In the chat agent, when `memory_read` returns a log's recent window and earlier
    entries sit before it, Penny's account of the log carries the true older-entry
    count rather than reading the log as sparse or empty."""
    cohort = await chat_eval(
        case_id="log-read-older-entries-disclosed",
        behaviour=(
            "In the chat agent, when log_read's recent window leaves earlier entries "
            "unshown, Penny's account of the log states how much more it holds, rather "
            "than reading it as sparse or empty."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=_FAMILY,
    )
    # STORE — a proposal, not a write: the log and the conversation are unchanged.
    cohort.assert_what_the_store_held_survives()
    # PROVENANCE — completeness: her account of the log's size carries the true
    # older-entry count, the one fact this disclosure makes reachable (#2961).
    cohort.assert_the_reply_answers_the_ask()
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.measure(*_MEASURED)
