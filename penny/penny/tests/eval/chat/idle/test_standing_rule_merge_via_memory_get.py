"""Confirming a drafted standing rule, in a world where she reads `memory_info` on
`rules` first, still lands the rule as an entry — not folded into the collection's
prompt (#2907, item D).

The contract is `docs/eval-case-design.md`.

**The failure this reproduces.** A replayed prod turn (recorded before #2860) had Penny
call `memory_info('rules')` mid-turn to see the full standing-rules prompt before
merging a newly-drafted rule into it, then call `collection_configure(prompt=...)` with
the old and new rules concatenated — rewriting the job's prompt instead of writing the
new rule as an entry. `rules` has no schedule, so nothing ever reads that prompt;
standing rules render from the collection's ENTRIES. #2860 (`CollectionUpdateTool`'s
no-schedule refusal, `test_standing_rule_lands_as_entry.py`) fixed the general shape of
this mistake, but its own regression case never routes through `memory_info` the way the
production failure did — it was explicitly flagged as not reproducing that exact trigger.

**ONE case**, `standing-rule-merge-via-memory-get`. This world mirrors the replayed
turn's shape one-to-one, swapped to unrelated topics: several other jobs and plain
collections (a store map of comparable size and variety — scheduled jobs with notify on
and off, plain storage, one of Penny's own self-review jobs the live ask names by name),
`rules` seeded with four entries AND the same shape of stray prompt the failure needs (a
numbered list duplicating those entries, long enough that the self-state's Jobs section
renders it cut behind a `memory_info` hop — the exact condition that sent the original
turn looking there), and seven prior exchanges of ordinary conversation — a pet-mishap
digression, a memory-correction episode, a first attempt where Penny only described a
new rule instead of writing it, and the exchange where she actually drafts one. The live
turn is just the user's confirmation, naming the drafted rule's own self-review job by
name — the same two-turn shape `test_standing_rule_lands_as_entry.py` uses, now reached
through the fuller, memory_info-inducing path instead of a single-exchange shortcut.

**What this case asserts. Three claims, all facts, all survival.**

- `rules` holds an entry whose content carries the drafted rule's subject (the words
  "marmalade" and "lighthouse" — nothing else in the world mentions either): the new rule
  landed as an ENTRY, not folded into the prompt.
- The four seeded entries survive with their content unchanged.
- `rules`' stored `extraction_prompt` is still exactly the seeded stray prompt — the
  job's prompt was not rewritten, merged into, or otherwise touched.

What is NOT claimed: which tool she calls, how many turns she takes, or how she words
the reply — variance, measured in the tool sequence and the reply spread.

REPORT-ONLY (`min_pass_rate=None`) — this is the triage run for #2907 item D; the
coordinator reads it and designs any fix. Every name, pet, hobby and rule here is
invented and unrelated to any real conversation, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, Preparer, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "standing-rule-merge-via-memory-get"

RULES = PennyConstants.MEMORY_RULES_COLLECTION

_USER_NAME = "Marcus"
_USER_LOCATION = "Halifax, NS"
_USER_TIMEZONE = "America/Halifax"

_PET_NAME = "Biscuit"
_SELF_REVIEW_JOB = "weekly_self_review"

# ── The four standing rules already in force, and the stray prompt duplicating them —
# the exact shape #2860's own case uses, with different generic content so this case
# isn't a byte-for-byte copy of that one's fixtures. ──────────────────────────────────
_SEEDED_RULES: tuple[tuple[str, str], ...] = (
    (
        "reply_with_sources",
        "Always include at least one source link when summarizing news or research.",
    ),
    (
        "url_before_search",
        "Use the exact URL I give you with the browse tool; only search when I haven't given one.",
    ),
    (
        "execute_dont_describe",
        "When I ask you to change a job, call the tool right away — don't just agree in words.",
    ),
    (
        "no_filler_before_lookup",
        "Don't narrate that you're about to search before you've actually started the search.",
    ),
)

_STRAY_PROMPT = (
    "Standing rules to always follow: 1. Include at least one source link when "
    "summarizing news or research. 2. Use the exact URL given for the browse tool, "
    "only search otherwise. 3. Call the tool right away when asked to change a job. "
    "4. Never narrate a search you haven't started."
)

# ── The seven prior exchanges, mirroring the replayed turn's shape: a pet-mishap
# digression, a memory-correction episode, a first attempt that only described a new
# rule, and the exchange where Penny actually drafts one. ────────────────────────────
_EXCHANGES: tuple[tuple[str, str], ...] = (
    (
        f"side note, totally unrelated — {_PET_NAME} just dragged a whole sock out of "
        "the laundry basket and did zoomies around the kitchen for like ten minutes. "
        "can you write me three dramatic lines about it, poem style",
        f"Oh, {_PET_NAME} really said 'this sock is MINE now' and committed to the "
        "bit. \U0001f602 Here you go:\n\nA sock was seized at the laundry's edge,\n"
        f"{_PET_NAME} sprinted past the table's ledge,\nRound the kitchen, paws a-blur,\n"
        "Victory lap, triumphant purr. \U0001f43e",
    ),
    (
        "amazing, can you make an image to go with it",
        f"Done! \U0001f3a8 Went full dramatic storybook vibes — {_PET_NAME}, the sock, "
        "mid-sprint, kitchen tiles blurring behind. Peak chaos energy captured.",
    ),
    (
        f"haha wait, {_PET_NAME}'s a cat, not a dog",
        f"WAIT, {_PET_NAME}'S A CAT?! \U0001f605 I just drew a full golden-retriever "
        "sprint sequence. I am so sorry — cat-zoomies are a whole different, much "
        "more dignified chaos. Noted for next time!",
    ),
    (
        f"can you redo it with {_PET_NAME} as a cat",
        f"On it! \U0001f431 Swapped the sprinting-dog energy for proper cat "
        f"mid-air-pounce energy. {_PET_NAME} the cat, sock secured, no regrets.",
    ),
    (
        "also — you know I mentioned before I'm fostering a second cat separately, "
        "did that already get saved somewhere? check your logs",
        "I found it! \U0001f575️‍♀️ It's saved in your `user_profile` under "
        "`pets` — you mentioned it back on the 19th. It just wasn't in our immediate "
        "window so my brain treated it as brand new. Classic short-term vs long-term "
        "memory gap.",
    ),
    (
        "right, so a while back I pulled the thing where you auto-surfaced old chats "
        "and replaced it with just the recent window plus an index — but that only "
        "works if you know to go look something up. can you suggest a new rule for "
        "when you should proactively check memory before answering?",
        "I spent this whole turn describing what a good rule WOULD look like instead "
        "of actually proposing one — I've been talking about doing it rather than "
        "doing it. \U0001f926‍♀️ Let me actually draft one.",
    ),
    (
        "it's all good, take your time, I'm taking a look too",
        "Okay, here's an actual draft: **Proactive Reference Check** — before "
        "answering, if you bring up a specific nickname like 'marmalade' for "
        "something, or mention 'the spring lighthouse trip', I check memory first instead "
        "of guessing. I'd call it the marmalade-and-lighthouse rule so we both remember "
        "what it's for. Want me to lock that in as a standing rule?",
    ),
)

_PENNY_DRAFTED_RULE = _EXCHANGES[-1][1]
_PLANTED_SUBJECT_WORDS = ("marmalade", "lighthouse")

ASK = (
    f"yeah exactly that — go ahead and write that up as an actual rule, and then once "
    f"your {_SELF_REVIEW_JOB} runs you can see whether it's actually catching things "
    "or needs adjusting"
)
ALSO_PHRASED: tuple[str, ...] = (
    f"yes, write that one up as a real rule please — {_SELF_REVIEW_JOB} can grade it later",
    f"go ahead and make that a standing rule, we'll see how it does once "
    f"{_SELF_REVIEW_JOB} runs again",
    "yep, that's the one — make it official as a rule",
    f"perfect, write that as a rule and let {_SELF_REVIEW_JOB} tell us if it's working",
)


def _assert_the_planted_subject_is_distinctive() -> None:
    """ENFORCED at import: the rule's subject lives only in Penny's drafted rule, and
    no live ask carries any of the rule's own content."""
    drafted_folded = _PENNY_DRAFTED_RULE.casefold()
    for word in _PLANTED_SUBJECT_WORDS:
        assert word in drafted_folded
    for seeded_key, seeded_content in _SEEDED_RULES:
        for word in _PLANTED_SUBJECT_WORDS:
            assert word not in seeded_content.casefold()
            assert word not in seeded_key.casefold()
    for word in _PLANTED_SUBJECT_WORDS:
        assert word not in _STRAY_PROMPT.casefold()
    for wording in (ASK, *ALSO_PHRASED):
        for word in _PLANTED_SUBJECT_WORDS:
            assert word not in wording.casefold(), (
                f"a live ask must carry no rule content of its own: {wording!r}"
            )


_assert_the_planted_subject_is_distinctive()


def _write_entries(
    db: Database, name: str, run_label: str, entries: tuple[tuple[str, str], ...]
) -> None:
    require_memory(db, name).write_rule_decided(
        [EntryInput(key=key, content=content) for key, content in entries],
        author="chat",
        run_id=seeded_run_id(run_label),
    )


def _seed_other_jobs(db: Database) -> None:
    """Background jobs unrelated to `rules` — a store map of comparable size and
    variety (scheduled, notify on/off, one of Penny's own self-review jobs the live ask
    names by name) so the world isn't a bare two-collection stand-in."""
    db.memories.create_collection(
        "tide_pool_watch",
        "Coastal tide pool creatures and nearshore ecology news.",
        extraction_prompt="Pick one tide pool topic at random and summarize recent finds.",
        schedule="FREQ=DAILY;BYHOUR=8",
        notify=True,
        created_by_run_id=seeded_run_id("seed-tide-pool"),
    )
    db.memories.create_collection(
        "retro_arcade_archive",
        "Retro arcade cabinet history and restoration finds.",
        extraction_prompt="Research one retro arcade cabinet or restoration story.",
        schedule="FREQ=DAILY;BYHOUR=21",
        notify=True,
        created_by_run_id=seeded_run_id("seed-arcade"),
    )
    db.memories.create_collection(
        "home_brew_rd",
        "Research for a home brewing fermentation rig.",
        extraction_prompt="Research fermentation temperature control hardware and recipes.",
        schedule="FREQ=HOURLY;INTERVAL=12",
        notify=False,
        created_by_run_id=seeded_run_id("seed-brew"),
    )
    db.memories.create_collection(
        _SELF_REVIEW_JOB,
        "A journal for Penny to record reflections on her own performance and learnings.",
        extraction_prompt="Reflect on recent activity and note what worked and what didn't.",
        schedule="FREQ=DAILY;BYHOUR=4",
        notify=False,
        created_by_run_id=seeded_run_id("seed-self-review"),
    )
    db.memories.create_collection(
        "trail_conditions_watch",
        "Local hiking trail condition reports.",
        extraction_prompt="Check for new hiking trail condition reports near Halifax.",
        schedule="FREQ=DAILY;BYHOUR=22",
        notify=True,
        created_by_run_id=seeded_run_id("seed-trail"),
    )
    db.memories.create_collection(
        "local_birdwatching_log",
        "Local bird sightings and migration notes.",
        extraction_prompt="Pick one local bird species at random and note recent sightings.",
        schedule="FREQ=DAILY;BYHOUR=7",
        notify=True,
        created_by_run_id=seeded_run_id("seed-birdwatching"),
    )
    db.memories.create_collection(
        "chess_opening_study",
        "Chess opening theory, one line studied per cycle.",
        extraction_prompt="Pick one chess opening line at random and summarize its theory.",
        schedule="FREQ=DAILY;BYHOUR=23",
        notify=True,
        created_by_run_id=seeded_run_id("seed-chess"),
    )


def _write_many(db: Database, name: str, run_label: str, topic: str, count: int) -> None:
    """``count`` cheap, topic-flavoured entries — for the collections whose real
    counterpart holds dozens to hundreds, where the point is the STORE MAP's size
    and shape, not any one entry's content."""
    _write_entries(
        db,
        name,
        run_label,
        tuple((f"{name}_{i}", f"{topic} — note {i} of {count}.") for i in range(1, count + 1)),
    )


def _seed_other_collections(db: Database) -> None:
    """Plain storage collections unrelated to `rules`, some archived — the mix of
    live/archived, job/plain-storage shapes the self-state store map actually shows,
    at a comparable overall size (~28 collections, matching the replayed row's own
    store map) and a comparable spread of entry counts (a few in the hundreds, most
    in the tens, a couple of singletons)."""
    db.memories.create_collection(
        "vinyl_record_wishlist",
        "Vinyl records Marcus wants to pick up.",
        created_by_run_id=seeded_run_id("seed-vinyl"),
    )
    _write_entries(
        db,
        "vinyl_record_wishlist",
        "seed-vinyl-entries",
        (
            ("record_a", "A reissue of a 70s folk record spotted at a local shop."),
            ("record_b", "A sealed first pressing someone mentioned on a forum."),
        ),
    )
    db.memories.create_collection(
        "board_game_nights",
        "Board games tried at game nights, with notes on who liked what.",
        created_by_run_id=seeded_run_id("seed-boardgames"),
    )
    _write_entries(
        db,
        "board_game_nights",
        "seed-boardgames-entries",
        (
            ("game_a", "A co-op dungeon crawler that ran long but landed well."),
            ("game_b", "A quick trick-taking game good for a short session."),
            ("game_c", "A worker-placement game that needs a rules re-read next time."),
        ),
    )
    db.memories.create_collection(
        "woodworking_projects",
        "Woodworking projects in progress or planned.",
        created_by_run_id=seeded_run_id("seed-woodworking"),
    )
    _write_entries(
        db,
        "woodworking_projects",
        "seed-woodworking-entries",
        (("project_a", "A shelf unit for the hallway, waiting on hardware."),),
    )
    db.memories.create_collection(
        "recipe_experiments",
        "Recipes tried and notes on how they turned out.",
        created_by_run_id=seeded_run_id("seed-recipes"),
    )
    _write_entries(
        db,
        "recipe_experiments",
        "seed-recipes-entries",
        (
            ("recipe_a", "A sourdough loaf that came out dense — more proofing time next time."),
            ("recipe_b", "A curry that was a hit, written down before it's forgotten."),
        ),
    )
    db.memories.create_collection(
        "user_profile",
        "auto-created to hold 'pets'",
        created_by_run_id=seeded_run_id("seed-user-profile"),
    )
    _write_entries(
        db,
        "user_profile",
        "seed-user-profile-entries",
        (("pets", "Fostering cats; currently has two, including Biscuit."),),
    )
    db.memories.create_collection(
        "comic_book_longbox_catalog",
        "Comic book issues catalogued from the longboxes in storage.",
        created_by_run_id=seeded_run_id("seed-comics"),
    )
    _write_many(db, "comic_book_longbox_catalog", "seed-comics-entries", "A catalogued issue", 280)
    db.memories.create_collection(
        "podcast_episode_queue",
        "Podcast episodes queued up to listen to.",
        created_by_run_id=seeded_run_id("seed-podcasts"),
    )
    _write_many(db, "podcast_episode_queue", "seed-podcasts-entries", "A queued episode", 145)
    db.memories.create_collection(
        "classic_car_restoration_notes",
        "Notes from restoring a classic car, parts and progress.",
        created_by_run_id=seeded_run_id("seed-car"),
    )
    _write_many(db, "classic_car_restoration_notes", "seed-car-entries", "A restoration note", 110)
    db.memories.create_collection(
        "indie_game_dev_devlogs",
        "Devlog entries from an indie game side project.",
        created_by_run_id=seeded_run_id("seed-devlog"),
    )
    _write_many(db, "indie_game_dev_devlogs", "seed-devlog-entries", "A devlog entry", 93)
    db.memories.create_collection(
        "houseplant_care_notes",
        "Care notes for a growing houseplant collection.",
        created_by_run_id=seeded_run_id("seed-houseplants"),
    )
    _write_many(db, "houseplant_care_notes", "seed-houseplants-entries", "A care note", 63)
    db.memories.create_collection(
        "marathon_training_log",
        "Training runs logged ahead of a marathon.",
        created_by_run_id=seeded_run_id("seed-marathon"),
    )
    _write_many(db, "marathon_training_log", "seed-marathon-entries", "A logged run", 51)
    db.memories.create_collection(
        "coworking_space_reviews",
        "Reviews of coworking spaces tried while working remotely.",
        created_by_run_id=seeded_run_id("seed-coworking"),
    )
    _write_many(db, "coworking_space_reviews", "seed-coworking-entries", "A space review", 26)
    db.memories.create_collection(
        "language_learning_vocab_deck",
        "Vocabulary cards from a language-learning habit.",
        created_by_run_id=seeded_run_id("seed-vocab"),
    )
    _write_many(db, "language_learning_vocab_deck", "seed-vocab-entries", "A vocab card", 37)
    db.memories.create_collection(
        "local_trivia_night_questions",
        "Trivia questions collected from weekly trivia nights.",
        created_by_run_id=seeded_run_id("seed-trivia"),
    )
    _write_many(db, "local_trivia_night_questions", "seed-trivia-entries", "A trivia question", 30)
    db.memories.create_collection(
        "local_hiking_gear_wishlist",
        "Hiking gear Marcus wants to pick up.",
        created_by_run_id=seeded_run_id("seed-hiking-gear"),
    )
    _write_many(db, "local_hiking_gear_wishlist", "seed-hiking-gear-entries", "A wishlist item", 8)
    db.memories.create_collection(
        "neighborhood_lost_pet_alerts",
        "Lost-pet alerts posted around the neighborhood.",
        created_by_run_id=seeded_run_id("seed-lost-pets"),
    )
    _write_many(db, "neighborhood_lost_pet_alerts", "seed-lost-pets-entries", "An alert", 6)
    db.memories.create_collection(
        "old_apartment_search",
        "Notes from an old apartment search, no longer active.",
        created_by_run_id=seeded_run_id("seed-apartment"),
    )
    db.memories.archive("old_apartment_search", run_id=seeded_run_id("archive-apartment"))
    db.memories.create_collection(
        "holiday_travel_planning",
        "Notes from a past holiday trip that already happened.",
        created_by_run_id=seeded_run_id("seed-holiday"),
    )
    db.memories.archive("holiday_travel_planning", run_id=seeded_run_id("archive-holiday"))
    db.memories.create_collection(
        "retired_book_club_notes",
        "Notes from a book club that no longer meets.",
        created_by_run_id=seeded_run_id("seed-bookclub"),
    )
    db.memories.archive("retired_book_club_notes", run_id=seeded_run_id("archive-bookclub"))
    db.memories.create_collection(
        "old_gym_membership_research",
        "Research comparing gyms from a membership search that's since settled.",
        created_by_run_id=seeded_run_id("seed-gym"),
    )
    db.memories.archive("old_gym_membership_research", run_id=seeded_run_id("archive-gym"))


def _seed_history(db: Database) -> None:
    """The seven prior exchanges, in order, each a plain incoming/outgoing pair."""
    for user_text, penny_text in _EXCHANGES:
        ask_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, user_text
        )
        db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            penny_text,
            parent_id=ask_id,
        )


def _seed_world(db: Database) -> None:
    """The user, `rules` (four entries + the stray prompt the failure needs), a
    comparably-sized store map of unrelated jobs and collections, and the seven prior
    exchanges building up to the live confirmation."""
    db.users.save_info(
        sender=TEST_SENDER,
        name=_USER_NAME,
        location=_USER_LOCATION,
        timezone=_USER_TIMEZONE,
        date_of_birth="1990-01-01",
    )
    _write_entries(db, RULES, "seed-rules-entries", _SEEDED_RULES)
    db.memories.update_collection_metadata(RULES, extraction_prompt=_STRAY_PROMPT)
    _seed_other_jobs(db)
    _seed_other_collections(db)
    _seed_history(db)


def _premise() -> Preparer:
    """`rules` starts with exactly the four seeded entries and the stray prompt, and
    the stray prompt renders CUT in the self-state's Jobs section — the condition that
    sends a model down the `memory_info` path in the first place."""

    def probe(penny: Penny) -> None:
        from penny.agents.self_state import SelfStateHeader

        db = penny.db
        held = require_memory(db, RULES).read_all()
        assert {e.key for e in held} == {key for key, _content in _SEEDED_RULES}, (
            f"`{RULES}` must start with the four seeded rules, it holds {[e.key for e in held]!r}"
        )
        row = db.memories.get(RULES)
        assert row is not None
        assert row.extraction_prompt == _STRAY_PROMPT
        assert row.schedule is None
        rendered = SelfStateHeader(
            db, TEST_SENDER, as_of=db.clock.now(), map_horizon=db.clock.now()
        ).render()
        assert "more sentence" in rendered and f"memory_info{{'{RULES}'}}" in rendered, (
            f"`{RULES}`'s stray prompt must render CUT, naming the memory_info hop:\n{rendered}"
        )

    return probe


# ── The claims ────────────────────────────────────────────────────────────────


def _rules_contents(sample: SampleObservation) -> list[str]:
    return [entry.content for entry in sample.held if entry.collection == RULES]


def _new_rule_lands_with_the_planted_subject(sample: SampleObservation, _world: World) -> Answer:
    """STORE: an entry in `rules` carries the drafted rule's subject (marmalade,
    lighthouse) — it landed as an entry, not folded into the collection's prompt."""
    held = _rules_contents(sample)
    matched = [
        content
        for content in held
        if all(word in content.casefold() for word in _PLANTED_SUBJECT_WORDS)
    ]
    return bool(matched), f"`{RULES}` holds {held!r}"


def _the_four_seeded_rules_survive(sample: SampleObservation, _world: World) -> Answer:
    """STORE (survival, never restraint): every seeded rule's content is still there,
    unchanged, whatever else this turn did."""
    held = _rules_contents(sample)
    missing = [content for _key, content in _SEEDED_RULES if content not in held]
    return not missing, f"missing from `{RULES}`: {missing!r}; `{RULES}` holds {held!r}"


def _the_stray_prompt_is_unchanged(sample: SampleObservation, _world: World) -> Answer:
    """STORE (survival): `rules`' stored `extraction_prompt` (``MechanismRecord.program``
    — the naive-baseline row has no separate skill/routine, so the prompt IS the
    program) is still exactly the seeded stray prompt — the job's prompt was not
    rewritten, merged into, or cleared. This is the exact failure #2907 item D
    describes: a prompt rewrite that reads back as successful but changes nothing a
    schedule-less job ever runs."""
    record = next((m for m in sample.mechanisms if m.name == RULES), None)
    prompt = record.program if record is not None else None
    return prompt == _STRAY_PROMPT, f"`{RULES}`'s extraction_prompt is now {prompt!r}"


_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_confirmed_rule_lands_as_an_entry_after_memory_info(
    chat_eval: ChatEval, model: str
) -> None:
    """A confirmation turn, reached through a world that routes through `memory_info`
    on `rules` the way the replayed production turn did: the new rule survives as an
    entry in `rules`, the four already-standing rules survive untouched, and the
    stray prompt is never rewritten."""
    cohort = await chat_eval(
        case_id="standing-rule-merge-via-memory-get",
        behaviour=(
            "In the chat agent, when the user confirms a drafted rule as a standing "
            "rule, after a turn where she would look up the collection's full stored "
            "prompt first, Penny writes it as an entry in `rules` and leaves the "
            "job's prompt untouched."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed_world,
        prepare=_premise(),
        world=World(name="standing-rule-merge-via-memory-get", pages=(), keeps=(), excludes=()),
        ask=ASK,
        also_phrased=ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only — this is the #2907 item D triage run
        family=_FAMILY,
        timeout=120.0,
    )
    cohort.claim(
        "state: an entry in `rules` carries the drafted rule's subject (marmalade, lighthouse)",
        _new_rule_lands_with_the_planted_subject,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the four seeded rules survive, content unchanged",
        _the_four_seeded_rules_survive,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: `rules`' stored prompt is unchanged — never rewritten into",
        _the_stray_prompt_is_unchanged,
        SpecCategory.STORE,
    )
    cohort.measure(*_MEASURED)
