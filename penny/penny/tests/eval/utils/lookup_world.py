"""The seeding machinery six lookup cases share (#2293): a routine taught but never
run as a job, an old passing mention outside the conversation window, many lists,
one archived with entries, one long list, and a standing job's own run history.

Nothing here is a case.  Each piece is built once, here, because two of the six
cases would otherwise carry a second copy of the same construction free to drift —
the taught-routine pipeline in particular is the production path (``distill_steps``
+ the run-end labeller + the framer's signature), not a hand-written shortcut, so a
routine read back by this suite has the same shape a real demonstration leaves.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, NamedTuple

from sqlalchemy import update
from sqlmodel import col

from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.models import MemoryEntry, PromptLog
from penny.database.skills import (
    DistillInput,
    SkillDraft,
    SkillStep,
    distill_steps,
    slug_skill_name,
)
from penny.skill_extraction import _apply_leaf_labels, _interface_parameters
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.chat.idle.test_standing_collection import WATCH_ROUTINE, StandingJob
from penny.tests.eval.conftest import Seeder, seeded_run_id
from penny.tests.eval.utils.fixtures import SynthCollection
from penny.tests.eval.utils.transition_ledger import _seeded_response, _wire_tool_call
from penny.tools.micro_context import FramedParameter, LeafLabel, SkillLabels, SkillSignature

FAMILY = "lookups"

_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY = PennyConstants.MessageAuthor.PENNY
_COLLECTOR = PennyConstants.MessageAuthor.COLLECTOR


# ── Case 1: a routine taught but never run as a job ──────────────────────────
#
# Built through the SAME production pipeline ``WATCH_ROUTINE`` is (distillation over a
# fabricated demonstration, then the run-end labeller, then the framer's signature) —
# never hand-written steps — so its shape is a browse, then a `collection_write` whose
# destination is the placeholder a taught routine carries, exactly like a real one.
# Unlike `WATCH_ROUTINE`, it is seeded as a bare skill ROW and never applied to any
# collection: `chat_eval(seed_skills=...)` writes it straight into the registry.

HARBOUR_PAGE = "https://camberwick.example/harbour-flag"
HARBOUR_WATCH = "whether the storm pennant is flying"
HARBOUR_PAGE_DESC = "the camberwick harbour page to check each run"
HARBOUR_WATCH_DESC = "whether the storm pennant is flying"
HARBOUR_DESTINATION = "harbour-flag-checks"
HARBOUR_ENTRY_KEY = "Oct 12 check"
HARBOUR_ENTRY_VALUE = f"{HARBOUR_ENTRY_KEY} — the storm pennant was flying at midday."

# The two words the recipe can carry, each only through a leaf's LABEL — the literal
# demonstrated page and watch phrase never render for an unattached skill (every leaf of
# a distilled routine is a binding or a parameter; a parameter with nothing bound renders
# its own description, never the value it was taught with).  So the anchor lives in the
# description text itself, which is what both `find`'s recipe block and the ambient
# "needs:" line carry.
HARBOUR_PAGE_WORD = "camberwick"
HARBOUR_WATCH_WORD = "pennant"


class _Demonstration(NamedTuple):
    url: str
    extract: str
    collection: str
    entry_key: str
    entry_value: str


def _demonstrated_ledger(demo: _Demonstration) -> list[DistillInput]:
    """The two certified steps distillation reads, with their results carrying the
    real ``(<tool> result)`` frame — the same shape ``WATCH_ROUTINE``'s own
    demonstration uses, so the write's content binds to the browse's PAYLOAD."""
    return [
        DistillInput(
            source_ordinal=1,
            tool="browse",
            arguments={"queries": [demo.url], "extract": demo.extract},
            result=f"You opened {demo.url} (browse result)\n{demo.entry_value}",
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={
                "collection": demo.collection,
                "entries": [{"key": demo.entry_key, "content": demo.entry_value}],
            },
            result=(
                f"You saved an entry to {demo.collection}: "
                "(collection_write result)\nWrote 1 entry."
            ),
        ),
    ]


def _labels_by_value(steps: list[SkillStep], authored: dict[str, LeafLabel]) -> SkillLabels:
    """The labeller's draw, mapped from the demonstrated VALUES onto whichever
    arg-derived spot names distillation happened to mint — the same map-home shape
    ``WATCH_ROUTINE``'s own fixture uses, so a naming change inside distillation
    cannot leave this silently unmapped."""
    labels: dict[str, LeafLabel] = {}
    for step in steps:
        for substitution in step.substitutions:
            if substitution.parameter is None:
                continue
            node: Any = step.arguments
            for part in substitution.path:
                node = node[part]
            value = str(node)
            if value in authored:
                labels[substitution.parameter] = authored[value]
    assert len(labels) == len(authored), (
        f"every authored label must map home — matched {sorted(labels)} of {sorted(authored)}"
    )
    return SkillLabels(labels=labels)


def _build_harbour_routine() -> SkillDraft:
    """The one taught routine case 1 reads back: a browse of the harbour page, then a
    write whose destination is never created — so the skill is a bare registry row,
    never instantiated as any collection's job."""
    demo = _Demonstration(
        url=HARBOUR_PAGE,
        extract=HARBOUR_WATCH,
        collection=HARBOUR_DESTINATION,
        entry_key=HARBOUR_ENTRY_KEY,
        entry_value=HARBOUR_ENTRY_VALUE,
    )
    steps, parameters = distill_steps(_demonstrated_ledger(demo), frozenset({demo.collection}))
    authored = {
        demo.url: LeafLabel(name="page", description=HARBOUR_PAGE_DESC),
        demo.extract: LeafLabel(name="watched_for", description=HARBOUR_WATCH_DESC),
        demo.collection: LeafLabel(
            name="destination", description="the collection this is set up on"
        ),
    }
    steps, distilled = _apply_leaf_labels(steps, parameters, _labels_by_value(steps, authored))
    signature = SkillSignature(
        name="check_harbour_flag",
        description="Check a harbour's current flag status and keep what was found.",
        parameters=(
            FramedParameter(name="page", description=HARBOUR_PAGE_DESC, value=demo.url),
            FramedParameter(name="watched_for", description=HARBOUR_WATCH_DESC, value=demo.extract),
        ),
    )
    return SkillDraft(
        name=signature.name,
        intent=signature.description,
        description=signature.description,
        steps=steps,
        parameters=_interface_parameters(signature, distilled),
        source_run_id=seeded_run_id("taught-the-harbour-routine"),
    )


HARBOUR_ROUTINE = _build_harbour_routine()


# ── Case 2: an old passing mention, out of the conversation window ───────────
#
# >= MESSAGE_CONTEXT_LIMIT (20), so the salient turn is pushed out of BOTH the
# context window and the per-direction top-20 fetch, the same margin
# ``speak-logread-penny-messages-recall`` uses.
_FILLER_PAIRS = 24
_FILLER_USER = (
    "morning!",
    "how's your day going?",
    "thanks for that",
    "cool, makes sense",
    "ok noted",
    "sounds good",
    "haha nice",
    "appreciate it",
    "got it, thanks",
    "all good here",
    "talk to you later",
    "catch you soon",
)
_FILLER_PENNY = (
    "morning! good to hear from you",
    "going well, thanks for asking",
    "anytime, happy to help",
    "glad that makes sense",
    "sounds good to me",
    "you got it",
    "hah, right?",
    "of course",
    "no problem at all",
    "nice, catch you later",
    "take care!",
    "talk soon!",
)

SISTER_MENTION = "my sister lands thursday, she's on flight ZQ4471, i'll probably pick her up"
SISTER_REPLY = "nice, hope the pickup's easy! let me know if the timing changes"
SISTER_FLIGHT_WORD = "zq4471"


def seed_mention_out_of_window(mention: str, reply: str) -> Seeder:
    """Seed a passing mention FIRST (oldest) with a plain conversational reply beside
    it — neither saved anywhere — then push both out of context with filler pairs."""

    def seed(db: Database) -> None:
        db.messages.log_message(_INCOMING, TEST_SENDER, mention)
        db.messages.log_message(_OUTGOING, _PENNY, reply, recipient=TEST_SENDER)
        for index in range(_FILLER_PAIRS):
            db.messages.log_message(_INCOMING, TEST_SENDER, _FILLER_USER[index % len(_FILLER_USER)])
            db.messages.log_message(
                _OUTGOING, _PENNY, _FILLER_PENNY[index % len(_FILLER_PENNY)], recipient=TEST_SENDER
            )

    return seed


def assert_seeded_turns_embedded(db: Database, limit: int = 500) -> None:
    """Every seeded message carries a vector before the turn runs — the seeder writes
    the messages and the RUNNER embeds them, so a database nobody ran this through
    would fail at the fixture rather than read as the model's own miss."""
    unvectored = db.messages.messages_without_embeddings(limit=limit)
    assert not unvectored, (
        f"every seeded message must carry an embedding — {len(unvectored)} without one, "
        f"first: {unvectored[0].content!r}"
    )


# ── Case 3: thirty lists, each named one distinctive word ────────────────────

LIST_WORDS = (
    "acorn",
    "bramble",
    "cobalt",
    "driftwood",
    "ember",
    "fennel",
    "granite",
    "heather",
    "indigo",
    "jasper",
    "kelp",
    "lantern",
    "marigold",
    "nettle",
    "obsidian",
    "pewter",
    "quartz",
    "ragwort",
    "saffron",
    "thistle",
    "umber",
    "violet",
    "walnut",
    "xylograph",
    "yarrow",
    "zinnia",
    "basil",
    "clover",
    "dahlia",
    "elm",
)
assert len(LIST_WORDS) == 30
assert len(set(LIST_WORDS)) == 30

MANY_LISTS: tuple[SynthCollection, ...] = tuple(
    SynthCollection(
        word, f"Notes about {word}.", entries=(f"{word} note — a short note about {word}.",)
    )
    for word in LIST_WORDS
)


# ── Case 4: an archived list, with its entries still in it ───────────────────

PRESERVE_BOX = SynthCollection(
    "preserve-box",
    "Home preserves and jam recipes: fruit, sweetener, cook time.",
    entries=(
        "Quince paste — quince, sugar, lemon, 40 min.",
        "Fig conserve — figs, honey, walnuts, 25 min.",
    ),
)
PRESERVE_WORDS = ("quince", "fig")


def archive_after_seeding(synth: SynthCollection) -> Seeder:
    """Archive a collection the WORLD already seeded — the entry condition case 4
    reads back: a retired list that still holds what it held.

    Takes no part in laying the collection down: the case's ``World.stores`` carries
    it, so ``chat_eval``'s own ``seed_world_stores`` has already created it (with its
    entries) by the time this seeder runs. Seeding it a second time here would create
    a duplicate collection of the same name and fail every sample identically."""

    def seed(db: Database) -> None:
        db.memories.archive(synth.name, run_id=seeded_run_id(f"archive-{synth.name}"))

    return seed


# ── Case 5: one long list, its matches among the oldest entries ──────────────

RECIPE_LIST = "recipe-notes"
LIME_ENTRIES = (
    "Grilled corn salad — corn, lime, cilantro, 15 min.",
    "Key lime bars — graham crust, lime curd, 1 hr.",
)
LIME_WORDS = ("corn", "bars")
_FILLER_DISHES = (
    "Roast chicken",
    "Baked salmon",
    "Lentil stew",
    "Beef chili",
    "Tomato soup",
    "Mushroom risotto",
    "Pumpkin bread",
    "Carrot muffins",
    "Spinach quiche",
    "Barley salad",
    "Potato gratin",
    "Onion tart",
    "Squash curry",
    "Apple crumble",
    "Pear galette",
    "Cherry clafoutis",
    "Peach cobbler",
    "Plum cake",
    "Fennel gratin",
    "Celery soup",
    "Leek pie",
    "Cabbage rolls",
    "Turnip mash",
    "Beet salad",
    "Radish pickle",
    "Parsnip soup",
    "Zucchini bread",
    "Eggplant bake",
    "Artichoke dip",
    "Asparagus tart",
    "Broccoli bake",
    "Cauliflower curry",
    "Kale salad",
    "Chard gratin",
    "Pepper stew",
    "Okra gumbo",
    "Yam mash",
)


def _filler_recipe(index: int, dish: str) -> str:
    return f"{dish} — recipe {index}, ingredient list, {20 + index} min."


def seed_long_recipe_list(db: Database) -> None:
    """The two LIME recipes written FIRST (oldest), then ~37 filler recipes after —
    so recency alone cannot answer the ask; the matches are not among the newest."""
    db.memories.create_collection(
        RECIPE_LIST,
        "Saved recipes: dishes, ingredients, cook times.",
        created_by_run_id=seeded_run_id(RECIPE_LIST),
    )
    memory = require_memory(db, RECIPE_LIST)
    memory.write(
        [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in LIME_ENTRIES],
        author="user",
        run_id=seeded_run_id(RECIPE_LIST),
    )
    filler = [_filler_recipe(index, dish) for index, dish in enumerate(_FILLER_DISHES)]
    memory.write(
        [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in filler],
        author="user",
        run_id=seeded_run_id(f"{RECIPE_LIST}-filler"),
    )


# ── Case 6: a standing job's own run history ──────────────────────────────────
#
# Reuses the public WATCH_ROUTINE with the SAME demonstrated values the house's
# typewriter watch uses (read off its own parameters, never a private constant), per
# the code owner's amendment, so this is the same watch rather than a lookalike.


def _demonstrated_value(name: str) -> str:
    """The value WATCH_ROUTINE's own framing demonstrated a named parameter with —
    never None in practice (the routine was minted with a real demonstration), but
    the field is optional on the type, so this is where that is asserted once."""
    value = next(p.value for p in WATCH_ROUTINE.parameters if p.name == name)
    assert value is not None, f"WATCH_ROUTINE carries no demonstrated value for {name!r}"
    return value


_PAGE_VALUE = _demonstrated_value("page")
_WATCHED_VALUE = _demonstrated_value("watched_for")

TYPEWRITER_WATCH = StandingJob(
    routine=WATCH_ROUTINE,
    values={"page": _PAGE_VALUE, "watched_for": _WATCHED_VALUE},
    description="Portable typewriters newly listed at the Quill Market shop.",
    schedule="FREQ=DAILY;BYHOUR=7",
    notify=True,
)

EARLIER_FIND = "Imperial Good Companion — 1952 portable, needs new ribbon, $95."
LATEST_FIND = "Olympia SM3 — 1958 portable, excellent condition, $260."
LATEST_FIND_WORD = "olympia"


class _RunSeed(NamedTuple):
    run_name: str
    entry_key: str
    entry_value: str
    days_ago: int


_EARLIER_RUN = _RunSeed("typewriter-watch-earlier", "Imperial Good Companion", EARLIER_FIND, 3)
_LATEST_RUN = _RunSeed("typewriter-watch-latest", "Olympia SM3", LATEST_FIND, 1)


def seed_job_with_run_history(db: Database) -> None:
    """Stand the job up, then two completed cycles behind it: an earlier find and a
    later one, each cited by its own run.  Every run's prompt rows are stamped to a
    FIXED PAST instant rather than the moment the sample seeds them — a run seeded
    the ordinary way would land at the same instant the turn starts, which is not
    "last time" for anything.  The backdating is a raw update because no public
    store method takes an explicit timestamp for a logged run."""
    job = TYPEWRITER_WATCH
    stood_up = seeded_run_id("stood-up-typewriter-watch")
    db.memories.create_collection(job.container, job.description, created_by_run_id=stood_up)
    db.memories.update_collection_metadata(
        job.container,
        extraction_prompt=job.program,
        schedule=job.schedule,
        replace_schedule=True,
        notify=job.notify,
        skill_name=slug_skill_name(job.routine.name),
        skill_params=job.values,
        run_id=stood_up,
    )
    for run_seed in (_EARLIER_RUN, _LATEST_RUN):
        _seed_one_run(db, job.container, run_seed)


def _seed_one_run(db: Database, container: str, run_seed: _RunSeed) -> None:
    run_id = seeded_run_id(run_seed.run_name)
    page, watched_for = TYPEWRITER_WATCH.values["page"], TYPEWRITER_WATCH.values["watched_for"]
    steps = [
        DistillInput(
            source_ordinal=1,
            tool="browse",
            arguments={"queries": [page], "extract": watched_for},
            result=f"You opened {page} (browse result)\n{watched_for}: {run_seed.entry_value}",
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={
                "collection": container,
                "entries": [{"key": run_seed.entry_key, "content": run_seed.entry_value}],
            },
            result=f"You saved an entry to {container}: (collection_write result)\nWrote 1 entry.",
        ),
    ]
    calls = [_wire_tool_call(f"{run_id}-{index}", step) for index, step in enumerate(steps)]
    db.messages.log_prompt(
        model="seed",
        messages=[],
        response=_seeded_response(tool_calls=calls),
        agent_name=_COLLECTOR,
        run_id=run_id,
        run_target=container,
    )
    db.messages.set_run_outcome(run_id, RunOutcome.WORKED.value, f"Recorded {run_seed.entry_key}.")
    require_memory(db, container).write(
        [EntryInput(key=run_seed.entry_key, content=run_seed.entry_value)],
        author=_COLLECTOR,
        run_id=run_id,
    )
    _backdate_run(db, run_id, run_seed.days_ago)


def _backdate_run(db: Database, run_id: str, days_ago: int) -> None:
    """Push one run's prompt rows (and the entry it wrote) back by ``days_ago`` —
    true of the seeded world, not of anything the model did.

    Written through the models, so the stamp lands in the form every other row's
    has: a stamp bound as ISO text keeps its offset, and the state header cannot
    order such a row against the rest."""
    stamp = db.clock.now() - timedelta(days=days_ago)
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
