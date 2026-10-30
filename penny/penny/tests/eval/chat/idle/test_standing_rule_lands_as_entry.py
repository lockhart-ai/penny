"""Confirming a drafted standing rule writes an entry, never a rewrite of the
collection's prompt (#2857).

The contract is `docs/eval-case-design.md`.

**The failure this reproduces.** Penny drafted a new standing rule, the user confirmed
it, and she read `rules` back with `memory_get`, found a STRAY prompt an earlier bad
turn had already written onto the collection, and merged the new rule into that prompt
with `collection_configure` instead of writing an entry. `rules` has no schedule, so
nothing ever reads that prompt — standing rules render from the collection's ENTRIES —
and her reply claimed the rule was now permanent when it wasn't.

**ONE case**, `standing-rule-lands-as-entry`. `rules` starts with four entries and the
same stray prompt the failure needs: a prose paragraph that merely lists those four
entries, written by an earlier (also wrong) turn, with no schedule on the collection.
One prior exchange has Penny drafting a new rule — in her own voice, proposing it,
naming a distinctive detail nothing else in the world repeats — and the live turn is
just the user's confirmation, with the rule's own content sitting in that one prior
exchange rather than in the live ask. This is the same two-turn shape #2758 and
#2630 both reproduced: a confirmation turn, not a turn that has to re-derive content
from scratch.

**What this case asserts. Two claims, both facts, both survival.**

- `rules` holds an entry whose content carries the drafted rule's subject (the words
  "basil" and "sunday", nothing else in the world mentions either) — the new rule
  landed as an ENTRY, not folded into the prompt.
- The four seeded entries survive with their content unchanged — the fix must not
  touch what was already there, including the stray prompt's own listing of them
  (its removal, if any, is a separate `prompt`-clearing move this case does not ask
  for and does not claim).

What is NOT claimed: which tool she calls, how many turns she takes, whether she
touches the stray `prompt` at all, or how she words the reply — variance, measured in
the tool sequence and the reply spread.

**What the case-only run on main showed.** In this particular world — a short
conversation where the rule's own content sits one exchange back and the live turn is
just a confirmation — gemma wrote the new rule as an entry via `collection_write` in
all 15 samples and never touched the stray prompt in any of them: this specific shape
does not reproduce the `memory_get`-driven prompt rewrite the issue describes, because
the model never called `memory_get` here at all. The case stays in the suite as a
durable, passing contract for the behaviour it does state (a confirmed drafted rule
lands as an entry) and as the regression guard for the render/refusal change; the
code owner may want a second case that forces a `memory_get` read before the
confirmation to exercise the exact mechanism the issue reported.

REPORT-ONLY (`min_pass_rate=None`). Every name and rule here is invented and unrelated
to any real conversation, because the repo is public.
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

_FAMILY = "standing-rule-lands-as-entry"

RULES = PennyConstants.MEMORY_RULES_COLLECTION

_USER_NAME = "Priya"
_USER_LOCATION = "Winnipeg, MB"
_USER_TIMEZONE = "America/Winnipeg"

# Four standing rules already in force — the content this case must see survive
# byte-for-byte, whatever the live turn does.
_SEEDED_RULES: tuple[tuple[str, str], ...] = (
    (
        "shoes_off_at_door",
        "Always take your shoes off at the door before walking through the house.",
    ),
    ("recycling_out_tuesday", "Put the recycling bin out every Tuesday night."),
    ("dishwasher_before_bed", "Start the dishwasher before going to bed if it's full."),
    ("no_phones_at_dinner", "No phones at the dinner table."),
)

# The stray prompt a prior (also wrong) turn left on `rules` — the exact shape the
# production failure needs: a prose paragraph that merely lists the four entries
# above, on a collection with no schedule, so nothing ever runs it.
_STRAY_PROMPT = (
    "Standing rules to always follow: 1. Shoes off at the door. 2. Recycling out "
    "every Tuesday night. 3. Start the dishwasher before bed if it's full. 4. No "
    "phones at the dinner table."
)

# Penny's own drafted rule, in the one prior exchange — "basil" and "sunday" are
# distinctive enough that only a model that actually carried this rule's content into
# a write could produce them; nothing else in the world mentions either. The claim
# matches on these two words rather than the drafted sentence verbatim (#2857, measured
# on the case-only run): gemma reliably writes the rule's SUBJECT as a new entry but
# freely paraphrases the sentence itself — dropping or reshaping "the quiet-hour
# watering" descriptor every time — so an exact-phrase match was asserting wording the
# model was never asked to preserve, not a fact about what landed.
_USER_PROMPT_FOR_DRAFT = (
    "ugh, the basil on the windowsill died again, I keep forgetting to water it"
)
_PENNY_DRAFTED_RULE = (
    "that's the third time this month — want me to make it a standing rule that you "
    "water the windowsill basil every Sunday morning, the quiet-hour watering? I can "
    "add it to your rules if you like."
)
_PLANTED_SUBJECT_WORDS = ("basil", "sunday")

ASK = "yes, write that as a rule"
ALSO_PHRASED: tuple[str, ...] = (
    "yep, make it a standing rule please",
    "yes please, add that as a rule",
)


def _assert_the_planted_subject_is_distinctive() -> None:
    """ENFORCED at import: the rule's subject lives only in Penny's drafted rule, and
    no live ask carries any of the rule's own content — the point of the two-turn
    shape."""
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
        assert "water" not in wording.casefold(), (
            f"a live ask must carry no rule content of its own: {wording!r}"
        )


_assert_the_planted_subject_is_distinctive()


def _seed_world(db: Database) -> None:
    """The user, four standing rules already in force, the stray prompt the
    reproduction needs, and the one prior exchange where Penny drafted the new
    rule."""
    db.users.save_info(
        sender=TEST_SENDER,
        name=_USER_NAME,
        location=_USER_LOCATION,
        timezone=_USER_TIMEZONE,
        date_of_birth="1990-01-01",
    )
    rules_run = seeded_run_id("seeded-rules")
    require_memory(db, RULES).write_rule_decided(
        [EntryInput(key=key, content=content) for key, content in _SEEDED_RULES],
        author="chat",
        run_id=rules_run,
    )
    db.memories.update_collection_metadata(RULES, extraction_prompt=_STRAY_PROMPT)
    draft_ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, _USER_PROMPT_FOR_DRAFT
    )
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        _PENNY_DRAFTED_RULE,
        parent_id=draft_ask_id,
    )


def _premise() -> Preparer:
    """`rules` starts with exactly the four seeded entries and the stray prompt —
    asserted before the live turn runs, so a later miss is the turn's, not the
    seed's."""

    def probe(penny: Penny) -> None:
        db = penny.db
        held = require_memory(db, RULES).read_all()
        assert {e.key for e in held} == {key for key, _content in _SEEDED_RULES}, (
            f"`{RULES}` must start with the four seeded rules, it holds {[e.key for e in held]!r}"
        )
        row = db.memories.get(RULES)
        assert row is not None
        assert row.extraction_prompt == _STRAY_PROMPT
        assert row.schedule is None

    return probe


# ── The claims ────────────────────────────────────────────────────────────────


def _rules_contents(sample: SampleObservation) -> list[str]:
    return [entry.content for entry in sample.held if entry.collection == RULES]


def _new_rule_lands_with_the_planted_subject(sample: SampleObservation, _world: World) -> Answer:
    """STORE: an entry in `rules` carries the drafted rule's subject (basil, Sunday) —
    it landed as an entry, not folded into the collection's prompt. Matched by subject
    rather than the drafted sentence's exact wording, which the model paraphrases
    freely (measured on the case-only run)."""
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


_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_confirmed_drafted_rule_lands_as_an_entry(chat_eval: ChatEval, model: str) -> None:
    """A confirmation turn whose new rule's content sits in the prior exchange: the
    new rule survives as an entry in `rules`, and the four already-standing rules
    survive untouched."""
    cohort = await chat_eval(
        case_id="standing-rule-lands-as-entry",
        behaviour=(
            "In the chat agent, when the user confirms a drafted rule as a standing "
            "rule, Penny writes it as an entry in `rules`."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed_world,
        prepare=_premise(),
        world=World(name="standing-rule-lands-as-entry", pages=(), keeps=(), excludes=()),
        ask=ASK,
        also_phrased=ALSO_PHRASED,
        samples_per_phrasing=5,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=120.0,
    )
    cohort.claim(
        "state: an entry in `rules` carries the drafted rule's subject (basil, Sunday)",
        _new_rule_lands_with_the_planted_subject,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the four seeded rules survive, content unchanged",
        _the_four_seeded_rules_survive,
        SpecCategory.STORE,
    )
    cohort.measure(*_MEASURED)
