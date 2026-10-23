"""The ``collection_write`` headline is honest about a merge (#2758).

The contract is `docs/eval-case-design.md`.

**ONE case**, ``rules-write-merge-is-honest``.  The ``rules`` collection already holds one
rule; the user asks for a new rule that restates the same policy in different words AND adds
a detail the stored content doesn't carry, with no key named — the entry-identity ladder
(#2715) decides whether the write is the same specific thing as the stored one and, since the
restatement adds something, a settled "same" MERGES rather than landing as an unchanged
duplicate.  When it merges, the store updates the EXISTING entry under its OWN key rather
than creating a second row, and the write's own result (`CollectionWriteTool.to_result_narration`,
fixed on this branch by #2758) tells the model so by naming the key that was actually
updated — never the key the model passed, which the write never created.

**The failure this reproduces, and where it actually shows (read from this case's own live
runs, not assumed).**  Before #2758 the TOOL RESULT's headline named the write's ARGUMENTS
(the key the model passed), not what the write did: on `main`, a cross-key merge's result
opens "You saved \"<the key she typed>\" to `rules`:" even though nothing under that key was
ever created — verbatim in this case's own `main` transcripts, e.g. `You saved "no tables,
use bullet points" to \\`rules\\`:` for a write the store actually landed under the existing
`no_markdown_tables`.  On this branch the same merge's result opens "You updated the existing
`no_markdown_tables` in `rules` — it matched what you wrote as the same thing:" — the exact
headline strings are pinned, whole-string, by the unit tests in
`penny/tests/tools/test_memory_tools.py`.

**What this case asserts, and what it tried and dropped.**  A STORE claim: `rules` ends
holding exactly ONE entry for the policy — the merge survives singularly, never a second row
for the same thing under a different name.  A sample where the ladder did NOT merge (wrote
the paraphrase as a genuinely new, second entry) fails this openly, which is the honest
reading: the claim's pass rate is itself how often the pair merged, never forced.

A reply-side PROVENANCE claim — every backtick- or quote-wrapped, key-shaped token the reply
names is a key `rules` actually holds — was tried and DROPPED after three live runs each
surfaced a different false-positive: ordinary contraction apostrophes read as a quote pair
("I'll ... I've"), a quoted restatement of the rule's own CONTENT (a full sentence, not a
key), and a casual scare-quoted nickname for the policy ("that \"no tables\" policy") that
names nothing at all.  Narrowing the heuristic further (penalising specific words or phrasing)
would be the "phrase lexicon" `docs/principles.md` §4.7 rules out.  The deeper reason it never
stabilised: gemma's reply style, on this ask, consistently paraphrases the tool result ("I
updated that rule...") rather than quoting the key the headline names — correct behaviour the
#2758 fix doesn't need reply-text policing to prove, since the mechanism is pinned exactly by
the unit tests and the raw tool-result text is readable verbatim in this case's own
transcripts (`docs/principles.md` §4.7's "a person reading one representative sample").  So
this case keeps only the claim that is both well-formed and falsifiable, and leaves the
headline's exact wording to the unit tests.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code owner's to
accept once the numbers have been read.  The rule's own subject (table formatting in replies)
is invented and unrelated to any real conversation, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.agents.self_state import standing_rules_instruction
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    Preparer,
    seeded_run_id,
)
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

_FAMILY = "collection-write-merge-headline"

RULES = PennyConstants.MEMORY_RULES_COLLECTION

_USER_NAME = "Marguerite"
_USER_LOCATION = "Halifax, NS"
_USER_TIMEZONE = "America/Halifax"

# The stored rule and the user's restatement of the SAME policy, worded differently AND
# carrying one NEW detail the stored content doesn't — the shape that reaches the identity
# ladder's VALUE question as "new_value" rather than "same_value" (#2715): a near-identical
# restatement reads as the SAME VALUE, which merges silently (``KEY_EXISTS_UNCHANGED`` /
# ``DUPLICATE_UNCHANGED``) and never exercises this case's headline at all — MEASURED in this
# case's own first run, where every sample's restatement was close enough to read as nothing
# new, and the single-entry STORE claim passed vacuously on an unchanged write every time.  A
# genuinely added qualifier (which surface the rule covers) is still the same specific rule —
# the identity question is still "yes" — so it reaches the merge path this case exists to
# narrate.  Invented, about table formatting, unrelated to any real conversation.
_SEEDED_KEY = "no_markdown_tables"
_SEEDED_CONTENT = "Never format a reply as a markdown table — turn it into a bulleted list instead."

ASK = (
    "hey, new rule for you: whenever you're about to answer me, don't lay it out as a "
    "table — turn it into bullet points instead, and that goes for your job reports too, "
    "not just chat replies"
)
ALSO_PHRASED: tuple[str, ...] = (
    "quick rule update — never answer me with a table, always break it into bullets "
    "instead, even for long comparison lists",
    "can you remember this as a rule: stop using tables in your replies, use bullet points "
    "instead, same goes for anything with more than a couple of rows",
)
_WORDINGS = (ASK, *ALSO_PHRASED)


def _assert_wordings_restate_the_policy_and_add_something_new() -> None:
    """ENFORCED at import: every wording restates the stored policy (tables, bullets) in its
    own words, adds a qualifier the stored content doesn't carry (so the pair reads as the
    same rule with a NEW value, not a duplicate), and never leaks the seeded key or its exact
    sentence — the ladder's identity question is answered from the CONTENT, never from a
    copied key or phrase."""
    seeded_first_words = " ".join(_SEEDED_CONTENT.split()[:4]).casefold()
    for wording in _WORDINGS:
        folded = wording.casefold()
        assert "table" in folded, f"a wording must name the table rule: {wording!r}"
        assert "bullet" in folded, f"a wording must name the bullet alternative: {wording!r}"
        assert _SEEDED_KEY not in folded, f"a wording must not leak the seeded key: {wording!r}"
        assert seeded_first_words not in folded, (
            f"a wording must not borrow the seeded rule's own words: {wording!r}"
        )
        assert "too" in folded or "even" in folded or "same goes" in folded, (
            f"a wording must add a qualifier the stored content lacks: {wording!r}"
        )


_assert_wordings_restate_the_policy_and_add_something_new()


def _seed_world(db: Database) -> None:
    """The user, and the one rule already standing."""
    db.users.save_info(
        sender=TEST_SENDER,
        name=_USER_NAME,
        location=_USER_LOCATION,
        timezone=_USER_TIMEZONE,
        date_of_birth="1990-01-01",
    )
    rule_run = seeded_run_id("seeded-table-rule")
    require_memory(db, RULES).write_rule_decided(
        [EntryInput(key=_SEEDED_KEY, content=_SEEDED_CONTENT)],
        author="chat",
        run_id=rule_run,
    )


def _premise() -> Preparer:
    """The seeded rule renders where standing rules render, and nothing else is in `rules`
    yet — asserted before the turn runs, so a later miss is the turn's, not the seed's."""

    def probe(penny: Penny) -> None:
        db = penny.db
        rendered = standing_rules_instruction(db) or ""
        assert _SEEDED_CONTENT in rendered, f"the seeded rule must render:\n{rendered!r}"
        held = require_memory(db, RULES).read_all()
        assert [e.key for e in held] == [_SEEDED_KEY], f"`{RULES}` must start with one rule"

    return probe


# ── The claims ────────────────────────────────────────────────────────────────


def _rules_entries(sample: SampleObservation) -> list[tuple[str | None, str]]:
    return [(entry.key, entry.content) for entry in sample.held if entry.collection == RULES]


def _rules_holds_one_entry_for_the_policy(sample: SampleObservation, _world: World) -> Answer:
    """STORE: the merge survives singularly — `rules` ends holding exactly one entry, never
    a second row for the same policy under a different name.  A sample where the ladder did
    NOT merge (wrote the paraphrase as a genuinely new, second entry) fails this openly,
    which is the honest reading: the claim's pass rate is itself how often the pair merged."""
    held = _rules_entries(sample)
    return len(held) == 1, f"`{RULES}` holds {held!r}"


_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_merge_headline_names_a_real_key(chat_eval: ChatEval, model: str) -> None:
    """A rule that restates a stored one, and the merge it can produce: the policy survives
    as one entry — the headline's own exact wording is pinned by the unit tests, and the raw
    tool-result text is readable verbatim in this case's transcripts."""
    cohort = await chat_eval(
        case_id="rules-write-merge-is-honest",
        behaviour=(
            "In the chat agent, when a new rule matches an existing one as the same thing, "
            "Penny's write merges onto the existing entry and the policy survives as one "
            "entry, never a second row under a different key."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed_world,
        prepare=_premise(),
        world=World(name="rules-write-merge-is-honest", pages=(), keeps=(), excludes=()),
        ask=ASK,
        also_phrased=ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=120.0,
    )
    cohort.claim(
        "state: `rules` holds exactly one entry for the policy",
        _rules_holds_one_entry_for_the_policy,
        SpecCategory.STORE,
    )
    cohort.measure(*_MEASURED)
