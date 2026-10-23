"""The ``collection_write`` headline is honest about a merge (#2758).

The contract is `docs/eval-case-design.md`.

**ONE case**, ``rules-write-merge-is-honest``.  The ``rules`` collection already holds one
rule; the user asks for a new rule that restates the same policy in different words, with no
key named — the entry-identity ladder (#2715) decides whether the write is the same specific
thing as the stored one.  When it is, the store merges the new content onto the EXISTING
entry under its OWN key rather than creating a second row, and the write's own result
(``CollectionWriteTool.to_result_narration``, fixed on this branch by #2758) tells the model so
by naming the key that was actually updated — never the key the model passed, which the write
never created.

**The failure this reproduces.**  Before #2758 the headline named the write's ARGUMENTS (the
key the model passed), not what the write did — so a merge recapped as "I saved a new rule
called `<the key she typed>`", a key `rules` never held under that name.  On `main` this case
is expected to reproduce that: a merged sample's reply names a key the store does not hold.
On this branch the headline names the key the write actually updated, so the same reply
claim passes.

**What is asserted.  Facts only.**
- STORE: `rules` ends holding exactly ONE entry for the policy — the merge survives
  singularly, never a second row for the same thing under a different name.
- PROVENANCE (reply): every backtick- or quote-wrapped, key-shaped token the reply names is a
  key `rules` actually holds when the turn ends.  This is the same family as "every specific
  value in the reply is sourced" (`assert_every_value_in_the_reply_is_sourced`) — a
  deterministic fact check, not a reply-wording claim: a reply that names no key-shaped token
  at all passes vacuously (nothing to be wrong about), and a reply that merely talks about the
  rule without quoting a key does too.  What it catches is the production shape: a quoted or
  backticked key that names nothing the store holds.

**What is NOT claimed.**  Whether the ladder actually merges this pair is model behaviour
(the decision model's call, #2715's own territory) — measured, never forced.  A sample that
writes the paraphrase under a brand-new key (no merge) passes both claims honestly: one
entry... no, TWO entries for two different keys, which fails the single-entry STORE claim
openly rather than silently, and the reply then names a key the store really does hold (its
own new one), which passes the PROVENANCE claim too.  So the STORE claim's pass rate is
itself a read of how often the ladder merged the pair; the PROVENANCE claim is this case's
own contract and is what should move between `main` and the branch.  Which tool she calls and
how she words the reply are variance, measured in the tool sequence and the reply spread.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code owner's to
accept once the numbers have been read.  The rule's own subject (table formatting in replies)
is invented and unrelated to any real conversation, because the repo is public.
"""

from __future__ import annotations

import re

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

# A key-shaped token the reply wraps in backticks or double quotes — the same wrapping
# `CollectionWriteTool`'s own narration uses for a key (`_keys_label` / `_merge_clause` in
# `penny/tools/memory_tools.py`), which a model recapping the write's result line carries
# over.  Excludes the collection's own name, which the headline also backtick-quotes.
#
# Single quotes are deliberately NOT a delimiter here (MEASURED): casual prose is full of
# apostrophes ("I'll", "I've", "didn't"), and a naive single-quote pair regex treats the
# apostrophe in one contraction as an OPENING quote and the apostrophe in a later, unrelated
# contraction as its CLOSE — capturing the entire run of prose between them as one "named
# key".  Every reply in this case's first measured run got flagged this way, none of them a
# real quoted key.  Backticks and double quotes carry no such risk (a reply has no reason to
# use either mid-word), so they stay.
_NAMED_TOKEN = re.compile(r"`([^`]+)`|\"([^\"]+)\"")

# A key is a short label, not a sentence: this case's own keys are one to three words
# (``no_markdown_tables``), and the issue's own motivating example is the same shape
# (``rule: always-cite``).  A reply's quoted RULE CONTENT is wrapped the same way (the model
# often repeats what a rule says back in quotes) and reads as many words ending in sentence
# punctuation — MEASURED in this case's own second run, where a quoted restatement of the
# rule's full sentence was flagged as a "named key" the store doesn't hold under that exact
# wording, which is true but not the bug this case exists to catch.  A token this shaped is
# content, not a key, and is excluded rather than counted.
_MAX_KEY_WORDS = 6
_SENTENCE_ENDINGS = (".", "!", "?")


def _reads_as_a_key(token: str) -> bool:
    if token.endswith(_SENTENCE_ENDINGS):
        return False
    return len(token.split()) <= _MAX_KEY_WORDS


def _reply_named_keys(reply: str) -> list[str]:
    """Every backtick- or quote-wrapped, key-shaped token in the reply, excluding the
    collection's own name and anything that reads as quoted sentence content rather than a
    key."""
    names = []
    for match in _NAMED_TOKEN.finditer(reply):
        token = next(group for group in match.groups() if group is not None).strip()
        if token and token.casefold() != RULES.casefold() and _reads_as_a_key(token):
            names.append(token)
    return names


def _rules_entries(sample: SampleObservation) -> list[tuple[str | None, str]]:
    return [(entry.key, entry.content) for entry in sample.held if entry.collection == RULES]


def _rules_holds_one_entry_for_the_policy(sample: SampleObservation, _world: World) -> Answer:
    """STORE: the merge survives singularly — `rules` ends holding exactly one entry, never
    a second row for the same policy under a different name.  A sample where the ladder did
    NOT merge (wrote the paraphrase as a genuinely new, second entry) fails this openly,
    which is the honest reading: the claim's pass rate is itself how often the pair merged."""
    held = _rules_entries(sample)
    return len(held) == 1, f"`{RULES}` holds {held!r}"


def _every_named_key_exists_in_the_store(sample: SampleObservation, _world: World) -> Answer:
    """PROVENANCE (reply): every key-shaped token the reply names is a key `rules` actually
    holds when the turn ends — the same family as "every specific value in the reply is
    sourced", read as a fact rather than wording.  The production bug this reproduces: a
    merge recapped by naming the key the model TYPED, which the write redirected elsewhere,
    so the reply named a key the store never held under that name."""
    held_keys = {key for key, _content in _rules_entries(sample) if key}
    named = _reply_named_keys(sample.reply)
    missing = [key for key in named if key not in held_keys]
    return not missing, f"the reply names {missing!r}, which `{RULES}` does not hold"


_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_merge_headline_names_a_real_key(chat_eval: ChatEval, model: str) -> None:
    """A rule that restates a stored one, and the merge it can produce: the reply never
    names a key `rules` doesn't hold, and the policy survives as one entry."""
    cohort = await chat_eval(
        case_id="rules-write-merge-is-honest",
        behaviour=(
            "In the chat agent, when a new rule matches an existing one as the same thing, "
            "Penny's reply never names a key `rules` does not hold, and the policy survives "
            "as one entry."
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
    cohort.claim(
        "reply: every key it names in `rules` is one the store holds",
        _every_named_key_exists_in_the_store,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.measure(*_MEASURED)
