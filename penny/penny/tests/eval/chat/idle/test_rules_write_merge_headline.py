"""The ``collection_write`` headline is honest about a merge (#2758).

The contract is `docs/eval-case-design.md`.

**ONE case**, ``rules-write-merge-is-honest``.  ``rules`` already holds two rules (one of them
the look-alike this case is about); several turns of unrelated small talk sit before the live
one; the user states a new rule in one turn, Penny asks whether they mean a STANDING rule, and
the live turn — the one this case drives — is just the user's confirmation, with the rule's
own content sitting in history rather than in the live ask.  That shape is deliberate: the
production failure this reproduces landed on a wrap-up step, deep in a long conversation, taken
without thinking — a confirmation turn is exactly that shape, and it is the shape #2758's own
fix targets (the model recapping a tool result it read without re-deriving it from scratch).

The user's stated rule restates an existing one's POLICY in different words AND adds a detail
the stored content doesn't carry — the shape that reaches the identity ladder's (#2715) VALUE
question as "new_value" rather than "same_value": a near-identical restatement reads as the
SAME VALUE, which settles as an unchanged duplicate and never reaches the merge path this case
exists to narrate — MEASURED in this case's own early runs, where a too-close restatement left
the write a no-op every time.  When the ladder settles "same" with something new added, the
store MERGES the new content onto the EXISTING entry under its OWN key rather than creating a
second row, and the write's own result (`CollectionWriteTool.to_result_narration`, fixed on
this branch by #2758) tells the model so by naming the key that was actually updated — never
the key the model passed, which the write never created.

**What this case asserts.  Two claims, both facts.**

- STORE: `rules` ends holding the untouched OTHER rule plus exactly ONE entry for the table
  policy — the merge survives singularly, never a second row for the same thing under a
  different name.  A sample where the ladder did NOT merge (wrote the restatement as a
  genuinely new, second entry) fails this openly: the claim's pass rate is itself how often
  the pair merged, never forced.
- EXACT KEY, no extraction: for every `key` this turn's `collection_write` calls PASSED that
  `rules` does NOT hold under that name at the end (because it merged into a different entry),
  the reply does not contain that exact string.  This is the strict version of the production
  failure — "I saved a new rule called `rule: research_and_cite`" when no such entry ever
  existed — read off the write's own argument (`SampleObservation.collection_write_keys`,
  read straight off the turn's prompt log, #2758), compared by exact substring, never
  tokenised and never a lexicon.  (An earlier version of this claim tried to spot a "named
  key" by pattern-matching the REPLY alone — backtick/quote-wrapped tokens — and that
  heuristic never stabilised: contraction apostrophes read as a quote pair, a quoted
  restatement of the rule's own content read as a "key", and a casual scare-quoted nickname
  for the policy named nothing at all.  Comparing against a KNOWN string the write itself
  passed has none of those failure modes.)

What is NOT claimed: which tool she calls, how many turns she takes to get there, or how she
words the reply beyond the one exact-string fact above — variance, measured in the tool
sequence and the reply spread.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code owner's to
accept once the numbers have been read.  Every rule's subject (table formatting, citing
sources) and the filler conversation (comparing camera prime lenses) are invented and
unrelated to any real conversation, because the repo is public.
"""

from __future__ import annotations

from datetime import datetime

import pytest

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

# The look-alike rule this case is about, and an unrelated second rule that must survive
# untouched — a write touching the wrong row would hide inside a single-entry count.
_SEEDED_KEY = "no_markdown_tables"
_SEEDED_CONTENT = "Never format a reply as a markdown table — turn it into a bulleted list instead."
_OTHER_KEY = "always_cite_sources"
_OTHER_CONTENT = "Whenever you search the web for something, include a source link in the reply."

# Filler small talk — comparing camera prime lenses, unrelated to either rule — pushed out of
# the chat history window by MESSAGE_CONTEXT_LIMIT below, so it is read as "a long conversation
# already happened" without actually being visible to the live turn.
_FILLER_PAIRS: tuple[tuple[str, str], ...] = (
    (
        "trying to decide between two prime lenses for my camera, any thoughts on 35mm vs "
        "50mm for portraits?",
        "the 50mm gives a tighter, more flattering crop for portraits; the 35mm reads more "
        "environmental",
    ),
    (
        "yeah, what about low light, is one noticeably faster?",
        "depends on the exact lens, but most 50mm primes open wider than their 35mm siblings "
        "in the same line",
    ),
    (
        "my budget's pretty tight, is it worth going third-party over the brand name glass?",
        "third-party primes from sigma or tamron usually match the sharpness for half the price",
    ),
    (
        "good to know, does the autofocus hold up on the cheaper ones?",
        "modern third-party autofocus is close enough for everyday shooting, just a touch "
        "slower on fast action",
    ),
    (
        "alright, i think i'm leaning toward the third-party one then",
        "solid choice, their art-series glass punches well above its price point",
    ),
    (
        "one more thing, do these lenses need a filter or is that overkill?",
        "a clear or uv filter mostly just protects the front element, not essential but cheap "
        "insurance",
    ),
)

# The two-turn flow (#2758's reproduction shape): the user states the rule, Penny asks whether
# it's a standing one, and ONLY the confirmation is the LIVE turn — the rule's own content
# lives in history, not in the ask this case drives.  The restatement adds "job reports too,
# not just chat replies", a detail `_SEEDED_CONTENT` doesn't carry.
_RULE_STATEMENT = (
    "a new rule is: whenever you're about to answer me, don't lay it out as a table — turn "
    "it into bullet points instead, and that goes for your job reports too, not just chat "
    "replies"
)
_CLARIFY_REPLY = (
    "just to check — do you want that as a standing rule I follow from here on, or just for "
    "this one reply?"
)

ASK = "a standing rule please"
ALSO_PHRASED: tuple[str, ...] = (
    "yeah, make it a standing rule",
    "as a standing rule, yes",
)


def _assert_the_rule_statement_restates_and_adds_something_new() -> None:
    """ENFORCED at import: the seeded rule-statement restates the stored policy (tables,
    bullets) in its own words, adds a qualifier the stored content doesn't carry (so the pair
    reads as the same rule with a NEW value, not a duplicate), and never leaks the seeded key
    or its exact sentence — the ladder's identity question is answered from the CONTENT, never
    from a copied key or phrase.  The live asks themselves carry no rule content at all — that
    is the point of the two-turn shape — so there is nothing of the rule's own wording to
    check in them."""
    folded = _RULE_STATEMENT.casefold()
    assert "table" in folded, "the rule statement must name the table rule"
    assert "bullet" in folded, "the rule statement must name the bullet alternative"
    assert _SEEDED_KEY not in folded, "the rule statement must not leak the seeded key"
    seeded_first_words = " ".join(_SEEDED_CONTENT.split()[:4]).casefold()
    assert seeded_first_words not in folded, (
        "the rule statement must not borrow the seeded rule's own words"
    )
    assert "too" in folded, "the rule statement must add a qualifier the stored content lacks"
    for wording in (ASK, *ALSO_PHRASED):
        assert "table" not in wording.casefold(), (
            f"a live ask must carry no rule content of its own: {wording!r}"
        )


_assert_the_rule_statement_restates_and_adds_something_new()


def _seed_world(db: Database) -> None:
    """The user, both rules already standing, the filler small talk, and the rule-statement
    exchange — with the window narrowed so only that last exchange is visible history."""
    db.users.save_info(
        sender=TEST_SENDER,
        name=_USER_NAME,
        location=_USER_LOCATION,
        timezone=_USER_TIMEZONE,
        date_of_birth="1990-01-01",
    )
    rules_run = seeded_run_id("seeded-rules")
    require_memory(db, RULES).write_rule_decided(
        [
            EntryInput(key=_SEEDED_KEY, content=_SEEDED_CONTENT),
            EntryInput(key=_OTHER_KEY, content=_OTHER_CONTENT),
        ],
        author="chat",
        run_id=rules_run,
    )
    for user_text, penny_text in _FILLER_PAIRS:
        ask_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, user_text
        )
        db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING, "penny", penny_text, parent_id=ask_id
        )
    rule_ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, _RULE_STATEMENT
    )
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        _CLARIFY_REPLY,
        parent_id=rule_ask_id,
    )
    # Narrows the chat history window to exactly the rule-statement exchange (#2746's
    # CHAT_HISTORY_TURNS trims by TURN count afterwards, but MESSAGE_CONTEXT_LIMIT is the raw
    # row fetch it trims FROM — set this low enough and the filler never enters the pool it
    # trims out of).
    db.config.set(
        "MESSAGE_CONTEXT_LIMIT",
        "2",
        "eval world: the window is exactly the rule-statement exchange",
    )


def _premise() -> Preparer:
    """Both rules are standing, and the visible window closes on the clarifying question —
    asserted before the live turn runs, so a later miss is the turn's, not the seed's."""

    def probe(penny: Penny) -> None:
        db = penny.db
        held = require_memory(db, RULES).read_all()
        assert {e.key for e in held} == {_SEEDED_KEY, _OTHER_KEY}, (
            f"`{RULES}` must start with both seeded rules, it holds {[e.key for e in held]!r}"
        )
        limit = int(penny.config.runtime.MESSAGE_CONTEXT_LIMIT)
        window = db.messages.get_messages_since(TEST_SENDER, since=datetime.min, limit=limit)
        contents = [m.content for m in window]
        assert contents == [_RULE_STATEMENT, _CLARIFY_REPLY], (
            f"the window must be exactly the rule-statement exchange, it holds {contents!r}"
        )

    return probe


# ── The claims ────────────────────────────────────────────────────────────────


def _rules_entries(sample: SampleObservation) -> list[tuple[str | None, str]]:
    return [(entry.key, entry.content) for entry in sample.held if entry.collection == RULES]


def _rules_holds_one_entry_for_the_policy(sample: SampleObservation, _world: World) -> Answer:
    """STORE: the OTHER rule survives untouched, and the merge survives singularly — exactly
    one entry for the table policy, never a second row for the same thing under a different
    name.  A sample where the ladder did NOT merge (wrote the restatement as a genuinely new,
    second entry) fails this openly: the claim's pass rate is itself how often the pair
    merged."""
    held = _rules_entries(sample)
    other_survived = (_OTHER_KEY, _OTHER_CONTENT) in held
    policy_entries = [entry for entry in held if entry[0] != _OTHER_KEY]
    return other_survived and len(policy_entries) == 1, f"`{RULES}` holds {held!r}"


def _reply_never_claims_a_merged_key(sample: SampleObservation, _world: World) -> Answer:
    """EXACT KEY, no extraction (#2758): for every key THIS TURN'S ``collection_write`` calls
    passed that `rules` does not hold under that name at the end — because it merged into a
    different entry — the reply does not contain that exact string.  A sample that wrote no
    key at all, or whose write landed under its own key (no merge), passes vacuously, which is
    the honest reading: nothing here was false to begin with."""
    held_keys = {key for key, _content in _rules_entries(sample) if key}
    merged_away = [key for key in sample.collection_write_keys if key not in held_keys]
    offending = [key for key in merged_away if key in sample.reply]
    return not offending, f"the reply contains {offending!r}, a key `{RULES}` does not hold"


_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_merge_headline_names_a_real_key(chat_eval: ChatEval, model: str) -> None:
    """A confirmation turn whose rule content sits in history, and the merge it can produce:
    the policy survives as one entry, and the reply never claims a key `rules` doesn't hold."""
    cohort = await chat_eval(
        case_id="rules-write-merge-is-honest",
        behaviour=(
            "In the chat agent, when a confirmed new rule matches an existing one as the same "
            "thing, Penny's write merges onto the existing entry, the policy survives as one "
            "entry, and her reply never claims a key `rules` doesn't hold."
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
        "state: `rules` holds exactly one entry for the policy, the other untouched",
        _rules_holds_one_entry_for_the_policy,
        SpecCategory.STORE,
    )
    cohort.claim(
        "reply: never claims a key `rules` doesn't hold",
        _reply_never_claims_a_merged_key,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.measure(*_MEASURED)
