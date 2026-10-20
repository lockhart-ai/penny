"""Standing rules: keeping, reading back and forgetting a rule the user gives (#2633).

The contract is `docs/eval-case-design.md`.

A standing rule is a thing the user asked Penny to always or never do.  It is an
ordinary entry in the seeded ``rules`` collection — one ``collection_write`` to
keep it, one ``collection_remove`` to forget it — and the ``### Standing rules``
section renders the collection in full wherever Penny acts (the chat self-state
header, every collector cycle's prompt, the notify document), so a rule holds
without a lookup.  These cases are the chat-surface contract for that
surface:

| case | the behaviour | what the store must show |
|---|---|---|
| keep one | ``rules-remember-always`` | an entry in ``rules`` carries the asked behaviour |
| list them | ``rules-list-standing`` | ``rules`` unchanged; the reply states each rule |
| forget one | ``rules-forget-one`` | the named rule gone; the other two unchanged |
| honour one | ``rules-honoured-in-chat-reply`` | the reply carries the rule's sign-off |

**The store map is the state the first case is answered from.**  ``rules`` seeds
empty, so with no rule set the section is omitted entirely (#2686) and the write
that keeps the first rule reads its destination off the store map, which always
lists ``rules`` with its description — the anchor is on screen before any rule
exists, which is the whole reachability contract of #2633.  The two later cases
seed three fictional rules through the store's own write path (author ``user``),
and the premise probe asserts the section renders them — a world that stopped
showing what the reply is answered from would make every sample measure
guesswork.

**What is NOT claimed.**  That she called ``collection_write`` / ``collection_remove``
is a ROUTE — a rule is an entry, and nothing here keys to a verb — so it is measured
in the tool sequence and never asserted.  How the reply words the keeping or the
forgetting is the reply's spread.  No wording of a reply is matched beyond the
answers tokens below, and no call is forced: the ask leads, and the real gate
answers whatever she draws.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code
owner's to accept once the numbers have been read.  Every rule is fictional — an
invented preference about linking, units and reply length — because the repo is
public.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.agents.self_state import standing_rules_section
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory.objects import EntryInput
from penny.penny import Penny
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import EVAL_MODELS, Preparer, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    token_stands_in,
)
from penny.tests.eval.utils.fixtures import SynthCollection
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "standing-rules"

# The collection every rule lives in — the anchor the section renders and the write
# verbs take.  Read off the constant, never restated.
RULES = PennyConstants.MEMORY_RULES_COLLECTION

# The three fictional rules the read-back and forget cases seed, oldest first.  Each
# one's TOKEN is the smallest datum that identifies it — it appears in the rule's own
# words (so the ambient section carries it and a reply citing the rule carries it),
# and in no wording of any ask below, enforced at import.
_PAGE_LINKS = ("page links", "always include a link when you mention a page")
_NO_METRIC = ("no metric units", "never use metric units")
_SHORT_REPLIES = ("short replies", "keep replies under three sentences")

_SEEDED_RULES = (_PAGE_LINKS, _NO_METRIC, _SHORT_REPLIES)
_SEEDED_TOKENS = tuple(
    (rule, token)
    for rule, token in zip(_SEEDED_RULES, ("link", "metric", "sentences"), strict=True)
)


def _seed_rules(db: Database, entries: tuple[tuple[str, str], ...]) -> None:
    """Lay the case's rules down through the store's own write path, under the keys
    the section will render — a rule is an entry like any other."""
    require_memory(db, RULES).write(
        [EntryInput(key=key, content=content) for key, content in entries],
        author="user",
        run_id=seeded_run_id(RULES),
    )


def _premise(rules: tuple[tuple[str, str], ...]) -> Preparer:
    """The case's premise, asserted inside the sample it belongs to: the shared
    standing-rules render shows the state the claims below are answered against —
    omitted entirely when nothing is seeded (#2686), every rule in full when they
    are.  Read through the header's own render so the probe and the surface cannot
    drift into asserting different words for one state; a seeded rule renders as
    its own sentence — ``- <content>``, no key, no stamp (#2647) — and the line is
    matched to the content whole (pinned whole in ``test_self_state.py``)."""

    def probe(penny: Penny) -> None:
        rendered = standing_rules_section(penny.db)
        if not rules:
            assert rendered == "", f"the section must be omitted when no rule stands:\n{rendered!r}"
            return
        lines = rendered.splitlines()[1:]  # drop the section title
        assert len(lines) == len(rules), f"the section must render every rule:\n{rendered!r}"
        for line, (_key, content) in zip(lines, rules, strict=True):
            assert line == f"- {content}", f"a seeded rule must render whole: {line!r}"

    return probe


class _RuleCase(NamedTuple):
    """One thing the user does with their standing rules, in five wordings.

    ``ask`` and ``also_phrased`` are five wordings of ONE message against one world.
    ``rules`` is what the ``rules`` collection holds when the turn begins (empty for
    the keep case — the state its write is answered from).  ``answers`` is what the
    reply owes, and only the read-back owes anything: an instruction's complete
    answer names nothing.
    """

    case_id: str
    behaviour: str
    ask: str
    also_phrased: tuple[str, ...]
    rules: tuple[tuple[str, str], ...] = ()
    answers: tuple[str, ...] = ()
    # A collection the world already holds when the turn begins — the fact an ordinary
    # question is answered from.  Empty for the three asks about the rules themselves;
    # the honour case carries the pantry its ask reads.
    stores: tuple[SynthCollection, ...] = ()

    @property
    def ground(self) -> World:
        """The world every arm is answered against — the rules collection is seeded
        by the migration itself and this case's own entries by ``seed=``, so the
        world's declared ``stores`` stay empty of rules: declaring them would lay the
        entries down a second time.  A case that answers from a fact in the store
        declares that collection here (the honour case's pantry); the driver lays it
        down through the same seed path.  No pages: nothing here is answered off the
        web, and a page set would hand a browsing sample something to talk about
        instead of letting the wrong turn read as one."""
        return World(
            name=self.case_id,
            pages=(),
            keeps=(),
            excludes=(),
            answers=self.answers,
            stores=self.stores,
        )


async def _drive(chat_eval: ChatEval, model: str, case: _RuleCase) -> Cohort:
    """Drive one case: the rules laid down through the store's own write path, the
    premise re-asserted before the turn."""
    return await chat_eval(
        case_id=case.case_id,
        behaviour=case.behaviour,
        area=Area.MEMORY,
        model=model,
        seed=_seed_rules_for(case),
        prepare=_premise(case.rules),
        world=case.ground,
        ask=case.ask,
        also_phrased=case.also_phrased,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )


# chat_eval's ``seed`` is a Seeder over the database; the case's rules are its whole
# argument, so one adapter serves every case (and seeds nothing when the case has none).
def _seed_rules_for(case: _RuleCase):
    def seed(db: Database) -> None:
        if case.rules:
            _seed_rules(db, case.rules)

    return seed


# ── The claims, as pure functions over one sample ─────────────────────────────


def _rules_entries(sample: SampleObservation) -> list[tuple[str | None, str]]:
    """Every entry ``rules`` holds when the sample ends, as (key, content) — the
    whole entry, key and content, because a fact in the key and a blurb in the body
    is a perfectly good way to store it."""
    return [(entry.key, entry.content) for entry in sample.held if entry.collection == RULES]


def _the_rule_is_kept(token: str):
    """An entry in ``rules`` carries the behaviour the ask named — the store's END
    STATE, read off the whole entry (key and content together), never the verb that
    wrote it.  A violating sample is nameable: one that stored it under a collection
    it invented, one that left it only in the reply, and one that stored something
    else entirely and said it had kept the rule."""

    def answer(sample: SampleObservation, _world: World) -> tuple[bool, str]:
        text = " ".join(f"{key or ''} {content}" for key, content in _rules_entries(sample))
        found = token in text.casefold()
        return found, f"`{RULES}` holds: {sorted(_rules_entries(sample))}"

    return answer


def _no_other_collection_is_created(sample: SampleObservation, _world: World) -> tuple[bool, str]:
    """Nothing new stands up beside ``rules`` — read off the registry's own
    born-this-run mark, so a collection the turn created under any name fails the
    claim by name."""
    born = [row.name for row in sample.mechanisms if row.born_this_run]
    return not born, f"created {born}"


def _the_named_rule_is_gone(token: str):
    """The rule the ask named no longer holds — its token appears in no entry the
    collection still carries.  A violating sample is nameable: one that removed the
    WRONG rule, one that rewrote the rule instead of removing it, and one that said
    it was forgotten while it is still there."""

    def answer(sample: SampleObservation, _world: World) -> tuple[bool, str]:
        still = [
            pair
            for pair in _rules_entries(sample)
            if token in f"{pair[0] or ''} {pair[1]}".casefold()
        ]
        return not still, f"still holding {token!r}: {still}"

    return answer


def _the_others_survive_unchanged(seeded: tuple[tuple[str, str], ...]):
    """The rules the ask did NOT name are still there, exactly as the turn found
    them — the whole entry compared, key and content, so a rewrite-in-place fails
    the same way a deletion does.  A violating sample is nameable: one that
    tidies while it forgets, dropping or rewording a rule it was not asked about."""

    def answer(sample: SampleObservation, _world: World) -> tuple[bool, str]:
        held = set(_rules_entries(sample))
        missing = [pair for pair in seeded if pair not in held]
        return not missing, f"no longer as seeded: {missing}"

    return answer


def _rules_unchanged(seeded: tuple[tuple[str, str], ...]):
    """``rules`` is exactly as the turn found it — the whole collection compared,
    key and content, against what the case seeded.  A violating sample is nameable:
    one that reworded the rule while answering, dropped it, or added a second one."""

    def answer(sample: SampleObservation, _world: World) -> tuple[bool, str]:
        held = set(_rules_entries(sample))
        return held == set(seeded), f"`{RULES}` holds {sorted(held)}"

    return answer


def _the_reply_carries_the_sign_off(sample: SampleObservation, _world: World) -> tuple[bool, str]:
    """The reply carries the sign-off the standing rule names — present in the message
    (#2684), however her wording or a trailing emoji sits around it.  The ask never
    mentions the rule, so the only place the word can come from is the section
    itself — the whole point of the honour case.  A violating sample is nameable:
    one that answered correctly and never signed off."""
    ok = token_stands_in(_SIGNOFF, sample.reply)
    return ok, f"the reply is {sample.reply!r}"


_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


# ═══ keep one ════════════════════════════════════════════════════════════════
#
# "Remember to always X."  The rules collection seeds EMPTY, so the write reads its
# destination off the section's own empty line — the anchor the render keeps on
# screen before the first rule exists.  The token is in the ask by nature (the ask
# IS the source of the behaviour), which is what makes the store claim a read: what
# landed in `rules` must carry what was said, wherever the draw chose to put it.

_REMEMBER = _RuleCase(
    case_id="rules-remember-always",
    behaviour=(
        "In the chat agent, when the user asks Penny to always do something, Penny "
        "keeps it as a standing rule."
    ),
    ask="remember to always include a link when you mention a page",
    also_phrased=(
        "from now on, always add a link whenever you bring up a page",
        "always link the page when you talk about one, ok?",
        "i want you to always include a link whenever you mention a page",
        "make it a rule: whenever you mention a page, include a link",
    ),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_always_ask_is_kept_as_a_standing_rule(chat_eval: ChatEval, model: str) -> None:
    """The rules collection is empty, the user asks Penny to always do something, and
    the behaviour is still there when the turn ends — in ``rules``, where the section
    renders it every turn afterwards."""
    cohort = await _drive(chat_eval, model, _REMEMBER)
    # STORE — the behaviour landed in `rules`, and nothing else was created.
    cohort.claim(
        "state: an entry in `rules` carries the behaviour they asked for",
        _the_rule_is_kept("link"),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: no other collection is created",
        _no_other_collection_is_created,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ list them ═══════════════════════════════════════════════════════════════
#
# "What are my standing rules?"  Three rules seeded, so the answer is entirely
# ambient — the section carries each rule in full, and the reply states each one.
# Which read she reaches for (the section itself, or one ``collection_list``) is a
# route, measured in the tool sequence.

_LISTED_TOKENS = tuple(token for _, token in _SEEDED_TOKENS)


def _answers_tokens_carried() -> None:
    """ENFORCED at import: each answer token is on its own rule's words, on no other
    rule's, and in no wording of the ask — a token the question carries would credit
    a reply for repeating the question back."""
    for rule, token in _SEEDED_TOKENS:
        key, content = rule
        assert token in f"{key} {content}".casefold(), (
            f"{token!r} must be on its own rule's words: {rule}"
        )
        for other, _ in _SEEDED_RULES:
            if other is not rule:
                assert token not in f"{other[0]} {other[1]}".casefold(), (
                    f"{token!r} must not be on another rule's words: {other}"
                )


_answers_tokens_carried()

_LIST_STANDING = _RuleCase(
    case_id="rules-list-standing",
    behaviour=(
        "In the chat agent, when the user asks what their standing rules are, Penny names each one."
    ),
    ask="what are my standing rules?",
    also_phrased=(
        "remind me what rules i've asked you to keep",
        "what have i told you to always or never do?",
        "list my standing rules for me",
        "what rules do you follow because i asked you to?",
    ),
    rules=_SEEDED_RULES,
    answers=_LISTED_TOKENS,
)

for _wording in (_LIST_STANDING.ask, *_LIST_STANDING.also_phrased):
    for _token in _LISTED_TOKENS:
        assert _token not in _wording.casefold(), (
            f"a read-back wording must not carry an answer token: {_wording!r}"
        )


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_she_names_every_standing_rule(chat_eval: ChatEval, model: str) -> None:
    """Three rules standing, and a question about all of them: the reply names each
    one, every value it states traces to the round it was given, and the rules are
    exactly as the turn found them.

    What is NOT claimed is how the listing is worded — a reply quoting the rules in
    any words is as faithful as another."""
    cohort = await _drive(chat_eval, model, _LIST_STANDING)
    # STORE — the rules are as the turn found them.
    cohort.assert_what_the_store_held_survives()

    # PROVENANCE — the reply states each rule, and every value it states was given.
    cohort.assert_the_reply_answers_the_ask()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ forget one ══════════════════════════════════════════════════════════════
#
# "Forget the rule about X."  Three rules seeded, and the ask names ONE — so the
# remove is determined, and the contract is both halves: the named rule is gone,
# and the two it did not name are untouched.  Every wording names its target in the
# rule's own words, so the section's rendered key is the anchor the remove reads.

_FORGET_TOKEN = "metric"

_FORGET_ONE = _RuleCase(
    case_id="rules-forget-one",
    behaviour=(
        "In the chat agent, when the user asks Penny to forget one standing rule, "
        "Penny removes that rule and keeps the others."
    ),
    ask="forget the rule about metric units",
    also_phrased=(
        "drop the never-use-metric-units rule",
        "i changed my mind — i'm fine with metric units, forget that rule",
        "remove the rule that says never to use metric units",
        "please forget the metric-units rule i gave you",
    ),
    rules=_SEEDED_RULES,
)

for _wording in (_FORGET_ONE.ask, *_FORGET_ONE.also_phrased):
    assert _FORGET_TOKEN in _wording.casefold(), (
        f"a forget wording must name the rule it targets: {_wording!r}"
    )


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_forgotten_rule_is_gone_and_the_others_stand(chat_eval: ChatEval, model: str) -> None:
    """The named rule is gone from ``rules`` and the other two are still there,
    exactly as the turn found them — the failure this exists to catch is a turn that
    forgets the wrong rule, or tidies the neighbours on the way past."""
    cohort = await _drive(chat_eval, model, _FORGET_ONE)
    # STORE — the named rule gone; the others untouched.
    cohort.claim(
        "state: the rule they named is gone",
        _the_named_rule_is_gone(_FORGET_TOKEN),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the other two are still there, unchanged",
        _the_others_survive_unchanged((_PAGE_LINKS, _SHORT_REPLIES)),
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# Every case, in one place — so the deterministic probe in ``test_eval_harness.py``
# can drive each one's seeder and premise without a GPU.
# ═══ honour one in the reply ═════════════════════════════════════════════════
#
# A rule about HOW every message ends, and an ordinary question the world answers from —
# the ask never mentions the rule, so the only place the sign-off can come from is the
# section itself, which is the whole reachability contract of #2633 made behavioural: a
# rule that renders and is never acted on is not a rule.  The sign-off token is exact
# and case-sensitive (``token_stands_in``) and may sit anywhere in the message — a
# trailing emoji after it is fine (#2684); how the reply words the answer around it is
# the reply's spread.

_SIGNOFF_KEY = "sign-off"
_SIGNOFF_CONTENT = 'End every message you send me with the word "Finished!"'
_SIGNOFF_RULE = (_SIGNOFF_KEY, _SIGNOFF_CONTENT)
_SIGNOFF = "Finished!"

# The small fact the honour case's turn answers from — a pantry with three items, so the
# reply has real work in front of the sign-off.  Seeded by the driver off the world's
# ``stores``, so the world render and the seed are one declaration.
_PANTRY = SynthCollection(
    "pantry",
    "What is in the pantry right now: staples and how much of each is left.",
    entries=(
        "Basmati rice — half a bag left, about 400 g.",
        "Chickpeas — two unopened tins.",
        "Olive oil — about a third of a bottle.",
    ),
)

# What the reply owes: each pantry item, named.  Chosen like every answers token set —
# the smallest datum unique in this world, on the item's own words, on no other source's,
# and in no wording of the ask (enforced at import below).
_PANTRY_WORDS = ("rice", "chickpeas", "olive oil")


def _honoured_tokens_stood_down() -> None:
    """ENFORCED at import: the sign-off token is on the rule's words and nothing else's,
    and each pantry token is on its item's words, on no other source's, and in no wording
    of the ask — a token the ask or the rule carried would credit the reply for repeating
    what it was handed."""
    pantry_words = " ".join(_PANTRY.entries).casefold()
    rule_words = f"{_SIGNOFF_KEY} {_SIGNOFF_CONTENT}".casefold()
    assert _SIGNOFF.casefold() in rule_words, "the sign-off token must be on the rule's words"
    assert _SIGNOFF.casefold() not in pantry_words, (
        "the sign-off token must not be on the pantry's words"
    )
    for token in _PANTRY_WORDS:
        assert token in pantry_words, f"{token!r} must be on its item's words: {_PANTRY.entries}"
        assert token not in rule_words, f"{token!r} must not be on the rule's words"


_honoured_tokens_stood_down()


_HONOUR_REPLY = _RuleCase(
    case_id="rules-honoured-in-chat-reply",
    behaviour=(
        "In the chat agent, when a standing rule asks for a sign-off on every "
        "message, Penny's reply carries it."
    ),
    ask="what do I have in the pantry?",
    also_phrased=(
        "what's in the pantry right now?",
        "can you tell me what food is in the pantry?",
        "check the pantry for me — what's in it?",
        "how are my pantry staples looking?",
    ),
    rules=(_SIGNOFF_RULE,),
    answers=_PANTRY_WORDS,
    stores=(_PANTRY,),
)

for _wording in (_HONOUR_REPLY.ask, *_HONOUR_REPLY.also_phrased):
    for _token in (*_PANTRY_WORDS, _SIGNOFF):
        assert _token.casefold() not in _wording.casefold(), (
            f"an honour wording must not carry an answer token: {_wording!r}"
        )


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_reply_ends_the_way_a_standing_rule_says(chat_eval: ChatEval, model: str) -> None:
    """One standing rule about how every message ends, and an ordinary question about
    the pantry — the reply answers the ask AND carries the sign-off the rule asks
    for, with the rules exactly as the turn found them."""
    cohort = await _drive(chat_eval, model, _HONOUR_REPLY)
    # STORE — the rule honoured on the delivered reply, and the rules untouched.
    cohort.claim(
        "state: the reply carries the rule's sign-off",
        _the_reply_carries_the_sign_off,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: `rules` is exactly as the turn found it",
        _rules_unchanged((_SIGNOFF_RULE,)),
        SpecCategory.STORE,
    )

    # PROVENANCE — the reply names the pantry items, and every value it states was given.
    cohort.assert_the_reply_answers_the_ask()
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.assert_every_value_in_the_store_is_sourced()

    cohort.measure(*_MEASURED)


RULE_CASES = (_REMEMBER, _LIST_STANDING, _FORGET_ONE, _HONOUR_REPLY)
