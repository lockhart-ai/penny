"""Email: answering from the message an ask describes, and a remark that asks nothing (#2209).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

**Two cases.**  The email tools reach a mailbox from plain language, and a message about email
either asks the mailbox something or it does not:

* ``email-answers-from-the-message-asked-about`` — asked what a named sender's email about a
  named subject says, the reply carries that message's figure and no neighbour's;
* ``email-remark-stays-idle`` — told the inbox is out of control, the machine stays in idle
  and the store is as it was.

**Asking by sender and asking by subject are ONE behaviour.**  Both are "find the message the
ask describes and answer from it", and against one mailbox a correct sample for either is
correct for the other: the reply carries the matching message's figure, and none of the
figures the messages beside it carry.  What differs is which field the search is made on —
the sender, the subject, the free text, or several at once — and that is a ROUTE, measured in
the tool sequence and never asserted.  So every wording names BOTH the sender and the subject,
and the mailbox is built so that neither alone picks the message out: the same sender sent a
second message about something else, and a second sender sent mail about the same subject.
A search on either field returns the matching message beside a neighbour, and choosing
between them is the behaviour.

**The remark is its own case**, because its correct end state is not the lookup's: a reply
that answers it carries no figure from the mailbox at all, so the lookup's claims fail it.  It
claims what SURVIVES — the machine still in idle and the store as it was — never that the
mailbox went unsearched: whether to look is the model's call on a remark about her inbox, and
it is measured in the tool sequence (``docs/principles.md`` §4.3).  Nor does it claim anything
about what her reply says.  A reply to a remark is advice in her own words — the apps, settings
and habits she would suggest — none of which the world was ever going to supply, so the reply
is measured as reply spread and never read for provenance.  What she stores is still read: a
value written to the store has to trace to something the turn was given.

**The mailbox is canned at the system boundary.**  The world declares five synthetic messages
(``World.mailbox``), and the driver installs them behind production's own Fastmail tool
builder — ``search_emails`` and ``read_emails``, the shipped descriptions, the shipped
summarising read — so only the backend is fake.  How a canned search matches is stated once,
in ``utils/mailbox.py``; it is deliberately lenient, so a query is answered with every message
carrying any of its words and the neighbours reach the model.

**The world also holds one collection** — the newsletters the user reads.  It is what gives a
preservation claim something to read on both cases: a remark that the inbox is out of control
can reasonably tempt a turn into pruning the user's subscriptions, and an email question is a
read.  Both claim the collection is still there, unchanged; neither claims nothing was written.

**What the claims read.**

* The FIGURE the lookup asks for is ``18,375``, asserted as its last digit group ``375`` — the
  part every rendering of the amount shares (``$18,375`` · ``18,375`` · ``18375``).  It is the
  only answer the ask requests: the sender and the subject are named in the ask itself, so a
  reply repeating them proves nothing about which message was read.
* The NEIGHBOURS' figures — ``145`` (the same sender's site-assessment invoice) and ``990``
  (the other sender's ``$14,990`` advertised price) — are the tokens the reply must not carry.
  Each answers no part of the ask, so a reply stating one has taken a neighbouring message for
  the one asked about.  Provenance cannot see that: once a search returned the neighbour, its
  figure IS something the model was given.
* Every specific value the store carries traces to what the turn was given — the user's words
  and the tool results, which hold the messages the turn actually opened — on both cases, and
  so does every specific value the lookup's reply carries.
* A route is never claimed: not which email tool ran, not whether she read the message or
  answered from its preview, not how many searches it took.

**Blind spots, stated.**  A figure the reply rounds ("about $18.4k") carries none of the
asserted digits, so it misses the answer claim and is invisible to the neighbour claim.  A
reply that names the neighbouring messages without their figures passes the neighbour claim,
which is a finding for a person reading the modal sample.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code owner's to
accept once the numbers have been read.  Every sender, address, subject and figure is invented,
because the repo is public.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.conversation_machine import ConversationState
from penny.penny import Penny
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    ChatEval,
    Preparer,
    collection_entries,
)
from penny.tests.eval.utils.assertions import Answer, Cohort, WorldClaim
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
    fold_typography,
)
from penny.tests.eval.utils.dispatch_world import assert_surface_carries
from penny.tests.eval.utils.fixtures import SynthCollection
from penny.tests.eval.utils.mailbox import CannedEmail
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

# Shared with the sibling dispatch stories (generate_image, choose, the muting contracts) so the
# report's families rollup reads chat-surface tool dispatch as one group.
_FAMILY = "nl-dispatch"

# The surface a Fastmail deployment carries, which is what the driver installs.
EMAIL_TOOLS = ("search_emails", "read_emails")

# What the lookup's reply owes, and what it must not carry — see the module docstring.
QUOTE_FIGURE = "375"
NEIGHBOUR_FIGURES = ("145", "990")


# ── The mailbox ──────────────────────────────────────────────────────────────
#
# Five messages.  The quote is the one the ask describes; the invoice shares its sender, the
# advert shares its subject, and the other two are ordinary mail.  Ids are opaque, as a real
# backend's are, so no id says which message is which.

_PRIYA = "Priya Nakamura"
_PRIYA_ADDRESS = "priya@brightleaf-solar.example"

THE_QUOTE = CannedEmail(
    id="Mqx7",
    subject="Your rooftop solar quote",
    sender=_PRIYA,
    address=_PRIYA_ADDRESS,
    received_at="2026-09-22T15:10:00Z",
    body=(
        "Hi — thanks again for having us out to look at the roof. Here is the quote for the "
        "rooftop solar install we talked about: 14 panels, a 5.6 kW system, installed price "
        f"$18,{QUOTE_FIGURE} including permits. The quote holds until October 31. Reply here "
        "when you'd like to book a start date.\n\nPriya Nakamura\nBrightleaf Solar"
    ),
)

_SAME_SENDER = CannedEmail(
    id="Mrt2",
    subject="Site assessment invoice",
    sender=_PRIYA,
    address=_PRIYA_ADDRESS,
    received_at="2026-09-12T18:40:00Z",
    body=(
        "Hi — here is the invoice for Thursday's site assessment visit: $145, already paid by "
        "card on the day, so nothing is owed. Your quote will follow once our designer has the "
        "roof measurements.\n\nPriya Nakamura\nBrightleaf Solar"
    ),
)

_SAME_SUBJECT = CannedEmail(
    id="Mvk9",
    subject="Rooftop solar quotes — this month only",
    sender="Sunvale Energy",
    address="offers@sunvale-energy.example",
    received_at="2026-09-25T13:00:00Z",
    body=(
        "Thinking about rooftop solar? Book a free home quote before October 15 and lock in "
        "our autumn pricing: complete installs from $14,990. Unsubscribe any time."
    ),
)

_LIBRARY = CannedEmail(
    id="Mhp4",
    subject="Your hold is ready",
    sender="Cedar Hill Library",
    address="holds@cedarhill-library.example",
    received_at="2026-09-27T09:05:00Z",
    body=(
        "The book you reserved, The Salt Road, is waiting at the front desk. Please pick it "
        "up within a week."
    ),
)

_NEWSLETTER = CannedEmail(
    id="Mzd6",
    subject="Harbor Weekly — this Friday's roundup",
    sender="Harbor Weekly",
    address="news@harborweekly.example",
    received_at="2026-09-26T11:00:00Z",
    body=(
        "This week: the ferry timetable changes, a new bakery on Quay Street, and the autumn "
        "market returns to the square on Saturday."
    ),
)

MAILBOX = (THE_QUOTE, _SAME_SENDER, _SAME_SUBJECT, _LIBRARY, _NEWSLETTER)

# The collection the store already holds — see the module docstring.
NEWSLETTERS = SynthCollection(
    "newsletters",
    "Newsletters the user is subscribed to and actually reads, with what each one covers.",
    entries=(
        "Harbor Weekly — local news roundup, every Friday",
        "The Seedling — gardening tips, every other Tuesday",
    ),
)


# ── The asks ─────────────────────────────────────────────────────────────────
#
# Five wordings of ONE message per case.  The lookup's every wording names the sender, the
# subject and the figure wanted; what varies is how a person says it.  The remark's every
# wording names email and asks nothing of it.

_LOOKUP_ASK = "did priya nakamura send me the rooftop solar quote yet? what does it come to?"
_LOOKUP_PHRASINGS = (
    "can you check my email for the rooftop solar quote from priya nakamura and tell me the price?",
    "what's the total on priya nakamura's rooftop solar quote? it should be in my email",
    "look through my inbox for the rooftop solar quote priya nakamura sent — how much is it?",
    "i think priya nakamura emailed me that rooftop solar quote, how much did it come out to?",
)

_REMARK = "honestly i get way too much email these days, my inbox is out of control"
_REMARK_PHRASINGS = (
    "my inbox has been a total mess lately, there's just so much email",
    "ugh, the amount of email i get now is ridiculous",
    "i swear my inbox doubles every week, it's too much email",
    "email has gotten completely out of hand for me lately",
)


# ── The worlds ───────────────────────────────────────────────────────────────
#
# One mailbox and one collection under both cases.  ``keeps`` and ``excludes`` are EMPTY on
# both, and that is a report: neither ask tells her to keep anything.  ``answers`` is the
# lookup's figure and nothing on the remark, which asks for nothing.

LOOKUP_WORLD = World(
    name="a mailbox with the quote, its sender's other mail and other mail on its subject",
    pages=(),
    keeps=(),
    excludes=(),
    answers=(QUOTE_FIGURE,),
    stores=(NEWSLETTERS,),
    mailbox=MAILBOX,
)

REMARK_WORLD = World(
    name="the same mailbox, and a remark about it",
    pages=(),
    keeps=(),
    excludes=(),
    stores=(NEWSLETTERS,),
    mailbox=MAILBOX,
)


class EmailCase(NamedTuple):
    """One case: its id, its sentence, its world and the five wordings of its one message."""

    case_id: str
    behaviour: str
    world: World
    ask: str
    also_phrased: tuple[str, ...]


LOOKUP = EmailCase(
    case_id="email-answers-from-the-message-asked-about",
    behaviour=(
        "In the chat agent, when the user asks what an email from a named sender about a named "
        "subject says, Penny answers with the figure that message carries and none of the "
        "figures the messages beside it carry, and the turn ends back in idle."
    ),
    world=LOOKUP_WORLD,
    ask=_LOOKUP_ASK,
    also_phrased=_LOOKUP_PHRASINGS,
)

REMARK = EmailCase(
    case_id="email-remark-stays-idle",
    behaviour=(
        "In the chat agent, when the user remarks on their email without asking anything of it, "
        "Penny stays in idle and everything the store already held is still there, unchanged."
    ),
    world=REMARK_WORLD,
    ask=_REMARK,
    also_phrased=_REMARK_PHRASINGS,
)

# Every case, in one place — so the deterministic pins in ``test_eval_harness.py`` can hold each
# world and its claims without a GPU.
EMAIL_CASES = (LOOKUP, REMARK)


# ── The premise, asserted inside each sample before the turn ─────────────────


def assert_mailbox_world(penny: Penny, case: EmailCase) -> None:
    """The world this case is answered in, asserted out loud: the email tools are on the chat
    surface, and the store holds the newsletters collection exactly as the world seeds it.

    The surface is read off the real ``get_tools``, so a mailbox that failed to install fails
    here naming the missing tools rather than scoring fifteen samples that were never offered
    one.  The collection is read back whole, so the preservation claim reads a store that
    really held what the world says."""
    assert_surface_carries(penny, case.case_id, EMAIL_TOOLS)
    for held in case.world.stores:
        stored = collection_entries(penny.db, held.name)
        assert stored == dict(held.keyed), (
            f"{case.case_id}: {held.name!r} must hold exactly what the world seeds — it holds "
            f"{stored}"
        )


def _probe(case: EmailCase) -> Preparer:
    def probe(penny: Penny) -> None:
        assert_mailbox_world(penny, case)

    return probe


# ── The claims ───────────────────────────────────────────────────────────────


def _carries_no_neighbour_figure(tokens: tuple[str, ...]) -> WorldClaim:
    """The reply carries none of the figures only a neighbouring message carries.

    Read through ``fold_typography``, the one definition every reply probe folds through.  The
    rationale names the figure the reply carried, which says which neighbour it was taken from."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        said = fold_typography(sample.reply)
        carried = [token for token in tokens if token in said]
        return not carried, f"the reply states {carried}, which only a neighbouring message carries"

    return answer


def claim_the_lookup(cohort: Cohort) -> None:
    """Every claim the lookup case makes, declared in one place so the deterministic pins
    answer the very set the case declares."""
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE — the figure the ask asked for is in the reply (a delivered message is a record the
    # store holds), and the collection the turn was not sent to is as it was.
    cohort.assert_the_reply_answers_the_ask()
    cohort.assert_what_the_store_held_survives()

    # PROVENANCE — nothing stored or said was invented, and the figure came from the message
    # asked about rather than from one beside it.
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.claim(
        "reply: it states no figure only a neighbouring message carries",
        _carries_no_neighbour_figure(NEIGHBOUR_FIGURES),
        SpecCategory.PROVENANCE,
        kind="reply",
    )


def claim_the_remark(cohort: Cohort) -> None:
    """Every claim the remark case makes — where the turn landed, what survived it, and that
    nothing it stored was invented."""
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE — what was already there survives.  The remark asks for nothing, so there is no
    # answer for the reply to owe.
    cohort.assert_what_the_store_held_survives()

    # PROVENANCE — the STORE half only.  The reply to a remark is advice in her own words, so it
    # is behaviour, measured as reply spread rather than read for provenance.
    cohort.assert_every_value_in_the_store_is_sourced()


# ``TOOL_SEQUENCE`` carries the route — which field she searched on, whether she read the message
# or answered from its preview, whether she looked at all on the remark.  ``ENTRIES_STORED``
# carries whether a turn chose to note something.  The registry features are out: neither ask
# teaches a routine, so they would read the same empty registry on every sample.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


async def _drive(chat_eval: ChatEval, model: str, case: EmailCase) -> Cohort:
    return await chat_eval(
        case_id=case.case_id,
        behaviour=case.behaviour,
        model=model,
        prepare=_probe(case),
        world=case.world,
        ask=case.ask,
        also_phrased=case.also_phrased,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,  # a search, a read and its summarising call, maybe a second search
    )


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_reply_comes_from_the_message_asked_about(
    chat_eval: ChatEval, model: str
) -> None:
    """A sender and a subject that only one message shares, and the figure it carries."""
    cohort = await _drive(chat_eval, model, LOOKUP)
    claim_the_lookup(cohort)
    cohort.measure(*_MEASURED)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_remark_about_email_stays_idle(chat_eval: ChatEval, model: str) -> None:
    """Email named, and nothing asked of it."""
    cohort = await _drive(chat_eval, model, REMARK)
    claim_the_remark(cohort)
    cohort.measure(*_MEASURED)
