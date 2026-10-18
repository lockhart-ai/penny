"""A price watch, one cycle at a time: the first reading, an unchanged one, a moved one
(#2017, the collector half of the cohort port), and a moved one on a real-length, decoy-rich
page (#2366).

**Four cases, because there are four behaviours.**  A watch does three different things and
each is a separate test with its own entry condition, its own single cycle and its own
assertions.  What selects the behaviour is the ENTRY CONDITION — what the collection already
holds when the cycle starts — not a script of cycles inside one test body:

* ``watch-writes-the-first-reading`` — the collection is EMPTY and the page says ``$499``.
  A first observation is news: record it, and say so once.
* ``watch-stays-quiet-when-the-reading-has-not-moved`` — the collection already holds
  ``$499`` and the page still says ``$499``.  The write gate STOPS: nothing is written and
  nothing reaches the send queue.
* ``watch-writes-and-tells-when-the-reading-moves`` — the collection holds ``$499`` and the
  page now says ``$449``.  The standing key is rewritten and the user is told once.

The fourth case, ``watch-tells-the-asking-price-from-other-prices-on-the-same-listing``, is the
SAME moved-reading behaviour as the third, on a different hazard: a page several kilobytes
long whose asking price is stated UNLABELLED, in a price ladder of other prices of the SAME
item — list, member, bundle, per-unit, financing, a protection plan — rather than other
products' prices, and rather than a matchable ``Current price:`` anchor.  It is a separate
case rather than a fifth arm of the third because what makes it worth running is the hazard
shape, not a sixth wording of the same one (#2366); see its own section below.

The middle case is the only one that can exist at all: the write gate's
``KEY_EXISTS_UNCHANGED`` STOP fires on a SECOND reading of a value already stored, so
structural silence has nothing to fire on until something is in the collection.  Preseeding it
is what gives that cycle its subject — and it is preseeded through the store's own write path,
under the key the program writes to, so the gate compares against a row of exactly the shape a
real cycle would have left.

**The arms are five wordings of one instruction, over one set of facts.**  A collector has no
user turn, so its natural-language surface is the ``extract`` instruction in its rendered
program and the prose of the page that answers it.  That instruction is written by the
``SkillSubstitution`` on the ``extract`` path and reaches the model through the shipped
instantiation seam — ``retarget_writes`` → ``bind_parameters`` → ``render_skill`` — so varying
it varies a draw rather than a hand-authored render.  Which page each arm reads is the other
half: same url, same product, same price, five catalogue-grade prose variants around a
byte-identical datum line.

**The FACTS are constant, and the claims hinge on them.**  One listing, one url, one pair of
prices: ``$499`` before the change and ``$449`` after.  So these cases can say what the store
holds by name.  The two prices are mutually exclusive — neither is a substring of the other —
because the moved case asserts one is present and the other gone.

**No claim reads the machine, and none reads how a run ended (#2496).**  A collector moves
no conversation machine, so there was never a landing to claim; and a cycle's run outcome and
stop reason are the run record's own fields — retired as claims — so what each case asserts is
what the store holds, what reached the user, and provenance.  How a cycle went is measured in
the tool sequence below, never claimed.

The first three cases are the whole of the collector's enactment coverage for a page whose
only hazard is OTHER PRODUCTS' prices.  A watch over a different page, a different datum or a
different routine is another program of the same behaviour, not another behaviour, so a case
for it would split one behaviour's samples across two numbers rather than measure anything
these three do not (#2007).  The fourth case is kept apart from that rule because its hazard —
an unlabelled price among several of the SAME item, on a page an order of magnitude larger —
is not a shape any of the first three's pages exercise.

Report-only (``min_pass_rate=None``).  All content is synthetic — the house listing fixture on
an invented marketplace — because the repo is public.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import pytest

from penny.database import Database
from penny.database.memory import EntryInput
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    CollectorCyclesEval,
    CycleArm,
    collection_entries,
    seeded_run_id,
)
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    StoredEntry,
)
from penny.tests.eval.utils.fixtures import LISTING_URL, CannedPage, datum
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "collector-enactment"

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"

# The run that laid the entry condition down.  A SEEDED id, structurally distinguishable from
# a live ``uuid4().hex``, so every "what did this sample write" reader excludes the preseeded
# row at the one chokepoint rather than by remembering to.
_SEED_RUN = seeded_run_id("watch-entry-condition")

# The one collection every arm's job is configured on.  One name, because one job.
_CONTAINER = "listing-price-watch"
_CONTAINER_DESCRIPTION = "The asking price on the listing I'm watching, as it stands."

# The calls the routine makes, in order — what the stored program must read back as under the
# strict rendered dialect, and therefore what each cycle's tool surface is scoped to.
_PROGRAM_CALLS = ("browse", "collection_write")

# The job's cadence.  Stated because a configured collection has one, though the cycle is
# driven through ``run_for``, which bypasses readiness.
_SCHEDULE = "FREQ=HOURLY"

# The key the program writes under, and therefore the key an entry condition must use: the
# write gate compares a candidate against what is stored UNDER THE SAME KEY, so a preseeded
# row under any other key would leave the unchanged case with nothing to be unchanged against.
_KEY = "asking price"

# ── The facts, held constant across every arm ────────────────────────────────
#
# One listing, one url, one pair of prices — the house listing every other case in the suite
# is built on, so a reader who knows `LISTING_URL` knows this world already.  Constant is what
# lets the claims below name a value: five listings would force every one of them back to a
# shape claim, and a shape claim cannot tell a watch that recorded the right price from one
# that recorded a plausible number.
_ITEM = "Aurora Deck 2"
_MATCH = "aurora-deck-2"

# The two readings, in their two roles.  The AMOUNT is the identity of the reading and what
# every claim matches on; the PRICE is how a listing displays it, and it is what the pages
# carry.  Keeping the two apart is what lets a page look like a real page while a claim answers
# the question it is actually asking.
#
# Both forms are MUTUALLY EXCLUSIVE — neither amount is a substring of the other, and neither
# price is a substring of the other, in the bare form or in the instruction-labelled pair a
# cycle may store since #1918 — because the moved case asserts one is present and the other
# gone.  ``499`` and ``4499`` would not have been.
_BASELINE_AMOUNT = "499"
_MOVED_AMOUNT = "449"
_BASELINE_PRICE = f"${_BASELINE_AMOUNT}"
_MOVED_PRICE = f"${_MOVED_AMOUNT}"

# The watched line itself, byte-identical on all five pages, and labelled as THE current price
# so the neighbouring items' prices further down each page cannot be read as it (round 5 of
# the enactment beat measured a wrong-price grab against an unlabelled one).
_DATUM = f"Current price: {_BASELINE_PRICE}"
_MOVED_DATUM = f"Current price: {_MOVED_PRICE}"


class Reading(NamedTuple):
    """One arm: how the job's ``extract`` step words what to look for, and the page prose that
    answers it.

    ``name`` is what this arm is called wherever it is identified — its world's name, and the
    row a reader opens when the report says a phrasing diverged.  ``extract`` is the
    substitution description that becomes ``extract={…}`` in the rendered program, which is the
    only natural language a collector cycle is handed.  ``body`` is this arm's whole page."""

    name: str
    extract: str
    body: str


class WatchCase(NamedTuple):
    """One behaviour: what the collection holds when the cycle starts, and what the page says.

    Two fields, because those two settings are the whole difference between the three cases.
    ``stored`` is the price already in the collection, or ``None`` for a collection that
    arrives empty; ``shows`` is the price on the page the single cycle reads."""

    case_id: str
    stored: str | None
    shows: str


FIRST_READING = WatchCase("watch-writes-the-first-reading", None, _BASELINE_PRICE)
UNCHANGED_READING = WatchCase(
    "watch-stays-quiet-when-the-reading-has-not-moved", _BASELINE_PRICE, _BASELINE_PRICE
)
MOVED_READING = WatchCase(
    "watch-writes-and-tells-when-the-reading-moves", _BASELINE_PRICE, _MOVED_PRICE
)
# Same entry condition as MOVED_READING — the collection holds the baseline, the page now
# shows the moved amount — paired with the messy, same-item-decoy page below instead of the
# cross-item-decoy one (#2366).  A separate id because the hazard, not the entry condition, is
# what this case exists to measure; see the module docstring.
SAME_ITEM_DECOYS_READING = WatchCase(
    "watch-tells-the-asking-price-from-other-prices-on-the-same-listing",
    _BASELINE_PRICE,
    _MOVED_PRICE,
)
CASES = (FIRST_READING, UNCHANGED_READING, MOVED_READING, SAME_ITEM_DECOYS_READING)


# The one sentence each case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The locus is the SHIPPED name of where the behaviour happens, and the four
# share an opening clause and differ only at the entry condition or the hazard — that
# parallelism is what the split reads like out loud.  The case id is a filename; this is the
# contract.
_BEHAVIOUR = {
    FIRST_READING.case_id: (
        "In a price-watch collector, when the job runs for the first time and its collection "
        "is still empty, Penny records the price the page shows and tells the user once — a "
        "first observation is news."
    ),
    UNCHANGED_READING.case_id: (
        "In a price-watch collector, when the page's price has not changed since she last "
        "recorded it, Penny writes nothing and says nothing."
    ),
    MOVED_READING.case_id: (
        "In a price-watch collector, when the page's price has changed since she last recorded "
        "it, Penny replaces the price she was holding with the new one and tells the user "
        "exactly once."
    ),
    SAME_ITEM_DECOYS_READING.case_id: (
        "In a price-watch collector, when the page's asking price has changed since she last "
        "recorded it and the page states it unlabelled, in a price ladder of several other "
        "figures for the SAME item — a list price, a member price, a bundle price, a "
        "per-unit price, a financing figure, a protection-plan price — Penny replaces the "
        "price she was holding with what a buyer pays for one, right now, names no other "
        "figure, and tells the user exactly once."
    ),
}

_SPEC_LINK = f"[{_ITEM} specification sheet]({LISTING_URL}/spec)"
_LISTING_LINK = f"[{_ITEM} listing]({LISTING_URL})"
_HEAD = f"Title: {_ITEM} — handheld console | faux-market\n{LISTING_URL}\n\n"

# Generic, number-free boilerplate every page below shares — real-listing furniture (a full
# spec table, a Q&A block, a returns paragraph, box contents, shipping options, a seller blurb,
# an accessibility note, a footer) that carries no dollar figure, so splicing it into any page
# below adds length and realism without touching that page's own decoys or its DATUM span.
# Shared across READINGS and MESSY_READINGS — one copy, not a near-duplicate per page set
# (#2366: the ruling that the first three cases' pages get real length too).
_FULL_SPECS = (
    "7-inch 1200p display with 512 GB of storage\n"
    "About four hours of battery per charge; the cell is replaceable\n"
    "Weight 640g · Original box and charger · One controller included\n"
    "Bluetooth 5.2 and a USB-C port for charging and wired audio\n"
    "Runs the open handheld OS, version 14\n"
    "Supports external storage up to 1 TB via its card slot\n"
    "Available in slate grey or sandstone\n"
    "Comes with a one-year manufacturer warranty, extendable at checkout\n"
)

_QA_BLOCK = (
    "Q: Does it come with a case? A: A soft pouch is included; a hard case is sold "
    "separately.\n"
    "Q: Is the storage expandable? A: Yes, via the card slot, up to 1 TB.\n"
    "Q: Does it work outside this region? A: The charger is dual-voltage; the plug may need "
    "an adapter.\n"
    "Q: Can I pick this up locally? A: No, this storefront ships only, no local pickup.\n"
)

_RETURN_POLICY = (
    "Returns are accepted within thirty days of delivery; return postage is the buyer's "
    "unless the item arrived faulty. Exchanges are processed once the returned unit is "
    "received and inspected. A protection plan purchased with this order is refunded in full "
    "if the unit is returned within the same window. Orders placed after the daily cutoff "
    "ship the next business day. This listing is provided by an independent seller on a "
    "fictional marketplace platform and is not sold or fulfilled by the platform itself.\n"
)

_BOX_CONTENTS = (
    "What's in the box\n"
    "One handheld console · One USB-C charging cable · One controller · A quick-start guide\n"
    "The protection plan, if purchased, is applied automatically at checkout and needs no "
    "separate registration.\n"
)

_SHIPPING_OPTIONS = (
    "Shipping options\n"
    "Standard delivery is tracked door-to-door; expedited delivery is available at checkout "
    "for an additional fee.\n"
    "Ships from a fictional warehouse in a plain, unbranded box.\n"
)

_COMPARE_BLOCK = (
    "Compare similar listings\n"
    "Buyers who viewed this also looked at two other open-box handhelds from the same "
    "storefront, each listed separately with its own asking price and condition notes.\n"
)

_SELLER_INFO_BLOCK = (
    "Seller information\n"
    "This storefront has operated on the marketplace for several years and ships from a "
    "single fictional warehouse. Customer messages are typically answered within one "
    "business day.\n"
)

_ACCESSIBILITY_NOTE = (
    "This listing supports screen readers; alt text is provided for the product image.\n"
)

_PAGE_FOOTER = (
    "Help · Track your order · Returns · Gift cards · Seller policies\n"
    "Prices and availability are accurate as of the date shown and may change.\n"
)


# Five catalogue-grade pages: same url, same product, same price, five voices.  Each carries
# far more than the job needs — a seller blurb, a specification block, neighbouring items with
# their OWN prices, housekeeping notes, and (below, #2366) the shared boilerplate above — because
# a real page does, and because a page thin enough to answer only the asked question cannot tell
# restraint from luck.  The neighbours are the load-bearing part: a cycle that grabs a price is
# only demonstrably reading the RIGHT one where wrong ones are on the page to grab.  Every
# markdown link sits at the CENTRE of its block, since a search-shaped read is trimmed to ±2
# lines around each solo link.  Each page is several kilobytes, real-listing length rather than
# the ~1 KB a page here once was — the ruling that these three cases' pages get real length too,
# not just the same-item-decoy case's.
READINGS = (
    Reading(
        "current-price",
        "the current price",
        _HEAD + f"{_ITEM} (open box, tested). Sold by a fictional reseller; stock moves weekly.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: nebula_resale (4.9 stars). The asking price last changed nine days ago.\n"
        "Condition: open box, tested. Dispatched within two working days.\n"
        "\n"
        "Specification\n"
        "7-inch 1200p display with 512 GB of storage\n"
        f"{_SPEC_LINK}\n"
        "About four hours of battery per charge; the cell is replaceable.\n"
        "Weight 640g · Original box and charger · One controller included\n"
        "\n"
        "Others from this seller\n"
        "Aurora Deck 1, refurbished — $329\n"
        "[Aurora Deck 1](https://faux-market.example/aurora-deck-1)\n"
        "Nebula Dock, charging stand — $59\n"
        "Both ship from the same fictional warehouse.\n"
        "\n"
        "Returns are accepted within thirty days; return postage is the buyer's.\n"
        "\n"
        "Full specifications\n"
        f"{_FULL_SPECS}"
        "\n"
        f"{_BOX_CONTENTS}"
        "\n"
        f"{_SHIPPING_OPTIONS}"
        "\n"
        "Questions & answers\n"
        f"{_QA_BLOCK}"
        "\n"
        f"{_COMPARE_BLOCK}"
        "\n"
        f"{_SELLER_INFO_BLOCK}"
        "\n"
        f"{_RETURN_POLICY}"
        "\n"
        f"{_ACCESSIBILITY_NOTE}"
        "\n"
        f"{_PAGE_FOOTER}",
    ),
    Reading(
        "asking-price",
        "the asking price on the page",
        _HEAD
        + "A fictional marketplace listing. The shop reprices this one whenever the shelf moves.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Offered by driftwood_games (4.8 stars). The asking price is reviewed every Monday.\n"
        "Condition: open box, tested. Two working days to dispatch.\n"
        "\n"
        "What you get\n"
        f"{_ITEM} handheld · 7-inch 1200p display · 512 GB\n"
        f"{_SPEC_LINK}\n"
        "The battery runs about four hours; charger and one controller are in the box.\n"
        "Weight 640g · Original packaging · No trade-ins accepted on this listing\n"
        "\n"
        "Also listed by this shop\n"
        "Aurora Deck Lite, boxed — $279\n"
        "[Aurora Deck Lite](https://faux-market.example/aurora-deck-lite)\n"
        "Spare stylus, twin pack — $19\n"
        "Everything ships from one fictional depot.\n"
        "\n"
        "Thirty-day returns. The buyer pays return postage.\n"
        "\n"
        "Full specifications\n"
        f"{_FULL_SPECS}"
        "\n"
        f"{_BOX_CONTENTS}"
        "\n"
        f"{_SHIPPING_OPTIONS}"
        "\n"
        "Questions & answers\n"
        f"{_QA_BLOCK}"
        "\n"
        f"{_COMPARE_BLOCK}"
        "\n"
        f"{_SELLER_INFO_BLOCK}"
        "\n"
        f"{_RETURN_POLICY}"
        "\n"
        f"{_ACCESSIBILITY_NOTE}"
        "\n"
        f"{_PAGE_FOOTER}",
    ),
    Reading(
        "asking-right-now",
        "what the listing is asking right now",
        _HEAD + "Listed by a fictional trade-in counter that re-prices its shelf every few days.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: quayside_swap (4.7 stars). What the counter asks moves with what it pays.\n"
        "Condition: open box, tested. Ready to dispatch inside two working days.\n"
        "\n"
        "Hardware\n"
        "A 7-inch 1200p screen and 512 GB of onboard storage\n"
        f"{_SPEC_LINK}\n"
        "Around four hours of play per charge; a workshop can swap the battery.\n"
        "Weight 640g · Boxed as it came · Charger included\n"
        "\n"
        "On the same shelf\n"
        "Aurora Deck 1, tested — $315\n"
        "[Aurora Deck 1](https://faux-market.example/aurora-deck-1)\n"
        "Moulded travel case — $44\n"
        "The counter posts its shelf every morning.\n"
        "\n"
        "Returns within thirty days, postage paid by the buyer.\n"
        "\n"
        "Full specifications\n"
        f"{_FULL_SPECS}"
        "\n"
        f"{_BOX_CONTENTS}"
        "\n"
        f"{_SHIPPING_OPTIONS}"
        "\n"
        "Questions & answers\n"
        f"{_QA_BLOCK}"
        "\n"
        f"{_COMPARE_BLOCK}"
        "\n"
        f"{_SELLER_INFO_BLOCK}"
        "\n"
        f"{_RETURN_POLICY}"
        "\n"
        f"{_ACCESSIBILITY_NOTE}"
        "\n"
        f"{_PAGE_FOOTER}",
    ),
    Reading(
        "listed-today",
        "the price it is listed at today",
        _HEAD + "A fictional clearance warehouse. Today's shelf, republished each morning.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: pier_clearance (4.6 stars). The asking price is set each day at opening.\n"
        "Condition: open box, tested. Two working days from order to dispatch.\n"
        "\n"
        "Specification\n"
        "Display 7-inch 1200p · Storage 512 GB\n"
        f"{_SPEC_LINK}\n"
        "Roughly four hours of battery; the pack is a serviceable part.\n"
        "Weight 640g · Original box and charger · One controller\n"
        "\n"
        "Today's other handhelds\n"
        "Aurora Deck Lite, open box — $265\n"
        "[Aurora Deck Lite](https://faux-market.example/aurora-deck-lite)\n"
        "Screen guard, twin pack — $12\n"
        "The whole shelf ships from one fictional warehouse.\n"
        "\n"
        "Returns accepted for thirty days; the buyer covers return postage.\n"
        "\n"
        "Full specifications\n"
        f"{_FULL_SPECS}"
        "\n"
        f"{_BOX_CONTENTS}"
        "\n"
        f"{_SHIPPING_OPTIONS}"
        "\n"
        "Questions & answers\n"
        f"{_QA_BLOCK}"
        "\n"
        f"{_COMPARE_BLOCK}"
        "\n"
        f"{_SELLER_INFO_BLOCK}"
        "\n"
        f"{_RETURN_POLICY}"
        "\n"
        f"{_ACCESSIBILITY_NOTE}"
        "\n"
        f"{_PAGE_FOOTER}",
    ),
    Reading(
        "listing-shows",
        "the price this listing shows",
        _HEAD
        + "A fictional consignment listing. The page shows whatever the owner last agreed to.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: lantern_consign (4.9 stars). The asking price is whatever this page shows.\n"
        "Condition: open box, tested. Dispatch within two working days.\n"
        "\n"
        "Specification sheet\n"
        "7-inch 1200p display · 512 GB storage · 640g\n"
        f"{_SPEC_LINK}\n"
        "The battery lasts about four hours and is replaceable at a workshop.\n"
        "Charger, one controller and the original box are included.\n"
        "\n"
        "Elsewhere in this consignment\n"
        "Aurora Deck 1, boxed — $340\n"
        "[Aurora Deck 1](https://faux-market.example/aurora-deck-1)\n"
        "Dock and cable set — $69\n"
        "All consigned stock ships from one fictional store.\n"
        "\n"
        "Thirty days to return; return postage is the buyer's.\n"
        "\n"
        "Full specifications\n"
        f"{_FULL_SPECS}"
        "\n"
        f"{_BOX_CONTENTS}"
        "\n"
        f"{_SHIPPING_OPTIONS}"
        "\n"
        "Questions & answers\n"
        f"{_QA_BLOCK}"
        "\n"
        f"{_COMPARE_BLOCK}"
        "\n"
        f"{_SELLER_INFO_BLOCK}"
        "\n"
        f"{_RETURN_POLICY}"
        "\n"
        f"{_ACCESSIBILITY_NOTE}"
        "\n"
        f"{_PAGE_FOOTER}",
    ),
)


# ── same-item decoys: a real-length, messier page for the same listing (#2366) ──
#
# The five pages above are already bigger than a "thin" fixture, but their only hazard is
# OTHER PRODUCTS' prices — never a second reading of the SAME item.  This page adds the
# hazard #2366 asks for: several prices that describe the item the job is watching, in
# different registers (list, member, bundle, per-unit, shipping, financing, a protection
# plan), on a page several kilobytes long — an order of magnitude closer to a real listing
# than the pages above.
#
# Round 1 (ordinary placement) and round 2 (the hardest decoys moved beside a ``Current
# price:`` label) both read 100% on checks and 0.000 variance, just at rising cost — because
# the world still handed the model an unambiguous anchor to match.  Round 3 removes that
# anchor: no line on this page says "current price" or "asking price".  The price a buyer
# pays for one, right now, is a SALE line in the register real listings actually use —
# ``Sale $449 (was $499)`` — read off the price ladder, not matched against a label.  The
# "was" figure is deliberately the case's own BASELINE ($499, what the collection already
# holds), because that is the realistic shape of a listing whose price just dropped, and it
# is the harder trap: a reading that latches onto "was" rather than "Sale" reproduces as
# "price unchanged," which is a different wrong answer than a decoy amount landing in the
# store.
#
# Every decoy amount below is a CONSTANT VALUE, reused byte-identical across all five arms —
# like ``_SPEC_LINK`` above — so the exclusion claim can name a token and mean it on every
# arm.  Every decoy is mutually exclusive, as a bare NUMERAL, from the baseline (499), the
# moved reading (449), and every other decoy: none is a substring of another, for the same
# reason ``_BASELINE_AMOUNT``/``_MOVED_AMOUNT`` are — a claim matching one must never
# accidentally match a different one.
_LIST_AMOUNT = "579"
_MEMBER_AMOUNT = "539"
_BUNDLE_AMOUNT = "899"
_PER_UNIT_AMOUNT = "433"
_SHIPPING_AMOUNT = "15"
_FOUR_PAY_AMOUNT = "112"
_WARRANTY_AMOUNT = "69"
_OTHER_ITEM_1_AMOUNT = "329"
_OTHER_ITEM_2_AMOUNT = "59"
_REVIEW_AMOUNT = "520"
_FLASH_SALE_AMOUNT = "489"
_PRICE_MATCH_AMOUNT = "619"

# Every decoy this case's exclusion claim checks for — the whole hazard, named once.  The
# baseline (499) and the moved amount (449) are NOT here: they are the tracked value and its
# prior reading, covered by ``_holds_the_moved_price_only`` below, not a decoy to exclude.
_SAME_ITEM_DECOY_AMOUNTS = (
    _LIST_AMOUNT,
    _MEMBER_AMOUNT,
    _BUNDLE_AMOUNT,
    _PER_UNIT_AMOUNT,
    _SHIPPING_AMOUNT,
    _FOUR_PAY_AMOUNT,
    _WARRANTY_AMOUNT,
    _OTHER_ITEM_1_AMOUNT,
    _OTHER_ITEM_2_AMOUNT,
    _REVIEW_AMOUNT,
    _FLASH_SALE_AMOUNT,
    _PRICE_MATCH_AMOUNT,
)

# Lines reused byte-identical across every arm — the price ladder, the decoys, a spec list, a
# Q&A block, a returns paragraph and a footer — none of which varies with the arm's wording.
# Only the voice around them (below) differs per arm, the same way ``_SPEC_LINK`` above is
# constant while the surrounding prose varies.  ``_SALE_LINE`` is the unlabelled line a correct
# reading must come from; everything else below it is a decoy in a different register,
# clustered right beside it rather than spread down the page — round 2's finding carried
# forward rather than undone.
_SALE_LINE = f"Sale {_MOVED_PRICE} (was {_BASELINE_PRICE})"
_FLASH_SALE_LINE = f"Today only: ${_FLASH_SALE_AMOUNT} for members who clip the coupon at checkout"
_LIST_LINE = f"List price: ${_LIST_AMOUNT}"
_MEMBER_LINE = f"Members: ${_MEMBER_AMOUNT}"
_PER_UNIT_LINE = f"${_PER_UNIT_AMOUNT} / unit (case of 3)"
_BUNDLE_LINE = f"Bundle of 3: ${_BUNDLE_AMOUNT}"
_FOUR_PAY_LINE = f"Or 4 payments of ${_FOUR_PAY_AMOUNT}"
_SHIPPING_LINE = f"Shipping: ${_SHIPPING_AMOUNT}, arrives in three to five business days"
_WARRANTY_LINE = f"Add a two-year protection plan: ${_WARRANTY_AMOUNT}"
_OTHER_ITEM_1_LINE = f"Aurora Deck 1, refurbished — ${_OTHER_ITEM_1_AMOUNT}"
_OTHER_ITEM_2_LINE = f"Nebula Dock, charging stand — ${_OTHER_ITEM_2_AMOUNT}"
_REVIEW_ONE = (
    f'"Paid ${_REVIEW_AMOUNT} for mine last spring and have zero regrets — still going '
    'strong." — a verified buyer\n'
)
_PRICE_MATCH_LINE = (
    f"Price-match guarantee: up to ${_PRICE_MATCH_AMOUNT} back as store credit if you find "
    "it cheaper elsewhere within 14 days"
)


class _MessyVoice(NamedTuple):
    """One arm's wording and the handful of sentences that make its page read differently
    from its siblings' — everything else on the page (the decoys, the specs, the Q&A, the
    policy, the footer) is the constant text above."""

    name: str
    extract: str
    intro: str
    seller: str
    condition: str
    spec_header: str
    review_two: str
    review_three: str
    closing: str


def _messy_body(voice: _MessyVoice) -> str:
    """One arm's whole page: the constant price ladder, decoys and boilerplate above, wrapped
    in this arm's own intro, seller line, condition, spec header, two further reviews and
    closing line.

    No line on this page is labelled "current price" or "asking price" (#2366 round 3) — the
    price ladder immediately after the intro is six lines in a row, each a different register
    real listings actually use, with nothing to match against but the shape of the register
    itself.  ``_SALE_LINE`` is the one a correct reading comes from; every other line in the
    ladder is a decoy, clustered right beside it rather than spread down the page — round 2's
    finding carried forward rather than undone."""
    return (
        _HEAD
        + f"{voice.intro}\n"
        + f"{_SALE_LINE}\n"
        + f"{_FLASH_SALE_LINE}\n"
        + f"{_LIST_LINE}\n"
        + f"{_MEMBER_LINE}\n"
        + f"{_PER_UNIT_LINE}\n"
        + f"{_BUNDLE_LINE}\n"
        + f"{_FOUR_PAY_LINE}\n"
        + f"{_LISTING_LINK}\n"
        + f"{voice.seller}\n"
        + f"{voice.condition}\n"
        "\n"
        "Delivery & payment\n"
        f"{_SHIPPING_LINE}\n"
        "\n"
        "Protect your purchase\n"
        f"{_WARRANTY_LINE}\n"
        "\n"
        f"{voice.spec_header}\n"
        f"{_FULL_SPECS}"
        f"{_SPEC_LINK}\n"
        "\n"
        "Frequently bought together\n"
        f"{_OTHER_ITEM_1_LINE}\n"
        "[Aurora Deck 1](https://faux-market.example/aurora-deck-1)\n"
        f"{_OTHER_ITEM_2_LINE}\n"
        "Both ship from the same fictional warehouse.\n"
        "\n"
        f"{_BOX_CONTENTS}"
        "\n"
        f"{_SHIPPING_OPTIONS}"
        "\n"
        "Customer reviews\n"
        f"{_REVIEW_ONE}"
        f"{voice.review_two}"
        f"{voice.review_three}"
        "\n"
        "Questions & answers\n"
        f"{_QA_BLOCK}"
        "\n"
        f"{_COMPARE_BLOCK}"
        "\n"
        f"{_SELLER_INFO_BLOCK}"
        "\n"
        f"{_RETURN_POLICY}"
        f"{_PRICE_MATCH_LINE}\n"
        "\n"
        f"{_ACCESSIBILITY_NOTE}"
        "\n"
        f"{voice.closing}\n"
        f"{_PAGE_FOOTER}"
    )


# Five voices over the one messy page — same five extract wordings as READINGS above, since
# they are already natural phrasings of "the price the job watches"; what is new is the page
# each pairs with.
_MESSY_VOICES = (
    _MessyVoice(
        name="current-price-messy",
        extract="the current price",
        intro="A fictional marketplace listing for a handheld console, open box and tested.",
        seller="Seller: nebula_resale (4.9 stars). The asking price last changed nine days ago.",
        condition="Condition: open box, tested. Dispatched within two working days.",
        spec_header="Specification",
        review_two=(
            '"Battery life is about what they claim, maybe a touch less." — a verified buyer\n'
        ),
        review_three='"Arrived well packaged, screen was spotless." — a verified buyer\n',
        closing=(
            "This listing is part of the nebula_resale storefront; browse their other gear "
            "from the seller page."
        ),
    ),
    _MessyVoice(
        name="asking-price-messy",
        extract="the asking price on the page",
        intro=(
            "A fictional marketplace listing. The shop reprices this one whenever the shelf moves."
        ),
        seller="Offered by driftwood_games (4.8 stars). The asking price is reviewed every Monday.",
        condition="Condition: open box, tested. Two working days to dispatch.",
        spec_header="What you get",
        review_two='"Controller feels great, a bit heavier than I expected." — a verified buyer\n',
        review_three='"Exactly as described, fast shipping." — a verified buyer\n',
        closing="driftwood_games ships everything from one fictional depot.",
    ),
    _MessyVoice(
        name="asking-right-now-messy",
        extract="what the listing is asking right now",
        intro="Listed by a fictional trade-in counter that re-prices its shelf every few days.",
        seller="Seller: quayside_swap (4.7 stars). What the counter asks moves with what it pays.",
        condition="Condition: open box, tested. Ready to dispatch inside two working days.",
        spec_header="Hardware",
        review_two='"Good value for an open-box unit, no complaints." — a verified buyer\n',
        review_three=(
            '"Screen has a couple of tiny scuffs but nothing you notice in use." — a verified '
            "buyer\n"
        ),
        closing="quayside_swap posts its shelf every morning.",
    ),
    _MessyVoice(
        name="listed-today-messy",
        extract="the price it is listed at today",
        intro="A fictional clearance warehouse. Today's shelf, republished each morning.",
        seller="Seller: pier_clearance (4.6 stars). The asking price is set each day at opening.",
        condition="Condition: open box, tested. Two working days from order to dispatch.",
        spec_header="Specification",
        review_two='"Bought as a backup unit, works perfectly." — a verified buyer\n',
        review_three=('"Box was a bit beat up but the unit itself was fine." — a verified buyer\n'),
        closing="The whole shelf ships from one fictional warehouse.",
    ),
    _MessyVoice(
        name="listing-shows-messy",
        extract="the price this listing shows",
        intro="A fictional consignment listing. The page shows whatever the owner last agreed to.",
        seller="Seller: lantern_consign (4.9 stars). The asking price is whatever this page shows.",
        condition="Condition: open box, tested. Dispatch within two working days.",
        spec_header="Specification sheet",
        review_two='"Seller was responsive and the handoff was smooth." — a verified buyer\n',
        review_three='"Works like new, battery holds a charge fine." — a verified buyer\n',
        closing="All consigned stock ships from one fictional store.",
    ),
)

MESSY_READINGS = tuple(
    Reading(voice.name, voice.extract, _messy_body(voice)) for voice in _MESSY_VOICES
)


def _page(case: WatchCase, reading: Reading) -> CannedPage:
    """The page THIS case's cycle reads, on THIS arm's prose.

    The moved variant is derived from the arm's own page by the shared single-span ``datum``
    edit rather than rebuilt from a template, so the two sides of the pair are one text and one
    span: the edit RAISES unless the watched line appears exactly once, which is what a rebuilt
    twin cannot check."""
    page = CannedPage(match=_MATCH, text=reading.body)
    return page if case.shows == _BASELINE_PRICE else datum(page, _DATUM, _MOVED_DATUM)


def _messy_page(case: WatchCase, reading: Reading) -> CannedPage:
    """The same-item-decoy case's page (#2366 round 3): no shared ``datum()`` edit, because
    this page carries no labelled span to edit — the asking price is a READING off an
    unlabelled sale line, not a match on a ``Current price:`` anchor.  ``MESSY_READINGS``
    bodies already carry the final moved-reading values (this case's only entry condition),
    so the page is used as authored; ``case`` is accepted only to keep the same shape as
    ``_page`` for ``_arm``'s ``page_builder`` parameter."""
    del case
    return CannedPage(match=_MATCH, text=reading.body)


def _world(
    case: WatchCase,
    reading: Reading,
    page_builder: Callable[[WatchCase, Reading], CannedPage] = _page,
) -> World:
    """This arm's ground: the page its single cycle reads.

    ``keeps``/``excludes`` are EMPTY, and deliberately.  Those token sets back
    ``assert_something_from_each_page_was_written``, which these cases never call — and there is
    nothing they could usefully hold: a ``keeps`` token has to appear on ONE page so a stored
    copy says which page it came from, and every arm here reads the same url, the same product
    and the same price.  Declaring tokens anyway would print "5 must-keep" in every report as
    though something had verified them.  A contract nothing reads is worse than no contract: it
    reads as a check that passed."""
    return World(
        name=reading.name,
        pages=(page_builder(case, reading),),
        keeps=(),
        excludes=(),
    )


def _prompt(reading: Reading) -> str:
    """The job's free-text instructions (#2455 naive baseline) — what a job's own
    ``extraction_prompt`` carries verbatim, with no skill/program rendering seam.

    ONE shape for every arm and every case — the same page, the same destination.  The
    one thing that differs is the EXTRACT phrase, which is what the test asserts moved —
    the same axis the deleted skill fixture varied through a substitution's description."""
    return (
        f"Read {LISTING_URL} and note {reading.extract}. If it's different from what "
        f"'{_CONTAINER}' already holds under the key '{_KEY}', update it there."
    )


def _seeder(case: WatchCase, reading: Reading):
    """The world this case's cycle starts in: a job already configured with its prompt,
    and the ENTRY CONDITION — then every claim that world makes, asserted out loud.

    The entry condition goes down through the store's own write path, under the key the
    prompt writes to, so the row carries the stamps and the embeddings a real cycle's
    write would have left.  Hand-inserting it would give the write gate something to
    compare against that no cycle could have produced, and the case would then measure
    the gate against a shape production never sees."""

    def seed(db: Database) -> None:
        db.memories.create_collection(
            _CONTAINER,
            _CONTAINER_DESCRIPTION,
            extraction_prompt=_prompt(reading),
            schedule=_SCHEDULE,
            notify=True,
        )
        if case.stored is not None:
            memory = db.memory(_CONTAINER)
            assert memory is not None, f"the job's container {_CONTAINER!r} must exist to seed"
            memory.write(
                [EntryInput(key=_KEY, content=case.stored)],
                author=_SEED_AUTHOR,
                run_id=_SEED_RUN,
            )
        _assert_the_watch_world(db, case, reading)

    return seed


def _assert_the_watch_world(db: Database, case: WatchCase, reading: Reading) -> None:
    """Everything the seeder is responsible for, asserted where it fails loudly."""
    row = db.memories.get(_CONTAINER)
    assert row is not None, f"the job's container {_CONTAINER!r} must exist"
    assert row.notify, "these cases measure the notification, so the job must be notifying"

    prompt = row.extraction_prompt or ""
    assert LISTING_URL in prompt, (
        f"the prompt must name {LISTING_URL!r} — a cycle can only fetch the page "
        f"something it reads names.  Prompt: {prompt!r}"
    )
    assert f"'{_CONTAINER}'" in prompt, (
        f"the prompt must name its own destination, {_CONTAINER!r}.  Prompt: {prompt!r}"
    )
    assert reading.extract in prompt, (
        f"this arm's instruction must reach the model as {reading.extract!r} — the "
        f"extract phrase is the whole arm axis.  Prompt: {prompt!r}"
    )
    _assert_the_entry_condition(db, case)


def _assert_the_entry_condition(db: Database, case: WatchCase) -> None:
    """The state the cycle starts from, which is what selects the behaviour being measured.

    An empty collection makes a first observation news; a collection already holding the
    reading is the only state the ``KEY_EXISTS_UNCHANGED`` STOP can fire against.  Getting this
    wrong silently turns one case into another — the quiet case run against an empty container
    would write, pass nothing, and look like a model failure — so it is asserted here rather
    than assumed."""
    held = collection_entries(db, _CONTAINER)
    if case.stored is None:
        assert not held, (
            "the container must be empty when the cycle starts, so a write is exactly a new "
            f"entry rather than a diff a claim has to compute — it holds {held}"
        )
        return
    assert held == {_KEY: case.stored}, (
        f"the cycle must start from exactly {{{_KEY!r}: {case.stored!r}}} — the write gate "
        f"compares a candidate against what is stored under the same key, and it holds {held}"
    )


def _arm(
    case: WatchCase,
    reading: Reading,
    page_builder: Callable[[WatchCase, Reading], CannedPage] = _page,
) -> CycleArm:
    """One arm of one case: this case's entry condition and page, under this arm's instruction.

    ``text`` is the instruction, because that is what makes this arm this arm."""
    return CycleArm(
        text=reading.extract,
        seed=_seeder(case, reading),
        pages=[page_builder(case, reading)],
        world=_world(case, reading, page_builder),
    )


def _arms(
    case: WatchCase,
    readings: Sequence[Reading] = READINGS,
    page_builder: Callable[[WatchCase, Reading], CannedPage] = _page,
) -> list[CycleArm]:
    """This case's five arms — five wordings of one instruction over five prose variants.

    ``readings`` defaults to the cross-item-decoy pages shared by the first three cases;
    the same-item-decoy case (#2366) passes ``MESSY_READINGS`` and ``_messy_page`` instead —
    the page set and how its page is built are the only things that differ."""
    return [_arm(case, reading, page_builder) for reading in readings]


# ── The claims ───────────────────────────────────────────────────────────────
#
# Every one of them reads END STATE: what the collection holds, what the run record says, and
# what reached the send queue.  Nothing reads a tool name or an ordering — a skill is an
# arbitrary tool sequence, so a route is measured in section B and never asserted.


def _held_text(sample: SampleObservation) -> str:
    """The collection as the cycle left it, key and content together.

    The WHOLE entry, because a fact in the key and a blurb in the body is a perfectly good way
    to store it, and a content-only read once reported a 25/32 model failure that was entirely
    its own bug."""
    return " ".join(entry.text for entry in sorted(sample.held, key=_held_key))


def _held_key(entry: StoredEntry) -> str:
    """One held entry's key, for a stable reading order.  A collection entry always has one;
    the empty string is the keyless-log shape the type allows and this collection never has."""
    return entry.key or ""


def _one_entry(sample: SampleObservation, _world: World) -> Answer:
    """The watch keeps ONE fact.  A watch that appends grows without bound and the user is
    told about a value they already have."""
    keys = sorted(_held_key(entry) for entry in sample.held)
    return len(sample.held) == 1, f"holds {len(sample.held)} entries: {keys}"


def _holds(amount: str):
    """A claim that the collection holds one named reading.

    The facts are the same on every arm, so the claim can name the value rather than asserting
    that something was stored: a shape claim cannot tell a watch that read the right price from
    one that produced a plausible number.

    Matched on the bare AMOUNT.  The identity of a reading is its number — that is what moves on
    the page, and what a watch exists to track — while which currency notation a draw happened to
    keep is not a question this claim answers.  ``449`` and ``499`` are mutually exclusive as
    numerals, so naming the amount costs the claim nothing it was measuring."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        held = _held_text(sample)
        return amount in held, f"expected {amount}; the collection holds {held!r}"

    return answer


def _holds_the_moved_price_only(sample: SampleObservation, _world: World) -> Answer:
    """The collection holds the reading the page moved TO, and no longer the one it moved FROM.

    Both halves are needed and neither is sufficient — a watch that appended the new reading
    beside the old one holds ``449`` while still telling the user something that is no longer
    true, and a watch that never re-read the page holds neither.

    Matched on the bare AMOUNTS, for the reason ``_holds`` gives: the identity of a reading is
    its number, and the notation a draw kept is not this claim's question."""
    held = _held_text(sample)
    ok = _MOVED_AMOUNT in held and _BASELINE_AMOUNT not in held
    return ok, (
        f"expected {_MOVED_AMOUNT} and not {_BASELINE_AMOUNT}; the collection holds {held!r}"
    )


def _told_once(sample: SampleObservation, _world: World) -> Answer:
    """The user was told, and told once.

    Read over the SEND QUEUE, which is what the user will actually receive — a cycle enqueues
    and the drainer is a separate schedule, so a pending-only read of outgoing messages reports
    a delivered notification as silence."""
    count = len(sample.notifications)
    return count == 1, f"{count} messages reached the send queue: {sample.notifications}"


def _told_nothing(sample: SampleObservation, _world: World) -> Answer:
    """Nothing reached the send queue.  No news is not a message saying there is no news."""
    return not sample.notifications, f"sent {sample.notifications}"


def _excludes_the_same_item_decoys(sample: SampleObservation, _world: World) -> Answer:
    """None of the SAME-ITEM price variants — list, member, bundle, per-unit, shipping,
    financing, warranty — nor the two cross-item neighbours nor the review's own figure ever
    reaches the store (#2366).

    The generic provenance claim cannot catch a wrong-price grab here: a decoy is still a
    real token ON the page, so it would read as "sourced" even though it is not the price the
    job exists to track.  Naming the decoy amounts and asserting their absence is what
    actually tests whether the cycle picked the right one."""
    held = _held_text(sample)
    present = [amount for amount in _SAME_ITEM_DECOY_AMOUNTS if amount in held]
    return not present, f"the store holds decoy amount(s) {present}: {held!r}"


# ── first reading: an empty collection, and a page ───────────────────────────


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_watch_writes_the_first_reading(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A collection arrives from apply EMPTY, so its first observation is a new key — a
    baseline, and a first observation is news."""
    cohort = await collector_cycles_eval(
        case_id=FIRST_READING.case_id,
        behaviour=_BEHAVIOUR[FIRST_READING.case_id],
        area=Area.BACKGROUND_COLLECTORS,
        model=model,
        collection=_CONTAINER,
        arms=_arms(FIRST_READING),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE
    cohort.claim("state: the collection holds one entry", _one_entry, SpecCategory.STORE)
    cohort.claim(
        f"state: the entry it holds carries {_BASELINE_AMOUNT}",
        _holds(_BASELINE_AMOUNT),
        SpecCategory.STORE,
    )
    cohort.claim("state: the user was told once", _told_once, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


# ── unchanged reading: the collection already holds what the page says ───────


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_watch_stays_quiet_when_the_reading_has_not_moved(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """The same reading twice over is silence — no write, and nothing entering a notification.

    REPLY_SPREAD is not measured here, and that is the point of the case: a correct cohort
    sends nothing, so a reply-spread reading would be blind on every sample by construction and
    would print a number where there is no measurement.  A sample that DID speak is caught by
    the send-queue claim and shows in ``transitions`` as ``quiet+told``."""
    cohort = await collector_cycles_eval(
        case_id=UNCHANGED_READING.case_id,
        behaviour=_BEHAVIOUR[UNCHANGED_READING.case_id],
        area=Area.BACKGROUND_COLLECTORS,
        model=model,
        collection=_CONTAINER,
        arms=_arms(UNCHANGED_READING),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )

    # STORE
    cohort.claim("state: the collection still holds one entry", _one_entry, SpecCategory.STORE)
    cohort.claim(
        f"state: the entry it holds still carries {_BASELINE_AMOUNT}",
        _holds(_BASELINE_AMOUNT),
        SpecCategory.STORE,
    )
    cohort.claim("state: nothing reached the send queue", _told_nothing, SpecCategory.STORE)

    # PROVENANCE — a correct cycle wrote nothing, so this claim is TRUE of it rather than
    # unasked; what it catches is a quiet cycle that wrote something the page never said.
    cohort.assert_every_value_in_the_store_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED)


# ── moved reading: the collection holds the old price, the page shows a new one ──


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_watch_writes_and_tells_when_the_reading_moves(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A moved reading is a write AND one notification, on the standing key."""
    cohort = await collector_cycles_eval(
        case_id=MOVED_READING.case_id,
        behaviour=_BEHAVIOUR[MOVED_READING.case_id],
        area=Area.BACKGROUND_COLLECTORS,
        model=model,
        collection=_CONTAINER,
        arms=_arms(MOVED_READING),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )

    # STORE
    cohort.claim(
        f"state: the collection holds {_MOVED_AMOUNT} and no longer {_BASELINE_AMOUNT}",
        _holds_the_moved_price_only,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the collection still holds one entry, rewritten", _one_entry, SpecCategory.STORE
    )
    cohort.claim("state: the user was told once", _told_once, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


# ── same-item decoys: a moved reading on a real-length, decoy-rich page (#2366) ──


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_watch_tells_the_asking_price_from_other_prices_on_the_same_listing(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """The same moved-reading behaviour as the case above, on a page several kilobytes long
    whose price is stated unlabelled, in a ladder of other prices of the SAME item — list,
    member, bundle, per-unit, financing, a protection plan — rather than other products'
    prices, and rather than a ``Current price:`` anchor to match against (#2366 round 3)."""
    cohort = await collector_cycles_eval(
        case_id=SAME_ITEM_DECOYS_READING.case_id,
        behaviour=_BEHAVIOUR[SAME_ITEM_DECOYS_READING.case_id],
        area=Area.BACKGROUND_COLLECTORS,
        model=model,
        collection=_CONTAINER,
        arms=_arms(SAME_ITEM_DECOYS_READING, MESSY_READINGS, page_builder=_messy_page),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )

    # STORE
    cohort.claim(
        f"state: the collection holds {_MOVED_AMOUNT} and no longer {_BASELINE_AMOUNT}",
        _holds_the_moved_price_only,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the collection still holds one entry, rewritten", _one_entry, SpecCategory.STORE
    )
    cohort.claim("state: the user was told once", _told_once, SpecCategory.STORE)
    cohort.claim(
        "state: none of the same-item or neighbouring decoy prices was stored",
        _excludes_the_same_item_decoys,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
