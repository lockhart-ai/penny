"""The delivery turn that writes a notifying job's message, in isolation: what it
makes of a finished cycle's report (#1911/#2363, ported to #2656's path).

No cycle runs — the driver seeds the job and the entries ONE finished cycle wrote,
enqueues the report row the way ``_notify_if_due`` does, and drives the real
delivery turn the send-queue drainer runs.  The ``job_report_eval`` driver mirrors
the browse extractor's shape: a fixed set of facts (the job's name and its cycle's
entries, held byte-identical across the arms) and five wordings of the one piece of
upstream free text the report carries — the collection's own ``description`` — the
same role the extractor's ``instruction``/``also_instructed`` play.

Two cases (identities kept from the composer they ported; the composer itself is
retired — the cases now measure the chat-agent delivery turn that replaced it):

* ``notify-composer-single-finding`` — the driver's own smoke test and cost baseline:
  one ordinary finding, nothing to summarise.
* ``notify-composer-many-findings`` — the adversarial case research 3/3 of the
  honest-failure meta (#2373) asked for: a cycle that wrote 18 real-sized findings of
  mixed relevance in one pass, across several marketplaces.  The composer's document
  capped the render at 8 and the claims named only those 8; the report the delivery
  turn reads carries the entries FULL and uncapped (#2656), so the claim strengthens
  to ALL 18 — every finding the cycle wrote must reach the user, not the 8 a budget
  happened to show.  (Claim change flagged for the code owner in the PR body.)

The marketplaces, brands and prices are real ones the model recognises (Reverb,
eBay, Craigslist, Amazon; Korg, Moog, Roland and the rest), per the code owner's
ruling of Oct 19 — the listings themselves are fictional, and the repo is public, so
nothing resembling a real person or a real live listing appears.

Every case is report-only; the thresholds are the code owner's.
"""

from __future__ import annotations

import pytest

from penny.database.memory import EntryInput
from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.notify_composer import JobReportEval
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import REPLY_SPREAD
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "notify-composer"

# ── notify-composer-many-findings ────────────────────────────────────────────────────────
# 18 findings from one cycle.  Under the composer these were capped at the first 8
# (``NOTIFY_WRITTEN_ENTRIES``) and the rest reduced to an honest "N more entries not
# shown" count; the report the delivery turn reads is UNCAPPED (#2656), so every one of
# the 18 is in front of the model and every one of them is claimed.
#
# Each finding's ANCHOR is its own item name — real content a friendly summary would
# naturally restate, never a separate id tag appended to the line: an appended tag reads to
# the model as an internal reference rather than a fact worth repeating (measured — the
# before-run's own transcripts show several samples correctly declining to echo a bare
# "zx01q-listing"-shaped tag while still naming the item in plain prose, which scored as a
# miss under the first anchor scheme even though nothing was dropped).
_MANY_FINDINGS_COLLECTION = "vintage-synth-watch"
_MANY_FINDINGS_MARKETS = ("Reverb", "eBay", "Craigslist")
_MANY_FINDINGS_CONTENT = (
    "a Korg MS-20 analog synth on Reverb, asking $410, described as lightly used",
    "a Moog Grandmother desktop module on eBay, listed new in box at $275",
    "a Korg gig bag sized for the MS-20 on Reverb, $40, not an instrument itself",
    "a Roland Juno-106 polysynth on Craigslist, price dropped from $700 to $610",
    "an Arturia MicroFreak semi-modular rig on eBay, $325, two units available",
    "an Akai Rhythm Wolf drum machine bundled free with a MicroFreak purchase on eBay, "
    "a different instrument category",
    "a Yamaha DX7 synth on Reverb, $540, auction ending soon",
    "a Novation Circuit groovebox on Craigslist, $190, seller offering a bundle discount",
    "a Doepfer eurorack case on eBay, $95",
    "an Elektron sequencer on Reverb, $220",
    "an Eowave ribbon controller on Craigslist, $60",
    "a Yamaha Reface synth on eBay, $480",
    "a Boss chorus pedal on Reverb, $35",
    "a Roland JX-3P synth on Craigslist, $390",
    "a Behringer patchbay on eBay, $70",
    "a Nord wavetable synth on Reverb, $455",
    "a Mackie mixer on Craigslist, $110",
    "a Teenage Engineering OP-Z percussion synth on eBay, $265",
)
_MANY_FINDINGS_ENTRIES = tuple(
    EntryInput(key=f"listing-{index:02d}", content=content)
    for index, content in enumerate(_MANY_FINDINGS_CONTENT, start=1)
)
# EVERY finding's item name — the report is uncapped now, so "every finding named" means
# every finding the cycle wrote, not the 8 a budget happened to render.
_MANY_FINDINGS_ANCHORS = (
    "Korg MS-20",
    "Moog Grandmother",
    "gig bag",
    "Juno-106",
    "MicroFreak",
    "Rhythm Wolf",
    "Yamaha DX7",
    "Novation Circuit",
    "eurorack",
    "Elektron",
    "ribbon controller",
    "Reface",
    "chorus pedal",
    "JX-3P",
    "patchbay",
    "Nord",
    "Mackie",
    "OP-Z",
)
_MANY_FINDINGS_WORLD = World(
    name="vintage-synth-watch-cycle",
    pages=(),
    keeps=(),
    excludes=(),
    answers=_MANY_FINDINGS_ANCHORS,
)
_MANY_FINDINGS_DESCRIPTION = (
    f"keeps an eye on {_MANY_FINDINGS_MARKETS[0]}, {_MANY_FINDINGS_MARKETS[1]}, and "
    f"{_MANY_FINDINGS_MARKETS[2]} for new or discounted vintage synthesizers"
)
_MANY_FINDINGS_ALSO_DESCRIBED = (
    f"watches {_MANY_FINDINGS_MARKETS[0]}, {_MANY_FINDINGS_MARKETS[1]}, and "
    f"{_MANY_FINDINGS_MARKETS[2]} and flags new synth listings or price cuts",
    f"checks the {_MANY_FINDINGS_MARKETS[0]}, {_MANY_FINDINGS_MARKETS[1]}, and "
    f"{_MANY_FINDINGS_MARKETS[2]} listings for anything synth-related that's new or cheaper",
    f"tracks vintage synthesizer listings and price drops across {_MANY_FINDINGS_MARKETS[0]}, "
    f"{_MANY_FINDINGS_MARKETS[1]}, and {_MANY_FINDINGS_MARKETS[2]}",
    f"scans {_MANY_FINDINGS_MARKETS[0]}, {_MANY_FINDINGS_MARKETS[1]}, and "
    f"{_MANY_FINDINGS_MARKETS[2]} for fresh synth listings and markdowns",
)

# ── notify-composer-single-finding ───────────────────────────────────────────────────────
# The anchor is the item's own name — real content, not an appended id tag (see the
# many-findings comment above for why).
_SINGLE_FINDING_COLLECTION = "garden-tool-watch"
_SINGLE_FINDING_ANCHOR = "Fiskars telescoping pruner"
_SINGLE_FINDING_ENTRIES = (
    EntryInput(
        key="listing-01",
        content=f"a {_SINGLE_FINDING_ANCHOR} on Amazon, listed new in box at $38",
    ),
)
_SINGLE_FINDING_WORLD = World(
    name="garden-tool-watch-cycle",
    pages=(),
    keeps=(),
    excludes=(),
    answers=(_SINGLE_FINDING_ANCHOR,),
)
_SINGLE_FINDING_DESCRIPTION = "keeps an eye on Amazon for new garden-tool listings"
_SINGLE_FINDING_ALSO_DESCRIBED = (
    "watches Amazon and flags new garden-tool listings",
    "checks the Amazon listings for anything garden-tool related that's new",
    "tracks new garden-tool listings on Amazon",
    "scans Amazon for fresh garden-tool listings",
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_notify_composer_names_one_ordinary_finding(
    job_report_eval: JobReportEval, model: str
) -> None:
    """The driver's own smoke test and cost baseline: one ordinary finding, nothing to
    summarise — what ``notify-composer-many-findings`` is read against."""
    cohort = await job_report_eval(
        case_id="notify-composer-single-finding",
        behaviour=(
            "In a finished job's delivery turn, when the cycle found one ordinary new "
            "entry, Penny's message names what the finding carries and invents nothing."
        ),
        area=Area.NOTIFICATIONS,
        model=model,
        collection=_SINGLE_FINDING_COLLECTION,
        entries=_SINGLE_FINDING_ENTRIES,
        world=_SINGLE_FINDING_WORLD,
        description=_SINGLE_FINDING_DESCRIPTION,
        also_described=_SINGLE_FINDING_ALSO_DESCRIBED,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=_FAMILY,
    )
    # STORE — the cycle's entries were seeded, and the delivery turn writes no store of
    # its own; the message it queues is the read.
    # PROVENANCE
    cohort.assert_the_reply_answers_the_ask()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_notify_composer_names_many_findings_at_once(
    job_report_eval: JobReportEval, model: str
) -> None:
    """A cycle that wrote 18 findings in one pass across three marketplaces, mixed
    relevance, real-sized text — the shape research 3/3 of the honest-failure meta
    (#2373) asked for.  The report the delivery turn reads carries every entry FULL
    and uncapped (#2656), so the claim is ALL 18 findings — the composer's document
    showed only the first 8 and the claim had to stop where the budget did."""
    cohort = await job_report_eval(
        case_id="notify-composer-many-findings",
        behaviour=(
            "In a finished job's delivery turn, when the cycle found many new entries at "
            "once, Penny's message names what the findings carry and invents nothing."
        ),
        area=Area.NOTIFICATIONS,
        model=model,
        collection=_MANY_FINDINGS_COLLECTION,
        entries=_MANY_FINDINGS_ENTRIES,
        world=_MANY_FINDINGS_WORLD,
        description=_MANY_FINDINGS_DESCRIPTION,
        also_described=_MANY_FINDINGS_ALSO_DESCRIBED,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=_FAMILY,
    )
    # STORE — the cycle's entries were seeded, and the delivery turn writes no store of
    # its own; the message it queues is the read.
    # PROVENANCE
    cohort.assert_the_reply_answers_the_ask()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(REPLY_SPREAD)
