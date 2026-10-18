"""The notify COMPOSER micro-context, in isolation: what it makes of a finished collector
cycle's assembled document (#1911/#2363).

No cycle runs and no store is seeded — the composer's whole input is one framework-rendered
document (``penny.notification.render_notification``), so a case is that one document.  The
``notify_composer_eval`` driver mirrors the browse extractor's shape: a fixed set of facts
(the cycle's own findings, held byte-identical across the arms) and five wordings of the one
piece of upstream free text the document carries — the collection's own ``description`` — the
same role the extractor's ``instruction``/``also_instructed`` play.

Two cases:

* ``notify-composer-single-finding`` — the driver's own smoke test and cost baseline: one
  ordinary finding, nothing to summarise.
* ``notify-composer-many-findings`` — the adversarial case research 3/3 of the honest-failure
  meta (#2373) asked for: a cycle that wrote 18 real-sized findings of mixed relevance in one
  pass, across several marketplaces, which is the shape ``render_notification`` actually
  renders bounded (#1934) — only the first ``PennyConstants.NOTIFY_WRITTEN_ENTRIES`` (8) are
  shown, the rest reduced to an honest "N more entries not shown" count.

All content is synthetic: invented marketplace names, invented items, invented prices —
nothing resembling a real person or listing.

Every case is report-only; the thresholds are the code owner's.
"""

from __future__ import annotations

import pytest

from penny.notification import NotificationInput, WrittenEntry
from penny.tests.eval.conftest import EVAL_MODELS, NotifyComposerEval
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import REPLY_SPREAD
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "notify-composer"

# ── notify-composer-many-findings ────────────────────────────────────────────────────────
# 18 findings from one cycle.  The document's own render caps written entries at
# NOTIFY_WRITTEN_ENTRIES (8, constants.py), so only the first 8 ever reach the model; the
# rest exist to produce the real overflow line a heavy cycle's document actually carries
# ("10 more entries not shown").  Two of the eight RENDERED findings (#3, #6) are
# deliberately borderline-relevant — true facts the cycle wrote down that are not quite "a
# synth" — to test that the composer reports them honestly rather than editorialising about
# relevance it was not asked to judge.
#
# Each finding's ANCHOR is its own item name — real content a friendly summary would
# naturally restate, never a separate id tag appended to the line: an appended tag reads to
# the model as an internal reference rather than a fact worth repeating (measured — the
# before-run's own transcripts show several samples correctly declining to echo a bare
# "zx01q-listing"-shaped tag while still naming the item in plain prose, which scored as a
# miss under the first anchor scheme even though nothing was dropped).
_MANY_FINDINGS_COLLECTION = "vintage-synth-watch"
_MANY_FINDINGS_MARKETS = ("Faraday Exchange", "Orbital Swap", "Nightjar Market")
_MANY_FINDINGS_CONTENT = (
    "a Korrigan-7 analog synth on Faraday Exchange, asking 410 credits, described as lightly used",
    "a Halcyon Drift desktop module on Orbital Swap, listed new in box at 275 credits",
    "a Korrigan Satchel carry case sized for the Korrigan series on Faraday Exchange, "
    "40 credits, not an instrument itself",
    "a Tamsin Eight polysynth on Nightjar Market, price dropped from 700 to 610 credits",
    "a Vesper Loop semi-modular rig on Orbital Swap, 325 credits, two units available",
    "a Fenlow Thumper drum machine bundled free with a Vesper Loop purchase on Orbital "
    "Swap, a different instrument category",
    "a Marrow FM synth on Faraday Exchange, 540 credits, auction ending soon",
    "a Quillfeather duo-oscillator synth on Nightjar Market, 190 credits, seller offering "
    "a bundle discount",
    "a Driftwood Six eurorack case on Orbital Swap, 95 credits",
    "a Pellucid Arp sequencer on Faraday Exchange, 220 credits",
    "a Rill & Moth ribbon controller on Nightjar Market, 60 credits",
    "a Second Harvest FM synth on Orbital Swap, 480 credits",
    "a Low Tide chorus pedal on Faraday Exchange, 35 credits",
    "a Candlewright string synth on Nightjar Market, 390 credits",
    "a Foxglove patchbay on Orbital Swap, 70 credits",
    "a Winter Reed wavetable synth on Faraday Exchange, 455 credits",
    "a Scrimshaw mixer on Nightjar Market, 110 credits",
    "a Juniper Line percussion synth on Orbital Swap, 265 credits",
)
_MANY_FINDINGS_WRITTEN = tuple(
    WrittenEntry(memory=_MANY_FINDINGS_COLLECTION, key=f"listing-{index:02d}", content=content)
    for index, content in enumerate(_MANY_FINDINGS_CONTENT, start=1)
)
_MANY_FINDINGS_BASE = NotificationInput(
    collection=_MANY_FINDINGS_COLLECTION,
    description="",
    routine=None,
    calls=(),
    written=_MANY_FINDINGS_WRITTEN,
    related=(),
)
# The item NAME of each of the first 8 findings — the ones NOTIFY_WRITTEN_ENTRIES actually
# renders into the document the draw reads.  The other 10 are never given to the model, so
# claiming them would be unfair (#2373 ruling): "every finding named" means every finding
# the model was shown, never the whole batch the cycle wrote.
_MANY_FINDINGS_ANCHORS = (
    "Korrigan-7",
    "Halcyon Drift",
    "Korrigan Satchel",
    "Tamsin Eight",
    "Vesper Loop",
    "Fenlow Thumper",
    "Marrow FM",
    "Quillfeather",
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
_SINGLE_FINDING_ANCHOR = "Fenwick & Vance telescoping pruner"
_SINGLE_FINDING_WRITTEN = (
    WrittenEntry(
        memory=_SINGLE_FINDING_COLLECTION,
        key="listing-01",
        content=f"a {_SINGLE_FINDING_ANCHOR} on Orbital Swap, listed new in box at 38 credits",
    ),
)
_SINGLE_FINDING_BASE = NotificationInput(
    collection=_SINGLE_FINDING_COLLECTION,
    description="",
    routine=None,
    calls=(),
    written=_SINGLE_FINDING_WRITTEN,
    related=(),
)
_SINGLE_FINDING_WORLD = World(
    name="garden-tool-watch-cycle",
    pages=(),
    keeps=(),
    excludes=(),
    answers=(_SINGLE_FINDING_ANCHOR,),
)
_SINGLE_FINDING_DESCRIPTION = "keeps an eye on Orbital Swap for new garden-tool listings"
_SINGLE_FINDING_ALSO_DESCRIBED = (
    "watches Orbital Swap and flags new garden-tool listings",
    "checks the Orbital Swap listings for anything garden-tool related that's new",
    "tracks new garden-tool listings on Orbital Swap",
    "scans Orbital Swap for fresh garden-tool listings",
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_notify_composer_names_one_ordinary_finding(
    notify_composer_eval: NotifyComposerEval, model: str
) -> None:
    """The driver's own smoke test and cost baseline: one ordinary finding, nothing to
    summarise, nothing capped — what ``notify-composer-many-findings`` is read against."""
    cohort = await notify_composer_eval(
        case_id="notify-composer-single-finding",
        behaviour=(
            "In the notify composer, when a cycle finds one ordinary new entry, Penny's "
            "notification names what the finding carries and invents nothing."
        ),
        area=Area.NOTIFICATIONS,
        model=model,
        base=_SINGLE_FINDING_BASE,
        world=_SINGLE_FINDING_WORLD,
        description=_SINGLE_FINDING_DESCRIPTION,
        also_described=_SINGLE_FINDING_ALSO_DESCRIBED,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=_FAMILY,
    )
    # STORE — empty, by report: a single call writes to no store.
    # PROVENANCE
    cohort.assert_the_reply_answers_the_ask()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_notify_composer_names_many_findings_at_once(
    notify_composer_eval: NotifyComposerEval, model: str
) -> None:
    """A cycle that wrote 18 findings in one pass across three marketplaces, mixed
    relevance, real-sized text — the shape research 3/3 of the honest-failure meta (#2373)
    asked for.  The document's own render caps written entries at 8
    (``PennyConstants.NOTIFY_WRITTEN_ENTRIES``), so "every finding named" claims only the 8
    the draw is actually shown, never the 10 it is never given."""
    cohort = await notify_composer_eval(
        case_id="notify-composer-many-findings",
        behaviour=(
            "In the notify composer, when a cycle finds many new entries at once, Penny's "
            "notification names what the findings carry and invents nothing."
        ),
        area=Area.NOTIFICATIONS,
        model=model,
        base=_MANY_FINDINGS_BASE,
        world=_MANY_FINDINGS_WORLD,
        description=_MANY_FINDINGS_DESCRIPTION,
        also_described=_MANY_FINDINGS_ALSO_DESCRIBED,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=_FAMILY,
    )
    # STORE — empty, by report: a single call writes to no store.
    # PROVENANCE
    cohort.assert_the_reply_answers_the_ask()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(REPLY_SPREAD)
