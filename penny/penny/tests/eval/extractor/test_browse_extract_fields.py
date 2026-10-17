"""The browse EXTRACTION micro-context, in isolation: what a page makes of an instruction.

An ``extract`` instruction routinely names several things at once — what each item is
called, where it points, what it says — and how much of that a real page carries decides
what comes back.  The contract (#1942) is keyed to the PAGE and never to the wording:
content carrying SOME of what was asked for is an ``EXTRACTED:`` read that gives whatever
it has for each thing and leaves out the rest, and ``NOT_PRESENT:`` is for content carrying
NONE of it.

Six cases over the three states a page can be in relative to its instruction — it carries
ALL of what was named, SOME of it, or NONE — with the middle state reached on three different
page shapes, two of them at REAL SIZE.  Each is five wordings of its instruction over one
fixed page:

* ``extract-fields-all-present`` — the CONTROL: a listing that answers everything asked of
  it.  The cheapest case in the suite, and the one that says a per-field read was not bought
  by loosening the ordinary case.
* ``extract-top-headlines-off-a-front-page`` — the failure as it was actually seen (#2360):
  a real-size news front page (~90 links, ~9K characters, the shape the content script
  returns for such a page) read for its top three headlines, where the first three story
  links carry no description.  Asked at real scale rather than the suite's usual few-link
  stand-in, because the undecidable loop the incident showed — re-reading the whole link
  list over and over while arguing whether a headline without its description is a partial
  unit or a missing one — only shows at that size.
* ``extract-top-headlines-past-a-run-on-label`` — a sharper version of the same incident
  (#2453): the same front page, with one more link spliced in just past the top three whose
  own label runs directly into its description with no separating space.  The real production
  draws read that run-on link correctly, then used the pattern they had just found there to
  re-litigate whether the top three — ordinary punctuation, ordinary spaces — were hiding the
  same split, burning several thousand extra tokens and, on the worst draws, costing the
  response its own ``EXTRACTED:`` tag.
* ``extract-fields-partly-present`` — the same front page read as an index of other pages:
  titles and links, and no summaries, under an instruction that asks for all three.
* ``extract-fields-partly-present-on-a-prose-page`` — the same shortfall reached by a second
  PAGE SHAPE: a prose section front printing a byline under each headline and no timestamp.
* ``extract-fields-none-present`` — the honesty guard: the front page carrying neither of
  the two things named must say so rather than answer anyway.

**The last two are a PAIR, and they are two cases rather than one.**  A behaviour states its
negative direction inside its own case where that direction shares the expected outcome;
here it does not — ``extract-fields-partly-present`` expects ``EXTRACTED`` and
``extract-fields-none-present`` expects ``NOT_PRESENT`` — so the negative direction is a case
of its own and the pair is named together.  A behaviour that always answered ``EXTRACTED:``
passes the first and fails the second.

**The arm is the ``extract`` instruction, and it is PENNY's own text** — production writes
it as the ``extract`` argument of a browse call, at the call site, by whatever agent made
the call.  So the five wordings are not five ways a person might ask; they are five ways the
CALLING DRAW might have worded the same request, which is variation that happens in
production today and that nothing else measures.  The FACTS are constant across the arms of a
case, because the claims name them.

Each case's page is what the CONTENT SCRIPT returns for a page of that shape, and the
document handed to the draw is built the way ``BrowseTool._page_section`` builds it, so the
extractor reads what production hands it.  A deterministic probe in ``make check`` holds each
fixture against the world it claims — every anchor really is on its page, exactly once —
because a case whose page does not carry what it says it carries measures nothing, and that
must fail before any GPU time rather than after.

Every case is report-only; the thresholds are the code owner's.  All content is synthetic.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.constants import PageReading
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    EXTRACT_MISSING,
    EXTRACT_OUTCOME,
    EXTRACT_REASON,
    EXTRACT_VALUE,
    ExtractorEval,
    FieldExpectation,
)
from penny.tests.eval.utils.assertions import Answer, WorldClaim
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    Consequence,
    SampleObservation,
    Share,
    ShareReading,
    SpecCategory,
    output_field,
    unsourced_specifics,
)
from penny.tests.eval.utils.fixtures import LISTING_URL
from penny.tests.eval.utils.worlds import World
from penny.tools.micro_context import MicroExtractOutcome, spoken_form

pytestmark = pytest.mark.eval

_FAMILY = "browse-extract"


class ExtractFixture(NamedTuple):
    """One agreed case: the page as the content script returns it, the instruction as a
    task would word it, and what each thing the instruction names should come back as —
    an anchor for the ones the page supplies, nothing for the ones it does not.

    ``behaviour`` is the one sentence the case exists to check, in the fixed form: "In <the
    locus>, when <X>, Penny <does Y>."  It rides the fixture so a case states it the same way
    whichever driver path runs it, and the report renders it before any number.

    ``expectations`` is what the ``make check`` coherence probe holds against the page.  Each
    anchor it names is the SAME constant the case's own provenance claim reads, so a page edit
    cannot leave a claim asserting a span the page no longer carries.

    ``reading`` (#2388) declares how the extension would have read ``page`` — ARTICLE (the
    default) or LINK_LIST — matching the shape the page's own comment says it is: a markdown
    link-index page declares LINK_LIST, a listing or prose page stays at the default."""

    case_id: str
    url: str
    page: str
    instruction: str
    behaviour: str
    expectations: tuple[FieldExpectation, ...]
    reading: PageReading = PageReading.ARTICLE


# ── The pages ────────────────────────────────────────────────────────────────
#
# A real-size news front page, in the link-index shape the content script returns for such a
# page: one markdown link per story, nav item or footer item, and nothing else — no article
# body, because an index page has none (#1942's `page_text.ts` split).  Shaped and sized like
# the page measured off the incident that opened #2360: ~9K characters, ~90 links across three
# header lines, ~15 one/two-word nav links, 38 story links, ~35 section/footer links and one
# closing line.  The first three story links in document order carry headline only — the
# ones `extract-top-headlines-off-a-front-page` asks for — and a few later ones (the shipyard,
# aquarium, foundry and apiary stories) carry a one-sentence description inside the link
# label, so the page is not uniformly one shape end to end.  Invented site, URLs and headlines.
_FRONT_PAGE_URL = "https://news-alpha.example/"
_FRONT_PAGE = (
    "Title: News Alpha — front page\n"
    "https://news-alpha.example/\n"
    "Local, national and world coverage, updated through the day\n"
    "[World](https://news-alpha.example/world)\n"
    "[Business](https://news-alpha.example/business)\n"
    "[Culture](https://news-alpha.example/culture)\n"
    "[Sports](https://news-alpha.example/sports)\n"
    "[Opinion](https://news-alpha.example/opinion)\n"
    "[Science](https://news-alpha.example/science)\n"
    "[Health](https://news-alpha.example/health)\n"
    "[Weather](https://news-alpha.example/weather)\n"
    "[Obituaries](https://news-alpha.example/obituaries)\n"
    "[Classifieds](https://news-alpha.example/classifieds)\n"
    "[Subscribe](https://news-alpha.example/subscribe)\n"
    "[Newsletters](https://news-alpha.example/newsletters)\n"
    "[Archive](https://news-alpha.example/archive)\n"
    "[Contact us](https://news-alpha.example/contact)\n"
    "[About us](https://news-alpha.example/about)\n"
    "[Harbour bridge reopens after a two-year refit that outlasted three mayors and two "
    "budget overhauls](https://news-alpha.example/world/2036/harbour-bridge-reopens-after-refit)\n"
    "[Dockside market changes hands after eighty years under the same founding family of "
    "traders](https://news-alpha.example/business/2036/dockside-market-changes-hands)\n"
    "[Lantern festival draws a record crowd to the old quarter despite a week of heavy rain "
    "forecast](https://news-alpha.example/world/2036/lantern-festival-draws-record-crowd)\n"
    "[Museum returns the borrowed mosaics a decade early after a quiet dispute over storage "
    "costs](https://news-alpha.example/culture/2036/museum-returns-borrowed-mosaics)\n"
    "[Ferry operator adds a night sailing for the winter after months of commuter "
    "complaints](https://news-alpha.example/business/2036/ferry-operator-adds-night-sailing)\n"
    "[City orchestra names a conductor from within the ranks for the first time in forty "
    "years](https://news-alpha.example/culture/2036/orchestra-names-conductor-from-ranks)\n"
    "[Bookshop reopens in the arcade under new owners who kept most of the old "
    "staff](https://news-alpha.example/culture/2036/bookshop-reopens-in-arcade)\n"
    "[Old lighthouse keeper's cottage is listed for sale after the trust ran out of "
    "funds](https://news-alpha.example/world/2036/lighthouse-cottage-listed-for-sale)\n"
    "[Rowing club wins the river cup for a third year running under a coach half their "
    "age](https://news-alpha.example/sports/2036/rowing-club-wins-river-cup)\n"
    "[Observatory volunteers log a record number of meteor sightings over the long "
    "weekend](https://news-alpha.example/science/2036/observatory-logs-meteor-sightings)\n"
    "[Clinic extends its evening hours after a sharp rise in walk-in appointments this "
    "winter](https://news-alpha.example/health/2036/clinic-extends-evening-hours)\n"
    "[Shipyard lands its biggest order in a decade from a regional ferry operator. The "
    "contract covers four vessels and secures roughly three hundred jobs through the next "
    "two winters.](https://news-alpha.example/business/2036/shipyard-lands-biggest-order)\n"
    "[Canal towpath reopens to cyclists after the flood barrier repairs finally "
    "finished](https://news-alpha.example/world/2036/canal-towpath-reopens-to-cyclists)\n"
    "[Theatre revives a forgotten local play for its first run in sixty "
    "years](https://news-alpha.example/culture/2036/theatre-revives-forgotten-local-play)\n"
    "[Stadium announces a new roof design after years of complaints about rain "
    "delays](https://news-alpha.example/sports/2036/stadium-announces-new-roof-design)\n"
    "[Brewery doubles its tasting room after demand outgrew the old converted "
    "barn](https://news-alpha.example/business/2036/brewery-doubles-tasting-room)\n"
    "[Tunnel repairs finally finish after eighteen months of diversions through the old "
    "quarter](https://news-alpha.example/world/2036/tunnel-repairs-finally-finish)\n"
    "[Aquarium hatches a rare species of seahorse for the first time in its history. "
    "Keepers say the hatchlings are already feeding well and will join the main display "
    "within a month.](https://news-alpha.example/science/2036/aquarium-hatches-rare-seahorse)\n"
    "[Vineyard opens its cellar for public tours after years of only private "
    "tastings](https://news-alpha.example/culture/2036/vineyard-opens-cellar-tours)\n"
    "[Quarry closure leaves the valley's last heavy haulage route without regular "
    "traffic](https://news-alpha.example/world/2036/quarry-closure-leaves-valley-route)\n"
    "[Sawmill switches to a four-day week after a long stretch of reduced "
    "orders](https://news-alpha.example/business/2036/sawmill-switches-four-day-week)\n"
    "[Hospital wing reopens after repairs that took twice as long as first "
    "estimated](https://news-alpha.example/health/2036/hospital-wing-reopens-after-repairs)\n"
    "[Foundry marks its hundredth year with an open day for the whole town. Former "
    "apprentices are travelling back for the day, some from as far as the "
    "capital.](https://news-alpha.example/world/2036/foundry-marks-hundredth-year)\n"
    "[Cannery adds a second shift after landing a new regional supply "
    "contract](https://news-alpha.example/business/2036/cannery-adds-second-shift)\n"
    "[Mill museum adds a working loom display after a two-year restoration "
    "project](https://news-alpha.example/culture/2036/mill-museum-adds-working-loom)\n"
    "[Dam inspection finds no new cracks after last spring's heavy flooding "
    "scare](https://news-alpha.example/world/2036/dam-inspection-finds-no-new-cracks)\n"
    "[Reservoir levels drop to a ten-year low after the driest spring on "
    "record](https://news-alpha.example/science/2036/reservoir-levels-drop-ten-year-low)\n"
    "[Greenhouse restoration finishes ahead of schedule thanks to a surge of volunteer "
    "labour](https://news-alpha.example/culture/2036/greenhouse-restoration-finishes-early)\n"
    "[Nursery school expansion clears its final planning hurdle after a long community "
    "review](https://news-alpha.example/world/2036/nursery-school-expansion-clears-hurdle)\n"
    "[Hatchery ships its first export order of the season to three neighbouring "
    "towns](https://news-alpha.example/business/2036/hatchery-ships-first-export-order)\n"
    "[Apiary collective reports its best honey harvest in a decade across every member "
    "farm. Members credit a mild spring and a late frost that never arrived this "
    "year.](https://news-alpha.example/culture/2036/apiary-collective-reports-best-harvest)\n"
    "[Orchard cooperative opens a new pressing house ahead of the autumn harvest "
    "rush](https://news-alpha.example/world/2036/orchard-cooperative-opens-pressing-house)\n"
    "[Tannery closes its oldest workshop after relocating production to the new "
    "site](https://news-alpha.example/business/2036/tannery-closes-oldest-workshop)\n"
    "[Forge apprenticeship programme doubles its intake after a surge of local "
    "interest](https://news-alpha.example/culture/2036/forge-apprenticeship-doubles-intake)\n"
    "[Distillery releases a limited cask after ageing it longer than originally "
    "planned](https://news-alpha.example/business/2036/distillery-releases-limited-cask)\n"
    "[Swim club opens a masters league after years of requests from older "
    "members](https://news-alpha.example/sports/2036/swim-club-opens-masters-league)\n"
    "[Cycling route gets a new signposted loop after the old markers finally wore "
    "out](https://news-alpha.example/sports/2036/cycling-route-gets-new-loop)\n"
    "[Pharmacy trials a late-night pickup window after a run of missed "
    "deliveries](https://news-alpha.example/health/2036/pharmacy-trials-late-night-pickup)\n"
    "[World news](https://news-alpha.example/world-news)\n"
    "[Business desk](https://news-alpha.example/business-desk)\n"
    "[Markets](https://news-alpha.example/markets)\n"
    "[Culture](https://news-alpha.example/culture)\n"
    "[Reviews](https://news-alpha.example/reviews)\n"
    "[Sports](https://news-alpha.example/sports)\n"
    "[Scores](https://news-alpha.example/scores)\n"
    "[Opinion](https://news-alpha.example/opinion)\n"
    "[Editorials](https://news-alpha.example/editorials)\n"
    "[Letters](https://news-alpha.example/letters)\n"
    "[Science](https://news-alpha.example/science)\n"
    "[Environment](https://news-alpha.example/environment)\n"
    "[Health](https://news-alpha.example/health)\n"
    "[Wellbeing](https://news-alpha.example/wellbeing)\n"
    "[Weather](https://news-alpha.example/weather)\n"
    "[Forecasts](https://news-alpha.example/forecasts)\n"
    "[Obituaries](https://news-alpha.example/obituaries)\n"
    "[Classifieds](https://news-alpha.example/classifieds)\n"
    "[Jobs](https://news-alpha.example/jobs)\n"
    "[Property](https://news-alpha.example/property)\n"
    "[Subscribe](https://news-alpha.example/subscribe)\n"
    "[Gift a subscription](https://news-alpha.example/gift-a-subscription)\n"
    "[Newsletters](https://news-alpha.example/newsletters)\n"
    "[RSS feeds](https://news-alpha.example/rss-feeds)\n"
    "[Mobile app](https://news-alpha.example/mobile-app)\n"
    "[Archive](https://news-alpha.example/archive)\n"
    "[Corrections](https://news-alpha.example/corrections)\n"
    "[Ethics policy](https://news-alpha.example/ethics-policy)\n"
    "[Privacy policy](https://news-alpha.example/privacy-policy)\n"
    "[Terms of use](https://news-alpha.example/terms-of-use)\n"
    "[Advertise](https://news-alpha.example/advertise)\n"
    "[Careers](https://news-alpha.example/careers)\n"
    "[Contact us](https://news-alpha.example/contact-us)\n"
    "[About us](https://news-alpha.example/about-us)\n"
    "[Help centre](https://news-alpha.example/help-centre)\n"
    "[Older stories](https://news-alpha.example/archive/2036)"
)

# The same front page, with ONE more link spliced in right after the third top headline
# (#2453): a label whose own title runs directly into its description with NO separating
# space or punctuation gap — the exact shape that cost a real front-page read minutes of
# deliberation and, on some draws, its own EXTRACTED: tag.  The title is two sentences
# (mirroring the real trigger), the second ending in a question mark that the description's
# first word follows with no gap at all.  Inserted early, beside the top three, because
# that is where the incident showed the model start treating the run-on link's own boundary
# as a hint that the top three might be hiding an un-signalled split too.  Fictional story,
# invented site.
_RUNON_LINK = (
    "[This clinic has forty staff. Why did the merger finally happen?The two practices had "
    "argued for years over whose name to keep, and the deal only closed once a retiring "
    "partner agreed to drop his. Patients say little has changed from one day to the next. "
    "Staff say the paperwork alone took months.]"
    "(https://news-alpha.example/health/2036/clinic-merger-why-so-long)\n"
)
_FRONT_PAGE_WITH_RUNON_LABEL = _FRONT_PAGE.replace(
    "lantern-festival-draws-record-crowd)\n", f"lantern-festival-draws-record-crowd)\n{_RUNON_LINK}"
)
assert _FRONT_PAGE_WITH_RUNON_LABEL != _FRONT_PAGE, "splice anchor not found on the front page"

# A section front that prints a byline under each headline and no timestamp — the same
# partly-answered shortfall reached by a different PAGE SHAPE.  It is a second CASE and not
# five more arms of the link-index one, because an arm varies the WORDING of one ask: this
# instruction names different things (a byline and a time, where the other names a link and a
# summary), and it names them of a page written as prose rather than as markdown links.  A
# different ask over a different page is a scenario, and a scenario gets its own 15.
#
# Real-size section front: about ten stories, each with a byline and a one- or two-sentence
# teaser — no timestamp on any of them, which is the shortfall this case holds on.
_SECTION_URL = "https://news-alpha.example/culture"
_SECTION = (
    "Culture\n"
    "\n"
    "Museum returns the borrowed mosaics a decade early\n"
    "by Wren Halloway\n"
    "The loan agreement had years left to run, but the lender asked for them back early for "
    "a touring show that opens in the spring, and the museum says it would rather hand them "
    "over on good terms than fight the request. Curators spent the week photographing every "
    "tile before the crates went out, in case a dispute over condition comes up later. Staff "
    "say the full collection will not travel again for at least another five years.\n"
    "\n"
    "City orchestra names a conductor from within the ranks\n"
    "by Ines Marlowe\n"
    "It is the first time in forty years the post has gone to someone already on the "
    "payroll, and the players say the choice was close to unanimous after a season of guest "
    "conductors failed to settle. The new conductor has led the viola section for nineteen "
    "years and will keep a reduced teaching load at the conservatory alongside the new post. "
    "The orchestra's board says the appointment was approved without a single dissenting "
    "vote.\n"
    "\n"
    "Bookshop reopens in the arcade under new owners\n"
    "by Oskar Penrith\n"
    "Most of the old staff stayed on, and the children's corner has doubled in size after "
    "the new owners knocked through to the unit next door, which had sat empty since the "
    "tailor's shop closed. The owners say the expansion was always the plan, and the delay "
    "was only ever about getting the lease signed. The old unit next door had been empty "
    "long enough that most regulars had forgotten what used to be there.\n"
    "\n"
    "Theatre revives a forgotten local play for its first run in sixty years\n"
    "by Talia Brecht\n"
    "The original script turned up in a box of donated costumes last spring, water-stained "
    "but legible enough for the dramaturg to reconstruct the missing pages from a newspaper "
    "review of the original run. Casting is already finished, and rehearsals begin as soon "
    "as the set carpenters finish the revolving stage. The theatre says ticket prices will "
    "stay the same as last season despite the extra cast member.\n"
    "\n"
    "Gallery hangs its winter show a month ahead of schedule\n"
    "by Soren Myklund\n"
    "Organisers say the early opening is meant to beat the worst of the holiday crowds, "
    "though the move also lets the gallery squeeze in an extra weekend of ticketed previews "
    "before the free weekdays begin. Three of the pieces on display have never been shown "
    "outside the artist's own studio before now. The gallery is already fielding requests "
    "from two other venues hoping to borrow the show once it closes.\n"
    "\n"
    "Choir society marks its centenary with a free concert in the square\n"
    "by Priya Oduya\n"
    "Founding members' descendants have been invited to sing in the final number, and the "
    "society has dug up a recording from its fiftieth anniversary to play before the "
    "concert starts. Organisers expect the square to fill well before the opening number, "
    "and extra seating is being brought in. The choir's archivist spent three weekends in "
    "the basement tracking down the right reel-to-reel player.\n"
    "\n"
    "Potters' guild opens its kilns to visitors for the first time in years\n"
    "by Callum Fenwick\n"
    "The open day replaces a members-only sale that ran for three decades, a change the "
    "guild says is meant to bring in younger members who never knew the sale existed in the "
    "first place. Visitors will be able to watch a full firing from loading to cooling "
    "across the open weekend. The guild says it hopes to make the open day an annual fixture "
    "rather than a one-off.\n"
    "\n"
    "Film society adds a second screen after years on a single projector\n"
    "by Marisol Quintard\n"
    "The upgrade was funded almost entirely by a single retiring patron's donation, and the "
    "society says it plans to use the second screen for matinees of films too niche to "
    "justify a full evening slot. The first title booked for the new screen is a restored "
    "print nobody local has seen since its original run. The society is already taking "
    "requests for a third screen, though nobody expects it to happen soon.\n"
    "\n"
    "Archive digitises its oldest ledgers after a grant from the heritage fund\n"
    "by Declan Ashworth\n"
    "Volunteers have been transcribing entries by hand ahead of the public launch, a job "
    "the archivist says has turned up at least two shipwrecks nobody had previously "
    "recorded in any local history. The full digitised ledger set is expected to go online "
    "sometime next year, pending one more funding round. The archivist says the biggest "
    "surprise so far was how many ledgers survived the old building's one flood.\n"
    "\n"
    "Puppet troupe tours the outer boroughs for the first time in its history\n"
    "by Nadia Steinholt\n"
    "The tour was planned around school terms rather than the usual festival calendar, "
    "which the troupe says should let far more children see a show without their parents "
    "needing to take a day off work. Every stop on the route was chosen by a borough that "
    "had never hosted the troupe before this year. The troupe's founder says this is the "
    "farthest the puppets have ever travelled from the workshop."
)

# An ordinary listing that answers everything asked of it — the control, where nothing about
# a per-field read should show at all.  It is at the house url (`LISTING_URL`) and the house
# price (`499`), which is what the rest of the suite shares and what the claims below name.
#
# The PAGE ITSELF is local, and deliberately not `fixtures.AURORA_LISTING_499`.  This case is
# the one that needs THREE fields answered, and the shared listing carries two: it has a title,
# a price, a seller and a self-link, and it says nothing about stock.  Adding a stock line to it
# would change the page every browse-driven case in the suite reads, which is a measurement this
# port has not taken and is not the port's to take.  So the copy is real and this is why it
# exists.  The url is IMPORTED (`LISTING_URL`) rather than retyped; the price is the same `499`
# the shared listing prints, held against this page by the coherence probe in `make check`.
#
# Real-length product page: the full listing copy (specs, shipping, seller notes, a Q&A
# section, early reviews) plus a "customers also viewed" rail of related items at OTHER
# prices, none of which is `499` and none of which repeats the exact item name — a product
# page's own real estate, not a five-line stand-in.
_LISTING = (
    "Aurora Deck 2, handheld console\n"
    "\n"
    "Price: $499\n"
    "In stock — three left, open box and tested\n"
    "Ships from a fictional warehouse within two days.\n"
    "\n"
    "About this item\n"
    "This handheld console has a seven inch display, a detachable controller grip and "
    "roughly six hours of battery life on a full charge under typical use. This unit is an "
    "open box return: the original packaging has been opened and resealed, and the console "
    "has been tested and wiped to factory settings before listing. The seller reports no "
    "cosmetic marks beyond light wear on the left grip, and the included cable is the "
    "original rather than a replacement. The screen protector that shipped with the unit "
    "has already been removed and is not included, though a replacement is cheap and widely "
    "available at most electronics counters.\n"
    "\n"
    "Specifications\n"
    "Display: seven inch, 1080p, sixty hertz.\n"
    "Storage: five hundred twelve gigabytes, expandable.\n"
    "Battery: roughly six hours of mixed use per charge.\n"
    "Weight: three hundred ninety grams without the grip attached.\n"
    "Connectivity: dual-band wireless and a single charging port.\n"
    "Audio: stereo speakers and a standard headphone jack.\n"
    "Included: console, grip, charging cable, and the original box insert.\n"
    "\n"
    "Shipping and returns\n"
    "Ships from a fictional warehouse within two days of payment clearing. Returns are "
    "accepted within thirty days provided the unit is returned in the condition it was "
    "sent, with the original cable included. Buyers outside the seller's usual delivery "
    "region should expect an extra day or two in transit depending on the carrier's "
    "regional hub. A restocking fee may apply if the console comes back without its "
    "original cable or box insert.\n"
    "\n"
    "Seller notes\n"
    "Seller: nebula_resale (4.9 stars). Listing updated daily. The seller has sold open-box "
    "electronics on this marketplace for six years and answers questions within a day on "
    "most listings, usually faster on weekdays than on weekends.\n"
    "\n"
    "Customer questions\n"
    "Several buyers have asked whether this unit includes a warranty: it does not, since "
    "open-box listings on this marketplace are sold as-is beyond the thirty-day return "
    "window above. Another recurring question is whether the grip is interchangeable with "
    "the original release's grip; the seller confirms it is not, since the connector was "
    "redesigned for this generation. A third question asked whether the battery has been "
    "cycle-tested; the seller says it was charged and drained twice before the listing "
    "photos were taken and held its charge both times.\n"
    "\n"
    "Early reviews of this generation\n"
    "Buyers of this model have generally praised the battery life over the previous "
    "generation, with several noting it comfortably lasts a long flight. A smaller number "
    "have flagged the detachable grip as fiddly to reattach one-handed, which the "
    "manufacturer has acknowledged without promising a fix. Most reviewers rate the display "
    "as a clear improvement over the previous generation's panel, particularly outdoors.\n"
    "\n"
    "Customers also viewed\n"
    "Aurora Deck 1, handheld console — $219. [View listing](https://faux-market.example/aurora-deck-1)\n"
    "Aurora Prime, handheld console — $760. [View listing](https://faux-market.example/aurora-prime)\n"
    "Carrying case for Aurora handhelds — $35. [View listing](https://faux-market.example/aurora-handheld-case)\n"
    "Charging dock for Aurora handhelds — $58. [View listing](https://faux-market.example/aurora-handheld-dock)\n"
    "Nimbus Play, handheld console — $349. [View listing](https://faux-market.example/nimbus-play)"
)


# ── The anchors ──────────────────────────────────────────────────────────────
#
# One constant per thing a page SUPPLIES, named once and read twice: the deterministic
# coherence probe holds it against the page through the fixture's expectations, and the case's
# own provenance claim reads the same string.  One source of truth, so a page edit cannot leave
# a claim asserting a span the page no longer carries.
#
# Each is the SMALLEST UNIQUE span of what it identifies, because everything around it the
# model may legitimately write another way.  A draw that trimmed a headline to its subject, or
# dropped a currency symbol, read the page exactly as well as one that copied the line, and an
# anchor carrying the whole line would fail it.  Unique on the page as well as small: the probe
# holds both, since a span appearing twice would be satisfied by the wrong one.
_ITEM_ANCHOR = "Aurora Deck 2"
_PRICE_ANCHOR = "499"
# The ASK is a BOOLEAN — "whether it is in stock" — so the anchor is the page's own phrase for
# that boolean, never the count beside it: a draw that answers "yes, it's in stock" has answered
# the ask in full without ever saying how many are left.  `in stock` is the smallest unique span
# naming the state itself, compared through the same `spoken_form` fold every other anchor here
# uses.
_STOCK_ANCHOR = "in stock"
# The count the page states beside the boolean — a detail the ask never requested.  How often a
# draw volunteers it anyway is measured, not claimed (`_STOCK_COUNT_VOLUNTEERED`, below): the
# page writes it as a word, so a draw that read the page perfectly may legitimately write `3`
# instead, and volunteering it at all is a free choice rather than a fact the ask owes.
_STOCK_COUNT_WORD = "three"

# The front page's first three story links in document order — headline only, no
# description — which `extract-top-headlines-off-a-front-page` asks for and
# `extract-fields-partly-present` reads its LEAD anchor from.
_TOP1_HEADLINE_ANCHOR = "Harbour bridge"
_TOP1_LINK_ANCHOR = "https://news-alpha.example/world/2036/harbour-bridge-reopens-after-refit"
_TOP2_HEADLINE_ANCHOR = "Dockside market"
_TOP2_LINK_ANCHOR = "https://news-alpha.example/business/2036/dockside-market-changes-hands"
_TOP3_HEADLINE_ANCHOR = "Lantern festival"
_TOP3_LINK_ANCHOR = "https://news-alpha.example/world/2036/lantern-festival-draws-record-crowd"

# `extract-fields-partly-present`'s other two anchors — a story from the middle of the link
# list and one from the last section before the footer — so its claim reads the draw across
# the WHOLE page rather than just its first few links.
_MID_HEADLINE_ANCHOR = "Quarry closure"
_MID_LINK_ANCHOR = "https://news-alpha.example/world/2036/quarry-closure-leaves-valley-route"
_LAST_HEADLINE_ANCHOR = "Pharmacy trials"
_LAST_LINK_ANCHOR = "https://news-alpha.example/health/2036/pharmacy-trials-late-night-pickup"

_SECTION_HEADLINE_ANCHOR = "City orchestra"
_SECTION_BYLINE_ANCHOR = "Ines Marlowe"

# The spliced-in run-on link's own title (#2453) — its smallest unique span, held against
# `_FRONT_PAGE_WITH_RUNON_LABEL` by the same coherence probe as every other anchor here.  It
# is never the subject of a claim on its own case (the ask is still "the top 3", and this
# link sits below them); it exists so a reader confirming the fixture coheres with what it
# claims can find the one thing that changed.
_RUNON_TITLE_ANCHOR = "forty staff"
assert _FRONT_PAGE_WITH_RUNON_LABEL.count(_RUNON_TITLE_ANCHOR) == 1


# ── The claims ───────────────────────────────────────────────────────────────
#
# A micro-context returns a typed result, and that result splits in two.  Its CLOSED field —
# the outcome enum — is strictly identifiable and asserted by EQUALITY.  Its OPEN fields —
# `value`, `reason` — are what the model wrote, so they are reachable only through fact
# alignment, in both directions: nothing omitted, and nothing invented.


def _landed_on(fixture: ExtractFixture) -> WorldClaim:
    """A claim that the draw committed to the outcome THIS FIXTURE'S FACTS make right.

    ``EXTRACTED`` is right on a page carrying some of what was asked for and wrong on a page
    carrying none of it — the same value, correct against one world and false against another.
    So the expectation is READ off the fixture rather than written beside it: an expectation
    carrying an anchor is a thing the page supplies, and a fixture that names at least one is
    a page with something to extract.

    That is the same declaration the ``make check`` coherence probe holds against the page, so
    a case cannot state a world to the probe and a different one to its own claim.  A case that
    cannot say which of its facts makes its expectation right is asserting a habit; this one
    has no way to."""
    supplies_something = any(one.anchor for one in fixture.expectations)
    wanted = (
        MicroExtractOutcome.EXTRACTED if supplies_something else MicroExtractOutcome.NOT_PRESENT
    )

    def answer(sample: SampleObservation, _world: World) -> Answer:
        drawn = sample.field(EXTRACT_OUTCOME)
        return drawn == wanted.value, f"came back {drawn}"

    return answer


def _carries(anchor: str) -> WorldClaim:
    """A claim that the answer carries one thing the PAGE supplies — the *nothing omitted*
    direction of fact alignment.

    One claim per supplied thing rather than one over all of them, because an instruction
    naming several things degrades one thing at a time and a single combined claim would
    report "some of it arrived" as a total failure.

    Compared through the shipped ``spoken_form``, so a value passes whether or not the draw
    kept the punctuation or the article in front of it — deliberately not an equality, since
    which words carry a fact has a little play in it and a scorer demanding one exact string
    would be answering for the draw."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        value = sample.field(EXTRACT_VALUE)
        carried = spoken_form(anchor) in spoken_form(value)
        return carried, f"not in the extracted value: {value!r}"

    return answer


def _nothing_invented(sample: SampleObservation, _world: World) -> Answer:
    """Every specific value in the answer traces to what the draw was GIVEN — the *nothing
    invented* direction.

    The extractor's own failure mode, and the strongest claim available to it: the draw's open
    fields are free text lifted off the page and production validates none of it, so a plausible
    headline the page never carried reaches the caller verbatim and is written down as read.

    Read over the WHOLE structured answer rather than over ``value`` alone, so the branch that
    populates ``reason`` instead is covered by the same claim."""
    invented = unsourced_specifics(sample.output_text, sample.given)
    return not invented, f"not on the page: {invented}"


def _names_the_shortfall(fixture: ExtractFixture, words: tuple[str, ...]) -> WorldClaim:
    """A claim that the draw's stated shortfall (#2360) names what the page lacks — every
    ``FieldExpectation`` this fixture declares with NO anchor, which is what was asked for
    and is genuinely absent from the page.

    Read off the fixture rather than written beside it, the same way :func:`_landed_on`
    reads its wanted outcome: a fixture's absent-anchor expectations ARE the shortfall a
    correct draw states, so a case that enlarges or narrows what the page lacks moves this
    claim with it rather than leaving it to drift out of sync.

    Compared through the same ``spoken_form`` fold ``_carries`` uses, but against the
    WORDS the case's OWN five wordings (the instruction and its phrasings) use for that
    absent part, not one field name: the sample's wording is the ask the model read, and
    the MISSING line it writes echoes whichever of those wordings it was given ("summary"
    in one arm, "the time it was posted" in another), so the claim passes when the line
    carries ANY of the case's words and fails otherwise — strictly true/false, no fuzzy
    match.  Every fixture this claim serves declares exactly one absent expectation, so
    ``words`` describes that one part; a fixture that ever declares two needs its own
    per-field words."""
    absent = [one.field for one in fixture.expectations if not one.anchor]

    def answer(sample: SampleObservation, _world: World) -> Answer:
        missing = sample.field(EXTRACT_MISSING)
        named = bool(missing) and all(
            any(spoken_form(word) in spoken_form(missing) for word in words) for _field in absent
        )
        return named, f"stated shortfall does not name any of {list(words)}: {missing!r}"

    return answer


def _stock_count_volunteered(sample: SampleObservation) -> ShareReading | None:
    """Whether the answer also states the COUNT beside the stock boolean — a detail the ask
    never requested.

    A SHARE, not a claim: keeping a value said about a single answer is a free choice, so the
    stock question measures how often she volunteers the count beyond what was asked, never
    whether she did — a rise would still show a lazy or over-literal reader, but it is read,
    not gated.

    ``None`` only if the sample produced nothing to read the fraction of (an absent extraction
    outcome); this fixture's page always supplies the boolean, so that reads as every pooled
    sample having a value to check."""
    value = sample.field(EXTRACT_VALUE)
    if not value:
        return None
    volunteered = spoken_form(_STOCK_COUNT_WORD) in spoken_form(value)
    return ShareReading(value=1.0 if volunteered else 0.0, evidence=[value] if volunteered else [])


_STOCK_COUNT_VOLUNTEERED = Share(
    "stock count volunteered",
    _stock_count_volunteered,
    open_above=0.0,
    says=(
        "of the samples that answered the stock boolean, the share that also stated the count "
        "the page supplies beside it, which the ask never requested"
    ),
)


# ── Case 1: everything asked for is on the page (the control) ─────────────────

_ALL_PRESENT = ExtractFixture(
    case_id="extract-fields-all-present",
    url=LISTING_URL,
    page=_LISTING,
    instruction="the item's name, its price and whether it is in stock",
    behaviour=(
        "In the browse-extract micro-context, when the page carries everything the "
        "instruction named, Penny comes back with all of it."
    ),
    expectations=(
        FieldExpectation("name", _ITEM_ANCHOR),
        FieldExpectation("price", _PRICE_ANCHOR),
        FieldExpectation("stock", _STOCK_ANCHOR),
    ),
)

# Five wordings of that one request.  Each names the same three things and none of them nudges
# at HOW the stock line should be rendered: a wording that asked "how many are left" would earn
# the count anchor above rather than measure it, which is the fixture lending its own phrasing
# to the prompt under test.
_ALL_PRESENT_PHRASINGS = (
    "what the item is called, what it costs, and whether it is available",
    "the product name, the price, and the stock status",
    "for the item on this page: its name, its price, and whether it can be bought",
    "pull out the name of the item, how much it costs, and whether it is in stock",
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_page_that_answers_everything_still_answers_everything(
    extractor_eval: ExtractorEval, model: str
) -> None:
    """The unchanged shape: three things named, three things on the page, three things back.

    It is here so a per-field read cannot be bought by loosening the ordinary case — if this
    moves, the change did something other than what it claims.

    **The STORE category is empty for this case, and that is the correct report.**  A
    micro-context is one call that returns a typed result — it moves no machine and writes
    nothing to any store — so there is no store claim to make.  The empty section says the
    shape has nothing to store, not that nobody ran the checklist.
    """
    cohort = await extractor_eval(
        case_id=_ALL_PRESENT.case_id,
        behaviour=_ALL_PRESENT.behaviour,
        area=Area.READING_A_PAGE,
        model=model,
        url=_ALL_PRESENT.url,
        page=_ALL_PRESENT.page,
        reading=_ALL_PRESENT.reading,
        instruction=_ALL_PRESENT.instruction,
        also_instructed=_ALL_PRESENT_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — which of the closed outcomes the draw committed to
    cohort.claim(
        "state: the draw read the page",
        _landed_on(_ALL_PRESENT),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the docstring.

    # PROVENANCE — what the page supplies arrived, and nothing else did
    cohort.claim(
        "state: the answer carries the item name the page supplies",
        _carries(_ITEM_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the price the page supplies",
        _carries(_PRICE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries whether the item is in stock",
        _carries(_STOCK_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: every specific value in the answer is on the page",
        _nothing_invented,
        SpecCategory.PROVENANCE,
    )

    # What is MEASURED — the draw's own structured fields, compared across the cohort, plus the
    # stock-count share above.  No tool sequence and no reply spread: a single call makes
    # neither.  `reason` is measured though a correct draw leaves it empty, for the reason
    # spelled out on `extract-fields-partly-present`: an all-empty reading is flagged BLIND
    # rather than counted as agreement, and a sample that came back NOT_PRESENT carries a
    # reason the others do not — an outlier at finer grain than `outcome` gives.
    cohort.measure(
        output_field(EXTRACT_OUTCOME),
        output_field(EXTRACT_VALUE, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_REASON, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_MISSING, consequence=Consequence.COSMETIC),
        _STOCK_COUNT_VOLUNTEERED,
    )


# ── Case 2: a real-size link index, and an instruction that also asks for summaries ──

_PARTLY_PRESENT = ExtractFixture(
    case_id="extract-fields-partly-present",
    url=_FRONT_PAGE_URL,
    page=_FRONT_PAGE,
    instruction=(
        "the headlines and their links, with a one-line summary of each where the page gives one"
    ),
    behaviour=(
        "In the browse-extract micro-context, when the page carries only some of what the "
        "instruction named, Penny comes back with the parts it has instead of reporting the "
        "page empty."
    ),
    # Spread across the WHOLE real-size page — the lead story, one from the middle of the
    # link list, and one from the last section before the footer — so the claim reads
    # whether the draw read the page's full breadth rather than just its first few links.
    expectations=(
        FieldExpectation("headline-lead", _TOP1_HEADLINE_ANCHOR),
        FieldExpectation("link-lead", _TOP1_LINK_ANCHOR),
        FieldExpectation("headline-mid", _MID_HEADLINE_ANCHOR),
        FieldExpectation("link-mid", _MID_LINK_ANCHOR),
        FieldExpectation("headline-last", _LAST_HEADLINE_ANCHOR),
        FieldExpectation("link-last", _LAST_LINK_ANCHOR),
        FieldExpectation("summary"),
    ),
    reading=PageReading.LINK_LIST,
)

# What the five wordings (the instruction and `_PARTLY_PRESENT_PHRASINGS` below) call the
# part the page lacks — every one of them says "summary", so that is the word the MISSING
# line echoes back on this case (#2360 round 3).
_PARTLY_PRESENT_SHORTFALL_WORDS = ("summary",)

# Two of the five hedge ("where the page gives one", "if there is one") and three do not, which
# is deliberate: whether the answer degrades per field is decided by what the PAGE carries,
# never by whether the instruction was worded to expect a gap.  Pooling both wordings into one
# cohort states that as a variance reading over one world.
_PARTLY_PRESENT_PHRASINGS = (
    "the headline of each story, the link to it, and a one-line summary if there is one",
    "for every story: its title, its url, and a short summary",
    "each headline with its link and a one-sentence summary",
    "pull out the story titles, the links they point at, and a brief summary of each",
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_page_with_titles_and_links_and_no_summaries_still_reads(
    extractor_eval: ExtractorEval, model: str
) -> None:
    """The regression itself, at REAL SIZE (#2360): a ~90-link, ~9K-character front page has
    two of the three things named and the third is genuinely not on it.  The read has to come
    back with the two, not answer as though the page were empty — and it has to do that
    reading the whole page, not just the links nearest the top.

    Its NEGATIVE direction is ``extract-fields-none-present`` below, a separate case because
    its expected outcome is the other one.

    **The STORE category is empty for this case, and that is the correct report.**  A
    micro-context is one call that returns a typed result — it moves no machine and writes
    nothing to any store — so there is no store claim to make.  The empty section says the
    shape has nothing to store, not that nobody ran the checklist.
    """
    cohort = await extractor_eval(
        case_id=_PARTLY_PRESENT.case_id,
        behaviour=_PARTLY_PRESENT.behaviour,
        area=Area.READING_A_PAGE,
        model=model,
        url=_PARTLY_PRESENT.url,
        page=_PARTLY_PRESENT.page,
        reading=_PARTLY_PRESENT.reading,
        instruction=_PARTLY_PRESENT.instruction,
        also_instructed=_PARTLY_PRESENT_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — which of the closed outcomes the draw committed to.  ``NOT_PRESENT`` here is the
    # regression itself: the answer that cost a whole round its headlines and links because the
    # page was short of the third thing.
    cohort.claim(
        "state: the draw read the page rather than reporting it empty",
        _landed_on(_PARTLY_PRESENT),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the docstring.

    # PROVENANCE — what the page supplies arrived, and nothing else did.  Three pairs
    # rather than one, spread across the page's breadth (see the fixture's comment above).
    cohort.claim(
        "state: the answer carries the lead headline the page supplies",
        _carries(_TOP1_HEADLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the lead headline's link",
        _carries(_TOP1_LINK_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries a mid-page headline the page supplies",
        _carries(_MID_HEADLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries that mid-page headline's link",
        _carries(_MID_LINK_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries a headline from the last section",
        _carries(_LAST_HEADLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries that last-section headline's link",
        _carries(_LAST_LINK_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: every specific value in the answer is on the page",
        _nothing_invented,
        SpecCategory.PROVENANCE,
    )

    # What is MEASURED — the draw's own structured fields, compared across the cohort.
    #
    # Variance is ORTHOGONAL to correctness: it does not ask whether a value is right, only
    # which samples diverge from the pack.  So `reason` belongs here even though this case
    # asserts the draw lands on `EXTRACTED` and `MicroContextResult` populates `reason` only
    # on `NOT_PRESENT` — a sample that came back NOT_PRESENT carries a reason the others do
    # not, and that is exactly an outlier worth surfacing, at finer grain than `outcome` gives.
    # On a run where every draw succeeds it reads nothing on all fifteen and is flagged blind,
    # which is the honest "nothing diverged this run" rather than a broken axis.
    #
    # No tool sequence and no reply spread: a single call makes neither.
    cohort.measure(
        output_field(EXTRACT_OUTCOME),
        output_field(EXTRACT_VALUE, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_REASON, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_MISSING, consequence=Consequence.COSMETIC),
    )


# ── Case 3: the same shortfall on a prose page ────────────────────────────────

_PARTLY_PRESENT_ON_A_PROSE_PAGE = ExtractFixture(
    case_id="extract-fields-partly-present-on-a-prose-page",
    url=_SECTION_URL,
    page=_SECTION,
    instruction="the headline, the byline and the published time for each story",
    behaviour=(
        "In the browse-extract micro-context, when a prose section front carries two of the "
        "three things the instruction named, Penny comes back with the two."
    ),
    expectations=(
        FieldExpectation("headline", _SECTION_HEADLINE_ANCHOR),
        FieldExpectation("byline", _SECTION_BYLINE_ANCHOR),
        FieldExpectation("published time"),
    ),
)

# What the five wordings (the instruction and `_PROSE_PAGE_PHRASINGS` below) call the part
# the page lacks, in their own words: "the published time for each story", "when it was
# published", "its publication time", "the times they went up", "the time it was posted".
# The MISSING line echoes whichever wording the sample was given (#2360 round 3), so the
# claim reads any of these rather than one field name.
_PROSE_PAGE_SHORTFALL_WORDS = (
    "published time",
    "when it was published",
    "publication time",
    "times they went up",
    "time it was posted",
)

_PROSE_PAGE_PHRASINGS = (
    "for each story: its headline, who wrote it, and when it was published",
    "each story's title, its author and its publication time",
    "pull out the headlines, the bylines and the times they went up",
    "the title of every story, the name of whoever wrote it, and the time it was posted",
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_prose_section_front_short_of_one_thing_still_reads(
    extractor_eval: ExtractorEval, model: str
) -> None:
    """The second PAGE SHAPE the contract has to hold on: prose with a byline under each
    headline, where the sibling case is markdown links.

    A world is fixed across a cohort's arms, so a page that is prose where the other is a link
    index is a world of its own and therefore a case of its own — the wording axis is already
    spent on the five instructions.  What it buys is the thing one page cannot say: that the
    per-field read follows the page's CONTENT and not the markup the content script happened to
    hand over.

    **The STORE category is empty for this case, and that is the correct report.**  A
    micro-context is one call that returns a typed result — it moves no machine and writes
    nothing to any store — so there is no store claim to make.  The empty section says the
    shape has nothing to store, not that nobody ran the checklist.
    """
    cohort = await extractor_eval(
        case_id=_PARTLY_PRESENT_ON_A_PROSE_PAGE.case_id,
        behaviour=_PARTLY_PRESENT_ON_A_PROSE_PAGE.behaviour,
        area=Area.READING_A_PAGE,
        model=model,
        url=_PARTLY_PRESENT_ON_A_PROSE_PAGE.url,
        page=_PARTLY_PRESENT_ON_A_PROSE_PAGE.page,
        reading=_PARTLY_PRESENT_ON_A_PROSE_PAGE.reading,
        instruction=_PARTLY_PRESENT_ON_A_PROSE_PAGE.instruction,
        also_instructed=_PROSE_PAGE_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — which of the closed outcomes the draw committed to
    cohort.claim(
        "state: the draw read the page rather than reporting it empty",
        _landed_on(_PARTLY_PRESENT_ON_A_PROSE_PAGE),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the docstring.

    # PROVENANCE — what the page supplies arrived, and nothing else did
    cohort.claim(
        "state: the answer carries the headline the page supplies",
        _carries(_SECTION_HEADLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the byline the page supplies",
        _carries(_SECTION_BYLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: every specific value in the answer is on the page",
        _nothing_invented,
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the stated shortfall names the published time the page does not give",
        _names_the_shortfall(_PARTLY_PRESENT_ON_A_PROSE_PAGE, _PROSE_PAGE_SHORTFALL_WORDS),
        SpecCategory.PROVENANCE,
    )

    # What is MEASURED — the draw's own structured fields, compared across the cohort.  No
    # tool sequence and no reply spread: a single call makes neither.  `reason` is measured
    # though a correct draw leaves it empty, for the reason spelled out on
    # `extract-fields-partly-present`: an all-empty reading is flagged BLIND rather
    # than counted as agreement, and a sample that came back NOT_PRESENT carries a reason
    # the others do not — an outlier at finer grain than `outcome` gives.
    cohort.measure(
        output_field(EXTRACT_OUTCOME),
        output_field(EXTRACT_VALUE, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_REASON, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_MISSING, consequence=Consequence.COSMETIC),
    )


# ── Case 4: a page with none of it (the honesty guard) ────────────────────────

_NONE_PRESENT = ExtractFixture(
    case_id="extract-fields-none-present",
    url=_FRONT_PAGE_URL,
    page=_FRONT_PAGE,
    instruction="the closing price of each company named and its ticker symbol",
    behaviour=(
        "In the browse-extract micro-context, when the page carries none of what the "
        "instruction named, Penny says plainly that it carries none of it rather than "
        "answering anyway."
    ),
    expectations=(
        FieldExpectation("closing price"),
        FieldExpectation("ticker symbol"),
    ),
    reading=PageReading.LINK_LIST,
)

_NONE_PRESENT_PHRASINGS = (
    "for every company mentioned: what its shares closed at and its ticker",
    "each company's closing share price and stock ticker symbol",
    "pull out the closing prices and the ticker symbols of the companies here",
    "the ticker symbol of every company named and the price its stock closed at",
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_page_carrying_none_of_it_still_says_so(
    extractor_eval: ExtractorEval, model: str
) -> None:
    """The over-correction guard, and the ``NOT_PRESENT`` half of the pair the module
    docstring names.

    A read that degrades per field must not degrade into answering anyway: this page carries
    neither of the two things named, so the honest answer is that it carries neither, and
    anything else is a value the page never held.  It is a case rather than a direction stated
    inside ``extract-fields-partly-present`` because its EXPECTED OUTCOME is the other one —
    a behaviour that always answered ``EXTRACTED:`` would pass that case and fail this.

    **Two categories report empty here, for two different reasons.**  STORE is empty for the
    shape's reason: one call returns a typed result and writes to no store.  The *nothing
    omitted* half of PROVENANCE is empty for this CASE's reason: nothing on this page answers
    the instruction, so there is no supplied thing for an answer to carry and no ``_carries``
    claim to make.  The *nothing invented* half is the whole point and is made below.

    **What is not asserted at all is the REASON's quality.**  It is free prose, so any check on
    it is the phrasing match ``docs/eval-case-design.md`` forbids; it rides the variance table
    instead.
    """
    cohort = await extractor_eval(
        case_id=_NONE_PRESENT.case_id,
        behaviour=_NONE_PRESENT.behaviour,
        area=Area.READING_A_PAGE,
        model=model,
        url=_NONE_PRESENT.url,
        page=_NONE_PRESENT.page,
        reading=_NONE_PRESENT.reading,
        instruction=_NONE_PRESENT.instruction,
        also_instructed=_NONE_PRESENT_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — which of the closed outcomes the draw committed to.  ``NOT_PRESENT`` is derived
    # from this fixture's own facts: the page carries neither named thing.
    cohort.claim(
        "state: the draw reports the page carries none of it",
        _landed_on(_NONE_PRESENT),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the docstring.

    # PROVENANCE — the *nothing invented* direction alone.  Its *nothing omitted* twin has
    # nothing to claim: the page supplies none of what was asked for, so there is no anchor an
    # answer could carry, and writing a vacuous claim would print a rate for a question nobody
    # asked.
    cohort.claim(
        "state: every specific value in the answer is on the page",
        _nothing_invented,
        SpecCategory.PROVENANCE,
    )

    # `value` and `reason` swap roles here relative to the sibling cases: a correct draw
    # populates `reason` and leaves `value` empty, so `value` reads nothing on every sample and
    # is flagged blind — the honest "nothing diverged this run" — while `reason` carries the
    # spread.  Both stay COSMETIC: they are prose, and a divergence in them is a different word
    # for the same outcome.  `missing` reads nothing on every sample too — a MISSING: line only
    # ever pairs with EXTRACTED, which this case never lands on.
    cohort.measure(
        output_field(EXTRACT_OUTCOME),
        output_field(EXTRACT_VALUE, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_REASON, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_MISSING, consequence=Consequence.COSMETIC),
    )


# ── Case 5: the top headlines off a real-size front page (#2360) ─────────────

_TOP_HEADLINES = ExtractFixture(
    case_id="extract-top-headlines-off-a-front-page",
    url=_FRONT_PAGE_URL,
    page=_FRONT_PAGE,
    instruction=(
        "the top 3 headlines on this page, with the title, a one-sentence description, "
        "and a link for each"
    ),
    behaviour=(
        "In the browse-extract micro-context, when a real-size front page carries the top "
        "headlines and their links but not their descriptions, Penny comes back with the "
        "headlines and links and invents no description."
    ),
    expectations=(
        FieldExpectation("headline-1", _TOP1_HEADLINE_ANCHOR),
        FieldExpectation("link-1", _TOP1_LINK_ANCHOR),
        FieldExpectation("headline-2", _TOP2_HEADLINE_ANCHOR),
        FieldExpectation("link-2", _TOP2_LINK_ANCHOR),
        FieldExpectation("headline-3", _TOP3_HEADLINE_ANCHOR),
        FieldExpectation("link-3", _TOP3_LINK_ANCHOR),
        FieldExpectation("description"),
    ),
    reading=PageReading.LINK_LIST,
)

# What the five wordings (the instruction and `_TOP_HEADLINES_PHRASINGS` below) call the
# part the page lacks: four say "description", the last says "a one-sentence summary" —
# the MISSING line echoes whichever wording the sample was given (#2360 round 3), so the
# claim reads any of these rather than one field name.
_TOP_HEADLINES_SHORTFALL_WORDS = ("description", "summary")

_TOP_HEADLINES_PHRASINGS = (
    "the three top headlines here, each with its title, a one-sentence description and a link",
    "give me the top three stories: the headline, a one-sentence description, and the link "
    "for each",
    "what are the top 3 headlines, with a title, a short one-sentence description and a link "
    "for each one",
    "pull the top three headlines from this page, including the title, a one-sentence "
    "summary, and a link for each",
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_real_size_front_page_gives_the_top_headlines_quickly(
    extractor_eval: ExtractorEval, model: str
) -> None:
    """The failure as it was actually seen (#2360): asked for the top headlines off a real
    news front page, the extractor reasoned until the browse tool's 300s timeout cancelled
    it, re-reading the whole 90-link list while arguing whether a headline without its
    description is a partial unit or a missing one.  ``extract-fields-partly-present`` holds
    the same shortfall at small scale (five links); this case holds it at the size the
    incident actually showed it: a ~90-link, ~9K-character page, where the loop over "which
    of what was asked is present" has far more to re-read before it can settle.

    **The STORE category is empty for this case, and that is the correct report.**  A
    micro-context is one call that returns a typed result — it moves no machine and writes
    nothing to any store — so there is no store claim to make.  The empty section says the
    shape has nothing to store, not that nobody ran the checklist.
    """
    cohort = await extractor_eval(
        case_id=_TOP_HEADLINES.case_id,
        behaviour=_TOP_HEADLINES.behaviour,
        area=Area.READING_A_PAGE,
        model=model,
        url=_TOP_HEADLINES.url,
        page=_TOP_HEADLINES.page,
        reading=_TOP_HEADLINES.reading,
        instruction=_TOP_HEADLINES.instruction,
        also_instructed=_TOP_HEADLINES_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — which of the closed outcomes the draw committed to
    cohort.claim(
        "state: the draw read the page rather than reporting it empty",
        _landed_on(_TOP_HEADLINES),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the docstring.

    # PROVENANCE — each of the top three headlines and its link arrived, and nothing else did
    cohort.claim(
        "state: the answer carries the first headline the page supplies",
        _carries(_TOP1_HEADLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the first headline's link",
        _carries(_TOP1_LINK_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the second headline the page supplies",
        _carries(_TOP2_HEADLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the second headline's link",
        _carries(_TOP2_LINK_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the third headline the page supplies",
        _carries(_TOP3_HEADLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the third headline's link",
        _carries(_TOP3_LINK_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: every specific value in the answer is on the page",
        _nothing_invented,
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the stated shortfall names the descriptions the page does not give",
        _names_the_shortfall(_TOP_HEADLINES, _TOP_HEADLINES_SHORTFALL_WORDS),
        SpecCategory.PROVENANCE,
    )

    # What is MEASURED — the draw's own structured fields, compared across the cohort.  No
    # tool sequence and no reply spread: a single call makes neither.
    cohort.measure(
        output_field(EXTRACT_OUTCOME),
        output_field(EXTRACT_VALUE, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_REASON, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_MISSING, consequence=Consequence.COSMETIC),
    )


# ── Case 6: the top headlines past a later link's run-on label (#2453) ───────

_TOP_HEADLINES_PAST_A_RUNON_LABEL = ExtractFixture(
    case_id="extract-top-headlines-past-a-run-on-label",
    url=_FRONT_PAGE_URL,
    page=_FRONT_PAGE_WITH_RUNON_LABEL,
    instruction=_TOP_HEADLINES.instruction,
    behaviour=(
        "In the browse-extract micro-context, when a real-size front page carries the top "
        "headlines without descriptions and a later link's own label runs directly into its "
        "description with no separating space, Penny still comes back with the top "
        "headlines and links, inventing no description."
    ),
    expectations=_TOP_HEADLINES.expectations,
    reading=PageReading.LINK_LIST,
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_real_size_front_page_still_gives_the_top_headlines_past_a_run_on_label(
    extractor_eval: ExtractorEval, model: str
) -> None:
    """The sharper trigger behind #2453: the same real-size front page as
    ``extract-top-headlines-off-a-front-page``, with one more link spliced in just past the
    top three — a label whose own title runs directly into a further sentence with NO space
    or punctuation gap before it (mirroring the production incident's own page exactly).

    The production thinking traces read this run-on link correctly on its own terms, then
    used the pattern it had just found — no space after the title's own closing punctuation
    — as a reason to go back and re-litigate whether the top three headlines, whose internal
    colons and question marks are followed by ordinary spaces, might be hiding an
    un-signalled split too.  That second-guessing is what burned several thousand extra
    tokens and, on the worst draws, cost the response its own ``EXTRACTED:`` tag.  The ask
    and the top three headlines are unchanged from the sibling case; only the page gained
    this one link, below them, so the only variable is whether its run-on shape still
    derails a read of headlines that never had that shape.

    **The STORE category is empty for this case, and that is the correct report.**  A
    micro-context is one call that returns a typed result — it moves no machine and writes
    nothing to any store — so there is no store claim to make.  The empty section says the
    shape has nothing to store, not that nobody ran the checklist.
    """
    cohort = await extractor_eval(
        case_id=_TOP_HEADLINES_PAST_A_RUNON_LABEL.case_id,
        behaviour=_TOP_HEADLINES_PAST_A_RUNON_LABEL.behaviour,
        area=Area.READING_A_PAGE,
        model=model,
        url=_TOP_HEADLINES_PAST_A_RUNON_LABEL.url,
        page=_TOP_HEADLINES_PAST_A_RUNON_LABEL.page,
        reading=_TOP_HEADLINES_PAST_A_RUNON_LABEL.reading,
        instruction=_TOP_HEADLINES_PAST_A_RUNON_LABEL.instruction,
        also_instructed=_TOP_HEADLINES_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — which of the closed outcomes the draw committed to
    cohort.claim(
        "state: the draw read the page rather than reporting it empty",
        _landed_on(_TOP_HEADLINES_PAST_A_RUNON_LABEL),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the docstring.

    # PROVENANCE — each of the top three headlines and its link arrived, and nothing else did
    cohort.claim(
        "state: the answer carries the first headline the page supplies",
        _carries(_TOP1_HEADLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the first headline's link",
        _carries(_TOP1_LINK_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the second headline the page supplies",
        _carries(_TOP2_HEADLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the second headline's link",
        _carries(_TOP2_LINK_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the third headline the page supplies",
        _carries(_TOP3_HEADLINE_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the answer carries the third headline's link",
        _carries(_TOP3_LINK_ANCHOR),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: every specific value in the answer is on the page",
        _nothing_invented,
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the stated shortfall names the descriptions the page does not give",
        _names_the_shortfall(_TOP_HEADLINES_PAST_A_RUNON_LABEL, _TOP_HEADLINES_SHORTFALL_WORDS),
        SpecCategory.PROVENANCE,
    )

    # What is MEASURED — the draw's own structured fields, compared across the cohort.  No
    # tool sequence and no reply spread: a single call makes neither.  Token cost is the
    # measure this case exists for (#2453): the run-on link should cost nothing beyond the
    # sibling case's own cost, not several thousand extra tokens of re-litigation.
    cohort.measure(
        output_field(EXTRACT_OUTCOME),
        output_field(EXTRACT_VALUE, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_REASON, consequence=Consequence.COSMETIC),
        output_field(EXTRACT_MISSING, consequence=Consequence.COSMETIC),
    )


# Every fixture, for the deterministic coherence probe in ``make check``.
FIXTURES = (
    _ALL_PRESENT,
    _TOP_HEADLINES,
    _TOP_HEADLINES_PAST_A_RUNON_LABEL,
    _PARTLY_PRESENT,
    _PARTLY_PRESENT_ON_A_PROSE_PAGE,
    _NONE_PRESENT,
)
