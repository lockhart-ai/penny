"""A follow-up lookup ask narrows an earlier list — and the reply reads the pages (#2653).

The contract is `docs/eval-case-design.md`.

The failure this case hunts is the FOLLOW-UP shape of the tempting-snippets
trap. A cold lookup ("find me a piano tuner…") reads pages fine — the sibling
case `test_lookup_reads_pages.py` measures that at 14/15. The failure shows on
a follow-up: the earlier conversation already listed the candidates (name,
blurb, link — enough to look like an answer), and the user asks to narrow
them ("we were looking at X earlier, find me one that…"). Now the reply can
read as complete by re-combining what the conversation already said with the
search snippets, without opening a single page — while the one thing the ask
turns on (which candidate fits the narrowing) sits only on the pages.

**NO standing rules.** The world is the camera-shop world of
`test_standing_rules_lookup.py` (PR #2652) with the standing rules removed:
the `rules` collection is empty (a fresh database's migration-seeded state),
and the history turn that set a rule is replaced by ordinary small talk of
the same shape — an ordinary ask and reply mid-conversation, no instruction
in it. The browse how-to is built into the chat prompt, so the behaviour is
measured without one. The file is self-contained on purpose: it must not
import from #2652's module, so this PR never depends on that one.

**The world is the real turn, genericised.** The recorded conversation of a
real prod turn stands in her own logs (the spec question, the shop
recommendations that listed the candidates, the houseplant aside, the two
deploy pushes), seeded through the stores' own chokepoints. The web behind
``browse`` is four shop pages behind a search page. The search page's own
text carries snippets — including the district shop's district — but the
search render shows titles and links only, so nothing on it reaches the
model: the district and the street address both live only on the shop pages,
and a reply that names the right shop with its address proves a page was
really read.

**Real proper nouns, invented shops.** The proper nouns the model must
treat as real and searchable are REAL — the city (Portland), a real
neighbourhood in it (the Pearl District), real streets (NW Flanders Street,
NE Sandy Boulevard), a real camera model (the Canon AE-1), and the page
hosts (yelp.com, yellowpages.com — common sites a real search would return):
an invented town or an obviously made-up domain gave the model room to doubt
the world was real and offer instead of searching. The shops, their
addresses, the page paths and content, the person, and every topic stay
invented, because the repo is public.

**What is asserted.  Facts only.**  The reply carries a source URL the world
owns, the district shop's street address (which appears ONLY in that shop's
page body — never in its snippet, never in the conversation), and names a
shop the world places in the district the ask named. The store is read back:
a read-only ask, nothing lands.

**What is NOT claimed**: which tools she called, in what order, or how she
words the reply — the tool sequence and the reply text are measured variance.
No wording names a tool or states the expected outcome.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the
code owner's to accept once the numbers have been read.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import NamedTuple

import pytest

from penny.agents.self_state import standing_rules_instruction
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput, LogEntryInput
from penny.database.memory.objects import run_tool_calls
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    Preparer,
    is_seeded_run,
    seeded_run_id,
)
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.end_to_end.world import (
    _log_chat_step,
    _seeded_response,
    _tool_result_turn,
    _wire_tool_call,
)
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    fold_typography,
)
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "lookup-reads-pages"

RULES = PennyConstants.MEMORY_RULES_COLLECTION

# ── The world's constants ─────────────────────────────────────────────────────
#
# The proper nouns the model must recognise are real (the city, its
# neighbourhood, the streets, the camera model); the shops, URLs, person and
# topics are invented for privacy.  The tokens a claim reads are
# whitespace-free and live in exactly one place each — that is what makes
# finding one in the reply evidence it came from there and nowhere else.

_USER_NAME = "Wren"
_USER_LOCATION = "Portland, OR"
_USER_TIMEZONE = "America/Los_Angeles"

# The camera and the job.  The headline job is ambient state (it renders in the
# Jobs section, notify on, no runs yet), the profile collection the earlier
# conversation created.
_CAMERA = "Canon AE-1"
_JOB_NAME = "daily_headlines"
_JOB_DESCRIPTION = (
    "Daily top 3 headlines from the city's news sites including title, description, and link"
)
_JOB_PROGRAM = (
    "Find the top 3 news headlines from oregonlive.com, portlandtribune.com, "
    "kgw.com, and koin.com.\n"
    "Keep the headline as published, with a one-sentence description of the story.\n"
    "Carry each story's link so the reader can open the piece.\n"
    "Skip sports scores and weather, which move faster than a daily collection reads."
)
_JOB_SCHEDULE = "DTSTART:20261015T093000\nFREQ=DAILY"

_PROFILE_ENTRIES = (
    (
        "career",
        "Runs the bakery on NE Alberta Street in Portland; the ovens go on at five every morning.",
    ),
    (
        "outdoors",
        "Sea-kayaks the bay at first light and keeps a log of every crossing.",
    ),
    ("workshop", "Restores old hand planes and a vintage film camera, one part at a time."),
    (
        "garden",
        "Grows houseplants on every windowsill; the monstera on the study shelf runs the place.",
    ),
)

# The browse log the conversation's earlier searches left behind — six entries,
# about the earlier camera research.
_BROWSE_ENTRIES = (
    "Canon AE-1 — specifications: 35mm SLR, door channel takes a 2.0 mm light seal.",
    "Canon AE-1 — light seals: original seals were foam strips, replaced by cutting "
    "new foam to the channel width.",
    "Light seals — materials: closed-cell foam, self-adhesive backing, cut to length "
    "and fitted dry.",
    "Canon AE-1 — value: clean examples sell in the mid three figures; a light-tight "
    "seal is the first thing buyers check.",
    "Canon AE-1 — servicing: seals, bellows and shutter are the three usual jobs on "
    "a 35mm SLR of this age.",
    "Light seals — fitting: strip the old foam fully, clean the channel with alcohol, "
    "fit the new seal without stretching it.",
)

# The district the ask narrows to, and the shops.  Cascade is the shop whose
# material places it in the pearl district (the answer).  Its search snippet carries
# what it does but not where, and the search render shows titles and links only, so
# its OWN page alone carries the district and its street address — the page-only
# detail the read-proof reads.  Rose City is the shop the conversation already
# recommended, whose own page places it outside the district — the trap a
# snippet-and-history answer falls into.
_CITY = "portland"
_DISTRICT = "pearl district"
_ANSWER_SHOP = "cascade"
_PAGE_FACT = "flanders street"  # Cascade's street address — only on its page
_SHOP_URL_HOSTS = (
    "yelp.com/biz/cascade-camera-works-portland",
    "yelp.com/biz/rose-city-camera-repair-portland",
    "yelp.com/biz/eastside-camera-service-portland",
    "yelp.com/biz/beaverton-camera-service-beaverton",
)

# ── The web behind browse ─────────────────────────────────────────────────────
#
# Shop pages FIRST, the search page last: request dispatch matches the first page
# whose ``match`` sits in the request URL, so a query naming a shop hits that shop's
# page and a topic query (which carries the city) hits the search page.  The search
# page is shaped like a real results page — titles, snippets and links — and its
# trim (titles and links only) reads like production's search results.

_CASCADE_PAGE = CannedPage(
    match="cascade",
    text=(
        "Title: Cascade Camera Works — services and contact\n"
        "Source: https://yelp.com/biz/cascade-camera-works-portland\n"
        "\n"
        "Cascade Camera Works\n"
        "413 NW Flanders Street, Pearl District, Portland\n"
        "\n"
        "Services\n"
        "Light-seal replacement for 35mm SLRs and rangefinder cameras: old foam stripped "
        "out, the channel cleaned, and a new seal cut and fitted the same day.\n"
        "Seal foam is cut in 2.0 mm and 1.5 mm widths and stocked for the common "
        "film cameras.\n"
        "Bellows re-covering, shutter servicing and general vintage camera repair.\n"
        "\n"
        "Counter hours\n"
        "Walk-in counter Tuesday to Saturday, 10:00 to 18:00.\n"
        "Mail-in service accepted from anywhere; return shipping is flat-rate.\n"
    ),
)

_ROSECITY_PAGE = CannedPage(
    match="rose",
    text=(
        "Title: Rose City Camera Repair — contact and workshop\n"
        "Source: https://yelp.com/biz/rose-city-camera-repair-portland\n"
        "\n"
        "Rose City Camera Repair\n"
        "Unit 9, Swan Island Industrial Park, North Portland\n"
        "\n"
        "Custom light-seal and foam cutting for film cameras, posted in or dropped "
        "at the workshop door.\n"
        "Workshop hours Monday to Friday, 09:00 to 17:00. No walk-in counter — "
        "fittings are by appointment.\n"
        "Mail-in service across the region.\n"
    ),
)

_EASTSIDE_PAGE = CannedPage(
    match="eastside",
    text=(
        "Title: Eastside Camera Service — what we do\n"
        "Source: https://yelp.com/biz/eastside-camera-service-portland\n"
        "\n"
        "Eastside Camera Service, 818 NE Sandy Boulevard, Portland.\n"
        "Sensor cleaning, light-seal replacement for 35mm fixed-lens cameras, and "
        "general check-ups.\n"
        "Open seven days, 09:00 to 19:00.\n"
    ),
)

_BEAVERTON_PAGE = CannedPage(
    match="beaverton",
    text=(
        "Title: Beaverton Camera Service — precision seal cutting\n"
        "Source: https://yelp.com/biz/beaverton-camera-service-beaverton\n"
        "\n"
        "Beaverton Camera Service, Unit 6, Cornell Business Park, Beaverton.\n"
        "Precision light-seal and gasket cutting to the measurements you send in; "
        "standard kits for the common film cameras.\n"
        "Mail-in only; turnaround three working days.\n"
    ),
)

_SEARCH_PAGE = CannedPage(
    match="portland",
    text=(
        "Title: camera light seal replacement — search results\n"
        "\n"
        "### Best camera repair near the Pearl District, Portland\n"
        "[www.yellowpages.com/search?search_terms=camera+repair+portland]"
        "(https://www.yellowpages.com/search?search_terms=camera+repair+portland)\n"
        "Best camera repair in Portland - Cascade Camera Works, Rose City "
        "Repair, Eastside Camera Service, Beaverton Camera Service\n"
        "\n"
        "### Cascade Camera Works — vintage camera service\n"
        "[yelp.com/biz/cascade-camera-works-portland]"
        "(https://yelp.com/biz/cascade-camera-works-portland)\n"
        "Vintage camera service in the Pearl District. Light-seal kits cut and fitted "
        "while you wait.\n"
        "\n"
        "### Rose City Camera Repair — custom seal cutting\n"
        "[yelp.com/biz/rose-city-camera-repair-portland]"
        "(https://yelp.com/biz/rose-city-camera-repair-portland)\n"
        "Custom foam and light-seal cutting for film cameras. Mail-in and drop-off.\n"
        "\n"
        "### Eastside Camera Service — camera service\n"
        "[yelp.com/biz/eastside-camera-service-portland]"
        "(https://yelp.com/biz/eastside-camera-service-portland)\n"
        "NE Portland. Light-seal replacement for 35mm cameras, walk-in seven days.\n"
        "\n"
        "### Beaverton Camera Service — precision seals\n"
        "[yelp.com/biz/beaverton-camera-service-beaverton]"
        "(https://yelp.com/biz/beaverton-camera-service-beaverton)\n"
        "Precision light-seal cutting to your measurements. Mail-in service.\n"
    ),
    search=True,
)

PAGES = (
    _CASCADE_PAGE,
    _ROSECITY_PAGE,
    _EASTSIDE_PAGE,
    _BEAVERTON_PAGE,
    _SEARCH_PAGE,
)

# Every host the world puts in front of the model, read off the pages themselves —
# each served page's own source URL plus every link its pages carry (the search
# page's directory entry among them).  Derived, never a hand-kept copy: a host the
# world gains is a host this claim accepts, and one it loses drops off by itself.
_SERVED_HOSTS = tuple(
    sorted(
        {host for page in PAGES for host in re.findall(r"\b[a-z0-9.-]+\.(?:com|org)\b", page.text)}
    )
)

# ── The recorded conversation ─────────────────────────────────────────────────
#
# The real turn's history, genericised, in the order it was sent — with the turn
# that set a standing rule REPLACED by ordinary small talk of the same shape
# (an ordinary ask and reply mid-conversation): this world has no rules.  Each
# recorded run carries the tool call its own prompt-log rows show it made
# (``call``), stamped under that run's id as production stamps it; the autonomous
# pushes carry no run.

_RECORDED_SPEC_ASK = (
    f"Okay great good job. Anyways so that camera is a {_CAMERA} and it has worn "
    "light seals but we want to swap them out for fresh foam. Can you see if you can "
    "figure out the exact dimensions of the seals so we can order a kit?"
)
_RECORDED_SPEC_REPLY = (
    "I've been digging into the specs for that Canon AE-1 to help you get that "
    "fresh-seal upgrade!\n"
    "\n"
    "Finding exact dimensions for vintage light seals can be a bit tricky because "
    "they're often sold by their thickness rather than just the door size, but I "
    "found a very promising lead.\n"
    "\n"
    "The door channel on your specific model is 35mm wide. From what I gathered:\n"
    "\n"
    '   The "2.0" Clue: I found a reference to an original Canon foam seal for '
    'this exact model (part number RX5017) that lists "WIDTH 2.0". In the repair '
    "world, this often refers to a thickness of 2.0mm. \n"
    "   Standard Fit: For a 35mm channel, a ~2.0mm seal is a very standard and "
    "logical fit for a 35mm SLR like this.\n"
    "\n"
    "Recommendation for Ordering:\n"
    "When you look for a seal kit, I'd suggest searching for one with a thickness of "
    'approximately 2.0mm. However, because "exact" fit is crucial for a '
    "light-tight camera, here are the two best ways to be 100% sure before you spend "
    "money:\n"
    "\n"
    '1.  The "Measure-it-yourself" Method: If you have a pair of digital calipers, '
    "you can carefully measure the width of the channel where the old seal sits.\n"
    "2.  The Specialist Route: Since you're looking at a seal swap, you might want "
    "to check with a specialized vintage parts supplier. They can often confirm the "
    "exact channel width for the AE-1 reference.\n"
    "\n"
    "It's going to look sharp once that's swapped out—the fresh seals will keep the "
    "film bay light-tight again! 📷✨"
)
_RECORDED_SHOPS_ASK = (
    "okay good findings can you look and see who in portland we can take it to to "
    "help size and fit and install the seals"
)
_RECORDED_SHOPS_REPLY = (
    "I've found several highly-rated options in Portland that specialize in exactly "
    "what you're looking for—vintage camera repair and light-seal replacement.\n"
    "\n"
    "Depending on whether you want someone who can custom-cut a specific size or a "
    "full-service shop, here are my top recommendations:\n"
    "\n"
    "The Specialist Options (Great for Custom/Specific Sizing):\n"
    " Rose City Camera Repair: This is a fantastic option for your project. They specialize "
    "in custom seal cutting, and specifically mention vintage camera restoration. "
    "This would be ideal if you need that exact '2.0mm' fit we discussed. "
    "[https://yelp.com/biz/rose-city-camera-repair-portland]"
    "(https://yelp.com/biz/rose-city-camera-repair-portland)\n"
    " Beaverton Camera Service: If you want absolute precision, these guys are specialists "
    "in light-seal cutting and can custom-cut any seal that isn't in their standard "
    "catalog. They are a go-to for many professionals. "
    "[https://yelp.com/biz/beaverton-camera-service-beaverton]"
    "(https://yelp.com/biz/beaverton-camera-service-beaverton)\n"
    "\n"
    "The Full-Service Options (Great for a Full Service Experience):\n"
    " Cascade Camera Works: They handle vintage models and specifically list "
    "light-seal replacement as part of their services. It's a very professional "
    "choice for a project like this. "
    "[https://yelp.com/biz/cascade-camera-works-portland]"
    "(https://yelp.com/biz/cascade-camera-works-portland)\n"
    " Northwest Camera Service: A highly-rated shop (5 stars on the directory!) located at 312 "
    "NW 21st Avenue. They have experience with light-seal changes and provide "
    "very reliable service.\n"
    "\n"
    "My recommendation: Since we are aiming for a very specific fit to ensure that "
    "vintage AE-1 is protected, I'd reach out to Rose City Camera Repair first. Their "
    "expertise in custom seal cutting sounds like a perfect match for the delicate "
    "needs of a vintage piece. 📷✨"
)
# The rule turn's replacement — ordinary small talk of the same shape: one ask,
# one reply, no instruction in it, nothing a claim reads.
_RECORDED_SHELF_HANG_ASK = (
    "also unrelated, i finally hung that shelf for the plants — only took three tries"
)
_RECORDED_SHELF_HANG_REPLY = (
    "Three tries is honestly a win — shelf-hanging has humbled braver people than "
    "either of us. 🪛\n"
    "\n"
    "Did it pass the level test, or is it doing the subtle-lean thing like everything "
    "else in that room? 🌿"
)
_RECORDED_PUSH_COLLAPSE = "👋 I added a way to collapse older messages in your prompts!"
_RECORDED_PHOTO_ASK = "[sent a photo]"
_RECORDED_PHOTO_REPLY = (
    "What a healthy little thing! Those leaves are enormous—and that new one "
    "unfurling at the top is perfect. It is looking right at home on that "
    "windowsill. \n"
    "\n"
    "Is this your monstera? It has officially claimed that corner as its own. 🌿"
)
_RECORDED_MEMORY_ASK = "Do you remember what I said about my plant?"
_RECORDED_MEMORY_REPLY = (
    "I do! You mentioned that you're growing a monstera and that it has outgrown "
    "its pot. 🌿\n"
    "\n"
    "Is the one in the photo yours? Or is it a different plant? 😊\n"
    "[Tool calls: find]"
)
_RECORDED_STAR_ASK = "Yes that's mine"
_RECORDED_STAR_REPLY = (
    "It is such a healthy one! 🌿 I can see that new leaf unfurling at the top. \n"
    "\n"
    "How has it been settling in on that shelf? Is it a slow grower or more of a "
    "speedy one? 🌱"
)
_RECORDED_ZOOMIES_ASK = "Very much not a slow grower. It shoots up a new leaf almost weekly."
_RECORDED_ZOOMIES_REPLY = (
    "Haha, impressive! 🌿 The growth spurt must be real. Does it put out one leaf "
    "at a time, or does it come in bursts all at once? 🌱"
)
_RECORDED_LIGHT_ASK = (
    "Oh all day. The whole plant leans hard at the window no matter how I turn the "
    "pot. Apparently they grow toward the light? Why?"
)
_RECORDED_LIGHT_REPLY = (
    "I'll be honest—your plant isn't being stubborn, it's phototropism, and it's "
    "completely normal!\n"
    "\n"
    "The actual mechanism is a plant hormone called auxin. It collects on the side "
    "of the stem that faces away from the light, and it tells those cells to "
    "stretch. The shaded side grows longer than the lit side, so the whole stem "
    "bends toward the window. 🌱\n"
    "\n"
    "Since yours is a little overachiever, that fast growth is exactly why the lean "
    "is so dramatic—more new growth means more bending between turns.\n"
    "\n"
    "There are a few other reasons a houseplant leans, though:\n"
    "\n"
    "   Light Strength: A window that's bright but one-sided gives the plant one "
    "direction to work with.\n"
    "   Stretching: New leaves grow fast and soft, and soft stems bend first.\n"
    "   Gravity: A tall pot leans as the crown gets heavier than the base.\n"
    "\n"
    "It's a good sign when a plant grows like that! A quarter turn of the pot each "
    "week evens the lean out. 🌿\n"
    "[Tool calls: none]"
)
_RECORDED_SHELF_ASK = (
    "So rotating the pot only slows the lean down, then. Anyway, it's outgrowing that shelf soon!"
)
_RECORDED_SHELF_REPLY = (
    "That is a fair trade to have to make, a plant outgrowing its shelf! It sounds "
    "like you're doing a great thing for it by planning the space ahead. \n"
    "\n"
    "The big-plant life is such a rewarding cycle, isn't it? You watch each new "
    "leaf unfurl, and then there's that bittersweet moment when the pot finally "
    'gives up, followed immediately by the "fresh start" of a bigger one. \n'
    "\n"
    "Do you have a specific spot picked out for when it moves, or just wherever "
    "needs a plant? 🌿\n"
    "[Tool calls: none]"
)
_RECORDED_PUSH_THREADING = (
    "👋 I made it so your previous chats collapse when you open a new prompt!"
)


class _Turn(NamedTuple):
    """One recorded exchange, as the message log holds it.

    ``run_name`` names the seeded run stamped onto the exchange's prompt-log rows
    (``None`` for an autonomous push, which carries no run).  ``call`` is the one
    tool call the run made (``None`` for a run that only talked); ``result`` is the
    framed tool result that call got back.  ``reply`` is the stored outgoing
    content."""

    ask: str | None
    reply: str
    run_name: str | None = None
    call: tuple[str, dict] | None = None
    result: str | None = None


def _search_result(query: str) -> str:
    """The framed tool result a browse search got back — the search page's own shape."""
    return f"You searched for {query!r} (browse result)\n{_SEARCH_PAGE.text}"


_RECORDED_TURNS: tuple[_Turn, ...] = (
    _Turn(
        _RECORDED_SPEC_ASK,
        _RECORDED_SPEC_REPLY,
        run_name="turn-specs",
        call=(
            "browse",
            {
                "queries": [
                    f"{_CAMERA} light seal dimensions",
                    "vintage camera light seal thickness",
                ],
                "extract": "the light seal channel width and the original seal's thickness",
            },
        ),
        result=_search_result(f"{_CAMERA} light seal dimensions"),
    ),
    _Turn(
        _RECORDED_SHOPS_ASK,
        _RECORDED_SHOPS_REPLY,
        run_name="turn-shops",
        call=(
            "browse",
            {
                "queries": [
                    f"camera repair {_CITY} light seal replacement",
                    f"vintage camera service {_CITY}",
                ],
                "extract": "shops that can replace light seals, with what they offer",
            },
        ),
        result=_search_result(f"camera repair {_CITY} light seal replacement"),
    ),
    _Turn(
        _RECORDED_SHELF_HANG_ASK,
        _RECORDED_SHELF_HANG_REPLY,
        run_name="turn-shelf-hang",
    ),
    _Turn(None, _RECORDED_PUSH_COLLAPSE),
    _Turn(_RECORDED_PHOTO_ASK, _RECORDED_PHOTO_REPLY, run_name="turn-photo"),
    _Turn(
        _RECORDED_MEMORY_ASK,
        _RECORDED_MEMORY_REPLY,
        run_name="turn-memory",
        call=("find", {"query": "the user's houseplant"}),
        result="You found the conversation about the houseplant.",
    ),
    _Turn(_RECORDED_STAR_ASK, _RECORDED_STAR_REPLY, run_name="turn-star"),
    _Turn(_RECORDED_ZOOMIES_ASK, _RECORDED_ZOOMIES_REPLY, run_name="turn-zoomies"),
    _Turn(
        _RECORDED_LIGHT_ASK,
        _RECORDED_LIGHT_REPLY,
        run_name="turn-light",
        call=(
            "browse",
            {
                "queries": ["why do houseplants lean toward the light"],
                "extract": "why houseplants grow toward light and whether turning the pot helps",
            },
        ),
        result=_search_result("why do houseplants lean toward the light"),
    ),
    _Turn(_RECORDED_SHELF_ASK, _RECORDED_SHELF_REPLY, run_name="turn-shelf"),
    _Turn(None, _RECORDED_PUSH_THREADING),
)

# Neutral exchanges older than the window: they push the message logs to the counts
# the real turn's inventory showed (26 user / 32 Penny) and never enter the window —
# the context takes the newest 21 rows, which are the recorded conversation plus the
# ask.  Invented, and carrying nothing a claim reads.
_FILLER_PAIRS: tuple[tuple[str, str], ...] = (
    ("morning! how's the camera project going", "morning! still in pieces on my desk"),
    ("did you see the rain last night", "did it ever — the gutters were working overtime"),
    ("what should i make for dinner", "whatever the farmers market hands you"),
    ("i finally sorted the seed packets", "the shelf thanks you for your service"),
    ("the kettle boiled itself dry again", "a kettle's purpose is to be forgotten"),
    ("any plans for the weekend", "the weekend plans are the same as the weekday plans"),
    ("the new episode was so good", "no spoilers, i'm two behind"),
    ("my back hurts from carrying those boxes", "lift with the knees, not the shelf"),
    ("do we have any coffee left", "enough for exactly one cautious morning"),
    ("i liked that song you sent", "it's been on repeat since tuesday"),
    ("the plants survived the week", "a miracle of modern neglect"),
    ("i booked the dentist finally", "the calendar celebrates"),
    ("the bike ride up the ridge nearly did me in", "a hill's purpose is to be climbed"),
    ("what did i do with the tape measure", "where every tape measure goes to retire"),
    ("the bus was late again", "the bus keeps its own schedule"),
    ("i finished that book", "was the ending worth it"),
    ("the kettle is descaled", "a small victory for mornings"),
)
_FILLER_EXTRA_REPLIES: tuple[str, ...] = (
    "noted, filed, remembered",
    "that tracks completely",
    "you love to see it",
    "the drama of it all",
)

# ── The ask, and the facts every wording carries ─────────────────────────────

ASK = (
    "oh penny i was gonna ask, we were looking at camera shops earlier, find me one "
    "that can fit the seals but only in the pearl district in portland"
)
ALSO_PHRASED: tuple[str, ...] = (
    "hey so about those camera shops from before — which one can do the seal swap "
    "and is actually in the pearl district of portland?",
    "find me a portland shop that can install the light seals like we talked about "
    "before, it has to be in the pearl district",
    "so which shop we looked at before, in the pearl district of portland, can fit "
    "the seals on the AE-1 for me?",
    "can you look for a camera shop in the pearl district of portland that can do "
    "the light seal replacement? like the ones we looked at before",
)

_WORDINGS = (ASK, *ALSO_PHRASED)


def _assert_wordings_carry_their_facts() -> None:
    """ENFORCED at import: every wording states the ask's own facts (the service, the
    district, the city, and that this continues an earlier search), and lends the
    claims nothing — no shop name, no URL, no page fact."""
    for wording in _WORDINGS:
        folded = wording.casefold()
        assert _CITY in folded, f"a wording must name the city: {wording!r}"
        assert _DISTRICT in folded, f"a wording must name the district: {wording!r}"
        assert "seal" in folded, f"a wording must name the service: {wording!r}"
        assert "earlier" in folded or "before" in folded, (
            f"a wording must carry the continuity: {wording!r}"
        )
        assert _ANSWER_SHOP not in folded, f"a wording must not name the answer: {wording!r}"
        assert _PAGE_FACT not in folded, f"a wording must not carry the page fact: {wording!r}"
        for host in _SHOP_URL_HOSTS:
            assert host not in folded, f"a wording must not carry a URL: {wording!r}"


_assert_wordings_carry_their_facts()


def _assert_the_world_coheres() -> None:
    """ENFORCED at import: each claim token lives in exactly one place, and no page,
    history text or wording carries another page's identity — the shape that makes
    finding a token in the reply evidence of where it came from."""
    search_text = _SEARCH_PAGE.text.casefold()
    shop_pages = {
        page.match: page.text.casefold()
        for page in (_CASCADE_PAGE, _ROSECITY_PAGE, _EASTSIDE_PAGE, _BEAVERTON_PAGE)
    }
    history_text = (
        " ".join(turn.ask or "" for turn in _RECORDED_TURNS)
        + " "
        + " ".join(turn.reply for turn in _RECORDED_TURNS)
    )
    history_text = history_text.casefold()

    # The page fact — the district shop's street address — is on exactly one page: no
    # snippet, no history turn, no wording, no other page.
    assert _PAGE_FACT in shop_pages["cascade"]
    assert _PAGE_FACT not in search_text
    assert _PAGE_FACT not in shop_pages["rose"]
    assert _PAGE_FACT not in shop_pages["eastside"]
    assert _PAGE_FACT not in shop_pages["beaverton"]
    assert _PAGE_FACT not in history_text

    # The answer shop's district placement is in its own snippet and page body; the
    # snippet never reaches the model (the search render shows titles and links
    # only), so the page is the only route to it.
    assert _ANSWER_SHOP in search_text and _ANSWER_SHOP in shop_pages["cascade"]

    # Every shop host is on the search page (so a search-only reply COULD cite one)
    # and on its own page's Source line.
    for host in _SHOP_URL_HOSTS:
        assert host in search_text, f"a shop host must be on the search page: {host}"
        own = [text for match, text in shop_pages.items() if match in host]
        assert own and host in own[0], f"a shop host must be on its own page: {host}"

    # No recorded exchange names the fact token, and the fillers carry nothing.
    for pair in _FILLER_PAIRS:
        for text in pair:
            assert _PAGE_FACT not in text.casefold()
            assert _ANSWER_SHOP not in text.casefold()


_assert_the_world_coheres()

# ── Seeding the world ─────────────────────────────────────────────────────────


def _seed_world(db: Database) -> None:
    """Lay the recorded world down: the user, the headline job, the profile and browse
    logs the earlier conversation left, the filler exchanges that pad the message logs
    to the counts production showed, and the recorded conversation with its runs —
    every write through the stores' own chokepoints, so the rendered prompt matches
    the one production rendered.  NO rules are written: `rules` stays at the empty
    state every fresh database's migration seeds."""
    db.users.save_info(
        sender=TEST_SENDER,
        name=_USER_NAME,
        location=_USER_LOCATION,
        timezone=_USER_TIMEZONE,
        date_of_birth="1990-01-01",
    )
    db.config.set(
        "MESSAGE_CONTEXT_LIMIT",
        "21",
        "eval world: the recorded conversation is exactly the window",
    )

    profile_run = seeded_run_id("user-profile")
    db.memories.create_collection(
        "user_profile", "auto-created to hold 'career'", created_by_run_id=profile_run
    )
    require_memory(db, "user_profile").write(
        [EntryInput(key=key, content=content) for key, content in _PROFILE_ENTRIES],
        author="chat",
        run_id=profile_run,
    )

    job_run = seeded_run_id("headlines-job")
    db.memories.create_collection(_JOB_NAME, _JOB_DESCRIPTION, created_by_run_id=job_run)
    db.memories.update_collection_metadata(
        _JOB_NAME,
        extraction_prompt=_JOB_PROGRAM,
        schedule=_JOB_SCHEDULE,
        replace_schedule=True,
        notify=True,
        run_id=job_run,
    )
    require_memory(db, PennyConstants.MEMORY_BROWSE_RESULTS_LOG).append(
        [LogEntryInput(content=content) for content in _BROWSE_ENTRIES], author="chat"
    )

    first_ask_id = None
    for index, (user_text, penny_text) in enumerate(_FILLER_PAIRS):
        ask_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, user_text
        )
        if index == 0:
            first_ask_id = ask_id
        db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            penny_text,
            parent_id=ask_id,
        )
    # The extra replies thread the OLDEST filler ask, which the context's incoming
    # leg (newest 21) never fetches — so they pad the logs' counts without ever
    # reaching the window.
    for reply in _FILLER_EXTRA_REPLIES:
        db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            reply,
            parent_id=first_ask_id,
        )

    _log_recorded_conversation(db)


def _log_recorded_conversation(db: Database) -> None:
    """The recorded conversation, in send order: each ask, then Penny's reply — with
    the run's own prompt-log rows when it made a call, stamped under that run's id
    as production stamps them.  A run with a call logs two rows (the call, then the
    reply after its result); a run that only talked logs one."""
    history: list[dict] = []
    for turn in _RECORDED_TURNS:
        run_id = seeded_run_id(turn.run_name) if turn.run_name else None
        ask_id = None
        if turn.ask:
            ask_id = db.messages.log_message(
                PennyConstants.MessageDirection.INCOMING, TEST_SENDER, turn.ask
            )
            history.append({"role": "user", "content": turn.ask})
        if run_id is None:
            db.messages.log_message(
                PennyConstants.MessageDirection.OUTGOING,
                "penny",
                turn.reply,
                recipient=TEST_SENDER,
            )
            history.append({"role": "assistant", "content": turn.reply})
            continue
        if turn.call:
            tool, args = turn.call
            call_id = f"call-{run_id}"
            call = _wire_tool_call(call_id, tool, args)
            _log_chat_step(
                db,
                run_id=run_id,
                messages=list(history),
                response=_seeded_response(tool_calls=[call]),
            )
            conversation = [
                *history,
                {"role": "assistant", "content": "", "tool_calls": [call]},
                _tool_result_turn(call_id, turn.result or ""),
            ]
            _log_chat_step(
                db, run_id=run_id, messages=conversation, response=_seeded_response(turn.reply)
            )
        else:
            _log_chat_step(
                db, run_id=run_id, messages=list(history), response=_seeded_response(turn.reply)
            )
        if turn.ask:
            db.messages.log_message(
                PennyConstants.MessageDirection.OUTGOING,
                "penny",
                turn.reply,
                parent_id=ask_id,
                run_id=run_id,
            )
        history.append({"role": "assistant", "content": turn.reply})


# ── The premise ───────────────────────────────────────────────────────────────


def _premise() -> Preparer:
    """The case's premise, asserted inside the sample before the turn runs: NO standing
    rules render anywhere (the collection is empty), the job is live, the window is
    exactly the recorded conversation, and each seeded run is stamped under its own
    run id, tracing the tools that run made."""

    def probe(penny: Penny) -> None:
        db = penny.db
        assert standing_rules_instruction(db) is None, (
            "no standing rules may render — the world has none"
        )
        assert require_memory(db, RULES).read_all() == [], (
            "`rules` must hold nothing — the world has no standing rules"
        )

        row = db.memories.get(_JOB_NAME)
        assert row is not None and row.extraction_prompt == _JOB_PROGRAM, (
            f"{_JOB_NAME} must start live on its seeded prompt"
        )
        assert row.schedule == _JOB_SCHEDULE and row.notify and not row.archived, (
            f"{_JOB_NAME} must start live, telling the user, on its daily rule"
        )
        info = db.users.get_info(TEST_SENDER)
        assert info is not None and info.name == _USER_NAME, (
            f"the seeded profile must name {_USER_NAME!r}"
        )

        limit = int(penny.config.runtime.MESSAGE_CONTEXT_LIMIT)
        window = db.messages.get_messages_since(TEST_SENDER, since=datetime.min, limit=limit)
        assert len(window) == 21 and window[-1].content == _RECORDED_PUSH_THREADING, (
            f"the window must close on the deploy push, it holds {len(window)} rows"
        )
        assert window[-3].content == _RECORDED_SHELF_ASK and (
            window[1].content == _RECORDED_SPEC_ASK
        ), "the window must be the recorded conversation, shelf reply closing on the push"

        # The history renders no tool lines, so the seeded runs' identity is what the
        # premise carries: each recorded run's prompt rows are stamped under their own
        # run id and trace the tools that run made.
        seeded_rows = {
            row.run_id: row
            for row in db.messages.recent_prompts(limit=100)
            if is_seeded_run(row.run_id)
        }
        for run_name, tools in (
            ("turn-specs", ("browse",)),
            ("turn-shops", ("browse",)),
            ("turn-shelf-hang", ()),
            ("turn-shelf", ()),
        ):
            group = [row for row in seeded_rows.values() if row.run_id == seeded_run_id(run_name)]
            assert group, f"the seeded run {run_name!r} must have its own prompt-log record"
            names = tuple(dict.fromkeys(name for name, _ in run_tool_calls(group)))
            assert names == tools, (
                f"the seeded run {run_name!r} must trace with {tools}, its records show {names}"
            )

    return probe


# ── The claims ────────────────────────────────────────────────────────────────


def _the_reply_carries_a_source_url(sample: SampleObservation, _world: World) -> Answer:
    """The reply carries a source URL from the world.  Any host the world's own pages
    carry counts, the search page's directory entry included: a URL the world serves
    or shows is a source the world owns.  A violating sample is nameable: one that
    replied from the earlier conversation's list and the search snippets' titles and
    names, naming shops and addresses but citing nothing."""
    said = fold_typography(sample.reply)
    hit = [host for host in _SERVED_HOSTS if host in said]
    return bool(hit), f"the reply cites none of {list(_SERVED_HOSTS)}"


def _the_reply_carries_the_page_fact(sample: SampleObservation, _world: World) -> Answer:
    """The reply carries the district shop's street address — the narrowing's proof
    that lives ONLY on the shop's own page: never in a search snippet, never in the
    conversation — so it can be in the reply only if that page was really read.  A
    violating sample is nameable: one that answered from the earlier list and the
    search snippets alone, which carry the shop's district but not its address."""
    said = fold_typography(sample.reply)
    return _PAGE_FACT in said, f"the reply never states the page fact {_PAGE_FACT!r}"


def _the_reply_names_a_district_shop(sample: SampleObservation, _world: World) -> Answer:
    """The reply names a shop the world places in the district the ask named.  A
    violating sample is nameable: one that recommends the shop the conversation
    already recommended, whose own page places it outside the district."""
    said = fold_typography(sample.reply)
    return _ANSWER_SHOP in said, f"the reply never names the district shop {_ANSWER_SHOP!r}"


_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_followup_lookup_reply_reads_the_pages(chat_eval: ChatEval, model: str) -> None:
    """A follow-up ask narrowing an earlier lookup, with no standing rules: the
    reply names a shop the world places in the district, carries that shop's
    page-only address, and cites a source URL the world owns.  The failure this
    case hunts — an answer re-combined from the earlier list and the search
    snippets, no page opened — fails the address claim and reads in the tool
    sequence."""
    cohort = await chat_eval(
        case_id="followup-lookup-reads-the-pages",
        behaviour=(
            "In the chat agent, when the earlier conversation already listed the "
            "candidates and the user asks to narrow them, Penny's reply carries the "
            "detail only the pages carry."
        ),
        area=Area.ANSWERING_FROM_THE_WEB,
        model=model,
        seed=_seed_world,
        prepare=_premise(),
        world=World(
            name="followup-lookup-reads-the-pages",
            pages=PAGES,
            keeps=(),  # empty by report: nothing here lands in a store; the deliverable
            excludes=(),  # is the reply, and the reply claims above carry it
            answers=(),
        ),
        ask=ASK,
        also_phrased=ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=900.0,
    )
    # STORE — a read-only ask, nothing seeded by the case lands: what the store
    # held when the turn began is still there.
    cohort.assert_what_the_store_held_survives()

    # PROVENANCE — the three things the narrowed ask owes the user: the shop that
    # fits, its page-only address, and a cited source.
    cohort.claim(
        "reply: it carries a source URL from the world",
        _the_reply_carries_a_source_url,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.claim(
        "reply: it carries the district shop's page-only address",
        _the_reply_carries_the_page_fact,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.claim(
        "reply: it names a shop the world places in the district",
        _the_reply_names_a_district_shop,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
