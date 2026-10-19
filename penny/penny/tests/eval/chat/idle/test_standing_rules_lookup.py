"""Standing rules: an ordinary web-lookup ask is honoured while rules stand (#2647).

The contract is `docs/eval-case-design.md`.

**ONE case**, ``rules-honoured-in-web-lookup``.  Two standing rules are in force —
one about citing a source URL, one about searching and then reading the promising
results directly — and the user asks an ordinary web question ("find me a shop that
can do X, only in Y").  The reply is the only deliverable: nothing lands in a store,
so what the case asserts is what reaches the user.

| case | the behaviour | what the reply must carry |
|---|---|---|
| honour both | ``rules-honoured-in-web-lookup`` | a source URL; a fact only a read page carries |

**The failure it was written from** (real use, cloned from the prod turn and
replayed): with both rules standing, the model called a rule's own key — a
snake_case name that reads as a tool — as if it were callable, got "no such tool",
searched twice, and replied from the search snippets alone: no URL, and nothing
from any result page's body.  The rules rendered as ``- <key>: <content> (added
…)`` in the chat prompt's instruction part, where a bare snake_case name followed
by an instruction reads exactly like the tool list two sections down.

**What is asserted.  Facts only.**  The reply carries a source URL the world owns,
and a fact that appears ONLY in a result page's body — never in its search snippet,
never in the conversation — so the fact can be in the reply only if a page was
really read.  The store is read back: the two rules are exactly as the turn found
them.  **What is NOT claimed**: that she called the rule's key as a tool, which
tools she called, in what order, or how she words the reply — the tool sequence and
the reply text are measured variance (the rule-as-tool call is the failure's
signature and is meant to VANISH from the sequence, not be asserted).  No wording
names a tool, mentions the rules, or states the expected outcome.

**The world.**  The recorded conversation of the real turn stands in her own logs
(the spec question, the shop recommendations, the turn that kept the second rule,
the cat fostering aside, the two deploy pushes), seeded through the stores' own
chokepoints — message rows paired with their prompt-log runs, so the history
renders the same tool lines production rendered.  The web behind ``browse`` is four
shop pages behind a search page: the search page carries titles, snippets and
links (including the mill-district shop's district, so a snippet-only reply CAN
name the right shop — the failure this case measures is the missing read and the
missing URL, not the district); the recommended shop's own page alone carries where
it actually is.  Every name, place, shop, URL, person, pet and topic is invented,
because the repo is public.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code
owner's to accept once the numbers have been read.
"""

from __future__ import annotations

from datetime import datetime
from typing import NamedTuple

import pytest

from penny.agents.self_state import standing_rules_instruction
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput, LogEntryInput
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    Preparer,
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

_FAMILY = "standing-rules"

# ── The world's constants ─────────────────────────────────────────────────────
#
# Every value here is invented (the real turn's values were swapped for these of
# the same shape before this file was committed).  The tokens a claim reads are
# whitespace-free and live in exactly one place each — that is what makes finding
# one in the reply evidence it came from there and nowhere else.

RULES = PennyConstants.MEMORY_RULES_COLLECTION

# The two standing rules the turn is answered under — the second one's KEY is the
# snake_case name the failing turn called as a tool.
_RULES = (
    (
        "always_cite_sources",
        "Whenever I ask you to search for things, always include the specific details "
        "and at least one source URL so I can dig deeper.",
    ),
    (
        "search_then_read_first",
        "When asked to look for something, perform a search first, pick promising "
        "results, and then browse them directly to read and extract information "
        "before responding.",
    ),
)

_USER_NAME = "Wren"
_USER_LOCATION = "Grelford, OR"
_USER_TIMEZONE = "America/Los_Angeles"

# The camera and the job.  The headline job is ambient state (it renders in the
# Jobs section, notify on, no runs yet), the profile collection the earlier
# conversation created.
_CAMERA = "Corvina CC-3"
_JOB_NAME = "daily_headlines"
_JOB_DESCRIPTION = (
    "Daily top 3 headlines from the city's news sites including title, description, and link"
)
_JOB_PROGRAM = (
    "Find the top 3 news headlines from harbourline.example, quarrypost.example, "
    "gullwind.example, and beaconwire.example.\n"
    "Keep the headline as published, with a one-sentence description of the story.\n"
    "Carry each story's link so the reader can open the piece.\n"
    "Skip sports scores and weather, which move faster than a daily collection reads."
)
_JOB_SCHEDULE = "DTSTART:20261015T093000\nFREQ=DAILY"

_PROFILE_ENTRIES = (
    (
        "career",
        "Software engineer in Grelford, Oregon; works at Skylark on the browser "
        "experimentation program called Northlight.",
    ),
    (
        "interests",
        "Plays guitar and records music in a home studio; plays video games; collects vinyls.",
    ),
    ("relationships", "Partner Jess — she's my best friend."),
    ("pets", "Fosters cats; currently fostering Toby."),
)

# The browse log the conversation's earlier searches left behind — six entries, the
# count the real turn's memory inventory showed, about the earlier camera research.
_BROWSE_ENTRIES = (
    "Corvina CC-3 — specifications: folding camera, 35mm film, door channel takes a "
    "2.0 mm light seal.",
    "Corvina CC-3 — light seals: original seals were foam strips, replaced by cutting "
    "new foam to the channel width.",
    "Light seals — materials: closed-cell foam, self-adhesive backing, cut to length "
    "and fitted dry.",
    "Corvina CC-3 — value: clean examples sell in the mid three figures; a light-tight "
    "seal is the first thing buyers check.",
    "Corvina CC-3 — servicing: seals, bellows and shutter are the three usual jobs on "
    "a folding camera of this age.",
    "Light seals — fitting: strip the old foam fully, clean the channel with alcohol, "
    "fit the new seal without stretching it.",
)

# The district the ask narrows to, and the shops.  Framinghouse is the shop whose
# material places it in the mill district (the answer); Sealcroft is the shop the
# conversation already recommended, whose OWN page alone says where it actually is.
_CITY = "grelford"
_DISTRICT = "mill district"
_ANSWER_SHOP = "framinghouse"
_FACT_TOKEN = "northolme"  # only on Sealcroft's page — the read-proof
_SHOP_URL_HOSTS = (
    "framinghouse.example.com",
    "sealcroft.example.com",
    "lightbench.example.com",
    "foamline.example.com",
)

# ── The web behind browse ─────────────────────────────────────────────────────
#
# Shop pages FIRST, the search page last: request dispatch matches the first page
# whose ``match`` sits in the request URL, so a query naming a shop hits that shop's
# page and a topic query (which carries the city) hits the search page.  The search
# page is shaped like a real results page — titles, snippets and links — so its trim
# (the ±2-line window around each solo link) reads like production's search results.

_FRAMINGHOUSE_PAGE = CannedPage(
    match="framinghouse",
    text=(
        "Title: Framinghouse Camera Works — services and contact\n"
        "Source: https://framinghouse.example.com/services\n"
        "\n"
        "Framinghouse Camera Works\n"
        "4 Cordage Lane, Mill District, Grelford\n"
        "\n"
        "Services\n"
        "Light-seal replacement for folding and rangefinder cameras: old foam stripped "
        "out, the channel cleaned, and a new seal cut and fitted the same day.\n"
        "Seal foam is cut in 2.0 mm and 1.5 mm widths and stocked for the common "
        "folding cameras.\n"
        "Bellows re-covering, shutter servicing and general vintage camera repair.\n"
        "\n"
        "Counter hours\n"
        "Walk-in counter Tuesday to Saturday, 10:00 to 18:00.\n"
        "Mail-in service accepted from anywhere; return shipping is flat-rate.\n"
    ),
)

_SEALCROFT_PAGE = CannedPage(
    match="sealcroft",
    text=(
        "Title: Sealcroft Repair — contact and workshop\n"
        "Source: https://sealcroft.example.com\n"
        "\n"
        "Sealcroft Repair\n"
        "Unit 9, Northolme Trading Estate, Northolme (Grelford county)\n"
        "\n"
        "Custom light-seal and foam cutting for folding cameras, posted in or dropped "
        "at the workshop door.\n"
        "Workshop hours Monday to Friday, 09:00 to 17:00. No walk-in counter — "
        "fittings are by appointment.\n"
        "Mail-in service across the region.\n"
    ),
)

_LIGHTBENCH_PAGE = CannedPage(
    match="lightbench",
    text=(
        "Title: Lightbench Co — what we do\n"
        "Source: https://lightbench.example.com\n"
        "\n"
        "Lightbench Co, 118 Foundry Street, North Grelford.\n"
        "Sensor cleaning, light-seal replacement for 35mm fixed-lens cameras, and "
        "general check-ups.\n"
        "Open seven days, 09:00 to 19:00.\n"
    ),
)

_FOAMLINE_PAGE = CannedPage(
    match="foamline",
    text=(
        "Title: Foamline Service — precision seal cutting\n"
        "Source: https://foamline.example.com\n"
        "\n"
        "Foamline Service, Unit 6, Ellerslie Industrial Park.\n"
        "Precision light-seal and gasket cutting to the measurements you send in; "
        "standard kits for the common folding cameras.\n"
        "Mail-in only; turnaround three working days.\n"
    ),
)

_SEARCH_PAGE = CannedPage(
    match="grelford",
    text=(
        "Title: camera light seal replacement — search results\n"
        "\n"
        "### Best camera repair near the mill district, Grelford\n"
        "[directory.example.com/camera-repair/grelford](https://directory.example.com/camera-repair/grelford)\n"
        "Best camera repair in Grelford - Framinghouse Camera Works, Sealcroft "
        "Repair, Lightbench Co, Foamline Service\n"
        "\n"
        "### Framinghouse Camera Works — vintage camera service\n"
        "[framinghouse.example.com/services](https://framinghouse.example.com/services)\n"
        "Vintage camera service in the mill district. Light-seal kits cut and fitted "
        "while you wait.\n"
        "\n"
        "### Sealcroft Repair — custom seal cutting\n"
        "[sealcroft.example.com](https://sealcroft.example.com)\n"
        "Custom foam and light-seal cutting for folding cameras. Mail-in and drop-off.\n"
        "\n"
        "### Lightbench Co — camera service\n"
        "[lightbench.example.com](https://lightbench.example.com)\n"
        "North Grelford. Light-seal replacement for 35mm cameras, walk-in seven days.\n"
        "\n"
        "### Foamline Service — precision seals\n"
        "[foamline.example.com](https://foamline.example.com)\n"
        "Precision light-seal cutting to your measurements. Mail-in service.\n"
    ),
)

PAGES = (
    _FRAMINGHOUSE_PAGE,
    _SEALCROFT_PAGE,
    _LIGHTBENCH_PAGE,
    _FOAMLINE_PAGE,
    _SEARCH_PAGE,
)

# ── The recorded conversation ─────────────────────────────────────────────────
#
# The real turn's history, genericised, in the order it was sent.  Each recorded run
# carries the tool call its own prompt-log rows show it made (``call``), so the
# history renders the tool lines production rendered; the autonomous pushes carry no
# run.  ``suffix`` is the tool-call line the model itself emitted into a reply —
# part of the recorded reply's stored text, as production stored it.

_RECORDED_SPEC_ASK = (
    f"Okay great good job. Anyways so that camera is a {_CAMERA} and it has worn "
    "light seals but we want to swap them out for fresh foam. Can you see if you can "
    "figure out the exact dimensions of the seals so we can order a kit?"
)
_RECORDED_SPEC_REPLY = (
    "I've been digging into the specs for that Corvina CC-3 to help you get that "
    "fresh-seal upgrade!\n"
    "\n"
    "Finding exact dimensions for vintage light seals can be a bit tricky because "
    "they're often sold by their thickness rather than just the door size, but I "
    "found a very promising lead.\n"
    "\n"
    "The door channel on your specific model is 35mm wide. From what I gathered:\n"
    "\n"
    '   The "2.0" Clue: I found a reference to an original Corvina foam seal for '
    'this exact model (part number RX5017) that lists "WIDTH 2.0". In the repair '
    "world, this often refers to a thickness of 2.0mm. \n"
    "   Standard Fit: For a 35mm channel, a ~2.0mm seal is a very standard and "
    "logical fit for a folding-door camera like this.\n"
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
    "exact channel width for the CC-3 reference.\n"
    "\n"
    "It's going to look sharp once that's swapped out—the fresh seals will keep the "
    "film bay light-tight again! 📷✨"
)
_RECORDED_SHOPS_ASK = (
    "okay good findings can you look and see who in grelford we can take it to to "
    "help size and fit and install the seals"
)
_RECORDED_SHOPS_REPLY = (
    "I've found several highly-rated options in Grelford that specialize in exactly "
    "what you're looking for—vintage camera repair and light-seal replacement.\n"
    "\n"
    "Depending on whether you want someone who can custom-cut a specific size or a "
    "full-service shop, here are my top recommendations:\n"
    "\n"
    "The Specialist Options (Great for Custom/Specific Sizing):\n"
    " Sealcroft Repair: This is a fantastic option for your project. They specialize "
    "in custom seal cutting, and specifically mention vintage camera restoration. "
    "This would be ideal if you need that exact '2.0mm' fit we discussed. "
    "[https://sealcroft.example.com](https://sealcroft.example.com)\n"
    " Foamline Service: If you want absolute precision, these guys are specialists "
    "in light-seal cutting and can custom-cut any seal that isn't in their standard "
    "catalog. They are a go-to for many professionals. "
    "[https://foamline.example.com](https://foamline.example.com)\n"
    "\n"
    "The Full-Service Options (Great for a Full Service Experience):\n"
    " Framinghouse Camera Works: They handle vintage models and specifically list "
    "light-seal replacement as part of their services. It's a very professional "
    "choice for a project like this. "
    "[https://framinghouse.example.com](https://framinghouse.example.com)\n"
    " Corvina Service Centre: A highly-rated shop (5 stars on Yelp!) located at 441 "
    "Chandler Street W. They have experience with light-seal changes and provide "
    "very reliable service.\n"
    "\n"
    "My recommendation: Since we are aiming for a very specific fit to ensure that "
    "vintage Corvina is protected, I'd reach out to Sealcroft Repair first. Their "
    "expertise in custom seal cutting sounds like a perfect match for the delicate "
    "needs of a vintage piece. 📷✨"
)
_RECORDED_RULE_ASK = (
    "oh i have a new rule for you to try: when i tell you to look for something, you "
    "do a browse search first, then pick some results, and browse them directly to "
    "read, store that as a rule"
)
_RECORDED_RULE_REPLY = (
    "I've just added that new rule to your rules collection! \n"
    "\n"
    "Specifically, I saved it as search_then_read_first. From now on, instead of "
    "just giving you what I see in those little search snippets, I'll perform the "
    "search, pick out the most promising links, and then head directly into the "
    "pages to read the full details before I come back to you. It'll definitely make "
    "my findings much more robust.\n"
    "\n"
    "I'm all set—what's on the agenda now? 🫡\n"
    "[Tool calls: collection_write]"
)
_RECORDED_PUSH_COLLAPSE = "👋 I added a way to collapse older messages in your prompts!"
_RECORDED_PHOTO_ASK = "[sent a photo]"
_RECORDED_PHOTO_REPLY = (
    "What a striking little soul! Those tabby markings are so gorgeous—and those "
    "eyes! They are looking right at you with such intense, curious energy. \n"
    "\n"
    "Is this one of yours? He looks like he's officially claimed that bookshelf as "
    "his sovereign territory. 🐾"
)
_RECORDED_MEMORY_ASK = "Do you remember what I said about my cat?"
_RECORDED_MEMORY_REPLY = (
    "I do! You mentioned that you're fostering cats and that you currently have "
    "Toby. 🐾\n"
    "\n"
    "Is the little tabby in the photo Toby? Or is he a different visitor? 😊\n"
    "[Tool calls: find]"
)
_RECORDED_STAR_ASK = "Yes that's Toby"
_RECORDED_STAR_REPLY = (
    "He is such a little star! 😻 I can see that curious energy in his eyes. \n"
    "\n"
    "How has he been settling in? Is he a little explorer or more of a couch "
    "potato? 🐾"
)
_RECORDED_ZOOMIES_ASK = "Very much not a couch potato. He's a little maniac."
_RECORDED_ZOOMIES_REPLY = (
    "Haha, oh no! 😹 The zoomies must be real. Is he one of those cats that does the "
    "midnight crazies, or is he just constant chaos all day long? 🐾"
)
_RECORDED_ALLERGY_ASK = (
    "Oh all day. El is allergic to cats but it varies by cat. Apparently tabbies are "
    "particularly bad for allergies? Why?"
)
_RECORDED_ALLERGY_REPLY = (
    "I'll be honest—it turns out it's probably not the tabby pattern itself that's "
    "the problem! There isn't any scientific evidence that says \"tabbies equal more "
    "allergies,\" but it's a really common thing people notice.\n"
    "\n"
    "The actual culprit is a protein called Fel d 1. It's found in a cat's saliva, "
    "skin (dander), and even their tears. When Toby grooms himself, he's spreading "
    'that protein all over his fur, and then when he goes into "maniac mode," he\'s '
    "essentially throwing that dander into the air! 🌪️\n"
    "\n"
    "Since you said he's a little maniac, that extra movement probably means more "
    "dander and saliva being kicked up around the house, which makes the allergies "
    "hit El a lot harder.\n"
    "\n"
    'There are a few other reasons why some cats are "worse" than others for '
    "allergies, though:\n"
    "\n"
    "   Protein Levels: Every individual cat has a different amount of that Fel d 1 "
    'protein. Some cats are just "high producers."\n'
    "   Gender & Neutering: Generally speaking, unneutered males have much higher "
    "levels of the protein than females or neutered cats. \n"
    "   Activity Level: Like we talked about, a high-energy cat is basically a tiny, "
    "furry particle dispenser!\n"
    "\n"
    "It's a tough situation when you're trying to foster! Is El able to manage the "
    "symptoms with meds, or does he just have to suffer through it for now? 🐾\n"
    "[Tool calls: none]"
)
_RECORDED_FOSTER_ASK = (
    "She just can't come over while he's here. Soon he will be adopted and then a "
    "new foster will arrive!"
)
_RECORDED_FOSTER_REPLY = (
    "That is a tough way to handle it, having to miss each other because of a little "
    "feline chaos! It's a real bummer, but it sounds like you're doing a great thing "
    "for him by getting him ready for his forever home. \n"
    "\n"
    "The foster life is such a bittersweet cycle, isn't it? You get so attached to "
    "these little maniacs, but then there's that bittersweet moment when they leave, "
    'followed immediately by the "fresh start" of a new guest. \n'
    "\n"
    "Do you have a specific type of cat you usually look for when fostering, or just "
    "whoever needs a home? 🐾\n"
    "[Tool calls: none]"
)
_RECORDED_PUSH_THREADING = (
    "👋 I made it so your previous chats collapse when you open a new prompt!"
)


class _Turn(NamedTuple):
    """One recorded exchange, as the message log holds it.

    ``run_name`` names the seeded run whose prompt-log rows the history's tool lines
    read (``None`` for an autonomous push, which carries no run and no line).
    ``call`` is the one tool call the run made (``None`` for a run that only
    talked); ``result`` is the framed tool result that call got back.  ``reply`` is
    the stored outgoing content — including any tool-call line the model itself
    emitted, as production stored it."""

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
        _RECORDED_RULE_ASK,
        _RECORDED_RULE_REPLY,
        run_name="turn-rule",
        call=(
            "collection_write",
            {
                "collection": "rules",
                "key": "search_then_read_first",
                "content": _RULES[1][1],
            },
        ),
        result="You saved the rule to `rules`.",
    ),
    _Turn(None, _RECORDED_PUSH_COLLAPSE),
    _Turn(_RECORDED_PHOTO_ASK, _RECORDED_PHOTO_REPLY, run_name="turn-photo"),
    _Turn(
        _RECORDED_MEMORY_ASK,
        _RECORDED_MEMORY_REPLY,
        run_name="turn-memory",
        call=("find", {"query": "the user's cat fostering"}),
        result="You found the conversation about fostering cats.",
    ),
    _Turn(_RECORDED_STAR_ASK, _RECORDED_STAR_REPLY, run_name="turn-star"),
    _Turn(_RECORDED_ZOOMIES_ASK, _RECORDED_ZOOMIES_REPLY, run_name="turn-zoomies"),
    _Turn(
        _RECORDED_ALLERGY_ASK,
        _RECORDED_ALLERGY_REPLY,
        run_name="turn-allergy",
        call=(
            "browse",
            {
                "queries": ["are tabby cats worse for allergies"],
                "extract": "whether tabby cats cause worse allergies and why",
            },
        ),
        result=_search_result("are tabby cats worse for allergies"),
    ),
    _Turn(_RECORDED_FOSTER_ASK, _RECORDED_FOSTER_REPLY, run_name="turn-foster"),
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
    ("i finally sorted the vinyl shelf", "the shelf thanks you for your service"),
    ("toby knocked a pen off the desk again", "a pen's purpose is to be knocked off"),
    ("any plans for the weekend", "the weekend plans are the same as the weekday plans"),
    ("the new episode was so good", "no spoilers, i'm two behind"),
    ("my back hurts from carrying those boxes", "lift with the knees, not the shelf"),
    ("do we have any coffee left", "enough for exactly one cautious morning"),
    ("i liked that song you sent", "it's been on repeat since tuesday"),
    ("the plants survived the week", "a miracle of modern neglect"),
    ("i booked the dentist finally", "the calendar celebrates"),
    ("toby stared at the wall for an hour", "a cat's job is never done"),
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
    "that can fit the seals but only in the mill district in grelford"
)
ALSO_PHRASED: tuple[str, ...] = (
    "hey so about those camera shops from before — which one can do the seal swap "
    "and is actually in the mill district of grelford?",
    "find me a grelford shop that can install the light seals like we talked about "
    "before, it has to be in the mill district",
    "so which shop we looked at before, in the mill district of grelford, can fit "
    "the seals on the CC-3 for me?",
    "can you look for a camera shop in the mill district of grelford that can do "
    "the light seal replacement? like the ones we looked at before",
)

_WORDINGS = (ASK, *ALSO_PHRASED)


def _assert_wordings_carry_their_facts() -> None:
    """ENFORCED at import: every wording states the ask's own facts (the service, the
    district, the city, and that this continues an earlier search), and lends the
    claims nothing — no shop name, no URL, no page fact, and nothing about the rules
    or what a good reply does."""
    for wording in _WORDINGS:
        folded = wording.casefold()
        assert _CITY in folded, f"a wording must name the city: {wording!r}"
        assert _DISTRICT in folded, f"a wording must name the district: {wording!r}"
        assert "seal" in folded, f"a wording must name the service: {wording!r}"
        assert "earlier" in folded or "before" in folded, (
            f"a wording must carry the continuity: {wording!r}"
        )
        assert _ANSWER_SHOP not in folded, f"a wording must not name the answer: {wording!r}"
        assert _FACT_TOKEN not in folded, f"a wording must not carry the page fact: {wording!r}"
        for host in _SHOP_URL_HOSTS:
            assert host not in folded, f"a wording must not carry a URL: {wording!r}"
        for key, content in _RULES:
            assert key not in folded, f"a wording must not name a rule key: {wording!r}"
            first_words = " ".join(content.split()[:4]).casefold()
            assert first_words not in folded, (
                f"a wording must not borrow a rule's words: {wording!r}"
            )


_assert_wordings_carry_their_facts()


def _assert_the_world_coheres() -> None:
    """ENFORCED at import: each claim token lives in exactly one place, and no page,
    history text or wording carries another page's identity — the shape that makes
    finding a token in the reply evidence of where it came from."""
    search_text = _SEARCH_PAGE.text.casefold()
    shop_pages = {
        page.match: page.text.casefold()
        for page in (_FRAMINGHOUSE_PAGE, _SEALCROFT_PAGE, _LIGHTBENCH_PAGE, _FOAMLINE_PAGE)
    }
    history_text = (
        " ".join(turn.ask or "" for turn in _RECORDED_TURNS)
        + " "
        + " ".join(turn.reply for turn in _RECORDED_TURNS)
    )
    history_text = history_text.casefold()

    # The page fact is on exactly one page — no snippet, no history turn, no wording.
    assert _FACT_TOKEN in shop_pages["sealcroft"]
    assert _FACT_TOKEN not in search_text
    assert _FACT_TOKEN not in shop_pages["framinghouse"]
    assert _FACT_TOKEN not in shop_pages["lightbench"]
    assert _FACT_TOKEN not in shop_pages["foamline"]
    assert _FACT_TOKEN not in history_text

    # The answer shop's district placement is in its own snippet and page; the fact
    # token's shop never claims the district (that reading costs the read).
    assert _ANSWER_SHOP in search_text and _ANSWER_SHOP in shop_pages["framinghouse"]

    # Every shop host is on the search page (so a snippet-only reply COULD cite one)
    # and on its own page's Source line.
    for host in _SHOP_URL_HOSTS:
        assert host in search_text, f"a shop host must be on the search page: {host}"
        own = [text for match, text in shop_pages.items() if match in host]
        assert own and host in own[0], f"a shop host must be on its own page: {host}"

    # No recorded exchange names the fact token, and the fillers carry nothing.
    for pair in _FILLER_PAIRS:
        for text in pair:
            assert _FACT_TOKEN not in text.casefold()
            assert _ANSWER_SHOP not in text.casefold()


_assert_the_world_coheres()

# ── Seeding the world ─────────────────────────────────────────────────────────


def _seed_world(db: Database) -> None:
    """Lay the recorded world down: the user, the headline job, the profile and browse
    logs the earlier conversation left, the two standing rules, the filler exchanges
    that pad the message logs to the counts production showed, and the recorded
    conversation with its runs — every write through the stores' own chokepoints, so
    the rendered prompt matches the one production rendered."""
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

    # The rules, written by the chat run that kept them (the recorded rule turn).
    rule_run = seeded_run_id("turn-rule")
    require_memory(db, RULES).write(
        [EntryInput(key=key, content=content) for key, content in _RULES],
        author="chat",
        run_id=rule_run,
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
    the run's own prompt-log rows when it made a call, so the history renders the
    tool lines production rendered.  A run with a call logs two rows (the call, then
    the reply after its result); a run that only talked logs one."""
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
    """The case's premise, asserted inside the sample before the turn runs: the rules
    render where she reads instructions carries both rules' sentences; the job is
    live; the window is exactly the recorded conversation; and the seeded runs pair
    so the history renders the recorded tool lines.  The render is asserted on the
    SENTENCES, not the line shape, so what the render looks like is the fix's
    business, never the probe's."""

    def probe(penny: Penny) -> None:
        db = penny.db
        rendered = standing_rules_instruction(db) or ""
        for _key, content in _RULES:
            assert content in rendered, f"the rules render must carry the rule:\n{rendered!r}"

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
        assert window[-3].content == _RECORDED_FOSTER_ASK and (
            window[1].content == _RECORDED_SPEC_ASK
        ), "the window must be the recorded conversation, foster reply closing on the push"

        traces = {t.run_id: t.tools for t in db.messages.chat_turn_tool_traces()}
        assert traces.get(seeded_run_id("turn-rule")) == ("collection_write",), (
            f"the rule turn must trace collection_write, it holds {traces}"
        )
        assert traces.get(seeded_run_id("turn-specs")) == ("browse",)
        assert traces.get(seeded_run_id("turn-foster")) == ()

    return probe


# ── The claims ────────────────────────────────────────────────────────────────


def _rules_entries(sample: SampleObservation) -> list[tuple[str | None, str]]:
    """Every entry ``rules`` holds when the sample ends, as (key, content)."""
    return [(entry.key, entry.content) for entry in sample.held if entry.collection == RULES]


def _rules_unchanged():
    """``rules`` is exactly as the turn found it — the whole collection compared,
    key and content, against what the case seeded.  A violating sample is nameable:
    one that reworded a rule while answering, dropped one, or added a second."""

    def answer(sample: SampleObservation, _world: World) -> tuple[bool, str]:
        held = set(_rules_entries(sample))
        return held == set(_RULES), f"`{RULES}` holds {sorted(held)}"

    return answer


def _the_reply_carries_a_source_url(sample: SampleObservation, _world: World) -> Answer:
    """The reply carries a source URL the world owns — the rule the user gave, made
    behavioural.  A violating sample is nameable: one that replied from the search
    snippets' titles and names, naming shops and addresses but citing nothing."""
    said = fold_typography(sample.reply)
    hit = [host for host in _SHOP_URL_HOSTS if host in said]
    return bool(hit), f"the reply cites none of {list(_SHOP_URL_HOSTS)}"


def _the_reply_carries_the_page_fact(sample: SampleObservation, _world: World) -> Answer:
    """The reply carries the fact that lives ONLY on a shop's own page — never in a
    search snippet, never in the conversation — so it can be in the reply only if a
    page was really read.  A violating sample is nameable: one that answered from
    the search snippets alone, which do not carry it."""
    said = fold_typography(sample.reply)
    return _FACT_TOKEN in said, f"the reply never states the page fact {_FACT_TOKEN!r}"


def _the_reply_names_a_district_shop(sample: SampleObservation, _world: World) -> Answer:
    """The reply names a shop the world places in the district the ask named.  A
    violating sample is nameable: one that recommends the shop the conversation
    already recommended, whose own page places it outside the district."""
    said = fold_typography(sample.reply)
    return _ANSWER_SHOP in said, f"the reply never names the district shop {_ANSWER_SHOP!r}"


_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


# ═══ honour both rules in the reply ══════════════════════════════════════════


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_reply_honours_the_standing_rules(chat_eval: ChatEval, model: str) -> None:
    """Two rules standing, an ordinary web ask: the reply cites a source URL and
    carries a fact only a read page carries, and the rules are exactly as the turn
    found them.  The failure this case was written from — a reply from snippets
    alone, after calling a rule's key as a tool — fails the two reply claims and
    reads in the tool sequence."""
    cohort = await chat_eval(
        case_id="rules-honoured-in-web-lookup",
        behaviour=(
            "In the chat agent, when standing rules ask for a cited, read-the-page "
            "web answer and the user asks for a shop lookup, Penny's reply cites a "
            "source URL and carries a fact only a read page carries."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed_world,
        prepare=_premise(),
        world=World(
            name="rules-honoured-in-web-lookup",
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
    # STORE — the rules are as the turn found them.
    cohort.claim(
        "state: `rules` is exactly as the turn found it", _rules_unchanged(), SpecCategory.STORE
    )

    # PROVENANCE — the two things the rules owe the user: a cited source, and a fact
    # that proves a page was read.
    cohort.claim(
        "reply: it carries a source URL from the world",
        _the_reply_carries_a_source_url,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.claim(
        "reply: it carries the fact only a read page carries",
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

    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
