"""Changing a running job's instructions to format its output, and the reply that
narrates the change instead (#2660).

The contract is `docs/eval-case-design.md`.

**The failure.** Chat, asked to update a running job's instructions so its output is
formatted a certain way every day: Penny replied that she had updated it, ended the turn
with no tool call, and the job's row never moved. Her thinking walked the whole
``collection_configure`` call and stopped one step short — the history she was shown carried
no past tool calls to imitate, only her own prose, so narrating read as the way a change
happens. This is #2630's failure again after #2651 removed the history's ``[Tool calls:
...]`` line (the model had copied that line into replies). The fix decided with the code
owner renders a past turn's real tool calls in the history as structured assistant
tool-call messages — the way her own calls look within a turn — never as text in her
reply's voice.

**The world is the prod turn's own conversation** (promptlog id 119, run
``a34ff4fd…``, 2026-10-19), rebuilt through the stores' own chokepoints: the exchanges the
history window held — a espresso thread with two real browse calls behind its replies, the
job's morning notification, a read of the job's record (``memory_get``), a read of the
morning run's trace (``run_get``), a read of the collection (``collection_list``), and the
two formatting passes the ask points back at — plus the job, the user's profile, the two
standing rules, the browse-results log, and the two collector cycles the morning left
behind (one WORKED, one FAILED). Every tool result a past turn got is the tool's own real
render (``Tool.format_result`` over the production render functions), never a
hand-written approximation.

**The claim that matters is the same kind #2630 made**: the job's stored instructions
change as asked — read off the store — while the job's schedule, its switch, its
singleness and the global mute are what they were. Tool calls and reply wording stay
measured.

**What is NOT claimed, and why.**  That she called ``collection_configure`` is a ROUTE —
the measured tool sequence names it, never a claim.  Whether the reply also carries a
copied tool-call line is measured variance, never asserted.  Nothing reads where a
conversation machine landed; there is none left to land on.

Every real proper noun in the recorded turn is swapped for another REAL one of the same
kind (another real city, real outlets, real hardware, a real newsletter header) — invented
names make the model suspect a test world and behave differently — while the topics they
name are invented, and nothing that stays is any value the recorded turn carried.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code owner's
to accept once the numbers have been read.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pytest

from penny.agents.self_state import SelfStateHeader
from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory import EntryInput, LogEntryInput
from penny.database.memory.objects import render_run_calls
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.end_to_end.world import (
    _log_chat_step,
    _seeded_response,
    _tool_result_turn,
    _wire_tool_call,
)
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    MechanismRecord,
    SampleObservation,
    SpecCategory,
    token_stands_in,
)
from penny.tests.eval.utils.worlds import World
from penny.tools.base import Tool
from penny.tools.memory_tools import MemoryMetadataTool, format_entries
from penny.tools.models import ToolResult

_HEADLINES_ASK = "the top 3 news headlines with their titles and URLs"

pytestmark = pytest.mark.eval

_FAMILY = "standing-collection-operations"

# The seeded rows' own model name: the suite's model, kept honest rather than invented
# (the same rule the shared world helpers follow).
_SEEDED_MODEL = EVAL_MODELS[0]

# ── The invented world ────────────────────────────────────────────────────────

_USER_NAME = "Sam"
_USER_LOCATION = "Seattle, WA"
_USER_TIMEZONE = "America/Los_Angeles"

_PROFILE_ENTRIES = (
    (
        "career",
        "Runs a small pottery studio in Seattle, Washington; teaches two evening classes a week.",
    ),
    ("interests", "Bike touring, board games, film photography."),
    ("relationships", "Partner Alex — they hike together on weekends."),
    ("pets", "Keeps a shelf of studio houseplants; the oldest is a monstera."),
)

_RULES_ENTRIES = (
    (
        "sources_with_links",
        "Whenever I ask you to search for things, always include the specific details "
        "and at least one source URL so I can dig deeper.",
    ),
    (
        "read_before_answering",
        "When asked to look for something, perform a search first, pick promising "
        "results, and then browse them directly to read and extract information before "
        "responding.",
    ),
)

_JOB_CONTAINER = "npr_daily_brief"
_JOB_DESCRIPTION = (
    "Top 3 headlines each morning from NPR News, with titles, descriptions, and links"
)
# The prompt the job was created with, and the one a later turn updated it to — the
# update is the change line the job's record renders.
_JOB_FIRST_PROGRAM = (
    "Go to npr.org. Find the top 3 news headlines. For each, extract "
    "the title, a one-sentence description or summary, and the URL. Save all three as "
    "a single entry for the day."
)
_JOB_PROGRAM = (
    "Find the top 3 news headlines from npr.org, theguardian.com, aljazeera.com, and "
    "dw.com. For each source, list the 3 headlines with their titles and URLs. Format "
    "the output clearly by source. Save as a single entry for the day."
)
_JOB_SOURCES = ("npr.org", "theguardian.com", "aljazeera.com", "dw.com")
# The sample's own day: the job runs each morning, and today's cycle has already fired.
_TODAY = "2026-10-14"
_JOB_SCHEDULE = f"DTSTART:{_TODAY.replace('-', '')}T093000\nFREQ=DAILY"

# The brief's own header — the invented anchor the format claim reads.
_BRIEF_HEADER = "Morning Briefing"

# The 12 stories this morning's cycle gathered — the brief's whole content.
_STORIES = {
    "npr.org": (
        "Community radio station tops its fundraiser goal a week early - "
        "https://www.npr.org/2026/10/14/1319428751/community-radio-fundraiser-goal",
        "Hillside trail reopens after winter slide repairs - "
        "https://www.npr.org/2026/10/14/1319429120/hillside-trail-reopens",
        "City council approves the night market's extended hours - "
        "https://www.npr.org/2026/10/14/1319429688/night-market-extended-hours",
    ),
    "theguardian.com": (
        "New library wing opens with a full lending collection - "
        "https://www.theguardian.com/us-news/2026/oct/14/library-wing-opens",
        "River cleanup crew logs its clearest water reading yet - "
        "https://www.theguardian.com/environment/2026/oct/14/river-cleanup-reading",
        "Bakery co-op outgrows its shared kitchen - "
        "https://www.theguardian.com/food/2026/oct/14/bakery-coop-expands",
    ),
    "aljazeera.com": (
        "Observatory dome restoration finishes ahead of schedule - "
        "https://www.aljazeera.com/news/2026/10/14/observatory-restoration-finishes",
        "Transit pilot adds a second evening loop - "
        "https://www.aljazeera.com/news/2026/10/14/transit-pilot-evening-loop",
        "Mural on the grain elevator restored by original painters - "
        "https://www.aljazeera.com/news/2026/10/14/grain-elevator-mural-restored",
    ),
    "dw.com": (
        "Regional chess league crowns a first-time champion - "
        "https://www.dw.com/en/regional-chess-league-first-time-champion/a-77531042",
        "Botanical conservatory reopens its fern house - "
        "https://www.dw.com/en/botanical-conservatory-fern-house-reopens/a-77531077",
        "Printers' guild exhibits a century of hand-set type - "
        "https://www.dw.com/en/printers-guild-hand-set-type-exhibit/a-77531103",
    ),
}

_SOURCE_LABELS = {
    "npr.org": "NPR News",
    "theguardian.com": "The Guardian",
    "aljazeera.com": "Al Jazeera",
    "dw.com": "Deutsche Welle",
}


_BROWSE_LOG_SITES = (
    (
        "https://www.npr.org",
        "NPR - Breaking News, Analysis, Music, Arts & Podcasts",
    ),
    (
        "https://www.theguardian.com",
        "The Guardian - Latest news, sport, business, opinion",
    ),
    (
        "https://www.aljazeera.com",
        "Al Jazeera - Breaking News, World News and Video",
    ),
    (
        "https://www.dw.com",
        "Deutsche Welle - News and analysis from Europe and beyond",
    ),
    (
        "https://www.coffeereview.com/comparison/linea-mini-vs-silvia-pro-x/",
        "Linea Mini vs Silvia Pro X: Dual Boiler Comparison",
    ),
)

_BROWSE_LOG_LINKS = (
    "[Privacy Notice (new window)]",
    "[Menu]",
    "[News]",
    "[World]",
    "[Sports]",
    "[Business]",
    "[Video]",
    "[Live]",
    "[Weather]",
    "[Newsletters]",
)


def _browse_log_entries() -> tuple[str, ...]:
    """The 15 page-fetch blobs the conversation's reads left behind — the count the
    self-state's memory map renders.  The bodies stand for the news-site fetches and the
    hardware-thread page; none reaches the prompt except through a ``log_read``, which no
    claim reads."""
    entries = []
    for index in range(15):
        url, title = _BROWSE_LOG_SITES[index % len(_BROWSE_LOG_SITES)]
        body = " ".join(_BROWSE_LOG_LINKS[index % 4 : index % 4 + 5])
        entries.append(f"## browse: {url}\nTitle: {title}\nURL: {url}\n\n{body}")
    return tuple(entries)


_BROWSE_LOG_ENTRIES = _browse_log_entries()


def _stored_brief() -> str:
    """The brief as the morning cycle stored it — one entry, sections by source."""
    blocks = []
    for host in _JOB_SOURCES:
        label = _SOURCE_LABELS[host]
        numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(_STORIES[host], start=1))
        blocks.append(f"{label}\n{numbered}")
    return "\n\n".join(blocks)


# The notification the morning cycle sent — the "news brief message above" the
# conversation reads back, which carried only the first source's stories.
_NOTIFICATION = (
    "Hey there! The job just finished up, and here is what it found for the NPR News "
    "headlines:\n\n"
    "First, there's Community radio station tops its fundraiser goal a week early "
    "(https://www.npr.org/2026/10/14/1319428751/community-radio-fundraiser-goal), then "
    "Hillside trail reopens after winter slide repairs (https://www.npr.org/2026/10/14/"
    "1319429120/hillside-trail-reopens), and finally City council approves the night "
    "market's extended hours (https://www.npr.org/2026/10/14/1319429688/"
    "night-market-extended-hours).\n\n"
    "The job mentioned there were more entries available during the run as well."
)

# ── The conversation the window held ─────────────────────────────────────────
#
# Ten exchanges plus the job's own notification, in send order — the same 21 rows
# the prod turn's history window (MESSAGE_CONTEXT_LIMIT = 20) held, the oldest
# incoming row falling out of the cap so the window opens on her reply.

_EX0_USER = "Oh sorry I was thinking of the Silvia"
_EX0_REPLY = (
    "No worries at all! That makes much more sense—the Silvia Pro X is the absolute "
    "workhorse of the prosumer espresso world. While it's not the café grade of the "
    "Linea Mini, it is incredibly capable for serious home shots because of that "
    "temperature-stable boiler.\n\n"
    "Here is the updated breakdown for the two machines:\n\n"
    "The Specs (The Real Deal)\n"
    "The difference in the boiler is the most important part here:\n"
    "   Linea Mini: single dual-use boiler, integrated PID (This is the one most cafés "
    "aim for).\n"
    "   Silvia Pro X: dual boiler with two PIDs, one per zone.\n\n"
    "Since you're picking one machine for the kitchen, let's assume you're looking at "
    "the Silvia Pro X.\n\n"
    "The Strategy: One Shot While You Steam\n"
    'The logic for a dual boiler stays the same, but your "workflow" changes. With two '
    "boilers you are looking at brewing and steaming at the same time.\n\n"
    "1.  The Grind: You'll still want a burr grinder with real stepless adjustment. Both "
    "machines reward a consistent, freshly ground dose more than anything else.\n"
    "2.  The Routine: Using the machine's own PID display, you can hold the brew "
    "temperature steady. You set it once: 93°C for a medium roast, and you leave it "
    "alone.\n"
    "3.  The Steam Math: With a dedicated steam boiler, you can texture milk while the "
    "next shot pulls, which means a two-drink round lands in about four minutes "
    "instead of ten.\n\n"
    "What the Shots Are Like?\n"
    "   The Body: Because the brew boiler holds its temperature through the whole "
    "shot, you'll see much more even extraction than a thermoblock gives you. It's "
    "highly forgiving of small grind errors.\n"
    '   The Routine: When you run a morning queue, your "bottleneck" is how fast the '
    "steam recovers. With the Pro X's second boiler there isn't one. As long as your "
    "water is decently filtered, you'll be golden.\n"
    '   The Verdict: A Silvia Pro X is a "pro-sumer" dream. It\'s significantly more '
    "affordable than a Linea Mini, but gives you enough machine to pull café-grade "
    "shots every morning.\n\n"
    "Pro-Tip: If you're switching from a thermoblock machine, I highly recommend "
    "weighing your doses. It allows you to dial in far faster than eyeballing, and it "
    "matters more than any boiler upgrade until it's a habit.\n\n"
    "Sources:\n"
    "   Linea Mini vs Silvia Pro X: Coffee Review "
    "(https://www.coffeereview.com/comparison/linea-mini-vs-silvia-pro-x/)\n"
    "   Rancilio Silvia Pro X Product Page: Whole Latte Love "
    "(https://www.wholelattelove.com/products/rancilio-silvia-pro-x)\n"
    "   Dialling In New Machines: Home-Barista Guides "
    "(https://www.home-barista.com/guides/dialing-in.html)\n"
    "[Tool calls: none]"
)

_EX1_USER = (
    "Oh yeah like this \n\nhttps://www.coffeereview.com/comparison/linea-mini-vs-silvia-pro-x/"
)

_EX1_REPLY = (
    "That is exactly the kind of proof I was talking about! 🚀 Those numbers are "
    "absolutely wild.\n\n"
    "I was looking at the readings on that page, and seeing those temperature and "
    "recovery figures for those machines is incredible. Something like a 1°C swing "
    "across a whole session (and ~35 s steam recovery on the Pro X) is just massive. "
    "\n\n"
    "It really proves the point: when a machine holds its brew temperature through a "
    "whole session, you aren't just \"getting by\"—you're getting café-grade consistency "
    "at home. It turns a fiddly morning routine into something that just repeats!\n\n"
    "It’s a huge jump from a thermoblock machine. It’s basically the difference between "
    "a kettle on a stove and a bench with a PID on it. ✈️\n\n"
    "Are you thinking of upgrading the kitchen setup for a specific routine? 🎮_"
)
_EX1_BROWSE_URL = "https://www.coffeereview.com/comparison/linea-mini-vs-silvia-pro-x/"
_EX0_BROWSE_BODY = (
    "## browse search: Linea Mini vs Silvia Pro X boiler temperature\n"
    "These are search results — titles and links only. You must read the actual pages "
    "before answering. Pick a URL from below and pass it in your next queries array to "
    "read it.\n\n"
    "### Linea Mini vs Silvia Pro X: Dual Boiler Comparison\n\n"
    "[coffeereview.com/comparison/linea-mini-vs-silvia-pro-x](https://www.coffeereview.com/"
    "comparison/linea-mini-vs-silvia-pro-x/)\n\n"
    "The Pro X holds brew temperature within about 1°C across a full session, and its "
    "steam boiler recovers in roughly 35 seconds. The Linea Mini trades that for a "
    "pre-heated group and an integrated shot timer.\n\n"
    "### Rancilio Silvia Pro X\n\n"
    "[wholelattelove.com/products/rancilio-silvia-pro-x](https://www.wholelattelove.com/"
    "products/rancilio-silvia-pro-x)\n\n"
    "Two independently controlled boilers, one for brewing and one for steaming, each "
    "with its own PID — deliver café-grade performance at home."
)
_EX1_BROWSE_BODY = (
    "## browse: https://www.coffeereview.com/comparison/linea-mini-vs-silvia-pro-x/\n"
    "temperature stability figures, steam recovery times, and how the dual-boiler setup "
    "performs compared to a single boiler: - ~93°C held within 1°C, ~35 s steam "
    "recovery, and ~9 bar at the group\n"
    "- the Linea Mini, the Silvia Pro X, and the Lelit Bianca\n"
    "— no side-by-side shot-timer data is provided on the page"
)

_EX2_USER = (
    "Well I’ve been looking at upgrading, for the kitchen! Right now I’m pulling shots "
    "on an old Gaggia Classic and the temperature wobbles so I’m always eyeballing the "
    "grind. But they got so expensive so quickly. Any dual-boiler now is $3k+. The "
    "Silvia Pro X is $1.7k so it isn’t so bad."
)
_EX2_REPLY = (
    "Wait, you're doing all this for the morning coffee? 🥹 That is genuinely the "
    "sweetest kind of project! I'm officially honored to be consulted on it. \n\n"
    'You’re hitting on the exact pain point of prosumer espresso: the "Capacity vs. '
    'Control" trade-off. It’s a really interesting comparison when you look at the '
    "math:\n\n"
    "The Gaggia Strategy (What you have now):\n"
    "   The Vibe: Compact, Cheap, Honest.\n"
    "   The Secret Sauce: Simplicity. Because everything is mechanical and visible, you "
    "learn what the machine is actually doing very easily. It’s stable and "
    "predictable.\n"
    "   The Downside: Like you’re seeing, a single thermoblock wobbles under a queue. "
    'It\'s more of a "marathon runner"—steady for one shot, but not built for back-to-'
    "back drinks.\n\n"
    "The Silvia Pro X Strategy (The dual-boiler idea):\n"
    "   The Vibe: High Control, Specific Pace.\n"
    "   The Secret Sauce: Raw thermal stability. Those two boilers are built to hold "
    "exact temperatures. As we saw in that comparison, the recovery figures are "
    "absolute rockets compared to what most people experience on single-boiler "
    "machines.\n"
    '   The Downside: The "Footprint Wall." It needs a big bench, a proper water '
    "plan, and it won't forgive a dull grinder — you'd have to feed it well or the "
    "stability goes to waste.\n\n"
    "It’s a really smart move, though. Even though it costs more than the Gaggia ever "
    "did, the speed at which the morning queue moves would be a massive jump. The "
    "second shot wouldn't just be faster; it would taste practically identical! 🚀\n\n"
    "If you ever do pull the trigger on that machine, just promise me you'll let me "
    "know so I can hear how the first proper flat white turns out! 😉 \n\n"
    'Are you leaning more toward the "control" route, or are you happy with the '
    "character the Gaggia gives the bigger, slower mornings? 🐾_"
)

_EX3_USER = "Okay take a look at the instructions in the news brief job and tell me what they say"
_EX3_REPLY = (
    "I just pulled up the full details for you! \n\n"
    "The instructions for the npr_daily_brief job are:\n\n"
    '"Find the top 3 news headlines from npr.org, theguardian.com, aljazeera.com, and '
    "dw.com. For each source, list the 3 headlines with their titles and URLs. Format "
    'the output clearly by source. Save as a single entry for the day."\n\n'
    "Basically, it's looking for a roundup of the three biggest things happening on each "
    "of those sites, keeping them organized by the source. 📰✨"
)

_EX4_USER = (
    "Right and now can you find the logs that show what you actually did on this mornings run?"
)

_EX4_REPLY = (
    "I found them! I pulled up the full, step-by-step breakdown of what happened during "
    "this morning's run of the npr_daily_brief job.\n\n"
    "Here is strictly what the logs recorded:\n\n"
    "Step 1: The Big Sweep\n"
    "You used the browse tool to visit several news sites at once (including npr.org, "
    "theguardian.com, aljazeera.com, and dw.com) to extract the top 3 headlines, titles, "
    "and URLs from each. This was the heavy lifting part of the process, as it required "
    "reading through several different sites.\n\n"
    "Step 2: The Deutsche Welle Deep Dive\n"
    "It looks like a second browse call was specifically triggered for dw.com to ensure "
    "those headlines were pulled correctly.\n\n"
    "Step 3: Saving the Results\n"
    "Finally, the collection_write tool was used to save everything into the "
    "npr_daily_brief collection. It organized the news by source like this:\n\n"
    "   NPR News: The stories about the radio fundraiser and the trail.\n"
    "   The Guardian: The stories about the library wing and the river.\n"
    "   Al Jazeera: The stories about the observatory and the transit pilot.\n"
    "   Deutsche Welle: The stories about the chess champion and the fern house.\n\n"
    'The job finished by saving the date "2026-10-14" to the collection to keep track '
    "of the run. 📰✨"
)

_EX5_USER = (
    "Okay and then looking at the  news brief message above, does that message include "
    "all the news stories?"
)

_EX5_REPLY = (
    "I just took a close look at it, and no, it definitely doesn't! \n\n"
    "The message only listed the stories from NPR News. It completely left out the "
    "headlines it found for The Guardian, Al Jazeera, and Deutsche Welle, even though "
    "the run log shows that it successfully found and saved them all into the "
    "collection. \n\n"
    "It's like it gave you the first chapter of the news instead of the whole book! 📰❌"
)

_EX6_USER = "Right okay I think I know why, thanks for looking"
_EX6_REPLY = (
    "You're so welcome! I'm glad we could dive into the logs and crack the case "
    "together. 🔍\n\n"
    "If you decide to tweak that extraction_prompt again to get the full news dump back "
    "into your notifications—or if anything else weird pops up—just let me know. I'm "
    "always down for a little detective work! 🕵️‍♀️✨"
)

_EX7_USER = (
    "What’s in the news brief collection? You tell me now what stories are in there and "
    "see if you can give me the daily brief based on what’s in there"
)

_EX7_REPLY = (
    "I see exactly what happened! I went into the collection and found all the missing "
    "pieces. \n\n"
    "The collection was actually holding everything perfectly—all 12 stories from the "
    "four different agencies. It looks like when I sent that brief earlier, I only pulled "
    "the first few lines of the actual entry instead of the whole thing. My bad! 🤦‍♀️\n\n"
    "Here is your actual full daily brief for today, October 14th:\n\n"
    "📻 NPR News\n"
    "   Community radio station tops its fundraiser goal a week early (Link "
    "(https://www.npr.org/2026/10/14/1319428751/community-radio-fundraiser-goal))\n"
    "   Hillside trail reopens after winter slide repairs (Link "
    "(https://www.npr.org/2026/10/14/1319429120/hillside-trail-reopens))\n"
    "   City council approves the night market's extended hours (Link "
    "(https://www.npr.org/2026/10/14/1319429688/night-market-extended-hours))\n\n"
    "🌍 The Guardian\n"
    "   New library wing opens with a full lending collection (Link "
    "(https://www.theguardian.com/us-news/2026/oct/14/library-wing-opens))\n"
    "   River cleanup crew logs its clearest water reading yet (Link "
    "(https://www.theguardian.com/environment/2026/oct/14/river-cleanup-reading))\n"
    "   Bakery co-op outgrows its shared kitchen (Link "
    "(https://www.theguardian.com/food/2026/oct/14/bakery-coop-expands))\n\n"
    "🗞️ Al Jazeera\n"
    "   Observatory dome restoration finishes ahead of schedule (Link "
    "(https://www.aljazeera.com/news/2026/10/14/observatory-restoration-finishes))\n"
    "   Transit pilot adds a second evening loop (Link "
    "(https://www.aljazeera.com/news/2026/10/14/transit-pilot-evening-loop))\n"
    "   Mural on the grain elevator restored by original painters (Link "
    "(https://www.aljazeera.com/news/2026/10/14/grain-elevator-mural-restored))\n\n"
    "🌍 Deutsche Welle\n"
    "   Regional chess league crowns a first-time champion (Link "
    "(https://www.dw.com/en/regional-chess-league-first-time-champion/a-77531042))\n"
    "   Botanical conservatory reopens its fern house (Link "
    "(https://www.dw.com/en/botanical-conservatory-fern-house-reopens/a-77531077))\n"
    "   Printers' guild exhibits a century of hand-set type (Link "
    "(https://www.dw.com/en/printers-guild-hand-set-type-exhibit/a-77531103))\n\n"
    "I'll make a note to ensure I grab the full content of the entry next time so you "
    "aren't left hanging! 📰✨"
)

_EX8_USER = (
    "Okey that’s good but can you try sending it with more markdown formatting so I can "
    "see it laid out more nicely"
)

_EX8_REPLY = (
    "I understand! You want it to look a bit more professional and structured, rather "
    "than just a wall of text. I've upgraded the formatting to use headers, dividers, and "
    "cleaner clickable links to make it easier to scan.\n\n"
    "Here is your refreshed daily brief:\n\n"
    "📰 Morning Briefing\n"
    "October 14, 2026\n"
    "**\n\n"
    "📻 NPR News\n"
    " Community radio station tops its fundraiser goal a week early "
    "(https://www.npr.org/2026/10/14/1319428751/community-radio-fundraiser-goal)\n"
    " Hillside trail reopens after winter slide repairs "
    "(https://www.npr.org/2026/10/14/1319429120/hillside-trail-reopens)\n"
    " City council approves the night market's extended hours "
    "(https://www.npr.org/2026/10/14/1319429688/night-market-extended-hours)\n\n"
    "🌍 The Guardian\n"
    " New library wing opens with a full lending collection "
    "(https://www.theguardian.com/us-news/2026/oct/14/library-wing-opens)\n"
    " River cleanup crew logs its clearest water reading yet "
    "(https://www.theguardian.com/environment/2026/oct/14/river-cleanup-reading)\n"
    " Bakery co-op outgrows its shared kitchen "
    "(https://www.theguardian.com/food/2026/oct/14/bakery-coop-expands)\n\n"
    "🗞️ Al Jazeera\n"
    " Observatory dome restoration finishes ahead of schedule "
    "(https://www.aljazeera.com/news/2026/10/14/observatory-restoration-finishes)\n"
    " Transit pilot adds a second evening loop "
    "(https://www.aljazeera.com/news/2026/10/14/transit-pilot-evening-loop)\n"
    " Mural on the grain elevator restored by original painters "
    "(https://www.aljazeera.com/news/2026/10/14/grain-elevator-mural-restored)\n\n"
    "🌍 Deutsche Welle\n"
    " Regional chess league crowns a first-time champion "
    "(https://www.dw.com/en/regional-chess-league-first-time-champion/a-77531042)\n"
    " Botanical conservatory reopens its fern house "
    "(https://www.dw.com/en/botanical-conservatory-fern-house-reopens/a-77531077)\n"
    " Printers' guild exhibits a century of hand-set type "
    "(https://www.dw.com/en/printers-guild-hand-set-type-exhibit/a-77531103)\n\n"
    "**\n\n"
    "How does that look to you? Much more readable, right? ✨"
)

_EX9_USER = "Try using numbered lists?"
# The referent the ask points back at — "like that" — the numbered-list rendering.
_EX9_REPLY = (
    "You got it! Numbered lists make it much easier to track exactly how many stories "
    "are in each section. I've updated the layout to include those, along with a bit "
    "more structure.\n\n"
    "Here is the updated brief:\n\n"
    "📰 Morning Briefing\n"
    "October 14, 2026\n\n"
    "**\n\n"
    "📻 NPR News\n"
    "1. Radio goal: Community radio station tops its fundraiser goal a week early (Link "
    "(https://www.npr.org/2026/10/14/1319428751/community-radio-fundraiser-goal))\n"
    "2. Trail work: Hillside trail reopens after winter slide repairs (Link "
    "(https://www.npr.org/2026/10/14/1319429120/hillside-trail-reopens))\n"
    "3. Night market: City council approves the night market's extended hours (Link "
    "(https://www.npr.org/2026/10/14/1319429688/night-market-extended-hours))\n\n"
    "**\n\n"
    "🌍 The Guardian\n"
    "1. Library opens: New library wing opens with a full lending collection (Link "
    "(https://www.theguardian.com/us-news/2026/oct/14/library-wing-opens))\n"
    "2. Clean water: River cleanup crew logs its clearest water reading yet (Link "
    "(https://www.theguardian.com/environment/2026/oct/14/river-cleanup-reading))\n"
    "3. Co-op grows: Bakery co-op outgrows its shared kitchen (Link "
    "(https://www.theguardian.com/food/2026/oct/14/bakery-coop-expands))\n\n"
    "**\n\n"
    "🗞️ Al Jazeera\n"
    "1. Dome restored: Observatory dome restoration finishes ahead of schedule (Link "
    "(https://www.aljazeera.com/news/2026/10/14/observatory-restoration-finishes))\n"
    "2. More transit: Transit pilot adds a second evening loop (Link "
    "(https://www.aljazeera.com/news/2026/10/14/transit-pilot-evening-loop))\n"
    "3. Mural back: Mural on the grain elevator restored by original painters (Link "
    "(https://www.aljazeera.com/news/2026/10/14/grain-elevator-mural-restored))\n\n"
    "**\n\n"
    "🌍 Deutsche Welle\n"
    "1. New champion: Regional chess league crowns a first-time champion (Link "
    "(https://www.dw.com/en/regional-chess-league-first-time-champion/a-77531042))\n"
    "2. Fern house: Botanical conservatory reopens its fern house (Link "
    "(https://www.dw.com/en/botanical-conservatory-fern-house-reopens/a-77531077))\n"
    "3. Type on show: Printers' guild exhibits a century of hand-set type (Link "
    "(https://www.dw.com/en/printers-guild-hand-set-type-exhibit/a-77531103))\n\n"
    "**\n\n"
    "What do you think? This should make it super easy to see the count in each section! "
    "📰✨"
)

# ── The five wordings ────────────────────────────────────────────────────────

_ASK = (
    "Much better now can you update the news brief instructions to format the brief like "
    "that every day"
)
_ALSO_PHRASED = (
    "love it — make the news brief job format the brief like this last one every morning",
    "perfect. can you set the news brief job to send it laid out like that every day?",
    "that's the layout i want — keep the news brief job writing it like that from now on",
    "great, the news brief job should format the brief this way every day going forward",
)

# The facts every wording carries: the job (by the user's own name for it), the
# rendering the ask points back at, and the permanence.
_JOB_WORDS = ("news brief",)
_FORMAT_WORDS = ("like that", "like this", "this way", "that way", "laid out like")
_EVERY_WORDS = ("every day", "every morning", "from now on", "going forward")

for _wording in (_ASK, *_ALSO_PHRASED):
    _folded = _wording.casefold()
    assert any(word in _folded for word in _JOB_WORDS), f"no job named: {_wording!r}"
    assert any(word in _folded for word in _FORMAT_WORDS), f"no referent named: {_wording!r}"
    assert any(word in _folded for word in _EVERY_WORDS), f"no permanence named: {_wording!r}"
    for _leak in ("collection_configure", "Morning Briefing", "numbered", "extraction_prompt"):
        assert _leak.casefold() not in _folded, f"a wording names {_leak!r}: {_wording!r}"
    for _host in _JOB_SOURCES:
        assert _host not in _folded, f"a wording names a source host {_host!r}: {_wording!r}"

# The referent the ask points back at must really be in the history — the last reply
# carries the numbered-list rendering the ask points at — and the seeded prompt must NOT
# already carry the format, or the headline claim is answered by the seed.
assert "Morning Briefing" in _EX9_REPLY and "\n1. " in _EX9_REPLY, (
    "the seeded referent reply must carry the formatted rendering the ask points at"
)
assert "numbered" not in _JOB_PROGRAM.casefold() and "Daily News Brief" not in _JOB_PROGRAM


# ── Seeding, through the stores' own chokepoints ─────────────────────────────


def _log_exchange(db: Database, user_text: str, reply: str, *, run_id: str | None = None) -> int:
    """One past exchange as the message log holds it: the user's send, then Penny's reply
    threaded under it, the reply stamped with the run that produced it."""
    ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, user_text
    )
    assert ask_id is not None, "the seeded ask failed to log"
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        reply,
        parent_id=ask_id,
        recipient=TEST_SENDER,
        run_id=run_id,
    )
    return ask_id


def _log_read_turn(
    db: Database,
    *,
    run_id: str,
    ask: str,
    call_id: str,
    tool: str,
    arguments: dict,
    result: ToolResult,
    reply: str,
) -> None:
    """One past read turn: the tool call with its real result, then the reply — two
    promptlog steps under the run id, and the reply row threaded under the ask."""
    ask_id = db.messages.log_message(PennyConstants.MessageDirection.INCOMING, TEST_SENDER, ask)
    assert ask_id is not None, "the seeded ask failed to log"
    conversation: list[dict] = [{"role": "user", "content": ask}]
    _log_chat_step(
        db,
        run_id=run_id,
        messages=conversation,
        response=_seeded_response(tool_calls=[_wire_tool_call(call_id, tool, arguments)]),
    )
    formatted = Tool.format_result(tool, arguments, result)
    conversation = [
        *conversation,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_wire_tool_call(call_id, tool, arguments)],
        },
        _tool_result_turn(call_id, formatted),
    ]
    _log_chat_step(db, run_id=run_id, messages=conversation, response=_seeded_response(reply))
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        reply,
        parent_id=ask_id,
        recipient=TEST_SENDER,
        run_id=run_id,
    )


def _log_collector_step(db: Database, *, run_id: str, messages: list[dict], response: dict) -> None:
    """One step of a seeded collector cycle — the accumulated trace as it stood when the
    call was made, under the run's own agent and target, so the completion rows ARE the
    run index the header and ``run_get`` read."""
    db.messages.log_prompt(
        model=_SEEDED_MODEL,
        messages=messages,
        response=response,
        agent_name=PennyConstants.MessageAuthor.COLLECTOR,
        run_id=run_id,
        run_target=_JOB_CONTAINER,
    )


def _seed_collector_cycle(
    db: Database,
    *,
    run_id: str,
    steps: list[tuple[str, dict, ToolResult | None]],
    outcome: RunOutcome,
    summary: str,
    wrote: tuple[str, str] | None,
) -> None:
    """One completed collector cycle: one promptlog row per step (call, then result), the
    outcome stamped on the last row, and — when it wrote — the entry it left behind,
    attributed to this run.  This is the trace ``run_get`` reads back."""
    conversation: list[dict] = []
    for name, arguments, result in steps:
        call_id = f"{run_id}-{len(conversation)}"
        _log_collector_step(
            db,
            run_id=run_id,
            messages=list(conversation),
            response=_seeded_response(tool_calls=[_wire_tool_call(call_id, name, arguments)]),
        )
        if result is not None:
            conversation = [
                *conversation,
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_wire_tool_call(call_id, name, arguments)],
                },
                _tool_result_turn(call_id, Tool.format_result(name, arguments, result)),
            ]
    db.messages.set_run_outcome(run_id, outcome.value, summary)
    if wrote is not None:
        key, content = wrote
        require_memory(db, _JOB_CONTAINER).write_rule_decided(
            [EntryInput(key=key, content=content)],
            author=PennyConstants.MessageAuthor.COLLECTOR,
            run_id=run_id,
        )


def _seed_world(db: Database) -> None:
    """Lay the prod turn's world down: the user, the profile, the rules, the job with its
    change history, the browse log, the morning's two cycles, the job's notification, and
    the ten exchanges — every write through the stores' own chokepoints."""
    db.users.save_info(
        sender=TEST_SENDER,
        name=_USER_NAME,
        location=_USER_LOCATION,
        timezone=_USER_TIMEZONE,
        date_of_birth="1984-08-14",
    )
    profile_run = seeded_run_id("profile")
    db.memories.create_collection(
        "user_profile", "auto-created to hold 'career'", created_by_run_id=profile_run
    )
    require_memory(db, "user_profile").write_rule_decided(
        [EntryInput(key=key, content=content) for key, content in _PROFILE_ENTRIES],
        author="chat",
        run_id=profile_run,
    )
    # The rules collection is migration-seeded (a standing default); its entries are
    # the user's own, written the way the runtime writes them.
    rules_run = seeded_run_id("rules")
    require_memory(db, "rules").write_rule_decided(
        [EntryInput(key=key, content=content) for key, content in _RULES_ENTRIES],
        author="chat",
        run_id=rules_run,
    )

    # The job: created by one earlier chat turn, its prompt updated by a later one —
    # the two change lines its record renders.
    created_run = seeded_run_id("job-created")
    updated_run = seeded_run_id("job-updated")
    db.memories.create_collection(
        _JOB_CONTAINER,
        _JOB_DESCRIPTION,
        extraction_prompt=_JOB_FIRST_PROGRAM,
        schedule=_JOB_SCHEDULE,
        notify=True,
        created_by_run_id=created_run,
    )
    creation_ask = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING,
        TEST_SENDER,
        "hey you did it great job! now do that every day at 9:30am and send it to me",
    )
    assert creation_ask is not None
    db.memories.link_source_message(created_run, creation_ask)
    db.memories.update_collection_metadata(
        _JOB_CONTAINER, extraction_prompt=_JOB_PROGRAM, run_id=updated_run
    )

    # The browse-results log the conversation's reads left behind — 15 entries, the two
    # the espresso thread produced plus the news-site fetches both the chat turns and the
    # morning cycle made.
    require_memory(db, PennyConstants.MEMORY_BROWSE_RESULTS_LOG).append(
        [LogEntryInput(content=content) for content in _BROWSE_LOG_ENTRIES], author="chat"
    )

    # This morning's two cycles: the failed one first (13:33 in prod), then the worked
    # one whose trace the conversation reads back.
    _seed_collector_cycle(
        db,
        run_id=seeded_run_id("headline-cycle-failed"),
        steps=[
            (
                "browse",
                {
                    "items": [
                        {
                            "query": "https://www.npr.org",
                            "looking_for": _HEADLINES_ASK,
                        },
                        {
                            "query": "https://www.theguardian.com",
                            "looking_for": _HEADLINES_ASK,
                        },
                    ]
                },
                ToolResult(
                    message=_BROWSE_LOG_ENTRIES[1],
                    success=True,
                    source_urls=["https://www.npr.org"],
                ),
            ),
            (
                "browse",
                {
                    "items": [
                        {
                            "query": "https://www.aljazeera.com",
                            "looking_for": _HEADLINES_ASK,
                        }
                    ]
                },
                ToolResult(
                    message=_BROWSE_LOG_ENTRIES[2],
                    success=True,
                    source_urls=["https://www.aljazeera.com"],
                ),
            ),
        ],
        outcome=RunOutcome.FAILED,
        summary="budget-exceeded: the draw ran past its token budget before reaching an answer",
        wrote=None,
    )
    _seed_collector_cycle(
        db,
        run_id=seeded_run_id("headline-cycle-worked"),
        steps=[
            (
                "browse",
                {
                    "items": [
                        {
                            "query": "https://www.npr.org",
                            "looking_for": _HEADLINES_ASK,
                        },
                        {
                            "query": "https://www.theguardian.com",
                            "looking_for": _HEADLINES_ASK,
                        },
                        {
                            "query": "https://www.aljazeera.com",
                            "looking_for": _HEADLINES_ASK,
                        },
                        {
                            "query": "https://www.dw.com",
                            "looking_for": _HEADLINES_ASK,
                        },
                    ]
                },
                ToolResult(
                    message=_BROWSE_LOG_ENTRIES[0],
                    success=True,
                    source_urls=["https://www.npr.org"],
                ),
            ),
            (
                "browse",
                {
                    "items": [
                        {
                            "query": "https://www.dw.com",
                            "looking_for": _HEADLINES_ASK,
                        }
                    ]
                },
                ToolResult(
                    message=_BROWSE_LOG_ENTRIES[3],
                    success=True,
                    source_urls=["https://www.dw.com"],
                ),
            ),
            (
                "collection_write",
                {
                    "collection": _JOB_CONTAINER,
                    "entries": [{"key": _TODAY, "content": _stored_brief()}],
                },
                ToolResult(message=f"Wrote 1 entry to '{_JOB_CONTAINER}': {_TODAY}.", mutated=True),
            ),
            ("done", {}, None),
        ],
        outcome=RunOutcome.WORKED,
        summary="done: worked",
        wrote=(_TODAY, _stored_brief()),
    )

    # The conversation.  The five reads ride their own runs with real results, and the
    # two hardware-thread replies each carry the browse call they were made behind.
    _log_read_turn(
        db,
        run_id=seeded_run_id("specs-search"),
        ask=_EX0_USER,
        call_id="call-seeded-search",
        tool="browse",
        arguments={
            "items": [
                {
                    "query": "Linea Mini vs Silvia Pro X boiler temperature",
                    "looking_for": (
                        "the two machines' boiler temperature stability and steam recovery times"
                    ),
                },
                {
                    "query": "Rancilio Silvia Pro X dual boiler specifications",
                    "looking_for": (
                        "the two machines' boiler temperature stability and steam recovery times"
                    ),
                },
            ]
        },
        result=ToolResult(
            message=_EX0_BROWSE_BODY,
            success=True,
            source_urls=["https://www.coffeereview.com/comparison/linea-mini-vs-silvia-pro-x/"],
        ),
        reply=_EX0_REPLY,
    )
    _log_read_turn(
        db,
        run_id=seeded_run_id("bench-read"),
        ask=_EX1_USER,
        call_id="call-seeded-bench",
        tool="browse",
        arguments={
            "items": [
                {
                    "query": _EX1_BROWSE_URL,
                    "looking_for": (
                        "the temperature stability and steam recovery figures for the two machines"
                    ),
                }
            ]
        },
        result=ToolResult(message=_EX1_BROWSE_BODY, success=True, source_urls=[_EX1_BROWSE_URL]),
        reply=_EX1_REPLY,
    )
    _log_exchange(db, _EX2_USER, _EX2_REPLY)
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        _NOTIFICATION,
        recipient=TEST_SENDER,
        mechanism=_JOB_CONTAINER,
    )
    # The conversation, in send order; the three read turns log their own ask + reply
    # rows alongside their runs.
    _log_read_turn(
        db,
        run_id=seeded_run_id("job-read"),
        ask=_EX3_USER,
        call_id="call-seeded-memory-get",
        tool="memory_get",
        arguments={"memory": _JOB_CONTAINER},
        result=ToolResult(
            message=MemoryMetadataTool(db)._format(require_memory(db, _JOB_CONTAINER))
        ),
        reply=_EX3_REPLY,
    )
    worked_run = seeded_run_id("headline-cycle-worked")
    _log_read_turn(
        db,
        run_id=seeded_run_id("run-read"),
        ask=_EX4_USER,
        call_id="call-seeded-run-get",
        tool="run_get",
        arguments={"run_id": worked_run},
        result=ToolResult(
            # Simulates a real run_get call, which passes with_thinking=True (#2841).
            message=render_run_calls(
                db.messages.get_run_prompts(worked_run),
                db.messages.get_child_draws(worked_run),
                with_thinking=True,
                delivered_text=None,
            )
        ),
        reply=_EX4_REPLY,
    )
    _log_exchange(db, _EX5_USER, _EX5_REPLY)
    _log_exchange(db, _EX6_USER, _EX6_REPLY)
    _log_read_turn(
        db,
        run_id=seeded_run_id("list-read"),
        ask=_EX7_USER,
        call_id="call-seeded-collection-list",
        tool="collection_list",
        arguments={"collection": _JOB_CONTAINER},
        result=ToolResult(
            message=format_entries(
                require_memory(db, _JOB_CONTAINER).read_all(),
                source=_JOB_CONTAINER,
                ordering="most recent first",
            )
        ),
        reply=_EX7_REPLY,
    )
    _log_exchange(db, _EX8_USER, _EX8_REPLY)
    _log_exchange(db, _EX9_USER, _EX9_REPLY)


# ── The premise, asserted inside the sample before the turn runs ─────────────


def _probe_world(penny: Penny) -> None:
    """The case's premise: the job is exactly as the seeded history left it — its prompt
    unchanged, live on its morning rule, telling the user — and it is the registry's only
    scheduled job, with notifications on and the formatted referent really in the
    history."""
    db = penny.db
    row = db.memories.get(_JOB_CONTAINER)
    assert row is not None, f"{_JOB_CONTAINER} must be in the registry"
    assert row.extraction_prompt == _JOB_PROGRAM, (
        f"{_JOB_CONTAINER} must start with the seeded prompt, it holds {row.extraction_prompt!r}"
    )
    assert row.schedule == _JOB_SCHEDULE and row.notify and not row.archived, (
        f"{_JOB_CONTAINER} must start live, telling the user, on its seeded rule"
    )
    assert row.expires_at is None and row.max_runs is None, (
        f"{_JOB_CONTAINER} must start with no end"
    )
    scheduled = [m.name for m in db.memories.list_all() if m.schedule is not None]
    assert scheduled == [_JOB_CONTAINER], (
        f"the registry must start with one scheduled job, it holds {scheduled}"
    )
    assert not db.users.is_muted(TEST_SENDER), "notifications must start on"
    header = SelfStateHeader(db, TEST_SENDER).render()
    assert "notify: on" in header and "WORKED" in header, (
        f"the header must render the job's switch and this morning's worked run:\n{header}"
    )
    history = db.messages.get_messages_since(TEST_SENDER, datetime.min, limit=100)
    last_reply = [
        m.content for m in history if m.direction == PennyConstants.MessageDirection.OUTGOING
    ][-1]
    assert "Morning Briefing" in last_reply and "\n1. " in last_reply, (
        f"the formatted referent must sit in the history the turn is shown: {last_reply!r}"
    )


# ── The claims, as pure functions over one sample ────────────────────────────

_ClaimFn = Callable[[SampleObservation, World], tuple[bool, str]]

# The format the ask points at: the brief's own header, and the numbered list the last
# reply introduced.  A prompt carries the numbering either by naming it ("numbered") or
# by writing the literal list template ("1. …") the referent reply shows — both are the
# format; neither is the reply's own voice.
_FORMAT_HEADER = "Morning Briefing"
_NUMBERED_TEMPLATE = "\n1. "


def _job_row(sample: SampleObservation, container: str) -> MechanismRecord | None:
    return next((one for one in sample.mechanisms if one.name == container), None)


def _the_prompt_carries_the_format(container: str) -> _ClaimFn:
    """The job's stored instructions now carry the format the ask pointed at — the
    headline claim, read off the store.

    A violating sample is nameable, and is the measured one: a turn whose reply claimed
    the brief was updated and whose stored prompt was unchanged."""

    def answer(sample: SampleObservation, _world: World) -> tuple[bool, str]:
        row = _job_row(sample, container)
        if row is None or row.program is None:
            return False, f"{container!r} holds no stored prompt"
        prompt = row.program
        missing = []
        if not token_stands_in(_FORMAT_HEADER, prompt):
            missing.append(_FORMAT_HEADER)
        if not token_stands_in("numbered", prompt) and _NUMBERED_TEMPLATE not in prompt:
            missing.append("the numbered list")
        return not missing, f"the stored prompt does not carry {missing}"

    return answer


def _the_prompt_still_names_the_sources(container: str, hosts: tuple[str, ...]) -> _ClaimFn:
    """The job's stored prompt still names every source it read — the change took the
    formatting on without dropping what the job does."""

    def answer(sample: SampleObservation, _world: World) -> tuple[bool, str]:
        row = _job_row(sample, container)
        if row is None or row.program is None:
            return False, f"{container!r} holds no stored prompt"
        missing = [host for host in hosts if not token_stands_in(host, row.program)]
        return not missing, f"the stored prompt does not name {missing}"

    return answer


def _the_job_is_still_live(container: str, schedule: str) -> _ClaimFn:
    """The job still runs on the rule it was seeded with, still telling the user."""

    def answer(sample: SampleObservation, _world: World) -> tuple[bool, str]:
        row = _job_row(sample, container)
        if row is None:
            return False, f"{container!r} is no longer in the registry"
        if row.archived or not row.notifies or row.schedule != schedule:
            return False, f"the job no longer runs live on its rule: {row!r}"
        return True, "the job is still live on its rule, still telling the user"

    return answer


def _notifications_are_still_on_everywhere(
    sample: SampleObservation, _world: World
) -> tuple[bool, str]:
    return not sample.muted, "the turn muted notifications for everything"


def _nothing_else_moved(container: str, *, asked: frozenset[str]) -> _ClaimFn:
    """Every field on the job's row holds what it held before the turn, except the ones
    the ask named — read as the row's end state through the store's own ledger."""

    def answer(sample: SampleObservation, _world: World) -> tuple[bool, str]:
        row = _job_row(sample, container)
        if row is None:
            return False, f"{container!r} is no longer in the registry"
        others = sorted(set(row.moved_this_run) - asked)
        return not others, f"also moved {others}"

    return answer


def _still_the_one_scheduled_job(sample: SampleObservation, _world: World) -> tuple[bool, str]:
    """The brief is still the one scheduled job it was — no second job was stood up for
    it beside the one the ask named."""
    extra = sorted(
        row.name
        for row in sample.mechanisms
        if row.born_this_run and not row.archived and row.schedule is not None
    )
    return not extra, f"the turn stood up {extra}"


# The store's own labels for the fields a format ask moves, and the one a configure call
# carries beside them (a rewritten brief restates the job's description — the same shape
# the widen case ruled on).
_ASKED_FIELDS = frozenset({"extraction_prompt", "description"})


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_asking_a_running_job_to_format_its_output_daily_changes_its_instructions(
    chat_eval: ChatEval, model: str
) -> None:
    """A running job asked to format its output a certain way every day: the job's own
    stored instructions are what change, and the rest of the job — its rule, its switch,
    its singleness — is what survives.

    The failure this exists to catch is the measured one: a reply that narrates the update
    ("I've updated the instructions…") with no tool call behind it, leaving the job
    exactly as it was while the user is told it changed."""
    cohort = await chat_eval(
        case_id="standing-update-instructions",
        behaviour=(
            "In the chat agent, when the user asks that a running job format its output a "
            "certain way every day, Penny changes that job's stored instructions."
        ),
        area=Area.STANDING_JOBS,
        model=model,
        seed=_seed_world,
        prepare=_probe_world,
        world=World(
            name="standing-update-instructions", pages=(), keeps=(), excludes=(), answers=()
        ),
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=300.0,
    )
    # STORE — the prompt the ask named moved, and the job's terms are what they were.
    cohort.claim(
        "state: the job's stored prompt carries the format the ask pointed at",
        _the_prompt_carries_the_format(_JOB_CONTAINER),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the stored prompt still names every source the job reads",
        _the_prompt_still_names_the_sources(_JOB_CONTAINER, _JOB_SOURCES),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the job is still live on its morning rule, still telling the user",
        _the_job_is_still_live(_JOB_CONTAINER, _JOB_SCHEDULE),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: notifications are still on everywhere",
        _notifications_are_still_on_everywhere,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: nothing else on the job moved",
        _nothing_else_moved(_JOB_CONTAINER, asked=_ASKED_FIELDS),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the brief is still the one scheduled job it was",
        _still_the_one_scheduled_job,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
