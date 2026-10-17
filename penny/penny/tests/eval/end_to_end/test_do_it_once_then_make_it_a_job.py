"""End-to-end: a task done once in chat, then made a daily job (#2433).

**The declared second kind** `docs/eval-case-design.md` names: a SCRIPT of real chat turns —
a one-off ask Penny carries out once in plain conversation, then "can you make that a daily
job?", then the user's walkthrough once Penny asks to be taught, and an acceptance giving the
job its terms — followed by the job those turns created, run through its OWN real cycles
against a page whose headlines change.

**Seen in production**: Penny did the one-off read, then on "make that a daily job?" reasoned
"I have already done that in previous turns… I'll assume the skill is now 'learned'," called
``collection_configure`` with an invented skill name, and — once that failed — followed the
tool's own failure text into browsing and writing an entry OUTSIDE any teach round, where no
skill can be minted.  Elicit turns mint no skill, so the round ended with a stray collection,
no skill and no job.  **Ruled** (code owner): doing the task once in conversation is not the
same as having been taught it — elicit should ask for the walkthrough regardless of what
already happened earlier in the conversation.

No routine, no job and no collection exist before the script runs: the world is COLD, so
whatever container the round builds and which collection the turns land the job on are left to
the model and read nowhere a claim depends on it.  The driver finds the created job
STRUCTURALLY (``end_to_end_eval``'s own read of whichever collection the script scheduled) and
runs ITS cycles; the claims below never name it.

**Claims read only:** a job she created's own terms (cadence, notify, end), what the store's
entries hold after each cycle, and provenance — never which collection the job sits on, the
routine's shape or bound values, or which conversation-machine states the script passed
through on the way (#2375's ruling, carried over from the first two end-to-end cases).

**The page is real-size** — several kilobytes: a title line, a dozen nav links, dozens of
other story links with their own headlines, and a footer of section links — the shape a real
news front page's link index reads as (#1942's `page_text.ts` split: an index page has no
article body, so the content script returns one markdown link per story). The baseline and
changed top-3 headlines are mutually exclusive substrings of each other and of the rest of the
page, so a sample that invented a plausible headline cannot satisfy a claim that names the one
the page actually carries.

Report-only (``min_pass_rate=None``). All content is synthetic, on an ``example`` domain,
because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.constants import PageReading
from penny.database import Database
from penny.database.memory import MemoryType
from penny.tests.eval.conftest import EVAL_MODELS, EndToEndArm, EndToEndEval
from penny.tests.eval.end_to_end.world import END_TO_END_WALK, held_text_at
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    JOB_TERMS,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.job_end import NO_END
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_CASE_ID = "end-to-end-do-it-once-then-make-it-a-job"
_FAMILY = "end-to-end"

_BEHAVIOUR = (
    "In the chat agent and the job it stands up, when the user has Penny do a task once in "
    "conversation and then asks to make it a daily job, Penny asks to be walked through it, "
    "learns it from the walkthrough, and stands up a job whose cycles record the page's "
    "headlines and replace them when they change."
)

# ── The front page the one-off ask and the walkthrough read, and the two readings its ──
# ── job's own cycles see ──────────────────────────────────────────────────────

_URL = "https://citydesk.example/front"

# The baseline top three headlines, in document order, each with a smallest-unique anchor the
# claims below read.
_TOP1 = "Harbor tunnel reopens three months ahead of schedule after a decade of delays"
_TOP2 = "Transit authority adds a night bus loop after years of commuter requests"
_TOP3 = "City council approves funding for a new downtown library branch"
_TOP1_ANCHOR = "Harbor tunnel"
_TOP2_ANCHOR = "Transit authority"
_TOP3_ANCHOR = "downtown library"

# The changed top three, mutually exclusive of the baseline three and of everything else the
# page carries, so a sample that invented a plausible headline cannot satisfy either claim.
_NEW1 = "Storm cleanup crews finish clearing the waterfront two days early"
_NEW2 = "Mayor unveils a plan to repave the old quarter's cobbled streets"
_NEW3 = "Volunteer crews reopen the riverside path after last week's washout"
_NEW1_ANCHOR = "Storm cleanup"
_NEW2_ANCHOR = "repave the old quarter"
_NEW3_ANCHOR = "riverside path"


def _front_page_text(top1: str, top2: str, top3: str) -> str:
    """One front page, in the link-index shape the content script returns for such a page —
    one markdown link per story, nav item or footer item, nothing else.  ``top1``/``top2``/
    ``top3`` are the only thing that ever varies between the baseline and the changed page:
    everything else — the nav, the dozen filler stories, the footer — stays byte-identical,
    the way a real front page keeps most of its lower stories in place while its top headlines
    turn over."""
    return (
        "Title: Citydesk — front page\n"
        f"{_URL}\n"
        "Local news and civic affairs for a fictional city, updated through the day\n"
        "[World](https://citydesk.example/world)\n"
        "[Local](https://citydesk.example/local)\n"
        "[Business](https://citydesk.example/business)\n"
        "[Opinion](https://citydesk.example/opinion)\n"
        "[Sports](https://citydesk.example/sports)\n"
        "[Arts](https://citydesk.example/arts)\n"
        "[Weather](https://citydesk.example/weather)\n"
        "[Obituaries](https://citydesk.example/obituaries)\n"
        "[Classifieds](https://citydesk.example/classifieds)\n"
        "[Subscribe](https://citydesk.example/subscribe)\n"
        "[Archive](https://citydesk.example/archive)\n"
        "[Contact us](https://citydesk.example/contact)\n"
        f"[{top1}](https://citydesk.example/local/2036/story-top-1)\n"
        f"[{top2}](https://citydesk.example/local/2036/story-top-2)\n"
        f"[{top3}](https://citydesk.example/local/2036/story-top-3)\n"
        "[Bakery on Fen Street marks fifty years under the same family after a quiet "
        "anniversary open house](https://citydesk.example/local/2036/bakery-fifty-years)\n"
        "[School board approves a redrawn bus route after months of parent "
        "complaints](https://citydesk.example/local/2036/school-board-redraws-bus-route)\n"
        "[Harbor market vendors ask for later closing hours through the summer "
        "season](https://citydesk.example/business/2036/harbor-market-later-hours)\n"
        "[Community garden expands onto the lot behind the old fire "
        "station](https://citydesk.example/local/2036/garden-expands-old-fire-station)\n"
        "[City pool reopens a week early after repairs finished ahead of "
        "plan](https://citydesk.example/local/2036/pool-reopens-early)\n"
        "[Orchestra announces a free concert in the park for the autumn "
        "equinox](https://citydesk.example/arts/2036/orchestra-free-park-concert)\n"
        "[Small business grants open for a second round after the first sold "
        "out](https://citydesk.example/business/2036/grants-second-round)\n"
        "[Cycling lane pilot gets a six-month extension after early ridership "
        "numbers](https://citydesk.example/local/2036/cycling-lane-extension)\n"
        "[Historic theatre begins a restoration expected to take two "
        "years](https://citydesk.example/arts/2036/theatre-restoration-begins)\n"
        "[Food bank reports its busiest month in three years ahead of the "
        "holidays](https://citydesk.example/local/2036/food-bank-busiest-month)\n"
        "[Youth sports league adds a second field after waitlists grew too "
        "long](https://citydesk.example/sports/2036/youth-league-second-field)\n"
        "[Downtown parking rates rise for the first time in five "
        "years](https://citydesk.example/business/2036/parking-rates-rise)\n"
        "[Volunteers finish repainting the footbridge ahead of the fall "
        "festival](https://citydesk.example/local/2036/footbridge-repainted)\n"
        "[Public library extends weekend hours on a trial "
        "basis](https://citydesk.example/local/2036/library-extends-weekend-hours)\n"
        "[Farmers market relocates to the square for the rest of the "
        "season](https://citydesk.example/business/2036/market-relocates-square)\n"
        "[Animal shelter waives adoption fees for the long "
        "weekend](https://citydesk.example/local/2036/shelter-waives-fees)\n"
        "[Road crews begin resurfacing the ring road in overnight "
        "shifts](https://citydesk.example/local/2036/ring-road-resurfacing)\n"
        "[Museum adds a late-night opening on the first Friday of each "
        "month](https://citydesk.example/arts/2036/museum-late-night-friday)\n"
        "[Little league finals move to the renovated riverside "
        "diamond](https://citydesk.example/sports/2036/finals-move-riverside-diamond)\n"
        "[World news](https://citydesk.example/world-news)\n"
        "[Business desk](https://citydesk.example/business-desk)\n"
        "[Opinion](https://citydesk.example/opinion-desk)\n"
        "[Sports scores](https://citydesk.example/scores)\n"
        "[Arts & culture](https://citydesk.example/arts-culture)\n"
        "[Weather forecasts](https://citydesk.example/forecasts)\n"
        "[Obituaries](https://citydesk.example/obituaries-desk)\n"
        "[Classifieds](https://citydesk.example/classifieds-desk)\n"
        "[Subscribe](https://citydesk.example/subscribe-now)\n"
        "[Newsletters](https://citydesk.example/newsletters)\n"
        "[About us](https://citydesk.example/about)\n"
        "[Contact the newsroom](https://citydesk.example/contact-newsroom)\n"
        "[Older stories](https://citydesk.example/archive/2036)"
    )


_BASELINE_PAGE = CannedPage(
    match="citydesk.example/front",
    text=_front_page_text(_TOP1, _TOP2, _TOP3),
    reading=PageReading.LINK_LIST,
)
_MOVED_PAGE = CannedPage(
    match="citydesk.example/front",
    text=_front_page_text(_NEW1, _NEW2, _NEW3),
    reading=PageReading.LINK_LIST,
)

# The entry condition is COLD — no routine, no job, no collection — and the one-off ask reads
# this same page once in plain conversation before the job exists, so the job's two cycles are:
# the page as the one-off ask and the walkthrough both read it, then the page with its top
# three headlines changed.
_CYCLES = ([_BASELINE_PAGE], [_MOVED_PAGE])

# The ground the SCRIPT's own turns are answered against — one page, installed once before any
# turn runs, the way a chat case's world always is.  ``keeps``/``excludes``/``answers`` are all
# empty by report: the script sets a job running later, so it asks to be TOLD rather than to be
# given a value back, and the ask names nothing to exclude.
_WORLD = World(name=_CASE_ID, pages=(_BASELINE_PAGE,), keeps=(), excludes=())

# The terms the acceptance gives: fires daily, no end stated, tells the user.  Read against
# ``_the_new_job`` in ``assertions.py`` — existentially, since "a job she created" is the
# claim, never "exactly one".
_CADENCE_SECONDS = 86400

# ── The five scripts: a one-off ask, "make it a job?", a walkthrough, an acceptance ──
#
# What varies is how a person says each of the four turns: which verb opens the one-off ask,
# how "daily job" gets said, how the walkthrough names its two actions, which word the
# acceptance opens with.  What does NOT vary is the url (named in the one-off ask and the
# walkthrough), "top 3"/"top three" headlines (named in the one-off ask and the walkthrough),
# that the second turn asks for a DAILY job, and that the acceptance gives the 8am cadence and
# a telling clause — every wording carries all of them, because the claims below hinge on them.
_SCRIPTS: tuple[tuple[str, str, str, str], ...] = (
    (
        f"can you pull the top 3 headlines off {_URL} for me?",
        "nice, can you make that into a daily job?",
        f"sure — go to {_URL}, pull the top 3 headlines and keep them",
        "great — do that every morning at 8am and tell me when they change",
    ),
    (
        f"grab the top 3 headlines from {_URL}, would you?",
        "could you turn that into a job that runs every day?",
        f"ok — head to {_URL}, pull the top 3 headlines and hang onto them",
        "yes — run it every morning at 8am and let me know if they change",
    ),
    (
        f"what are the top 3 headlines on {_URL} right now?",
        "can you set that up as a daily job?",
        f"alright — visit {_URL}, find the top 3 headlines and save them",
        "do it every morning at 8am and message me when they change",
    ),
    (
        f"check {_URL} and tell me the top 3 headlines",
        "can you make that run as a job every day?",
        f"yep — open {_URL}, get the top 3 headlines and remember them",
        "sounds good — every morning at 8am, and tell me if they change",
    ),
    (
        f"could you read {_URL} and give me its top 3 headlines?",
        "can that become a daily job for you?",
        f"sure thing — go to {_URL}, pull out the top 3 headlines and keep them around",
        "perfect — check it every morning at 8am and let me know when they change",
    ),
)

_TOP3_TOKENS = ("top 3", "top three")
_DAILY_TOKENS = ("daily", "every day")
_CADENCE_TOKEN = "8am"
_TELLING_CLAUSES = ("tell me", "let me know", "message")


def _assert_every_script_carries_the_facts() -> None:
    """Every one of the five scripts' four turns carries the facts the claims below read —
    the url and "top 3" in the one-off ask and the walkthrough, "daily" in the second turn, and
    the acceptance's 8am cadence and telling clause — so a dropped fact fails in this assertion
    rather than as a puzzling score after a paid run."""
    for ask, make_it_a_job, walkthrough, acceptance in _SCRIPTS:
        assert _URL in ask, f"{_CASE_ID}: this one-off ask names no page — {ask!r}"
        assert any(token in ask for token in _TOP3_TOKENS), (
            f"{_CASE_ID}: this one-off ask names no top-3 headlines — {ask!r}"
        )
        assert any(token in make_it_a_job for token in _DAILY_TOKENS), (
            f"{_CASE_ID}: this turn never asks for a DAILY job — {make_it_a_job!r}"
        )
        assert _URL in walkthrough, f"{_CASE_ID}: this walkthrough names no page — {walkthrough!r}"
        assert any(token in walkthrough for token in _TOP3_TOKENS), (
            f"{_CASE_ID}: this walkthrough names no top-3 headlines — {walkthrough!r}"
        )
        assert _CADENCE_TOKEN in acceptance, (
            f"{_CASE_ID}: this acceptance gives no {_CADENCE_TOKEN!r} cadence — {acceptance!r}"
        )
        assert any(clause in acceptance for clause in _TELLING_CLAUSES), (
            f"{_CASE_ID}: this acceptance never asks to be told — {acceptance!r}"
        )


_assert_every_script_carries_the_facts()


def _seed(db: Database) -> None:
    """The entry condition: no routine taught, no job running.  Asserted rather than assumed,
    so a world that drifted fails here rather than after a live run.

    Checked against COLLECTIONS only, never the whole store — a fresh eval database is built
    exactly like production (``create_tables()`` then ``migrate()``), and migrations seed
    system log memories (``penny-messages``, the browse-results log, …) into every database,
    cold teach or not."""
    assert not db.skills.list_all(), "the registry must start empty for a cold teach"
    collections = [row for row in db.memories.list_all() if row.type == MemoryType.COLLECTION]
    assert not collections, f"no collection may exist yet for a cold teach, found {collections}"


_ARMS = tuple(
    EndToEndArm(
        label=script[0], turns=script, seed=_seed, seed_skills=None, world=_WORLD, cycles=_CYCLES
    )
    for script in _SCRIPTS
)


# ── The claims ───────────────────────────────────────────────────────────────


def _store_holds_the_top_headlines(sample: SampleObservation, _world: World) -> Answer:
    held = held_text_at(sample.cycles, 0)
    anchors = (_TOP1_ANCHOR, _TOP2_ANCHOR, _TOP3_ANCHOR)
    ok = all(anchor in held for anchor in anchors)
    return ok, f"expected all of {anchors}; held {held!r}"


def _store_holds_the_changed_headlines(sample: SampleObservation, _world: World) -> Answer:
    held = held_text_at(sample.cycles, 1)
    anchors = (_NEW1_ANCHOR, _NEW2_ANCHOR, _NEW3_ANCHOR)
    ok = all(anchor in held for anchor in anchors)
    return ok, f"expected all of {anchors}; held {held!r}"


_MEASURED = (END_TO_END_WALK, TOOL_SEQUENCE, ENTRIES_STORED, JOB_TERMS, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_doing_it_once_then_asking_for_a_job_is_taught_through_a_walkthrough(
    end_to_end_eval: EndToEndEval, model: str
) -> None:
    """A one-off ask Penny carries out once in plain conversation, then "make that a daily
    job?", a walkthrough and an acceptance with terms — all driven for real — then the job
    those turns created runs for real across a front page whose top headlines change."""
    cohort = await end_to_end_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.BACKGROUND_COLLECTORS,
        model=model,
        arms=list(_ARMS),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — empty; see the module docstring. The script's own machine walk is read as part
    # of the end-to-end WALK feature below, never as a claim about which states it passed
    # through.

    # STORE — the job's own terms, read existentially over whatever it created.
    cohort.assert_a_new_job_fires_every(_CADENCE_SECONDS)
    cohort.assert_a_new_job_notifies()
    cohort.assert_a_new_job_ends_when_asked(NO_END)

    # STORE — the two cycles, each read as what the created job(s) held at that position,
    # union over however many jobs the script created.
    cohort.claim(
        "state: after the first cycle the store holds the page's top headlines",
        _store_holds_the_top_headlines,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: after the second cycle the store holds the changed headlines",
        _store_holds_the_changed_headlines,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
