"""Per-edge conversation-state classifier contracts (#1706, beats 1–2): the
idle out-edges, every direction, from the cold-start shape through a populated
skill registry.

Each case sweeps a ten-phrasing pool deterministically (sample i →
``pool[i % 10]``), so at N=10 one run covers every phrasing exactly once and the
per-check cells map 1:1 to phrasings — the input-variation doctrine's first
native customer.  Snapshots are built PER SAMPLE by the production
``build_snapshot`` (embed + resolve_by_meaning pre-pass), so what varies between
the beat-1 and beat-2 cases is exactly what varies in production: whether the
registry holds skills.

**Beat 1 (empty registry — the cold-start shape)**: apply is structurally
withheld, the live union is elicit vs idle.  FIRE = request-shaped asks for
routines nothing covers; HOLD = ordinary conversation incl. the named boundary
case — a PASSING MENTION of a watchable thing (recurrence words describing the
USER's own habit, topic twins of fire phrasings).  Gated at 0.8 (two clean 1.00
baseline runs at N=10, turn-audited).

**Beat 2 (two seeded skills — a price-watch plus a distractor)**: the union
grows to three and the apply draw must ALSO bind WHICH skill (the SKILL: line,
membership-validated).  APPLY-FIRE = asks the price-watch skill covers (the
distractor measures wrong-skill selection); UNCOVERED = request-shaped asks
neither skill covers, which must still elicit with plausible candidates
dangling (the false-apply temptation — several are deliberate near-misses:
watching a NUMBER or page that isn't a price); HOLD-WITH-SKILLS = beat 1's hold
pool verbatim under candidates (does chat stay chat when apply is on offer?);
MIXED = chat preamble + a covered ask in one message (the named mixed-message
boundary → apply).

**Beat 3 (the parked machine — elicit's out-edges)**: the machine is parked in
elicit (anchor = the instigating ask, ``penny_last_turn`` = the teach question —
the parked-snapshot fields' first live use), and the user's reply resolves it:
STEPS = instructions telling the assistant HOW (→ learn, incl. schedule-worded
steps as realistic difficulty); CLARIFYING = questions back / partials without
the how (→ elicit); BAIL = call-offs and topic changes (→ idle, the break-out
edge).  Report-only until baselines are read.

Fictional-but-believable fixtures throughout (the repo is public).
"""

from __future__ import annotations

import pytest

from penny.conversation_machine import ConversationState
from penny.tests.eval.conftest import ClassifierEval, eval_skill

pytestmark = pytest.mark.eval

_FAMILY = "state-classifier"

# ── Beat 1: the cold-start shape (no skills seeded) ───────────────────────────

# Fire direction — a routine is being asked for and nothing covers it.
_FIRE_POOL = [
    "hey can you keep an eye on the harbor ferry timetable for me?",
    "can you watch the price on ridgelinefoxes.example/den-camera-kit?",
    "i want you to check the tide tables every morning and tell me if low tide is before 9",
    "could you track when the farmers market vendor list changes?",
    "keep tabs on the library's new-arrivals page for me",
    "watch harborseals.example/colony-count and let me know when the number moves",
    "start collecting the daily specials from the corner bakery's site, ok?",
    "monitor the trailhead conditions page — i want to know when the pass opens",
    "hey, track auction listings for vintage synths for me",
    "would you keep an eye out for when the ferry adds the late sailing?",
]

# Hold direction — ordinary conversation, incl. the passing-mention boundary
# (phrasings 3/4/6/9 mention watchable things or the user's OWN checking habit;
# 9 is a topic twin of fire phrasing 9).
_HOLD_POOL = [
    "morning! how's it going?",
    "what's the tallest mountain in the andes?",
    "the ferry was packed again this morning, could barely get a seat",
    "i've been checking the auction listings every day lately",
    "thanks, that was really helpful",
    "lol the bakery ran out of croissants before 8 again",
    "what time is it in lisbon right now?",
    "my sister might visit next weekend, thinking we'll hit the tidepools",
    "prices on vintage synths are getting ridiculous these days",
    "remind me what we talked about yesterday?",
]


async def test_idle_to_elicit_fires_on_uncovered_requests(
    classifier_eval: ClassifierEval,
) -> None:
    """Fire: a request-shaped ask for a routine no skill covers classifies
    elicit — the entry edge of the whole teach loop."""
    await classifier_eval(
        case_id="idle-elicit-fire",
        state=ConversationState.IDLE,
        pool=_FIRE_POOL,
        expected=ConversationState.ELICIT,
        min_pass_rate=0.8,
        family=_FAMILY,
    )


async def test_idle_holds_on_chat_and_passing_mentions(
    classifier_eval: ClassifierEval,
) -> None:
    """Hold: chat, questions, and passing mentions of watchable things classify
    idle — don't chase a mention into a teach loop."""
    await classifier_eval(
        case_id="idle-elicit-hold",
        state=ConversationState.IDLE,
        pool=_HOLD_POOL,
        expected=ConversationState.IDLE,
        min_pass_rate=0.8,
        family=_FAMILY,
    )


# ── Beat 2: a populated registry (price-watch + a distractor skill) ───────────

_PRICE_SKILL = "watch a listing price for changes"
_SEEDED_SKILLS = [
    eval_skill(
        _PRICE_SKILL,
        "keep an eye on a product or listing page and record its current price",
        {"url": "the product or listing page whose price to watch"},
    ),
    eval_skill(
        "collect daily cafe specials",
        "read a cafe or bakery menu page and save the day's specials each morning",
        {"url": "the cafe's menu page"},
    ),
]

# Apply direction — every ask is covered by the price-watch skill; the specials
# skill rides along as the in-context distractor (wrong-skill selection is a
# scored miss on the "named the covering skill" check).
_APPLY_POOL = [
    "can you watch the price on ridgelinefoxes.example/den-camera-kit?",
    "keep an eye on the price of the harbor kayak rental page for me",
    "track what the vintage synth on brasscat.example/listings/modular-iii is going for",
    "watch tidepool-optics.example/spotting-scope and tell me when the price moves",
    "can you keep tabs on the price of the ferry season pass?",
    "monitor the den camera kit listing — i want to know if it gets cheaper",
    "follow the price on that surfboard listing at driftline.example/boards/7-2",
    "hey, watch the campsite fee page and note the current rate",
    "could you track the price of the espresso grinder on beanhouse.example?",
    "keep watching what the old pinball machine is listed at",
]

# Uncovered direction — routine setups CLEARLY outside both seeded skills
# (no page-watching, no menu-reading: reminders, chat-extraction lists,
# summaries, tallies).  The contract case proves transition reasoning on
# unambiguous inputs; the genuinely fuzzy watch-shaped near-misses live in
# the report-only idle-coverage-boundary case below.
_UNCOVERED_POOL = [
    "every friday can you remind me to water the plants?",
    "keep a running list of every restaurant i mention to you",
    "every morning teach me one new portuguese word",
    "at the end of each week, summarize what we talked about",
    "keep track of how many times i go to the gym each week",
    "each evening save a one-line note about how my day went",
    "whenever i mention a book, add it to my reading list",
    "keep a tally by species of the birds i tell you i saw",
    "every sunday plan out three dinner ideas and save them for me",
    "log every movie i tell you i've watched",
]

# Cross-domain non-coverage — the STARK version of the non-match test (the
# code-owner ruling on runs 4-10: a watch-shaped request against a watch-shaped
# skill is legitimately COVERED — the model's "it fits" reading was correct, so
# the old watch-adjacent near-miss pool measured a non-distinction and is
# retired).  Here the seeded discovery skill and the requests share the same
# VERB shape (find/collect/watch for new X) in starkly different domains — a
# job-listings skill does not cover restaurants, houses, or concerts.
_CROSS_DOMAIN_SKILLS = [
    eval_skill(
        "find new job listings",
        "search the job boards for newly posted listings matching a role and save them",
        {"role": "the kind of job to look for", "boards": "the job boards to search"},
    ),
    _SEEDED_SKILLS[0],
]

_CROSS_DOMAIN_POOL = [
    "keep a list of new restaurants opening downtown",
    "find me new podcasts about gardening each week",
    "watch for new houses coming on the market in our neighborhood",
    "collect new science fiction releases at the library each month",
    "keep track of new hiking trails the parks department opens",
    "keep a list of new murals going up around the city",
    "watch for new classes at the community center",
    "collect newly announced concerts happening near us",
    "keep an eye out for new vendors joining the farmers market",
    "find new coffee roasters popping up in town",
]

# Mixed-message boundary — chat preamble + a covered ask in ONE message: the
# routine half wins (apply, with the skill bound); the chat half never
# suppresses it.  Every ask names the price EXPLICITLY (a contract case tests
# the transition on clear inputs; paraphrase tolerance is the advisory
# boundary case's business).
_MIXED_POOL = [
    "morning! oh and can you watch the price on the den camera kit listing?",
    "bakery ran out of croissants again lol — anyway, watch the espresso grinder price for me?",
    "that hike was gorgeous. also, track the spotting scope's price for me?",
    "thanks, super helpful! one more thing — watch the kayak rental price?",
    "my sister's visiting next weekend. btw can you keep tabs on the surfboard listing price?",
    "what a day. anyway — monitor the modular synth listing price for me, ok?",
    "the ferry was packed this morning. oh — watch the season pass price too?",
    "haha fair enough. hey, can you track the pinball machine's asking price?",
    "good morning! quick one: keep watching the den camera kit price?",
    "nice, that worked. also can you watch the price on the campsite booking page?",
]


async def test_idle_to_apply_fires_and_binds_the_covering_skill(
    classifier_eval: ClassifierEval,
) -> None:
    """Apply fire: an ask a seeded skill covers classifies apply AND binds that
    skill by name — with a distractor skill in the candidate list."""
    await classifier_eval(
        case_id="idle-apply-fire",
        state=ConversationState.IDLE,
        pool=_APPLY_POOL,
        expected=ConversationState.APPLY,
        expected_skill=_PRICE_SKILL,
        seed_skills=_SEEDED_SKILLS,
        min_pass_rate=0.8,
        family=_FAMILY,
    )


async def test_idle_still_elicits_when_no_candidate_covers(
    classifier_eval: ClassifierEval,
) -> None:
    """The false-apply guard: request-shaped asks neither skill covers must
    still elicit — plausible candidates dangling in context are not coverage."""
    await classifier_eval(
        case_id="idle-elicit-uncovered",
        state=ConversationState.IDLE,
        pool=_UNCOVERED_POOL,
        expected=ConversationState.ELICIT,
        seed_skills=_SEEDED_SKILLS,
        min_pass_rate=0.8,
        family=_FAMILY,
    )


async def test_same_verb_different_domain_still_elicits(
    classifier_eval: ClassifierEval,
) -> None:
    """The stark non-coverage contract: a discovery skill in one domain does
    not cover discovery requests in another — same verb shape, different
    world."""
    await classifier_eval(
        case_id="idle-elicit-cross-domain",
        state=ConversationState.IDLE,
        pool=_CROSS_DOMAIN_POOL,
        expected=ConversationState.ELICIT,
        seed_skills=_CROSS_DOMAIN_SKILLS,
        min_pass_rate=None,
        family=_FAMILY,
    )


async def test_idle_holds_on_chat_with_candidates_dangling(
    classifier_eval: ClassifierEval,
) -> None:
    """Beat 1's hold pool verbatim, now with candidates rendered: chat stays
    chat even when apply is on offer (incl. the price-adjacent topic twin)."""
    await classifier_eval(
        case_id="idle-hold-with-skills",
        state=ConversationState.IDLE,
        pool=_HOLD_POOL,
        expected=ConversationState.IDLE,
        seed_skills=_SEEDED_SKILLS,
        min_pass_rate=0.8,
        family=_FAMILY,
    )


async def test_mixed_chat_plus_covered_ask_applies(
    classifier_eval: ClassifierEval,
) -> None:
    """The mixed-message boundary: a chat preamble plus a covered ask in one
    message classifies apply with the skill bound — the routine half wins."""
    await classifier_eval(
        case_id="idle-apply-mixed",
        state=ConversationState.IDLE,
        pool=_MIXED_POOL,
        expected=ConversationState.APPLY,
        expected_skill=_PRICE_SKILL,
        seed_skills=_SEEDED_SKILLS,
        min_pass_rate=0.8,
        family=_FAMILY,
    )


# ── Beat 3: the parked machine — elicit's out-edges ───────────────────────────

# The parked-elicit context: the instigating ask (beat 1's first fire phrasing —
# continuity) and the teach question the reply answers.  Replies are only
# classifiable against what they answer, so both parked-snapshot fields render.
_FERRY_ASK = "hey can you keep an eye on the harbor ferry timetable for me?"
_TEACH_QUESTION = (
    "I don't know how to do that yet — can you teach me? "
    "What should I read, look for, and remember?"
)

# Steps direction — the reply tells the assistant HOW: what to read, what to
# look for, what to remember (2/9 carry schedule words — realistic difficulty,
# not a separate case).
_STEPS_POOL = [
    "sure — read harborferries.example/timetable and remember the first morning departure",
    "here's what to do: open harborferries.example/timetable and save the harbor loop times",
    "check harborferries.example/timetable every morning and note any changes to the schedule",
    "it's easy: read harborferries.example/timetable and remember the last sailing of the day",
    "go to the ferry timetable, find the weekend sailings, and remember those",
    "read the timetable at harborferries.example and save the departure times",
    "ok: fetch the timetable page, pull out the morning departures, and store them",
    "look at harborferries.example/timetable and remember whatever sailings are listed",
    "each day, read the timetable and write down the first and last departure",
    "grab the times off the timetable page and keep them somewhere you can check later",
]

# Clarifying direction — still on-task, but the teach question is NOT answered:
# questions back, hedges, partials without the how.
_CLARIFYING_POOL = [
    "what do you mean teach you? like explain the steps?",
    "hmm, what kinds of things can you actually read?",
    "what would you need from me to do that?",
    "i'm not sure — what do you usually watch for people?",
    "does it matter which page i give you?",
    "hang on, let me find the right link first",
    "can you even open websites on your own?",
    "what format do you want the steps in?",
    "wait, would this be every day or just once?",
    "let me think about what exactly i want you to look for",
]

# Bail direction — the break-out edge: call-offs and topic changes.
_BAIL_POOL = [
    "actually never mind, don't worry about it",
    "forget it — what's the weather looking like tomorrow?",
    "eh, it's not that important. anyway how was your night?",
    "no no, not that. let's drop it",
    "let's skip it for now",
    "on second thought i'll just check it myself",
    "changing topics — did anything interesting happen in the news today?",
    "scratch that. can you tell me a joke instead?",
    "nah, leave it. what time is it in lisbon right now?",
    "actually let's not — i'd rather talk about dinner plans",
]


async def test_parked_elicit_steps_arrive(classifier_eval: ClassifierEval) -> None:
    """Steps arrived: a reply that tells the assistant how — what to read, look
    for, remember — classifies learn (the demo round begins)."""
    await classifier_eval(
        case_id="elicit-learn-steps",
        state=ConversationState.ELICIT,
        pool=_STEPS_POOL,
        expected=ConversationState.LEARN,
        penny_last_turn=_TEACH_QUESTION,
        task_anchor=_FERRY_ASK,
        min_pass_rate=None,
        family=_FAMILY,
    )


async def test_parked_elicit_still_clarifying(classifier_eval: ClassifierEval) -> None:
    """Still clarifying: a question back or a partial without the how leaves the
    machine parked in elicit — the teach question is not answered yet."""
    await classifier_eval(
        case_id="elicit-still-clarifying",
        state=ConversationState.ELICIT,
        pool=_CLARIFYING_POOL,
        expected=ConversationState.ELICIT,
        penny_last_turn=_TEACH_QUESTION,
        task_anchor=_FERRY_ASK,
        min_pass_rate=None,
        family=_FAMILY,
    )


async def test_parked_elicit_bails_out(classifier_eval: ClassifierEval) -> None:
    """The break-out edge: a call-off or topic change routes back to idle — the
    parked teach loop never traps the conversation."""
    await classifier_eval(
        case_id="elicit-bail",
        state=ConversationState.ELICIT,
        pool=_BAIL_POOL,
        expected=ConversationState.IDLE,
        penny_last_turn=_TEACH_QUESTION,
        task_anchor=_FERRY_ASK,
        min_pass_rate=None,
        family=_FAMILY,
    )
