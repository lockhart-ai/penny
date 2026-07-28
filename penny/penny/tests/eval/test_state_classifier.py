"""Per-edge conversation-state classifier contracts (#1706, beats 1–5): the
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
withheld, the live union is learn vs elicit vs idle.  FIRE = request-shaped asks
for routines nothing covers (no steps in the message → elicit); HOLD = ordinary
conversation incl. the named boundary case — a PASSING MENTION of a watchable
thing (recurrence words describing the USER's own habit, topic twins of fire
phrasings).  UNPROMPTED TEACHING = the same intents but WITH the steps in the
message → learn directly, skipping the teach question; fire is its paired guard
(same asks, no steps).  Gated at 0.8 (two clean 1.00 baseline runs at N=10,
turn-audited).

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
edge).

**Beat 4 (parked learn — re-entry after a failed demo round)**: the machine is
parked in learn (the demo round did not complete; ``penny_last_turn`` = the
honest failure report) with NOTHING in the registry — a failed round taught no
skill — so the skill-gated apply edge is structurally withheld and the union is
binary: a corrected set of instructions (→ learn), or a bail (→ idle).  There is
no path back to elicit (code-owner ruling: elicit exists to GET instructions,
and they have been given).  RETRY = corrections that CARRY the new instructions;
QUESTIONS = post-failure questions and doubts, no instructions carried; BAIL =
give-ups, topic changes, and withdrawals of the instructions.

**Beat 5 (the parked machine — request's out-edges)**: idle grows a fifth
edge, and apply splits with it.  A covered ask whose message CARRIES what the
skill needs applies; a covered ask MISSING it goes to request instead (the
paired split — same intents, the url present or absent is the only difference),
where the assistant names the skill and asks.  Parked in request
(``penny_last_turn`` = that question, anchor = the instigating ask), the reply
resolves it: DETAILS ARRIVE = the page, or a confirmation (→ apply, skill still
bound); WRONG SKILL = the proposal rejected but the task still wanted (→ elicit);
BAIL = call-offs (→ idle).  Both apply and request are SKILL-GATED, so each also
binds its skill by name.

**Beat 6 (parked learn — the round RAN and the offer is on the table)**: the
same parked state as beat 4 with the two things a COMPLETED round changes — the
taught skill is in the registry (so apply is offered) and ``penny_last_turn`` is
the report + the offer to set it running.  ACCEPT = taking that offer up, in the
shapes an acceptance arrives in (a bare yes, a cadence, an end condition, a
notify ask) → apply, binding the just-taught skill; none of them restates the
page, which is the whole reason the edge exists rather than a request that asks
for what was just read.  Its paired guard is CORRECTIONS under that same offer —
a correction must still be a correction when apply is on the table — and it
carries its own pool rather than re-running beat 4's, because beat 4's answers a
FAILED round and those phrasings contradict a last turn that reports what it
saved.  A pool has to cohere with the turn it answers or the case measures
contradiction-handling instead of the boundary it names.

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


# Unprompted teaching — the user volunteers the routine WITH its steps, so the
# machine goes straight to learn from idle, skipping the teach question (the
# fire pool above is the paired guard: same intents, no steps → elicit).
_UNPROMPTED_TEACH_POOL = [
    "hey lemme teach you how to check the ferry: open harborferries.example/timetable "
    "and remember the first morning departure",
    "here's how i want you to track the tides — read the tide table page and note when "
    "low tide is before 9",
    "i'll show you how this works: go to the library's new-arrivals page and save any "
    "new mystery titles",
    "let me teach you my routine — open the bakery's site, find the daily special, write it down",
    "this is how you do it: read the trailhead conditions page and remember whether the "
    "pass is open",
    "want to learn how i do this? open the vendor list page and note which stalls are new",
    "ok teaching time — read harborseals.example/colony-count and save the number",
    "here's the routine: check the ferry site, find the late sailing, remember if it's listed",
    "i'll walk you through it: open the community pool page and note the summer hours",
    "let me show you — read the birding club's sightings board and save any new species",
]


async def test_idle_to_learn_on_unprompted_teaching(
    classifier_eval: ClassifierEval,
) -> None:
    """Teaching can arrive UNPROMPTED: a message carrying the routine AND its
    steps goes straight to learn from idle, skipping the teach question."""
    await classifier_eval(
        case_id="idle-learn-unprompted",
        state=ConversationState.IDLE,
        pool=_UNPROMPTED_TEACH_POOL,
        expected=ConversationState.LEARN,
        min_pass_rate=None,
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
# Apply direction — covered AND fully bound: every ask names the page the
# price-watch skill needs, so the machine can enact immediately.  Asks that
# leave the url out are the request pool below (the paired split).
_APPLY_POOL = [
    "can you watch the price on ridgelinefoxes.example/den-camera-kit?",
    "keep an eye on the price at harborkayak.example/rentals/sea-touring",
    "track what the vintage synth on brasscat.example/listings/modular-iii is going for",
    "watch tidepool-optics.example/spotting-scope and tell me when the price moves",
    "keep tabs on the ferry season pass price at harborferries.example/passes",
    "monitor ridgelinefoxes.example/den-camera-kit — i want to know if it gets cheaper",
    "follow the price on that surfboard listing at driftline.example/boards/7-2",
    "watch campsite fees at pinehollow.example/rates and note the current one",
    "could you track the espresso grinder price at beanhouse.example/grinders/ek43?",
    "keep watching what brasscat.example/listings/pinball-1979 is listed at",
]

# Request-details direction — the skill plainly fits, but the page it needs is
# NOT in the message: the machine names the skill and asks, never refuses and
# never guesses a url.
_MISSING_INPUT_POOL = [
    "keep an eye on the price of the harbor kayak rental page for me",
    "can you keep tabs on the price of the ferry season pass?",
    "monitor the den camera kit listing — i want to know if it gets cheaper",
    "hey, watch the campsite fee page and note the current rate",
    "could you track the price of the espresso grinder at that roaster?",
    "keep watching what the old pinball machine is listed at",
    "watch the price on that spotting scope i was looking at",
    "track the surfboard listing price for me",
    "can you follow the vintage synth's price?",
    "keep an eye on what the sea-touring kayak goes for",
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
    "every week, check which new vendors joined the farmers market",
    "keep a running list of new coffee roasters opening in town",
]

# Mixed-message boundary — chat preamble + a covered ask in ONE message: the
# routine half wins (apply, with the skill bound); the chat half never
# suppresses it.  Every ask names the price EXPLICITLY and carries the page the
# skill needs, so the ONE variable under test is the chat preamble — boundness
# is the fire/request split's business, not this case's.
_MIXED_POOL = [
    "morning! oh and can you watch the price on ridgelinefoxes.example/den-camera-kit?",
    "croissants again, ugh — anyway, watch beanhouse.example/grinders/ek43's price?",
    "that hike was gorgeous. also, track tidepool-optics.example/spotting-scope's price?",
    "thanks! one more thing — watch harborkayak.example/rentals/sea-touring's price?",
    "sister visits next weekend. btw keep tabs on driftline.example/boards/7-2's price?",
    "what a day. anyway — monitor the price at brasscat.example/listings/modular-iii, ok?",
    "the ferry was packed today. oh — watch harborferries.example/passes' price too?",
    "haha fair enough. hey, track the price at brasscat.example/listings/pinball-1979?",
    "good morning! quick one: watch the price at ridgelinefoxes.example/den-camera-kit?",
    "nice, that worked. also can you watch the price on pinehollow.example/rates?",
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


async def test_idle_to_request_details_when_inputs_missing(
    classifier_eval: ClassifierEval,
) -> None:
    """Covered but unbound: the skill fits and the page it needs is absent, so
    the machine names the skill and asks — never refusing, never guessing."""
    await classifier_eval(
        case_id="idle-request",
        state=ConversationState.IDLE,
        pool=_MISSING_INPUT_POOL,
        expected=ConversationState.REQUEST,
        expected_skill=_PRICE_SKILL,
        seed_skills=_SEEDED_SKILLS,
        min_pass_rate=None,
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
        min_pass_rate=0.8,
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


# ── Beat 5: the parked machine — request's out-edges ─────────────────────────

# The parked context: the assistant named the skill and asked for what it needs.
_REQUEST_QUESTION = "Sounds like my watch-a-listing-price skill — which page should I watch?"
_KAYAK_ASK = "keep an eye on the price of the harbor kayak rental page for me"

# Details arrive (a url, or a plain confirmation) → apply, skill still bound.
_REQUEST_APPLY_POOL = [
    "harborkayak.example/rentals/sea-touring",
    "here you go: harborkayak.example/rentals/sea-touring",
    "yeah that's the one — harborkayak.example/rentals/sea-touring",
    "the page is harborkayak.example/rentals/sea-touring",
    "use harborkayak.example/rentals/sea-touring",
    "yes please, and the page is harborkayak.example/rentals",
    "that skill's right — watch harborkayak.example/rentals",
    "correct. the listing lives at harborkayak.example/rentals/sea-touring",
    "sure, go ahead — harborkayak.example/rentals/sea-touring",
    "yep. harborkayak.example/rentals/sea-touring is the page",
]

# Wrong skill, task still wanted → elicit (teach me the right routine).
_WRONG_SKILL_POOL = [
    "no, that's not what i meant — i want something different",
    "that skill isn't right for this, it's a different kind of thing",
    "not that one. what i want is something else entirely",
    "nope, wrong skill — this isn't about prices",
    "that's not it. i need something you don't know how to do yet",
    "no, i don't want the price watcher for this",
    "that skill doesn't fit what i'm after",
    "not quite — this is a different job than that one",
    "no, that's the wrong routine for what i need",
    "that isn't what i had in mind, i want another thing done",
]

# Bail from the parked question → idle.
_REQUEST_BAIL_POOL = [
    "actually never mind, forget the whole thing",
    "eh, drop it — what's the weather tomorrow?",
    "let's skip it for now",
    "you know what, don't bother",
    "changing topics — tell me a joke instead",
    "no worries, i'll just check it myself",
    "leave it. how was your night?",
    "not important, let's move on",
    "cancel that, i'd rather plan dinner",
    "forget it — did anything happen in the news today?",
]


async def test_parked_details_arrive_applies(classifier_eval: ClassifierEval) -> None:
    """Details arrive: a url (or a plain confirmation) moves the parked machine
    to apply with the same skill still bound — the negotiation completes."""
    await classifier_eval(
        case_id="request-apply",
        state=ConversationState.REQUEST,
        pool=_REQUEST_APPLY_POOL,
        expected=ConversationState.APPLY,
        expected_skill=_PRICE_SKILL,
        penny_last_turn=_REQUEST_QUESTION,
        task_anchor=_KAYAK_ASK,
        seed_skills=_SEEDED_SKILLS,
        min_pass_rate=None,
        family=_FAMILY,
    )


async def test_parked_details_wrong_skill_elicits(classifier_eval: ClassifierEval) -> None:
    """Wrong skill, task still wanted: the machine returns to elicit — the
    proposal was rejected, so the routine has to be taught."""
    await classifier_eval(
        case_id="request-elicit",
        state=ConversationState.REQUEST,
        pool=_WRONG_SKILL_POOL,
        expected=ConversationState.ELICIT,
        penny_last_turn=_REQUEST_QUESTION,
        task_anchor=_KAYAK_ASK,
        seed_skills=_SEEDED_SKILLS,
        min_pass_rate=None,
        family=_FAMILY,
    )


async def test_parked_details_bails(classifier_eval: ClassifierEval) -> None:
    """The break-out edge: a call-off or topic change drops the negotiation
    back to idle."""
    await classifier_eval(
        case_id="request-bail",
        state=ConversationState.REQUEST,
        pool=_REQUEST_BAIL_POOL,
        expected=ConversationState.IDLE,
        penny_last_turn=_REQUEST_QUESTION,
        task_anchor=_KAYAK_ASK,
        seed_skills=_SEEDED_SKILLS,
        min_pass_rate=None,
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
    "is this something you're able to do from your side?",
    "can you even open websites on your own?",
    "what format do you want the steps in?",
    "wait, would this be every day or just once?",
    "do you need the exact address or just the site name?",
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
        min_pass_rate=0.8,
        family=_FAMILY,
    )


async def test_parked_elicit_steps_arrive_with_skills_populated(
    classifier_eval: ClassifierEval,
) -> None:
    """The populated-registry stress: the SAME steps replies, but with the
    beat-2 skills seeded — the Known skills section renders watch-adjacent
    candidates while the user is mid-teaching something NEW.  Existing skills
    must not demote teaching to still-clarifying or a bail; the paired delta
    against elicit-learn-steps isolates the candidates' effect."""
    await classifier_eval(
        case_id="elicit-learn-steps-with-skills",
        state=ConversationState.ELICIT,
        pool=_STEPS_POOL,
        expected=ConversationState.LEARN,
        penny_last_turn=_TEACH_QUESTION,
        task_anchor=_FERRY_ASK,
        seed_skills=_SEEDED_SKILLS,
        min_pass_rate=0.8,
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
        min_pass_rate=0.8,
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
        min_pass_rate=0.8,
        family=_FAMILY,
    )


# ── Beat 4: parked learn — re-entry after a failed demo round ─────────────────

# The parked-learn context: the demo round failed (the honest failure report is
# the assistant's last turn), the machine holds in learn, and the reply decides
# retry vs re-explain vs bail.
_FAILED_ROUND_REPORT = (
    "I tried, but the timetable page wouldn't load, so I couldn't save anything. "
    "Should I try again, or is there a different page I should read?"
)

# Retry direction — corrections and try-agains actionable NOW (the correction
# IS new instruction: a fixed url, a different column, a narrower value).
_RETRY_POOL = [
    "try again — the page should load now",
    "no, read the SECOND table on the page, not the first one",
    "you saved the arrival time — i wanted the departure time, fix that",
    "use harborferries.example/timetable-v2 instead, the old link is dead",
    "almost — but remember the last sailing too, not just the first",
    "run it once more, i think the site was just down",
    "the times you grabbed are for weekdays — get the weekend ones",
    "same steps, but save them under 'ferry times' instead",
    "close! the departure column is the one on the left",
    "redo it and this time keep only the morning sailings",
]

# Bail direction — the break-out edge from a failed round: give-ups and topic
# changes; the task dies, the conversation moves on.
_LEARN_BAIL_POOL = [
    "you know what, forget it — this isn't working",
    "let's give up on this one. how's your evening been?",
    "eh, never mind the ferry thing. what's the weather tomorrow?",
    "drop it for now, i'll set it up some other time",
    "this is more trouble than it's worth, let's move on",
    "abandon that — tell me a joke instead",
    "let's shelve it. did anything happen in the news today?",
    "no worries, i'll just check the ferry site myself from now on",
    "scrap those steps — i'll write you better instructions in a minute",
    "forget those instructions, they were wrong. i'll send new ones shortly",
]


async def test_parked_learn_retries_on_corrections(classifier_eval: ClassifierEval) -> None:
    """Retry: a correction or try-again actionable now stays in learn — the
    correction-loop invariant (a failed round holds its state)."""
    await classifier_eval(
        case_id="learn-retry-corrections",
        state=ConversationState.LEARN,
        pool=_RETRY_POOL,
        expected=ConversationState.LEARN,
        penny_last_turn=_FAILED_ROUND_REPORT,
        task_anchor=_FERRY_ASK,
        min_pass_rate=0.8,
        family=_FAMILY,
    )


# Questions direction — post-failure questions and doubts about how: engaged,
# but carrying no instructions, so the machine falls to idle (learn is two-way;
# there is no path back to elicit once instructions have been given).
_POST_FAILURE_QUESTION_POOL = [
    "what went wrong exactly? which page did you open?",
    "hm, what can you actually read then?",
    "did the link work at all, or did nothing come up?",
    "wait, which url did you try?",
    "is there something about that page you can't handle?",
    "what do you mean it wouldn't load — did you get an error?",
    "so what part failed, the reading or the saving?",
    "would a different page work better for you?",
    "what kind of pages usually work?",
    "huh, it loads fine for me — what did you see?",
]


async def test_parked_learn_questions_fall_to_idle(classifier_eval: ClassifierEval) -> None:
    """Engaged questions still carry no instructions, so they fall to idle —
    the two-way learn state has no path back to elicit."""
    await classifier_eval(
        case_id="learn-questions-idle",
        state=ConversationState.LEARN,
        pool=_POST_FAILURE_QUESTION_POOL,
        expected=ConversationState.IDLE,
        penny_last_turn=_FAILED_ROUND_REPORT,
        task_anchor=_FERRY_ASK,
        min_pass_rate=0.8,
        family=_FAMILY,
    )


async def test_parked_learn_bails_out(classifier_eval: ClassifierEval) -> None:
    """The break-out edge from a failed round: a give-up, a topic change, or a
    withdrawal of the instructions routes to idle — parked learn is binary, so
    anything that is not a corrected set of instructions is a bail."""
    await classifier_eval(
        case_id="learn-bail",
        state=ConversationState.LEARN,
        pool=_LEARN_BAIL_POOL,
        expected=ConversationState.IDLE,
        penny_last_turn=_FAILED_ROUND_REPORT,
        task_anchor=_FERRY_ASK,
        min_pass_rate=0.8,
        family=_FAMILY,
    )


# ── Beat 6: parked learn — the round RAN, and the offer is on the table ───────

# The taught routine, as the run-end extractor would have left it: a generic
# name + description over the demonstration, alongside an unrelated distractor
# (wrong-skill selection is a scored miss on the bound-skill check).
_TIMETABLE_SKILL = "check a timetable page for a departure time"
_TAUGHT_SKILLS = [
    eval_skill(
        _TIMETABLE_SKILL,
        "read a schedule page and record the departure time it lists",
        {"url": "the timetable page to read"},
    ),
    _SEEDED_SKILLS[1],
]

# The parked-learn context beat 4 does NOT cover: the demo round SUCCEEDED, so
# the assistant's last turn reports what it did and offers to set it running —
# which is where the machine stands when the user answers that offer.
_TAUGHT_ROUND_REPORT = (
    "Read harborferries.example/timetable and saved the first morning departure "
    "(6:40am) to ferry-times. Want me to keep it up to date on its own?"
)

# Accept direction — the offer taken up, in the shapes an acceptance actually
# arrives in: a bare yes, a cadence, an end condition, a notify ask.  None of
# them restates the page: the round that just ran supplied it.
#
# Phrasings 4 and 9 stack ALL THREE terms in one message (how often · until when
# · tell me), which is what a real acceptance looks like and what the enactment
# case sends verbatim — the shape that exposed this edge's hardest failure, where
# the terms were read as steps of the routine and the message classified learn.
# A pool of one-term acceptances hid that; these two are the case for it.
_ACCEPT_POOL = [
    "yes please, set that up",
    "yeah do that every morning",
    "perfect — keep it running and tell me when it changes",
    "perfect — do that every hour until 10pm tonight and tell me if it changes",
    "sure, go ahead and make that automatic",
    "yep, run that daily from now on",
    "please do — and let me know if the time moves",
    "that's exactly it, keep doing that",
    "yes — check it every 30 minutes until midnight and message me if it moves",
    "yes, and ping me when the departure changes",
]


async def test_parked_learn_accepts_the_offer_and_applies(
    classifier_eval: ClassifierEval,
) -> None:
    """The edge the demonstrated round's own offer creates: accepting it moves
    the parked machine to apply with the just-taught skill bound.  Nothing in
    these messages names the page — the round that just ran is what supplies
    it, which is why this edge exists at all rather than routing through a
    request that asks for what was just read."""
    await classifier_eval(
        case_id="learn-apply-accept",
        state=ConversationState.LEARN,
        pool=_ACCEPT_POOL,
        expected=ConversationState.APPLY,
        expected_skill=_TIMETABLE_SKILL,
        penny_last_turn=_TAUGHT_ROUND_REPORT,
        task_anchor=_FERRY_ASK,
        seed_skills=_TAUGHT_SKILLS,
        min_pass_rate=None,
        family=_FAMILY,
    )


# Corrections to a round that RAN — this beat's own pool, not beat 4's.  Beat 4's
# retry pool answers a FAILED round ("try again — the page should load now", "run
# it once more, i think the site was just down"), so under this beat's last turn —
# which reports what it saved and offers to keep it up to date — those phrasings
# contradict the story they are answering, and the classifier is being asked to
# resolve a fixture that does not cohere rather than the boundary under test.
# (Measured: exactly those two phrasings missed at N=10; the eight carrying real
# corrections held.)  A correction to a SUCCESSFUL round fixes what it saved —
# the wrong column, the wrong day, too little, the wrong place to put it.
_POST_SUCCESS_CORRECTION_POOL = [
    "no, read the SECOND table on the page, not the first one",
    "you grabbed the departure — i actually wanted the arrival time, fix that",
    "almost — but remember the last sailing too, not just the first",
    "the times you grabbed are for weekdays, i want the weekend ones",
    "close! the departure column is the one on the left",
    "redo it and this time keep only the morning sailings",
    "save it under 'morning ferries' instead",
    "use harborferries.example/timetable-v2 — that's the page i actually meant",
    "6:40 is the weekday one, grab the sunday departure instead",
    "also note the ferry name next to the time, not just the time",
]


async def test_parked_learn_corrections_still_learn_with_the_skill_offered(
    classifier_eval: ClassifierEval,
) -> None:
    """The paired over-correction guard: the SAME parked state with apply now on
    offer must still route a correction to learn.  Beat 4 measured this with an
    empty registry, where apply was structurally withheld — so it could not have
    caught the failure this edge introduces (a correction read as acceptance,
    freezing the wrong routine into a schedule)."""
    await classifier_eval(
        case_id="learn-apply-corrections-guard",
        state=ConversationState.LEARN,
        pool=_POST_SUCCESS_CORRECTION_POOL,
        expected=ConversationState.LEARN,
        penny_last_turn=_TAUGHT_ROUND_REPORT,
        task_anchor=_FERRY_ASK,
        seed_skills=_TAUGHT_SKILLS,
        min_pass_rate=None,
        family=_FAMILY,
    )
