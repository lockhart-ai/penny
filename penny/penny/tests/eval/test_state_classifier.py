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
boundary → apply).  Report-only until baselines are read.

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

# Uncovered direction — request-shaped asks NEITHER seeded skill covers, so the
# classifier must still elicit with plausible candidates dangling in context
# (the false-apply temptation).  Several are deliberate near-misses: watching a
# NUMBER or a page that isn't a price (colony count, timetable, waitlist).
_UNCOVERED_POOL = [
    "hey can you keep an eye on the harbor ferry timetable for me?",
    "could you track when the farmers market vendor list changes?",
    "monitor the trailhead conditions page — i want to know when the pass opens",
    "keep tabs on the library's new-arrivals page for me",
    "watch harborseals.example/colony-count and let me know when the number moves",
    "i want you to check the tide tables every morning and tell me if low tide is before 9",
    "would you keep an eye out for when the ferry adds the late sailing?",
    "track when the community pool posts its summer schedule",
    "let me know when the birding club updates the sightings board",
    "keep an eye on the marina's slip waitlist page for me",
]

# Mixed-message boundary — chat preamble + a covered ask in ONE message: the
# routine half wins (apply, with the skill bound); the chat half never
# suppresses it.
_MIXED_POOL = [
    "morning! oh and can you watch the price on the den camera kit listing?",
    "bakery ran out of croissants again lol — anyway, watch the espresso grinder price for me?",
    "that hike was gorgeous. also, track what the spotting scope is going for?",
    "thanks, super helpful! one more thing — watch the kayak rental price?",
    "my sister's visiting next weekend. btw can you keep tabs on the surfboard listing price?",
    "what a day. anyway — monitor the modular synth listing price for me, ok?",
    "the ferry was packed this morning. oh — watch the season pass price too?",
    "haha fair enough. hey, can you track the pinball machine's asking price?",
    "good morning! quick one: keep watching the den camera kit price?",
    "nice, that worked. also can you follow the campsite fee page for rate changes?",
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
        min_pass_rate=None,
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
        min_pass_rate=None,
        family=_FAMILY,
    )
