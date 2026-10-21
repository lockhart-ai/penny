"""The collection write's entry-identity ladder, in isolation: the DECISION step.

When a write lands on a collection that already holds entries, the ladder decides what
becomes of it: the deterministic rules settle every pair they are sure of, an embedding
shortlist ranks the few stored entries a candidate still needs to be asked about, and the
decision model answers — identity first (the same specific thing, or two items side by
side), value only on yes (does the stored entry already say everything the new one says?).
A wrong decision is visible in the store: two rows for one thing, or one row that lost a
value it should have kept. That is what these cases measure.

**Every pair here is one the deterministic rules do NOT settle** — verified structurally by
``test_each_identity_pair_reaches_the_decision_model`` (in this directory, run by ``make
check``): the rules-settled territory is unit-tested in ``tests/``; what these cases measure
is exactly what reaches the decision model. The ladder's own questions, its degraded mode
and its outcome rules are likewise mechanism, pinned by the mock suite — what a live run
adds is how the decision model actually answers.

**The arm is the PAIR**, for the reason the collector's arm is the page: the ladder takes no
user words, so there is no ask to reword — what varies across the arms is which look-alike
SHAPE the pair is (an abbreviation, a middle initial, a paraphrase, a recipe gaining a step,
a price moving; a substring collision, a prefix collision, two books of one series, two
teams of one city, two models of one line). The facts are fixed within an arm — the stored
entry, the filler and the new entry are byte-identical across that arm's samples — which is
what lets a claim name the value the store must carry.

**Claims are store-state only.** The ladder moves no machine and sends nothing; what a
wrong decision costs is visible in the collection's rows, so the claims read the collection's
entries after the write, whole (key and content), through each entry's smallest unique datum
— the anchor the coherence probe holds against the arm's declared entries. Tool calls are
not asserted (the write is the driver's own single call, not a model route), and the reply
category does not exist here: no chat turn is driven.

Two cases, because the two directions have different expected end states:

* ``identity-merges-true-variants`` — the new entry names the SAME specific thing a stored
  entry already holds. The collection ends with ONE entry for the pair (the stored one
  unchanged when it already says everything, rewritten when the new entry adds something)
  plus the untouched filler. The unsure outcome rides a write-as-new, so an unsure ladder
  fails these claims the honest way: three rows where one belongs.
* ``identity-keeps-look-alikes-apart`` — the new entry is a look-alike of the stored entry
  but a different specific thing. The collection ends with BOTH entries plus the filler,
  each carrying its own value. A wrong merge fails these claims the honest way: two rows
  where three belong.

Both are report-only; the thresholds are the code owner's.
"""

from __future__ import annotations

import pytest

from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.identity import ENTRY_COUNT, IdentityArm, IdentityEval
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import SpecCategory

pytestmark = pytest.mark.eval

_FAMILY = "entry-identity"

_MERGE_BEHAVIOUR = (
    "In the collection write's entry-identity ladder, when the new entry names the same "
    "specific thing a stored entry already holds, Penny ends with one entry for it, "
    "carrying the surviving value."
)

_APART_BEHAVIOUR = (
    "In the collection write's entry-identity ladder, when the new entry is a look-alike "
    "of a stored entry but a different specific thing, Penny ends with both entries — the "
    "stored one carrying its own value and the new one written."
)


def _carries(entry, anchor: str) -> bool:
    """Whether one stored entry's WHOLE row (key and content) carries the anchor."""
    return anchor.lower() in f"{entry.key} {entry.content}".lower()


def _held_keys(sample) -> list[str]:
    return [entry.key or "" for entry in sample.held]


# ── Case 1: a true variant merges into the entry that holds it ────────────────
#
# Five pairs the rules leave to the decision model, each the SAME thing under another
# name or wording. The first three arms expect the stored entry UNCHANGED (it already
# says everything the new entry says — the abbreviation, the dropped middle initial and
# the paraphrase add nothing), so the surviving value is the STORED entry's; the last two
# expect a MERGE (the recipe gains a step, the price moves), so the surviving value is
# the NEW entry's. ``survivor_anchor`` says which, per arm, and the claim reads it.

_MERGE_ARMS = (
    IdentityArm(
        text="abbreviation",
        collection="music-likes",
        description="Bands and artists the user likes",
        stored=(
            "godspeed you! black emperor",
            "Godspeed You! Black Emperor, the Montreal post-rock collective",
        ),
        filler=("portishead", "Portishead, the Bristol trip-hop band"),
        new=("gy!be", "GY!BE"),
        stored_anchor="montreal post-rock",
        filler_anchor="bristol trip-hop",
        new_anchor="gy!be",
        survivor="stored",
        survivor_anchor="montreal post-rock",
    ),
    IdentityArm(
        text="name variant",
        collection="authors",
        description="Writers the user has asked about",
        stored=("philip k. dick", "Philip K. Dick, the author of Ubik"),
        filler=("ursula k. le guin", "Ursula K. Le Guin, the author of The Dispossessed"),
        new=("philip dick", "Philip Dick, the science-fiction novelist"),
        stored_anchor="author of ubik",
        filler_anchor="the dispossessed",
        new_anchor="science-fiction novelist",
        survivor="stored",
        survivor_anchor="author of ubik",
    ),
    IdentityArm(
        text="paraphrase",
        collection="outdoors",
        description="What the user does outdoors",
        stored=("sea kayaking", "Likes sea kayaking"),
        filler=("rock climbing", "Goes rock climbing on weekends"),
        new=("kayaking on the ocean", "Enjoys kayaking on the ocean"),
        stored_anchor="sea kayaking",
        filler_anchor="rock climbing",
        new_anchor="kayaking on the ocean",
        survivor="stored",
        survivor_anchor="sea kayaking",
    ),
    IdentityArm(
        text="recipe step",
        collection="recipes",
        description="The user's recipes",
        stored=(
            "fajitas",
            "Chicken fajitas: peppers, onions, chili powder, tortillas; serves four",
        ),
        filler=("pancakes", "Weekend pancakes: flour, eggs, milk, butter"),
        new=(
            "chicken fajitas",
            "Chicken fajitas: peppers, onions, chili powder, tortillas, "
            "plus a 10-minute lime marinade",
        ),
        stored_anchor="serves four",
        filler_anchor="flour, eggs, milk",
        new_anchor="lime marinade",
        survivor="new",
        survivor_anchor="lime marinade",
    ),
    IdentityArm(
        text="price update",
        collection="typewriter-watch",
        description="Asking prices of typewriters on local listings",
        stored=("olympia sm3", "Olympia SM3, asking $180 on local listings"),
        filler=("hermes rocket", "Hermes Rocket, asking $95 on local listings"),
        new=("olympia sm3 typewriter", "Olympia SM3, asking $150 on local listings"),
        stored_anchor="$180",
        filler_anchor="$95",
        new_anchor="$150",
        survivor="new",
        survivor_anchor="$150",
    ),
)


def _holds_survivor_and_filler(sample, world) -> tuple[bool, str | None]:
    """The merge case's end state: exactly TWO entries — the pair's single survivor,
    carrying the value that survives, and the untouched filler beside it."""
    arm = _MERGE_ARMS[sample.arm]
    if len(sample.held) != 2:
        return False, f"the collection ended with {_held_keys(sample)}, expected two entries"
    survivor_key = next(
        (entry.key for entry in sample.held if _carries(entry, arm.survivor_anchor)), None
    )
    filler_key = next(
        (entry.key for entry in sample.held if _carries(entry, arm.filler_anchor)), None
    )
    if survivor_key is None or filler_key is None:
        return False, (
            f"the collection ended with {_held_keys(sample)}; nothing carries the surviving "
            f"value ({arm.survivor_anchor!r}) or the filler ({arm.filler_anchor!r})"
        )
    return True, None


# ── Case 2: look-alikes stay apart ───────────────────────────────────────────
#
# Five pairs the rules leave to the decision model, each TWO things that read alike —
# a substring of the stored name, a shared prefix, two books of one series, two teams
# of one city, two models of one product line. The collection ends with all three
# entries, each carrying its own value.

_APART_ARMS = (
    IdentityArm(
        text="substring collision",
        collection="music-likes",
        description="Bands and artists the user likes",
        stored=("pink floyd", "Pink Floyd, the progressive rock band"),
        filler=("genesis", "Genesis, Peter Gabriel's old band"),
        new=("pink", "P!nk, the pop singer"),
        stored_anchor="progressive rock",
        filler_anchor="peter gabriel",
        new_anchor="pop singer",
        survivor="new",
        survivor_anchor="pop singer",
    ),
    IdentityArm(
        text="prefix collision",
        collection="listening",
        description="Music the user listens to",
        stored=("portishead", "Portishead, the Bristol trip-hop band"),
        filler=("massive attack", "Massive Attack, the Bristol trip-hop duo"),
        new=("portugal. the man", "Portugal. The Man, the rock band from Alaska"),
        stored_anchor="trip-hop band",
        filler_anchor="trip-hop duo",
        new_anchor="from alaska",
        survivor="new",
        survivor_anchor="from alaska",
    ),
    IdentityArm(
        text="same series",
        collection="books",
        description="Books the user has read",
        stored=("dune", "Dune, the Frank Herbert novel"),
        filler=("neuromancer", "Neuromancer, the William Gibson novel"),
        new=("dune messiah", "Dune Messiah, the Frank Herbert sequel"),
        stored_anchor="the frank herbert novel",
        filler_anchor="william gibson",
        new_anchor="dune messiah",
        survivor="new",
        survivor_anchor="dune messiah",
    ),
    IdentityArm(
        text="same-city teams",
        collection="teams-followed",
        description="Teams the user follows",
        stored=("boston celtics", "Follows the Boston Celtics, the NBA team"),
        filler=("lakers", "Follows the Los Angeles Lakers, the Western Conference team"),
        new=("boston red sox", "Follows the Boston Red Sox, the MLB team"),
        stored_anchor="nba",
        filler_anchor="los angeles",
        new_anchor="mlb",
        survivor="new",
        survivor_anchor="mlb",
    ),
    IdentityArm(
        text="two models",
        collection="handhelds",
        description="Game handhelds the user has an eye on",
        stored=("steam deck oled", "Valve Steam Deck OLED, the 512GB model"),
        filler=("rog ally", "The ASUS ROG Ally handheld"),
        new=("steam deck lcd", "Valve Steam Deck LCD, the 256GB model"),
        stored_anchor="oled",
        filler_anchor="asus",
        new_anchor="256gb",
        survivor="new",
        survivor_anchor="256gb",
    ),
)


def _holds_stored_filler_and_new(sample, world) -> tuple[bool, str | None]:
    """The keep-apart case's end state: exactly THREE entries, each of the pair plus the
    filler still carrying its own value."""
    arm = _APART_ARMS[sample.arm]
    if len(sample.held) != 3:
        return False, f"the collection ended with {_held_keys(sample)}, expected three entries"
    missing = [
        name
        for name, anchor in (
            ("stored", arm.stored_anchor),
            ("filler", arm.filler_anchor),
            ("new", arm.new_anchor),
        )
        if not any(_carries(entry, anchor) for entry in sample.held)
    ]
    if missing:
        return False, (
            f"the collection ended with {_held_keys(sample)}; no entry carries the "
            f"{', '.join(missing)} value(s)"
        )
    return True, None


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_true_variant_merges_into_the_entry_that_holds_it(
    identity_eval: IdentityEval, model: str
) -> None:
    """The new entry names what the stored entry already holds; one entry ends up there.

    **The PROVENANCE category is empty by report, not by omission:** no model-authored
    value enters the store on this path — the stored entry, the filler and the new entry
    are the case's own declared inputs, and the only model in the loop (the decision
    model) writes nothing — so there is no value a provenance check could fail."""
    cohort = await identity_eval(
        case_id="identity-merges-true-variants",
        behaviour=_MERGE_BEHAVIOUR,
        area=Area.MEMORY,
        model=model,
        arms=_MERGE_ARMS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE — the collection's rows after the write, read whole
    cohort.claim(
        "state: the collection holds exactly the pair's one entry and the untouched filler",
        _holds_survivor_and_filler,
        SpecCategory.STORE,
    )
    cohort.measure(ENTRY_COUNT)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_look_alikes_stay_apart(identity_eval: IdentityEval, model: str) -> None:
    """The new entry is a look-alike, not the same thing; both entries end up there.

    **The PROVENANCE category is empty by report, not by omission:** no model-authored
    value enters the store on this path — the stored entry, the filler and the new entry
    are the case's own declared inputs, and the only model in the loop (the decision
    model) writes nothing — so there is no value a provenance check could fail."""
    cohort = await identity_eval(
        case_id="identity-keeps-look-alikes-apart",
        behaviour=_APART_BEHAVIOUR,
        area=Area.MEMORY,
        model=model,
        arms=_APART_ARMS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE — the collection's rows after the write, read whole
    cohort.claim(
        "state: the stored entry, the filler and the new entry all remain, "
        "each carrying its own value",
        _holds_stored_filler_and_new,
        SpecCategory.STORE,
    )
    cohort.measure(ENTRY_COUNT)
