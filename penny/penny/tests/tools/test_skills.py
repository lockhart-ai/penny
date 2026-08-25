"""Skill substrate tests (#1590) — the render, provenance inference, certified-by
-execution, and the seed library, driven through the tool entry points with
deterministic fixtures and fictional content only.
"""

from __future__ import annotations

import json

import pytest

from penny.database import Database
from penny.database.skills import (
    DERIVED_NAME_MAX_LENGTH,
    WRITE_TARGET_DESCRIPTION,
    DistillInput,
    SkillDraft,
    SkillParameter,
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    bind_parameters,
    build_binding_content,
    derive_collection_name,
    distill_steps,
    render_skill,
    render_spoken_turns,
    retarget_writes,
    unbound_required_parameters,
)
from penny.tools.skill_tools import SkillReadTool

# ── Fixtures: a fictional "watch the elevation of a peak" demonstration ────────
#
# Structural provenance (#1659): the browse query and the extract instruction are
# non-binding string leaves → HOLES; the extracted reading ("1,842 m") flows
# step 1 → step 2 as a BINDING.  The leaf holding "elevations" additionally carries
# the ATTACHMENT MARK (#1783) because that value names one of Penny's own
# collections — nothing here looks at which tool the leaf sits on.  All fictional.

_UTTERANCE = "Save the Zephyr Ridge elevation to my notes"
_EXTRACTED_VALUE = "1,842 m"

# The registry the demonstration ran against: the collections a routine could be
# ATTACHED to.  ``distill_steps`` compares demonstrated VALUES against this set — it
# never learns which tools exist, let alone which of them write.
_ATTACHMENT_NAMES = frozenset({"elevations", "knowledge", "prices", "fruits", "notes", "headlines"})

# Every real tool call the framework logs carries the universal ``reasoning``
# think-aloud (``Tool.to_ollama_tool`` injects it) — the model's per-call
# narration.  The fixtures inject it so a demonstration matches a REAL promptlog
# (the #1661 divergence: the old fixtures omitted it, so distill never had to
# strip it and a real run surfaced nonsense ``reasoning`` holes).  Distill drops
# the top-level ``reasoning`` outright — never a hole, never a stored/rendered arg.
_REASONING = "because the user asked me to save it"

_BROWSE_ARGS = {"queries": ["Zephyr Ridge elevation"], "extract": "the elevation above sea level"}
_WRITE_ARGS = {
    "memory": "elevations",
    "entries": [{"key": "Zephyr Ridge elevation", "content": _EXTRACTED_VALUE}],
}

_BROWSE_OK = (
    f"You used `browse` and here's the result: (browse result)\nEXTRACTED: {_EXTRACTED_VALUE}"
)
_WRITE_OK = (
    "You saved an entry to elevations: (collection_write result)\n"
    "Wrote 1 entry to 'elevations': Zephyr Ridge elevation."
)
_BROWSE_FAILED = (
    "You searched for 'Zephyr Ridge elevation' but couldn't read anything — no page was "
    "read, so there's nothing current to quote (browse result)\n"
    "## browse error: unreachable"
)


def _elevation_steps() -> list[SkillStep]:
    """The distilled steps for the fixture, built directly (independent of the
    inference path) so the render is pinned in isolation."""
    return [
        SkillStep(
            ordinal=1,
            source_ordinal=1,
            tool="browse",
            arguments=dict(_BROWSE_ARGS),
            substitutions=[
                SkillSubstitution(path=["queries", 0], kind=SkillSubKind.HOLE, parameter="queries"),
                SkillSubstitution(path=["extract"], kind=SkillSubKind.HOLE, parameter="extract"),
            ],
        ),
        SkillStep(
            ordinal=2,
            source_ordinal=2,
            tool="collection_write",
            arguments=json.loads(json.dumps(_WRITE_ARGS)),
            substitutions=[
                SkillSubstitution(
                    path=["memory"],
                    kind=SkillSubKind.PLACEHOLDER,
                    description=WRITE_TARGET_DESCRIPTION,
                    attachment=True,
                ),
                SkillSubstitution(
                    path=["entries", 0, "key"], kind=SkillSubKind.HOLE, parameter="queries"
                ),
                SkillSubstitution(
                    path=["entries", 0, "content"], kind=SkillSubKind.BINDING, step=1
                ),
            ],
        ),
    ]


# ── The render: with-holes and the money literal ──────────────────────────────

_WITH_HOLES = (
    "1. browse(queries=[{queries}], extract={extract})\n"
    "2. collection_write(memory={the collection this is set up on}, "
    "entries=[{'key': {queries}, 'content': the value from step 1}])"
)

# THE money literal — steps + bound params → the numbered TEXT recipe.  Holes
# substituted verbatim; the binding reads as a legible instruction; the ATTACHMENT-marked
# leaf stays a placeholder because only ATTACHING the routine to a collection decides
# it — ``params`` can never bind it, so an uninstantiated skill never names a
# collection.  The prompt a collection actually runs adds ``retarget_writes`` (below).
_MONEY_LITERAL = (
    "1. browse(queries=['Cinder Peak elevation'], extract='the elevation above sea level')\n"
    "2. collection_write(memory={the collection this is set up on}, "
    "entries=[{'key': 'Cinder Peak elevation', 'content': the value from step 1}])"
)


def test_render_skill_with_holes_is_the_template():
    """An unbound skill renders holes as ``{name}`` placeholders and the binding as
    a legible instruction — the with-holes recipe the read surface shows."""
    assert render_skill(_elevation_steps()) == _WITH_HOLES


def test_render_skill_bound_is_the_money_literal():
    """steps + bound params → the numbered text recipe: every hole substituted with the
    param value verbatim, the binding kept legible, and the attachment-marked leaf still
    the placeholder — a skill hardcodes nothing from its demonstration (#1777)."""
    rendered = render_skill(
        _elevation_steps(),
        {"queries": "Cinder Peak elevation", "extract": "the elevation above sea level"},
    )
    assert rendered == _MONEY_LITERAL
    assert "elevations" not in rendered  # the demonstrated collection is never named


# ── Provenance inference: binding / candidate / attachment mark in one run ─────


def test_distill_classifies_every_leaf_the_same_way_and_marks_by_value():
    """Structural provenance (#1659): a value that flowed from a prior result is a
    BINDING; every other string leaf is a REQUIRED candidate (shared values collapse to
    one).  #1783: EVERY leaf goes through that one process — the collection name is a
    candidate like any other and reaches the labeller — and it additionally carries the
    ATTACHMENT mark because its VALUE names one of Penny's collections, never because
    of the tool it sits on."""
    inputs = [
        DistillInput(source_ordinal=1, tool="browse", arguments=_BROWSE_ARGS, result=_BROWSE_OK),
        DistillInput(
            source_ordinal=2, tool="collection_write", arguments=_WRITE_ARGS, result=_WRITE_OK
        ),
    ]
    steps, parameters = distill_steps(inputs, _ATTACHMENT_NAMES)

    # Three candidates — the browse query, the extract instruction, and the collection
    # name; the write KEY reuses the query's candidate (same value → one shared).  The
    # collection name is NOT exempt from candidacy: the labeller adjudicates it, and
    # only then does it stop being a parameter (see test_skill_extraction.py).
    assert parameters == [
        SkillParameter(name="queries", required=True),
        SkillParameter(name="extract", required=True),
        SkillParameter(name="memory", required=True),
    ]
    # Every candidate is required, so an unbound instantiation refuses naming each one
    # (the refusal carries the whole SkillParameter — name + description); binding them
    # all clears the validation (#1591/#1659, no silent default).
    assert [p.name for p in unbound_required_parameters(parameters, {})] == [
        "queries",
        "extract",
        "memory",
    ]
    assert unbound_required_parameters(parameters, {"queries": "x", "extract": "y"}) == [
        SkillParameter(name="memory", required=True)
    ]

    # Step 1: the query and the extract instruction are both candidates, neither marked
    # (neither value names a collection).
    step1 = {tuple(s.path): s for s in steps[0].substitutions}
    assert step1[("queries", 0)].kind == SkillSubKind.HOLE
    assert step1[("queries", 0)].parameter == "queries"
    assert step1[("queries", 0)].attachment is False
    assert step1[("extract",)].kind == SkillSubKind.HOLE
    assert step1[("extract",)].attachment is False

    # Step 2: the key is the SHARED 'queries' candidate; the content is a BINDING to
    # step 1's result; the collection name is a candidate carrying the mark.
    step2 = {tuple(s.path): s for s in steps[1].substitutions}
    assert step2[("entries", 0, "key")].kind == SkillSubKind.HOLE
    assert step2[("entries", 0, "key")].parameter == "queries"
    assert step2[("entries", 0, "key")].attachment is False
    assert step2[("entries", 0, "content")].kind == SkillSubKind.BINDING
    assert step2[("entries", 0, "content")].step == 1
    assert step2[("memory",)].kind == SkillSubKind.HOLE
    assert step2[("memory",)].parameter == "memory"
    assert step2[("memory",)].attachment is True
    # The demonstrated collection survives as the verbatim ledger copy (provenance),
    # and never as a rendered literal — pre-labeller it is a {memory} hole, and the
    # instantiation seam binds it to the collection the routine is attached to.
    assert steps[1].arguments["memory"] == "elevations"
    assert "elevations" not in render_skill(steps)
    assert "collection_write(memory={memory}" in render_skill(steps)
    assert "collection_write(memory='peak-notes'" in render_skill(
        retarget_writes(steps, "peak-notes")
    )


def test_distill_marks_by_value_across_arbitrary_tools():
    """#1783's load-bearing case, and the one the old rule could not express: a routine
    whose sequence is a ``log_append`` (a write tool that was never in the whitelist),
    a plugin call (a tool nothing has heard of), and a read of the routine's own
    collection.  The mark follows the VALUE — the collection name wherever it appears,
    including on the read — and every other leaf, log name and calendar id alike, is an
    ordinary candidate the labeller adjudicates."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="collection_read_latest",
            arguments={"memory": "notes", "limit": 5},
            result="You read notes: (collection_read_latest result)\n1 entry",
        ),
        DistillInput(
            source_ordinal=2,
            tool="create_event",
            arguments={"calendar_id": "cal-77", "title": "Ridge survey"},
            result="You created an event: (create_event result)\nCreated.",
        ),
        DistillInput(
            source_ordinal=3,
            tool="log_append",
            arguments={"memory": "survey-log", "content": "Ridge survey booked"},
            result="You appended to survey-log: (log_append result)\nAppended.",
        ),
        DistillInput(
            source_ordinal=4,
            tool="collection_write",
            arguments={"memory": "notes", "entries": [{"key": "k", "content": "c"}]},
            result=_WRITE_OK,
        ),
    ]
    steps, parameters = distill_steps(inputs, _ATTACHMENT_NAMES)
    marked = {
        (step.tool, tuple(sub.path))
        for step in steps
        for sub in step.substitutions
        if sub.attachment
    }
    # 'notes' is a collection, so BOTH the read and the write carry the mark; the log
    # name and the calendar id are values the mark knows nothing about.
    assert marked == {("collection_read_latest", ("memory",)), ("collection_write", ("memory",))}
    assert {p.name for p in parameters} >= {"calendar_id", "title", "memory"}
    # Applying the routine to a collection binds every marked leaf and NOTHING else —
    # the log stays the log, the calendar id stays the calendar id.
    retargeted = retarget_writes(steps, "ridge-notes")
    assert retargeted[0].arguments["memory"] == "ridge-notes"
    assert retargeted[1].arguments == {"calendar_id": "cal-77", "title": "Ridge survey"}
    assert retargeted[2].arguments["memory"] == "survey-log"
    assert retargeted[3].arguments["memory"] == "ridge-notes"


def test_a_bound_leaf_is_never_marked_even_when_its_value_names_a_collection():
    """The mark only reaches an UNEXPLAINED leaf.  A value a prior step produced is
    already explained — it came from that step — so there is nothing for an attachment
    to decide, even when the value happens to match a collection's name.  Marking it
    would let the render seam overwrite a real binding with the collection's own name,
    which is a stale literal written into every cycle, forever."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="browse",
            arguments={"queries": ["which list holds the readings"]},
            result="You opened the index (browse result)\nelevations",
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={"memory": "notes", "entries": [{"key": "k", "content": "elevations"}]},
            result=_WRITE_OK,
        ),
    ]
    steps, _ = distill_steps(inputs, _ATTACHMENT_NAMES)
    subs = {tuple(s.path): s for s in steps[1].substitutions}
    bound = subs[("entries", 0, "content")]
    assert bound.kind == SkillSubKind.BINDING and bound.step == 1
    assert bound.attachment is False  # 'elevations' is a collection, but it BOUND
    # Only the write's own unexplained collection leaf is rebound; the binding survives.
    retargeted = retarget_writes(steps, "ridge-notes")
    assert retargeted[1].arguments["memory"] == "ridge-notes"
    assert retargeted[1].arguments["entries"][0]["content"] == "elevations"
    assert "'content': the value from step 1" in render_skill(retargeted)


def test_distill_binds_a_wrapped_prior_result():
    """A binding is structural, not equality: the value binds when it CONTAINS a
    prior result (the model wrapped '$499' into 'Price: $499 today'), #1659."""
    inputs = [
        DistillInput(
            source_ordinal=1, tool="browse", arguments={"queries": ["gadget price"]}, result="$499"
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={
                "memory": "prices",
                "entries": [{"key": "gadget", "content": "Price: $499 today"}],
            },
            result=_WRITE_OK,
        ),
    ]
    steps, _ = distill_steps(inputs, _ATTACHMENT_NAMES)
    content = {tuple(s.path): s for s in steps[1].substitutions}[("entries", 0, "content")]
    assert content.kind == SkillSubKind.BINDING and content.step == 1


def test_distill_does_not_bind_a_trivial_overlap():
    """The binding guard: a sub-``_MIN_BINDING_OVERLAP`` coincidence never binds — a
    1-char prior result contained in a longer arg stays a hole, not a false binding."""
    inputs = [
        DistillInput(source_ordinal=1, tool="browse", arguments={"queries": ["fruit"]}, result="a"),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={"memory": "fruits", "entries": [{"key": "k", "content": "banana"}]},
            result=_WRITE_OK,
        ),
    ]
    steps, _ = distill_steps(inputs, _ATTACHMENT_NAMES)
    content = {tuple(s.path): s for s in steps[1].substitutions}[("entries", 0, "content")]
    # 'a' (len 1) is inside 'banana' but too trivial to bind → 'banana' stays a hole.
    assert content.kind == SkillSubKind.HOLE


# ── #1809: a page is not a returned value — a mid-token match never binds ──────
#
# A browse WITHOUT ``extract`` returns the PAGE, so the payload a leaf is tested
# against is a document, and a document contains most short strings by accident.
# The listing's own URL carries the product's slug, so a write key the assistant
# slugs from the same product is a substring of the payload — while the values the
# round genuinely copied out sit against punctuation ('$499.', '"Aurora Deck 2"').

_FAUX_LISTING_URL = "https://faux-market.example/aurora-deck-2"
_FAUX_LISTING_PAGE = (
    "You opened the Aurora Deck 2 listing (browse result)\n"
    f"## browse {_FAUX_LISTING_URL}:\n"
    '"Aurora Deck 2" — the listed price is $499.\n'
    f"Permalink: {_FAUX_LISTING_URL}\n"
    "In stock."
)

# The recipe the round distils to, whichever label the assistant picked: named slots
# for the two keys IT chose, the fetched values for the two contents it copied.
_LISTING_RECIPE = (
    "1. browse(queries=[{queries}])\n"
    "2. collection_write(memory={memory}, entries=["
    "{'key': {key}, 'content': the value from step 1}, "
    "{'key': {key-2}, 'content': the value from step 1}])"
)


@pytest.mark.parametrize("write_key", ["aurora-deck-2", "aurora-deck-2-price"])
def test_a_write_key_inside_the_pages_url_is_not_a_binding(write_key):
    """A key that appears in the browsed page only as a FRAGMENT of its URL never came
    from the browse — it stays a candidate parameter (#1809).

    Plain containment made the classification turn on an accident of the page: the same
    round, the same leaf, two labels — 'aurora-deck-2' is inside the listing's own URL
    and false-bound to the price, while 'aurora-deck-2-price' is not and classified
    correctly.  The consequence was a routine that wrote each cycle's price under a key
    that was also the price, so nothing ever overwrote anything.  A binding means the
    model COPIED the result, so the value must fill the tokens it lands in; a slice out
    of the middle of one ('…example/aurora-deck-2') is a coincidence of characters.

    Both labels now agree — and the two values the round DID copy still bind even
    though each sits against punctuation in the page ('is $499.', '"Aurora Deck 2"'),
    which is what keeps the narrowing from costing real bindings."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="browse",
            arguments={"queries": [_FAUX_LISTING_URL]},
            result=_FAUX_LISTING_PAGE,
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={
                "memory": "prices",
                "entries": [
                    {"key": write_key, "content": "$499"},
                    {"key": "listing-title", "content": "Aurora Deck 2"},
                ],
            },
            result=_WRITE_OK,
        ),
    ]
    steps, parameters = distill_steps(inputs, _ATTACHMENT_NAMES)
    subs = {tuple(s.path): s for s in steps[1].substitutions}

    assert subs[("entries", 0, "key")].kind == SkillSubKind.HOLE
    assert {parameter.name for parameter in parameters} == {"queries", "memory", "key", "key-2"}
    # Both copied values bind despite their punctuation margins — the price to the
    # figure the page states, the title to the quoted heading.
    for path in (("entries", 0, "content"), ("entries", 1, "content")):
        assert subs[path].kind == SkillSubKind.BINDING
        assert subs[path].step == 1
    # The recipe therefore reads the right way round — a named slot for each key, the
    # fetched value for each content — and never names the demonstrated label.
    assert render_skill(steps) == _LISTING_RECIPE


# ── #1933: a repeat is one input given twice, never a step's output ───────────
#
# A tool result QUOTES the call it answers: a browse section opens with the url it was
# handed (``## browse: <url>``) and an extract result opens with the instruction it was
# handed.  So a demonstration that fetches three pages and then re-fetches one of them
# found that url sitting in the earlier step's result — provenance was asked first,
# answered "it came from step 1", and the routine came out saying
# ``browse(queries=[the value from step 1])``: a program that no longer says which page
# it reads, and whose second fetch would follow whatever the first one happened to
# return.  The url was ALREADY a parameter, minted by the first fetch, and that is what
# a repeat is.  All fictional.

_DIGEST_URLS = (
    "https://news-alpha.example/today",
    "https://news-beta.example/today",
    "https://news-gamma.example/today",
)
_DIGEST_EXTRACT = "the top headline"
_DIGEST_HEADLINES = (
    "Harbour ferries add a night run",
    "Tram works close the lower loop",
    "Library extends its winter hours",
)


def _digest_section(url: str, headline: str) -> str:
    """One page's section of a batched browse-with-``extract`` result: the header naming
    the url the call was GIVEN, then the instruction it was GIVEN, then the value the
    page actually produced (``BrowseTool``'s per-page shape — two echoes and one
    output)."""
    return f"## browse: {url}\n{_DIGEST_EXTRACT}: {headline}"


_DIGEST_RESULT = (
    "You opened three news pages (browse result)\n"
    + "\n".join(
        _digest_section(url, headline)
        for url, headline in zip(_DIGEST_URLS, _DIGEST_HEADLINES, strict=True)
    )
    + "\nFull page content saved to browse-results#41, browse-results#42, "
    "browse-results#43 — read it there for anything more."
)
_DIGEST_WRITE_OK = (
    "You saved entries to headlines: (collection_write result)\nWrote 3 entries to 'headlines'."
)

# The recipe the round distils to: each page named by its own slot, each stored value
# fetched, and the re-read pointed at the SAME page as the first fetch.
_DIGEST_RECIPE = (
    "1. browse(queries=[{queries}, {queries-2}, {queries-3}], extract={extract})\n"
    "2. collection_write(memory={memory}, entries=["
    "{'key': {key}, 'content': the value from step 1}, "
    "{'key': {key-2}, 'content': the value from step 1}, "
    "{'key': {key-3}, 'content': the value from step 1}])\n"
    "3. browse(queries=[{queries}], extract={extract})"
)


def test_a_repeat_of_an_earlier_argument_joins_its_parameter():
    """The observed production shape (#1933): three pages fetched with one ``extract``,
    their headlines written, then ONE of those pages fetched again with the same
    instruction.

    Both leaves of the repeat carry values the first step was called with, and both of
    them turn up in that step's result because the tool quotes its own arguments — so
    plain provenance-first classification bound them to step 1's *output*.  The collapse
    rule ("identical values collapse to ONE candidate") is asked first now, so the repeat
    joins the parameter the first fetch minted: one page, one slot, bound once at
    instantiation.  The values genuinely COPIED out of the payload — the three headlines
    — still bind, which is what keeps this from being a blanket retreat from binding."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="browse",
            arguments={"queries": list(_DIGEST_URLS), "extract": _DIGEST_EXTRACT},
            result=_DIGEST_RESULT,
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={
                "memory": "headlines",
                "entries": [
                    {"key": "alpha-top-headline", "content": _DIGEST_HEADLINES[0]},
                    {"key": "beta-top-headline", "content": _DIGEST_HEADLINES[1]},
                    {"key": "gamma-top-headline", "content": _DIGEST_HEADLINES[2]},
                ],
            },
            result=_DIGEST_WRITE_OK,
        ),
        DistillInput(
            source_ordinal=3,
            tool="browse",
            arguments={"queries": [_DIGEST_URLS[0]], "extract": _DIGEST_EXTRACT},
            result=_digest_section(_DIGEST_URLS[0], _DIGEST_HEADLINES[0]),
        ),
    ]
    steps, parameters = distill_steps(inputs, _ATTACHMENT_NAMES)
    first_fetch = {tuple(s.path): s for s in steps[0].substitutions}
    write = {tuple(s.path): s for s in steps[1].substitutions}
    repeat = {tuple(s.path): s for s in steps[2].substitutions}

    # The re-read joins the first fetch's own slots — no leaf of it is a binding.
    for leaf in (("queries", 0), ("extract",)):
        assert repeat[leaf].kind == SkillSubKind.HOLE
        assert repeat[leaf].parameter == first_fetch[leaf].parameter
    assert all(sub.kind == SkillSubKind.HOLE for sub in steps[2].substitutions)
    # The repeat added no parameter of its own: three pages, one instruction, one
    # collection, three keys.
    assert [parameter.name for parameter in parameters] == [
        "queries",
        "queries-2",
        "queries-3",
        "extract",
        "memory",
        "key",
        "key-2",
        "key-3",
    ]
    # What the pages PRODUCED still binds — one headline per stored entry.
    for index in range(3):
        content = write[("entries", index, "content")]
        assert content.kind == SkillSubKind.BINDING and content.step == 1
        assert write[("entries", index, "key")].kind == SkillSubKind.HOLE
    # So the program reads coherently end to end, and never says which pages the
    # demonstration happened to read.
    assert render_skill(steps) == _DIGEST_RECIPE


_TECH_SECTION_URL = "https://news-alpha.example/tech"
_TECH_HEADLINE = "Harbour ferries add a night run"
_TECH_SEARCH_RESULT = (
    "You searched for 'news-alpha tech section' (browse result)\n"
    "## browse search: news-alpha tech section\n"
    "Tech — News Alpha\n"
    f"{_TECH_SECTION_URL}"
)


def test_an_echoed_argument_binds_to_the_step_that_produced_it():
    """The other half of #1933, where the collapse rule cannot help: a value a step
    genuinely PRODUCED, passed straight into the next call — so that leaf BOUND and no
    parameter was ever minted for it — and used once more afterwards.

    The middle step's result quotes the url it was handed, so the third step's copy of
    it matched there first and bound to the fetch rather than to the search that found
    it.  A span the step was CALLED with is excluded from its result, so the binding
    lands on the step that actually produced the value — the redirect is to the truth,
    not away from binding."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="browse",
            arguments={"queries": ["news-alpha tech section"]},
            result=_TECH_SEARCH_RESULT,
        ),
        DistillInput(
            source_ordinal=2,
            tool="browse",
            arguments={"queries": [_TECH_SECTION_URL], "extract": _DIGEST_EXTRACT},
            result=_digest_section(_TECH_SECTION_URL, _TECH_HEADLINE),
        ),
        DistillInput(
            source_ordinal=3,
            tool="browse",
            arguments={"queries": [_TECH_SECTION_URL], "extract": _DIGEST_EXTRACT},
            result=_digest_section(_TECH_SECTION_URL, _TECH_HEADLINE),
        ),
    ]
    steps, _ = distill_steps(inputs, _ATTACHMENT_NAMES)
    fetch = {tuple(s.path): s for s in steps[1].substitutions}
    repeat = {tuple(s.path): s for s in steps[2].substitutions}

    # The search DID produce the url — the fetch binds to it, as it always did.
    assert fetch[("queries", 0)].kind == SkillSubKind.BINDING
    assert fetch[("queries", 0)].step == 1
    # And so does the re-read, rather than to the fetch that merely quoted it back.
    assert repeat[("queries", 0)].kind == SkillSubKind.BINDING
    assert repeat[("queries", 0)].step == 1
    # The instruction was never a result: it is the parameter the first fetch minted.
    assert repeat[("extract",)].kind == SkillSubKind.HOLE
    assert repeat[("extract",)].parameter == fetch[("extract",)].parameter


_KEEL_LANTERN_PAGE = (
    "You searched for lantern (browse result)\n"
    "## browse: lantern\n"
    "the price: $499 for the keel lantern"
)


def test_an_argument_word_inside_a_fetched_value_keeps_its_binding():
    """The echo exclusion is per OCCURRENCE, not a redaction of the payload.

    A search term turns up twice in this result: once in the header quoting the query,
    and once inside the sentence the page produced.  Excluding the ECHO must not cost the
    sentence — a value copied out of the payload that merely CONTAINS an argument word is
    still a copy, and demoting it would leave the routine with a required parameter
    holding a price nobody could supply (the #1770 harm).  The occurrence that lands
    clear of the echo is what binds."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="browse",
            arguments={"queries": ["lantern"], "extract": "the price"},
            result=_KEEL_LANTERN_PAGE,
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={
                "memory": "prices",
                "entries": [{"key": "keel lantern price", "content": "$499 for the keel lantern"}],
            },
            result=_WRITE_OK,
        ),
    ]
    steps, _ = distill_steps(inputs, _ATTACHMENT_NAMES)
    write = {tuple(s.path): s for s in steps[1].substitutions}

    content = write[("entries", 0, "content")]
    assert content.kind == SkillSubKind.BINDING and content.step == 1
    # And the key the assistant chose still isn't one: it appears nowhere in the page.
    assert write[("entries", 0, "key")].kind == SkillSubKind.HOLE


def test_distill_strips_the_top_level_reasoning_thinkaloud():
    """#1661: the universal top-level ``reasoning`` think-aloud every real call carries
    is stripped at distill — it adds NO hole, never lands in a stored step's
    arguments, and never renders (it is per-run narration; the executing model
    supplies its own reasoning at run time)."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="browse",
            arguments={**_BROWSE_ARGS, "reasoning": _REASONING},
            result=_BROWSE_OK,
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={**_WRITE_ARGS, "reasoning": "because it flowed from step 1"},
            result=_WRITE_OK,
        ),
    ]
    steps, holes = distill_steps(inputs, _ATTACHMENT_NAMES)
    # The same candidates as the reasoning-free run above — the think-aloud added none.
    assert holes == [
        SkillParameter(name="queries", required=True),
        SkillParameter(name="extract", required=True),
        SkillParameter(name="memory", required=True),
    ]
    assert all("reasoning" not in step.arguments for step in steps)
    assert "reasoning=" not in render_skill(steps)


def test_distill_keeps_a_nested_key_named_reasoning():
    """Only the TOP-LEVEL ``reasoning`` is stripped (#1661): a NESTED arg that merely
    shares the name is real routine data — it stays in the stored step (and, as a
    non-binding string leaf, is a hole like any other)."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="collection_write",
            arguments={
                "memory": "notes",
                "reasoning": "top-level narration — stripped",
                "entries": [{"key": "k", "content": "c", "reasoning": "nested — kept"}],
            },
            result=_WRITE_OK,
        ),
    ]
    steps, _ = distill_steps(inputs, _ATTACHMENT_NAMES)
    args = steps[0].arguments
    assert "reasoning" not in args  # the top-level think-aloud is gone
    assert args["entries"][0]["reasoning"] == "nested — kept"  # the nested key is untouched


# ── skill_read: render one / list all ─────────────────────────────────────────


def _seed_skill(db: Database, name: str) -> None:
    """Persist a skill directly (the extractor's post-labeller output shape) so the read
    surface can be tested independently of the extraction path."""
    db.skills.upsert(
        SkillDraft(
            name=name,
            intent=_UTTERANCE,
            description=_UTTERANCE,
            steps=_elevation_steps(),
            parameters=[
                SkillParameter(name="queries", required=True),
                SkillParameter(name="extract", required=True),
            ],
            source_run_id="run-A",
        ),
        author="chat",
        description_embedding=None,
    )


@pytest.mark.bare_db
@pytest.mark.asyncio
async def test_skill_read_renders_one_and_lists_all(db):
    """``skill_read(name)`` renders one full recipe; bare ``skill_read()`` lists
    every skill; an unknown name is an actionable miss."""
    _seed_skill(db, "Watch elevation")
    read = SkillReadTool(db)

    one = await read.execute(name="Watch elevation")
    # render_skill_full indents the recipe block two spaces under `steps:` (#1668).
    indented_recipe = "\n".join(f"  {line}" for line in _WITH_HOLES.splitlines())
    assert one.success and "skill 'Watch elevation'" in one.message
    assert indented_recipe in one.message

    listing = await read.execute()
    assert listing.success and "- Watch elevation:" in listing.message

    missing = await read.execute(name="nope")
    assert not missing.success and "No skill named 'nope'" in missing.message


# ── The empty registry: honest empty state, no seeds ──────────────────────────

# Pinned literal: the honest empty-registry listing.  Migration 0084 ships the
# skill table EMPTY (no seed library — every skill is distilled from a chat run),
# so this is what a fresh install's skill_read() returns.
_EMPTY_LISTING = (
    "No skills yet — teach one by demonstrating a flow here in chat, and I'll learn it "
    "automatically."
)


@pytest.mark.asyncio
async def test_fresh_migrated_registry_is_empty_and_reads_honestly(db):
    """A prod-identical DB (create_tables + migrate) has the skill table and ZERO
    rows — no seeds — and skill_read() renders the honest empty state verbatim."""
    assert db.skills.list_all() == []
    listing = await SkillReadTool(db).execute()
    assert listing.success
    assert listing.message == _EMPTY_LISTING


# ── Attachment binding at apply (#1629/#1783, pure) ────────────────────────────


def test_retarget_writes_binds_the_write_memory_to_the_target():
    """The attachment-marked ``memory`` placeholder is BOUND to the target
    collection's name, and the render reflects it — a skill demoed against
    'elevations' renders its write to the collection it's applied to."""
    steps = _elevation_steps()  # step 2's memory leaf is the #1777 placeholder
    retargeted = retarget_writes(steps, "cinder-elevation")
    rendered = render_skill(
        retargeted, {"queries": "Cinder Peak", "extract": "the elevation above sea level"}
    )
    assert rendered == (
        "1. browse(queries=['Cinder Peak'], extract='the elevation above sea level')\n"
        "2. collection_write(memory='cinder-elevation', "
        "entries=[{'key': 'Cinder Peak', 'content': the value from step 1}])"
    )
    # Pure: the source steps are untouched (a skill is target-agnostic at rest).
    assert steps[1].arguments["memory"] == "elevations"
    assert steps[1].substitutions[0].kind == SkillSubKind.PLACEHOLDER


def test_retarget_writes_binds_a_migrated_legacy_write_target():
    """A skill taught before #1783 carries no mark of its own — migration 0103 sets one
    on the leaf 0101 gave a placeholder, and from then on it binds like any other marked
    leaf.  So an instantiated collection's program acts on its own collection whatever
    shape the stored skill was taught in."""
    migrated = SkillStep(
        ordinal=1,
        source_ordinal=1,
        tool="collection_write",
        arguments={"memory": "elevations", "entries": [{"key": "k", "content": "c"}]},
        substitutions=[
            SkillSubstitution(
                path=["memory"],
                kind=SkillSubKind.PLACEHOLDER,
                description=WRITE_TARGET_DESCRIPTION,
                attachment=True,
            )
        ],
    )
    retargeted = retarget_writes([migrated], "peak-notes")
    assert render_skill(retargeted) == (
        "1. collection_write(memory='peak-notes', entries=[{'key': 'k', 'content': 'c'}])"
    )


def test_retarget_writes_binds_only_what_carries_the_mark():
    """The deliberate consequence of binding by mark (#1783): a leaf with NO mark is not
    bound, whatever tool it belongs to.  An unmigrated pre-#1783 row would therefore
    keep its demonstrated literal — which is exactly why migration 0103 exists, and why
    nothing in this path may fall back to a tool whitelist."""
    unmarked = SkillStep(
        ordinal=1,
        source_ordinal=1,
        tool="collection_write",
        arguments={"memory": "elevations", "entries": [{"key": "k", "content": "c"}]},
        substitutions=[],
    )
    assert retarget_writes([unmarked], "peak-notes") == [unmarked]


def test_a_user_named_destination_stays_a_parameter():
    """The labeller judged the USER to have named the destination, so the leaf keeps its
    parameter and carries no mark — the attachment must not overwrite a choice the user
    made.  Two such destinations therefore stay distinct, with no mechanism added for
    it: ``params`` binds each one separately (#1783)."""
    step = SkillStep(
        ordinal=1,
        source_ordinal=1,
        tool="collection_write",
        arguments={"memory": "reading-list", "entries": [{"key": "k", "content": "c"}]},
        substitutions=[
            SkillSubstitution(path=["memory"], kind=SkillSubKind.HOLE, parameter="destination")
        ],
    )
    assert retarget_writes([step], "somewhere-else") == [step]
    assert render_skill([step], {"destination": "weekend-reading"}) == (
        "1. collection_write(memory='weekend-reading', entries=[{'key': 'k', 'content': 'c'}])"
    )


def test_retarget_writes_binds_a_marked_hole_and_drops_its_substitution():
    """A marked leaf that is still a HOLE (the labeller left no verdict on it, so the
    render seam is what fills it) is bound to the target and its substitution dropped —
    else the render would put a marker back over the collection's own name."""
    step = SkillStep(
        ordinal=1,
        source_ordinal=1,
        tool="collection_write",
        arguments={"memory": "elevations", "entries": [{"key": "k", "content": "c"}]},
        substitutions=[
            SkillSubstitution(
                path=["memory"], kind=SkillSubKind.HOLE, parameter="memory", attachment=True
            )
        ],
    )
    retargeted = retarget_writes([step], "target-b")
    assert retargeted[0].arguments["memory"] == "target-b"
    assert all(sub.path != ["memory"] for sub in retargeted[0].substitutions)
    assert "memory='target-b'" in render_skill(retargeted, {})


def test_retarget_writes_leaves_unmarked_steps_untouched():
    """A step with no marked leaf (a browse) is passed through unchanged — binding
    reads the mark, never the tool."""
    steps = _elevation_steps()
    retargeted = retarget_writes(steps, "target-b")
    assert retargeted[0].arguments == steps[0].arguments  # the browse step is identical


# ── The runtime join: framed parameters onto the program's leaves (#1907) ─────
#
# The post-#1828 shape a taught routine really has: every spot is a PLACEHOLDER
# carrying what belongs there, the write target additionally marked, and the
# interface is the framer's — each parameter carrying the value the round
# DEMONSTRATED it with, which is the only thing the two draws share.

_TIMETABLE_URL = "https://harbour-ferry.example/timetable"
_DEMONSTRATED_LINE = "the first sailing"


def _timetable_steps() -> list[SkillStep]:
    return [
        SkillStep(
            ordinal=1,
            source_ordinal=1,
            tool="browse",
            arguments={"queries": [_TIMETABLE_URL], "extract": _DEMONSTRATED_LINE},
            substitutions=[
                SkillSubstitution(
                    path=["queries", 0],
                    kind=SkillSubKind.PLACEHOLDER,
                    description="the url of the timetable page to browse each run",
                ),
                SkillSubstitution(
                    path=["extract"],
                    kind=SkillSubKind.PLACEHOLDER,
                    description="the text or line to look for on that page",
                ),
            ],
        ),
        SkillStep(
            ordinal=2,
            source_ordinal=2,
            tool="collection_write",
            arguments={
                "memory": "sailings",
                "entries": [{"key": "first sailing", "content": "07:10"}],
            },
            substitutions=[
                SkillSubstitution(
                    path=["memory"],
                    kind=SkillSubKind.PLACEHOLDER,
                    description=WRITE_TARGET_DESCRIPTION,
                    attachment=True,
                ),
                SkillSubstitution(
                    path=["entries", 0, "key"],
                    kind=SkillSubKind.PLACEHOLDER,
                    description="the key the extracted value is stored under",
                ),
                SkillSubstitution(
                    path=["entries", 0, "content"], kind=SkillSubKind.BINDING, step=1
                ),
            ],
        ),
    ]


def _timetable_parameters() -> list[SkillParameter]:
    return [
        SkillParameter(
            name="url",
            description="the timetable page to read",
            value=_TIMETABLE_URL,
        ),
        SkillParameter(
            name="line",
            description="which sailing to look for",
            value=_DEMONSTRATED_LINE,
        ),
    ]


def test_bind_parameters_writes_the_bound_values_into_the_leaves_they_fill():
    """The join, whole (#1907): the page a run fetches and the thing it looks for read
    as themselves in the program, instead of as descriptions of themselves.

    Each is found by the value the DEMONSTRATION put there — the framer recorded the
    user's word for the page, the ledger recorded what the browse actually carried, and
    where those agree that leaf is the parameter's site.  The entry key is claimed by
    nobody and keeps saying what belongs in it (it is a per-run value, not a term of the
    job), and the write target is the ATTACHMENT's, bound to the collection the routine
    was applied to and never to a parameter."""
    steps = bind_parameters(
        retarget_writes(_timetable_steps(), "ferry-departures"),
        _timetable_parameters(),
        {"url": "https://northpier.example/departures", "line": "the dawn sailing"},
    )

    assert render_skill(steps) == (
        "1. browse(queries=['https://northpier.example/departures'], "
        "extract='the dawn sailing')\n"
        "2. collection_write(memory='ferry-departures', "
        "entries=[{'key': {the key the extracted value is stored under}, "
        "'content': the value from step 1}])"
    )


def test_bind_parameters_says_so_by_name_when_a_parameter_got_no_value():
    """Visible degradation over silent success: a parameter with nothing bound to it
    leaves its leaf naming the gap, not the description it had.

    The description reads like an instruction — "the url of the timetable page to browse
    each run" is something a collector could try to act on — so leaving it standing over
    an unsupplied parameter is a step that looks runnable and is not.  The parameter's
    own name is what the leaf says, because that name is the key someone rebinds with."""
    steps = bind_parameters(
        retarget_writes(_timetable_steps(), "ferry-departures"),
        _timetable_parameters(),
        {"line": "the dawn sailing"},
    )

    assert render_skill(steps) == (
        "1. browse(queries=[{no value was supplied for the parameter 'url'}], "
        "extract='the dawn sailing')\n"
        "2. collection_write(memory='ferry-departures', "
        "entries=[{'key': {the key the extracted value is stored under}, "
        "'content': the value from step 1}])"
    )


def test_bind_parameters_joins_nothing_for_a_routine_that_recorded_no_values():
    """A routine taught before the join existed — or one whose framing failed — declares
    parameters with no demonstrated value, so nothing can say which leaf each fills.  The
    program renders exactly as it did before rather than guessing a site."""
    steps = retarget_writes(_timetable_steps(), "ferry-departures")
    unrecorded = [SkillParameter(name="url", description="the timetable page to read")]

    assert bind_parameters(steps, unrecorded, {"url": "https://northpier.example/x"}) == steps


def test_bind_parameters_declines_a_value_two_parameters_both_claim():
    """Two parameters demonstrated with the SAME value: nothing in the program tells
    their leaves apart, so neither binds.  A leaf that keeps saying what belongs in it is
    recoverable; one silently carrying the other parameter's value is a wrong write every
    cycle."""
    steps = retarget_writes(_timetable_steps(), "ferry-departures")
    twins = [
        SkillParameter(name="url", value=_TIMETABLE_URL),
        SkillParameter(name="mirror", value=_TIMETABLE_URL),
    ]

    bound = bind_parameters(
        steps, twins, {"url": "https://a.example", "mirror": "https://b.example"}
    )
    assert "https://a.example" not in render_skill(bound)
    assert "https://b.example" not in render_skill(bound)


def test_bind_parameters_never_claims_the_attachment_marked_leaf():
    """Where a routine writes is decided by what it is applied to (#1827 principle 4),
    so a parameter demonstrated with the collection's own name still binds nothing there
    — the attachment fills it at the seam before this runs, and the mark is what says so.
    """
    steps = _timetable_steps()
    impostor = [SkillParameter(name="destination", value="sailings")]

    bound = bind_parameters(steps, impostor, {"destination": "somewhere-else"})
    assert "somewhere-else" not in render_skill(bound)
    assert (
        render_skill(retarget_writes(bound, "ferry-departures")).count("memory='ferry-departures'")
        == 1
    )


# ── Asking for a routine again: the binder's document + the derived name (#1867)


def test_build_binding_content_renders_the_signature_then_the_users_words():
    """The binder's document, WHOLE — the surface the binding draw actually reads.

    The routine comes first (what it is called, what it is for, one line per declared
    parameter) and the user's own words come last, because the words are what is being
    read: the draw is told what to look for, then handed the text to look in.  EVERY
    declared parameter renders, so none of them can only be answered by guessing."""
    spoken = render_spoken_turns(
        (
            "can you keep an eye on the sailing board at https://saltmarsh.example/board?",
            "tell me when the dawn crossing turns up",
        )
    )

    content = build_binding_content(
        spoken,
        "check_sailing_board",
        "read a sailing board and report the status of one entry",
        [
            SkillParameter(name="url", description="the URL of the board to read each run"),
            SkillParameter(name="keyword", description="which entry to look for on it"),
        ],
    )

    assert content == (
        "The routine that has been asked for:\n"
        "name: check_sailing_board\n"
        "what it is for: read a sailing board and report the status of one entry\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: the URL of the board to read each run\n"
        "- keyword: which entry to look for on it\n"
        "\n"
        "What the user said, in their own words:\n"
        "can you keep an eye on the sailing board at https://saltmarsh.example/board?\n"
        "tell me when the dawn crossing turns up"
    )
    # The document ENDS with the user's turns, byte for byte — the same string the span
    # check tests each drawn value against, so the two can never describe different text.
    assert content.endswith(spoken)


def test_build_binding_content_renders_a_parameter_that_carries_no_description():
    """A signature can carry a parameter nobody described (a framing that fell back).  Its
    line still renders — the draw has to answer for it either way — with the description
    omitted cleanly rather than as an empty tail, and a signature with no parameters at all
    says so instead of leaving a heading over nothing."""
    described = build_binding_content(
        "do the thing", "run_it", "runs it", [SkillParameter(name="url")]
    )
    assert "What it needs, one line each:\n- url\n" in described

    assert "What it needs, one line each:\n(nothing)\n" in build_binding_content(
        "do the thing", "run_it", "runs it", []
    )


def test_derive_collection_name_slugs_the_skill_and_normalises_a_url():
    """The scheme at its simplest: the skill's name slugged, then the value's host and
    path.  The whole point is that the name is obvious without resolving anything."""
    assert (
        derive_collection_name("monitor_price", ["https://faux-market.example/keel-lantern"])
        == "monitor-price-faux-market-example-keel-lantern"
    )


def test_derive_collection_name_reads_one_page_written_four_ways_as_one_job():
    """Identity is the page, not how it was typed: the scheme, a ``www.``, a tracking
    query, a fragment and a trailing slash are all dropped, so four spellings of one page
    derive one name and find-or-create hands back the job the user already has."""
    spellings = [
        "https://harborbakery.example/menu",
        "http://www.harborbakery.example/menu/",
        "harborbakery.example/menu?utm_source=mail",
        "https://harborbakery.example/menu#specials",
    ]
    derived = {derive_collection_name("fetch_daily_special", [one]) for one in spellings}
    assert derived == {"fetch-daily-special-harborbakery-example-menu"}


def test_derive_collection_name_slugs_a_phrase_that_is_not_a_url():
    """A value that is not an address is slugged as the phrase it is — and a phrase is
    never read as a url, so a sentence with a full stop in it keeps everything after the
    stop."""
    assert (
        derive_collection_name("track_symbol", ["a first look. then the rest"])
        == "track-symbol-a-first-look-then-the-rest"
    )
    assert derive_collection_name("track_symbol", ["VLT"]) == "track-symbol-vlt"


def test_derive_collection_name_joins_several_values_in_declared_order():
    """Every declared parameter contributes, in the order the signature declares them —
    which is what makes two jobs differing only in their second value two names."""
    assert (
        derive_collection_name(
            "check_ferry_timetable", ["https://northpier.example/departures", "dawn sailing"]
        )
        == "check-ferry-timetable-northpier-example-departures-dawn-sailing"
    )

    assert derive_collection_name(
        "check_ferry_timetable", ["https://northpier.example/departures", "late sailing"]
    ) != derive_collection_name(
        "check_ferry_timetable", ["https://northpier.example/departures", "dawn sailing"]
    )


def test_derive_collection_name_shortens_on_whole_tokens_and_keeps_every_value():
    """The length policy: the name stays inside the reading budget by dropping WHOLE
    trailing tokens — no hash, no ellipsis, nothing mid-word — and the budget is spent
    round-robin, so a second parameter is still represented when the first one is long
    enough to have eaten it under a plain tail truncation."""
    long_path = "https://faux-market.example/very-long-product-slug-that-keeps-on-going"

    single = derive_collection_name("monitor_price", [long_path])
    assert len(single) <= DERIVED_NAME_MAX_LENGTH
    assert single == "monitor-price-faux-market-example-very-long-product-slug-that"

    pair = derive_collection_name("monitor_price", [long_path, "brass lantern"])
    assert len(pair) <= DERIVED_NAME_MAX_LENGTH
    assert pair == "monitor-price-faux-market-example-very-long-brass-lantern"
    # The first value is shortened further than it was on its own, and BOTH values are
    # still in the name — which is the whole reason the budget is spent in turn rather
    # than front to back.
    assert "brass" in pair and "lantern" in pair


def test_derive_collection_name_handles_a_value_with_nothing_sluggable_in_it():
    """A value carrying no alphanumerics contributes no tokens rather than an empty
    separator run, and a routine pointed at nothing derives its own name — degraded, but
    still a readable name rather than a hyphen."""
    assert derive_collection_name("monitor_price", ["!!!"]) == "monitor-price"
    assert derive_collection_name("monitor_price", []) == "monitor-price"
