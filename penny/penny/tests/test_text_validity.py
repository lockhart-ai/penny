"""Unit tests for the pure text-validity detectors.

``is_degenerate_run`` is the shared fingerprint for a gpt-oss punctuation-collapse
("...??…?..").  Because a false positive would discard a *healthy* model response
(and, at the write gate, refuse legitimate content), the zero-false-positive
contract on real punctuation is load-bearing — so it's pinned here against a
corpus of legitimate punctuation that must never match and a corpus of collapse
shapes captured from the prompt log that must always match.
"""

from __future__ import annotations

from penny.text_validity import (
    check_extraction_prompt_tools,
    degenerate_reason,
    extract_tool_call_names,
    is_degenerate_run,
    is_degenerate_tool_name,
)

# Legitimate punctuation that must NEVER be flagged — conversational ellipses,
# emphatic marks, list/code notation.  A hit here would throw out good output.
LEGITIMATE = [
    "Wait... what?!",
    "Hmm...?",
    "Really...?",
    "Anyway… let's go",
    "to be continued…",
    "The list includes a, b, c...",
    "He said 'well...' and left",
    "[1, 2, 3, ...]",
    "def f(*args): ...",
    "Score: 9/10 — amazing!",
    "What?! No way!",
    "So good!!",
    "Is that true?",
    "Loading, please wait...",
    "one… two… three",
    "Yes!! Finally!",
    "huh?!",
    "Heads up — a new title dropped, details inside.",
]

# Degeneration-collapse runs (ASCII dots, the ellipsis char, and the non-breaking
# separators the model laces through them) that must ALWAYS be flagged.
DEGENERATE = [
    "...??…?..?????..?",
    "… ……?? ……………?????",
    "AI\xa0……?",
    "New Prague\xa0…\xa0…\xa0…\xa0…\xa0…",
    "Delivered deliver...???",
    "the summary is ...??",
    "...…………—………... !…..",
    "West …\xa0…\xa0……“\xa0……\xa0…",
    "Got it...?? ..",
    "Hi there! ......???",
    "..??",
    "New restaurant … … … … openings",
]


def test_is_degenerate_run_never_flags_legitimate_punctuation():
    flagged = [text for text in LEGITIMATE if is_degenerate_run(text)]
    assert flagged == [], f"false positives on legitimate punctuation: {flagged}"


def test_is_degenerate_run_flags_every_collapse_shape():
    missed = [text for text in DEGENERATE if not is_degenerate_run(text)]
    assert missed == [], f"missed degeneration collapses: {missed}"


# Collapse-shaped tool-call NAMES (shapes seen in the prompt log, plus synthetic
# variants per collapse character) that must ALWAYS be flagged — the collapse
# landing in the name field instead of content.
DEGENERATE_TOOL_NAMES = [
    "Functions?????",
    "funcs.done?",
    "read_simpar?",
    "collection_write…",
    "done..",
    "log_read!!",
]

# Names that must NEVER be flagged, even though none is in the registry: a
# plausible near-miss identifier keeps the executor's tool-not-found path (with
# its "Did you mean X?" hint), and a Harmony-token-wrapped valid name is the
# Harmony-strip repair case (#1306), not poison.
PLAUSIBLE_TOOL_NAMES = [
    "collection_metadata",
    "read_last",
    "functions.done",
    "done<|channel|>commentary",
    "example_function_name",
]


def test_is_degenerate_tool_name_flags_every_collapse_shape():
    missed = [name for name in DEGENERATE_TOOL_NAMES if not is_degenerate_tool_name(name)]
    assert missed == [], f"missed degenerate tool names: {missed}"


def test_is_degenerate_tool_name_never_flags_plausible_identifiers():
    flagged = [name for name in PLAUSIBLE_TOOL_NAMES if is_degenerate_tool_name(name)]
    assert flagged == [], f"false positives on plausible tool names: {flagged}"


def test_degenerate_reason_rejects_wordful_poison():
    """A collapse embedded in otherwise-wordful text clears the blank/URL/bail-out
    checks, so the run detector is what keeps it out of the corpus and off the wire."""
    reason = degenerate_reason("Delivered a find about Boss ..??.. gear")
    assert reason is not None
    assert "degenerate" in reason.lower()
    # A clean summary with a normal trailing ellipsis is still accepted.
    assert degenerate_reason("A new Boss delay pedal dropped this week…") is None


# ── extraction_prompt tool-call extraction + validation ──────────────────────

# A realistic multi-step extraction_prompt (genericized), mixing canonical
# ``tool(args)`` calls with the prose parentheticals ("(e.g. the first)") a recipe
# naturally carries.  Only the real calls should be extracted — the no-space rule
# separating ``browse(`` from ``result (e.g.`` is what keeps a prose aside from
# reading as a call.
_REALISTIC_PROMPT = (
    "Collect strategy board games — one category at a time.\n"
    "1. Pick one category: euro, co-op, deckbuilder (randomly).\n"
    '2. browse(["{category} board games"])  # search\n'
    "3. From the results select one (e.g. the first, or at random) and store its url.\n"
    "4. browse([game_url])  # read the page\n"
    "5. From that page extract the title, designer, and a one-line hook.\n"
    '6. collection_write("board-games", entries=[{key: "{title}", content: "..."}])\n'
    "7. done()"
)


def test_extract_tool_call_names_finds_calls_and_skips_prose_parentheticals():
    """The canonical ``tool(`` calls are extracted in order; a prose parenthetical
    with a space before ``(`` ("select one (e.g. ...)") is never mistaken for one."""
    assert extract_tool_call_names(_REALISTIC_PROMPT) == ["browse", "collection_write", "done"]


def test_extract_tool_call_names_ignores_bare_and_spaced_parentheticals():
    """Prose that opens a paren after a space, an uppercase word, or a dotted name
    is not a call — the extractor must return nothing for any of these."""
    for prose in [
        "pick one platform/genre combination (randomly) and note it",
        "read the article (skimming is fine) before writing",
        "the release date (YYYY) goes in the entry",
        "call functions.browse eventually",  # dotted / not call-shaped
        "Store the summary somewhere useful",  # capitalised, no paren
    ]:
        assert extract_tool_call_names(prose) == [], prose


def test_extract_tool_call_names_ignores_the_pluralisation_idiom():
    """``word(s)`` / ``word(es)`` / ``word(ies)`` is prose ("store the url(s)"), not a
    call — a bare short plural suffix closing the paren must never be extracted, or a
    legitimate step would be wrongly rejected.  A ``done()``-style empty call still is."""
    for prose in [
        "store the url(s) you visited",
        "skip any duplicate(s) already on record",
        "note the tag(s) and category(ies)",
        "read all the class(es) it belongs to",
        "list the page(s) you opened",
    ]:
        assert extract_tool_call_names(prose) == [], prose
    # The exclusion is specific to the plural suffix — an empty call is still a call.
    assert extract_tool_call_names("finally done()") == ["done"]


# The rule takes the valid surface as a parameter, so it's exercised against a small
# contrived set here — the real collector surface is discovered at the tool call site
# (``memory_tools``) and integration-tested there, not duplicated into this unit.
_CONTRIVED_TOOLS = frozenset({"browse", "collection_write", "send_message", "done"})


def test_check_extraction_prompt_tools_accepts_calls_in_the_surface():
    """A prompt whose every call is in the supplied surface — including a legitimate
    ``send_message`` step on a notify collector — passes cleanly."""
    assert check_extraction_prompt_tools(_REALISTIC_PROMPT, _CONTRIVED_TOOLS) is None
    notify_prompt = (
        'Watch for news.\n1. browse(["updates"])\n'
        '2. collection_write("news", entries=[{key: "k", content: "c"}])\n'
        '3. send_message("heads up")\n4. done()'
    )
    assert check_extraction_prompt_tools(notify_prompt, _CONTRIVED_TOOLS) is None


def test_check_extraction_prompt_tools_flags_a_call_outside_the_surface():
    """A call not in the surface (a hallucinated ``extract_text(...)``) is rejected with
    an actionable message naming the offender and the tools that DO exist."""
    prompt = (
        'Collect things.\n1. browse(["x"])\n2. extract_text(page)  # not a real tool\n'
        '3. collection_write("things", entries=[{key: "k", content: "c"}])\n4. done()'
    )
    error = check_extraction_prompt_tools(prompt, _CONTRIVED_TOOLS)
    assert error is not None
    assert "extract_text" in error
    # Names a real tool the model can use instead (the available surface).
    assert "collection_write" in error
    # A tool that IS in the surface, though present in the prompt, is NOT reported.
    assert "browse" not in error.split("only these tools:")[0]
