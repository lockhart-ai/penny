"""Unit tests for the pure text-validity detectors.

``is_degenerate_run`` is the shared fingerprint for a gpt-oss punctuation-collapse
("...??…?..").  Because a false positive would discard a *healthy* model response
(and, at the write gate, refuse legitimate content), the zero-false-positive
contract on real punctuation is load-bearing — so it's pinned here against a
corpus of legitimate punctuation that must never match and a corpus of collapse
shapes captured from the prompt log that must always match.
"""

from __future__ import annotations

from penny.text_validity import degenerate_reason, is_degenerate_run, is_degenerate_tool_name

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
