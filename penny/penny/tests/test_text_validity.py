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
    call_as_text_tool_name,
    degenerate_reason,
    is_degenerate_run,
    is_degenerate_tool_name,
    is_embedded_call_as_text,
    suggest_tool_name,
)

# A chat agent's registered tool names, for the embedded-call-as-text tests below.
REGISTERED_TOOLS = frozenset(
    {"browse", "generate_image", "collection_configure", "find", "memory_get"}
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
    "San Diego\xa0…\xa0…\xa0…\xa0…\xa0…",
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


def test_is_embedded_call_as_text_catches_a_call_object_after_prose():
    """The prod leak (#2787): one line of prose, then the call serialized as JSON
    text — a shape the whole-reply-only ``is_call_as_text_bail`` can't see.  Caught
    by naming a registered tool as a string value.  (Before #2813 a bare-args
    object carrying only the framework-injected ``reasoning`` key, with no tool
    name at all, was also caught here — that field and the signal it carried are
    both gone now.)"""
    assert is_embedded_call_as_text(
        'On it!\n\n{"action": "generate_image", "action_input": '
        '"{\\"description\\": \\"a lighthouse in a storm\\"}"}',
        REGISTERED_TOOLS,
    )


def test_is_embedded_call_as_text_catches_textual_call_syntax():
    """A draft that writes the call as ``tool_name(arg=...)`` prose instead of
    JSON is caught the same way, keyed to the live registry — a real argument
    name followed by ``=``, not the literal word ``identifier``."""
    assert is_embedded_call_as_text(
        'Sure!\n\ngenerate_image(description="a lighthouse")', REGISTERED_TOOLS
    )


def test_is_embedded_call_as_text_textual_syntax_never_flags_lookalikes():
    """Word-boundary anchored on the tool name, and anchored on ``(`` itself —
    a longer word ending in a tool's name, or a mention with a space before the
    parenthesis, must not trip it."""
    # "rebrowse(" ends in "browse(" but is a different identifier entirely.
    assert not is_embedded_call_as_text('rebrowse(x="1")', REGISTERED_TOOLS)
    # A parenthetical aside after the tool's name, not a call at all.
    assert not is_embedded_call_as_text("I'll browse (quickly) for that", REGISTERED_TOOLS)


def test_is_embedded_call_as_text_never_flags_ordinary_prose_or_mentions():
    """Zero-false-positive discipline (#2787): plain prose, a sentence that merely
    NAMES a tool, a JSON code sample with none of the call shapes, and a value that
    only CONTAINS a tool's name as a substring must all pass through untouched."""
    negatives = [
        "Lake Baikal is the deepest lake in the world!",
        "I can use browse for that if you'd like me to look it up.",
        'Here\'s an example object: {"color": "teal", "count": 3}',
        '{"description": "let\'s browse the shelves at the library this weekend"}',
    ]
    flagged = [text for text in negatives if is_embedded_call_as_text(text, REGISTERED_TOOLS)]
    assert flagged == [], f"false positives: {flagged}"


def test_is_embedded_call_as_text_never_flags_a_call_quoted_in_inline_code():
    """The prod leak (#2855): a rule proposal that QUOTES the calls it would make,
    each inside inline code, is a worked example — not a leak — so it must not be
    discarded and rerolled.  ``call_as_text_tool_name`` must agree there is no
    tool name to retry against."""
    reply = (
        "Here's a rule for looking things up in memory:\n\n"
        "1. **Trigger:** the user asks where something is.\n"
        '2. **Tool call:** `find(query="Biscuit")`\n'
        '3. **Fallback:** if nothing turns up, try `memory_get(memory="pets")`.\n\n'
        "Want me to save this as a rule?"
    )
    assert not is_embedded_call_as_text(reply, REGISTERED_TOOLS)
    assert call_as_text_tool_name(reply, REGISTERED_TOOLS) is None


def test_is_embedded_call_as_text_still_flags_the_same_call_written_bare():
    """The identical call, written bare in prose rather than quoted in inline
    code, is still the leak #2787 caught — the inline-code strip must not widen
    into a blanket exemption."""
    reply = 'Let me check. find(query="Biscuit")'
    assert is_embedded_call_as_text(reply, REGISTERED_TOOLS)
    assert call_as_text_tool_name(reply, REGISTERED_TOOLS) == "find"


def test_is_embedded_call_as_text_still_flags_a_call_inside_a_fenced_block():
    """A FENCED ``` block is a different shape than inline code and is never
    stripped — a call written there still leaks exactly as before."""
    reply = 'Here you go:\n\n```\nfind(query="Biscuit")\n```\n'
    assert is_embedded_call_as_text(reply, REGISTERED_TOOLS)
    assert call_as_text_tool_name(reply, REGISTERED_TOOLS) == "find"


def test_suggest_tool_name_falls_through_when_the_rename_target_is_absent():
    """The table remembers renames, never what EXISTS: a retired name whose replacement
    is not on the surface takes the string leg like any other miss, so a suggestion
    always names something callable (#1944)."""
    assert suggest_tool_name("collection_metadata", frozenset({"memory_get"})) == ("memory_get")
    assert suggest_tool_name("collection_metadata", frozenset({"browse"})) is None
    assert suggest_tool_name("collection_writ", frozenset({"collection_write"})) == (
        "collection_write"
    )
    # skill_read (#2305) resolves to skill_get the same way.
    assert suggest_tool_name("skill_read", frozenset({"skill_get"})) == "skill_get"
    assert suggest_tool_name("skill_read", frozenset({"browse"})) is None


def test_suggest_tool_name_finds_a_real_name_wrapped_inside_a_malformed_one():
    """A malformed emission often wraps a real tool name inside namespacing
    (``functions.done``) or a stray argument fragment (``browse(extract: the
    price``) — neither a rename nor a spelling match, but exactly one available
    name sits inside it as a whole token once split on non-word characters (#2510).

    The call itself still goes REJECTED (principles.md §2 item 7): this only
    changes what the did-you-mean names, never what the executor runs."""
    assert suggest_tool_name("functions.done", frozenset({"done", "browse"})) == "done"
    assert suggest_tool_name("browse(extract: the price", frozenset({"done", "browse"})) == "browse"
    # Two available names both sit inside the requested one — picking between them
    # would be a guess, so the containment leg stays silent.
    assert suggest_tool_name("functions.browse.done", frozenset({"done", "browse"})) is None
    # A name with no real tool token inside it, and nothing close by spelling,
    # still gets nothing.
    assert suggest_tool_name("functions.nonexistent", frozenset({"done", "browse"})) is None


def test_suggest_tool_name_containment_leg_never_shadows_the_earlier_legs():
    """The containment leg is last: a retired name still resolves through the
    rename table, and an ordinary typo still resolves through string distance,
    even when the requested name would also satisfy the containment leg."""
    # "skill_read" IS a retired name (resolves to skill_get) AND contains no
    # available name as a token — the rename leg answers first either way.
    assert suggest_tool_name("skill_read", frozenset({"skill_get"})) == "skill_get"
    # "memory_gt" is a plain typo of "memory_get" — the string leg answers it
    # before the containment leg is ever consulted.
    assert suggest_tool_name("memory_gt", frozenset({"memory_get"})) == "memory_get"
