"""Content-validity primitives — pure text predicates, no DB / no model.

The one home for "is this text usable?" rules, kept dependency-light (the
standard library plus the ``constants`` leaf) so every layer can import it
without dragging in the database or agent packages.  Two callers that must agree
share these:

  * the memory write path (``Collection.write`` / the ``exists`` probe) rejects
    degenerate corpus content via :func:`degenerate_reason` — a SUBSTRING poison
    check (a '…'/'.'/'?' collapse run anywhere corrupts a stored entry);
  * the ``send_message`` tool's ``args_model`` validator AND the run-health
    classifier's ``⚠ HALF-FORMED SEND`` flag both gate on
    :func:`half_formed_send_reason` — one definition for what Penny refuses to
    send and what she flags as a regression.  This judges the message AS A WHOLE
    (blank / bare-URL / bail-out / unfinished-or-truncated TAIL), NOT on a substring
    hit, so a substantive message that merely quotes a degenerate fragment mid-text
    is delivered.  Catching an in-flight collapse in the model's own output stays
    the agent-loop reroll guard's job (``is_degenerate_run`` in ``agents/base.py``).

It is also where the agent loop's **invalid-draw** detectors live — the
call-shaped-text family (:func:`is_call_as_text_bail`, :func:`is_embedded_call_as_text`,
:func:`is_done_json_bail`, :func:`is_call_fragment_reply`) and the say-nothing draw
(:func:`is_empty_draw`, #1937), beside the transport artifacts (:func:`is_degenerate_run`,
:func:`has_leaked_harmony_envelope`, :func:`is_degenerate_tool_name`).  Since #1839
all of them feed ONE mechanism — ``Agent._unusable_output_condition``'s
discard-and-reroll — so they belong in one leaf rather than split between here and
the validator chain.  :func:`strip_think_tags` sits with them because every reader
that asks "what did this draw actually SAY?" must normalise it the same way.

Living here (rather than inside ``database/memory/_similarity``) is what lets
``tools/models.py`` import :func:`half_formed_send_reason` without triggering the
``penny.database`` package import (which would close an import cycle back through
``penny.agents``).  The ``database.memory`` package re-exports these names from
here, so its public import surface is unchanged.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Collection

from penny.constants import PennyConstants

_WORD_TOKEN_RE = re.compile(r"\w+")

# A collection's extraction_prompt drives the collector each cycle; one too short
# to carry a numbered recipe leaves the collector with nothing to do.
EXTRACTION_PROMPT_MIN_CHARS = 25

# Matches content that is a bare URL with no surrounding description.
_BARE_URL_RE = re.compile(r"^https?://\S+$")

# LLM bail-out phrases that produce useless knowledge entries.
_WRITE_BAILOUT_PHRASES: frozenset[str] = frozenset(
    {
        "not sure",
        "i'm not sure",
        "i am not sure",
        "i cannot help with that",
        "i can't help with that",
        "i don't know",
        "i do not know",
        "n/a",
        "no information",
        "no information available",
        "unable to summarize",
        "unable to provide a summary",
        "no summary available",
        "no content available",
        "content not available",
        "page not available",
        "page not found",
        "we can't find that page",
        "we cannot find that page",
        "content unavailable",
        "access denied",
        "error",
    }
)

# A message that TRAILS OFF at its end into a run of dots followed by
# question/exclamation spam with no closing clause — the fingerprint of a
# half-formed generation.  The real case this targets: a notifier cycle that sent
# "Hi there! ......???" before the actual notification.  Deliberately narrow (≥3
# dots immediately followed by ≥2 ?/!) so legitimate punctuation ("Wait... what?!",
# "Hmm...?") is never caught — and TAIL-ANCHORED (the run must END the message,
# modulo a closing quote/paren and trailing whitespace), because "unfinished" is a
# whole-message property.  An identical run EMBEDDED mid-message (`she sent "Hi
# there! ......???" then the real one`) is NOT an unfinished send — it's a
# substantive message that quotes the fragment (a message reporting or quoting a
# bad send it observed is the canonical case).  Catching an embedded collapse run
# in the model's OWN output is the agent-loop reroll guard's job (`is_degenerate_run`
# on raw output, in `agents/base.py`); this predicate only judges whether the
# OUTGOING message is itself a complete one.
_UNFINISHED_FRAGMENT_RE = re.compile(r"""\.{3,}\s*[?!]{2,}\s*["'”’)\]]*\s*$""")

# Separator characters gpt-oss threads through a degeneration collapse — the run
# is rarely pure punctuation; it's laced with the "smart typography" the model
# slides into just before it loses coherence: NBSP / narrow-NBSP (both already
# matched by `\s`), plus zero-width space, soft hyphen, and the hyphen/dash family
# (spelled as escapes so the source carries no invisible bytes).
_DEGEN_SEP = r"\s\u200b\u00ad\u2010\u2011\u2013\u2014"

# The fingerprint of a gpt-oss degeneration collapse — a run of `.` / `…` / `?`
# (optionally `!`) that the model emits when it "gives up" mid-generation on a
# large context, e.g.  "...??…?..?????"  or  "New … … … … …".  This is the
# *poison*: it corrupts the tool-call argument or message it lands in, and once
# fed back into the conversation it degrades the next step too (measured ~4×
# elevated collapse once a step has gone bad).  A union of high-precision
# sub-patterns, each individually rare in real text — tuned against the prompt
# log corpus for zero false positives on legitimate punctuation ("Wait... what?!",
# "Really...?", "to be continued…", code "[1, 2, 3, ...]"):
#   1. 3+ ellipsis chars in a row (nobody types "………")
#   2. two ellipsis-runs bridged only by punctuation/spacing ("… … …", "...??..")
#   3. two punctuation clusters separated only by spacing/dashes ("?? ..??")
#   4. a mixed cluster of ≥2 dots/ellipses AND ≥2 ?/! ("..??", "...???")
#   5. one long ≥5-char run of mixed `.`/`…`/`?`
_DEGENERATE_RUN_RE = re.compile(
    r"…{3,}"
    r"|(?:…|\.\.\.)[" + _DEGEN_SEP + r"?!.]*(?:…|\.\.\.)"
    r"|[.?!]{2,}[" + _DEGEN_SEP + r"]+[.?!…]{2,}"
    r"|[.…]{2,}[?!]{2,}|[?!]{2,}[.…]{2,}"
    r"|[.…?]{5,}"
)


def is_degenerate_run(content: str) -> bool:
    """True if ``content`` contains a gpt-oss degeneration-collapse run.

    The shared detector for the *poison* fingerprint, used at every gate that
    keeps it from spreading: the agent loop discards + re-rolls model output that
    matches (never appending it — that feeds the collapse back in), and the corpus
    write / send gates (via :func:`degenerate_reason`) refuse content that carries
    it so no poison reaches ``memory_entry`` / ``messagelog``.  Substring match —
    a run anywhere in otherwise-wordful text ("Delivered deliver...???") still
    counts, because that entry/message is corrupt.
    """
    return bool(_DEGENERATE_RUN_RE.search(content))


# The Harmony (gpt-oss's native format) tool-call envelope, leaked into
# ``message.content`` as literal text.  Stock Ollama parses the envelope into
# structured ``tool_calls``, but some remote OpenAI-compatible backends serving
# gpt-oss fail to and leak the whole call as prose, e.g.
#   "<|start|>assistant<|channel|>analysis to=functions.browse code<|message|><|call|>"
# with ``tool_calls`` empty — so a chat text turn finalizes the raw envelope as the
# reply and delivers it to the user verbatim.  The pipe-bracketed control tokens
# (``<|start|>`` … ``<|call|>``) and the ``to=functions.`` recipient marker are
# Harmony sentinels that never occur in legitimate prose or code, so any hit means
# the call leaked and the output is unusable (the real call was never routed through
# the tool-call channel).  Sibling of
# :func:`penny.llm.models.strip_harmony_control_tokens`, which repairs a leak in the
# tool-call NAME field; this catches the whole call leaking into content, which the
# name path never sees.  Kept high-precision (specific token names, not a bare
# ``<|``) for zero false positives, the same corpus discipline as the collapse regex.
_HARMONY_ENVELOPE_RE = re.compile(
    r"<\|(?:start|end|channel|message|call|constrain|return)\|>|to=functions\."
)


# A reply that is nothing but an empty JSON object — ``{}`` / ``{ }`` / ``{}\n``.
# Anchored to the WHOLE (already lstripped) reply, so a message that merely mentions
# ``{}`` mid-prose ("an empty object is `{}`") never matches — only a reply that IS
# the bare object, which Penny's voice never sends.
_EMPTY_OBJECT_REPLY_RE = re.compile(r"\{\s*\}\s*$")


def is_call_fragment_reply(content: str) -> bool:
    """True if a would-be user-facing reply is a bare JSON-ish call fragment.

    The observed leak (#1570 field audit): the model emits something like
    ``{"memory":"rip? wait we need …"}`` as its final TEXT — a mangled slice of a
    tool call's argument object — and, being neither the full call envelope
    (:func:`is_call_as_text_bail`'s shape) nor a punctuation collapse, it sailed
    through every guard and was SENT to the user verbatim.  The agent-loop
    reroll guard treats it like the other unusable-output conditions: discard
    and re-draw on the unchanged context.

    Two fragment shapes match, both anchored to the WHOLE stripped reply so a
    reply that merely CONTAINS a quoted/embedded snippet mid-prose never does
    (zero-false-positive discipline — Penny's voice never SENDS a bare JSON
    object; a reply *about* JSON carries prose around it):

      * an **empty JSON object** — ``{}`` / ``{ }`` / ``{}\\n`` (#1732): the
        degenerate tail of a forced tool-call attempt that leaked into the text
        channel, the shape the #1731 nudge-loop spiral terminated in, which the old
        ``{"`` gate missed (an empty object has no string key); and
      * a **fragment with a string key** — the stripped text OPENS with ``{"`` but
        is NOT the full call ENVELOPE (both ``"name"`` and ``"arguments"``
        present).  The envelope is excluded here because
        :func:`is_call_as_text_bail` owns it — one condition per shape, so the
        rerolled draw is logged under the condition that actually describes it.
    """
    stripped = content.lstrip()
    if _EMPTY_OBJECT_REPLY_RE.match(stripped):
        return True
    if not stripped.startswith('{"'):
        return False
    return not ('"name"' in stripped and '"arguments"' in stripped)


# The tool-call envelope the model also emits as TEXT: ``{"name": …, "arguments": {…}}``.
_CALL_ENVELOPE_KEYS = frozenset({"name", "arguments"})

# An inline code span — text between single backticks, never crossing a newline
# (CommonMark inline code is one line; a backtick pair that spans lines is stray
# punctuation, not a code span).  A FENCED ``` block is a different shape and never
# matches this pattern, so it stays checked.
_INLINE_CODE_SPAN_RE = re.compile(r"`[^`\n]*`")


def _strip_inline_code(content: str) -> str:
    """``content`` with every inline code span removed (#2855).

    A tool call quoted inside inline code — e.g. a rule proposal's "**Tool call:**
    `find(query=\"Biscuit\")`" — is a mention, not a call the model is trying to
    route through the tool channel.  The embedded-call-as-text detectors below
    scan the result of this, not the raw reply, so a quoted example never trips
    them while the same call written bare in prose, or inside a fenced ``` block
    (never touched by this pattern), still does.
    """
    return _INLINE_CODE_SPAN_RE.sub(" ", content)


def _lone_json_object(content: str) -> dict | None:
    """The parsed object when ``content`` is nothing but one JSON object, else ``None``.

    Both call-as-text detectors key on the whole reply BEING a serialized object —
    ordinary prose never is, which is what keeps a genuine answer from tripping them.
    """
    text = content.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def is_done_json_bail(content: str) -> bool:
    """True when a plain-text collector draw is really an argless ``done()`` call the
    model failed to route through the tool channel (#1569).

    The one convergent shape on Harmony backends is the named envelope —
    ``{"name": "done", "arguments": {}}`` (or ``{"name": "done"}``): a lone JSON
    object naming ``done``, with at most an ``arguments`` dict alongside.  ``done`` is
    argless, so the envelope's ``arguments`` are irrelevant — the model meant to
    finish and the call never ran.  Any other name or extra key is not this shape (a
    collector's draw is invalid either way, #1839 — this only decides which condition
    the discarded draw is logged under).
    """
    parsed = _lone_json_object(content)
    if parsed is None or not set(parsed) <= _CALL_ENVELOPE_KEYS:
        return False
    if parsed.get("name") != PennyConstants.DONE_TOOL_NAME:
        return False
    return "arguments" not in parsed or isinstance(parsed["arguments"], dict)


def is_call_as_text_bail(content: str) -> bool:
    """True when a plain-text draw is really a tool call the model failed to route
    through the tool channel — gpt-oss's Harmony call-as-text fallback.

    Unlike :func:`is_done_json_bail` this keys on the *call shape*, not one named
    tool's envelope, because chat has many tools and no ``done()``: the full
    envelope, ``{"name": "browse", "arguments": {…}}``.

    (Before #2813 a bare-args shape — ``{"queries": [...], "reasoning": "…"}`` —
    was also caught here, keyed on the framework-injected ``reasoning`` field every
    tool's schema used to carry.  That field is gone, and nothing replaces the
    signal here: a bare-args leak with no envelope and no registered tool name
    inside it (see :func:`is_embedded_call_as_text`, which keeps the name signal)
    no longer trips this detector.  It still falls through to
    :func:`is_call_fragment_reply`, which flags any ``{"…`` that isn't the full
    envelope — so the draw is still discarded and rerolled, just under a
    different, still-accurate condition.)

    The whole content must be a lone JSON object — a normal prose reply never is, so
    a genuine answer can't trip this.  That discriminator is what lets chat treat a
    call-shaped draw as invalid (#1839) while its plain conversational reply stays a
    perfectly valid terminal state.
    """
    parsed = _lone_json_object(content)
    if parsed is None:
        return False
    if set(parsed) == _CALL_ENVELOPE_KEYS:
        return isinstance(parsed.get("name"), str) and isinstance(parsed.get("arguments"), dict)
    return False


def is_embedded_call_as_text(content: str, registered_tools: Collection[str]) -> bool:
    """True when ``content`` carries a tool call written as text ANYWHERE in the
    reply (#2787) — the shape :func:`is_call_as_text_bail` misses because it only
    judges the WHOLE reply: a line of prose followed by the call object, e.g. the
    observed prod leak, one line of "On it!" then
    ``{"action": "generate_image", "action_input": "…"}``, sent to the user
    verbatim with the image never generated.

    Keyed to the agent's LIVE tool registry, never a fixed tool list (#1783's
    lesson — a plugin tool is covered without a change).  Two shapes:

      - a JSON object anywhere in the text — found by scanning for every ``{`` and
        trying ``json.JSONDecoder.raw_decode`` from there — that names one of
        ``registered_tools`` as a STRING VALUE (an exact match, never a substring:
        a value that merely *contains* a tool's name, or a sentence that mentions
        one, must not trip this); or
      - the textual call syntax a draft sometimes falls back to instead of JSON —
        the old paren form, ``registered_tool_name(`` followed by an argument name
        and ``=`` (e.g. ``generate_image(description="…")``), or gemma's brace form
        (#2863/#2867), ``registered_tool_name{`` followed by either ``}`` (argless)
        or an argument name and ``:`` (e.g. ``run_get{run: '…'}``) — both
        word-boundary anchored on the tool name so neither ever fires inside a
        longer word (``rebrowse(`` is not ``browse(``).

    (Before #2813 a JSON object carrying only the framework-injected ``reasoning``
    key, with no tool name anywhere in it, also tripped this.  That field is gone
    and nothing replaces the signal — see :func:`is_call_as_text_bail` for what
    still catches a bare-args leak with no name in it.)

    Both shapes scan with inline code spans stripped first (#2855, via
    :func:`_strip_inline_code`), so a call QUOTED in inline code — a rule
    proposal's own worked example — reads as a mention rather than a leak; a bare
    call in prose, or one inside a fenced ``` block, is untouched by the strip and
    still trips this.
    """
    return _has_embedded_call_object(content, registered_tools) or _has_textual_call_syntax(
        content, registered_tools
    )


def call_as_text_tool_name(content: str, registered_tools: Collection[str]) -> str | None:
    """The tool name a call-as-text draft named, when the shape carries one (#2790)
    — for the retry note, which names the tool rather than speaking only of "a call".

    Mirrors :func:`is_call_as_text_bail` and :func:`is_embedded_call_as_text`'s own
    detection without changing either's signature or behaviour: the whole-reply
    envelope's ``"name"`` field, the textual call syntax's matched name, or a
    registered tool named as a string value in an embedded object — tried in that
    order, the whole-envelope shape first since it is the one unambiguous "name".
    ``None`` when nothing call-shaped is found at all.  (Every shape that DOES
    carry a name now also HAS one, per #2813's removal of the nameless
    reasoning-only signal — see :func:`is_call_as_text_bail` and
    :func:`is_embedded_call_as_text` — so ``None`` is no longer reachable through
    a genuinely call-shaped draft; callers still handle it as a fallback.)

    The two mirrored legs scan with inline code stripped (#2855), the same text
    :func:`is_embedded_call_as_text` matched against — so the retry note always
    names the tool the check actually flagged, never one read off the raw reply."""
    whole = _lone_json_object(content)
    if whole is not None and isinstance(whole.get("name"), str):
        return whole["name"]
    name = _textual_call_syntax_name(content, registered_tools)
    if name is not None:
        return name
    return _embedded_call_object_tool_name(content, registered_tools)


# The textual call syntax a draft sometimes falls back to instead of JSON,
# right after a registered tool's name — the old paren form (an argument name
# then ``=``), or gemma's brace form (#2863/#2867): either an empty ``{}``
# (argless, e.g. ``done{}``), an argument name then ``:`` (e.g.
# ``run_get{run: '…'}``), or a positional argument (a quote or ``<`` for list
# syntax, e.g. ``memory_get{'film-finds'}``).  One constant shared by
# :func:`_has_textual_call_syntax` and :func:`_textual_call_syntax_name` so
# their patterns can't drift apart.
_TEXTUAL_CALL_SYNTAX_SUFFIX = r"(?:\(\s*[A-Za-z_]\w*\s*=|\{\s*(?:\}|[A-Za-z_]\w*\s*:|['\"<]))"


def _textual_call_syntax_name(content: str, registered_tools: Collection[str]) -> str | None:
    """The registered tool name matched by :func:`_has_textual_call_syntax`'s own
    pattern, or ``None`` — same anchoring (including the inline-code strip), returning
    the match instead of a bool."""
    if not registered_tools:
        return None
    names = "|".join(re.escape(name) for name in registered_tools)
    pattern = re.compile(rf"\b({names}){_TEXTUAL_CALL_SYNTAX_SUFFIX}")
    match = pattern.search(_strip_inline_code(content))
    return match.group(1) if match else None


def _embedded_call_object_tool_name(content: str, registered_tools: Collection[str]) -> str | None:
    """The registered tool a JSON object embedded anywhere in ``content`` names as a
    string value — see :func:`_has_embedded_call_object` (including the inline-code
    strip), returning the name instead of a bool."""
    stripped = _strip_inline_code(content)
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped, index)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        for value in parsed.values():
            if isinstance(value, str) and value in registered_tools:
                return value
    return None


def _has_embedded_call_object(content: str, registered_tools: Collection[str]) -> bool:
    """A JSON object anywhere in ``content`` that names a registered tool as a
    string value — see :func:`is_embedded_call_as_text`.

    Scans the content with inline code spans stripped (#2855) — a call QUOTED in
    inline code (e.g. a rule proposal's own example) is a mention, not a leak; a
    fenced ``` block is a different shape and stays checked.
    """
    stripped = _strip_inline_code(content)
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped, index)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and _is_call_shaped_object(parsed, registered_tools):
            return True
    return False


def _is_call_shaped_object(parsed: dict, registered_tools: Collection[str]) -> bool:
    """A parsed JSON object names a registered tool as a string VALUE (an exact
    match — never a substring hit).

    Before #2813 this also matched a bare ``reasoning`` key, every real call's
    serialized arguments having carried one.  That field is gone and nothing
    replaces the signal: a bare-args object with no registered tool name
    anywhere in it is no longer call-shaped by this predicate (though
    :func:`is_call_fragment_reply` still catches most such leaks — see
    :func:`is_call_as_text_bail`)."""
    return any(isinstance(value, str) and value in registered_tools for value in parsed.values())


def _has_textual_call_syntax(content: str, registered_tools: Collection[str]) -> bool:
    """``registered_tool_name(`` followed by an argument name and ``=`` (the old
    paren form), or ``registered_tool_name{`` followed by either ``}`` (argless)
    or an argument name and ``:`` (gemma's brace form, #2863/#2867 — e.g.
    ``run_get{run: '…'}``) — the textual call syntax some drafts fall back to
    instead of JSON.  One compiled alternation over every registered name, each
    ``re.escape``d and ``\\b``-anchored so a longer word ending in a tool's name
    (``rebrowse(``) never matches, built from :data:`_TEXTUAL_CALL_SYNTAX_SUFFIX`
    — the one pattern :func:`_textual_call_syntax_name` also matches against, so
    the two can't drift apart.

    Scans with inline code spans stripped (#2855), so a call quoted in inline
    code (e.g. a rule proposal's own example) reads as a mention, not a leak; a
    bare call in prose, or one inside a fenced ``` block, still matches."""
    if not registered_tools:
        return False
    names = "|".join(re.escape(name) for name in registered_tools)
    pattern = re.compile(rf"\b(?:{names}){_TEXTUAL_CALL_SYNTAX_SUFFIX}")
    return bool(pattern.search(_strip_inline_code(content)))


def has_leaked_harmony_envelope(content: str) -> bool:
    """True if ``content`` carries a leaked Harmony tool-call envelope.

    A backend that doesn't fully parse gpt-oss's Harmony format leaks the tool
    call into the text channel as literal control-token prose instead of parsing
    it into ``tool_calls``.  The agent-loop reroll guard treats that exactly like
    a degeneration collapse — discard the poisoned output and re-draw on the
    unchanged context, since the leak is intermittent (a fresh draw usually comes
    back clean) — rather than reconstructing the call from the envelope grammar.
    See :data:`_HARMONY_ENVELOPE_RE`.
    """
    return bool(_HARMONY_ENVELOPE_RE.search(content))


# Matches <think>...</think> blocks emitted inline by some models (e.g. DeepSeek-R1,
# Qwen3) instead of on the response's own thinking field.
_THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def strip_think_tags(content: str) -> tuple[str, str | None]:
    """Strip ``<think>…</think>`` blocks from content.

    Returns ``(cleaned_content, extracted_thinking)`` where ``extracted_thinking``
    is the concatenated text of every stripped block.  It lives beside the
    invalid-draw predicates because it is the shared NORMALISATION every reader of a
    draw's substance applies first — the empty-draw guard, the refusal and
    hallucinated-URL validators, and the loop's final-answer build — so a draw that is
    nothing but reasoning reads the same way at all four.
    """
    thinking_parts: list[str] = []

    def _collect(match: re.Match) -> str:
        thinking_parts.append(match.group(1).strip())
        return ""

    cleaned = _THINK_TAG_PATTERN.sub(_collect, content).strip()
    extracted = "\n\n".join(thinking_parts) if thinking_parts else None
    return cleaned, extracted


def is_empty_draw(content: str) -> bool:
    """True when a call-less draw says nothing — blank, separators-only, or a bare
    ``<think>`` block with no body (#1937).

    The say-nothing member of the invalid-draw family: chat's valid terminal state is
    a reply, and a draw carrying fewer than ``MIN_RESPONSE_LETTERS`` letters is not one
    — a bare ``\\n\\n---`` or a lone emoji would otherwise be finalized as Penny's
    answer.  Counting LETTERS (not word tokens) is what keeps a markdown separator and
    a punctuation run in the same class as the empty string.

    Ordered LAST in chat's conditions: the bare ``{}`` tail of a forced call attempt has
    no letters either, and ``CALL_FRAGMENT_REPLY`` is the honest name for that shape
    (#1732), so it is tried first and this one names what is left.
    """
    effective, _ = strip_think_tags(content.strip())
    letters = sum(1 for character in effective if character.isalpha())
    return letters < PennyConstants.MIN_RESPONSE_LETTERS


# The collapse fingerprint scoped to a tool-call NAME — any `?` / `!` / `…`
# anywhere, or a run of 2+ dots.  A name is a short identifier, so the prose
# heuristics in `_DEGENERATE_RUN_RE` (which need two clusters or a 5-char run to
# avoid flagging legitimate punctuation) are too lax here: `funcs.done?` or
# `read_simpar?` carries a single collapse character and is already poison.
# Deliberately CONSERVATIVE about everything else — a single interior dot
# (`functions.done`, a namespacing near-miss) and non-collapse junk such as a
# Harmony-wrapped valid name (`done<|channel|>commentary`, a repair case for the
# Harmony-token strip, not poison) must NOT match, so those calls keep the
# executor's helpful tool-not-found path ("Did you mean X?").
_DEGENERATE_TOOL_NAME_RE = re.compile(r"[?!…]|\.{2,}")


def is_degenerate_tool_name(name: str) -> bool:
    """True if a tool-call NAME carries the degeneration-collapse fingerprint.

    The punctuation collapse (:func:`is_degenerate_run`) also lands in the tool
    NAME field (``Functions?????``, ``funcs.done?``, ``read_simpar?``); the
    content-level detector misses those, and feeding back a tool-not-found error
    keeps the poison in context.  The agent loop treats an *unregistered* name
    that matches this as degenerate output — discard-and-reroll, same as content
    poison.  Only collapse characters match (``?``/``!``/``…``/dot runs): a
    plausible near-miss identifier or a Harmony-token-wrapped valid name stays on
    the tool-not-found path, where the "Did you mean X?" hint can repair it.
    """
    return bool(_DEGENERATE_TOOL_NAME_RE.search(name))


# A message cut off mid-thought on an ellipsis TAIL — one-or-more "…" or 3+ ASCII
# dots, optionally a single trailing ?/!/. — the model self-truncating.  Real
# failures: "...the original …", "all-time-best ‑ …?", "Hello world...".  A
# conversational "…" with text after it ("Anyway… 🤓") isn't the tail, so it's safe.
_TRUNCATION_TAIL_RE = re.compile(r"(?:…+|\.{3,})\s*[?!.]?\s*$")


def is_unfinished_fragment(content: str) -> bool:
    """True if ``content`` ENDS in an ellipsis + ?/! spam tail — a half-formed message.

    Complements :func:`degenerate_reason` (which only catches blank / bare-URL /
    bail-out content): a message can carry word tokens yet still trail off into a
    half-formed tail a user should never have received.  Tail-anchored — the run
    must end the message — so an identical fragment EMBEDDED mid-message (a
    deliberate quote of observed evidence) is not flagged; an in-flight collapse
    landing mid-output is the agent-loop reroll guard's concern, not this send
    gate's.  See :data:`_UNFINISHED_FRAGMENT_RE`.
    """
    return bool(_UNFINISHED_FRAGMENT_RE.search(content))


def is_truncated(content: str) -> bool:
    """True if ``content`` ends on an ellipsis tail — cut off mid-thought.

    The aggressive tail check (catches a lone "Hmm...?") — appropriate for an
    OUTBOUND message Penny is about to send, where a trailing-ellipsis fragment is
    junk the user shouldn't receive.  Folded into :func:`half_formed_send_reason`
    so the send gate and the run-health flag share one definition of half-formed.
    """
    return bool(_TRUNCATION_TAIL_RE.search(content))


def is_blank(content: str) -> bool:
    """Return True if ``content`` carries no word tokens at all.

    The conservative "is this empty?" predicate — whitespace, punctuation, or
    ellipsis only.  Distinct from the fuller :func:`degenerate_reason` (which
    also rejects bare URLs and bail-out phrases): a blank check is safe for any
    text field, including log appends where a bare URL may be legitimate.
    """
    return not _WORD_TOKEN_RE.findall(content)


def degenerate_reason(content: str) -> str | None:
    """Return a rejection reason if ``content`` is too degenerate to store.

    Catches empty/pure-punctuation strings, bare URLs, and known LLM
    bail-out phrases.  Returns ``None`` when content is acceptable.
    Applied at collection write time to keep the corpus clean.
    """
    stripped = content.strip()
    if is_blank(stripped):
        return "content has no word tokens (empty, punctuation, or ellipsis only)"
    if _BARE_URL_RE.match(stripped):
        return "content is a bare URL with no descriptive text"
    if stripped.lower() in _WRITE_BAILOUT_PHRASES:
        return f"content matches a known LLM bail-out phrase: {stripped!r}"
    if is_degenerate_run(stripped):
        return "content carries a degenerate punctuation run (a '…'/'.'/'?' collapse)"
    return None


def _empty_send_reason(stripped: str) -> str | None:
    """Actionable reason a send has no real content — blank, bare URL, or a
    bail-out phrase — or ``None``.  Each names what's wrong AND the next move.

    This is the send gate's content-shape check.  It deliberately does NOT include
    the SUBSTRING degenerate-run check that :func:`degenerate_reason` (the corpus
    write gate) carries: an embedded '…'/'.'/'?' collapse run inside an otherwise
    substantive, deliberate message (a message quoting a bad send it observed) is a
    real message the user should receive — catching a genuine
    in-flight collapse in the model's OWN output is the agent-loop reroll guard's
    job (`is_degenerate_run`).  Only a WHOLE-message half-formed shape (blank /
    bare-URL / bail-out here, plus the tail checks in the caller) is refused.
    """
    if is_blank(stripped):
        return (
            "your message has no real content — it is empty or only "
            "punctuation/ellipsis.  Compose a complete, substantive message and "
            "send it again."
        )
    if _BARE_URL_RE.match(stripped):
        return (
            "your message is a bare URL with no surrounding text.  Add a sentence "
            "describing the link, then send it again."
        )
    if stripped.lower() in _WRITE_BAILOUT_PHRASES:
        return (
            f"your message is a bail-out phrase ({stripped!r}), not a real reply.  "
            "Compose the actual message you meant to send."
        )
    return None


def half_formed_send_reason(content: str) -> str | None:
    """Return why ``content`` is not a complete message a user should receive, or None.

    The single definition of a "half-formed send", shared by the ``send_message``
    tool's pre-send gate (which refuses it before delivery) and the run-health
    classifier's after-the-fact ``⚠ HALF-FORMED SEND`` flag — so what Penny
    refuses to send and what she flags as a regression are one rule.  Judged on the
    message AS A WHOLE, not on a substring hit: a blank / bare-URL / bail-out body
    (via :func:`_empty_send_reason`), an unfinished ellipsis+?/! TAIL
    (``"Hi there! ......???"``, via :func:`is_unfinished_fragment`), or an
    ellipsis-truncated TAIL (``"...the original …"``, via :func:`is_truncated`).  A
    substantive message that merely EMBEDS such a fragment mid-text (a message
    quoting a degenerate send it observed) is a complete message and
    passes — the substring poison check stays the corpus write gate's
    (:func:`degenerate_reason`) and the agent-loop reroll guard's concern.  Each
    reason is actionable: it names the specific defect and the next move.
    """
    stripped = content.strip()
    if reason := _empty_send_reason(stripped):
        return reason
    if is_unfinished_fragment(stripped):
        return (
            "your message trails off at the end into a '.'/'…' run and '?'/'!' spam "
            "with no closing clause — a half-formed, degenerate tail.  Finish the "
            "sentence on a complete clause (no placeholder punctuation) and send it "
            "again."
        )
    if is_truncated(stripped):
        return (
            "your message ends on an ellipsis ('…' or '...'), cut off mid-thought.  "
            "Finish the sentence and send the complete message."
        )
    return None


def check_extraction_prompt(prompt: str | None) -> str | None:
    """Return an error string if prompt is set but too short, else None.

    The string-returning form, used where a prompt is read from a stored row and
    its absence/shortness only *gates* an action (the collector's readiness
    check) rather than rejecting a tool call.  There is no arg-validator twin: the
    model surface takes no ``extraction_prompt`` argument (#1658 — a routine is only
    ever a render of a demonstrated skill), so nothing validates one on the way in.
    """
    if prompt is None or len(prompt) >= EXTRACTION_PROMPT_MIN_CHARS:
        return None
    return (
        f"extraction_prompt is too short ({len(prompt)} chars — minimum "
        f"{EXTRACTION_PROMPT_MIN_CHARS}).  Provide a full numbered-step prompt "
        f"(see the collection_configure description for the required shape)."
    )


def check_description(description: str) -> str | None:
    """Return an error string if a required description is blank, else None.

    The description doubles as the stage-1 routing anchor, so a blank one
    would create a memory that can never be matched.  Reject it loudly rather
    than embedding an empty string.
    """
    if is_blank(description):
        return "description cannot be blank — provide a content-reflective one-line summary."
    return None


# Tool names this project has CORRECTED, each resolved to the name in use today (#1944).
# Read off the migrations that rewrote stored ``extraction_prompt``s, and resolved
# TRANSITIVELY — a name corrected twice points at where it ended up, never at an
# intermediate that is also gone.  Two histories land here, and they are one question at
# a miss: names that were RENAMED out from under callers —
#   0059       collection_metadata → memory_metadata
#   0057       log_read_next → log_read, log_read_recent → log_read
#   0059       read_latest → collection_read_latest
#   #1631      collection_update → collection_set  (create + update collapsed onto one
#              front door; migrations 0037/0041 rewrote this name inside COLLECTOR
#              prompts to ``update_entry`` instead, which was a surface-scope correction
#              for a tool a cycle cannot call, not a rename — so it is deliberately not
#              what this table says)
# and names that were never tools at all, recurring enough for a migration to sweep them:
#   0039/0059  read_last → read_latest → collection_read_latest
#   0040/0057  log_read_log → log_read_next → log_read
#   0111       collection_read_latest → collection_list,
#              collection_delete_entry → collection_remove
#   #2812      collection_sample → collection_list (order='random' is the same draw);
#              collection_unarchive → collection_archive (archived=False is the same
#              restore); notifications_mute / notifications_unmute → notifications_set
#              (on=False / on=True) — three tool-PAIR folds, each a retired name
#              resolving to the one tool that replaced it plus the argument that
#              recovers the retired direction
# Why a table rather than more string distance: what a name became is a fact about this
# project's history, and the two names need not look alike.  ``collection_metadata`` is
# the measured case — the model guessed the retired name, and the string leg answered
# with ``collection_set`` (a near neighbour by characters, a different tool entirely)
# while the tool it wanted, ``memory_metadata``, is the one this table names.  Nothing
# here is authoritative about what EXISTS: a suggestion is only made when the current
# name is actually on the surface being suggested from, so an entry whose target that
# surface lacks falls through to the string leg like any other miss.
LEGACY_TOOL_RENAMES: dict[str, str] = {
    "collection_metadata": "memory_get",
    "collection_update": "collection_configure",
    "collection_set": "collection_configure",
    "memory_metadata": "memory_get",
    "log_read_log": "log_read",
    "log_read_next": "log_read",
    "log_read_recent": "log_read",
    # Each of these two once pointed at collection_read_latest; #2307 renamed THAT
    # tool too, so both are repointed straight at the name it ended up with —
    # resolution is one hop, never a chain.
    "read_last": "collection_list",
    "read_latest": "collection_list",
    "collection_read_latest": "collection_list",
    "collection_delete_entry": "collection_remove",
    # A value stored under a key is written with the collection's one write call.
    "update_entry": "collection_write",
    # collection_read_random pointed at collection_sample, which #2812 folded into
    # collection_list(order='random') — repointed straight at the tool it ended up
    # with, one hop, never a chain.
    "collection_read_random": "collection_list",
    "collection_sample": "collection_list",
    # The archive/unarchive pair folded onto one tool (#2812): both retired names
    # point at the tool that replaced them, which takes the direction as `archived`.
    "collection_unarchive": "collection_archive",
    # The mute/unmute pair folded onto one tool (#2812): both retired names point
    # at the tool that replaced them, which takes the direction as `on`.
    "notifications_mute": "notifications_set",
    "notifications_unmute": "notifications_set",
    # A key either reads back something or it doesn't — collection_get answers that.
    "exists": "collection_get",
    # Reading one skill by name is skill_get (#2305).
    "skill_read": "skill_get",
    # A source's runs are listed by run_list, one is read by run_get; the retired
    # per-collector verb resolved through the browse it overlapped.
    "read_run_calls": "run_list",
    "get_event": "run_get",
    "collector_run_history": "run_list",
}


def suggest_tool_name(requested: str, available: Collection[str]) -> str | None:
    """The tool ``requested`` most likely meant, or ``None`` when nothing is close.

    Three legs, MEANING first, then spelling, then containment (#2510).  A name this
    project retired is resolved through :data:`LEGACY_TOOL_RENAMES` to the tool that
    replaced it — a historical fact no character comparison can recover — and only
    when that replacement is on ``available``, so the suggestion always names something
    callable.  Next the string leg (``difflib`` over the available names, gated on
    ``PennyConstants.DID_YOU_MEAN_STRING_CUTOFF``), which is what catches an ordinary
    typo and stays silent on a genuinely unrelated guess.  Last, a containment leg: a
    malformed emission (``functions.done``, ``browse(extract: the price``) often wraps
    a real tool name inside namespacing or a stray argument fragment — split on
    anything that isn't a letter, digit or underscore and, when exactly one available
    name equals one of those tokens, suggest it; two or more matches suggest nothing,
    since picking between them would be a guess.

    Shared by the two surfaces that answer an unknown tool name — the executor's live
    tool-not-found result and the stored-``extraction_prompt`` check below — so the two
    cannot drift apart.
    """
    renamed = LEGACY_TOOL_RENAMES.get(requested)
    if renamed is not None and renamed in available:
        return renamed
    close = difflib.get_close_matches(
        requested, available, n=1, cutoff=PennyConstants.DID_YOU_MEAN_STRING_CUTOFF
    )
    if close:
        return close[0]
    tokens = set(_WORD_TOKEN_RE.findall(requested))
    contained = [name for name in available if name in tokens]
    return contained[0] if len(contained) == 1 else None


def require_non_blank_description(value: str) -> str:
    """Arg-validator: raise if a required ``description`` is blank."""
    if error := check_description(value):
        raise ValueError(error)
    return value


def require_non_blank_log_content(value: str) -> str:
    """Arg-validator: raise if a log append's ``content`` is blank.

    Blank-only is refused (a bare URL is still a valid log entry, unlike the
    collection corpus filter), so nothing empty joins an append-only stream.
    """
    if is_blank(value):
        raise ValueError("log entry content is blank — provide non-empty text.")
    return value


def require_non_degenerate_content(value: str) -> str:
    """Arg-validator: raise if a collection entry's replacement ``content`` is
    degenerate (blank, bare URL, or a known bail-out phrase).

    Mirrors the corpus write filter (:func:`degenerate_reason`) at the tool
    surface, pointing the model at ``collection_remove`` if removal was the
    intent rather than a replacement.
    """
    if reason := degenerate_reason(value):
        raise ValueError(
            f"replacement content rejected — {reason}. Provide the full replacement "
            f"text, or use collection_remove if you meant to remove the entry."
        )
    return value
