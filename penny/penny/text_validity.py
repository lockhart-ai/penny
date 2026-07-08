"""Content-validity primitives — pure text predicates, no DB / no model.

The one home for "is this text usable?" rules, kept dependency-light (standard
library + ``PennyConstants`` only) so every layer can import it without dragging in
the database or agent packages.  Two callers that must agree share these:

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

Living here (rather than inside ``database/memory/_similarity``) is what lets
``tools/models.py`` import :func:`half_formed_send_reason` without triggering the
``penny.database`` package import (which would close an import cycle back through
``penny.agents``).  The ``database.memory`` package re-exports these names from
here, so its public import surface is unchanged.
"""

from __future__ import annotations

import difflib
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
# substantive message that quotes the fragment (a `quality`-collector suggestion
# reporting a bad send is the canonical case).  Catching an embedded collapse run
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
    substantive, deliberate message (a `quality` suggestion quoting the bad send it
    observed) is a real message the user should receive — catching a genuine
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
    substantive message that merely EMBEDS such a fragment mid-text (a `quality`
    suggestion quoting the degenerate send it observed) is a complete message and
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
    check) rather than rejecting a tool call.  The arg-validator form
    (:func:`require_extraction_prompt`) wraps this for the tool surface.
    """
    if prompt is None or len(prompt) >= EXTRACTION_PROMPT_MIN_CHARS:
        return None
    return (
        f"extraction_prompt is too short ({len(prompt)} chars — minimum "
        f"{EXTRACTION_PROMPT_MIN_CHARS}).  Provide a full numbered-step prompt "
        f"(see the collection_create description for the required shape)."
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


def require_extraction_prompt(value: str) -> str:
    """Arg-validator: raise if a required ``extraction_prompt`` is too short.

    The ``Annotated`` validator form of :func:`check_extraction_prompt` — the
    same rule, raised as a ``ValueError`` so the tool's ``args_model`` rejects
    the call before ``execute`` with the actionable envelope.
    """
    if error := check_extraction_prompt(value):
        raise ValueError(error)
    return value


# Canonical tool-call notation in a numbered extraction_prompt is ``tool(args)`` — a
# lowercase snake_case identifier IMMEDIATELY followed by ``(``, no space (the
# prompt-writing guide's ``N. tool(args) — purpose`` shape).  Two precision levers
# keep a prose word from reading as a call:
#   * NO space before ``(`` — English prose puts a space before an aside
#     ("one result (e.g. the first)") while a call never does ("browse(").
#   * NOT the pluralisation idiom — ``word(s)`` / ``word(es)`` / ``word(ies)``
#     ("store the url(s)", "skip duplicate(s)") is prose, not a ``done()``-style
#     empty call, so a bare short plural suffix immediately closing the paren is
#     excluded via the negative lookahead.
# Not preceded by a word char or ``.`` so a dotted/wrapped artifact
# (``functions.browse(``) isn't split into a spurious bare name.  (A no-space
# parenthetical enumeration like ``category(euro/co-op)`` is a rare residual — the
# common case is the plural suffix.)
_TOOL_CALL_RE = re.compile(r"(?<![\w.])([a-z_][a-z0-9_]*)\((?!(?:s|es|ies)\))")


def extract_tool_call_names(prompt: str) -> list[str]:
    """Return the tool-call names written in ``prompt``, in first-seen order.

    A pure text scan for the canonical ``tool(args)`` call shape (see
    :data:`_TOOL_CALL_RE`); it does not know which names are real tools — the
    caller checks membership.  Order-preserving and de-duplicated, so a repeated
    call is reported once.
    """
    seen: dict[str, None] = {}
    for match in _TOOL_CALL_RE.finditer(prompt):
        seen.setdefault(match.group(1), None)
    return list(seen)


def check_extraction_prompt_tools(prompt: str, valid_tools: Collection[str]) -> str | None:
    """Return an error naming any tool call outside ``valid_tools``, else None.

    Every ``tool(args)`` in a collector's ``extraction_prompt`` must name a tool the
    collector can actually run.  The caller supplies that surface (``valid_tools``) —
    so this stays a pure text rule that doesn't know the tool registry, and a test can
    exercise it with a small contrived set.  A fictitious call (a hallucinated
    ``extract_text(...)`` the model wrote into its own recipe) would otherwise be
    persisted into a prompt the collector later tries to run and then fail every
    cycle, so it is rejected at write time with an actionable message (the offending
    tool, a did-you-mean, and the tools that DO exist), mirroring the executor's live
    tool-not-found response.
    """
    unknown = [name for name in extract_tool_call_names(prompt) if name not in valid_tools]
    if not unknown:
        return None
    close = difflib.get_close_matches(unknown[0], valid_tools, n=1, cutoff=0.6)
    did_you_mean = f" Did you mean '{close[0]}'?" if close else ""
    offending = ", ".join(repr(name) for name in unknown)
    available = ", ".join(sorted(valid_tools))
    return (
        f"extraction_prompt calls {offending}, which a collector cannot use."
        f"{did_you_mean} Rewrite the step(s) using only these tools: {available}."
    )


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
    surface, pointing the model at ``collection_delete_entry`` if removal was the
    intent rather than a replacement.
    """
    if reason := degenerate_reason(value):
        raise ValueError(
            f"replacement content rejected — {reason}. Provide the full replacement "
            f"text, or use collection_delete_entry if you meant to remove the entry."
        )
    return value


def is_low_info(content: str) -> bool:
    """Return True if ``content`` carries less than the configured minimum word
    count and should be filtered from similarity scoring.

    The filter targets entries that geometrically dominate cosine rankings on
    short keyword anchors despite having no topical payload — empty strings,
    lone punctuation, stock greetings, bare URL fragments.  Entries that pass
    still appear in other recall paths (recent / all / read_latest); only the
    relevant-mode similarity corpus is filtered.
    """
    return len(_WORD_TOKEN_RE.findall(content)) < PennyConstants.MEMORY_RELEVANT_MIN_WORDS
