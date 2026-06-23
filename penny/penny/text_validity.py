"""Content-validity primitives — pure text predicates, no DB / no model.

The one home for "is this text usable?" rules, kept dependency-light (only ``re``
+ ``PennyConstants``) so every layer can import it without dragging in the
database or agent packages.  Two callers that must agree share these:

  * the memory write path (``Collection.write`` / the ``exists`` probe) rejects
    degenerate corpus content via :func:`degenerate_reason`;
  * the ``send_message`` tool's ``args_model`` validator AND the run-health
    classifier's ``⚠ HALF-FORMED SEND`` flag both gate on
    :func:`half_formed_send_reason` — one definition for what Penny refuses to
    send and what she flags as a regression.

Living here (rather than inside ``database/memory/_similarity``) is what lets
``tools/models.py`` import :func:`half_formed_send_reason` without triggering the
``penny.database`` package import (which would close an import cycle back through
``penny.agents``).  The ``database.memory`` package re-exports these names from
here, so its public import surface is unchanged.
"""

from __future__ import annotations

import re

from penny.constants import PennyConstants

_WORD_TOKEN_RE = re.compile(r"\w+")

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
        "no content available",
        "content not available",
        "page not available",
        "content unavailable",
        "access denied",
        "error",
    }
)

# A message that trails off into a run of dots followed by question/exclamation
# spam with no closing clause — the fingerprint of a half-formed generation.  The
# real case this targets: a notifier cycle that sent "Hi there! ......???" before
# the actual notification.  Deliberately narrow (≥3 dots immediately followed by
# ≥2 ?/!) so legitimate punctuation ("Wait... what?!", "Hmm...?") is never caught.
_UNFINISHED_FRAGMENT_RE = re.compile(r"\.{3,}\s*[?!]{2,}")

# A message cut off mid-thought on an ellipsis TAIL — one-or-more "…" or 3+ ASCII
# dots, optionally a single trailing ?/!/. — the model self-truncating.  Real
# failures: "...the original …", "all-time-best ‑ …?", "Hello world...".  A
# conversational "…" with text after it ("Anyway… 🤓") isn't the tail, so it's safe.
_TRUNCATION_TAIL_RE = re.compile(r"(?:…+|\.{3,})\s*[?!.]?\s*$")


def is_unfinished_fragment(content: str) -> bool:
    """True if ``content`` ends in ellipsis + ?/! spam — a half-formed message.

    Complements :func:`degenerate_reason` (which only catches blank / bare-URL /
    bail-out content): a message can carry word tokens yet still be an unfinished
    fragment a user should never have received.
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
    return None


def half_formed_send_reason(content: str) -> str | None:
    """Return why ``content`` is not a real message a user should receive, or None.

    The single definition of a "half-formed send", shared by the ``send_message``
    tool's pre-send gate (which refuses it before delivery) and the run-health
    classifier's after-the-fact ``⚠ HALF-FORMED SEND`` flag — so what Penny
    refuses to send and what she flags as a regression are one rule.  Combines the
    corpus content filter (blank / bare-URL / bail-out phrase, via
    :func:`degenerate_reason`), the unfinished-fragment fingerprint
    (``"Hi there! ......???"``, via :func:`is_unfinished_fragment`), and the
    ellipsis-tail truncation (``"...the original …"``, via :func:`is_truncated`).
    """
    reason = degenerate_reason(content)
    if reason is not None:
        return reason
    if is_unfinished_fragment(content):
        return "content is an unfinished fragment (ellipsis run + ?/! with no closing clause)"
    if is_truncated(content):
        return "content ends on an ellipsis ('…' or '...'), cut off mid-thought"
    return None


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
