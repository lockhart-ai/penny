"""Model-refusal phrase detection.

Shared by the agent loop (which retries when the model emits a refusal
mid-conversation) and ``SendMessageTool`` (which refuses to dispatch a
refusal as if it were the user-facing reply).
"""

from __future__ import annotations

import re

# Substrings that indicate the model has produced a refusal rather than
# substantive content.  Lowercase for case-insensitive matching.
#
# An apology ("I'm sorry", "I apologize") is deliberately NOT here: it's
# just as often an honest report of a real failure ("I'm sorry, the image
# server didn't answer") as a refusal, and unlike "I can't"/"as an AI" an
# apology carries no refusal of its own — it only READS as one next to a
# phrase that does.  Dropping it means a short, honest "I'm sorry, X
# failed." is never withheld or discarded as a refusal.
REFUSAL_PHRASES = (
    "i can't",
    "i cannot",
    "i'm unable",
    "i am unable",
    "as an ai",
    "as a language model",
)

# A genuine refusal is one short, contentless sentence — "I'm sorry, but I
# can't help with that." is under 40 characters, and the longest templated
# refusal phrasings we've seen stay well under 150.  Past this many
# characters the reply has moved on to real content, whether it opens with
# an apology for an honest tool failure that goes on to explain what
# happened, or hedges mid-answer ("...but I can suggest...") — neither is
# a refusal, so the length bound keeps them from being discarded and
# redrawn for nothing.
MAX_REFUSAL_LENGTH = 160

# Markdown emphasis/heading/quote markers and common emoji blocks a model
# sometimes opens a reply with, stripped before the first sentence is read
# so decoration never hides (or fakes) an opening refusal phrase.
_LEADING_DECORATION = re.compile(
    "^["
    r"\s*_#>`~\-\"'“”‘’"
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "←-⇿"
    "⌀-⏿"
    "️‍"
    "]+"
)

# The first sentence ends at the first ., ! or ? (followed by whitespace or
# the end of the string); a reply with no terminator at all IS its own
# first sentence.
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def _first_sentence(content: str) -> str:
    stripped = _LEADING_DECORATION.sub("", content)
    match = _SENTENCE_END.search(stripped)
    return stripped[: match.end()] if match else stripped


def is_refusal(content: str) -> bool:
    """Return True if ``content`` OPENS with a refusal phrase and is short.

    A refusal is read off the first sentence only, so a good answer that
    merely mentions "I can't fix X myself, but I can..." mid-text is never
    one; the length bound keeps a long, honest apology for a real failure
    (which goes on to say what happened) from reading as a refusal just
    because it opens politely.
    """
    if not content or len(content) > MAX_REFUSAL_LENGTH:
        return False
    first_sentence = _first_sentence(content).lower()
    return any(phrase in first_sentence for phrase in REFUSAL_PHRASES)
