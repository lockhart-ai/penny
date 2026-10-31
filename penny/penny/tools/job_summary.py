"""``job_summary_line`` — the ONE render of a job's one-line summary (#2495, #2685).

``reads <url> — "<instructions>"`` is the second line the self-state Jobs section
hangs under a live job's row, and the same line ``collection_configure`` refuses
when she copies it back as the job's ``prompt`` — the check and the display read
this one function, so what the tool refuses is exactly what the state shows.

The module lives in ``tools/`` because both consumers import it at the top:
``agents.self_state`` (the render) and ``tools.memory_tools`` (the refusal), and
``agents.base`` imports ``tools.memory_tools`` — a home inside ``agents/`` would
re-create the import cycle the shared function exists to end."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from penny.constants import PennyConstants

if TYPE_CHECKING:
    from penny.database.models import MemoryRow

# The first ``https?://`` URL in a job's prompt (#2495) — the page the job reads,
# rendered whole on its instructions line no matter how the prompt is capped.
_FIRST_URL_RE = re.compile(r"https?://\S+")
# A sentence boundary in a job's prompt: a period followed by whitespace (or the end)
# that neither follows a digit (a list's ``1.`` numbering belongs to its sentence, not
# to a boundary) nor sits inside one (a decimal).  The match spans only the period, so
# slicing keeps it with the sentence it ends.
_SENTENCE_END_RE = re.compile(r"(?<![0-9])\.(?=\s|$)")


def job_summary_line(row: MemoryRow) -> str:
    """``reads <url> — "<instructions>"`` — the second line the Jobs section hangs
    under a live job's row (#2495), without its two-space indent.

    The page is the FIRST ``https?://`` URL anywhere in the job's full prompt — an
    anchor, so it renders whole no matter how the instructions are capped — and the
    instructions are the prompt's own words, whole when they fit the character
    budget, otherwise the first sentence cut at a word boundary with a note naming
    the ``memory_get`` hop to the whole prompt.  The ``reads`` clause is omitted
    when the prompt names no page; the quoted instructions always render, so the
    line never disappears entirely.

    ONE render for two consumers: the Jobs section above, and
    ``collection_configure``'s refusal of the summary passed back as a ``prompt``
    (#2685) — the check and the display read the same code, so what the tool
    refuses is exactly what the state shows."""
    page = _page(row)
    instructions = _instructions(row)
    if page is None:
        return f'"{instructions}"'
    return f'reads {page} — "{instructions}"'


def _page(row: MemoryRow) -> str | None:
    """The first ``https?://`` URL in the job's FULL prompt — read whole, never
    through the character cap, since a dropped sentence must not hide the page.
    Trailing punctuation (.,;:)!?'") is stripped so a URL followed by a period
    doesn't carry it."""
    match = _FIRST_URL_RE.search(row.extraction_prompt or "")
    if not match:
        return None
    url = match.group(0)
    # Strip trailing punctuation that often appears at sentence ends
    return url.rstrip(".,;:)!?'\"") or url


def _instructions(row: MemoryRow) -> str:
    """The job's prompt as the instructions line shows it: whole when it fits the
    character budget (#2495), otherwise its first sentence — hard-cut at a word
    boundary inside the budget when even that sentence is longer — followed by a
    note counting the sentences cut and naming the ``memory_get('<name>')`` hop to
    the whole prompt. Nothing is cut silently."""
    prompt = row.extraction_prompt or ""
    cap = PennyConstants.SELF_STATE_JOB_PROMPT_CHARS
    if len(prompt) <= cap:
        return prompt
    sentences = _sentences(prompt)
    first = sentences[0]
    first_was_longer_than_cap = len(first) > cap
    if first_was_longer_than_cap:
        first = _cut_at_word(first, cap)
    cut = len(sentences) - 1

    # When the FIRST sentence itself is cut (because it's longer than cap) and
    # it's the only sentence, say "the rest is cut". Otherwise, count additional sentences.
    if len(sentences) == 1 and first_was_longer_than_cap:
        return f"{first} (the rest is cut — memory_get{{'{row.name}'}} shows the whole prompt)"
    else:
        noun = "sentence" if cut == 1 else "sentences"
        return (
            f"{first} ({cut} more {noun} cut — memory_get{{'{row.name}'}} shows the whole prompt)"
        )


def _sentences(prompt: str) -> list[str]:
    """The prompt's sentences — each period-terminated span a ``_SENTENCE_END_RE``
    boundary splits off, its period kept. A prompt with no boundary is one
    sentence; a prompt of nothing but boundaries is still one."""
    ends = [match.start() for match in _SENTENCE_END_RE.finditer(prompt)]
    bounds = [0, *(end + 1 for end in ends), len(prompt)]
    parts = [prompt[start:end].strip() for start, end in zip(bounds, bounds[1:], strict=False)]
    sentences = [part for part in parts if part]
    return sentences or [prompt]


def _cut_at_word(text: str, cap: int) -> str:
    """``text`` hard-cut inside ``cap`` characters at its last word boundary — a
    cut that never lands mid-word (a single word longer than the cap renders
    whole-and-overflowing rather than chopped, since there is no boundary to land
    on)."""
    cut = text[:cap]
    space = cut.rfind(" ")
    return cut[:space] if space > 0 else cut
