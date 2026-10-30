"""Detects a raw-template thinking draw that has started repeating itself word
for word, so the streaming transport (``client.py``, #2842) can cut it short
mid-generation instead of burning its whole token budget.

Measured against prod: three chat turns ran their thinking to the full
16,000-token budget before being cut by it, each one a literal loop (the same
reply redrafted about 30 times ending "Final. Final. Final…", an "Actually,
I'll check X too… Actually, I'll just answer" repeated many times, one
sentence repeated 68 times). Two browse-extract page-reader steps separately
looped copying the same page text until their own 8,000-token cap.

The check: every ``CHECK_INTERVAL_CHARS`` of new thought text, take the last
``REPEAT_WINDOW_CHARS`` characters; if that exact text already occurs
``REPEAT_THRESHOLD`` or more times in the thinking so far, it is looping.
Measured offline over 333 prod traces longer than 2,000 characters: caught all
3 real budget-cut loops with 73-79% of the trace still unwritten, and flagged
4 more (the two page-reader loops, two triple-drafted replies) — no false
alarms. The three constants are that measurement's own result, not a
per-deployment knob nobody has evidence to retune.

Runs on the THOUGHT channel only, never the reply — ``ThoughtStreamTracker``
is fed the RAW completion text accumulated so far after every streamed chunk
and only ever looks at the slice it can prove is inside an open thought
channel (``raw_template.THOUGHT_MARKER_PAIRS`` — gemma's own channel syntax,
or oMLX's own rendering of it as ``<think>...</think>``).
"""

from __future__ import annotations

from dataclasses import dataclass

from penny.llm.raw_template import THOUGHT_MARKER_PAIRS

REPEAT_WINDOW_CHARS = 400
REPEAT_THRESHOLD = 3
CHECK_INTERVAL_CHARS = 200


@dataclass(frozen=True)
class ThinkingLoopCut:
    """Where a detected loop should be cut.

    ``opener``/``kept_body``/``closer`` are exactly what the model itself
    streamed before the cut, minus the discarded repeat — a continuation
    prompt built as ``original_prompt + opener + kept_body + closer`` reads,
    to the model, as a turn it already started and then (truthfully) already
    closed.  ``window``/``discarded_chars`` are for the discard detail a
    caller logs, not for building the continuation.
    """

    opener: str
    kept_body: str
    closer: str
    window: str
    discarded_chars: int

    def continuation_suffix(self) -> str:
        """Appended straight after the ORIGINAL prompt to build the
        one-more-call continuation prompt (#2842)."""
        return f"{self.opener}{self.kept_body}{self.closer}"

    def discard_detail(self) -> str:
        """The ``promptlog.discard_detail`` text for the cut draw's own row —
        names the repeat that triggered the cut, not just that one did."""
        return (
            f"thought's last {REPEAT_WINDOW_CHARS} characters repeated "
            f"{REPEAT_THRESHOLD}+ times; kept {len(self.kept_body)} chars, "
            f"discarded {self.discarded_chars}"
        )


def find_loop_cut(opener: str, thought_body: str) -> ThinkingLoopCut | None:
    """Whether ``thought_body`` (the thought channel's text so far, with
    ``opener`` already stripped off) is now looping, and if so, where to cut
    it. Pure — the caller (``ThoughtStreamTracker`` below, in production; a
    contrived string in a test) decides how often to call this.

    Occurrences are counted OVERLAPPING (every start position, not
    ``str.count``'s non-overlapping stride) — a tight repeat (e.g. a 50-char
    sentence looping under a 400-char window) recurs every ~50 characters, far
    closer together than the window's own 400-char stride, so a
    non-overlapping count would need roughly 3 window-widths of pure
    repetition before ever reaching the threshold: caught far later than the
    issue's own measurement (73-79% of the trace still unwritten) implies.
    """
    if len(thought_body) < REPEAT_WINDOW_CHARS:
        return None
    window = thought_body[-REPEAT_WINDOW_CHARS:]
    first = _first_occurrence_if_repeated(thought_body, window, REPEAT_THRESHOLD)
    if first is None:
        return None
    closer = next(c for o, c in THOUGHT_MARKER_PAIRS if o == opener)
    return ThinkingLoopCut(
        opener=opener,
        kept_body=thought_body[:first],
        closer=closer,
        window=window,
        discarded_chars=len(thought_body) - first,
    )


def _first_occurrence_if_repeated(text: str, sub: str, min_count: int) -> int | None:
    """The start index of ``sub``'s FIRST occurrence in ``text`` — "the start
    of the first repeat" the recovery keeps everything up to — but only once
    ``sub`` has recurred at least ``min_count`` times overall (every start
    position counted, so occurrences may overlap). ``None`` short of that
    count. Stops scanning the moment the count is confirmed, not after a full
    pass — a loop many times longer than the threshold still costs one
    bounded scan."""
    first_index: int | None = None
    seen = 0
    index = -1
    while True:
        index = text.find(sub, index + 1)
        if index == -1:
            return None
        seen += 1
        if first_index is None:
            first_index = index
        if seen == min_count:
            return first_index


def _opened_thought(buffer: str) -> tuple[str, str, bool] | None:
    """``(opener, body, closed)`` for whichever marker pair ``buffer`` opens
    with — ``body`` truncated at the close marker when it has already
    arrived — or ``None`` before enough of the buffer is in to tell (too
    short, or this draw never opens a thought channel at all — a
    ``thinking_enabled=False`` draw's own output carries no marker, since the
    template closes the channel in the PROMPT rather than the completion)."""
    for opener, closer in THOUGHT_MARKER_PAIRS:
        if not buffer.startswith(opener):
            continue
        body = buffer[len(opener) :]
        end = body.find(closer)
        return (opener, body, False) if end == -1 else (opener, body[:end], True)
    return None


class ThoughtStreamTracker:
    """Fed the raw completion text accumulated so far after every streamed
    chunk (#2842). Re-runs the repetition check only every
    ``CHECK_INTERVAL_CHARS`` new characters of THOUGHT text (never on every
    chunk — the check is an O(n) scan) and the moment the close marker
    arrives, so a loop completed right at the channel's own close is still
    caught. A no-op once the thought channel has closed — there is nothing
    left to loop over, and (per ``_opened_thought``'s own note) a
    continuation's freshly closed channel never reopens one."""

    def __init__(self) -> None:
        self._checked_through = 0
        self.closed = False

    def feed(self, buffer: str) -> ThinkingLoopCut | None:
        if self.closed:
            return None
        opened = _opened_thought(buffer)
        if opened is None:
            return None
        opener, body, closed = opened
        self.closed = closed
        if len(body) - self._checked_through < CHECK_INTERVAL_CHARS and not closed:
            return None
        self._checked_through = len(body)
        return find_loop_cut(opener, body)
