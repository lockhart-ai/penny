"""Pure-function tests for the thinking-loop detector (#2842) — the repeated-
thought check that cuts a streamed raw-template draw short instead of letting
it run to its token budget. No I/O, no mocks: contrived texts in, a cut (or
``None``) out."""

from __future__ import annotations

from penny.llm.thinking_loop import (
    CHECK_INTERVAL_CHARS,
    REPEAT_THRESHOLD,
    REPEAT_WINDOW_CHARS,
    RepeatStreamTracker,
    ThoughtStreamTracker,
    find_loop_cut,
    find_repeat_cut,
)

_GEMMA_OPENER = "<|channel>thought\n"
_GEMMA_CLOSER = "<channel|>"
_OMLX_OPENER = "<think>\n"
_OMLX_CLOSER = "</think>"


def _distinct_window() -> str:
    """A string exactly ``REPEAT_WINDOW_CHARS`` long with NO internal
    periodicity (every 4-digit group distinct) — so it can only match itself
    at its own exact alignment. A run of one repeated character (``"x" *
    400``) would instead "overlap-match" at nearly every offset within a
    longer run of the same character, making an exact occurrence count
    meaningless for a boundary test."""
    window = "".join(f"{i:04d}" for i in range(100))
    assert len(window) == REPEAT_WINDOW_CHARS
    return window


def _redrafted_reply_loop() -> str:
    """The "same reply redrafted ~30 times ending 'Final. Final. Final…'"
    prod shape named in the issue — the whole draft-then-"Final." unit
    repeats, so the tail reads exactly as the issue describes it."""
    unit = (
        "Let me think about how to phrase this reply to the user about their "
        "schedule for tomorrow, covering the three meetings and the one open slot. "
        "Final. "
    )
    return unit * 30


def _actually_ill_loop() -> str:
    """The "Actually, I'll check X too… Actually, I'll just answer" prod shape."""
    return "Actually, I'll check the calendar too. Actually, I'll just answer. " * 40


def _single_sentence_loop() -> str:
    """The "one sentence repeated 68 times" prod shape."""
    return "The user wants to know if the meeting moved. " * 68


def _non_looping_thinking() -> str:
    """A long thought that never repeats itself — the false-alarm guard."""
    topics = [
        "the calendar",
        "the open invoice",
        "the weather forecast",
        "the grocery list",
        "the flight times",
        "the last message",
        "the reminder",
        "the draft email",
        "the meeting notes",
        "the budget",
        "the travel plans",
        "the follow-up task",
    ]
    return "".join(
        f"Now considering {topic} in detail, step {i}. " for i, topic in enumerate(topics * 10)
    )


class TestFindLoopCut:
    def test_short_thinking_never_cuts(self) -> None:
        assert find_loop_cut(_GEMMA_OPENER, "too short to loop") is None

    def test_long_non_looping_thinking_never_cuts(self) -> None:
        body = _non_looping_thinking()
        assert len(body) > REPEAT_WINDOW_CHARS * 3
        assert find_loop_cut(_GEMMA_OPENER, body) is None

    def test_redrafted_reply_loop_is_cut(self) -> None:
        body = _redrafted_reply_loop()
        cut = find_loop_cut(_GEMMA_OPENER, body)
        assert cut is not None
        assert cut.opener == _GEMMA_OPENER
        assert cut.closer == _GEMMA_CLOSER
        # What's kept is a strict PREFIX of the full body — the repeat itself,
        # and everything after it, is what gets discarded.
        assert body.startswith(cut.kept_body)
        assert len(cut.kept_body) < len(body)
        assert cut.discarded_chars > 0

    def test_actually_ill_loop_is_cut(self) -> None:
        cut = find_loop_cut(_GEMMA_OPENER, _actually_ill_loop())
        assert cut is not None

    def test_single_sentence_loop_is_cut(self) -> None:
        cut = find_loop_cut(_GEMMA_OPENER, _single_sentence_loop())
        assert cut is not None

    def test_omlx_spelling_is_cut_the_same_way(self) -> None:
        """The detector is agnostic to which provider's own marker spelling
        carried the thought — oMLX's ``<think>`` closes with ``</think>``,
        never gemma's own ``<channel|>``."""
        cut = find_loop_cut(_OMLX_OPENER, _single_sentence_loop())
        assert cut is not None
        assert cut.closer == _OMLX_CLOSER

    def test_exactly_two_repeats_does_not_cut(self) -> None:
        """Below ``REPEAT_THRESHOLD`` (3) is not a loop — a sentence said
        twice is ordinary emphasis, not runaway repetition.

        ``_distinct_window`` is deliberately NON-periodic internally (unlike
        a run of one repeated character), so it can only match itself at
        its own exact alignment — this body has EXACTLY two occurrences,
        not "at least two from incidental overlap"."""
        window = _distinct_window()
        body = window + window
        assert find_loop_cut(_GEMMA_OPENER, body) is None

    def test_exactly_three_repeats_cuts_at_the_first_occurrence(self) -> None:
        window = _distinct_window()
        body = "preamble " + window + window + window
        cut = find_loop_cut(_GEMMA_OPENER, body)
        assert cut is not None
        assert cut.kept_body == "preamble "
        assert cut.window == window

    def test_continuation_suffix_reads_as_the_models_own_closed_turn(self) -> None:
        cut = find_loop_cut(_GEMMA_OPENER, _single_sentence_loop())
        assert cut is not None
        suffix = cut.continuation_suffix()
        assert suffix.startswith(_GEMMA_OPENER)
        assert suffix.endswith(_GEMMA_CLOSER)
        assert cut.kept_body in suffix

    def test_discard_detail_names_the_window_and_threshold(self) -> None:
        cut = find_loop_cut(_GEMMA_OPENER, _single_sentence_loop())
        assert cut is not None
        detail = cut.discard_detail()
        assert str(REPEAT_WINDOW_CHARS) in detail
        assert str(REPEAT_THRESHOLD) in detail
        assert str(len(cut.kept_body)) in detail


class TestThoughtStreamTracker:
    def test_never_opened_is_never_cut(self) -> None:
        """A ``thinking_enabled=False`` draw's own output carries no thought
        marker at all — the tracker must never mistake ordinary reply text
        for a thought channel."""
        tracker = ThoughtStreamTracker()
        reply = "Sure, here's the answer you were after. " * 50
        buffer = ""
        for chunk in _chunks(reply):
            buffer += chunk
            assert tracker.feed(buffer) is None

    def test_a_loop_is_caught_mid_stream_not_only_at_the_end(self) -> None:
        """Streamed incrementally, the SAME text that loops when checked whole
        is caught partway through, before every character has arrived — the
        whole point of running the check WHILE streaming."""
        full = _GEMMA_OPENER + _single_sentence_loop() + _GEMMA_CLOSER + "the reply"
        tracker = ThoughtStreamTracker()
        buffer = ""
        cut = None
        chars_seen_at_cut = None
        for chunk in _chunks(full, size=37):
            buffer += chunk
            cut = tracker.feed(buffer)
            if cut is not None:
                chars_seen_at_cut = len(buffer)
                break
        assert cut is not None
        assert chars_seen_at_cut is not None
        assert chars_seen_at_cut < len(full), "caught before the whole stream ever arrived"

    def test_a_cut_tracker_is_closed_and_feeds_nothing_more(self) -> None:
        tracker = ThoughtStreamTracker()
        full = _GEMMA_OPENER + _single_sentence_loop()
        cut = None
        buffer = ""
        for chunk in _chunks(full):
            buffer += chunk
            cut = tracker.feed(buffer)
            if cut is not None:
                break
        assert cut is not None
        # Once cut, further text (e.g. the rest of a stream the caller hasn't
        # closed yet) is never re-checked.
        assert tracker.feed(buffer + "more text") is None

    def test_a_closed_non_looping_channel_stays_closed_and_uncut(self) -> None:
        full = _GEMMA_OPENER + "a short, ordinary thought" + _GEMMA_CLOSER + "the reply"
        tracker = ThoughtStreamTracker()
        buffer = ""
        for chunk in _chunks(full):
            buffer += chunk
            assert tracker.feed(buffer) is None
        assert tracker.closed

    def test_checks_happen_every_check_interval_not_every_chunk(self) -> None:
        """A single-character-at-a-time stream still only runs the O(n) scan
        every ``CHECK_INTERVAL_CHARS`` — proven indirectly: the non-looping
        body below is longer than one interval, and feeding it one char at a
        time never raises and never cuts, so the tracker tolerates a stream
        far chattier than the oMLX/CoreWeave chunking actually measured."""
        body = _non_looping_thinking()
        assert len(body) > CHECK_INTERVAL_CHARS * 2
        full = _GEMMA_OPENER + body
        tracker = ThoughtStreamTracker()
        buffer = ""
        for char in full:
            buffer += char
            assert tracker.feed(buffer) is None


class TestCleanBoundary:
    """#2843: a cut landing mid-draft let a real prod continuation resume an
    unfinished line and loop again. ``find_loop_cut``/``find_repeat_cut`` must
    both back the cut up to a clean boundary — a paragraph break, or failing
    that a sentence end — never leave a half-written line as the kept text."""

    def test_thought_cut_backs_up_to_the_last_paragraph_break(self) -> None:
        """The digest shape: the repeat starts mid-list-item, with a
        complete earlier paragraph (a markdown-style digest entry) before
        it."""
        complete_entry = "- [A story about an appeal][1] after a two-year wait.\n\n"
        mid_draft_entry = "- [Another story, still being drafted"
        looping_tail = (" after a two-year appeal][2]. Continuing to draft the same entry. ") * 10
        body = complete_entry + mid_draft_entry + looping_tail
        cut = find_loop_cut(_GEMMA_OPENER, body)
        assert cut is not None
        assert cut.kept_body == complete_entry
        assert cut.kept_body.endswith("\n\n")
        assert mid_draft_entry not in cut.kept_body

    def test_thought_cut_falls_back_to_the_last_sentence_end(self) -> None:
        """No blank line anywhere before the repeat — the next-best clean
        boundary is the end of the last complete sentence."""
        prefix = "First I considered the weather. Then I started on the next point"
        looping_tail = (" about whether to reschedule, still deciding whether to reschedule. ") * 10
        body = prefix + looping_tail
        cut = find_loop_cut(_GEMMA_OPENER, body)
        assert cut is not None
        assert cut.kept_body == "First I considered the weather."

    def test_with_neither_a_paragraph_break_nor_a_sentence_end(self) -> None:
        """No clean boundary exists at all before the repeat — falls back to
        the repeat's own start (today's prior behaviour), never crashes."""
        body = "no punctuation here at all " + _distinct_window() * 3
        cut = find_loop_cut(_GEMMA_OPENER, body)
        assert cut is not None
        assert cut.kept_body == "no punctuation here at all "

    def test_repeat_cut_also_backs_up_to_a_paragraph_break(self) -> None:
        """``find_repeat_cut`` (the continuation's own second-loop guard,
        content included) gets the identical clean-boundary treatment."""
        complete_entry = "Here is the first full point, settled.\n\n"
        mid_draft = "Now starting the second point, which keeps repeating"
        looping_tail = " itself over and over without ever finishing. " * 14
        text = complete_entry + mid_draft + looping_tail
        cut = find_repeat_cut(text)
        assert cut is not None
        assert cut.kept_text == complete_entry


class TestRepeatStreamTrackerAndFindRepeatCut:
    """The continuation's own guard (#2843) — the WHOLE output, content
    included, no thought-channel marker awareness at all."""

    def test_short_text_never_cuts(self) -> None:
        assert find_repeat_cut("not nearly enough text to loop") is None

    def test_a_repeat_in_plain_reply_content_is_cut(self) -> None:
        """No thought markers anywhere — exactly what a continuation's own
        reply CONTENT looks like once its prompt already closed thinking."""
        sentence = "I will keep listing this same point. "
        text = sentence * 16
        cut = find_repeat_cut(text)
        assert cut is not None
        assert cut.discarded_chars > 0

    def test_tracker_catches_a_loop_in_streamed_content(self) -> None:
        """Mirrors ``_consume_stream``'s own calling convention: ``feed`` on
        every chunk, then ONE ``final_check`` once the stream ends (#2843) —
        without it, a stream that ends short of another interval boundary
        (the common case) would miss a loop sitting in that final stretch,
        exactly what this fixture's own length triggers."""
        sentence = "I will keep listing this same point. "
        full = sentence * 16
        tracker = RepeatStreamTracker()
        buffer = ""
        cut = None
        for chunk in _chunks(full):
            buffer += chunk
            cut = tracker.feed(buffer)
            if cut is not None:
                break
        if cut is None:
            cut = tracker.final_check(buffer)
        assert cut is not None

    def test_tracker_never_cuts_long_non_looping_content(self) -> None:
        tracker = RepeatStreamTracker()
        buffer = ""
        for chunk in _chunks(_non_looping_thinking()):
            buffer += chunk
            assert tracker.feed(buffer) is None
        assert tracker.final_check(buffer) is None


def _chunks(text: str, size: int = 23) -> list[str]:
    """Split ``text`` into chunks of ``size`` — an arbitrary, non-round
    number so a test can't pass by accident of chunk boundaries lining up
    with marker lengths."""
    return [text[i : i + size] for i in range(0, len(text), size)]
