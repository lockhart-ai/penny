"""The conversation map's mechanism, pinned deterministically (#2752, #2982).

The map is a render over stored rows, so every test seeds rows through the
stores' own write paths (``log_message``, ``write_rule_decided``, the
browse-results append with its run id, ``mutations.record``, ``log_prompt``)
and asserts on ``ConversationMap.render()`` whole — never by substring.  What
the mock tests pin here is what ``make check`` owns: the line render (the
user's words whole to ``USER_CLIP_CHARS``, the recorded outcome, the
autonomous-notification line), the window boundary shared with the history's
own grow/cut cycle (#2756/#2757), the collapse rules, and the two boundings.
The live-model contract is the ``chat/idle`` refer-back case beside them.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

from sqlmodel import col, update

from penny.agents.base import Agent
from penny.agents.conversation_map import MESSAGE_POOL, USER_CLIP_CHARS, ConversationMap
from penny.agents.related_exchanges import RelatedExchanges
from penny.config_params import RuntimeParams
from penny.constants import (
    ChatPromptType,
    MutationAction,
    MutationActor,
    MutationEntityType,
    PennyConstants,
)
from penny.database.memory import EntryInput, LogEntryInput
from penny.database.models import MessageLog
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.utils.clock import PinnedClock
from penny.tests.schema_template import migrated_db

_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY = PennyConstants.MessageAuthor.PENNY


def _map(db) -> ConversationMap:
    anchor = db.messages.session_anchor(TEST_SENDER)
    return ConversationMap(db, TEST_SENDER, RuntimeParams(db), anchor)


def _exchange(db, user: str, reply: str, *, run_id: str | None = None) -> None:
    db.messages.log_message(_INCOMING, TEST_SENDER, user)
    db.messages.log_message(_OUTGOING, _PENNY, reply, recipient=TEST_SENDER, run_id=run_id)


def _autonomous_send(db, collection: str, run_id: str, *, prompt_type: str | None) -> None:
    """One autonomous send, stamped the way the framework stamps it (#1568):
    an outgoing row with a non-null ``mechanism`` and the delivery run's id,
    plus that run's own promptlog row carrying the delivery turn's
    ``prompt_type`` — the stamp the map's kind label reads."""
    db.messages.log_message(
        _OUTGOING,
        _PENNY,
        f"the composed {collection} report message",
        recipient=TEST_SENDER,
        mechanism=collection,
        run_id=run_id,
    )
    db.messages.log_prompt(
        model="test-model",
        messages=[{"role": "user", "content": "the delivery turn's trigger"}],
        response={"content": f"the composed {collection} report message"},
        agent_name="chat",
        prompt_type=prompt_type,
        run_id=run_id,
    )


def _backdate(db, message_ids: list[int], days: float) -> None:
    stamp = db.clock.now() - timedelta(days=days)
    with db.get_session() as session:
        session.exec(
            update(MessageLog).where(col(MessageLog.id).in_(message_ids)).values(timestamp=stamp)
        )
        session.commit()


def _last_ids(db, count: int) -> list[int]:
    rows = db.messages.recent_direct_messages(since=datetime.min, limit=count)
    return [row.id for row in rows if row.id is not None]


def _filler(db, pairs: int) -> None:
    """Short, outcome-less exchanges that pad the window out — comfortably past
    any grow/cut phase of the default window, so the seeded exchange ahead of
    them is older than the window whatever cycle position the count lands on."""
    for index in range(pairs):
        _exchange(db, f"filler question {index}", f"filler answer {index}")


def _write_entry(db, collection: str, key: str, run_id: str) -> None:
    db.memories.create_collection(
        collection, f"Notes about {collection}.", created_by_run_id=run_id
    )
    require_memory(db, collection).write_rule_decided(
        [EntryInput(key=key, content=f"{key} — a note.")], author=_PENNY, run_id=run_id
    )


def _outcome_exchange(db, index: int) -> None:
    """One outcome-bearing exchange — never collapse fodder, so counting its
    own line in the rendered map is unambiguous."""
    run_id = f"{index:032x}"
    _write_entry(db, f"list-{index}", f"key-{index}", run_id)
    _exchange(db, f"exchange ask {index}", f"exchange reply {index}", run_id=run_id)


def _long_exchange(db, index: int) -> None:
    """One plain exchange whose ask is longer than ``SHORT_CHARS`` — never
    collapse fodder, so each renders its own line, without needing a recorded
    outcome (a seeded day of 100+ of these stays cheap)."""
    _exchange(
        db,
        f"a busy day's question {index}, asked at enough length that " + "y" * 80,
        f"the reply {index}",
    )


def _content_lines(rendered: str) -> list[str]:
    """The map's own content lines — everything between the header and the
    fixed closing anchor."""
    if not rendered:
        return []
    return rendered.split("\n")[1:-1]


def _render_bounded(map: ConversationMap, seconds: float = 10.0) -> str:
    """``render()`` on a worker thread, joined with a timeout — the ceiling
    fold must always end, so a spin (a fold pass that changes nothing) fails
    the test instead of hanging the suite (#3014)."""
    outcome: dict[str, str] = {}

    def run() -> None:
        try:
            outcome["rendered"] = map.render()
        except Exception as exc:  # surfaced below, on the calling thread
            outcome["error"] = repr(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), "render() never returned — the fold loop spun"
    if "error" in outcome:
        raise RuntimeError(outcome["error"])
    return outcome["rendered"]


# ── The line render ──────────────────────────────────────────────────────────


def test_a_line_renders_the_users_words_whole_and_the_outcome(db) -> None:
    run_id = "a" * 32
    ask = (
        "which bands did the Talking Heads members form after the band broke up, "
        "and who was in each one?"
    )
    _write_entry(db, "albums-to-hear", "remain-in-light", run_id)
    db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG).append(
        [LogEntryInput(content="## browse: https://example.com/a\npage")],
        author=_PENNY,
        run_id=run_id,
    )
    db.mutations.record(
        entity_type=MutationEntityType.COLLECTION,
        entity_name="albums-to-hear",
        action=MutationAction.CREATED,
        actor=MutationActor.USER_RUN,
        run_id=run_id,
    )
    _exchange(
        db,
        ask,
        "here you go **Tom Tom Club** and **The Modern Lovers** — see https://example.com/a",
        run_id=run_id,
    )
    ids = _last_ids(db, 2)
    _filler(db, 20)  # comfortably past the default window's grow/cut cycle
    _backdate(db, ids, 1.0)

    rendered = _map(db).render()
    lines = [line for line in rendered.split("\n")[1:] if line.startswith("- ")]
    # The exchange's own line, older than the window — the user's own words
    # WHOLE (the ask is under the clip), the outcome from the records:
    assert any(
        f'"{ask}"' in line
        and "read 1 page" in line
        and "saved 'remain-in-light' → albums-to-hear" in line
        and "created albums-to-hear" in line
        for line in lines
    ), rendered
    # Her reply's own markup is never the map's content: the items it marked
    # and the URL it kept are gone (#2982).
    assert "named:" not in rendered
    assert "Tom Tom Club" not in rendered
    assert "example.com" not in rendered


def test_a_long_message_keeps_its_head_and_its_tail(db) -> None:
    long_ask = "opening words " + "x" * 400 + " and the conclusion everyone needed"
    _exchange(db, long_ask, "a short answer")
    ids = _last_ids(db, 2)
    _filler(db, 20)
    _backdate(db, ids, 1.0)
    rendered = _map(db).render()
    # The clip: head + " … " + tail, counted inside USER_CLIP_CHARS — the
    # conclusion, which came last, still rides the line.
    assert long_ask[: (USER_CLIP_CHARS - 3 + 1) // 2] in rendered
    assert "and the conclusion everyone needed" in rendered
    assert " … " in rendered
    line = next(line for line in rendered.split("\n") if " … " in line)
    assert len(line) < len(long_ask) + 40  # the line is bounded, the ask was not


def test_a_reply_with_no_outcome_renders_no_arrow(db) -> None:
    _exchange(db, "what did you find?", "nothing usable came back, honestly.")
    ids = _last_ids(db, 2)
    _filler(db, 20)
    _backdate(db, ids, 1.0)
    rendered = _map(db).render()
    assert "→" not in rendered and "nothing usable" not in rendered


# ── The autonomous-notification lines (#2982) ───────────────────────────────


def test_an_autonomous_send_renders_its_notification_line(db) -> None:
    """A mechanism-stamped outgoing row older than the window renders as its
    own line — kind label off the delivery run's ``prompt_type`` stamp, the
    job's name, and the verbatim ``run_get`` anchor."""
    report_run = "c" * 32
    _autonomous_send(db, "morning-brief", report_run, prompt_type=ChatPromptType.JOB_REPORT)
    ids = _last_ids(db, 1)
    _filler(db, 20)
    _backdate(db, ids, 1.0)
    rendered = _map(db).render()
    assert f"job report: morning-brief (run {report_run}) → you replied" in rendered, rendered
    # The delivered message's own prose never rides the map.
    assert "the composed morning-brief report message" not in rendered


def test_a_sandbox_report_and_the_startup_notice_render_their_kinds(db) -> None:
    sandbox_run = "d" * 32
    changelog_run = "e" * 32
    _autonomous_send(db, "tabletop-lore", sandbox_run, prompt_type=ChatPromptType.SANDBOX_REPORT)
    _autonomous_send(db, "changelog", changelog_run, prompt_type=ChatPromptType.CHANGELOG)
    ids = _last_ids(db, 2)
    _filler(db, 20)
    _backdate(db, ids, 1.0)
    rendered = _map(db).render()
    assert f"sandbox report: tabletop-lore (run {sandbox_run}) → you replied" in rendered
    # The startup notice renders WITHOUT its mechanism — ``changelog`` is the
    # memory row's own name, internal naming, never a job name to anchor.
    assert f"startup notice (run {changelog_run}) → you replied" in rendered
    assert "changelog" not in rendered


def test_an_unknown_delivery_kind_renders_the_fallback_label(db) -> None:
    """A future delivery mechanism — a prompt_type the labels table doesn't
    name — renders the generic label with the row's own mechanism, visibly,
    never silently dropped (visible degradation over silent success)."""
    future_run = "f" * 32
    _autonomous_send(db, "new-thing", future_run, prompt_type="new_delivery_kind")
    ids = _last_ids(db, 1)
    _filler(db, 20)
    _backdate(db, ids, 1.0)
    rendered = _map(db).render()
    assert f"notification: new-thing (run {future_run}) → you replied" in rendered, rendered


def test_a_notification_newer_than_the_window_renders_nothing(db) -> None:
    """The window boundary is the window's first USER exchange (#2982): a
    notification still inside the history renders no line, whatever its
    position in the send order."""
    _filler(db, 20)
    _autonomous_send(db, "fresh-job", "9" * 32, prompt_type=ChatPromptType.JOB_REPORT)
    # The notification is the newest item — inside the window by position.
    assert "fresh-job" not in _map(db).render()


# ── The window boundary — shared with the history's grow/cut cycle (#2757) ──


def test_the_map_is_byte_identical_across_two_non_cut_turns(db) -> None:
    """Between two turns neither of which is a cut, the history window's own
    first exchange doesn't move (``Agent._history_window_size``), so the set of
    exchanges the map covers doesn't change either — the map renders
    BYTE-IDENTICAL, not merely a growing prefix (#2757).  The cap sits one
    above the window, so the session alternates between window and window+1
    without ever cutting."""
    db.config.set("CHAT_HISTORY_TURNS", "2", "test")
    db.config.set("CHAT_SESSION_MAX_TURNS", "3", "test")
    for index in range(4):  # total=4 is the state right after a cut: older=2
        _outcome_exchange(db, index)
    assert Agent._history_window_size(4, 4, 2, 3) == 2  # sanity: older = 4 - 2 = 2
    before = _map(db).render()
    assert len(_content_lines(before)) == 2

    _outcome_exchange(db, 4)  # total=5 — still growing, same 2 older exchanges
    assert Agent._history_window_size(5, 5, 2, 3) == 3  # older = 5 - 3 = 2, unchanged
    after = _map(db).render()

    assert after == before


def test_the_map_grows_by_the_exchanges_that_left_on_a_cut_turn(db) -> None:
    """On a cut turn — the one turn the history window's own first exchange
    moves — the map gains exactly the exchanges the history dropped, whole, at
    its end; every earlier line stays byte-identical (#2757).  The cap drives
    the cut: the session passes ``CHAT_SESSION_MAX_TURNS`` on the 6th turn of
    a two-exchange window (#3077)."""
    db.config.set("CHAT_HISTORY_TURNS", "2", "test")
    db.config.set("CHAT_SESSION_MAX_TURNS", "3", "test")
    for index in range(5):  # total=5: growing, older=2 (exchanges 0, 1)
        _outcome_exchange(db, index)
    assert Agent._history_window_size(5, 5, 2, 3) == 3
    before = _map(db).render()
    before_lines = _content_lines(before)
    assert len(before_lines) == 2

    _outcome_exchange(db, 5)  # total=6 — the cap cut: window drops back to 2
    assert Agent._history_window_size(6, 6, 2, 3) == 2  # older = 6 - 2 = 4
    after = _map(db).render()
    after_lines = _content_lines(after)

    assert after_lines[: len(before_lines)] == before_lines  # a pure prefix
    assert len(after_lines) == 4
    assert "exchange ask 2" in after_lines[2]
    assert "exchange ask 3" in after_lines[3]


def test_the_map_grows_by_the_notifications_that_left_on_a_cut_turn(db) -> None:
    """A notification between two user exchanges ages out with the exchange
    ahead of it: the slice runs to the window's first USER exchange, so a
    notification that precedes it rides the map (#2982)."""
    db.config.set("CHAT_HISTORY_TURNS", "2", "test")
    db.config.set("CHAT_SESSION_MAX_TURNS", "3", "test")
    _outcome_exchange(db, 0)
    _autonomous_send(db, "old-job", "7" * 32, prompt_type=ChatPromptType.JOB_REPORT)
    for index in range(1, 5):
        _outcome_exchange(db, index)
    # total=5 user turns: growing, window=3, older = exchanges 0-1 + the
    # notification between them.
    assert Agent._history_window_size(5, 5, 2, 3) == 3
    rendered = _map(db).render()
    lines = _content_lines(rendered)
    assert len(lines) == 3
    assert "old-job" in lines[1]


def test_the_map_and_the_related_exchanges_share_one_boundary(db) -> None:
    """The map's older-than set and the related-exchanges boundary are
    computed from the SAME session anchor (#3077): the newest exchange the map
    shows is exactly the one whose user turn the related-exchanges boundary
    row holds — the history's own oldest kept user turn.  One anchor, so the
    three readers can never disagree about where the window starts."""
    db.config.set("CHAT_HISTORY_TURNS", "2", "test")
    db.config.set("CHAT_SESSION_MAX_TURNS", "3", "test")
    for index in range(5):  # one session, cap 3: window = 2 + 5 % 2 = 3
        _outcome_exchange(db, index)

    anchor = db.messages.session_anchor(TEST_SENDER)
    boundary = RelatedExchanges(
        db, TEST_SENDER, RuntimeParams(db), anchor, [0.0], frozenset()
    )._boundary()
    assert boundary is not None and boundary.content == "exchange ask 2"

    lines = _content_lines(_map(db).render())
    assert len(lines) == 2  # the exchanges older than the window's first kept turn
    assert "exchange ask 0" in lines[0]
    assert "exchange ask 1" in lines[1]


def test_an_empty_map_renders_nothing(db) -> None:
    """Fewer exchanges than the window: nothing is older than it, so the map
    renders empty — the static system message it rides in stays byte-identical
    to a build with no map at all (pinned end to end in
    ``test_full_chat_system_prompt_composition``)."""
    _outcome_exchange(db, 0)
    assert _map(db).render() == ""


# ── The date horizon ─────────────────────────────────────────────────────────


def test_the_date_horizon_drops_whole_days_from_the_front(db) -> None:
    oldest_ids: list[int] = []
    for index in range(6):
        run_id = f"{index:032x}"
        _write_entry(db, f"list-{index}", f"key-{index}", run_id)
        _exchange(db, f"exchange ask {index}", f"exchange reply {index}", run_id=run_id)
        if index == 0:
            oldest_ids = _last_ids(db, 2)
    _filler(db, 20)
    _backdate(db, oldest_ids, 3.0)  # the OLDEST exchange, outside the 2-day horizon

    rendered = _map(db).render()
    assert "exchange ask 0" not in rendered  # left by the horizon, never shown half-rendered
    assert "exchange ask 1" in rendered


# ── The collapse rule ────────────────────────────────────────────────────────


def test_consecutive_short_exchanges_collapse_into_one_ranged_line(db) -> None:
    for index in range(3):
        _exchange(db, f"quick hello {index}", f"quick answer {index}")
    _exchange(db, "a real ask that needs doing properly today", "did it — saved.", run_id="b" * 32)
    _write_entry(db, "done-list", "did-it", "b" * 32)  # the recorded outcome keeps it un-collapsed
    _filler(db, 20)

    rendered = _map(db).render()
    assert "3 short messages" in rendered
    assert "quick hello" not in rendered and "quick answer" not in rendered
    # The outcome-bearing exchange renders its own full line — collapse fodder
    # is the outcome-less ones.
    assert "a real ask that needs doing properly today" in rendered


# ── The one-to-one ids (#3085) ──────────────────────────────────────────────


def test_one_to_one_user_ids_names_exchanges_with_their_own_line(db) -> None:
    """``one_to_one_user_ids()`` runs the SAME pipeline ``render()`` runs: a
    lone line — outcome-bearing or a lone short exchange — carries its user
    id, while a RANGED short-collapse line and a notification line carry
    none, since neither shows one exchange's own words whole."""
    run_id = "a" * 32
    _write_entry(db, "list-0", "key-0", run_id)
    _exchange(db, "a real ask with a recorded outcome", "did it — saved.", run_id=run_id)
    outcome_ids = _last_ids(db, 2)

    _exchange(db, "hi", "hey")  # a lone short exchange, between two non-short ones
    lone_short_ids = _last_ids(db, 2)

    _long_exchange(db, 0)  # breaks it from the run below — never collapse fodder
    break_ids = _last_ids(db, 2)

    for index in range(3):
        _exchange(db, f"quick hello {index}", f"quick answer {index}")
    collapsed_ids = _last_ids(db, 6)

    report_run = "c" * 32
    _autonomous_send(db, "morning-brief", report_run, prompt_type=ChatPromptType.JOB_REPORT)

    _filler(db, 20)  # comfortably past the default window's grow/cut cycle

    map_ = _map(db)
    rendered = map_.render()
    assert '"hi"' in rendered  # sanity: the lone short exchange kept its own line
    assert "3 short messages" in rendered  # sanity: the run collapsed
    assert "morning-brief" in rendered  # sanity: the notification rendered

    one_to_one = map_.one_to_one_user_ids()
    assert outcome_ids[0] in one_to_one
    assert lone_short_ids[0] in one_to_one
    assert break_ids[0] in one_to_one
    assert not (one_to_one & set(collapsed_ids))


# ── The ceiling ──────────────────────────────────────────────────────────────


def test_past_the_ceiling_the_oldest_whole_day_collapses(db) -> None:
    """Six outcome-bearing exchanges older than a two-exchange window, five of
    them yesterday and one today, ceiling 5: the oldest day folds into ONE
    line naming its log_read read, and the map lands under the ceiling
    without a single exchange trimmed line-by-line."""
    db.config.set("CHAT_HISTORY_TURNS", "2", "test")
    db.config.set("CHAT_SESSION_MAX_TURNS", "3", "test")
    # The backdated days read as gaps, so the session idle threshold is raised
    # past them: one session over everything, and the cap-3 cycle lands the
    # window back at 2 exactly as a flat window would (#3077).
    db.config.set("CHAT_SESSION_IDLE_SECONDS", str(7 * 24 * 3600), "test")
    for index in range(8):  # window=2: older = 8 - 2 = 6 (exchanges 0-5)
        run_id = f"{index:032x}"
        _write_entry(db, f"list-{index}", f"key-{index}", run_id)
        _exchange(db, f"exchange ask {index}", f"exchange reply {index}", run_id=run_id)
        ids = _last_ids(db, 2)
        if index < 5:
            _backdate(db, ids, 1.0)  # the oldest five: yesterday
    assert Agent._history_window_size(8, 8, 2, 3) == 2  # sanity: older = 8 - 2 = 6
    db.config.set("CHAT_MAP_MAX_LINES", "5", "test ceiling")

    rendered = _map(db).render()
    content_lines = list(rendered.split("\n")[1:-1])
    day_lines = [line for line in content_lines if "exchanges — memory_access" in line]
    assert len(day_lines) == 1, rendered
    assert day_lines[0].startswith("- ")
    assert (
        "memory_access{target: 'user-messages', action: 'read', payload: {day: '" in (day_lines[0])
    )
    assert len(content_lines) <= 5
    # Never trimmed line-by-line: every exchange the day line replaced is gone whole.
    assert "exchange ask 0" not in rendered and "exchange ask 4" not in rendered
    assert "exchange ask 5" in rendered


def test_the_fold_folds_the_oldest_unfolded_day_and_terminates(db) -> None:
    """The oldest day already a single fold line, the NEXT day alone over the
    ceiling: each pass folds the oldest day that still has more than one line
    and is not the newest, never an already-folded day — folding that one
    again would change nothing, and the pre-fix loop spanned forever on it
    (#3014).  The render returns, with that next day folded too."""
    db.config.set("CHAT_HISTORY_TURNS", "2", "test")
    db.config.set("CHAT_SESSION_MAX_TURNS", "3", "test")
    # The backdated days read as gaps, so the session idle threshold is raised
    # past them — one session, and the cap-3 cycle lands the window back at 2
    # exactly as a flat window would (#3077).
    db.config.set("CHAT_SESSION_IDLE_SECONDS", str(7 * 24 * 3600), "test")
    db.config.set("CHAT_MAP_MAX_LINES", "3", "test ceiling")
    db.config.set("CHAT_MAP_DAYS", "4", "test horizon")
    for index in range(10):
        _outcome_exchange(db, index)
        ids = _last_ids(db, 2)
        if index < 2:
            _backdate(db, ids, 3.0)  # the oldest day: 2 exchanges
        elif index < 6:
            _backdate(db, ids, 2.0)  # the next day: 4 exchanges — alone over the ceiling
        elif index < 8:
            _backdate(db, ids, 1.0)  # the newest day: two exchanges, in the map
    # total=10, window=2: the map covers the oldest 8 — day A ×2, day B ×4, day C ×2.

    rendered = _render_bounded(_map(db))
    content_lines = _content_lines(rendered)
    day_lines = [line for line in content_lines if "exchanges — memory_access" in line]
    assert len(day_lines) == 2, rendered  # day A already folded, then day B folded too
    assert any("· 2 exchanges — memory_access" in line for line in day_lines), rendered
    assert any("· 4 exchanges — memory_access" in line for line in day_lines), rendered
    assert len(content_lines) == 4, rendered  # 2 fold lines + the newest day's 2 lines
    assert "exchange ask 6" in rendered and "exchange ask 7" in rendered  # never folded
    assert "exchange ask 0" not in rendered and "exchange ask 2" not in rendered


def test_only_the_newest_day_over_the_ceiling_stops_the_fold(db) -> None:
    """Only the newest day is over the ceiling: the oldest day folds, then no
    day but the newest has more than one line left — the fold stops and the
    render returns with the map still over the ceiling, the newest day
    untouched (#3014).  The pre-fix loop spanned forever re-folding the
    already-folded oldest day."""
    db.config.set("CHAT_HISTORY_TURNS", "2", "test")
    db.config.set("CHAT_SESSION_MAX_TURNS", "3", "test")
    # The backdated days read as gaps, so the session idle threshold is raised
    # past them — one session, and the cap-3 cycle lands the window back at 2
    # exactly as a flat window would (#3077).
    db.config.set("CHAT_SESSION_IDLE_SECONDS", str(7 * 24 * 3600), "test")
    db.config.set("CHAT_MAP_MAX_LINES", "3", "test ceiling")
    db.config.set("CHAT_MAP_DAYS", "4", "test horizon")
    for index in range(8):
        _outcome_exchange(db, index)
        ids = _last_ids(db, 2)
        if index < 2:
            _backdate(db, ids, 3.0)  # the oldest day: 2 exchanges
        elif index < 6:
            _backdate(db, ids, 1.0)  # the newest day: 4 exchanges — alone over the ceiling
    # total=8, window=2: the map covers the oldest 6 — day A ×2, day C ×4.

    rendered = _render_bounded(_map(db))
    content_lines = _content_lines(rendered)
    day_lines = [line for line in content_lines if "exchanges — memory_access" in line]
    assert len(day_lines) == 1, rendered  # only the oldest day folded
    assert "· 2 exchanges — memory_access" in day_lines[0], rendered
    assert len(content_lines) == 5, rendered  # over the ceiling — the fold stopped
    for index in range(2, 6):
        assert f"exchange ask {index}" in rendered  # the newest day stays unfolded


def test_a_folded_day_over_the_pool_keeps_its_true_count(db) -> None:
    """A folded day holding more rows than ``MESSAGE_POOL``: its fold line's
    count is the day's true count off the store, never whatever part of the
    day the bounded read still holds — so as later-day messages push more of
    the day's front out of the pool, the fold line is byte-identical between
    renders instead of shrinking turn by turn (#3056, the prod Nov 5 churn
    that busted the cached prefix at the map)."""
    db.config.set("CHAT_HISTORY_TURNS", "2", "test")
    db.config.set("CHAT_SESSION_MAX_TURNS", "3", "test")
    # The window must stay smaller than the session for day B to ride the map
    # beside day A — with the default cap it swallows day B whole, leaving one
    # over-ceiling day the fold correctly never folds (#3077).
    db.config.set("CHAT_MAP_DAYS", "4", "test horizon")
    db.config.set("CHAT_MAP_MAX_LINES", "50", "test ceiling")
    # Each exchange is 2 rows, so this day holds more rows than the pool — the
    # read keeps only the day's tail, and later messages keep shaving it.
    day_a_count = MESSAGE_POOL // 2 + 20
    for index in range(day_a_count):
        _long_exchange(db, index)
    day_a_ids = _last_ids(db, 2 * day_a_count)
    _backdate(db, day_a_ids, 2.0)  # the folded day
    for index in range(6):  # a later day with lines of its own — the fold needs one
        _long_exchange(db, 100 + index)
    _backdate(db, _last_ids(db, 12), 1.0)

    # Sanity: the horizon's read is saturated — the day really did outgrow the pool.
    assert len(db.messages.recent_direct_messages(since=datetime.min, limit=MESSAGE_POOL)) == (
        MESSAGE_POOL
    )

    before = _render_bounded(_map(db))
    before_fold = next(
        line for line in _content_lines(before) if "exchanges — memory_access" in line
    )
    assert f"· {day_a_count} exchanges — memory_access" in before_fold, before
    assert "a busy day's question 0" not in before  # the day folded whole, not line-by-line

    _long_exchange(db, 200)  # later-day messages: the pool drops more of day A's front
    _long_exchange(db, 201)
    after = _render_bounded(_map(db))
    after_fold = next(line for line in _content_lines(after) if "exchanges — memory_access" in line)

    assert after_fold == before_fold  # byte-identical — a pure function of the day
    # The fold replaced the day's lines at the map's front, so everything the
    # first render showed is still there, in order.
    assert _content_lines(after)[: len(_content_lines(before))] == _content_lines(before)


def test_the_horizon_reads_the_databases_clock_not_the_machines(tmp_path) -> None:
    """The map's horizon reads the database's clock, so in eval samples with
    a pinned clock far in the past, seeded older exchanges still render.  If
    the horizon used the machine's clock, it would be far in the future and
    would exclude every seeded exchange."""
    far_past = datetime(2020, 3, 15, 14, 0, tzinfo=UTC)
    db = migrated_db(str(tmp_path / "t.db"), clock=PinnedClock(far_past))

    # Seed well past the window so several exchanges are unambiguously older
    # than it — the session cap sits AT the window, so the size stays flat at
    # the window for any total past it (#3077).
    db.config.set("CHAT_SESSION_MAX_TURNS", "5", "test")
    window_size = int(RuntimeParams(db).CHAT_HISTORY_TURNS)
    for index in range(3 * window_size):
        _exchange(db, f"old question {index}", f"old answer {index}")

    # The oldest exchanges should render because the horizon reads the DB
    # clock, which is far in the past, not the machine's (real) clock — if it
    # used the real clock, every one of these 2020 exchanges would be outside
    # the horizon and the map would render empty.  Every seeded exchange here
    # is short and outcome-less, so they collapse into one ranged line rather
    # than keeping their own text — the clock's own date is what proves the
    # horizon read the right one.
    rendered = _map(db).render()
    assert rendered, (
        "the map should render something because the horizon reads the "
        "database's clock (far in the past), not the machine's (real) clock"
    )
    assert "Mar 15" in rendered, rendered
