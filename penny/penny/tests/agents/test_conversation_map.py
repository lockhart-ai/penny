"""The conversation map's mechanism, pinned deterministically (#2752).

The map is a render over stored rows, so every test seeds rows through the
stores' own write paths (``log_message``, ``write_rule_decided``, the
browse-results append with its run id, ``mutations.record``) and asserts on
``ConversationMap.render()`` whole — never by substring.  What the mock tests
pin here is what ``make check`` owns: the line render, the window boundary
shared with the history's own grow/cut cycle (#2756/#2757), the collapse
rules, and the two boundings.  The live-model contract is the ``chat/idle``
refer-back case beside them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlmodel import col, update

from penny.agents.base import Agent
from penny.agents.conversation_map import CLIP_CHARS, ConversationMap
from penny.config_params import RuntimeParams
from penny.constants import MutationAction, MutationActor, MutationEntityType, PennyConstants
from penny.database.memory import EntryInput, LogEntryInput
from penny.database.models import MessageLog
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.utils.clock import PinnedClock
from penny.tests.schema_template import migrated_db

_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY = PennyConstants.MessageAuthor.PENNY


def _map(db) -> ConversationMap:
    return ConversationMap(db, TEST_SENDER, RuntimeParams(db))


def _exchange(db, user: str, reply: str, *, run_id: str | None = None) -> None:
    db.messages.log_message(_INCOMING, TEST_SENDER, user)
    db.messages.log_message(_OUTGOING, _PENNY, reply, recipient=TEST_SENDER, run_id=run_id)


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


def _content_lines(rendered: str) -> list[str]:
    """The map's own content lines — everything between the header and the
    fixed closing anchor."""
    if not rendered:
        return []
    return rendered.split("\n")[1:-1]


# ── The line render ──────────────────────────────────────────────────────────


def test_a_line_renders_clip_outcome_and_names(db) -> None:
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
    # The exchange's own line, older than the window — the user's own
    # words clipped at the named length, the outcome from the records, the names
    # her reply marked:
    assert any(
        f'"{ask[:CLIP_CHARS]}…"' in line
        and "read 1 page" in line
        and "saved 'remain-in-light' → albums-to-hear" in line
        and "created albums-to-hear" in line
        and "named: Tom Tom Club, The Modern Lovers" in line
        for line in lines
    ), rendered
    # A stored URL never rides the map — the name is the link's text.
    assert "example.com" not in rendered


def test_a_reply_marking_nothing_contributes_no_names(db) -> None:
    _exchange(db, "what did you find?", "nothing usable came back, honestly.")
    ids = _last_ids(db, 2)
    _filler(db, 20)
    _backdate(db, ids, 1.0)
    rendered = _map(db).render()
    assert "named:" not in rendered and "→" not in rendered


# ── The window boundary — shared with the history's grow/cut cycle (#2757) ──


def test_the_map_is_byte_identical_across_two_non_cut_turns(db) -> None:
    """Between two turns neither of which is a cut, the history window's own
    first exchange doesn't move (``Agent._history_window_size``), so the set of
    exchanges the map covers doesn't change either — the map renders
    BYTE-IDENTICAL, not merely a growing prefix (#2757)."""
    db.config.set("CHAT_HISTORY_TURNS", "2", "test")
    for index in range(4):  # total=4 is the state right after a cut: older=2
        _outcome_exchange(db, index)
    assert Agent._history_window_size(4, 2) == 2  # sanity: older = 4 - 2 = 2
    before = _map(db).render()
    assert len(_content_lines(before)) == 2

    _outcome_exchange(db, 4)  # total=5 — still growing, same 2 older exchanges
    assert Agent._history_window_size(5, 2) == 3  # older = 5 - 3 = 2, unchanged
    after = _map(db).render()

    assert after == before


def test_the_map_grows_by_the_exchanges_that_left_on_a_cut_turn(db) -> None:
    """On a cut turn — the one turn the history window's own first exchange
    moves — the map gains exactly the exchanges the history dropped, whole, at
    its end; every earlier line stays byte-identical (#2757)."""
    db.config.set("CHAT_HISTORY_TURNS", "2", "test")
    for index in range(5):  # total=5: growing, older=2 (exchanges 0, 1)
        _outcome_exchange(db, index)
    assert Agent._history_window_size(5, 2) == 3
    before = _map(db).render()
    before_lines = _content_lines(before)
    assert len(before_lines) == 2

    _outcome_exchange(db, 5)  # total=6 — the cut: window drops back to 2
    assert Agent._history_window_size(6, 2) == 2  # older = 6 - 2 = 4
    after = _map(db).render()
    after_lines = _content_lines(after)

    assert after_lines[: len(before_lines)] == before_lines  # a pure prefix
    assert len(after_lines) == 4
    assert "exchange ask 2" in after_lines[2]
    assert "exchange ask 3" in after_lines[3]


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


# ── The ceiling ──────────────────────────────────────────────────────────────


def test_past_the_ceiling_the_oldest_whole_day_collapses(db) -> None:
    """Six outcome-bearing exchanges older than a two-exchange window, five of
    them yesterday and one today, ceiling 5: the oldest day folds into ONE
    line naming its log_read read, and the map lands under the ceiling
    without a single exchange trimmed line-by-line."""
    db.config.set("CHAT_HISTORY_TURNS", "2", "test")
    for index in range(8):  # window=2: older = 8 - 2 = 6 (exchanges 0-5)
        run_id = f"{index:032x}"
        _write_entry(db, f"list-{index}", f"key-{index}", run_id)
        _exchange(db, f"exchange ask {index}", f"exchange reply {index}", run_id=run_id)
        ids = _last_ids(db, 2)
        if index < 5:
            _backdate(db, ids, 1.0)  # the oldest five: yesterday
    assert Agent._history_window_size(8, 2) == 2  # sanity: older = 8 - 2 = 6
    db.config.set("CHAT_MAP_MAX_LINES", "5", "test ceiling")

    rendered = _map(db).render()
    content_lines = list(rendered.split("\n")[1:-1])
    day_lines = [line for line in content_lines if "exchanges — log_read" in line]
    assert len(day_lines) == 1, rendered
    assert day_lines[0].startswith("- ")
    assert "log_read{'user-messages', day: '" in day_lines[0]
    assert len(content_lines) <= 5
    # Never trimmed line-by-line: every exchange the day line replaced is gone whole.
    assert "exchange ask 0" not in rendered and "exchange ask 4" not in rendered
    assert "exchange ask 5" in rendered


def test_the_horizon_reads_the_databases_clock_not_the_machines(tmp_path) -> None:
    """The map's horizon reads the database's clock, so in eval samples with
    a pinned clock far in the past, seeded older exchanges still render.  If
    the horizon used the machine's clock, it would be far in the future and
    would exclude every seeded exchange."""
    far_past = datetime(2020, 3, 15, 14, 0, tzinfo=UTC)
    db = migrated_db(str(tmp_path / "t.db"), clock=PinnedClock(far_past))

    # Seed well past the window's own grow/cut ceiling (2 * CHAT_HISTORY_TURNS - 1)
    # so several exchanges are unambiguously older than the window, whatever
    # phase of the cycle the total count lands on.
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
