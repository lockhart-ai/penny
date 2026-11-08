"""Integration tests for the chat pre-warm (#3078, #3164).

Covers: the warm's rendered prefix is a true TOKEN prefix of the next real
turn's own request, both mid-session and once the session has actually
rotated — which requires the warm's own render to carry NO generation prompt
(#3164): the real turn continues past the warm's own messages with a NEW
turn (the per-turn dynamic system message), never with ``<|turn>model``, so
a warm rendered WITH the generation prompt can never be a byte prefix of the
real request. Both prefix tests below first show that the OLD render (with
the generation prompt) fails the prefix check, then that the fixed render
(without it, ending on a gemma special token instead) passes it. Also
covers: the ``ChatPrewarmTask`` gate (no warm while a collector cycle runs,
never twice for the same prefix, a fresh warm once the newest model call
moves, no ``promptlog`` row ever written); and a foreground turn cancelling
an in-flight warm leaves the chat agent's own shared state untouched.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import select

from penny.constants import PennyConstants
from penny.database.models import PromptLog
from penny.llm import raw_template
from penny.llm.client import LlmClient
from penny.llm.models import LlmMessage, UsageTiming
from penny.scheduler.base import BackgroundScheduler
from penny.scheduler.chat_prewarm import ChatPrewarmTask
from penny.scheduler.schedules import PeriodicSchedule
from penny.tests.conftest import TEST_SENDER, wait_until

# gemma4's own fixed generation-prompt marker (confirmed by reading
# ``llm/templates/gemma4_chat_template.jinja`` directly, not assumed) when
# thinking is on and the prefix does not end in a tool call/response. A REAL
# turn's own request always carries this (``add_generation_prompt`` defaults
# True); the warm must NEVER carry it (#3164) — see the module docstring.
_GENERATION_PROMPT_MARKER = "<|turn>model\n"


def _rendered_dicts(
    dict_messages: list[dict], tools: list[dict] | None, *, add_generation_prompt: bool = True
) -> str:
    """The exact bytes ``raw_template.render_prompt`` would send for an
    already-dict-shaped message/tool list (the shape ``MockLlmClient``
    records a real request in) — through the SAME translation
    ``LlmClient.chat()``'s raw-template branch applies first, never a
    second render path. ``add_generation_prompt`` defaults True (a real
    turn's own request); the warm's own render passes False (#3164),
    mirroring ``LlmClient.render_prewarm_prompt``."""
    translated_messages = LlmClient._translate_messages(dict_messages)
    translated_tools = LlmClient._translate_tools(tools) if tools else None
    return raw_template.render_prompt(
        translated_messages, translated_tools, True, add_generation_prompt
    )


def _rendered(
    messages: list[LlmMessage], tools: list[dict] | None, *, add_generation_prompt: bool = True
) -> str:
    """The warm's own ``list[LlmMessage]`` output, rendered the same way."""
    return _rendered_dicts(
        [message.to_log_dict() for message in messages],
        tools,
        add_generation_prompt=add_generation_prompt,
    )


def _seed_closed_exchange(db, sender: str, ask: str, reply: str, at) -> None:
    ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, sender, ask, timestamp=at
    )
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        reply,
        parent_id=ask_id,
        timestamp=at + timedelta(seconds=5),
        recipient=sender,
    )


class _FakeCycleGate:
    def __init__(self, running: bool = False) -> None:
        self.running = running

    def is_cycle_running(self) -> bool:
        return self.running


class _FakeModelClient:
    """Stands in for ``LlmClient`` in a ``ChatPrewarmTask`` test — the three
    members the task actually reads/calls. ``render_prewarm_prompt`` renders
    for real (through the same ``_rendered`` helper the end-to-end tests
    above use), never a stub, so a test exercising the dedup key sees
    exactly what the real ``LlmClient.render_prewarm_prompt`` (#3133) would
    produce for the same messages/tools."""

    def __init__(self, *, raw_template_enabled: bool = True) -> None:
        self.raw_template_enabled = raw_template_enabled
        self.prewarm_requests: list[dict] = []

    def render_prewarm_prompt(self, messages: list[LlmMessage], tools: list[dict] | None) -> str:
        return _rendered(messages, tools, add_generation_prompt=False)

    async def prewarm(self, messages: list[LlmMessage], tools: list[dict] | None) -> UsageTiming:
        self.prewarm_requests.append({"messages": messages, "tools": tools})
        return UsageTiming(prompt_tokens=100, cached_tokens=0, completion_tokens=1)


@pytest.mark.asyncio
async def test_mid_session_prefix_is_a_prefix_of_the_real_next_turns_request(
    signal_server, make_config, running_penny, test_user_info, mock_llm
) -> None:
    """The warm's own render must match what a REAL turn, driven end to end,
    actually sends — never a second, hand-built implementation of the same
    path, which could only ever agree with itself (#3078 review). The past
    reply cites a URL, so the real turn's own history rewrites it to a
    handle (#2717) — exercising the exact conversion the warm must mirror."""
    config = make_config()
    async with running_penny(config) as penny:
        db = penny.db
        now = db.clock.now()
        _seed_closed_exchange(
            db,
            TEST_SENDER,
            "check this out",
            "sure, see https://example.com/article for details",
            now - timedelta(minutes=5),
        )

        warm_messages, warm_tools = await penny.chat_agent.build_prewarm_prefix(TEST_SENDER)
        # The OLD render (with the generation prompt) — kept only to show it
        # fails the prefix check: its own trailing marker is never what the
        # real turn's own continuation actually sends at that position.
        old_warm_text = _rendered(warm_messages, warm_tools, add_generation_prompt=True)
        assert old_warm_text.endswith(_GENERATION_PROMPT_MARKER)

        # The FIXED render (#3164) — no generation prompt, ending on whichever
        # gemma special token already closes the prefix's own last message.
        warm_text = _rendered(warm_messages, warm_tools, add_generation_prompt=False)
        assert not warm_text.endswith(_GENERATION_PROMPT_MARKER)
        assert warm_text.rstrip("\n").endswith(raw_template.STOP_SEQUENCE)

        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(request, "all set"),
            answers_micro_contexts=False,
        )
        await signal_server.push_message(sender=TEST_SENDER, content="what's new?")
        await signal_server.wait_for_message(timeout=10.0)

        real_request = mock_llm.requests[0]
        real_text = _rendered_dicts(real_request["messages"], real_request.get("tools"))

        assert not real_text.startswith(old_warm_text), (
            "the generation-prompt tail must never be a prefix of the real turn's prompt"
        )
        assert real_text.startswith(warm_text)


class _FixedClock:
    """A controllable clock (#3125) — mutable, never advancing on its own, so
    the idle gap it measures is stated rather than raced against real test
    execution time.  Passed into ``make_config(clock=...)`` so EVERY store
    (``db.messages`` included) shares this EXACT instance from construction
    — unlike reassigning ``db.clock`` afterward, which only the Database
    facade's own direct readers would see, never ``MessageStore._clock``
    (captured once at ``__init__``, #3125's own new ``now()`` reads:
    ``record_idle_rotation``/``record_cap_rotation``). Advanced by mutating
    ``_when`` in place, never by replacing the instance, so the background
    scheduler's own ``ChatSessionRotationTask`` (sharing this same clock)
    reads the identical value a test-driven call would.

    ``_when`` is kept NAIVE — the same convention every stored timestamp and
    every test seed literal already follows — and ``now()`` tags it UTC-aware
    on the way out, matching the real ``Clock``'s own contract (other
    renders, e.g. ``map_horizon_start``'s ``.astimezone(zone)``, assume an
    aware instant)."""

    def __init__(self, when: datetime) -> None:
        self._when = when

    def now(self) -> datetime:
        return self._when.replace(tzinfo=UTC)


def _seed_history_longer_than_the_window(db, sender: str, start: datetime) -> None:
    """Four closed exchanges, a minute apart — more than a small
    ``CHAT_HISTORY_TURNS`` override, so a real cut actually drops the
    oldest ones instead of being masked by ``total <= window``."""
    for n in range(4):
        _seed_closed_exchange(
            db, sender, f"old ask {n}", f"old reply {n}", start + timedelta(minutes=n)
        )


@pytest.mark.asyncio
async def test_post_rotation_prefix_matches_once_the_session_actually_rotates(
    signal_server, make_config, running_penny, test_user_info, mock_llm
) -> None:
    """Once the idle gap has passed, the NEXT real turn's own anchor comes
    back rotated — assert the warm's prediction matches it, with ONE real
    turn after the gap (#3125: the prior version drove two real turns,
    which encoded the one-message lag rather than catching it — the first
    turn after a gap is the one whose own anchor must already be rotated
    when ITS OWN dispatch reads it, not the one after).

    The rotation is driven explicitly here (``record_idle_rotation``, the
    same write the idle scheduler task makes) so the warm's prefix and the
    real turn's own inline rotation (``_dispatch_turn``'s own call, before
    its read) agree on an ALREADY-settled boundary — calling the write
    again from the turn itself is then the no-op #3125 designs for."""
    clock = _FixedClock(datetime(2026, 1, 1, 9, 0))
    config = make_config(chat_session_idle_seconds=5, chat_history_turns=2, clock=clock)
    async with running_penny(config) as penny:
        db = penny.db
        start = clock.now().replace(tzinfo=None)
        _seed_history_longer_than_the_window(db, TEST_SENDER, start)
        clock._when = start + timedelta(minutes=10, seconds=5)  # past the 5s idle gap

        rotated = db.messages.record_idle_rotation(
            TEST_SENDER, int(config.runtime.CHAT_SESSION_IDLE_SECONDS)
        )
        assert rotated is True, "the session must actually have rotated"
        anchor = db.messages.session_anchor(TEST_SENDER)
        assert anchor.session_turns == 0

        warm_messages, warm_tools = await penny.chat_agent.build_prewarm_prefix(TEST_SENDER)
        # Same demonstration as the mid-session test above: the OLD render
        # (with the generation prompt) fails the prefix check post-rotation
        # too — the fix is not specific to one session shape.
        old_warm_text = _rendered(warm_messages, warm_tools, add_generation_prompt=True)
        assert old_warm_text.endswith(_GENERATION_PROMPT_MARKER)

        warm_text = _rendered(warm_messages, warm_tools, add_generation_prompt=False)
        assert not warm_text.endswith(_GENERATION_PROMPT_MARKER)
        assert warm_text.rstrip("\n").endswith(raw_template.STOP_SEQUENCE)

        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(request, "all set"),
            answers_micro_contexts=False,
        )
        await signal_server.push_message(sender=TEST_SENDER, content="what's new?")
        await signal_server.wait_for_message(timeout=10.0)

        real_request = mock_llm.requests[0]
        real_text = _rendered_dicts(real_request["messages"], real_request.get("tools"))

        assert not real_text.startswith(old_warm_text), (
            "the generation-prompt tail must never be a prefix of the real turn's prompt"
        )
        assert real_text.startswith(warm_text)
        # The history actually cut to CHAT_HISTORY_TURNS (2): the two oldest
        # seeded exchanges aged out, the two newest survived.
        assert "old ask 0" not in real_text
        assert "old ask 1" not in real_text
        assert "old ask 2" in real_text
        assert "old ask 3" in real_text


@pytest.mark.asyncio
async def test_first_turn_after_a_gap_cuts_history_even_with_no_rotation_task_run(
    signal_server, make_config, running_penny, test_user_info, mock_llm
) -> None:
    """The real turn's own inline rotation (``_dispatch_turn``'s own
    ``record_idle_rotation``/``record_cap_rotation`` calls, before its one
    read) is what guarantees this even when nothing else — the idle
    scheduler task included — ever got to it (#3125): a restart inside the
    window, the eval harness driving turns with no scheduler, or simply a
    tick that hasn't landed yet. No rotation is ever recorded explicitly
    here; the FIRST real turn after the gap does it for itself."""
    clock = _FixedClock(datetime(2026, 1, 1, 9, 0))
    config = make_config(chat_session_idle_seconds=5, chat_history_turns=2, clock=clock)
    async with running_penny(config) as penny:
        db = penny.db
        start = clock.now().replace(tzinfo=None)
        _seed_history_longer_than_the_window(db, TEST_SENDER, start)
        clock._when = start + timedelta(minutes=10, seconds=5)  # past the 5s idle gap

        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(request, "all set"),
            answers_micro_contexts=False,
        )
        await signal_server.push_message(sender=TEST_SENDER, content="what's new?")
        await signal_server.wait_for_message(timeout=10.0)

        real_request = mock_llm.requests[0]
        real_text = _rendered_dicts(real_request["messages"], real_request.get("tools"))

        assert "old ask 0" not in real_text
        assert "old ask 1" not in real_text
        assert "old ask 2" in real_text
        assert "old ask 3" in real_text


@pytest.mark.asyncio
async def test_no_warm_while_a_collector_cycle_is_running(
    make_config, running_penny, test_user_info, mock_llm
) -> None:
    config = make_config()
    async with running_penny(config) as penny:
        _seed_closed_exchange(
            penny.db, TEST_SENDER, "hi", "hey", penny.db.clock.now() - timedelta(minutes=5)
        )
        model_client = _FakeModelClient()
        task = ChatPrewarmTask(
            chat_agent=penny.chat_agent,
            model_client=model_client,
            cycle_gate=_FakeCycleGate(running=True),
            runtime=penny.config.runtime,
        )

        fired = await task.execute()

        assert fired is False
        assert model_client.prewarm_requests == []


@pytest.mark.asyncio
async def test_no_second_warm_for_an_unchanged_key(
    make_config, running_penny, test_user_info, mock_llm
) -> None:
    config = make_config()
    async with running_penny(config) as penny:
        _seed_closed_exchange(
            penny.db, TEST_SENDER, "hi", "hey", penny.db.clock.now() - timedelta(minutes=5)
        )
        model_client = _FakeModelClient()
        task = ChatPrewarmTask(
            chat_agent=penny.chat_agent,
            model_client=model_client,
            cycle_gate=_FakeCycleGate(running=False),
            runtime=penny.config.runtime,
        )

        first = await task.execute()
        second = await task.execute()

        assert first is True
        assert second is False
        assert len(model_client.prewarm_requests) == 1


@pytest.mark.asyncio
async def test_no_second_warm_across_idle_ticks_when_history_carries_a_past_tool_call(
    make_config, running_penny, test_user_info, mock_llm
) -> None:
    """#3133: a past run that called a tool mints a FRESH id for that call
    every time history is rebuilt (``new_history_replay_call_id``, #2915) —
    bytes that never reach the RENDERED prompt (the raw template only
    matches an id to pair a call with its result, never prints the id
    itself) but DO land in ``LlmMessage.to_log_dict()``. Hashing that
    pre-wire shape — the OLD ``_prefix_hash`` — made the dedup key differ on
    every tick even though the actual prefix never changed. Reproduces the
    prod symptom (hash different every ~65s, prompt/cached tokens
    identical): three ticks, no state change between them, the clock
    advanced a real idle gap each time — must still send only once."""
    clock = _FixedClock(datetime(2026, 1, 1, 9, 0))
    config = make_config(clock=clock)
    async with running_penny(config) as penny:
        db = penny.db
        start = clock.now().replace(tzinfo=None)
        ask_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "check the weather and save it",
            timestamp=start - timedelta(minutes=5),
        )
        run_id = "warm-test-run-1"
        first_messages = [{"role": "user", "content": "check the weather and save it"}]
        first_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "browse",
                                    "arguments": json.dumps({"url": "https://example.com"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        db.messages.log_prompt(
            model="m", messages=first_messages, response=first_response, run_id=run_id
        )
        db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            "Checked and saved — sunny, 18C!",
            parent_id=ask_id,
            run_id=run_id,
            timestamp=start - timedelta(minutes=4),
            recipient=TEST_SENDER,
        )

        model_client = _FakeModelClient()
        task = ChatPrewarmTask(
            chat_agent=penny.chat_agent,
            model_client=model_client,
            cycle_gate=_FakeCycleGate(running=False),
            runtime=penny.config.runtime,
        )

        assert await task.execute() is True
        clock._when = clock._when + timedelta(seconds=30)
        assert await task.execute() is False
        clock._when = clock._when + timedelta(seconds=30)
        assert await task.execute() is False

        assert len(model_client.prewarm_requests) == 1


@pytest.mark.asyncio
async def test_a_new_warm_fires_once_the_newest_model_call_moves(
    make_config, running_penny, test_user_info, mock_llm
) -> None:
    """A collector cycle's own call leaves the chat prefix's TEXT unchanged
    but can evict it from the server's cache — the dedup key's newest-call
    timestamp is what makes its completion trigger a fresh warm anyway."""
    config = make_config()
    async with running_penny(config) as penny:
        _seed_closed_exchange(
            penny.db, TEST_SENDER, "hi", "hey", penny.db.clock.now() - timedelta(minutes=5)
        )
        model_client = _FakeModelClient()
        task = ChatPrewarmTask(
            chat_agent=penny.chat_agent,
            model_client=model_client,
            cycle_gate=_FakeCycleGate(running=False),
            runtime=penny.config.runtime,
        )
        assert await task.execute() is True

        penny.db.messages.log_prompt(
            model="test-model",
            messages=[],
            response={},
            agent_name="collector",
            run_id="run-cycle",
            run_target="some-job",
        )

        assert await task.execute() is True
        assert len(model_client.prewarm_requests) == 2


@pytest.mark.asyncio
async def test_no_promptlog_row_is_ever_written_by_the_warm(
    make_config, running_penny, test_user_info, mock_llm
) -> None:
    config = make_config()
    async with running_penny(config) as penny:
        _seed_closed_exchange(
            penny.db, TEST_SENDER, "hi", "hey", penny.db.clock.now() - timedelta(minutes=5)
        )
        with penny.db.get_session() as session:
            before = len(session.exec(select(PromptLog)).all())

        model_client = _FakeModelClient()
        task = ChatPrewarmTask(
            chat_agent=penny.chat_agent,
            model_client=model_client,
            cycle_gate=_FakeCycleGate(running=False),
            runtime=penny.config.runtime,
        )
        assert await task.execute() is True

        with penny.db.get_session() as session:
            after = len(session.exec(select(PromptLog)).all())
        assert after == before


@pytest.mark.asyncio
async def test_foreground_start_cancels_an_in_flight_warm_leaving_no_shared_state(
    make_config, running_penny, test_user_info, mock_llm
) -> None:
    """A user message arriving mid-warm cancels it — the scheduler's existing
    foreground-priority cancel (``notify_foreground_start``), no new
    mechanism (#3078). Nothing of the warm's own build is left on the chat
    agent afterward: it never wrote ``self._session_state`` or installed
    tools onto ``self._tool_registry`` in the first place.

    A fresh, isolated ``BackgroundScheduler`` — not ``penny.scheduler``'s own
    (whose chat-prewarm schedule stays disabled by the default test
    overrides) — the same pattern ``test_scheduler.py``'s own cancellation
    test uses, so this test's timing never races the real scheduler's."""
    config = make_config()
    async with running_penny(config) as penny:
        _seed_closed_exchange(
            penny.db, TEST_SENDER, "hi", "hey", penny.db.clock.now() - timedelta(minutes=5)
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        class _SlowModelClient(_FakeModelClient):
            async def prewarm(self, messages, tools):
                entered.set()
                await release.wait()
                return await super().prewarm(messages, tools)

        model_client = _SlowModelClient()
        task = ChatPrewarmTask(
            chat_agent=penny.chat_agent,
            model_client=model_client,
            cycle_gate=_FakeCycleGate(running=False),
            runtime=penny.config.runtime,
        )
        scheduler = BackgroundScheduler(
            schedules=[
                PeriodicSchedule(agent=task, interval=lambda: 9999.0, idle_threshold=lambda: 0.0)
            ],
            tick_interval=0.01,
        )
        run_task = asyncio.create_task(scheduler.run())
        try:
            await wait_until(lambda: entered.is_set(), timeout=5.0)
            scheduler.notify_foreground_start()
            # Deterministic, not a sleep: the scheduler's own ``finally`` clears
            # ``_active_task`` the moment the cancellation it just issued has
            # actually landed and been processed.
            await wait_until(lambda: scheduler._active_task is None, timeout=5.0)

            assert model_client.prewarm_requests == []  # never actually completed
        finally:
            release.set()
            scheduler.stop()
            run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await run_task

        # Nothing the warm would have written is left lying on the agent —
        # it never touched this shared state to begin with.
        assert penny.chat_agent._session_state is None
        assert penny.chat_agent._related_exchanges is None
        assert penny.chat_agent._tool_registry.get_all() == []

        with penny.db.get_session() as session:
            rows = session.exec(select(PromptLog)).all()
        assert rows == []
