"""Integration tests for the chat pre-warm (#3078).

Covers: the warm's rendered prefix is a true prefix of the next real turn's
own request, both mid-session and once the session has actually rotated; the
``ChatPrewarmTask`` gate (no warm while a collector cycle runs, never twice
for the same prefix, a fresh warm once the newest model call moves, no
``promptlog`` row ever written); and a foreground turn cancelling an
in-flight warm leaves the chat agent's own shared state untouched.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta

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
# thinking is on and the prefix does not end in a tool call/response.
# ``raw_template.render_prompt`` always sends ``add_generation_prompt=True``,
# so this is the warm's own trailing text — stripped before the prefix check.
_GENERATION_PROMPT_MARKER = "<|turn>model\n"


def _rendered_dicts(dict_messages: list[dict], tools: list[dict] | None) -> str:
    """The exact bytes ``raw_template.render_prompt`` would send for an
    already-dict-shaped message/tool list (the shape ``MockLlmClient``
    records a real request in) — through the SAME translation
    ``LlmClient.chat()``'s raw-template branch applies first, never a
    second render path."""
    translated_messages = LlmClient._translate_messages(dict_messages)
    translated_tools = LlmClient._translate_tools(tools) if tools else None
    return raw_template.render_prompt(translated_messages, translated_tools, True)


def _rendered(messages: list[LlmMessage], tools: list[dict] | None) -> str:
    """The warm's own ``list[LlmMessage]`` output, rendered the same way."""
    return _rendered_dicts([message.to_log_dict() for message in messages], tools)


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
    """Stands in for ``LlmClient`` in a ``ChatPrewarmTask`` test — only the
    two members the task actually reads/calls."""

    def __init__(self, *, raw_template_enabled: bool = True) -> None:
        self.raw_template_enabled = raw_template_enabled
        self.prewarm_requests: list[dict] = []

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
        idle_seconds = int(penny.config.runtime.CHAT_SESSION_IDLE_SECONDS)

        warm_messages, warm_tools = await penny.chat_agent.build_prewarm_prefix(
            TEST_SENDER, idle_seconds, db.clock.now()
        )
        warm_text = _rendered(warm_messages, warm_tools)
        assert warm_text.endswith(_GENERATION_PROMPT_MARKER)
        warm_prefix_text = warm_text[: -len(_GENERATION_PROMPT_MARKER)]

        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(request, "all set"),
            answers_micro_contexts=False,
        )
        await signal_server.push_message(sender=TEST_SENDER, content="what's new?")
        await signal_server.wait_for_message(timeout=10.0)

        real_request = mock_llm.requests[0]
        real_text = _rendered_dicts(real_request["messages"], real_request.get("tools"))

        assert real_text.startswith(warm_prefix_text)


class _FixedClock:
    """A controllable ``db.clock`` substitute — a plain, reassignable
    attribute (``Database.clock``), read only by stored-timestamp writes and
    anchor reads, so swapping it mid-test advances "now" for those alone."""

    def __init__(self, when: datetime) -> None:
        self._when = when

    def now(self) -> datetime:
        return self._when


@pytest.mark.asyncio
async def test_post_rotation_prefix_matches_once_the_session_actually_rotates(
    signal_server, make_config, running_penny, test_user_info, mock_llm
) -> None:
    """Once the idle gap has passed, the NEXT real turn's own anchor comes
    back rotated — assert the warm's prediction matches it (#3078: one
    anchor rule, no separate rotate flag), driven end to end.

    A turn's own incoming row is logged only AFTER its run finishes
    (``_finish_turn``, deliberately deferred so it never doubles into the
    turn's own recall) — so a turn never sees ITSELF as the live, open row;
    the rotation only becomes visible once that row is stored and a LATER
    turn reads it. Two real turns, with the idle gap advanced on
    ``db.clock`` (a plain, reassignable attribute — nothing else reads it
    besides stored-timestamp writes and anchor reads) between them, so the
    SECOND turn's own dispatch is the one the warm's prediction is checked
    against."""
    config = make_config(chat_session_idle_seconds=5)
    async with running_penny(config) as penny:
        db = penny.db
        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(
                request, "sure, see https://example.com/article for details"
            ),
            answers_micro_contexts=False,
        )
        await signal_server.push_message(sender=TEST_SENDER, content="check this out")
        await signal_server.wait_for_message(timeout=10.0)
        mock_llm.requests.clear()

        idle_seconds = int(penny.config.runtime.CHAT_SESSION_IDLE_SECONDS)
        rotated_now = db.clock.now() + timedelta(seconds=idle_seconds + 30)
        db.clock = _FixedClock(rotated_now)

        # The session has actually rotated by this point — not a weaker,
        # still-mid-session case that the prefix check below couldn't tell
        # apart from one.
        anchor = db.messages.anchor_for_arrival(TEST_SENDER, idle_seconds, db.clock.now())
        assert anchor.session_turns == 0, "the session must actually have rotated"

        warm_messages, warm_tools = await penny.chat_agent.build_prewarm_prefix(
            TEST_SENDER, idle_seconds, db.clock.now()
        )
        warm_text = _rendered(warm_messages, warm_tools)
        assert warm_text.endswith(_GENERATION_PROMPT_MARKER)
        warm_prefix_text = warm_text[: -len(_GENERATION_PROMPT_MARKER)]

        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(request, "all set"),
            answers_micro_contexts=False,
        )
        await signal_server.push_message(sender=TEST_SENDER, content="what's new?")
        await signal_server.wait_for_message(timeout=10.0)

        real_request = mock_llm.requests[0]
        real_text = _rendered_dicts(real_request["messages"], real_request.get("tools"))

        assert real_text.startswith(warm_prefix_text)


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
