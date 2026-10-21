"""The delivery turn, end to end through the drainer (#2656).

A notifying cycle's report row reaches the user through the SEND-QUEUE DRAINER:
the drainer builds the :class:`JobReport` from the cycle's written entries, runs
ONE chat-agent turn on it, and what the turn writes is what gets gated, stored on
the row and delivered through the channel.  These tests drive that whole step
through the drainer's public ``execute()`` — the real chat agent, the real
send-queue store, a real LLM client behind the mock SDK — against a stub channel,
which is the system boundary (the channel owns delivery and logging in
production).

The chat agent's reply here is the model's to write; what the tests pin is the
mechanism around it: the turn read the rendered report, the message is attributed
to the collection that queued the row, the row's own audit trail says what
happened, and a compose that produced nothing usable is retried bounded and then
failed VISIBLY — never silently dropped, never counted as a message the user got.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import cast

import pytest
from sqlmodel import Session, col, select

from penny.agents.chat import ChatAgent
from penny.channels.base import MessageChannel
from penny.constants import ChatPromptType, CycleTrigger, PennyConstants
from penny.database.memory import EntryInput
from penny.database.models import PromptLog, SendQueueItem
from penny.llm.client import LlmClient
from penny.scheduler.send_queue_drainer import SendQueueDrainer
from penny.tests.conftest import require_memory
from penny.tools.link_handles import CITATION_LINE


class _StubChannel:
    """Captures what the drainer delivers — the boundary the real channel sits at."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_response(self, **kwargs) -> int | None:
        self.sent.append(kwargs)
        return 1


def _llm(db) -> LlmClient:
    return LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        db=db,
        max_retries=1,
        retry_delay=0.0,
    )


def _make_delivery(db, test_config) -> tuple[SendQueueDrainer, _StubChannel]:
    chat = ChatAgent(
        model_client=_llm(db),
        db=db,
        config=test_config,
        embedding_model_client=_llm(None),
    )
    drainer = SendQueueDrainer(db=db, config=test_config, chat_agent=chat)
    channel = _StubChannel()
    # The stub is the test double at the delivery boundary; the real channel type is
    # what production binds, and the drainer reads nothing else off it.
    drainer.set_channel(cast(MessageChannel, channel))
    return drainer, channel


def _register_user(db) -> None:
    db.users.save_info(
        sender="+15551230000",
        name="Test User",
        location="Seattle, WA",
        timezone="America/Los_Angeles",
        date_of_birth="1990-01-01",
    )


def _seed_cycle_writes(db, run_id: str) -> None:
    """A job plus the entry its cycle wrote — the report's whole payload."""
    _register_user(db)
    db.memories.create_collection("indie-metroidvanias", "Indie metroidvania releases", notify=True)
    require_memory(db, "indie-metroidvanias").write_rule_decided(
        [
            EntryInput(
                key="Cinder Drift",
                content="Cinder Drift — a hand-drawn metroidvania. https://ex.example/cd",
            )
        ],
        author="producer",
        run_id=run_id,
    )


def _enqueue(db, run_id: str, *, trigger: CycleTrigger = CycleTrigger.ON_DEMAND):
    return db.send_queue.enqueue_report("indie-metroidvanias", trigger, run_id)


_REPORT_PROMPT_HEAD = 'Your job "indie-metroidvanias" just finished a run'


def _reply_handler(mock_llm, reply: str):
    """Answer ONLY the delivery turn — anything else is a test bug made loud."""

    def handler(request: dict, count: int):
        user_turns = [m.get("content", "") for m in request["messages"] if m.get("role") == "user"]
        assert any(str(c).startswith(_REPORT_PROMPT_HEAD) for c in user_turns), (
            f"the mock was asked something other than the delivery turn: {user_turns!r}"
        )
        return mock_llm._make_text_response(request, reply)

    return handler


@pytest.mark.asyncio
async def test_a_report_composes_through_one_chat_turn_and_delivers_once(db, test_config, mock_llm):
    """THE DELIVERY (#2656): a report row goes in with no content; the drainer builds
    the report from the cycle's entries, the chat agent writes the message with its
    whole context, the message is stored on the row, and exactly one send reaches
    the channel — attributed to the collection whose cycle produced it, with the
    turn's own run id stamped on the delivery (#2630's pairing)."""
    _seed_cycle_writes(db, "cycle-run-1")
    row_id = _enqueue(db, "cycle-run-1")
    reply = "Hey! Cinder Drift just dropped — a hand-drawn metroidvania. https://ex.example/cd"
    mock_llm.set_response_handler(_reply_handler(mock_llm, reply))
    drainer, channel = _make_delivery(db, test_config)

    assert await drainer.execute() is True
    assert len(channel.sent) == 1
    sent = channel.sent[0]
    assert sent["content"] == reply
    assert sent["author"] == "indie-metroidvanias"
    assert sent["mechanism"] == "indie-metroidvanias"
    assert sent["recipient"] == "+15551230000"
    assert sent["run_id"] is not None and len(str(sent["run_id"])) == 32

    # The queue's own audit trail: the row carries what went out and that it went.
    with Session(db.engine) as session:
        row = session.get(SendQueueItem, row_id)
        assert row is not None
        assert row.content == reply
        assert row.sent_at is not None
    assert db.send_queue.next_pending() is None

    # The delivery turn is its own ledger-visible run on the chat agent, under the
    # report's own prompt type.
    with Session(db.engine) as session:
        report_rows = list(
            session.exec(
                select(PromptLog).where(
                    col(PromptLog.prompt_type) == ChatPromptType.JOB_REPORT.value
                )
            ).all()
        )
    assert report_rows, "the delivery turn logged under its own prompt type"
    assert all(row.agent_name == "chat" for row in report_rows)


@pytest.mark.asyncio
async def test_the_delivery_turn_reads_the_report_whole(db, test_config, mock_llm):
    """What the chat agent was shown is the deterministic template, whole: the job's
    name, what it is for, why it ran, and every written entry with its key and FULL
    content — asserted whole off the promptlog row the run itself logged."""
    _seed_cycle_writes(db, "cycle-run-1")
    _enqueue(db, "cycle-run-1")
    mock_llm.set_response_handler(_reply_handler(mock_llm, "On it — Cinder Drift is out."))
    drainer, _channel = _make_delivery(db, test_config)

    await drainer.execute()

    with Session(db.engine) as session:
        rows = list(
            session.exec(select(PromptLog).where(col(PromptLog.prompt_type) == "job_report")).all()
        )
    assert rows, "the delivery turn logged its prompt"
    user_turn = next(
        message["content"]
        for row in rows
        for message in json.loads(row.messages)
        if message.get("role") == "user"
    )
    assert user_turn == (
        'Your job "indie-metroidvanias" just finished a run and asked to tell the user.\n'
        "What the job is for: Indie metroidvania releases\n"
        "It ran just now because the user asked it to run.\n"
        "\n"
        "## What the job wrote this run\n"
        "These entries are what the job kept for the user this run — the message "
        "tells them about every one.\n"
        # The entry's stored URL renders as the delivery turn's handle (#2717), and
        # the one plain line telling the model how to cite rides after the entries.
        "- into `indie-metroidvanias`, under `Cinder Drift`: Cinder Drift — a hand-drawn "
        f"metroidvania. [1]\n{CITATION_LINE}\n"
        "\n"
        "Write the one message to send the user about this run, from what the job wrote. It "
        "can be as long as it takes to cover every entry — a job's report isn't a quick "
        "text."
    )


@pytest.mark.asyncio
async def test_an_unusable_reply_is_retried_bounded_then_failed_visibly(db, test_config, mock_llm):
    """A turn that produced nothing whole is a COMPOSE failure: the row's attempts
    count up while it stays pending, and once the budget is spent the row is stamped
    ``failed_at`` — kept, never delivered, and the pending queries exclude it
    structurally so the drainer can never pick it up again.  Visible degradation,
    never silence and never a message the user never received."""
    _seed_cycle_writes(db, "cycle-run-1")
    row_id = _enqueue(db, "cycle-run-1")
    # The unusable reply is one the turn's own loop ACCEPTS (a bare URL — a
    # complete-enough draw for the agent loop) but the send gate's whole-message rule
    # rejects: the loop owns what a valid DRAW is, the gate owns what a sendable
    # MESSAGE is.
    mock_llm.set_response_handler(_reply_handler(mock_llm, "https://ex.example/cd"))
    drainer, channel = _make_delivery(db, test_config)

    for attempt in range(1, PennyConstants.COLLECTOR_RETRY_ATTEMPTS):
        assert await drainer.execute() is False
        assert channel.sent == []
        with Session(db.engine) as session:
            row = session.get(SendQueueItem, row_id)
            assert row is not None
            assert row.attempts == attempt
            assert row.failed_at is None
        assert db.send_queue.next_pending() is not None

    # The budget's last attempt stamps the row failed — terminally.
    assert await drainer.execute() is False
    with Session(db.engine) as session:
        row = session.get(SendQueueItem, row_id)
        assert row is not None
        assert row.failed_at is not None
    assert db.send_queue.next_pending() is None


@pytest.mark.asyncio
async def test_a_withheld_report_is_a_decline_not_a_failure(db, test_config, mock_llm):
    """A reply the send gate withholds ON PURPOSE is a decision, not a retry: the row
    is stamped cancelled and no attempt is counted, where the not-whole reply above
    counts against the budget.  Here the withhold is the mute taken between the
    enqueue and the compose — the lane says WHEN a message goes out, not WHETHER.
    (A refusal-shaped reply is screened earlier still, by the turn's own loop; the
    gate's refusal arm guards what arrives past it.)"""
    _seed_cycle_writes(db, "cycle-run-1")
    row_id = _enqueue(db, "cycle-run-1")
    db.users.set_muted("+15551230000")
    mock_llm.set_response_handler(_reply_handler(mock_llm, "Cinder Drift is out!"))
    drainer, channel = _make_delivery(db, test_config)

    assert await drainer.execute() is False
    assert channel.sent == []
    with Session(db.engine) as session:
        row = session.get(SendQueueItem, row_id)
        assert row is not None
        assert row.cancelled_at is not None
        assert row.failed_at is None
        assert row.attempts == 0
    assert db.send_queue.next_pending() is None


@pytest.mark.asyncio
async def test_a_cancelled_compose_leaves_the_row_untouched(db, test_config, mock_llm):
    """Foreground priority cancels the compose mid-turn like any background work:
    the row stays pending with no attempt counted — the attempt did not fail, it was
    interrupted — and the next idle window tries again."""
    _seed_cycle_writes(db, "cycle-run-1")
    row_id = _enqueue(db, "cycle-run-1")

    def cancelling_handler(request: dict, count: int):
        raise asyncio.CancelledError

    mock_llm.set_response_handler(cancelling_handler)
    drainer, channel = _make_delivery(db, test_config)

    with pytest.raises(asyncio.CancelledError):
        await drainer.execute()

    assert channel.sent == []
    with Session(db.engine) as session:
        row = session.get(SendQueueItem, row_id)
        assert row is not None
        assert row.attempts == 0
        assert row.failed_at is None
        assert row.cancelled_at is None
    assert db.send_queue.next_pending() is not None


@pytest.mark.asyncio
async def test_an_ordinary_row_never_enters_the_model(db, test_config, mock_llm):
    """A row that already carries its content is delivered with no model call at
    all — the compose step is only ever a report row's, and the mock answers
    nothing else loudly."""
    _register_user(db)
    db.send_queue.enqueue(content="a queued status", collection="tide-times")
    mock_llm.set_response_handler(
        lambda request, count: pytest.fail("the drainer called the model for an ordinary row")
    )
    drainer, channel = _make_delivery(db, test_config)

    assert await drainer.execute() is True
    assert len(channel.sent) == 1
    assert channel.sent[0]["content"] == "a queued status"
    assert channel.sent[0]["mechanism"] == "tide-times"
    assert channel.sent[0]["run_id"] is None


@pytest.mark.asyncio
async def test_a_report_for_a_vanished_collection_never_reaches_the_model(
    db, test_config, mock_llm
):
    """A collection archived before its report composes already had its pending rows
    cancelled at the archive chokepoint (#1634) — so the report never composes and
    the model is never asked.  The drainer's own vanished-collection guard (a row
    that survived without its collection, e.g. a hand-edited store) sits behind
    that: build_job_report names the failure instead of composing from a
    half-assembled report."""
    _register_user(db)
    db.memories.create_collection("gone-job", "a job that will not exist at compose time")
    row_id = db.send_queue.enqueue_report("gone-job", CycleTrigger.CADENCE, uuid.uuid4().hex)
    db.memories.archive("gone-job")

    mock_llm.set_response_handler(
        lambda request, count: pytest.fail("the drainer called the model for a vanished collection")
    )
    drainer, channel = _make_delivery(db, test_config)
    assert await drainer.execute() is False
    assert channel.sent == []
    with Session(db.engine) as session:
        row = session.get(SendQueueItem, row_id)
        assert row is not None
        assert row.cancelled_at is not None
        assert row.failed_at is None
        assert row.attempts == 0
