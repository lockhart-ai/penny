"""Tests for the iOS channel registration and durable outbox contract."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import pytest
from sqlmodel import Session

from penny.channels.ios.apns import ApnsClient, ApnsConfig, ApnsEnvironment, ApnsError
from penny.channels.ios.channel import PUSH_GREETING_TITLE, TEST_PUSH_MESSAGE, IosChannel
from penny.channels.ios.models import (
    IOS_MSG_TYPE_ACK,
    IOS_MSG_TYPE_PULL,
    IOS_MSG_TYPE_REGISTER,
    IOS_RESP_TYPE_MESSAGES,
    IOS_RESP_TYPE_OUTBOX_CHANGED,
    IOS_RESP_TYPE_REGISTERED,
)
from penny.constants import ChannelType
from penny.database import Database
from penny.database.memory import Inclusion, RecallMode
from penny.database.migrate import migrate
from penny.database.models import Schedule


class FakeWs:
    """Minimal websocket double for unit tests."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


class FakeApns:
    """Records preview notifications without touching the network."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False

    async def send_preview(self, **kwargs) -> None:
        self.sent.append(kwargs)

    async def close(self) -> None:
        self.closed = True


class RejectingApns(FakeApns):
    """APNs double that rejects every preview as an invalid token."""

    async def send_preview(self, **kwargs) -> None:
        self.sent.append(kwargs)
        raise ApnsError(400, "BadDeviceToken")


def _make_db(tmp_path) -> Database:
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    migrate(db_path)
    return db


def _make_channel(db: Database, apns=None, is_primary: bool = True) -> IosChannel:
    return IosChannel(
        host="localhost",
        port=9999,
        message_agent=MagicMock(),
        db=db,
        pairing_token="pair-me",
        apns_client=apns,
        is_primary_channel=is_primary,
    )


async def _ios_admin_request(channel: IosChannel, payload: dict) -> dict:
    ws = FakeWs()
    await channel._process_raw_message(
        cast(Any, ws),
        json.dumps(payload),
        "ios-keychain-id",
    )
    return ws.sent[0]


@pytest.mark.asyncio
async def test_listen_returns_after_close(tmp_path):
    db = _make_db(tmp_path)
    channel = IosChannel(host="localhost", port=0, message_agent=MagicMock(), db=db)
    task = asyncio.create_task(channel.listen())
    try:
        await asyncio.wait_for(_wait_for_server(channel), timeout=2.0)
        await channel.close()
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def _wait_for_server(channel: IosChannel) -> None:
    while channel._server is None:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_register_creates_default_ios_device_and_registration(tmp_path):
    db = _make_db(tmp_path)
    channel = _make_channel(db)
    ws = FakeWs()
    server_ws = cast(Any, ws)

    device_id = await channel._handle_register(
        server_ws,
        {
            "type": IOS_MSG_TYPE_REGISTER,
            "device_id": "ios-keychain-id",
            "label": "Robert's iPhone",
            "pairing_token": "pair-me",
            "apns_token": "apns-token",
            "apns_environment": "sandbox",
            "app_version": "1.0",
        },
    )

    assert device_id == "ios-keychain-id"
    device = db.devices.get_by_identifier("ios-keychain-id")
    assert device is not None
    assert device.channel_type == ChannelType.IOS
    assert device.is_default is True
    assert db.devices.get_default_identifier() == "ios-keychain-id"
    assert device.id is not None
    registration = db.ios.get_registration(device.id)
    assert registration is not None
    assert registration.apns_token == "apns-token"
    assert ws.sent[-1]["type"] == IOS_RESP_TYPE_REGISTERED
    assert ws.sent[-1]["is_default"] is True
    assert ws.sent[-1]["pending_count"] == 0


@pytest.mark.asyncio
async def test_sidecar_registration_does_not_steal_default_device(tmp_path):
    """In sidecar mode (Signal primary), an iOS register must not claim the default."""
    db = _make_db(tmp_path)
    db.devices.register(ChannelType.SIGNAL, "+15550000000", "Signal", is_default=True)
    channel = _make_channel(db, is_primary=False)
    ws = FakeWs()
    server_ws = cast(Any, ws)

    device_id = await channel._handle_register(
        server_ws,
        {
            "type": IOS_MSG_TYPE_REGISTER,
            "device_id": "ios-keychain-id",
            "label": "iPhone",
            "pairing_token": "pair-me",
            "apns_token": "apns-token",
        },
    )

    assert device_id == "ios-keychain-id"
    device = db.devices.get_by_identifier("ios-keychain-id")
    assert device is not None and device.id is not None
    assert device.is_default is False
    assert db.devices.get_default_identifier() == "+15550000000"
    registration = db.ios.get_registration(device.id)
    assert registration is not None
    assert registration.apns_token == "apns-token"
    assert ws.sent[-1]["type"] == IOS_RESP_TYPE_REGISTERED
    assert ws.sent[-1]["is_default"] is False


@pytest.mark.asyncio
async def test_admin_prompt_logs_request_returns_unfiltered_runs(tmp_path):
    db = _make_db(tmp_path)
    channel = _make_channel(db)
    db.messages.log_prompt(
        model="test-model",
        messages=[{"role": "user", "content": "chat"}],
        response={"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        agent_name="chat",
        run_id="run-chat",
        duration_ms=100,
    )
    db.messages.log_prompt(
        model="test-model",
        messages=[{"role": "user", "content": "collector"}],
        response={"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7}},
        agent_name="collector",
        run_id="run-collector",
        run_target="recipes",
        duration_ms=200,
    )

    response = await _ios_admin_request(channel, {"type": "prompt_logs_request"})

    assert response["type"] == "prompt_logs_response"
    assert [run["run_id"] for run in response["runs"]] == ["run-collector", "run-chat"]
    assert {run["agent_name"] for run in response["runs"]} == {"chat", "collector"}
    assert response["has_more"] is False


@pytest.mark.asyncio
async def test_admin_schedules_request_returns_existing_schedules(tmp_path, monkeypatch):
    db = _make_db(tmp_path)
    monkeypatch.setattr(db.users, "get_primary_sender", lambda: "testuser")
    channel = _make_channel(db)
    with Session(db.engine) as session:
        session.add(
            Schedule(
                user_id="testuser",
                user_timezone="America/Vancouver",
                cron_expression="0 9 * * *",
                prompt_text="check the news",
                timing_description="daily at 9",
            )
        )
        session.commit()

    response = await _ios_admin_request(channel, {"type": "schedules_request"})

    assert response["type"] == "schedules_response"
    assert response["error"] is None
    assert response["schedules"][0]["prompt_text"] == "check the news"
    assert response["schedules"][0]["cron_expression"] == "0 9 * * *"


@pytest.mark.asyncio
async def test_admin_memories_request_returns_memory_records(tmp_path):
    db = _make_db(tmp_path)
    channel = _make_channel(db)
    db.memories.create_collection(
        "recipes",
        "Meal notes",
        Inclusion.RELEVANT,
        RecallMode.RECENT,
        intent="Remember food preferences",
    )

    response = await _ios_admin_request(channel, {"type": "memories_request"})

    assert response["type"] == "memories_response"
    names = {memory["name"] for memory in response["memories"]}
    assert "recipes" in names


@pytest.mark.asyncio
async def test_admin_config_request_returns_runtime_params(tmp_path):
    db = _make_db(tmp_path)
    channel = _make_channel(db)

    response = await _ios_admin_request(channel, {"type": "config_request"})

    assert response["type"] == "config_response"
    assert response["params"]
    assert {"key", "value", "default", "description", "type", "group"}.issubset(
        response["params"][0]
    )


@pytest.mark.asyncio
async def test_send_raw_queues_outbox_and_sends_push_preview_when_disconnected(tmp_path, caplog):
    db = _make_db(tmp_path)
    apns = FakeApns()
    channel = _make_channel(db, apns=apns)
    ws = FakeWs()
    server_ws = cast(Any, ws)
    await channel._handle_register(
        server_ws,
        {
            "type": IOS_MSG_TYPE_REGISTER,
            "device_id": "ios-keychain-id",
            "label": "iPhone",
            "pairing_token": "pair-me",
            "apns_token": "apns-token",
        },
    )
    channel._connections.clear()

    with caplog.at_level("INFO", logger="penny.channels.ios.channel"):
        external_id = await channel._send_raw(
            "ios-keychain-id",
            "**Found a fare drop.** Vancouver to Tokyo is cheaper today. https://example.com",
            source_name="flight-deals",
        )

    assert external_id is not None
    assert any(
        "Sending iOS preview notification to APNs" in record.message
        and f"outbox_id={external_id}" in record.message
        and "source_name=flight-deals" in record.message
        for record in caplog.records
    )
    device = db.devices.get_by_identifier("ios-keychain-id")
    assert device is not None and device.id is not None
    pending = db.ios.pending_for_device(device.id)
    assert len(pending) == 1
    assert pending[0].source_type == "collector"
    assert pending[0].source_name == "flight-deals"
    assert pending[0].push_title == PUSH_GREETING_TITLE
    assert "Found a fare drop" in pending[0].push_summary
    assert apns.sent[0]["device_token"] == "apns-token"
    assert apns.sent[0]["outbox_id"] == external_id
    assert apns.sent[0]["title"] == PUSH_GREETING_TITLE
    assert apns.sent[0]["body"] == "Found a fare drop."
    assert apns.sent[0]["badge"] == 1
    assert apns.sent[0]["environment"] == "sandbox"


@pytest.mark.asyncio
async def test_send_raw_forwards_production_environment_to_apns(tmp_path):
    db = _make_db(tmp_path)
    apns = FakeApns()
    channel = _make_channel(db, apns=apns)
    ws = FakeWs()
    server_ws = cast(Any, ws)
    await channel._handle_register(
        server_ws,
        {
            "type": IOS_MSG_TYPE_REGISTER,
            "device_id": "ios-keychain-id",
            "label": "iPhone",
            "pairing_token": "pair-me",
            "apns_token": "apns-token",
            "apns_environment": "production",
        },
    )
    channel._connections.clear()

    await channel._send_raw("ios-keychain-id", "hello from Penny", source_name="notifier")

    device = db.devices.get_by_identifier("ios-keychain-id")
    assert device is not None and device.id is not None
    registration = db.ios.get_registration(device.id)
    assert registration is not None and registration.apns_environment == "production"
    assert apns.sent[0]["environment"] == "production"


@pytest.mark.asyncio
async def test_send_preview_selects_host_from_device_environment(monkeypatch, caplog):
    requests: list[dict] = []

    class FakeHttp:
        async def post(self, url, *, headers, json):
            requests.append({"url": url})
            return httpx.Response(200)

    client = object.__new__(ApnsClient)
    client._config = ApnsConfig(
        team_id="TEAMID",
        key_id="KEYID",
        key_path="/unused/AuthKey_KEYID.p8",
        bundle_id="com.example.Penny",
        sandbox=True,  # global default host is sandbox
    )
    client._http = FakeHttp()
    monkeypatch.setattr(
        client,
        "_provider_token",
        lambda environment=None: "provider-token",
    )

    async def send(environment):
        await client.send_preview(
            device_token="device-token",
            title=PUSH_GREETING_TITLE,
            body="hi",
            badge=1,
            outbox_id=1,
            source_type=None,
            source_name=None,
            environment=environment,
        )

    await send("production")
    await send("sandbox")
    await send(None)  # unset -> global default (sandbox)
    with caplog.at_level("WARNING", logger="penny.channels.ios.apns"):
        await send("unexpected")  # unrecognized -> warn + global default (sandbox)

    hosts = [request["url"].split("/3/device/")[0] for request in requests]
    assert hosts == [
        "https://api.push.apple.com",
        "https://api.sandbox.push.apple.com",
        "https://api.sandbox.push.apple.com",
        "https://api.sandbox.push.apple.com",
    ]
    assert any(
        "Unrecognized APNs environment 'unexpected'" in record.getMessage()
        for record in caplog.records
    )


def test_apns_provider_token_uses_production_credentials(monkeypatch):
    encoded: list[dict] = []
    client = object.__new__(ApnsClient)
    client._config = ApnsConfig(
        team_id="SANDBOXTEAM",
        key_id="SANDBOXKEY",
        key_path="/unused/AuthKey_SANDBOXKEY.p8",
        bundle_id="com.example.Penny",
        sandbox=True,
        production_team_id="PRODTEAM",
        production_key_id="PRODKEY",
        production_key_path="/unused/AuthKey_PRODKEY.p8",
    )
    client._private_keys = {
        "/unused/AuthKey_SANDBOXKEY.p8": "sandbox-private-key",
        "/unused/AuthKey_PRODKEY.p8": "production-private-key",
    }
    client._tokens = {}

    def fake_encode(payload, private_key, *, algorithm, headers):
        encoded.append(
            {
                "payload": payload,
                "private_key": private_key,
                "algorithm": algorithm,
                "headers": headers,
            }
        )
        return f"token-{headers['kid']}"

    monkeypatch.setattr("penny.channels.ios.apns.jwt.encode", fake_encode)
    monkeypatch.setattr("penny.channels.ios.apns.time.time", lambda: 1234)

    token = client._provider_token(ApnsEnvironment.PRODUCTION)

    assert token == "token-PRODKEY"
    assert encoded == [
        {
            "payload": {"iss": "PRODTEAM", "iat": 1234},
            "private_key": "production-private-key",
            "algorithm": "ES256",
            "headers": {"alg": "ES256", "kid": "PRODKEY"},
        }
    ]


@pytest.mark.asyncio
async def test_send_raw_logs_and_disables_push_when_apns_rejects_token(tmp_path, caplog):
    db = _make_db(tmp_path)
    apns = RejectingApns()
    channel = _make_channel(db, apns=apns)
    ws = FakeWs()
    server_ws = cast(Any, ws)
    await channel._handle_register(
        server_ws,
        {
            "type": IOS_MSG_TYPE_REGISTER,
            "device_id": "ios-keychain-id",
            "label": "iPhone",
            "pairing_token": "pair-me",
            "apns_token": "bad-apns-token",
        },
    )
    channel._connections.clear()

    with caplog.at_level("WARNING", logger="penny.channels.ios.channel"):
        external_id = await channel._send_raw("ios-keychain-id", "hello from Penny")

    device = db.devices.get_by_identifier("ios-keychain-id")
    assert device is not None and device.id is not None
    registration = db.ios.get_registration(device.id)
    assert registration is not None
    assert registration.push_enabled is False
    assert external_id is not None
    assert any(
        "APNs rejected iOS device token; disabling push" in record.message
        and "BadDeviceToken" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_test_push_phrase_forces_apns_even_when_websocket_connected(tmp_path):
    db = _make_db(tmp_path)
    apns = FakeApns()
    channel = _make_channel(db, apns=apns)
    ws = FakeWs()
    server_ws = cast(Any, ws)
    await channel._handle_register(
        server_ws,
        {
            "type": IOS_MSG_TYPE_REGISTER,
            "device_id": "ios-keychain-id",
            "label": "iPhone",
            "pairing_token": "pair-me",
            "apns_token": "apns-token",
        },
    )

    await channel._handle_chat_message(
        {"type": "message", "content": "send me a test push"}, "ios-keychain-id"
    )

    assert len(apns.sent) == 1
    assert apns.sent[0]["device_token"] == "apns-token"
    assert apns.sent[0]["title"] == PUSH_GREETING_TITLE
    assert apns.sent[0]["body"] == TEST_PUSH_MESSAGE
    assert apns.sent[0]["badge"] == 1
    device = db.devices.get_by_identifier("ios-keychain-id")
    assert device is not None and device.id is not None
    outbox = db.ios.pending_for_device(device.id)
    assert len(outbox) == 1
    assert outbox[0].source_type == "test_push"
    assert outbox[0].source_name == "test_push"
    assert outbox[0].push_sent_at is not None


@pytest.mark.asyncio
async def test_apns_preview_payload_includes_greeting_summary_and_badge(monkeypatch):
    requests: list[dict] = []

    class FakeHttp:
        async def post(self, url, *, headers, json):
            requests.append({"url": url, "headers": headers, "json": json})
            return httpx.Response(200)

    client = object.__new__(ApnsClient)
    client._config = ApnsConfig(
        team_id="TEAMID",
        key_id="KEYID",
        key_path="/unused/AuthKey_KEYID.p8",
        bundle_id="com.example.Penny",
        sandbox=True,
    )
    client._http = FakeHttp()
    monkeypatch.setattr(
        client,
        "_provider_token",
        lambda environment=None: "provider-token",
    )

    await client.send_preview(
        device_token="device-token",
        title=PUSH_GREETING_TITLE,
        body="Found a fare drop.",
        badge=3,
        outbox_id=123,
        source_type="collector",
        source_name="flight-deals",
        thread_id="penny-flight-deals",
    )

    assert requests[0]["json"]["aps"]["alert"] == {
        "title": PUSH_GREETING_TITLE,
        "body": "Found a fare drop.",
    }
    assert requests[0]["json"]["aps"]["badge"] == 3
    assert requests[0]["json"]["aps"]["thread-id"] == "penny-flight-deals"
    assert requests[0]["json"]["outbox_id"] == 123
    assert requests[0]["json"]["source_type"] == "collector"
    assert requests[0]["json"]["source_name"] == "flight-deals"


@pytest.mark.asyncio
async def test_send_raw_notifies_connected_client_without_push(tmp_path):
    db = _make_db(tmp_path)
    apns = FakeApns()
    channel = _make_channel(db, apns=apns)
    ws = FakeWs()
    server_ws = cast(Any, ws)
    await channel._handle_register(
        server_ws,
        {
            "type": IOS_MSG_TYPE_REGISTER,
            "device_id": "ios-keychain-id",
            "label": "iPhone",
            "pairing_token": "pair-me",
            "apns_token": "apns-token",
        },
    )

    await channel._send_raw("ios-keychain-id", "hello from Penny", source_name="notifier")

    assert apns.sent == []
    assert ws.sent[-1]["type"] == IOS_RESP_TYPE_OUTBOX_CHANGED
    assert ws.sent[-1]["pending_count"] == 1


@pytest.mark.asyncio
async def test_pull_and_ack_messages(tmp_path):
    db = _make_db(tmp_path)
    channel = _make_channel(db)
    ws = FakeWs()
    server_ws = cast(Any, ws)
    await channel._handle_register(
        server_ws,
        {
            "type": IOS_MSG_TYPE_REGISTER,
            "device_id": "ios-keychain-id",
            "label": "iPhone",
            "pairing_token": "pair-me",
        },
    )
    await channel._send_raw("ios-keychain-id", "first", source_name="notifier")
    await channel._send_raw("ios-keychain-id", "second", source_name="notifier")

    device = db.devices.get_by_identifier("ios-keychain-id")
    assert device is not None and device.id is not None
    assert db.ios.pending_count(device.id) == 2

    await channel._handle_pull(server_ws, {"type": IOS_MSG_TYPE_PULL}, "ios-keychain-id")

    messages_payload = ws.sent[-1]
    assert messages_payload["type"] == IOS_RESP_TYPE_MESSAGES
    assert [m["content"] for m in messages_payload["messages"]] == ["first", "second"]
    ids = [m["id"] for m in messages_payload["messages"]]

    await channel._handle_ack(server_ws, {"type": IOS_MSG_TYPE_ACK, "ids": ids}, "ios-keychain-id")

    assert ws.sent[-1]["count"] == 2
    assert db.ios.pending_for_device(device.id) == []
    assert db.ios.pending_count(device.id) == 0
