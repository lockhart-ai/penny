"""Pydantic models for the iOS WebSocket protocol."""

from __future__ import annotations

from pydantic import BaseModel, Field

IOS_MSG_TYPE_ACK = "ack_messages"
IOS_MSG_TYPE_HEARTBEAT = "heartbeat"
IOS_MSG_TYPE_MESSAGE = "message"
IOS_MSG_TYPE_PULL = "pull_messages"
IOS_MSG_TYPE_REGISTER = "register"

IOS_RESP_TYPE_ACKED = "messages_acked"
IOS_RESP_TYPE_MESSAGES = "messages"
IOS_RESP_TYPE_OUTBOX_CHANGED = "outbox_changed"
IOS_RESP_TYPE_REGISTERED = "registered"
IOS_RESP_TYPE_STATUS = "status"
IOS_RESP_TYPE_TYPING = "typing"


class IosRegister(BaseModel):
    """Register or refresh an iOS device connection."""

    type: str = IOS_MSG_TYPE_REGISTER
    device_id: str
    label: str
    pairing_token: str | None = None
    device_secret: str | None = None
    apns_token: str | None = None
    apns_environment: str = "sandbox"
    app_version: str | None = None


class IosIncomingMessage(BaseModel):
    """A foreground chat message from the iOS app."""

    type: str = IOS_MSG_TYPE_MESSAGE
    content: str


class IosPullMessages(BaseModel):
    """Request unacknowledged outbox rows."""

    type: str = IOS_MSG_TYPE_PULL
    limit: int = Field(default=50, ge=1, le=200)


class IosAckMessages(BaseModel):
    """Acknowledge outbox rows after the app has persisted/displayed them."""

    type: str = IOS_MSG_TYPE_ACK
    ids: list[int]


class IosOutboxRecord(BaseModel):
    """One outbox row returned to the app."""

    id: int
    created_at: str
    content: str
    attachments: list[str] = []
    source_type: str | None = None
    source_name: str | None = None
    source_hint: str | None = None
    push_title: str
    push_summary: str


class IosStatus(BaseModel):
    """Connection status or protocol error."""

    type: str = IOS_RESP_TYPE_STATUS
    connected: bool = True
    error: str | None = None


class IosRegistered(BaseModel):
    """Successful registration response."""

    type: str = IOS_RESP_TYPE_REGISTERED
    device_id: str
    is_default: bool
    pending_count: int


class IosMessages(BaseModel):
    """Outbox response."""

    type: str = IOS_RESP_TYPE_MESSAGES
    messages: list[IosOutboxRecord]


class IosMessagesAcked(BaseModel):
    """Ack response."""

    type: str = IOS_RESP_TYPE_ACKED
    count: int


class IosOutboxChanged(BaseModel):
    """Server hint that the app should pull messages."""

    type: str = IOS_RESP_TYPE_OUTBOX_CHANGED
    pending_count: int


class IosTyping(BaseModel):
    """Typing indicator."""

    type: str = IOS_RESP_TYPE_TYPING
    active: bool
