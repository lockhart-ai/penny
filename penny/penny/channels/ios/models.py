"""Pydantic models for the iOS WebSocket protocol."""

from __future__ import annotations

import base64
import binascii

from pydantic import BaseModel, Field

IOS_MSG_TYPE_ACK = "ack_messages"
IOS_MSG_TYPE_EMBEDDING_REQUEST = "embedding_request"
IOS_MSG_TYPE_HEARTBEAT = "heartbeat"
IOS_MSG_TYPE_HISTORY = "history_request"
IOS_MSG_TYPE_MESSAGE = "message"
IOS_MSG_TYPE_PULL = "pull_messages"
IOS_MSG_TYPE_REGISTER = "register"
IOS_MSG_TYPE_TEST_NOTIFICATION = "test_notification"

IOS_RESP_TYPE_ACKED = "messages_acked"
IOS_RESP_TYPE_EMBEDDING = "embedding_response"
IOS_RESP_TYPE_MESSAGES = "messages"
IOS_RESP_TYPE_OUTBOX_CHANGED = "outbox_changed"
IOS_RESP_TYPE_REGISTERED = "registered"
IOS_RESP_TYPE_STATUS = "status"
IOS_RESP_TYPE_TYPING = "typing"
IOS_RESP_TYPE_AGENT_PROGRESS = "agent_progress"
IOS_RESP_TYPE_CONVERSATION_STATE = "conversation_state"

# Photo-attachment limits a ``message`` frame's ``images`` must satisfy (#3084).
# The app downscales before sending (JPEG, long edge ≤ 2048 px); these are the
# server-side backstop, enforced with an honest ``status`` error frame rather
# than a silent drop.
IOS_INCOMING_MAX_IMAGES = 4
IOS_INCOMING_IMAGE_MAX_BYTES = 6 * 1024 * 1024
IOS_INCOMING_IMAGE_MIME_TYPES = ("image/jpeg", "image/png")

# The error codes a violating message frame is answered with.
IOS_IMAGE_ERROR_TOO_MANY = "too_many_images"
IOS_IMAGE_ERROR_TYPE = "unsupported_image_type"
IOS_IMAGE_ERROR_SIZE = "image_too_large"
IOS_IMAGE_ERROR_UNDECODABLE = "undecodable_image"


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
    # Photo attachments as data URIs (#3084) — may carry the message alone, with
    # empty ``content``. Validated against the channel limits (above) before the
    # message is forwarded anywhere.
    images: list[str] = Field(default_factory=list)


def validate_incoming_images(images: list[str]) -> str | None:
    """The error code for the first photo-attachment rule a message's ``images``
    violate (#3084), or ``None`` when every image is acceptable.

    Checked in ``_handle_chat_message`` — after the frame parses — so the
    connection gets a named ``status`` error instead of a silent drop. Each
    image must be a ``data:image/<allowed>;base64,`` URI whose decoded payload
    fits ``IOS_INCOMING_IMAGE_MAX_BYTES``.
    """
    if len(images) > IOS_INCOMING_MAX_IMAGES:
        return IOS_IMAGE_ERROR_TOO_MANY
    for uri in images:
        header, sep, encoded = uri.partition(";base64,")
        mime = header[len("data:") :] if header.startswith("data:") else None
        if not sep or mime not in IOS_INCOMING_IMAGE_MIME_TYPES:
            return IOS_IMAGE_ERROR_TYPE
        try:
            data = base64.b64decode(encoded, validate=True)
        except binascii.Error, ValueError:
            return IOS_IMAGE_ERROR_UNDECODABLE
        if len(data) > IOS_INCOMING_IMAGE_MAX_BYTES:
            return IOS_IMAGE_ERROR_SIZE
    return None


class IosTestNotification(BaseModel):
    """Request a diagnostic APNs notification for the registered device."""

    type: str = IOS_MSG_TYPE_TEST_NOTIFICATION


class IosPullMessages(BaseModel):
    """Request unacknowledged outbox rows."""

    type: str = IOS_MSG_TYPE_PULL
    limit: int = Field(default=50, ge=1, le=200)


class IosHistoryRequest(BaseModel):
    """Request one older page from the shared message log."""

    type: str = IOS_MSG_TYPE_HISTORY
    limit: int = Field(default=50, ge=1, le=200)
    before: str | None = None
    channel_types: list[str] | None = None
    include_attachments: bool = True
    count_only: bool = False


class IosAckMessages(BaseModel):
    """Acknowledge outbox rows after the app has persisted/displayed them."""

    type: str = IOS_MSG_TYPE_ACK
    ids: list[int]


class IosEmbeddingRequest(BaseModel):
    """Request an embedding for client-side semantic search."""

    type: str = IOS_MSG_TYPE_EMBEDDING_REQUEST
    request_id: str
    text: str


class IosEmbeddingResponse(BaseModel):
    """Embedding response correlated to an iOS client request."""

    type: str = IOS_RESP_TYPE_EMBEDDING
    request_id: str
    embedding: str | None = None
    error: str | None = None


class IosOutboxRecord(BaseModel):
    """One outbox row returned to the app."""

    id: int
    message_id: int | None = None
    run_id: str | None = None
    outbox_id: int | None = None
    created_at: str
    content: str
    attachments: list[str] = Field(default_factory=list)
    source_type: str | None = None
    source_name: str | None = None
    source_hint: str | None = None
    push_title: str
    push_summary: str
    direction: str | None = None
    channel_type: str | None = None
    device_label: str | None = None
    device_identifier: str | None = None
    parent_id: int | None = None
    embedding: str | None = None
    # The producing run's tool-call count and wall time, run start → this reply
    # (#3041). None — and absent from the wire (`exclude_none`) — when the
    # message has no run; older clients ignore them all the same.
    tool_calls: int | None = None
    duration_ms: int | None = None


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
    """Outbox or history response."""

    type: str = IOS_RESP_TYPE_MESSAGES
    messages: list[IosOutboxRecord]
    mode: str = "outbox"
    next_cursor: str | None = None
    has_more: bool = False
    total_count: int | None = None
    attachments_included: bool = True


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


class IosConversationState(BaseModel):
    """The conversation machine's settled state."""

    type: str = IOS_RESP_TYPE_CONVERSATION_STATE
    label: str


class IosAgentProgressTool(BaseModel):
    """A redacted tool descriptor for client-side progress formatting."""

    name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class IosAgentProgress(BaseModel):
    """Ephemeral agent progress sent only to connected iOS clients."""

    type: str = IOS_RESP_TYPE_AGENT_PROGRESS
    event: str
    run_id: str
    agent: str
    scope: str
    step: int | None = None
    max_steps: int | None = None
    tools: list[IosAgentProgressTool] = Field(default_factory=list)
    outcome: str | None = None
    # The step's reasoning, trimmed to IOS_PROGRESS_THINKING_MAX_CHARS (#3040).
    # Foreground (chat) runs only — background collectors don't stream thinking.
    thinking: str | None = None


class IosDataReadRequest(BaseModel):
    """Optional correlation and common filters for the read-only data browser."""

    type: str
    request_id: str
    query: str | None = None
    agent_name: str | None = None
    flagged_only: bool = False
    offset: int = Field(default=0, ge=0)


class IosDataReadError(BaseModel):
    """A failed read, correlated without exposing internal exception details."""

    type: str = "data_read_error"
    request_id: str
    error: str
