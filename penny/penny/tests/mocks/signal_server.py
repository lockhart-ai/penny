"""Mock Signal server for integration testing."""

import asyncio
import json
import time
import uuid

from aiohttp import WSMsgType, web


class MockSignalServer:
    """Mock Signal API server with WebSocket and REST endpoints."""

    def __init__(self) -> None:
        self.outgoing_messages: list[dict] = []
        self.typing_events: list[tuple[str, str]] = []  # (action, recipient)
        # Append-only log of every reaction op the channel sent. Each entry is
        # ``{"op": "send"|"remove", "reaction": ..., "target_author": ...,
        # "timestamp": ...}`` so tests can assert the full progress sequence.
        self.reaction_events: list[dict] = []
        self._websockets: list[web.WebSocketResponse] = []
        self._attachments: dict[str, bytes] = {}  # attachment_id -> binary data
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self.port: int | None = None
        # Queue of (status, body) tuples to return for upcoming /v2/send requests.
        # When non-empty, the next send request pops and returns the queued response
        # instead of the default 200 success. Useful for simulating transient errors.
        self._send_response_queue: list[tuple[int, dict]] = []

    def queue_send_error(self, status: int, body: dict) -> None:
        """Queue a specific HTTP response for the next /v2/send request."""
        self._send_response_queue.append((status, body))

    async def start(self, port: int = 0) -> None:
        """Start the mock server on specified port (0 = random available port)."""
        self._app = web.Application()
        self._app.router.add_post("/v2/send", self._handle_send)
        self._app.router.add_put("/v1/typing-indicator/{phone_number}", self._handle_typing_start)
        self._app.router.add_delete("/v1/typing-indicator/{phone_number}", self._handle_typing_stop)
        self._app.router.add_post("/v1/reactions/{phone_number}", self._handle_reaction_send)
        self._app.router.add_delete("/v1/reactions/{phone_number}", self._handle_reaction_remove)
        self._app.router.add_get(
            "/v1/attachments/{attachment_id}", self._handle_attachment_download
        )
        self._app.router.add_get("/v1/receive/{phone_number}", self._handle_websocket)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "localhost", port)
        await self._site.start()

        # Get the actual port (important when port=0)
        server = self._site._server
        assert isinstance(server, asyncio.Server)
        sock = server.sockets[0]
        self.port = sock.getsockname()[1]

    async def stop(self) -> None:
        """Stop the mock server and close all connections."""
        for ws in self._websockets:
            await ws.close()
        self._websockets.clear()

        if self._runner:
            await self._runner.cleanup()

    async def push_message(
        self,
        sender: str,
        content: str,
        quote: dict | None = None,
    ) -> None:
        """Push an incoming message to all connected WebSocket clients."""
        ts = int(time.time() * 1000)
        envelope = {
            "envelope": {
                "source": sender,
                "sourceNumber": sender,
                "sourceUuid": str(uuid.uuid4()),
                "sourceName": "Test User",
                "sourceDevice": 1,
                "timestamp": ts,
                "serverReceivedTimestamp": ts,
                "serverDeliveredTimestamp": ts,
                "dataMessage": {
                    "timestamp": ts,
                    "message": content,
                    "expiresInSeconds": 0,
                    "isExpirationUpdate": False,
                    "viewOnce": False,
                    "quote": quote,
                },
            },
            "account": "+15551234567",
        }

        for ws in self._websockets:
            if not ws.closed:
                await ws.send_json(envelope)

    async def push_image_message(
        self,
        sender: str,
        image_data: bytes,
        content_type: str = "image/jpeg",
        text: str | None = None,
    ) -> None:
        """Push a message with an image attachment to all connected WebSocket clients."""
        ts = int(time.time() * 1000)
        attachment_id = str(uuid.uuid4())

        # Store the binary data for download
        self._attachments[attachment_id] = image_data

        envelope = {
            "envelope": {
                "source": sender,
                "sourceNumber": sender,
                "sourceUuid": str(uuid.uuid4()),
                "sourceName": "Test User",
                "sourceDevice": 1,
                "timestamp": ts,
                "serverReceivedTimestamp": ts,
                "serverDeliveredTimestamp": ts,
                "dataMessage": {
                    "timestamp": ts,
                    "message": text,
                    "expiresInSeconds": 0,
                    "isExpirationUpdate": False,
                    "viewOnce": False,
                    "attachments": [
                        {
                            "contentType": content_type,
                            "id": attachment_id,
                            "size": len(image_data),
                        }
                    ],
                },
            },
            "account": "+15551234567",
        }

        for ws in self._websockets:
            if not ws.closed:
                await ws.send_json(envelope)

    async def push_reaction(
        self,
        sender: str,
        emoji: str,
        target_timestamp: int,
        target_author: str | None = None,
    ) -> None:
        """Push a reaction message to all connected WebSocket clients."""
        ts = int(time.time() * 1000)
        envelope = {
            "envelope": {
                "source": sender,
                "sourceNumber": sender,
                "sourceUuid": str(uuid.uuid4()),
                "sourceName": "Test User",
                "sourceDevice": 1,
                "timestamp": ts,
                "serverReceivedTimestamp": ts,
                "serverDeliveredTimestamp": ts,
                "dataMessage": {
                    "timestamp": ts,
                    "message": "",
                    "reaction": {
                        "emoji": {"value": emoji},
                        "targetAuthor": target_author or "+15551234567",
                        "targetAuthorNumber": target_author or "+15551234567",
                        "targetSentTimestamp": target_timestamp,
                        "isRemove": False,
                    },
                },
            },
            "account": "+15551234567",
        }

        for ws in self._websockets:
            if not ws.closed:
                await ws.send_json(envelope)

    async def wait_for_message(self, timeout: float = 10.0) -> dict:
        """Wait for an outgoing message to be captured."""
        start = time.time()
        initial_count = len(self.outgoing_messages)
        while time.time() - start < timeout:
            if len(self.outgoing_messages) > initial_count:
                return self.outgoing_messages[-1]
            await asyncio.sleep(0.05)
        msg = f"Timeout waiting for outgoing message after {timeout}s"
        raise TimeoutError(msg)

    async def wait_for_message_containing(self, substring: str, timeout: float = 10.0) -> dict:
        """Wait for an outgoing message whose body contains ``substring``.

        Useful when the channel sends a sequence of edits to the same
        message and the test cares about the final form, not the initial
        "thinking..." placeholder.
        """
        start = time.time()
        while time.time() - start < timeout:
            for msg in reversed(self.outgoing_messages):
                if substring in (msg.get("message") or ""):
                    return msg
            await asyncio.sleep(0.05)
        msg = f"Timeout waiting for outgoing message containing {substring!r} after {timeout}s"
        raise TimeoutError(msg)

    async def _handle_send(self, request: web.Request) -> web.Response:
        """Handle POST /v2/send - capture outgoing messages."""
        data = await request.json()
        # Return a queued error response if one is staged
        if self._send_response_queue:
            status, body = self._send_response_queue.pop(0)
            return web.json_response(body, status=status)
        self.outgoing_messages.append(data)
        return web.json_response({"timestamp": int(time.time() * 1000)})

    async def _handle_reaction_send(self, request: web.Request) -> web.Response:
        """Handle POST /v1/reactions/{phone_number}."""
        data = await request.json()
        self.reaction_events.append({"op": "send", **data})
        return web.Response(status=204)

    async def _handle_reaction_remove(self, request: web.Request) -> web.Response:
        """Handle DELETE /v1/reactions/{phone_number}."""
        try:
            data = await request.json()
        except json.JSONDecodeError:
            data = {}
        self.reaction_events.append({"op": "remove", **data})
        return web.Response(status=204)

    async def _handle_attachment_download(self, request: web.Request) -> web.Response:
        """Handle GET /v1/attachments/{attachment_id} - serve attachment data."""
        attachment_id = request.match_info["attachment_id"]
        data = self._attachments.get(attachment_id)
        if data is None:
            return web.Response(status=404)
        return web.Response(body=data, content_type="image/jpeg")

    async def _handle_typing_start(self, request: web.Request) -> web.Response:
        """Handle PUT /v1/typing-indicator/{phone_number}."""
        try:
            data = await request.json()
            recipient = data.get("recipient", "")
        except json.JSONDecodeError:
            recipient = ""
        self.typing_events.append(("start", recipient))
        return web.Response()

    async def _handle_typing_stop(self, request: web.Request) -> web.Response:
        """Handle DELETE /v1/typing-indicator/{phone_number}."""
        try:
            data = await request.json()
            recipient = data.get("recipient", "")
        except json.JSONDecodeError:
            recipient = ""
        self.typing_events.append(("stop", recipient))
        return web.Response()

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle GET /v1/receive/{phone_number} WebSocket connection."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._websockets.append(ws)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._websockets.remove(ws)

        return ws
