"""Client for Penny's local mflux image-generation server (see ``image_server/``).

The server runs on the host, outside Docker (it needs Metal), serves exactly one
model at a time, and answers with a named ``error`` on every failure rather than
retrying anything itself — retry policy lives here, on the client, where it can
tell "the server said no" from "the server didn't answer":

- a non-2xx response that names a reason (an unsupported/misconfigured model, a
  bad request, a generation failure) is NOT retried — that is a firm answer, and
  retrying it would just burn three more denoising runs on the same "no";
- ``503`` (a generation is already in flight; the server does one at a time) IS
  retried, bounded, with the configured delay — the one condition worth a short wait;
- a connection error (the server is down or unreachable) is also retried, bounded,
  the same budget, since that may be transient (the server cold-starting, a port
  not yet listening).
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_BUSY_STATUS = 503


class ImageServerError(RuntimeError):
    """The image server answered with its own named reason for a failure."""


class _GenerateResponse(BaseModel):
    """The server's success response: base64-encoded PNG image data."""

    image: str


class _HealthResponse(BaseModel):
    """The server's ``/health`` response."""

    status: str
    served_model: str
    model_loaded: bool


class MfluxImageClient:
    """Client for the local mflux image-generation server."""

    def __init__(
        self,
        api_url: str,
        model: str,
        *,
        width: int,
        height: int,
        steps: int,
        max_retries: int,
        retry_delay: float,
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.width = width
        self.height = height
        self.steps = steps
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def generate_image(self, prompt: str) -> str:
        """Generate an image from a text prompt.

        Returns base64-encoded PNG image data.  Raises ``ImageServerError`` naming
        the server's own reason for a non-2xx response that wasn't a busy/transient
        condition worth retrying; raises the underlying connection error if every
        retry on a transient condition was exhausted.
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await self._post_generate(prompt)
            except Exception as error:
                last_error = error
                logger.warning(
                    "Image server connection error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    error,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                continue

            if response.status_code == _BUSY_STATUS:
                last_error = ImageServerError(self._reason(response))
                logger.warning(
                    "Image server busy (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    last_error,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                continue

            if response.status_code >= 400:
                raise ImageServerError(self._reason(response))

            return _GenerateResponse(**response.json()).image

        logger.error("Image generation failed after %d attempts: %s", self.max_retries, last_error)
        assert last_error is not None
        raise last_error

    async def _post_generate(self, prompt: str) -> httpx.Response:
        async with httpx.AsyncClient(timeout=120) as http_client:
            return await http_client.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "width": self.width,
                    "height": self.height,
                    "steps": self.steps,
                },
            )

    @staticmethod
    def _reason(response: httpx.Response) -> str:
        try:
            return str(response.json().get("error") or response.text)
        except Exception:
            return response.text or f"HTTP {response.status_code}"

    async def health(self) -> tuple[bool, str]:
        """Probe the server for the preflight. Returns ``(ok, detail)``; never raises."""
        try:
            async with httpx.AsyncClient(timeout=10) as http_client:
                response = await http_client.get(f"{self.api_url}/health")
                response.raise_for_status()
                health = _HealthResponse(**response.json())
        except Exception as error:
            return False, str(error)
        if health.served_model != self.model:
            return False, (
                f"server at {self.api_url} serves {health.served_model!r}, "
                f"not the configured {self.model!r}"
            )
        return True, f"model {self.model!r} ready (loaded={health.model_loaded})"
