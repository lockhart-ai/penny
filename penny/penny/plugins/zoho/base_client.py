"""Shared OAuth base for the Zoho Calendar and Projects API clients."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from penny.constants import PennyConstants
from penny.plugins.zoho.models import ZohoSession

logger = logging.getLogger(__name__)


class ZohoOAuthClient:
    """Shared OAuth 2.0 + HTTP lifecycle for Zoho API clients.

    Both ``ZohoCalendarClient`` and ``ZohoProjectsClient`` subclass this: it owns
    the refresh-token exchange, the cached access-token session, the auth headers,
    and the ``httpx.AsyncClient`` lifecycle. Subclasses declare their service
    label (which names them in the token-refresh log line) and may override
    ``_on_token_refreshed`` to capture service-specific fields from the token
    response (Zoho Projects records the API domain that way).
    """

    # Names the service in the token-refresh log line; overridden per subclass.
    service_label: str = "Zoho"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        *,
        timeout: float = PennyConstants.ZOHO_CLIENT_TIMEOUT,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._session: ZohoSession | None = None
        self._http = httpx.AsyncClient(timeout=timeout)

    async def _ensure_access_token(self) -> str:
        """Ensure we have a valid access token, refreshing if needed."""
        now = time.time()
        if self._session and self._session.expires_at > now + 60:
            return self._session.access_token

        resp = await self._http.post(
            PennyConstants.ZOHO_TOKEN_URL,
            data={
                "refresh_token": self._refresh_token,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"Zoho OAuth error: {data.get('error')}")

        expires_in = data.get("expires_in", 3600)
        self._session = ZohoSession(
            access_token=data["access_token"],
            expires_at=now + expires_in,
        )
        self._on_token_refreshed(data)
        logger.info("%s access token refreshed, expires in %ds", self.service_label, expires_in)
        return self._session.access_token

    def _on_token_refreshed(self, data: dict[str, Any]) -> None:
        """Capture service-specific fields from the token response (no-op by default)."""

    async def _get_headers(self) -> dict[str, str]:
        """Get headers with current access token."""
        token = await self._ensure_access_token()
        return {
            "Authorization": f"Zoho-oauthtoken {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
