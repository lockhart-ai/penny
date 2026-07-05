"""APNs client for iOS preview notifications."""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
import jwt

logger = logging.getLogger(__name__)

APNS_SANDBOX_HOST = "api.sandbox.push.apple.com"
APNS_PRODUCTION_HOST = "api.push.apple.com"


class ApnsEnvironment(StrEnum):
    """APNs delivery environment a device's push token was minted for.

    A build's APNs token is only valid against one host: development/ad-hoc
    builds get a sandbox token, while TestFlight and App Store builds get a
    production one.  The client reports which at registration, so the send host
    is chosen per device rather than from the global default alone.
    """

    SANDBOX = "sandbox"
    PRODUCTION = "production"

    @property
    def host(self) -> str:
        return APNS_PRODUCTION_HOST if self is ApnsEnvironment.PRODUCTION else APNS_SANDBOX_HOST

    @classmethod
    def from_value(cls, value: str | None) -> ApnsEnvironment | None:
        """Parse a stored environment string; None if unset or unrecognized."""
        if value is None:
            return None
        try:
            return cls(value)
        except ValueError:
            return None


class ApnsError(Exception):
    """APNs request failed."""

    def __init__(self, status_code: int, reason: str) -> None:
        super().__init__(f"APNs error {status_code}: {reason}")
        self.status_code = status_code
        self.reason = reason

    @property
    def invalid_token(self) -> bool:
        """Whether APNs says this token should no longer be used."""
        return self.reason in {"BadDeviceToken", "Unregistered", "DeviceTokenNotForTopic"}


@dataclass
class ApnsConfig:
    """APNs token-auth configuration."""

    team_id: str
    key_id: str
    key_path: str
    bundle_id: str
    sandbox: bool = True

    @property
    def host(self) -> str:
        default = ApnsEnvironment.SANDBOX if self.sandbox else ApnsEnvironment.PRODUCTION
        return default.host


class ApnsClient:
    """Minimal APNs HTTP/2 client using provider-token authentication."""

    def __init__(self, config: ApnsConfig) -> None:
        self._config = config
        self._private_key = Path(config.key_path).read_text()
        self._token: str | None = None
        self._token_iat = 0
        self._http = httpx.AsyncClient(http2=True, timeout=10.0)

    async def send_preview(
        self,
        *,
        device_token: str,
        title: str,
        body: str,
        badge: int,
        outbox_id: int,
        source_type: str | None,
        source_name: str | None,
        thread_id: str | None = None,
        environment: str | None = None,
    ) -> None:
        """Send a visible preview notification for one outbox row.

        The host is chosen from the device's registered ``environment`` (sandbox
        vs. production), falling back to the global default when the device did
        not report a recognized value.
        """
        payload: dict[str, Any] = {
            "aps": {
                "alert": {"title": title, "body": body},
                "badge": badge,
                "sound": "default",
            },
            "outbox_id": outbox_id,
        }
        if thread_id:
            payload["aps"]["thread-id"] = thread_id
        if source_type:
            payload["source_type"] = source_type
        if source_name:
            payload["source_name"] = source_name

        response = await self._http.post(
            f"https://{self._host_for(environment)}/3/device/{device_token}",
            headers={
                "authorization": f"bearer {self._provider_token()}",
                "apns-topic": self._config.bundle_id,
                "apns-push-type": "alert",
                "apns-priority": "10",
            },
            json=payload,
        )
        if response.status_code < 300:
            return
        reason = "unknown"
        with suppress(Exception):
            reason = response.json().get("reason") or reason
        raise ApnsError(response.status_code, reason)

    def _host_for(self, environment: str | None) -> str:
        """Resolve the APNs host for a device, falling back to the global default."""
        resolved = ApnsEnvironment.from_value(environment)
        if resolved is not None:
            return resolved.host
        if environment is not None:
            logger.warning(
                "Unrecognized APNs environment %r; using global default host %s",
                environment,
                self._config.host,
            )
        return self._config.host

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()

    def _provider_token(self) -> str:
        """Return a cached provider token; APNs allows reuse for up to one hour."""
        now = int(time.time())
        if self._token and now - self._token_iat < 50 * 60:
            return self._token
        self._token_iat = now
        self._token = jwt.encode(
            {"iss": self._config.team_id, "iat": now},
            self._private_key,
            algorithm="ES256",
            headers={"alg": "ES256", "kid": self._config.key_id},
        )
        return self._token
