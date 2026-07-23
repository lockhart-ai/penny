"""Fastmail JMAP API client."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from penny.constants import PennyConstants
from penny.email.models import EmailAddress, EmailDetail, EmailSummary
from penny.html_utils import strip_html
from penny.plugins.fastmail.models import JmapSession

logger = logging.getLogger(__name__)

JMAP_CAPABILITIES = [
    "urn:ietf:params:jmap:core",
    "urn:ietf:params:jmap:mail",
]


class JmapClient:
    """Fastmail JMAP API client."""

    def __init__(
        self,
        api_token: str,
        *,
        timeout: float,
        max_body_length: int,
        search_limit: int,
    ) -> None:
        self._api_token = api_token
        self._max_body_length = max_body_length
        self._search_limit = search_limit
        self._session: JmapSession | None = None
        self._http = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_token}"},
        )

    async def _ensure_session(self) -> JmapSession:
        """Fetch and cache JMAP session (apiUrl, accountId)."""
        if self._session:
            return self._session

        resp = await self._http.get(PennyConstants.JMAP_SESSION_URL)
        resp.raise_for_status()
        data = resp.json()

        self._session = JmapSession(
            api_url=data["apiUrl"],
            account_id=data["primaryAccounts"]["urn:ietf:params:jmap:mail"],
        )
        logger.info("JMAP session established: account_id=%s", self._session.account_id)
        return self._session

    async def _call(self, method_calls: list[list[Any]]) -> list[list[Any]]:
        """Make a JMAP method call."""
        session = await self._ensure_session()
        body = {
            "using": JMAP_CAPABILITIES,
            "methodCalls": method_calls,
        }
        resp = await self._http.post(session.api_url, json=body)
        resp.raise_for_status()
        return resp.json()["methodResponses"]

    async def search_emails(
        self,
        text: str | None = None,
        from_addr: str | None = None,
        subject: str | None = None,
        after: str | None = None,
        before: str | None = None,
    ) -> list[EmailSummary]:
        """Search emails and return summaries."""
        filter_obj: dict[str, Any] = {}
        if text:
            filter_obj["text"] = text
        if from_addr:
            filter_obj["from"] = from_addr
        if subject:
            filter_obj["subject"] = subject
        if after:
            filter_obj["after"] = after
        if before:
            filter_obj["before"] = before

        session = await self._ensure_session()

        responses = await self._call(
            [
                [
                    "Email/query",
                    {
                        "accountId": session.account_id,
                        "filter": filter_obj,
                        "sort": [{"property": "receivedAt", "isAscending": False}],
                        "limit": self._search_limit,
                    },
                    "0",
                ],
                [
                    "Email/get",
                    {
                        "accountId": session.account_id,
                        "#ids": {
                            "resultOf": "0",
                            "name": "Email/query",
                            "path": "/ids",
                        },
                        "properties": ["id", "subject", "from", "receivedAt", "preview"],
                    },
                    "1",
                ],
            ]
        )

        emails_data = responses[1][1].get("list", [])
        logger.info("JMAP search returned %d email(s)", len(emails_data))

        return [
            EmailSummary(
                id=e["id"],
                subject=e.get("subject", "(no subject)"),
                from_addresses=[EmailAddress(**a) for a in (e.get("from") or [])],
                received_at=e.get("receivedAt", ""),
                preview=e.get("preview", ""),
            )
            for e in emails_data
        ]

    async def read_emails(self, email_ids: list[str]) -> list[EmailDetail]:
        """Fetch full email bodies by IDs."""
        if not email_ids:
            return []

        session = await self._ensure_session()

        responses = await self._call(
            [
                [
                    "Email/get",
                    {
                        "accountId": session.account_id,
                        "ids": email_ids,
                        "properties": [
                            "id",
                            "subject",
                            "from",
                            "to",
                            "receivedAt",
                            "textBody",
                            "htmlBody",
                            "bodyValues",
                        ],
                        "fetchTextBodyValues": True,
                        "fetchHTMLBodyValues": True,
                    },
                    "0",
                ],
            ]
        )

        emails_data = responses[0][1].get("list", [])
        results: list[EmailDetail] = []

        for e in emails_data:
            body_values = e.get("bodyValues", {})

            # Try plain text body first
            text_body = ""
            text_parts = e.get("textBody", [])
            for part in text_parts:
                part_id = part.get("partId")
                if part_id and part_id in body_values:
                    text_body += body_values[part_id].get("value", "")

            # Fall back to HTML body with tag stripping
            if not text_body:
                html_parts = e.get("htmlBody", [])
                for part in html_parts:
                    part_id = part.get("partId")
                    if part_id and part_id in body_values:
                        html_content = body_values[part_id].get("value", "")
                        text_body += strip_html(html_content)

            # Truncate long bodies
            if len(text_body) > self._max_body_length:
                text_body = text_body[: self._max_body_length] + "\n\n[truncated]"

            results.append(
                EmailDetail(
                    id=e["id"],
                    subject=e.get("subject", "(no subject)"),
                    from_addresses=[EmailAddress(**a) for a in (e.get("from") or [])],
                    to_addresses=[EmailAddress(**a) for a in (e.get("to") or [])],
                    received_at=e.get("receivedAt", ""),
                    text_body=text_body,
                )
            )

        return results

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
