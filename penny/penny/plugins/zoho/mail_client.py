"""Zoho Mail API client."""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from penny.constants import PennyConstants
from penny.email.models import EmailAddress, EmailDetail, EmailSummary
from penny.html_utils import strip_html
from penny.plugins.zoho.mail_models import ZohoAccount, ZohoFolder, ZohoSession

logger = logging.getLogger(__name__)

# Regex for Zoho date format DD-MMM-YYYY (e.g., 12-Sep-2017)
_ZOHO_DATE_RE = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{4}$")

# Month abbreviations for ISO 8601 → Zoho date conversion
_MONTH_ABBREVS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


class ZohoClient:
    """Zoho Mail API client.

    Uses OAuth 2.0 with client credentials to access Zoho Mail API.
    Requires a refresh token to be obtained via the OAuth flow.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        *,
        timeout: float,
        max_body_length: int,
        search_limit: int,
        list_limit: int,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._max_body_length = max_body_length
        self._search_limit = search_limit
        self._list_limit = list_limit
        self._session: ZohoSession | None = None
        self._account: ZohoAccount | None = None
        self._folders: list[ZohoFolder] | None = None
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
        logger.info("Zoho access token refreshed, expires in %ds", expires_in)
        return self._session.access_token

    async def _get_headers(self) -> dict[str, str]:
        """Get headers with current access token."""
        token = await self._ensure_access_token()
        return {
            "Authorization": f"Zoho-oauthtoken {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def _ensure_account(self) -> ZohoAccount:
        """Fetch and cache the primary Zoho Mail account."""
        if self._account:
            return self._account

        headers = await self._get_headers()
        resp = await self._http.get(PennyConstants.ZOHO_ACCOUNTS_URL, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        accounts = data.get("data", [])
        if not accounts:
            raise RuntimeError("No Zoho Mail accounts found")

        # Use the first (primary) account
        acct = accounts[0]

        # emailAddress can be a list of dicts or a single dict
        email_addr_field = acct.get("emailAddress", [])
        if isinstance(email_addr_field, list) and email_addr_field:
            email_address = email_addr_field[0].get("mailId", "")
        elif isinstance(email_addr_field, dict):
            email_address = email_addr_field.get("mailId", "")
        else:
            email_address = ""

        self._account = ZohoAccount(
            account_id=str(acct["accountId"]),
            email_address=email_address,
            display_name=acct.get("displayName"),
        )
        logger.info(
            "Zoho account: %s (%s)",
            self._account.email_address,
            self._account.account_id,
        )
        return self._account

    async def get_folders(self) -> list[ZohoFolder]:
        """Fetch and cache all folders for the account."""
        if self._folders is not None:
            return self._folders

        account = await self._ensure_account()
        headers = await self._get_headers()

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/folders"
        resp = await self._http.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        folders_data = data.get("data", [])
        self._folders = [
            ZohoFolder(
                folder_id=str(f["folderId"]),
                folder_name=f.get("folderName", ""),
                folder_type=f.get("folderType", ""),
                path=f.get("path", ""),
                is_archived=bool(f.get("isArchived", 0)),
            )
            for f in folders_data
        ]
        logger.info("Loaded %d Zoho folders", len(self._folders))
        return self._folders

    async def get_folder_by_name(self, name: str) -> ZohoFolder | None:
        """Get a folder by name (case-insensitive)."""
        folders = await self.get_folders()
        name_lower = name.lower()
        for folder in folders:
            if folder.folder_name.lower() == name_lower:
                return folder
        return None

    async def get_folder_by_type(self, folder_type: str) -> ZohoFolder | None:
        """Get a folder by type (e.g., 'Inbox', 'Sent', 'Drafts')."""
        folders = await self.get_folders()
        for folder in folders:
            if folder.folder_type == folder_type:
                return folder
        return None

    async def list_emails(
        self,
        folder_name: str | None = None,
        limit: int | None = None,
    ) -> list[EmailSummary]:
        """List emails from a specific folder."""
        account = await self._ensure_account()
        headers = await self._get_headers()

        # Get the folder ID
        if folder_name:
            folder = await self.get_folder_by_name(folder_name)
            if not folder:
                logger.warning("Folder not found: %s", folder_name)
                return []
            folder_id = folder.folder_id
        else:
            # Default to Inbox
            folder = await self.get_folder_by_type("Inbox")
            if not folder:
                logger.warning("Inbox folder not found")
                return []
            folder_id = folder.folder_id

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/messages/view"
        params = {
            "folderId": folder_id,
            "limit": limit if limit is not None else self._list_limit,
            "includeto": "true",
        }

        resp = await self._http.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        emails_data = data.get("data", [])
        logger.info("Listed %d email(s) from folder %s", len(emails_data), folder_name or "Inbox")

        return [self._parse_email_summary(e) for e in emails_data]

    async def search_emails(
        self,
        text: str | None = None,
        from_addr: str | None = None,
        subject: str | None = None,
        after: str | None = None,
        before: str | None = None,
    ) -> list[EmailSummary]:
        """Search emails and return summaries.

        Uses Zoho's search syntax to build the searchKey parameter.
        Zoho syntax: parameter:value with :: between multiple conditions.
        Docs: https://www.zoho.com/mail/help/search-syntax.html
        """
        search_parts = []
        if text:
            escaped = text.replace('"', "")
            if " " in escaped:
                search_parts.append(f'entire:"{escaped}"')
            else:
                search_parts.append(f"entire:{escaped}")
        if from_addr:
            search_parts.append(f"sender:{from_addr.replace('"', '')}")
        if subject:
            escaped = subject.replace('"', "")
            if " " in escaped or ":" in escaped:
                search_parts.append(f'subject:"{escaped}"')
            else:
                search_parts.append(f"subject:{escaped}")
        # Zoho date format is DD-MMM-YYYY — convert from ISO 8601 if needed
        if after:
            zoho_date = self._to_zoho_date(after)
            if zoho_date:
                search_parts.append(f"fromDate:{zoho_date}")
        if before:
            zoho_date = self._to_zoho_date(before)
            if zoho_date:
                search_parts.append(f"toDate:{zoho_date}")

        # Join with :: for AND logic between conditions
        search_key = "::".join(search_parts) if search_parts else "newMails"
        logger.info("Zoho search query: %s", search_key)

        account = await self._ensure_account()
        headers = await self._get_headers()

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/messages/search"
        params = {
            "searchKey": search_key,
            "limit": self._search_limit,
            "includeto": "true",
        }

        resp = await self._http.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        emails_data = data.get("data", [])
        logger.info("Zoho search returned %d email(s)", len(emails_data))

        return [self._parse_email_summary(e) for e in emails_data]

    def _parse_email_summary(self, e: dict[str, Any]) -> EmailSummary:
        """Parse a Zoho email response into an EmailSummary."""
        from_addr = e.get("fromAddress", "")
        from_name = e.get("sender", "")

        return EmailSummary(
            id=self._make_email_id(e),
            subject=e.get("subject", "(no subject)"),
            from_addresses=[EmailAddress(name=from_name or None, email=from_addr)],
            received_at=self._format_timestamp(e.get("receivedTime", 0)),
            preview=e.get("summary", ""),
        )

    def _make_email_id(self, e: dict[str, Any]) -> str:
        """Create a folderId:messageId composite ID."""
        folder_id = e.get("folderId", "")
        message_id = e.get("messageId", "")
        return f"{folder_id}:{message_id}"

    def _format_timestamp(self, ts: int | str) -> str:
        """Format a Unix timestamp (ms) to ISO 8601."""
        if not ts:
            return ""
        try:
            ts_int = int(ts)
            dt = datetime.fromtimestamp(ts_int / 1000, tz=UTC)
            return dt.isoformat()
        except ValueError, TypeError:
            return str(ts)

    @staticmethod
    def _is_valid_zoho_date(date_str: str) -> bool:
        """Check if date string is in Zoho format DD-MMM-YYYY (e.g., 12-Sep-2017)."""
        return bool(_ZOHO_DATE_RE.match(date_str))

    @staticmethod
    def _convert_to_zoho_date(date_str: str) -> str | None:
        """Convert ISO 8601 date string to Zoho DD-MMM-YYYY format.

        Accepts formats like 2026-01-15, 2026-01-15T00:00:00Z, etc.
        Returns None if parsing fails.
        """
        try:
            # Strip time/timezone suffix for simple date parsing
            date_part = date_str.split("T")[0]
            parts = date_part.split("-")
            if len(parts) != 3:
                return None
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            if not (1 <= month <= 12):
                return None
            return f"{day}-{_MONTH_ABBREVS[month - 1]}-{year}"
        except ValueError, IndexError:
            return None

    def _to_zoho_date(self, date_str: str) -> str | None:
        """Accept Zoho DD-MMM-YYYY or ISO 8601 date, return Zoho format or None."""
        if self._is_valid_zoho_date(date_str):
            return date_str
        return self._convert_to_zoho_date(date_str)

    async def read_emails(self, email_ids: list[str]) -> list[EmailDetail]:
        """Fetch full email bodies by IDs."""
        if not email_ids:
            return []

        headers = await self._get_headers()
        results: list[EmailDetail] = []

        for email_id in email_ids:
            try:
                detail = await self._fetch_email_detail(email_id, headers)
                if detail:
                    results.append(detail)
            except Exception as e:
                logger.warning("Failed to fetch email %s: %s", email_id, e)

        return results

    async def _fetch_email_detail(
        self,
        email_id: str,
        headers: dict[str, str],
    ) -> EmailDetail | None:
        """Fetch a single email's full content using the content endpoint."""
        # Extract folder_id and message_id from the composite ID
        if ":" not in email_id:
            logger.warning("Invalid email ID format (no colon): %s", email_id)
            return None

        folder_id, message_id = email_id.split(":", 1)
        if not folder_id or not message_id:
            logger.warning("Invalid email ID format (empty parts): %s", email_id)
            return None

        account = await self._ensure_account()

        # Use the content endpoint with folderId from search results
        content_url = (
            f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}"
            f"/folders/{folder_id}/messages/{message_id}/content"
        )

        logger.debug("Fetching email content from: %s", content_url)
        resp = await self._http.get(
            content_url, headers=headers, params={"includeBlockContent": "true"}
        )
        resp.raise_for_status()
        content_data = resp.json().get("data", {})

        text_body = content_data.get("content") or ""

        # Strip HTML if content contains HTML tags
        if text_body and re.search(r"<[a-zA-Z][^>]*>", text_body):
            text_body = strip_html(text_body)

        # Truncate long bodies
        if len(text_body) > self._max_body_length:
            text_body = text_body[: self._max_body_length] + "\n\n[truncated]"

        # Get metadata from content response or use defaults
        from_addr = content_data.get("fromAddress", "")
        from_name = content_data.get("sender", "")
        to_list = (
            content_data.get("toAddress", "").split(",") if content_data.get("toAddress") else []
        )
        subject = content_data.get("subject", "(no subject)")
        received_time = content_data.get("receivedTime", 0)

        return EmailDetail(
            id=email_id,
            subject=subject,
            from_addresses=[EmailAddress(name=from_name or None, email=from_addr)],
            to_addresses=[EmailAddress(email=addr.strip()) for addr in to_list if addr.strip()],
            received_at=self._format_timestamp(received_time),
            text_body=text_body,
        )

    async def draft_response(
        self,
        to_addresses: list[str],
        subject: str,
        content: str,
        cc_addresses: list[str] | None = None,
        bcc_addresses: list[str] | None = None,
        in_reply_to: str | None = None,
        mail_format: str = "plaintext",
    ) -> str | None:
        """Save an email draft to the Drafts folder.

        Args:
            to_addresses: List of recipient email addresses
            subject: Email subject line
            content: Email body content
            cc_addresses: Optional CC recipients
            bcc_addresses: Optional BCC recipients
            in_reply_to: Optional Message-ID of email being replied to
            mail_format: 'plaintext' or 'html' (default: plaintext)

        Returns:
            The draft message ID if successful, None otherwise.
        """
        account = await self._ensure_account()
        headers = await self._get_headers()

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/messages"

        payload: dict[str, Any] = {
            "fromAddress": account.email_address,
            "toAddress": ",".join(to_addresses),
            "subject": subject,
            "content": content,
            "mode": "draft",
            "mailFormat": mail_format,
        }

        if cc_addresses:
            payload["ccAddress"] = ",".join(cc_addresses)
        if bcc_addresses:
            payload["bccAddress"] = ",".join(bcc_addresses)
        if in_reply_to:
            payload["inReplyTo"] = in_reply_to

        logger.info("Saving draft to %s: %s", to_addresses, subject)

        resp = await self._http.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Extract the draft message ID from response
        draft_data = data.get("data", {})
        message_id = draft_data.get("messageId")
        if message_id:
            logger.info("Draft saved successfully: messageId=%s", message_id)
            return str(message_id)

        logger.warning("Draft saved but no messageId returned: %s", data)
        return None

    async def create_folder(
        self,
        name: str,
        parent_folder_id: str | None = None,
    ) -> ZohoFolder | None:
        """Create a new email folder, optionally nested under a parent."""
        account = await self._ensure_account()
        headers = await self._get_headers()

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/folders"
        payload: dict[str, Any] = {"folderName": name}
        if parent_folder_id:
            payload["parentFolderId"] = parent_folder_id

        logger.info("Creating folder: %s (parent: %s)", name, parent_folder_id)
        resp = await self._http.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        folder_data = data.get("data", {})
        if folder_data.get("folderId"):
            self._folders = None
            return ZohoFolder(
                folder_id=str(folder_data["folderId"]),
                folder_name=folder_data.get("folderName", name),
                folder_type=folder_data.get("folderType", ""),
                path=folder_data.get("path", ""),
                is_archived=False,
            )
        logger.warning("Folder creation returned no folderId: %s", data)
        return None

    async def create_nested_folder(self, path: str) -> ZohoFolder | None:
        """Create a folder path, creating parent folders as needed.

        Args:
            path: Folder path like "Clients/John Smith" or "Accounting/Expenses/AWS"

        Returns:
            The final (deepest) folder, or None if creation failed.
        """
        parts = [p.strip() for p in path.split("/") if p.strip()]
        if not parts:
            return None

        parent_id: str | None = None
        current_folder: ZohoFolder | None = None

        for part in parts:
            existing = await self.get_folder_by_name(part)
            if existing:
                parent_id = existing.folder_id
                current_folder = existing
                continue

            current_folder = await self.create_folder(part, parent_folder_id=parent_id)
            if not current_folder:
                logger.error("Failed to create folder: %s", part)
                return None
            parent_id = current_folder.folder_id

        return current_folder

    async def move_messages(
        self,
        message_ids: list[str],
        dest_folder_id: str,
    ) -> bool:
        """Move messages to a destination folder."""
        if not message_ids:
            return True

        account = await self._ensure_account()
        headers = await self._get_headers()

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/updatemessage"

        pure_message_ids = []
        for mid in message_ids:
            if ":" in mid:
                _, msg_id = mid.split(":", 1)
                pure_message_ids.append(msg_id)
            else:
                pure_message_ids.append(mid)

        payload = {
            "mode": "moveMessage",
            "messageId": pure_message_ids,
            "destfolderId": dest_folder_id,
        }

        logger.info("Moving %d message(s) to folder %s", len(pure_message_ids), dest_folder_id)
        resp = await self._http.put(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", {}).get("code", 0)
        if status == 200:
            logger.info("Successfully moved %d message(s)", len(pure_message_ids))
            return True

        logger.warning("Move messages returned status: %s", data)
        return False

    async def get_labels(self) -> list[dict[str, Any]]:
        """Get all labels for the account."""
        account = await self._ensure_account()
        headers = await self._get_headers()

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/labels"
        params = {"fields": "labelId,displayName,color"}

        resp = await self._http.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        labels = data.get("data", [])
        logger.info("Loaded %d labels", len(labels))
        return labels

    async def get_label_by_name(self, name: str) -> dict[str, Any] | None:
        """Get a label by name (case-insensitive)."""
        labels = await self.get_labels()
        name_lower = name.lower()
        for label in labels:
            if label.get("displayName", "").lower() == name_lower:
                return label
        return None

    async def create_label(self, name: str, color: str = "#4285f4") -> dict[str, Any] | None:
        """Create a new label."""
        account = await self._ensure_account()
        headers = await self._get_headers()

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/labels"
        payload = {"displayName": name, "color": color}

        logger.info("Creating label: %s", name)
        resp = await self._http.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        label_data = data.get("data", {})
        if label_data.get("labelId"):
            return label_data
        logger.warning("Label creation returned no labelId: %s", data)
        return None

    async def apply_label(
        self,
        message_ids: list[str],
        label_id: str,
    ) -> bool:
        """Apply a label to messages."""
        if not message_ids:
            return True

        account = await self._ensure_account()
        headers = await self._get_headers()

        url = f"{PennyConstants.ZOHO_API_BASE}/accounts/{account.account_id}/updatemessage"

        pure_message_ids = []
        for mid in message_ids:
            if ":" in mid:
                _, msg_id = mid.split(":", 1)
                pure_message_ids.append(msg_id)
            else:
                pure_message_ids.append(mid)

        payload = {
            "mode": "applyLabel",
            "messageId": pure_message_ids,
            "labelId": [label_id],
        }

        logger.info("Applying label %s to %d message(s)", label_id, len(pure_message_ids))
        resp = await self._http.put(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", {}).get("code", 0)
        if status == 200:
            logger.info("Successfully applied label to %d message(s)", len(pure_message_ids))
            return True

        logger.warning("Apply label returned status: %s", data)
        return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
