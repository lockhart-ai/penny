"""Email rule matching and application logic for the Zoho plugin."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlmodel import select

from penny.database.models import EmailRule
from penny.email.models import EmailDetail, EmailSummary

if TYPE_CHECKING:
    from penny.database import Database
    from penny.zoho.client import ZohoClient

logger = logging.getLogger(__name__)


class RuleMatcher:
    """Matches emails against rule conditions."""

    @staticmethod
    def matches(email: EmailSummary | EmailDetail, condition: dict) -> bool:
        """Check if an email matches a rule condition.

        Supported condition fields:
        - from: Sender email or domain (partial match)
        - subject_contains: Text in subject (case-insensitive)
        - body_contains: Text in body (case-insensitive, EmailDetail only)
        """
        if "from" in condition:
            from_pattern = condition["from"].lower()
            sender_match = False
            for addr in email.from_addresses:
                if addr.email and from_pattern in addr.email.lower():
                    sender_match = True
                    break
                if addr.name and from_pattern in addr.name.lower():
                    sender_match = True
                    break
            if not sender_match:
                return False

        if "subject_contains" in condition:
            pattern = condition["subject_contains"].lower()
            if pattern not in email.subject.lower():
                return False

        if "body_contains" in condition and isinstance(email, EmailDetail) and email.text_body:
            pattern = condition["body_contains"].lower()
            if pattern not in email.text_body.lower():
                return False

        return True


class RuleExecutor:
    """Executes rule actions on matched emails."""

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, email_ids: list[str], action: dict) -> dict[str, bool]:
        """Execute a rule action on a list of emails.

        Supported action fields:
        - move_to: Folder path to move emails to
        - label: Label name to apply
        """
        results: dict[str, bool] = {}

        if "move_to" in action:
            folder_path = action["move_to"]
            folder = await self._client.create_nested_folder(folder_path)
            if folder:
                success = await self._client.move_messages(email_ids, folder.folder_id)
                results["move_to"] = success
                if success:
                    logger.info("Moved %d email(s) to '%s'", len(email_ids), folder_path)
            else:
                results["move_to"] = False
                logger.warning("Failed to create folder: %s", folder_path)

        if "label" in action:
            label_name = action["label"]
            label = await self._client.get_label_by_name(label_name)
            if not label:
                label = await self._client.create_label(label_name)

            if label:
                label_id = label.get("labelId", "")
                success = await self._client.apply_label(email_ids, label_id)
                results["label"] = success
                if success:
                    logger.info("Applied label '%s' to %d email(s)", label_name, len(email_ids))
            else:
                results["label"] = False
                logger.warning("Failed to create label: %s", label_name)

        return results


async def apply_email_rules(
    db: Database,
    zoho_client: ZohoClient,
    user_id: str,
    emails: list[EmailSummary],
) -> dict[str, list[str]]:
    """Apply all active email rules to a list of emails."""
    results: dict[str, list[str]] = {}

    with db.get_session() as session:
        rules = list(
            session.exec(
                select(EmailRule)
                .where(EmailRule.user_id == user_id)
                .where(EmailRule.provider == "zoho")
                .where(EmailRule.enabled == True)  # noqa: E712
            )
        )

        if not rules:
            logger.debug("No email rules configured for user %s", user_id)
            return results

        logger.info("Applying %d email rule(s) to %d email(s)", len(rules), len(emails))

        matcher = RuleMatcher()
        executor = RuleExecutor(zoho_client)

        for rule in rules:
            condition = rule.get_condition()
            action = rule.get_action()
            matched_ids: list[str] = []

            for email in emails:
                if matcher.matches(email, condition):
                    matched_ids.append(email.id)

            if matched_ids:
                logger.info("Rule '%s' matched %d email(s)", rule.name, len(matched_ids))
                await executor.execute(matched_ids, action)
                rule.last_applied_at = datetime.now(UTC)
                session.add(rule)
                results[rule.name] = matched_ids

        session.commit()

    return results


def format_rule_results(results: dict[str, list[str]]) -> str:
    """Format rule application results for display."""
    if not results:
        return "No email rules were applied."

    lines = ["**Email Rules Applied:**\n"]
    total_processed = 0

    for rule_name, email_ids in results.items():
        count = len(email_ids)
        total_processed += count
        lines.append(f"- **{rule_name}**: {count} email(s)")

    lines.append(f"\nTotal: {total_processed} email(s) processed by rules.")
    return "\n".join(lines)
