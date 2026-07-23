"""Email rule matching and application logic for the Zoho plugin."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from penny.email.models import EmailDetail, EmailSummary
from penny.plugins.zoho.rule_models import EmailRuleAction, EmailRuleCondition

if TYPE_CHECKING:
    from penny.database import Database
    from penny.plugins.zoho.mail_client import ZohoClient

logger = logging.getLogger(__name__)


class RuleMatcher:
    """Matches emails against rule conditions."""

    @staticmethod
    def matches(email: EmailSummary | EmailDetail, condition: EmailRuleCondition) -> bool:
        """Check if an email matches a rule condition.

        Supported condition fields:
        - from: Sender email or domain (partial match)
        - subject_contains: Text in subject (case-insensitive)
        - body_contains: Text in body (case-insensitive, EmailDetail only)
        """
        if condition.from_ is not None:
            from_pattern = condition.from_.lower()
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

        if condition.subject_contains is not None:
            pattern = condition.subject_contains.lower()
            if pattern not in email.subject.lower():
                return False

        if (
            condition.body_contains is not None
            and isinstance(email, EmailDetail)
            and email.text_body
        ):
            pattern = condition.body_contains.lower()
            if pattern not in email.text_body.lower():
                return False

        return True


class RuleExecutor:
    """Executes rule actions on matched emails."""

    def __init__(self, zoho_client: ZohoClient) -> None:
        self._client = zoho_client

    async def execute(self, email_ids: list[str], action: EmailRuleAction) -> dict[str, bool]:
        """Execute a rule action on a list of emails.

        Supported action fields:
        - move_to: Folder path to move emails to
        - label: Label name to apply
        """
        results: dict[str, bool] = {}

        if action.move_to is not None:
            folder_path = action.move_to
            folder = await self._client.create_nested_folder(folder_path)
            if folder:
                success = await self._client.move_messages(email_ids, folder.folder_id)
                results["move_to"] = success
                if success:
                    logger.info("Moved %d email(s) to '%s'", len(email_ids), folder_path)
            else:
                results["move_to"] = False
                logger.warning("Failed to create folder: %s", folder_path)

        if action.label is not None:
            label_name = action.label
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
    emails: list[EmailSummary],
    provider: str,
) -> dict[str, list[str]]:
    """Apply all active rules for a provider to a list of emails."""
    results: dict[str, list[str]] = {}

    rules = db.email_rules.list_active(provider)
    if not rules:
        logger.debug("No email rules configured for provider %s", provider)
        return results

    logger.info("Applying %d email rule(s) to %d email(s)", len(rules), len(emails))

    matcher = RuleMatcher()
    executor = RuleExecutor(zoho_client)

    for rule in rules:
        condition = EmailRuleCondition.model_validate_json(rule.condition)
        action = EmailRuleAction.model_validate_json(rule.action)
        matched_ids = [email.id for email in emails if matcher.matches(email, condition)]

        if matched_ids:
            logger.info("Rule '%s' matched %d email(s)", rule.name, len(matched_ids))
            await executor.execute(matched_ids, action)
            if rule.id is not None:
                db.email_rules.mark_applied(rule.id)
            results[rule.name] = matched_ids

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
