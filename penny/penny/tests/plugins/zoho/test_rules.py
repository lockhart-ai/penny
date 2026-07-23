"""Tests for Zoho email rule matching and formatting."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from penny.constants import PennyConstants
from penny.email.models import EmailAddress, EmailSummary
from penny.plugins.zoho.rule_models import EmailRuleAction, EmailRuleCondition
from penny.plugins.zoho.rules import RuleMatcher, apply_email_rules, format_rule_results


@pytest.fixture
def sample_email() -> EmailSummary:
    return EmailSummary(
        id="F001:M1",
        subject="Your AWS invoice for July",
        from_addresses=[EmailAddress(name="AWS", email="aws@amazon.com")],
        received_at="2026-07-20T10:00:00+00:00",
        preview="Invoice #12345",
    )


def _condition(**fields: str) -> EmailRuleCondition:
    return EmailRuleCondition.model_validate(fields)


def test_rule_matcher_from_domain(sample_email: EmailSummary) -> None:
    matcher = RuleMatcher()
    assert matcher.matches(sample_email, _condition(**{"from": "amazon.com"})) is True
    assert matcher.matches(sample_email, _condition(**{"from": "google.com"})) is False


def test_rule_matcher_subject_contains(sample_email: EmailSummary) -> None:
    matcher = RuleMatcher()
    assert matcher.matches(sample_email, _condition(subject_contains="invoice")) is True
    assert matcher.matches(sample_email, _condition(subject_contains="receipt")) is False


def test_rule_matcher_combined_conditions(sample_email: EmailSummary) -> None:
    matcher = RuleMatcher()
    assert (
        matcher.matches(sample_email, _condition(**{"from": "aws", "subject_contains": "July"}))
        is True
    )
    assert (
        matcher.matches(sample_email, _condition(**{"from": "aws", "subject_contains": "receipt"}))
        is False
    )


@pytest.mark.asyncio
async def test_apply_email_rules_matches_and_marks_applied(db, sample_email: EmailSummary) -> None:
    """apply_email_rules runs active rules, executes the action, and stamps applied."""
    db.email_rules.create(
        provider=PennyConstants.PROVIDER_ZOHO,
        name="AWS invoices",
        condition=EmailRuleCondition.model_validate({"from": "amazon.com"}).model_dump_json(
            by_alias=True, exclude_none=True
        ),
        action=EmailRuleAction(move_to="Accounting/Expenses/AWS").model_dump_json(
            by_alias=True, exclude_none=True
        ),
    )

    client = MagicMock()
    folder = MagicMock(folder_id="F123")
    client.create_nested_folder = AsyncMock(return_value=folder)
    client.move_messages = AsyncMock(return_value=True)

    results = await apply_email_rules(db, client, [sample_email], PennyConstants.PROVIDER_ZOHO)

    assert results == {"AWS invoices": ["F001:M1"]}
    client.move_messages.assert_awaited_once_with(["F001:M1"], "F123")
    applied = db.email_rules.list_active(PennyConstants.PROVIDER_ZOHO)
    assert applied[0].last_applied_at is not None


@pytest.mark.asyncio
async def test_apply_email_rules_no_rules(db, sample_email: EmailSummary) -> None:
    """With no rules configured, apply_email_rules is an empty no-op."""
    client = MagicMock()
    results = await apply_email_rules(db, client, [sample_email], PennyConstants.PROVIDER_ZOHO)
    assert results == {}


def test_format_rule_results() -> None:
    results = {
        "AWS invoices": ["F001:M1", "F001:M2"],
        "GitHub notifications": ["F002:M3"],
    }
    text = format_rule_results(results)
    assert "AWS invoices" in text
    assert "2 email(s)" in text
    assert "GitHub notifications" in text
    assert "Total: 3 email(s) processed" in text


def test_format_rule_results_empty() -> None:
    assert format_rule_results({}) == "No email rules were applied."
