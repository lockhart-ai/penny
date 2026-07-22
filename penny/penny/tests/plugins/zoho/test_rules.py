"""Tests for Zoho email rule matching and formatting."""

from __future__ import annotations

import pytest

from penny.jmap.models import EmailAddress, EmailSummary
from penny.plugins.zoho.rules import RuleMatcher, format_rule_results


@pytest.fixture
def sample_email() -> EmailSummary:
    return EmailSummary(
        id="F001:M1",
        subject="Your AWS invoice for July",
        from_addresses=[EmailAddress(name="AWS", email="aws@amazon.com")],
        received_at="2026-07-20T10:00:00+00:00",
        preview="Invoice #12345",
    )


def test_rule_matcher_from_domain(sample_email: EmailSummary) -> None:
    matcher = RuleMatcher()
    assert matcher.matches(sample_email, {"from": "amazon.com"}) is True
    assert matcher.matches(sample_email, {"from": "google.com"}) is False


def test_rule_matcher_subject_contains(sample_email: EmailSummary) -> None:
    matcher = RuleMatcher()
    assert matcher.matches(sample_email, {"subject_contains": "invoice"}) is True
    assert matcher.matches(sample_email, {"subject_contains": "receipt"}) is False


def test_rule_matcher_combined_conditions(sample_email: EmailSummary) -> None:
    matcher = RuleMatcher()
    assert (
        matcher.matches(
            sample_email,
            {"from": "aws", "subject_contains": "July"},
        )
        is True
    )
    assert (
        matcher.matches(
            sample_email,
            {"from": "aws", "subject_contains": "receipt"},
        )
        is False
    )


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
