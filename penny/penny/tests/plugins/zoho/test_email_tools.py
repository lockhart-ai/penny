"""Tests for Zoho email organisation plugin tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from penny.constants import PennyConstants
from penny.plugins.zoho.email_tools import (
    APPLY_LABEL_FAILED,
    CREATE_FOLDER_FAILED,
    LIST_LABELS_FAILED,
    MOVE_EMAILS_FAILED,
    ApplyLabelTool,
    CreateEmailRuleTool,
    CreateFolderTool,
    ListEmailRulesTool,
    ListLabelsTool,
    MoveEmailsTool,
)
from penny.plugins.zoho.rule_models import EmailRuleAction, EmailRuleCondition
from penny.tools.models import ToolResult


@pytest.mark.asyncio
async def test_move_emails_to_existing_folder():
    """MoveEmailsTool moves messages when the folder already exists."""
    client = MagicMock()
    folder = MagicMock(folder_id="F123", folder_name="Invoices")
    client.get_folder_by_name = AsyncMock(return_value=folder)
    client.move_messages = AsyncMock(return_value=True)

    tool = MoveEmailsTool(client)
    result = await tool.execute(
        message_ids=["F001:M1", "F001:M2"],
        folder_path="Invoices",
        create_if_missing=True,
    )

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.mutated is True
    client.move_messages.assert_awaited_once_with(["F001:M1", "F001:M2"], "F123")


@pytest.mark.asyncio
async def test_move_emails_creates_folder_when_missing():
    """MoveEmailsTool creates the destination folder if it doesn't exist."""
    client = MagicMock()
    client.get_folder_by_name = AsyncMock(return_value=None)
    folder = MagicMock(folder_id="F123", folder_name="Invoices")
    client.create_nested_folder = AsyncMock(return_value=folder)
    client.move_messages = AsyncMock(return_value=True)

    tool = MoveEmailsTool(client)
    result = await tool.execute(
        message_ids=["F001:M1"],
        folder_path="Accounting/Expenses/AWS",
        create_if_missing=True,
    )

    assert result.success is True
    client.create_nested_folder.assert_awaited_once_with("Accounting/Expenses/AWS")


@pytest.mark.asyncio
async def test_move_emails_no_message_ids():
    """MoveEmailsTool fails gracefully when no message IDs are supplied."""
    client = MagicMock()
    tool = MoveEmailsTool(client)
    result = await tool.execute(
        message_ids=[],
        folder_path="Invoices",
        create_if_missing=True,
    )

    assert isinstance(result, ToolResult)
    assert result.success is False
    client.move_messages.assert_not_called()


@pytest.mark.asyncio
async def test_create_folder_tool():
    """CreateFolderTool returns success when the folder is created."""
    client = MagicMock()
    folder = MagicMock(folder_id="F123", folder_name="Clients/Acme")
    client.create_nested_folder = AsyncMock(return_value=folder)

    tool = CreateFolderTool(client)
    result = await tool.execute(folder_path="Clients/Acme")

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.mutated is True


@pytest.mark.asyncio
async def test_apply_label_tool_creates_label_when_missing():
    """ApplyLabelTool creates a missing label before applying it."""
    client = MagicMock()
    client.get_label_by_name = AsyncMock(return_value=None)
    client.create_label = AsyncMock(return_value={"labelId": "L123", "displayName": "Done"})
    client.apply_label = AsyncMock(return_value=True)

    tool = ApplyLabelTool(client)
    result = await tool.execute(
        message_ids=["F001:M1"],
        label_name="Done",
        create_if_missing=True,
    )

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.mutated is True
    client.create_label.assert_awaited_once_with("Done")
    client.apply_label.assert_awaited_once_with(["F001:M1"], "L123")


@pytest.mark.asyncio
async def test_list_labels_tool():
    """ListLabelsTool returns a formatted label list."""
    client = MagicMock()
    client.get_labels = AsyncMock(
        return_value=[
            {"labelId": "L1", "displayName": "Work", "color": "#4285f4"},
            {"labelId": "L2", "displayName": "Personal", "color": "#34a853"},
        ]
    )

    tool = ListLabelsTool(client)
    result = await tool.execute()

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert "Work" in result.message
    assert "Personal" in result.message


@pytest.mark.asyncio
async def test_create_email_rule_persists_via_store(db):
    """create_email_rule stores a typed rule through db.email_rules and echoes it."""
    tool = CreateEmailRuleTool(db, PennyConstants.PROVIDER_ZOHO)
    result = await tool.run(
        name="AWS invoices",
        condition={"from": "aws@amazon.com"},
        action={"move_to": "Accounting/Expenses/AWS"},
    )

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.mutated is True
    assert (
        result.message
        == """Email rule 'AWS invoices' saved.

Condition: from=aws@amazon.com
Action: move_to=Accounting/Expenses/AWS

Note: saved rules aren't applied automatically yet — this stores the rule for future use."""
    )

    rules = db.email_rules.list_active(PennyConstants.PROVIDER_ZOHO)
    assert len(rules) == 1
    assert rules[0].name == "AWS invoices"
    assert rules[0].provider == "zoho"
    assert rules[0].enabled is True
    assert rules[0].last_applied_at is None
    assert EmailRuleCondition.model_validate_json(rules[0].condition).from_ == "aws@amazon.com"
    assert EmailRuleAction.model_validate_json(rules[0].action).move_to == "Accounting/Expenses/AWS"


@pytest.mark.asyncio
async def test_list_email_rules_renders_active_rules(db):
    """list_email_rules reads active rules via the store and renders them."""
    create = CreateEmailRuleTool(db, PennyConstants.PROVIDER_ZOHO)
    await create.run(
        name="AWS invoices",
        condition={"from": "aws@amazon.com"},
        action={"move_to": "Accounting/Expenses/AWS"},
    )

    result = await ListEmailRulesTool(db, PennyConstants.PROVIDER_ZOHO).run()

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert (
        result.message
        == """Found 1 active email rule(s):

1. **AWS invoices**
   Condition: from=aws@amazon.com
   Action: move_to=Accounting/Expenses/AWS
"""
    )


@pytest.mark.asyncio
async def test_list_email_rules_when_none(db):
    """list_email_rules reports an empty rule set honestly."""
    result = await ListEmailRulesTool(db, PennyConstants.PROVIDER_ZOHO).run()
    assert result.success is True
    assert result.message == "No email rules configured."


@pytest.mark.asyncio
async def test_create_email_rule_rejects_empty_condition(db):
    """An all-empty condition is refused with an actionable, field-naming message."""
    result = await CreateEmailRuleTool(db, PennyConstants.PROVIDER_ZOHO).run(
        name="catch all",
        condition={},
        action={"move_to": "Somewhere"},
    )

    assert result.success is False
    assert result.mutated is False
    assert result.narration == "You tried to use `create_email_rule` but the arguments were wrong:"
    assert result.message == (
        "condition (object: Rule condition. Supported fields: 'from' (sender email/domain), "
        "'subject_contains' (text in subject), 'body_contains' (text in body)): a rule condition "
        "needs at least one of: from, subject_contains, body_contains. "
        "Call create_email_rule(<valid arguments>) again."
    )
    assert db.email_rules.list_active(PennyConstants.PROVIDER_ZOHO) == []


@pytest.mark.asyncio
async def test_create_email_rule_rejects_unknown_field(db):
    """An unknown condition field is refused, not silently absorbed."""
    result = await CreateEmailRuleTool(db, PennyConstants.PROVIDER_ZOHO).run(
        name="typo rule",
        condition={"sender": "aws"},
        action={"label": "invoices"},
    )

    assert result.success is False
    assert result.mutated is False
    assert result.message == (
        "unknown parameter 'condition.sender' "
        "(valid parameters: from, subject_contains, body_contains). "
        "Call create_email_rule(<valid arguments>) again."
    )
    assert db.email_rules.list_active(PennyConstants.PROVIDER_ZOHO) == []


# ── Mail organisation tools report their own client failures (#2369) ─────────
# Each tool's backend call is wrapped: a raised failure is caught and narrated — what
# happened, with no order attached — rather than falling through to the executor's generic
# "check the arguments" narration, which makes no sense for a backend outage. Mocked at the
# backend boundary (the client each tool holds), the same seam every other test in this file
# mocks at. The catch is narrow — ``httpx.HTTPError`` (a non-2xx response or a transport
# error) plus ``RuntimeError`` (``ZohoClient`` raises this on its own OAuth token refresh and
# on a missing mail account) — so a bug in this file's own logic is never mislabeled as a
# mailbox outage. ``create_email_rule``/``list_email_rules`` reach only ``db.email_rules``
# (no network client), so they get no such test.


@pytest.mark.asyncio
async def test_move_emails_reports_a_raised_move():
    client = MagicMock()
    folder = MagicMock(folder_id="F123")
    client.get_folder_by_name = AsyncMock(return_value=folder)
    client.move_messages = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await MoveEmailsTool(client).execute(
        message_ids=["F001:M1"],
        folder_path="Invoices",
    )

    assert result.success is False
    assert result.message == MOVE_EMAILS_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_create_folder_reports_a_raised_create():
    client = MagicMock()
    client.create_nested_folder = AsyncMock(
        side_effect=RuntimeError("Zoho OAuth error: invalid_grant")
    )

    result = await CreateFolderTool(client).execute(folder_path="Clients/Acme")

    assert result.success is False
    assert result.message == CREATE_FOLDER_FAILED.format(e="Zoho OAuth error: invalid_grant")


@pytest.mark.asyncio
async def test_apply_label_reports_a_raised_apply():
    client = MagicMock()
    client.get_label_by_name = AsyncMock(return_value={"labelId": "L1"})
    client.apply_label = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await ApplyLabelTool(client).execute(
        message_ids=["F001:M1"],
        label_name="Done",
    )

    assert result.success is False
    assert result.message == APPLY_LABEL_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_list_labels_reports_a_raised_listing():
    client = MagicMock()
    client.get_labels = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await ListLabelsTool(client).execute()

    assert result.success is False
    assert result.message == LIST_LABELS_FAILED.format(e="connection refused")
