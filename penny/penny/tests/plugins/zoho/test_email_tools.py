"""Tests for Zoho email organisation plugin tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from penny.plugins.zoho.email_tools import (
    ApplyLabelTool,
    CreateFolderTool,
    ListLabelsTool,
    MoveEmailsTool,
)
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
