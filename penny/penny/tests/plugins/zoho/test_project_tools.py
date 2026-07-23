"""Tests for the Zoho Projects tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from penny.constants import PennyConstants
from penny.plugins.zoho.project_tools import CreateTaskTool, UpdateTaskTool


@pytest.mark.asyncio
async def test_create_task_reports_failure_when_tasklist_creation_fails():
    """A failed auto-creation of the task list is an honest failure (``success=False``),
    not a silent success — and no task is created."""
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.get_task_lists = AsyncMock(return_value=[])
    client.create_task_list = AsyncMock(return_value=None)
    client.create_task = AsyncMock()

    result = await CreateTaskTool(client).execute(project_name="Acme", name="Ship it")

    assert result.success is False
    assert "Failed to create task list" in result.message
    client.create_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_task_reports_missing_project_as_failure():
    """A missing project is an honest failure, not a silent success."""
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=None)

    result = await CreateTaskTool(client).execute(project_name="Ghost", name="Ship it")

    assert result.success is False
    assert "Project not found" in result.message


@pytest.mark.asyncio
async def test_update_task_reports_missing_task_as_failure():
    """A task that can't be found is an honest failure — the update didn't happen."""
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.get_tasks = AsyncMock(return_value=[])

    result = await UpdateTaskTool(client).execute(project_name="Acme", task_name="Ghost")

    assert result.success is False
    assert "Task not found" in result.message


@pytest.mark.asyncio
async def test_create_task_defaults_to_general_tasklist():
    """With no tasklist_name, the task is filed under the default 'General' list — the value
    now named PennyConstants.ZOHO_PROJECTS_DEFAULT_TASKLIST, unchanged by the constant swap."""
    assert PennyConstants.ZOHO_PROJECTS_DEFAULT_TASKLIST == "General"

    project = MagicMock(id="P1")
    project.name = "Acme"
    tasklist = MagicMock(id="TL1")
    tasklist.name = "General"
    task = MagicMock()
    task.name = "Ship it"

    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=project)
    client.get_task_lists = AsyncMock(return_value=[])
    client.create_task_list = AsyncMock(return_value=tasklist)
    client.create_task = AsyncMock(return_value=task)

    result = await CreateTaskTool(client).execute(project_name="Acme", name="Ship it")

    assert result.success
    client.create_task_list.assert_awaited_once_with("P1", "General")
