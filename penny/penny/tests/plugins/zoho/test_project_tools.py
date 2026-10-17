"""Tests for the Zoho Projects tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from penny.constants import PennyConstants
from penny.plugins.zoho.project_tools import (
    CREATE_PROJECT_FAILED,
    CREATE_TASK_FAILED,
    CREATE_TASK_LIST_FAILED,
    LIST_PROJECTS_FAILED,
    LIST_TASK_LISTS_FAILED,
    LIST_TASKS_FAILED,
    PROJECT_DETAILS_FAILED,
    UPDATE_TASK_FAILED,
    CreateProjectTool,
    CreateTaskListTool,
    CreateTaskTool,
    GetProjectDetailsTool,
    ListProjectsTool,
    ListTaskListsTool,
    ListTasksTool,
    UpdateTaskTool,
)


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


# ── Projects tools report their own client failures (#2369) ──────────────────
# Each tool's backend call is wrapped: a raised failure is caught and narrated — what
# happened, with no order attached — rather than falling through to the executor's generic
# "check the arguments" narration, which makes no sense for a backend outage. Mocked at the
# backend boundary (the client each tool holds), the same seam every other test in this file
# mocks at. The catch is narrow — ``httpx.HTTPError`` (a non-2xx response or a transport
# error) plus ``RuntimeError`` (the projects client's shared ``ZohoOAuthClient`` raises this
# on a failed token refresh) — so a bug in this file's own logic is never mislabeled as a
# Projects outage.


@pytest.mark.asyncio
async def test_list_projects_reports_a_raised_listing():
    client = MagicMock()
    client.get_projects = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await ListProjectsTool(client).execute()

    assert result.success is False
    assert result.message == LIST_PROJECTS_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_get_project_details_reports_a_raised_fetch():
    client = MagicMock()
    client.get_project_by_name = AsyncMock(
        side_effect=RuntimeError("Zoho OAuth error: invalid_grant")
    )

    result = await GetProjectDetailsTool(client).execute(project_name="Acme")

    assert result.success is False
    assert result.message == PROJECT_DETAILS_FAILED.format(e="Zoho OAuth error: invalid_grant")


@pytest.mark.asyncio
async def test_create_project_reports_a_raised_create():
    client = MagicMock()
    client.create_project = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await CreateProjectTool(client).execute(name="Acme")

    assert result.success is False
    assert result.message == CREATE_PROJECT_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_list_task_lists_reports_a_raised_listing():
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.get_task_lists = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await ListTaskListsTool(client).execute(project_name="Acme")

    assert result.success is False
    assert result.message == LIST_TASK_LISTS_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_create_task_list_reports_a_raised_create():
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.create_task_list = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await CreateTaskListTool(client).execute(project_name="Acme", name="Phase 2")

    assert result.success is False
    assert result.message == CREATE_TASK_LIST_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_list_tasks_reports_a_raised_listing():
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.get_tasks = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await ListTasksTool(client).execute(project_name="Acme")

    assert result.success is False
    assert result.message == LIST_TASKS_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_create_task_reports_a_raised_create():
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.get_task_lists = AsyncMock(return_value=[])
    client.create_task_list = AsyncMock(return_value=MagicMock(id="TL1"))
    client.create_task = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await CreateTaskTool(client).execute(project_name="Acme", name="Ship it")

    assert result.success is False
    assert result.message == CREATE_TASK_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_update_task_reports_a_raised_update():
    task = MagicMock(id="T1")
    task.name = "Ship it"
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.get_tasks = AsyncMock(return_value=[task])
    client.update_task = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await UpdateTaskTool(client).execute(project_name="Acme", task_name="Ship it")

    assert result.success is False
    assert result.message == UPDATE_TASK_FAILED.format(e="connection refused")
