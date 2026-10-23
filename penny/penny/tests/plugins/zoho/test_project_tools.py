"""Tests for the Zoho Projects tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from penny.constants import PennyConstants
from penny.llm.call_link import CallLink
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

    result = await CreateTaskTool(client).execute(
        CallLink.new_top(), project_name="Acme", name="Ship it"
    )

    assert result.success is False
    assert "Failed to create task list" in result.message
    client.create_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_task_reports_missing_project_as_failure():
    """A missing project is an honest failure, not a silent success."""
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=None)

    result = await CreateTaskTool(client).execute(
        CallLink.new_top(), project_name="Ghost", name="Ship it"
    )

    assert result.success is False
    assert "Project not found" in result.message


@pytest.mark.asyncio
async def test_update_task_reports_missing_task_as_failure():
    """A task that can't be found is an honest failure — the update didn't happen."""
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.get_tasks = AsyncMock(return_value=[])

    result = await UpdateTaskTool(client).execute(
        CallLink.new_top(), project_name="Acme", task_name="Ghost"
    )

    assert result.success is False
    assert "Task not found" in result.message


@pytest.mark.asyncio
async def test_update_task_renders_from_the_confirmed_response():
    """The change summary is built from what the backend CONFIRMS it did (the
    returned task's own fields), never the call's own args (#2768) — a field the
    backend rejects within an otherwise-successful call would otherwise be
    claimed as landed."""
    task = MagicMock(id="T1")
    task.name = "Ship it"
    updated = MagicMock()
    updated.name = "Ship it v2"
    updated.priority = "high"
    updated.completion_percentage = 50
    updated.description = "Now with more detail"
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.get_tasks = AsyncMock(return_value=[task])
    client.update_task = AsyncMock(return_value=updated)

    result = await UpdateTaskTool(client).execute(
        CallLink.new_top(),
        project_name="Acme",
        task_name="Ship it",
        new_name="Ship it v2",
        priority="high",
        completion_percentage=50,
        description="Now with more detail",
    )

    assert result.success is True
    assert result.mutated is True
    assert result.message == (
        "Task 'Ship it v2' updated: name -> 'Ship it v2', priority -> high, "
        "progress -> 50%, description -> Now with more detail"
    )


@pytest.mark.asyncio
async def test_update_task_names_other_matches_when_the_name_is_ambiguous():
    """More than one task sharing the name picks the first, but says so (#2768) —
    a silent pick would read as an unambiguous update when it wasn't."""
    first = MagicMock(id="T1")
    first.name = "Ship it"
    second = MagicMock(id="T2")
    second.name = "Ship it"
    updated = MagicMock()
    updated.name = "Ship it"
    updated.priority = "high"
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.get_tasks = AsyncMock(return_value=[first, second])
    client.update_task = AsyncMock(return_value=updated)

    result = await UpdateTaskTool(client).execute(
        CallLink.new_top(), project_name="Acme", task_name="Ship it", priority="high"
    )

    assert result.success is True
    await_args = client.update_task.await_args
    assert await_args is not None
    assert await_args.kwargs["task_id"] == "T1"
    assert (
        "Heads up: 2 tasks named 'Ship it' in this project — updated the first "
        "one; the others weren't touched."
    ) in result.message


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

    result = await CreateTaskTool(client).execute(
        CallLink.new_top(), project_name="Acme", name="Ship it"
    )

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

    result = await ListProjectsTool(client).execute(CallLink.new_top())

    assert result.success is False
    assert result.message == LIST_PROJECTS_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_get_project_details_reports_a_raised_fetch():
    client = MagicMock()
    client.get_project_by_name = AsyncMock(
        side_effect=RuntimeError("Zoho OAuth error: invalid_grant")
    )

    result = await GetProjectDetailsTool(client).execute(CallLink.new_top(), project_name="Acme")

    assert result.success is False
    assert result.message == PROJECT_DETAILS_FAILED.format(e="Zoho OAuth error: invalid_grant")


@pytest.mark.asyncio
async def test_create_project_reports_a_raised_create():
    client = MagicMock()
    client.create_project = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await CreateProjectTool(client).execute(CallLink.new_top(), name="Acme")

    assert result.success is False
    assert result.message == CREATE_PROJECT_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_list_task_lists_reports_a_raised_listing():
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.get_task_lists = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await ListTaskListsTool(client).execute(CallLink.new_top(), project_name="Acme")

    assert result.success is False
    assert result.message == LIST_TASK_LISTS_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_create_task_list_reports_a_raised_create():
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.create_task_list = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await CreateTaskListTool(client).execute(
        CallLink.new_top(), project_name="Acme", name="Phase 2"
    )

    assert result.success is False
    assert result.message == CREATE_TASK_LIST_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_list_tasks_reports_a_raised_listing():
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.get_tasks = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await ListTasksTool(client).execute(CallLink.new_top(), project_name="Acme")

    assert result.success is False
    assert result.message == LIST_TASKS_FAILED.format(e="connection refused")


@pytest.mark.asyncio
async def test_create_task_reports_a_raised_create():
    client = MagicMock()
    client.get_project_by_name = AsyncMock(return_value=MagicMock(id="P1"))
    client.get_task_lists = AsyncMock(return_value=[])
    client.create_task_list = AsyncMock(return_value=MagicMock(id="TL1"))
    client.create_task = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await CreateTaskTool(client).execute(
        CallLink.new_top(), project_name="Acme", name="Ship it"
    )

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

    result = await UpdateTaskTool(client).execute(
        CallLink.new_top(), project_name="Acme", task_name="Ship it"
    )

    assert result.success is False
    assert result.message == UPDATE_TASK_FAILED.format(e="connection refused")
