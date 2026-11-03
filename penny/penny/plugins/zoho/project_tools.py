"""Zoho Projects tools — LLM-callable tools for project management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import Field

from penny.constants import PennyConstants
from penny.llm.call_link import CallLink
from penny.tools.base import PLAN_ARG_DESCRIPTION, Tool
from penny.tools.models import ToolArgs, ToolResult
from penny.tools.plan_tool import ResolvingMutationTool

logger = logging.getLogger(__name__)

# The account a client failure gets (#2369): what happened, with no order attached — the
# model reads the state and reasons from it, same as every other tool result. Catches the
# projects client's own failures — a non-2xx response or a transport error (both
# ``httpx.HTTPError``), or a failed OAuth token refresh (``RuntimeError``, raised by the
# shared ``ZohoOAuthClient``) — never a bug in this file's own logic, which still propagates
# to the executor's generic handler as today.
LIST_PROJECTS_FAILED = "Projects couldn't be listed ({e}), so nothing was read from Projects."
PROJECT_DETAILS_FAILED = (
    "That project's details couldn't be fetched ({e}), so nothing was read from Projects."
)
CREATE_PROJECT_FAILED = "The project couldn't be created ({e}), so nothing was added to Projects."
LIST_TASK_LISTS_FAILED = (
    "The task lists couldn't be listed ({e}), so nothing was read from Projects."
)
CREATE_TASK_LIST_FAILED = (
    "The task list couldn't be created ({e}), so nothing was added to Projects."
)
LIST_TASKS_FAILED = "Tasks couldn't be listed ({e}), so nothing was read from Projects."
CREATE_TASK_FAILED = "The task couldn't be created ({e}), so nothing was added to Projects."
UPDATE_TASK_FAILED = "The task couldn't be updated ({e}), so nothing was changed in Projects."


class ListProjectsArgs(ToolArgs):
    """Arguments for listing projects."""


class GetProjectArgs(ToolArgs):
    """Arguments for getting project details."""

    project_name: str = Field(description="Project name to look up")


class CreateProjectArgs(ToolArgs):
    """Arguments for creating a project."""

    name: str = Field(description="Project name")
    description: str | None = Field(default=None, description="Project description")
    start_date: str | None = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: str | None = Field(default=None, description="End date (YYYY-MM-DD)")
    # The structural plan mark (#2947, following #2933) — every ``mutates=True``
    # tool declares this literal field itself (``Tool.__init_subclass__`` asserts
    # it's here, rather than synthesising it, so a type checker can see it on
    # ``args.plan``).
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


class ListTasksArgs(ToolArgs):
    """Arguments for listing tasks."""

    project_name: str = Field(description="Project name to list tasks from")


class CreateTaskArgs(ToolArgs):
    """Arguments for creating a task."""

    project_name: str = Field(description="Project name to add task to")
    name: str = Field(description="Task name")
    tasklist_name: str | None = Field(
        default=None, description="Task list name (creates 'General' if not specified)"
    )
    description: str | None = Field(default=None, description="Task description")
    start_date: str | None = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: str | None = Field(default=None, description="End date (YYYY-MM-DD)")
    priority: str = Field(default="none", description="Priority: none, low, medium, high")
    assignees: list[str] | None = Field(default=None, description="List of assignee ZPUIDs")
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


class UpdateTaskArgs(ToolArgs):
    """Arguments for updating a task."""

    project_name: str = Field(description="Project name containing the task")
    task_name: str = Field(description="Task name to update")
    new_name: str | None = Field(default=None, description="New task name")
    description: str | None = Field(default=None, description="New description")
    priority: str | None = Field(default=None, description="New priority")
    status_id: str | None = Field(default=None, description="New status ID")
    completion_percentage: int | None = Field(
        default=None, description="Completion percentage (0-100)"
    )
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


class CreateTaskListArgs(ToolArgs):
    """Arguments for creating a task list."""

    project_name: str = Field(description="Project name to add task list to")
    name: str = Field(description="Task list name")
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


class ListProjectsTool(Tool):
    """List all projects in the portal."""

    name = "list_projects"
    description = (
        "List all projects in the Zoho Projects portal. Returns project names, "
        "statuses, and owners. Use this to discover what projects exist before "
        "creating tasks or viewing project details."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    args_model = ListProjectsArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing projects"

    def __init__(self, projects_client: Any) -> None:
        self._client = projects_client

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """List all projects."""
        try:
            projects = await self._client.get_projects()
        except (httpx.HTTPError, RuntimeError) as e:
            logger.exception("Listing projects failed")
            return ToolResult(message=LIST_PROJECTS_FAILED.format(e=e), success=False)
        if not projects:
            return ToolResult(message="No projects found.")

        lines = [f"Found {len(projects)} project(s):\n"]
        for proj in projects:
            status = f" ({proj.status})" if proj.status else ""
            lines.append(f"- **{proj.name}**{status}")
            if proj.owner_name:
                lines.append(f"  Owner: {proj.owner_name}")
            if proj.start_date or proj.end_date:
                dates = f"{proj.start_date or '?'} to {proj.end_date or '?'}"
                lines.append(f"  Dates: {dates}")
        return ToolResult(message="\n".join(lines))


class GetProjectDetailsTool(Tool):
    """Get detailed information about a project."""

    name = "get_project_details"
    description = (
        "Get detailed information about a specific project including its "
        "description, dates, owner, and status. Use this to understand a "
        "project before adding tasks or making changes."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "project_name": {
                "type": "string",
                "description": "Project name to look up",
            },
        },
        "required": ["project_name"],
    }
    args_model = GetProjectArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Getting project details"

    def __init__(self, projects_client: Any) -> None:
        self._client = projects_client

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """Get project details."""
        args = GetProjectArgs(**kwargs)

        try:
            project = await self._client.get_project_by_name(args.project_name)
        except (httpx.HTTPError, RuntimeError) as e:
            logger.exception("Fetching project details failed")
            return ToolResult(message=PROJECT_DETAILS_FAILED.format(e=e), success=False)
        if not project:
            return ToolResult(message=f"Project not found: {args.project_name}")

        lines = [f"**{project.name}**\n"]
        if project.status:
            lines.append(f"Status: {project.status}")
        if project.description:
            lines.append(f"Description: {project.description}")
        if project.owner_name:
            lines.append(f"Owner: {project.owner_name}")
        if project.start_date:
            lines.append(f"Start Date: {project.start_date}")
        if project.end_date:
            lines.append(f"End Date: {project.end_date}")

        return ToolResult(message="\n".join(lines))


class CreateProjectTool(ResolvingMutationTool[CreateProjectArgs]):
    """Create a new project."""

    name = "create_project"
    mutates = True
    description = (
        "Create a new project in the Zoho Projects portal. Specify the project "
        "name and optionally a description and dates. Use this when starting "
        "new initiatives or organizing work into projects.\n"
        "\n"
        "Pass `plan=true` to see what this call would do without doing it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Project name",
            },
            "description": {
                "type": "string",
                "description": "Optional project description",
            },
            "start_date": {
                "type": "string",
                "description": "Optional start date (YYYY-MM-DD)",
            },
            "end_date": {
                "type": "string",
                "description": "Optional end date (YYYY-MM-DD)",
            },
        },
        "required": ["name"],
    }
    args_model = CreateProjectArgs
    failure_message = CREATE_PROJECT_FAILED

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating project"

    def __init__(self, projects_client: Any) -> None:
        self._client = projects_client

    async def resolve(self, args: CreateProjectArgs) -> CreateProjectArgs:
        """Nothing to look up — the whole call is the args themselves."""
        return args

    def describe(self, resolved: CreateProjectArgs) -> ToolResult:
        lines = [f"create project '{resolved.name}'."]
        if resolved.description:
            lines.append(f"Description: {resolved.description}")
        if resolved.start_date or resolved.end_date:
            lines.append(f"Dates: {resolved.start_date or '?'} to {resolved.end_date or '?'}")
        return ToolResult(message="\n".join(lines))

    async def apply(self, resolved: CreateProjectArgs) -> ToolResult:
        """The one mutating call — create the project."""
        project = await self._client.create_project(
            name=resolved.name,
            description=resolved.description,
            start_date=resolved.start_date,
            end_date=resolved.end_date,
        )
        if project:
            return ToolResult(
                message=f"Project '{project.name}' created successfully (ID: {project.id})",
                mutated=True,
            )
        return ToolResult(message=f"Failed to create project: {resolved.name}", success=False)


class ListTaskListsTool(Tool):
    """List task lists (milestones) in a project."""

    name = "list_task_lists"
    description = (
        "List all task lists (milestones) in a project. Task lists help organize "
        "tasks into logical groups. Use this to see what task lists exist before "
        "creating tasks."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "project_name": {
                "type": "string",
                "description": "Project name to list task lists from",
            },
        },
        "required": ["project_name"],
    }
    args_model = GetProjectArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing task lists"

    def __init__(self, projects_client: Any) -> None:
        self._client = projects_client

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """List task lists in a project."""
        args = GetProjectArgs(**kwargs)

        try:
            project = await self._client.get_project_by_name(args.project_name)
            if not project:
                return ToolResult(message=f"Project not found: {args.project_name}")

            tasklists = await self._client.get_task_lists(project.id)
        except (httpx.HTTPError, RuntimeError) as e:
            logger.exception("Listing task lists failed")
            return ToolResult(message=LIST_TASK_LISTS_FAILED.format(e=e), success=False)
        if not tasklists:
            return ToolResult(message=f"No task lists found in project '{project.name}'")

        lines = [f"Task lists in '{project.name}':\n"]
        for tl in tasklists:
            status = f" ({tl.status})" if tl.status else ""
            lines.append(f"- **{tl.name}**{status}")
        return ToolResult(message="\n".join(lines))


@dataclass
class _ResolvedCreateTaskList:
    """What ``CreateTaskListTool.resolve`` works out before the mutating
    call — the project it belongs to."""

    args: CreateTaskListArgs
    project: Any


class CreateTaskListTool(ResolvingMutationTool[CreateTaskListArgs]):
    """Create a new task list (milestone) in a project."""

    name = "create_task_list"
    mutates = True
    description = (
        "Create a new task list (milestone) in a project. Task lists help "
        "organize tasks into logical groups like phases or categories. "
        "Use this to structure work within a project.\n"
        "\n"
        "Pass `plan=true` to see what this call would do without doing it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "project_name": {
                "type": "string",
                "description": "Project name to add task list to",
            },
            "name": {
                "type": "string",
                "description": "Task list name",
            },
        },
        "required": ["project_name", "name"],
    }
    args_model = CreateTaskListArgs
    failure_message = CREATE_TASK_LIST_FAILED

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating task list"

    def __init__(self, projects_client: Any) -> None:
        self._client = projects_client

    async def resolve(self, args: CreateTaskListArgs) -> _ResolvedCreateTaskList | ToolResult:
        """Resolve the target project — read-only, safe under ``plan``."""
        project = await self._client.get_project_by_name(args.project_name)
        if not project:
            return ToolResult(
                message=f"Project not found: {args.project_name}. "
                "Check the project name and try again.",
                success=False,
            )
        return _ResolvedCreateTaskList(args=args, project=project)

    def describe(self, resolved: _ResolvedCreateTaskList) -> ToolResult:
        return ToolResult(
            message=(
                f"create task list '{resolved.args.name}' in project '{resolved.project.name}'."
            )
        )

    async def apply(self, resolved: _ResolvedCreateTaskList) -> ToolResult:
        """The one mutating call — create the task list."""
        tasklist = await self._client.create_task_list(resolved.project.id, resolved.args.name)
        if tasklist:
            return ToolResult(
                message=(
                    f"Task list '{tasklist.name}' created in project '{resolved.project.name}' "
                    f"(ID: {tasklist.id})"
                ),
                mutated=True,
            )
        return ToolResult(
            message=f"Failed to create task list: {resolved.args.name}", success=False
        )


class ListTasksTool(Tool):
    """List tasks in a project."""

    name = "list_tasks"
    description = (
        "List all tasks in a project. Returns task names, statuses, priorities, "
        "and assignees. Use this to see what work is planned or in progress."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "project_name": {
                "type": "string",
                "description": "Project name to list tasks from",
            },
        },
        "required": ["project_name"],
    }
    args_model = ListTasksArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing tasks"

    def __init__(self, projects_client: Any) -> None:
        self._client = projects_client

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """List tasks in a project."""
        args = ListTasksArgs(**kwargs)

        try:
            project = await self._client.get_project_by_name(args.project_name)
            if not project:
                return ToolResult(message=f"Project not found: {args.project_name}")

            tasks = await self._client.get_tasks(project.id)
        except (httpx.HTTPError, RuntimeError) as e:
            logger.exception("Listing tasks failed")
            return ToolResult(message=LIST_TASKS_FAILED.format(e=e), success=False)
        if not tasks:
            return ToolResult(message=f"No tasks found in project '{project.name}'")

        lines = [f"Tasks in '{project.name}':\n"]
        for task in tasks:
            status = f" [{task.status}]" if task.status else ""
            priority = f" ({task.priority})" if task.priority and task.priority != "none" else ""
            lines.append(f"- **{task.name}**{status}{priority}")
            if task.tasklist_name:
                lines.append(f"  Task List: {task.tasklist_name}")
            if task.owners:
                lines.append(f"  Owners: {', '.join(task.owners)}")
            if task.completion_percentage > 0:
                lines.append(f"  Progress: {task.completion_percentage}%")
        return ToolResult(message="\n".join(lines))


@dataclass
class _ResolvedCreateTask:
    """What ``CreateTaskTool.resolve`` works out before the mutating call(s)
    — the project, the task list's NAME and whether it already exists (as
    an existing object, or ``None`` meaning "doesn't exist yet, would be
    created" — resolve never calls ``create_task_list`` itself, that's a
    mutation deferred to ``apply``)."""

    args: CreateTaskArgs
    project: Any
    tasklist_name: str
    tasklist: Any | None


class CreateTaskTool(ResolvingMutationTool[CreateTaskArgs]):
    """Create a new task in a project."""

    name = "create_task"
    mutates = True
    description = (
        "Create a new task in a project. Specify the task name, and optionally "
        "a description, dates, priority, and assignees. Tasks must belong to a "
        "task list - if not specified, a 'General' task list will be created.\n"
        "\n"
        "Pass `plan=true` to see what this call would do without doing it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "project_name": {
                "type": "string",
                "description": "Project name to add task to",
            },
            "name": {
                "type": "string",
                "description": "Task name",
            },
            "tasklist_name": {
                "type": "string",
                "description": "Task list name (creates 'General' if not specified)",
            },
            "description": {
                "type": "string",
                "description": "Optional task description",
            },
            "start_date": {
                "type": "string",
                "description": "Optional start date (YYYY-MM-DD)",
            },
            "end_date": {
                "type": "string",
                "description": "Optional end date (YYYY-MM-DD)",
            },
            "priority": {
                "type": "string",
                "description": "Priority: none, low, medium, high (default: none)",
            },
            "assignees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of assignee ZPUIDs",
            },
        },
        "required": ["project_name", "name"],
    }
    args_model = CreateTaskArgs
    failure_message = CREATE_TASK_FAILED

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating task"

    def __init__(self, projects_client: Any) -> None:
        self._client = projects_client

    async def resolve(self, args: CreateTaskArgs) -> _ResolvedCreateTask | ToolResult:
        """Resolve the project and look up the task list by name — never
        create it here, even when it's missing, so a plan never calls a
        mutating endpoint."""
        project = await self._client.get_project_by_name(args.project_name)
        if not project:
            return ToolResult(
                message=f"Project not found: {args.project_name}. "
                "Check the project name and try again.",
                success=False,
            )

        tasklist_name = args.tasklist_name or PennyConstants.ZOHO_PROJECTS_DEFAULT_TASKLIST
        tasklists = await self._client.get_task_lists(project.id)
        tasklist = next(
            (tl for tl in tasklists if tl.name.lower() == tasklist_name.lower()),
            None,
        )
        return _ResolvedCreateTask(
            args=args, project=project, tasklist_name=tasklist_name, tasklist=tasklist
        )

    def describe(self, resolved: _ResolvedCreateTask) -> ToolResult:
        """The plan headline — names whether the task list already exists."""
        tasklist_note = (
            f"existing task list '{resolved.tasklist_name}'"
            if resolved.tasklist
            else f"task list '{resolved.tasklist_name}' (doesn't exist yet, would be created)"
        )
        lines = [
            f"create task '{resolved.args.name}' in project '{resolved.project.name}' "
            f"({tasklist_note})."
        ]
        if resolved.args.priority != "none":
            lines.append(f"Priority: {resolved.args.priority}")
        if resolved.args.start_date or resolved.args.end_date:
            lines.append(
                f"Dates: {resolved.args.start_date or '?'} to {resolved.args.end_date or '?'}"
            )
        return ToolResult(message="\n".join(lines))

    async def apply(self, resolved: _ResolvedCreateTask) -> ToolResult:
        """Create the task list if it was missing, then the task."""
        args = resolved.args
        tasklist = resolved.tasklist
        if not tasklist:
            tasklist = await self._client.create_task_list(
                resolved.project.id, resolved.tasklist_name
            )
            if not tasklist:
                return ToolResult(
                    message=f"Failed to create task list: {resolved.tasklist_name}",
                    success=False,
                )

        task = await self._client.create_task(
            project_id=resolved.project.id,
            name=args.name,
            tasklist_id=tasklist.id,
            description=args.description,
            start_date=args.start_date,
            end_date=args.end_date,
            priority=args.priority,
            owner_zpuids=args.assignees,
        )
        if task:
            return ToolResult(
                message=(
                    f"Task '{task.name}' created in project '{resolved.project.name}' "
                    f"(Task List: {tasklist.name})"
                ),
                mutated=True,
            )
        return ToolResult(message=f"Failed to create task: {args.name}", success=False)


@dataclass
class _ResolvedUpdateTask:
    """What ``UpdateTaskTool.resolve`` works out before the mutating call —
    the project and the matched task, read-only."""

    args: UpdateTaskArgs
    project: Any
    task: Any
    ambiguity_note: str


class UpdateTaskTool(ResolvingMutationTool[UpdateTaskArgs]):
    """Update an existing task."""

    name = "update_task"
    mutates = True
    description = (
        "Update an existing task in a project. Can change the name, description, "
        "priority, status, or completion percentage. Use this to track progress "
        "or reprioritize work.\n"
        "\n"
        "Pass `plan=true` to see what this call would do without doing it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "project_name": {
                "type": "string",
                "description": "Project name containing the task",
            },
            "task_name": {
                "type": "string",
                "description": "Current task name to update",
            },
            "new_name": {
                "type": "string",
                "description": "New task name (optional)",
            },
            "description": {
                "type": "string",
                "description": "New description (optional)",
            },
            "priority": {
                "type": "string",
                "description": "New priority: none, low, medium, high (optional)",
            },
            "status_id": {
                "type": "string",
                "description": "New status ID (optional)",
            },
            "completion_percentage": {
                "type": "integer",
                "description": "Completion percentage 0-100 (optional)",
            },
        },
        "required": ["project_name", "task_name"],
    }
    args_model = UpdateTaskArgs
    failure_message = UPDATE_TASK_FAILED

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Updating task"

    def __init__(self, projects_client: Any) -> None:
        self._client = projects_client

    async def resolve(self, args: UpdateTaskArgs) -> _ResolvedUpdateTask | ToolResult:
        """Resolve the project and find the task to update — read-only."""
        project = await self._client.get_project_by_name(args.project_name)
        if not project:
            return ToolResult(
                message=f"Project not found: {args.project_name}. "
                "Check the project name and try again.",
                success=False,
            )

        tasks = await self._client.get_tasks(project.id)
        matching_tasks = [t for t in tasks if t.name.lower() == args.task_name.lower()]
        if not matching_tasks:
            return ToolResult(
                message=f"Task not found: {args.task_name}. Check the task name and try again.",
                success=False,
            )
        task = matching_tasks[0]
        ambiguity_note = self._ambiguous_match_note(matching_tasks)
        return _ResolvedUpdateTask(
            args=args, project=project, task=task, ambiguity_note=ambiguity_note
        )

    def describe(self, resolved: _ResolvedUpdateTask) -> ToolResult:
        """The plan headline — the changed fields, named from args."""
        changes = []
        if resolved.args.new_name is not None:
            changes.append(f"name -> '{resolved.args.new_name}'")
        if resolved.args.priority is not None:
            changes.append(f"priority -> {resolved.args.priority}")
        if resolved.args.completion_percentage is not None:
            changes.append(f"progress -> {resolved.args.completion_percentage}%")
        if resolved.args.description is not None:
            changes.append(f"description -> {resolved.args.description}")
        change_str = ", ".join(changes) if changes else "no changes"
        return ToolResult(
            message=(
                f"update task '{resolved.task.name}' in project '{resolved.project.name}': "
                f"{change_str}{resolved.ambiguity_note}"
            )
        )

    async def apply(self, resolved: _ResolvedUpdateTask) -> ToolResult:
        """The one mutating call — update the task."""
        updated = await self._client.update_task(
            project_id=resolved.project.id,
            task_id=resolved.task.id,
            name=resolved.args.new_name,
            description=resolved.args.description,
            priority=resolved.args.priority,
            status_id=resolved.args.status_id,
            completion_percentage=resolved.args.completion_percentage,
        )
        return self._build_update_result(resolved.args, updated, resolved.ambiguity_note)

    @staticmethod
    def _ambiguous_match_note(matching_tasks: list[Any]) -> str:
        """``""`` for an unambiguous match (the common case, byte-identical to
        before); otherwise names how many tasks share the name, since a silent
        first-match pick (#2768) would read as unambiguous."""
        if len(matching_tasks) <= 1:
            return ""
        return (
            f"\n\n(Heads up: {len(matching_tasks)} tasks named '{matching_tasks[0].name}' "
            "in this project — updated the first one; the others weren't touched.)"
        )

    @staticmethod
    def _build_update_result(args: UpdateTaskArgs, updated: Any, ambiguity_note: str) -> ToolResult:
        """Render the update from the state the backend CONFIRMS in its response
        (#2768), never the call's own args: the backend can reject a field within
        an otherwise-successful call, so echoing the request would claim a value
        that didn't land.  Mirrors ``CreateTaskTool``'s own render, which already
        reads the returned task rather than its own args."""
        if not updated:
            return ToolResult(message=f"Failed to update task: {args.task_name}", success=False)

        changes = []
        if args.new_name is not None:
            changes.append(f"name -> '{updated.name}'")
        if args.priority is not None:
            changes.append(f"priority -> {updated.priority}")
        if args.completion_percentage is not None:
            changes.append(f"progress -> {updated.completion_percentage}%")
        if args.description is not None:
            changes.append(f"description -> {updated.description}")

        change_str = ", ".join(changes) if changes else "no changes"
        return ToolResult(
            message=f"Task '{updated.name}' updated: {change_str}{ambiguity_note}",
            mutated=True,
        )


def project_tools(projects_client: Any) -> list[Tool]:
    """Return all Zoho Projects tools bound to the given client."""
    return [
        ListProjectsTool(projects_client),
        GetProjectDetailsTool(projects_client),
        CreateProjectTool(projects_client),
        ListTaskListsTool(projects_client),
        CreateTaskListTool(projects_client),
        ListTasksTool(projects_client),
        CreateTaskTool(projects_client),
        UpdateTaskTool(projects_client),
    ]
