"""Zoho Projects tools — LLM-callable tools for project management."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from penny.tools.base import Tool
from penny.tools.models import ToolArgs, ToolResult

logger = logging.getLogger(__name__)


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


class CreateTaskListArgs(ToolArgs):
    """Arguments for creating a task list."""

    project_name: str = Field(description="Project name to add task list to")
    name: str = Field(description="Task list name")


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

    async def execute(self, **kwargs: Any) -> ToolResult:
        """List all projects."""
        projects = await self._client.get_projects()
        if not projects:
            return ToolResult(message="No projects found.")

        lines = [f"Found {len(projects)} project(s):\n"]
        for proj in projects:
            status = f" ({proj.status})" if proj.status else ""
            lines.append(f"- **{proj.name}**{status}")
            if proj.owner_name:
                lines.append(f"  Owner: {proj.owner_name}")
            if proj.start_date or proj.end_date:
                dates = f"{proj.start_date or '?'} to {proj.end_date or '?' }"
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

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Get project details."""
        args = GetProjectArgs(**kwargs)

        project = await self._client.get_project_by_name(args.project_name)
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


class CreateProjectTool(Tool):
    """Create a new project."""

    name = "create_project"
    description = (
        "Create a new project in the Zoho Projects portal. Specify the project "
        "name and optionally a description and dates. Use this when starting "
        "new initiatives or organizing work into projects."
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

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating project"

    def __init__(self, projects_client: Any) -> None:
        self._client = projects_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Create a new project."""
        args = CreateProjectArgs(**kwargs)

        project = await self._client.create_project(
            name=args.name,
            description=args.description,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        if project:
            return ToolResult(
                message=f"Project '{project.name}' created successfully (ID: {project.id})",
                mutated=True,
            )
        return ToolResult(message=f"Failed to create project: {args.name}", success=False)


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

    async def execute(self, **kwargs: Any) -> ToolResult:
        """List task lists in a project."""
        args = GetProjectArgs(**kwargs)

        project = await self._client.get_project_by_name(args.project_name)
        if not project:
            return ToolResult(message=f"Project not found: {args.project_name}")

        tasklists = await self._client.get_task_lists(project.id)
        if not tasklists:
            return ToolResult(message=f"No task lists found in project '{project.name}'")

        lines = [f"Task lists in '{project.name}':\n"]
        for tl in tasklists:
            status = f" ({tl.status})" if tl.status else ""
            lines.append(f"- **{tl.name}**{status}")
        return ToolResult(message="\n".join(lines))


class CreateTaskListTool(Tool):
    """Create a new task list (milestone) in a project."""

    name = "create_task_list"
    description = (
        "Create a new task list (milestone) in a project. Task lists help "
        "organize tasks into logical groups like phases or categories. "
        "Use this to structure work within a project."
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

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating task list"

    def __init__(self, projects_client: Any) -> None:
        self._client = projects_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Create a task list."""
        args = CreateTaskListArgs(**kwargs)

        project = await self._client.get_project_by_name(args.project_name)
        if not project:
            return ToolResult(message=f"Project not found: {args.project_name}")

        tasklist = await self._client.create_task_list(project.id, args.name)
        if tasklist:
            return ToolResult(
                message=(
                    f"Task list '{tasklist.name}' created in project '{project.name}' "
                    f"(ID: {tasklist.id})"
                ),
                mutated=True,
            )
        return ToolResult(message=f"Failed to create task list: {args.name}", success=False)


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

    async def execute(self, **kwargs: Any) -> ToolResult:
        """List tasks in a project."""
        args = ListTasksArgs(**kwargs)

        project = await self._client.get_project_by_name(args.project_name)
        if not project:
            return ToolResult(message=f"Project not found: {args.project_name}")

        tasks = await self._client.get_tasks(project.id)
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


class CreateTaskTool(Tool):
    """Create a new task in a project."""

    name = "create_task"
    description = (
        "Create a new task in a project. Specify the task name, and optionally "
        "a description, dates, priority, and assignees. Tasks must belong to a "
        "task list - if not specified, a 'General' task list will be created."
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

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating task"

    def __init__(self, projects_client: Any) -> None:
        self._client = projects_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Create a task."""
        args = CreateTaskArgs(**kwargs)

        project = await self._client.get_project_by_name(args.project_name)
        if not project:
            return ToolResult(message=f"Project not found: {args.project_name}")

        tasklist_name = args.tasklist_name or "General"
        tasklists = await self._client.get_task_lists(project.id)
        tasklist = next(
            (tl for tl in tasklists if tl.name.lower() == tasklist_name.lower()),
            None,
        )

        if not tasklist:
            tasklist = await self._client.create_task_list(project.id, tasklist_name)
            if not tasklist:
                return ToolResult(message=f"Failed to create task list: {tasklist_name}")

        task = await self._client.create_task(
            project_id=project.id,
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
                    f"Task '{task.name}' created in project '{project.name}' "
                    f"(Task List: {tasklist.name})"
                ),
                mutated=True,
            )
        return ToolResult(message=f"Failed to create task: {args.name}", success=False)


class UpdateTaskTool(Tool):
    """Update an existing task."""

    name = "update_task"
    description = (
        "Update an existing task in a project. Can change the name, description, "
        "priority, status, or completion percentage. Use this to track progress "
        "or reprioritize work."
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

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Updating task"

    def __init__(self, projects_client: Any) -> None:
        self._client = projects_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Update a task."""
        args = UpdateTaskArgs(**kwargs)

        project = await self._client.get_project_by_name(args.project_name)
        if not project:
            return ToolResult(message=f"Project not found: {args.project_name}")

        tasks = await self._client.get_tasks(project.id)
        task = next(
            (t for t in tasks if t.name.lower() == args.task_name.lower()),
            None,
        )

        if not task:
            return ToolResult(message=f"Task not found: {args.task_name}")

        updated = await self._client.update_task(
            project_id=project.id,
            task_id=task.id,
            name=args.new_name,
            description=args.description,
            priority=args.priority,
            status_id=args.status_id,
            completion_percentage=args.completion_percentage,
        )

        if updated:
            changes = []
            if args.new_name:
                changes.append(f"name -> '{args.new_name}'")
            if args.priority:
                changes.append(f"priority -> {args.priority}")
            if args.completion_percentage is not None:
                changes.append(f"progress -> {args.completion_percentage}%")
            if args.description:
                changes.append("description updated")

            change_str = ", ".join(changes) if changes else "no changes"
            return ToolResult(
                message=f"Task '{task.name}' updated: {change_str}",
                mutated=True,
            )
        return ToolResult(message=f"Failed to update task: {args.task_name}", success=False)


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
