"""Zoho Projects API client."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from penny.constants import PennyConstants
from penny.plugins.zoho.models import (
    ZohoPortal,
    ZohoProject,
    ZohoSession,
    ZohoTask,
    ZohoTaskList,
)

logger = logging.getLogger(__name__)


class ZohoProjectsClient:
    """Zoho Projects API client.

    Uses OAuth 2.0 with client credentials to access Zoho Projects API.
    Defaults to the first (and typically only) portal.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        *,
        timeout: float = 30.0,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._session: ZohoSession | None = None
        self._http = httpx.AsyncClient(timeout=timeout)
        self._default_portal: ZohoPortal | None = None
        self._api_domain: str | None = None

    async def _ensure_access_token(self) -> str:
        """Ensure we have a valid access token, refreshing if needed."""
        now = time.time()
        if self._session and self._session.expires_at > now + 60:
            return self._session.access_token

        resp = await self._http.post(
            PennyConstants.ZOHO_TOKEN_URL,
            data={
                "refresh_token": self._refresh_token,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"Zoho OAuth error: {data.get('error')}")

        expires_in = data.get("expires_in", 3600)
        self._session = ZohoSession(
            access_token=data["access_token"],
            expires_at=now + expires_in,
        )

        api_domain = data.get("api_domain", "https://www.zohoapis.com")
        self._api_domain = api_domain.rstrip("/")

        logger.info("Zoho Projects access token refreshed, expires in %ds", expires_in)
        return self._session.access_token

    async def _get_headers(self) -> dict[str, str]:
        """Get headers with current access token."""
        token = await self._ensure_access_token()
        return {
            "Authorization": f"Zoho-oauthtoken {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def get_portals(self) -> list[ZohoPortal]:
        """Fetch all portals for the user."""
        headers = await self._get_headers()
        url = f"{PennyConstants.ZOHO_PROJECTS_API_BASE}/portals"

        logger.debug("Fetching portals from: %s", url)
        resp = await self._http.get(url, headers=headers)
        if resp.status_code != 200:
            logger.error(
                "Zoho Projects API error: status=%s, url=%s, response=%s",
                resp.status_code,
                url,
                resp.text,
            )
        resp.raise_for_status()
        data = resp.json()

        portals_data = data if isinstance(data, list) else data.get("portals", [])
        portals = [
            ZohoPortal(
                id=str(p.get("id", "")),
                name=p.get("name", ""),
                is_default=p.get("default", False) or idx == 0,
            )
            for idx, p in enumerate(portals_data)
        ]
        logger.info("Loaded %d portals", len(portals))
        return portals

    async def get_default_portal(self) -> ZohoPortal | None:
        """Get the default (first) portal."""
        if self._default_portal:
            return self._default_portal

        portals = await self.get_portals()
        if portals:
            self._default_portal = portals[0]
            return self._default_portal
        return None

    async def get_projects(self, portal_id: str | None = None) -> list[ZohoProject]:
        """Fetch all projects from a portal."""
        if not portal_id:
            portal = await self.get_default_portal()
            if not portal:
                logger.error("No portal available")
                return []
            portal_id = portal.id

        headers = await self._get_headers()
        url = f"{PennyConstants.ZOHO_PROJECTS_API_BASE}/portal/{portal_id}/projects"

        resp = await self._http.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        projects_data = data if isinstance(data, list) else data.get("projects", [])
        projects = [
            ZohoProject(
                id=str(p.get("id", "")),
                name=p.get("name", ""),
                status=self._extract_status_name(p.get("status")),
                description=p.get("description"),
                start_date=p.get("start_date"),
                end_date=p.get("end_date"),
                owner_name=(p.get("owner") or {}).get("name"),
            )
            for p in projects_data
        ]
        logger.info("Loaded %d projects from portal %s", len(projects), portal_id)
        return projects

    async def create_project(
        self,
        name: str,
        *,
        portal_id: str | None = None,
        description: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> ZohoProject | None:
        """Create a new project."""
        if not portal_id:
            portal = await self.get_default_portal()
            if not portal:
                logger.error("No portal available for project creation")
                return None
            portal_id = portal.id

        headers = await self._get_headers()
        url = f"{PennyConstants.ZOHO_PROJECTS_API_BASE}/portal/{portal_id}/projects"

        payload: dict[str, Any] = {"name": name}
        if description:
            payload["description"] = description
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date

        logger.info("Creating project '%s' in portal %s", name, portal_id)
        resp = await self._http.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        p = data.get("projects", [{}])[0] if data.get("projects") else {}
        if p.get("id"):
            return ZohoProject(
                id=str(p["id"]),
                name=p.get("name", name),
                status=self._extract_status_name(p.get("status")),
                description=p.get("description"),
                start_date=p.get("start_date"),
                end_date=p.get("end_date"),
            )

        logger.warning("Project creation returned no id: %s", data)
        return None

    async def get_task_lists(
        self, project_id: str, portal_id: str | None = None
    ) -> list[ZohoTaskList]:
        """Fetch task lists (milestones) for a project."""
        if not portal_id:
            portal = await self.get_default_portal()
            if not portal:
                return []
            portal_id = portal.id

        headers = await self._get_headers()
        url = (
            f"{PennyConstants.ZOHO_PROJECTS_API_BASE}/portal/{portal_id}"
            f"/projects/{project_id}/tasklists"
        )

        resp = await self._http.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        tasklists_data = data if isinstance(data, list) else data.get("tasklists", [])
        tasklists = [
            ZohoTaskList(
                id=str(tl.get("id", "")),
                name=tl.get("name", ""),
                status=tl.get("status"),
                flag=tl.get("flag"),
            )
            for tl in tasklists_data
        ]
        logger.info("Loaded %d task lists from project %s", len(tasklists), project_id)
        return tasklists

    async def create_task_list(
        self,
        project_id: str,
        name: str,
        *,
        portal_id: str | None = None,
        flag: str = "internal",
    ) -> ZohoTaskList | None:
        """Create a new task list (milestone) in a project."""
        if not portal_id:
            portal = await self.get_default_portal()
            if not portal:
                return None
            portal_id = portal.id

        headers = await self._get_headers()
        url = (
            f"{PennyConstants.ZOHO_PROJECTS_API_BASE}/portal/{portal_id}"
            f"/projects/{project_id}/tasklists"
        )

        payload = {"name": name, "flag": flag}

        logger.info("Creating task list '%s' in project %s", name, project_id)
        resp = await self._http.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        tl = data.get("tasklists", [{}])[0] if data.get("tasklists") else {}
        if tl.get("id"):
            return ZohoTaskList(
                id=str(tl["id"]),
                name=tl.get("name", name),
                status=tl.get("status"),
                flag=tl.get("flag", flag),
            )

        logger.warning("Task list creation returned no id: %s", data)
        return None

    async def get_tasks(self, project_id: str, portal_id: str | None = None) -> list[ZohoTask]:
        """Fetch all tasks for a project."""
        if not portal_id:
            portal = await self.get_default_portal()
            if not portal:
                return []
            portal_id = portal.id

        headers = await self._get_headers()
        url = (
            f"{PennyConstants.ZOHO_PROJECTS_API_BASE}/portal/{portal_id}"
            f"/projects/{project_id}/tasks"
        )

        resp = await self._http.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        tasks_data = data if isinstance(data, list) else data.get("tasks", [])
        tasks = [
            ZohoTask(
                id=str(t.get("id", "")),
                name=t.get("name", ""),
                status=self._extract_status(t),
                priority=t.get("priority"),
                description=t.get("description"),
                start_date=t.get("start_date"),
                end_date=t.get("end_date"),
                completion_percentage=t.get("percent_complete", 0),
                tasklist_id=str(t.get("tasklist", {}).get("id", "")) if t.get("tasklist") else None,
                tasklist_name=t.get("tasklist", {}).get("name") if t.get("tasklist") else None,
                owners=[o.get("name", "") for o in t.get("owners", [])],
            )
            for t in tasks_data
        ]
        logger.info("Loaded %d tasks from project %s", len(tasks), project_id)
        return tasks

    def _extract_status_name(self, status: Any) -> str | None:
        """Extract status name from status field (may be dict or string in v3 API)."""
        if status is None:
            return None
        if isinstance(status, dict):
            return status.get("name") or status.get("id")
        return str(status)

    def _extract_status(self, task_data: dict[str, Any]) -> str | None:
        """Extract status name from task data."""
        return self._extract_status_name(task_data.get("status"))

    async def create_task(
        self,
        project_id: str,
        name: str,
        tasklist_id: str,
        *,
        portal_id: str | None = None,
        description: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        priority: str = "none",
        owner_zpuids: list[str] | None = None,
    ) -> ZohoTask | None:
        """Create a new task in a project."""
        if not portal_id:
            portal = await self.get_default_portal()
            if not portal:
                return None
            portal_id = portal.id

        headers = await self._get_headers()
        url = (
            f"{PennyConstants.ZOHO_PROJECTS_API_BASE}/portal/{portal_id}"
            f"/projects/{project_id}/tasks"
        )

        payload: dict[str, Any] = {
            "name": name,
            "tasklist": {"id": tasklist_id},
            "priority": priority,
        }
        if description:
            payload["description"] = description
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date
        if owner_zpuids:
            payload["owners_and_work"] = {"owners": [{"zpuid": zpuid} for zpuid in owner_zpuids]}

        logger.info("Creating task '%s' in project %s", name, project_id)
        resp = await self._http.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        t = data.get("tasks", [{}])[0] if data.get("tasks") else {}
        if t.get("id"):
            return ZohoTask(
                id=str(t["id"]),
                name=t.get("name", name),
                status=self._extract_status(t),
                priority=t.get("priority", priority),
                description=t.get("description"),
                start_date=t.get("start_date"),
                end_date=t.get("end_date"),
                tasklist_id=tasklist_id,
            )

        logger.warning("Task creation returned no id: %s", data)
        return None

    async def update_task(
        self,
        project_id: str,
        task_id: str,
        *,
        portal_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        status_id: str | None = None,
        priority: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        completion_percentage: int | None = None,
        owner_zpuids: list[str] | None = None,
    ) -> ZohoTask | None:
        """Update an existing task."""
        if not portal_id:
            portal = await self.get_default_portal()
            if not portal:
                return None
            portal_id = portal.id

        headers = await self._get_headers()
        url = (
            f"{PennyConstants.ZOHO_PROJECTS_API_BASE}/portal/{portal_id}"
            f"/projects/{project_id}/tasks/{task_id}"
        )

        payload: dict[str, Any] = {}
        if name:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if status_id:
            payload["status"] = {"id": status_id}
        if priority:
            payload["priority"] = priority
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date
        if completion_percentage is not None:
            payload["completion_percentage"] = completion_percentage
        if owner_zpuids is not None:
            payload["owners_and_work"] = {"owners": [{"add": [{"zpuid": z} for z in owner_zpuids]}]}

        if not payload:
            logger.warning("No fields to update for task %s", task_id)
            return None

        logger.info("Updating task %s in project %s", task_id, project_id)
        resp = await self._http.patch(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        t = data.get("tasks", [{}])[0] if data.get("tasks") else {}
        if t.get("id"):
            return ZohoTask(
                id=str(t["id"]),
                name=t.get("name", ""),
                status=self._extract_status(t),
                priority=t.get("priority"),
                description=t.get("description"),
                start_date=t.get("start_date"),
                end_date=t.get("end_date"),
                completion_percentage=t.get("percent_complete", 0),
            )

        logger.warning("Task update returned no id: %s", data)
        return None

    async def get_project_by_name(
        self, name: str, portal_id: str | None = None
    ) -> ZohoProject | None:
        """Find a project by name (case-insensitive)."""
        projects = await self.get_projects(portal_id)
        name_lower = name.lower().strip()

        for proj in projects:
            if proj.name.lower() == name_lower:
                return proj

        for proj in projects:
            if name_lower in proj.name.lower():
                return proj

        return None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
