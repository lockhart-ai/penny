"""Tests for the Zoho Projects API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from penny.plugins.zoho.projects_client import ZohoProjectsClient


@pytest.mark.asyncio
async def test_get_projects_handles_null_owner():
    """A project whose ``owner`` relationship is JSON null parses without an
    ``AttributeError`` (``owner_name`` is simply ``None``)."""
    request = httpx.Request("GET", "https://projectsapi.zoho.com/api/v3/portal/p1/projects")
    with patch("penny.plugins.zoho.projects_client.httpx.AsyncClient") as mock_client_cls:
        client = ZohoProjectsClient(client_id="id", client_secret="secret", refresh_token="token")
        client._get_headers = AsyncMock(return_value={})
        client.get_default_portal = AsyncMock(return_value=MagicMock(id="p1"))
        mock_http = mock_client_cls.return_value
        mock_http.get = AsyncMock(
            return_value=httpx.Response(
                200,
                json={"projects": [{"id": "1", "name": "Website", "owner": None}]},
                request=request,
            )
        )

        projects = await client.get_projects()

    assert len(projects) == 1
    assert projects[0].name == "Website"
    assert projects[0].owner_name is None
