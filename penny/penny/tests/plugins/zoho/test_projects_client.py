"""Tests for the Zoho Projects API client — endpoints + the shared OAuth base."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from penny.plugins.zoho.projects_client import ZohoProjectsClient


def _response(payload: dict, status_code: int = 200) -> MagicMock:
    """A stand-in for an httpx.Response with sync json()/raise_for_status()."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = ""
    resp.json = MagicMock(return_value=payload)
    resp.raise_for_status = MagicMock()
    return resp


def _make_projects_client() -> ZohoProjectsClient:
    """A projects client whose httpx client is a harmless mock (no real socket)."""
    with patch("penny.plugins.zoho.base_client.httpx.AsyncClient"):
        return ZohoProjectsClient(client_id="id", client_secret="secret", refresh_token="refresh")


@pytest.mark.asyncio
async def test_get_headers_refreshes_token_via_shared_base():
    """Token refresh works for the Projects client through the shared OAuth base."""
    client = _make_projects_client()
    client._http = MagicMock()
    client._http.post = AsyncMock(
        return_value=_response({"access_token": "tok-p", "expires_in": 3600})
    )

    headers = await client._get_headers()

    assert headers["Authorization"] == "Zoho-oauthtoken tok-p"
    client._http.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_token_refresh_captures_api_domain():
    """The Projects override captures (and trims) the token response's api_domain."""
    client = _make_projects_client()
    client._http = MagicMock()
    client._http.post = AsyncMock(
        return_value=_response(
            {
                "access_token": "tok-p",
                "expires_in": 3600,
                "api_domain": "https://custom.zohoapis.com/",
            }
        )
    )

    await client._ensure_access_token()

    assert client._api_domain == "https://custom.zohoapis.com"


@pytest.mark.asyncio
async def test_token_refresh_defaults_api_domain_when_absent():
    """No api_domain in the response → the documented default."""
    client = _make_projects_client()
    client._http = MagicMock()
    client._http.post = AsyncMock(
        return_value=_response({"access_token": "tok-p", "expires_in": 3600})
    )

    await client._ensure_access_token()

    assert client._api_domain == "https://www.zohoapis.com"


@pytest.mark.asyncio
async def test_get_projects_handles_null_owner():
    """A project whose ``owner`` relationship is JSON null parses without an
    ``AttributeError`` (``owner_name`` is simply ``None``). The httpx client is
    constructed on the shared base, so that's where it is patched."""
    request = httpx.Request("GET", "https://projectsapi.zoho.com/api/v3/portal/p1/projects")
    with patch("penny.plugins.zoho.base_client.httpx.AsyncClient") as mock_client_cls:
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
    # URL is built from ZOHO_PROJECTS_PROJECTS_PATH — value unchanged by the constant swap.
    assert mock_http.get.call_args.args[0] == (
        "https://projectsapi.zoho.com/api/v3/portal/p1/projects"
    )
