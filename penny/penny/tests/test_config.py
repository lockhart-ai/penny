"""Tests for ``Config.load()`` env-var → ``Config`` field wiring."""

import httpx

from penny.config import Config
from penny.constants import PennyConstants
from penny.llm import LlmClient


class TestLlmTimeoutEnvWiring:
    """``LLM_TIMEOUT`` env var threads through ``Config`` and into ``LlmClient``."""

    def test_env_var_sets_config_llm_timeout(self, monkeypatch):
        """``LLM_TIMEOUT=120`` lands as ``Config.llm_timeout == 120.0``."""
        monkeypatch.setenv("LLM_TIMEOUT", "120")
        monkeypatch.setenv("SIGNAL_NUMBER", "+15551234567")  # satisfy channel validation

        config = Config.load()

        assert config.llm_timeout == 120.0

    def test_unset_env_var_leaves_config_llm_timeout_none(self, monkeypatch):
        """``LLM_TIMEOUT`` absent → ``Config.llm_timeout is None`` (use SDK default)."""
        monkeypatch.delenv("LLM_TIMEOUT", raising=False)
        monkeypatch.setenv("SIGNAL_NUMBER", "+15551234567")

        config = Config.load()

        assert config.llm_timeout is None

    def test_client_timeout_overrides_only_read_write(self):
        """Constructing ``LlmClient(timeout=120)`` configures httpx read/write
        to 120s while keeping the connect timeout at
        ``PennyConstants.LLM_CONNECT_TIMEOUT_SECONDS``.
        """
        client = LlmClient(
            api_url="http://localhost:11434",
            model="m",
            max_retries=1,
            retry_delay=0.0,
            timeout=120.0,
        )

        timeout_obj = client.client.timeout
        assert isinstance(timeout_obj, httpx.Timeout)
        assert timeout_obj.read == 120.0
        assert timeout_obj.write == 120.0
        assert timeout_obj.connect == PennyConstants.LLM_CONNECT_TIMEOUT_SECONDS

    def test_client_without_timeout_does_not_set_explicit_httpx_timeout(self):
        """When ``timeout`` is omitted, ``LlmClient`` does not pass an explicit
        ``timeout`` to the OpenAI SDK — the SDK's own default applies."""
        client = LlmClient(
            api_url="http://localhost:11434",
            model="m",
            max_retries=1,
            retry_delay=0.0,
            timeout=None,
        )

        # The httpx Timeout object the SDK ends up with is its default —
        # the read deadline is 600s, not our caller-supplied number.
        timeout_obj = client.client.timeout
        assert timeout_obj.read == 600.0
