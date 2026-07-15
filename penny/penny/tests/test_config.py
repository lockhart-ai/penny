"""Tests for ``Config.load()`` env-var → ``Config`` field wiring."""

import httpx
import pytest

from penny.config import Config
from penny.constants import ChannelType, PennyConstants
from penny.llm import LlmClient


class TestLlmTimeoutEnvWiring:
    """``LLM_TIMEOUT`` env var threads through ``Config`` and into ``LlmClient``."""

    def test_env_var_sets_config_llm_timeout(self, monkeypatch):
        """``LLM_TIMEOUT=120`` lands as ``Config.llm_timeout == 120.0``."""
        monkeypatch.setenv("LLM_TIMEOUT", "120")
        monkeypatch.setenv("SIGNAL_NUMBER", "+15551234567")  # satisfy channel validation
        monkeypatch.setenv("LLM_EMBEDDING_MODEL", "embeddinggemma")  # required prerequisite

        config = Config.load()

        assert config.llm_timeout == 120.0

    def test_unset_env_var_leaves_config_llm_timeout_none(self, monkeypatch):
        """``LLM_TIMEOUT`` absent → ``Config.llm_timeout is None`` (use SDK default)."""
        monkeypatch.delenv("LLM_TIMEOUT", raising=False)
        monkeypatch.setenv("SIGNAL_NUMBER", "+15551234567")
        monkeypatch.setenv("LLM_EMBEDDING_MODEL", "embeddinggemma")

        config = Config.load()

        assert config.llm_timeout is None


class TestEmbeddingModelRequired:
    """``LLM_EMBEDDING_MODEL`` is a hard prerequisite — ``Config.load`` fails fast."""

    def test_missing_embedding_model_raises_at_load(self, monkeypatch):
        """Unset ``LLM_EMBEDDING_MODEL`` → ``Config.load`` raises an actionable error."""
        monkeypatch.setenv("SIGNAL_NUMBER", "+15551234567")  # satisfy channel validation
        monkeypatch.delenv("LLM_EMBEDDING_MODEL", raising=False)
        # Config.load() calls _load_dotenv(), which re-reads the on-disk .env and would
        # repopulate LLM_EMBEDDING_MODEL (defeating the delenv above) in any checkout whose
        # .env sets it — so this test passed in CI/worktrees (no .env) but failed in the
        # primary checkout. Neutralize the on-disk read so the assertion holds everywhere.
        monkeypatch.setattr("penny.config._load_dotenv", lambda: None)

        with pytest.raises(ValueError, match="LLM_EMBEDDING_MODEL is required"):
            Config.load()

    def test_embedding_model_lands_on_config(self, monkeypatch):
        """A configured ``LLM_EMBEDDING_MODEL`` threads onto ``Config``."""
        monkeypatch.setenv("SIGNAL_NUMBER", "+15551234567")
        monkeypatch.setenv("LLM_EMBEDDING_MODEL", "embeddinggemma")

        config = Config.load()

        assert config.llm_embedding_model == "embeddinggemma"

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


class TestIosChannelDetection:
    """iOS can run as a sidecar instead of replacing Signal."""

    def test_signal_remains_primary_when_ios_enabled(self, monkeypatch):
        monkeypatch.setattr("penny.config._load_dotenv", lambda: None)
        monkeypatch.delenv("CHANNEL_TYPE", raising=False)
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DISCORD_CHANNEL_ID", raising=False)
        monkeypatch.setenv("SIGNAL_NUMBER", "+15551234567")
        monkeypatch.setenv("IOS_ENABLED", "true")
        monkeypatch.setenv("LLM_EMBEDDING_MODEL", "embeddinggemma")

        config = Config.load()

        assert config.channel_type == ChannelType.SIGNAL
        assert config.ios_enabled is True

    def test_ios_only_still_detects_ios_primary(self, monkeypatch):
        monkeypatch.setattr("penny.config._load_dotenv", lambda: None)
        monkeypatch.delenv("CHANNEL_TYPE", raising=False)
        monkeypatch.delenv("SIGNAL_NUMBER", raising=False)
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DISCORD_CHANNEL_ID", raising=False)
        monkeypatch.setenv("IOS_ENABLED", "true")
        monkeypatch.setenv("LLM_EMBEDDING_MODEL", "embeddinggemma")

        config = Config.load()

        assert config.channel_type == ChannelType.IOS
        assert config.ios_enabled is True


class TestIosApnsProductionConfig:
    """Production APNs credentials are optional overrides for production tokens."""

    def test_production_apns_credentials_land_on_config(self, monkeypatch):
        monkeypatch.setattr("penny.config._load_dotenv", lambda: None)
        monkeypatch.setenv("SIGNAL_NUMBER", "+15551234567")
        monkeypatch.setenv("IOS_ENABLED", "true")
        monkeypatch.setenv("LLM_EMBEDDING_MODEL", "embeddinggemma")
        monkeypatch.setenv("IOS_APNS_TEAM_ID", "SANDBOXTEAM")
        monkeypatch.setenv("IOS_APNS_KEY_ID", "SANDBOXKEY")
        monkeypatch.setenv(
            "IOS_APNS_KEY_PATH",
            "/penny/data/private/AuthKey_SANDBOXKEY.p8",
        )
        monkeypatch.setenv("IOS_APNS_PRODUCTION_TEAM_ID", "PRODTEAM")
        monkeypatch.setenv("IOS_APNS_PRODUCTION_KEY_ID", "PRODKEY")
        monkeypatch.setenv(
            "IOS_APNS_PRODUCTION_KEY_PATH",
            "/penny/data/private/AuthKey_PRODKEY.p8",
        )
        monkeypatch.setenv("IOS_BUNDLE_ID", "com.example.Penny")
        monkeypatch.setenv("IOS_APNS_PRODUCTION_BUNDLE_ID", "com.example.PennyTestflight")

        config = Config.load()

        assert config.ios_apns_production_team_id == "PRODTEAM"
        assert config.ios_apns_production_key_id == "PRODKEY"
        assert config.ios_apns_production_key_path == "/penny/data/private/AuthKey_PRODKEY.p8"
        assert config.ios_apns_production_bundle_id == "com.example.PennyTestflight"

    def test_partial_production_apns_credentials_raise(self, monkeypatch):
        monkeypatch.setattr("penny.config._load_dotenv", lambda: None)
        monkeypatch.setenv("SIGNAL_NUMBER", "+15551234567")
        monkeypatch.setenv("IOS_ENABLED", "true")
        monkeypatch.setenv("LLM_EMBEDDING_MODEL", "embeddinggemma")
        monkeypatch.delenv("IOS_APNS_PRODUCTION_TEAM_ID", raising=False)
        monkeypatch.delenv("IOS_APNS_PRODUCTION_KEY_PATH", raising=False)
        monkeypatch.delenv("IOS_APNS_PRODUCTION_BUNDLE_ID", raising=False)
        monkeypatch.setenv("IOS_APNS_PRODUCTION_KEY_ID", "PRODKEY")

        with pytest.raises(ValueError, match="IOS_APNS_PRODUCTION_TEAM_ID"):
            Config.load()


class TestMaxStepsEnvWiring:
    """The ``MAX_STEPS`` runtime param threads from the env-override tier (#1601).

    The env tier reads ``os.getenv(param.key)`` — the key is ``MAX_STEPS``. Docs once
    named this env var ``MESSAGE_MAX_STEPS``, which the tier never reads, so a value set
    under that name was silently ignored. ``MAX_STEPS`` is the canonical name.
    """

    def _base_env(self, monkeypatch):
        """Neutralize the on-disk .env read and satisfy channel + embedding validation."""
        monkeypatch.setattr("penny.config._load_dotenv", lambda: None)
        monkeypatch.setenv("SIGNAL_NUMBER", "+15551234567")
        monkeypatch.setenv("LLM_EMBEDDING_MODEL", "embeddinggemma")

    def test_max_steps_env_var_lands_on_runtime(self, monkeypatch):
        """``MAX_STEPS=15`` (the canonical key) flows into ``config.runtime.MAX_STEPS``."""
        self._base_env(monkeypatch)
        monkeypatch.setenv("MAX_STEPS", "15")

        config = Config.load()

        assert config.runtime.MAX_STEPS == 15

    def test_default_holds_when_nothing_overrides(self, monkeypatch):
        """Unset → the ConfigParam default (20, equal to BACKGROUND_MAX_STEPS) applies."""
        self._base_env(monkeypatch)
        monkeypatch.delenv("MAX_STEPS", raising=False)

        config = Config.load()

        assert config.runtime.MAX_STEPS == 20
