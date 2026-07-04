"""Startup setup-health / preflight checks.

Consolidates Penny's startup prerequisite checks into one legible surface so a
fresh deploy fails fast on a hard misconfiguration (unreachable LLM endpoint,
an unresolvable chat or embedding model) and surfaces soft degradations (a
missing vision/image model, a disconnected browser addon, a mis-routed primary
channel) as visible warnings instead of silent no-ops.

The ``Preflight`` orchestrator runs each check and returns a ``PreflightReport``;
the caller logs the report and raises ``PreflightError`` on any hard failure so
startup aborts with an actionable message instead of letting every downstream
call fail opaquely (a wrong model id 404ing on every request, memory silently
running embedding-blind, notifications routed to the wrong device).
"""

from __future__ import annotations

import logging
from enum import StrEnum

from pydantic import BaseModel

from penny.config import Config
from penny.llm.client import LlmClient
from penny.llm.image_client import OllamaImageClient
from penny.llm.models import LlmConnectionError, LlmError

logger = logging.getLogger(__name__)


class PreflightError(Exception):
    """Raised when a hard startup prerequisite fails — aborts startup."""


class CheckStatus(StrEnum):
    """Outcome of a single preflight check."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class PreflightCheck(StrEnum):
    """Stable identifier for each preflight check (log/label key)."""

    LLM_ENDPOINT = "llm-endpoint"
    EMBEDDING_MODEL = "embedding-model"
    VISION_MODEL = "vision-model"
    IMAGE_MODEL = "image-model"
    BROWSER_ADDON = "browser-addon"
    PRIMARY_CHANNEL = "primary-channel"


_STATUS_ICON: dict[CheckStatus, str] = {
    CheckStatus.OK: "✓",
    CheckStatus.WARN: "⚠",
    CheckStatus.FAIL: "✗",
}

_STATUS_LEVEL: dict[CheckStatus, int] = {
    CheckStatus.OK: logging.INFO,
    CheckStatus.WARN: logging.WARNING,
    CheckStatus.FAIL: logging.ERROR,
}


def model_available(model: str, available: list[str]) -> bool:
    """Whether a model id resolves against an endpoint's model list.

    Tag-tolerant: some backends report a model without its ``:tag`` suffix, so
    ``gpt-oss:20b`` matches a listed ``gpt-oss`` and vice versa.
    """
    base = model.split(":")[0]
    return any(listed == model or listed.split(":")[0] == base for listed in available)


class CheckResult(BaseModel):
    """The outcome of one preflight check."""

    name: PreflightCheck
    status: CheckStatus
    detail: str

    def render(self) -> str:
        """One-line human summary: ``<icon> <name>: <detail>``."""
        return f"{_STATUS_ICON[self.status]} {self.name}: {self.detail}"


class PreflightReport(BaseModel):
    """The collected results of a preflight run."""

    results: list[CheckResult]

    @property
    def failures(self) -> list[CheckResult]:
        """The hard-failing checks (startup must abort if any)."""
        return [result for result in self.results if result.status is CheckStatus.FAIL]

    @property
    def has_failures(self) -> bool:
        """Whether any hard prerequisite failed."""
        return bool(self.failures)

    def log(self, log: logging.Logger) -> None:
        """Emit the report as a startup log summary, one line per check."""
        log.info("Setup preflight:")
        for result in self.results:
            log.log(_STATUS_LEVEL[result.status], "  %s", result.render())

    def failure_summary(self) -> str:
        """Newline-joined render of the hard failures (for the raised error)."""
        return "\n".join(result.render() for result in self.failures)


class Preflight:
    """Runs the startup setup-health checks and returns a ``PreflightReport``.

    Channel facts (browser connectivity, resolved routing) are passed in as
    plain values snapshotted by the caller, so the orchestrator stays decoupled
    from the concrete channel wiring and independently testable.
    """

    def __init__(
        self,
        *,
        config: Config,
        model_client: LlmClient,
        embedding_client: LlmClient,
        vision_client: LlmClient | None,
        image_client: OllamaImageClient | None,
        browser_enabled: bool,
        browser_connected: bool,
        configured_channel_type: str,
        resolved_channel_type: str | None,
    ):
        self.config = config
        self.model_client = model_client
        self.embedding_client = embedding_client
        self.vision_client = vision_client
        self.image_client = image_client
        self.browser_enabled = browser_enabled
        self.browser_connected = browser_connected
        self.configured_channel_type = configured_channel_type
        self.resolved_channel_type = resolved_channel_type

    async def run(self) -> PreflightReport:
        """Run every applicable check and collect the results — summary method."""
        candidates: list[CheckResult | None] = [
            await self._check_llm_endpoint(),
            await self._check_embedding_model(),
            await self._check_vision_model(),
            await self._check_image_model(),
            self._check_browser_addon(),
            self._check_primary_channel(),
        ]
        return PreflightReport(results=[result for result in candidates if result is not None])

    # ── Hard-fail checks ─────────────────────────────────────────────────

    async def _check_llm_endpoint(self) -> CheckResult:
        """LLM endpoint reachable + configured chat model resolves (hard fail)."""
        name = PreflightCheck.LLM_ENDPOINT
        model = self.config.llm_model
        url = self.config.llm_api_url
        try:
            available = await self.model_client.list_models()
        except LlmConnectionError as error:
            return self._fail(
                name,
                f"LLM endpoint unreachable at {url}: {error}. Start your LLM server "
                f"(e.g. Ollama) and check LLM_API_URL.",
            )
        except LlmError as error:
            return self._warn(
                name, f"reached {url} but could not list models to verify {model!r}: {error}"
            )
        if model_available(model, available):
            return self._ok(name, f"chat model {model!r} available at {url}")
        return self._fail(
            name,
            f"chat model {model!r} not available at {url} — pull it (`ollama pull {model}`) "
            f"or set LLM_MODEL to an installed model.",
        )

    async def _check_embedding_model(self) -> CheckResult:
        """Embedding endpoint reachable + embedding model resolves (hard fail).

        The embedding model is a required prerequisite (memory dedup + recall
        depend on it), so an unreachable endpoint or an unresolvable model is a
        hard failure, not a degraded mode.
        """
        name = PreflightCheck.EMBEDDING_MODEL
        model = self.config.llm_embedding_model
        url = self.config.llm_embedding_api_url or self.config.llm_api_url
        try:
            available = await self.embedding_client.list_models()
        except LlmConnectionError as error:
            return self._fail(
                name,
                f"embedding endpoint unreachable at {url}: {error}. Penny's memory "
                f"(dedup + recall) requires it.",
            )
        except LlmError as error:
            return self._warn(
                name,
                f"reached {url} but could not list models to verify embedding model "
                f"{model!r}: {error}",
            )
        if model_available(model, available):
            return self._ok(name, f"embedding model {model!r} available at {url}")
        return self._fail(
            name,
            f"embedding model {model!r} not available at {url} — pull it "
            f"(`ollama pull {model}`) or fix LLM_EMBEDDING_MODEL. Memory (dedup + recall) "
            f"depends on it.",
        )

    # ── Soft-warn checks ─────────────────────────────────────────────────

    async def _check_vision_model(self) -> CheckResult | None:
        """Vision model resolves if configured (soft warn)."""
        if not self.config.llm_vision_model or self.vision_client is None:
            return None
        name = PreflightCheck.VISION_MODEL
        model = self.config.llm_vision_model
        url = self.config.llm_vision_api_url or self.config.llm_api_url
        try:
            available = await self.vision_client.list_models()
        except LlmError as error:
            return self._warn(
                name,
                f"could not verify vision model {model!r} at {url}: {error}. Image "
                f"understanding degraded until it resolves.",
            )
        if model_available(model, available):
            return self._ok(name, f"vision model {model!r} available at {url}")
        return self._warn(
            name,
            f"vision model {model!r} not available at {url} — pull it "
            f"(`ollama pull {model}`). Image understanding degraded until then.",
        )

    async def _check_image_model(self) -> CheckResult | None:
        """Image-generation model resolves if configured (soft warn)."""
        if not self.config.llm_image_model or self.image_client is None:
            return None
        name = PreflightCheck.IMAGE_MODEL
        model = self.config.llm_image_model
        url = self.config.image_api_url
        available = await self.image_client.list_models()
        if model_available(model, available):
            return self._ok(name, f"image model {model!r} available at {url}")
        return self._warn(
            name,
            f"image model {model!r} not available at {url} (or the endpoint is unreachable) "
            f"— pull it (`ollama pull {model}`). /draw disabled until then.",
        )

    def _check_browser_addon(self) -> CheckResult | None:
        """Browser addon connectivity if the browser channel is enabled (soft warn)."""
        if not self.browser_enabled:
            return None
        name = PreflightCheck.BROWSER_ADDON
        if self.browser_connected:
            return self._ok(name, "browser addon connected")
        return self._warn(
            name,
            "no browser addon connected — web search and page reading are degraded "
            "until an addon connects.",
        )

    def _check_primary_channel(self) -> CheckResult | None:
        """Configured primary channel is the one proactive sends resolve to (soft warn).

        Guards the routing-bug class where proactive messages went to the addon
        instead of the configured channel.
        """
        name = PreflightCheck.PRIMARY_CHANNEL
        configured = self.configured_channel_type
        resolved = self.resolved_channel_type
        if resolved is None:
            return self._warn(name, "no default channel resolved for proactive sends.")
        if resolved != configured:
            return self._warn(
                name,
                f"proactive sends will route to the {resolved!r} channel, but the configured "
                f"primary is {configured!r} — notifications may be misrouted.",
            )
        return self._ok(name, f"proactive sends route to the configured {configured!r} channel.")

    # ── Result builders ──────────────────────────────────────────────────

    @staticmethod
    def _ok(name: PreflightCheck, detail: str) -> CheckResult:
        return CheckResult(name=name, status=CheckStatus.OK, detail=detail)

    @staticmethod
    def _warn(name: PreflightCheck, detail: str) -> CheckResult:
        return CheckResult(name=name, status=CheckStatus.WARN, detail=detail)

    @staticmethod
    def _fail(name: PreflightCheck, detail: str) -> CheckResult:
        return CheckResult(name=name, status=CheckStatus.FAIL, detail=detail)
