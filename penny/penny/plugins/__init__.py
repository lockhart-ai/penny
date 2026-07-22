"""Plugin system for Penny integrations.

Plugins are self-contained integrations (Zoho, Fastmail, InvoiceNinja, etc.)
that register tools into Penny at startup based on the PLUGINS environment
variable. Each plugin lives in plugins/<name>/ and must expose PLUGIN_CLASS
pointing to a subclass of Plugin.

Usage in .env:
    PLUGINS=["zoho", "invoiceninja"]
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from penny.config import Config
    from penny.tools.base import Tool

logger = logging.getLogger(__name__)

CAPABILITY_EMAIL = "email"
CAPABILITY_CALENDAR = "calendar"
CAPABILITY_PROJECT = "project"
CAPABILITY_INVOICING = "invoicing"


class Plugin(ABC):
    """Base class for all Penny plugins.

    Each plugin encapsulates a third-party service integration and hooks into
    Penny by contributing LLM-callable tools. Plugins are loaded on startup;
    disabled or unconfigured plugins are skipped silently.
    """

    name: str
    capabilities: list[str]

    @classmethod
    @abstractmethod
    def is_configured(cls, config: Config) -> bool:
        """Return True if required credentials are present in the environment."""
        ...

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """Return LLM-callable tools contributed by this plugin."""
        ...

    async def close(self) -> None:  # noqa: B027
        """Clean up any open resources (HTTP clients, etc.). Override if needed."""


def load_plugins(config: Config) -> list[Plugin]:
    """Import and instantiate plugins listed in config.plugins.

    For each name in config.plugins:
    - Imports penny.plugins.<name>
    - Reads PLUGIN_CLASS from the module
    - Calls is_configured(); skips if credentials missing
    - Instantiates and returns the plugin
    """
    plugins: list[Plugin] = []
    for name in config.plugins:
        try:
            module = importlib.import_module(f"penny.plugins.{name}")
        except ImportError:
            logger.error("Plugin '%s' not found — no module penny.plugins.%s", name, name)
            continue

        plugin_cls = getattr(module, "PLUGIN_CLASS", None)
        if plugin_cls is None:
            logger.error("Plugin '%s' module missing PLUGIN_CLASS", name)
            continue

        if not plugin_cls.is_configured(config):
            logger.warning("Plugin '%s' is not configured (missing credentials), skipping", name)
            continue

        try:
            plugin = plugin_cls(config)
            plugins.append(plugin)
            logger.info(
                "Loaded plugin '%s' (capabilities: %s)",
                name,
                plugin.capabilities,
            )
        except Exception:
            logger.exception("Failed to instantiate plugin '%s'", name)
    return plugins


__all__ = [
    "CAPABILITY_EMAIL",
    "CAPABILITY_CALENDAR",
    "CAPABILITY_PROJECT",
    "CAPABILITY_INVOICING",
    "Plugin",
    "load_plugins",
]
