"""Plugin system for Penny connectors.

A plugin is a self-contained connector for a third-party service (an email
provider, a calendar, a project tracker, ...) that contributes LLM-callable
tools to Penny's tool surface.  Each plugin lives in ``penny/plugins/<name>/``
and exposes a module-level ``PLUGIN_CLASS`` pointing to a :class:`Plugin`
subclass.

Enable plugins with the ``PLUGINS`` env var (a JSON array or a comma-separated
list; see ``config._parse_plugins``)::

    PLUGINS=["fastmail", "zoho"]
    PLUGINS=fastmail,zoho

At startup ``Penny._init_plugins`` calls :func:`load_plugins`, collects each
loaded plugin's ``get_tools()``, and threads them onto the *shared* base-``Agent``
tool surface (via each agent's ``plugin_tools`` argument) — so the tools reach
every agent shape, the chat agent and the background collectors alike.  A plugin
whose credentials are absent is skipped with a visible warning rather than
failing Penny's startup, so one unconfigured connector never takes the whole
agent down.
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

# Every plugin module must expose a module-level attribute of this name pointing
# to its Plugin subclass — the single entry point load_plugins looks up.
PLUGIN_CLASS_ATTR = "PLUGIN_CLASS"


class Plugin(ABC):
    """Base class for a Penny connector plugin.

    A plugin contributes LLM-callable tools to Penny's shared tool surface
    (chat + background collectors).  Subclasses declare their ``name`` and
    implement :meth:`is_configured` (are the required credentials present in the
    environment?) and :meth:`get_tools` (the tools to register).  Override
    :meth:`close` to release any long-lived resources the plugin holds (HTTP
    clients, sessions, ...).
    """

    name: str

    def __init__(self, config: Config) -> None:
        self._config = config

    @classmethod
    @abstractmethod
    def is_configured(cls, config: Config) -> bool:
        """Return True if the plugin's required credentials are present."""
        ...

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """Return the LLM-callable tools this plugin contributes."""
        ...

    async def close(self) -> None:  # noqa: B027
        """Release any resources held by the plugin. Override if needed."""


def load_plugins(config: Config) -> list[Plugin]:
    """Instantiate the plugins named in ``config.plugins``.

    Each name is loaded independently: an unknown, malformed, or unconfigured
    plugin is logged and skipped so it can't block the others or Penny's
    startup.  Returns the plugins that loaded successfully, in listed order.
    """
    plugins: list[Plugin] = []
    for name in config.plugins:
        plugin = _load_one(name, config)
        if plugin is not None:
            plugins.append(plugin)
    return plugins


def _load_one(name: str, config: Config) -> Plugin | None:
    """Import and instantiate one plugin by name; None when unavailable.

    Returns None (with a visible log) for the three "not usable here" states —
    no such module, a module missing ``PLUGIN_CLASS``, or credentials absent —
    so ``load_plugins`` skips it and keeps loading the rest.
    """
    try:
        module = importlib.import_module(f"penny.plugins.{name}")
    except ImportError:
        logger.error("Plugin '%s' not found — no module penny.plugins.%s", name, name)
        return None

    plugin_cls = getattr(module, PLUGIN_CLASS_ATTR, None)
    if plugin_cls is None:
        logger.error("Plugin '%s' module is missing %s", name, PLUGIN_CLASS_ATTR)
        return None

    if not plugin_cls.is_configured(config):
        logger.warning("Plugin '%s' is not configured (missing credentials), skipping", name)
        return None

    plugin = plugin_cls(config)
    logger.info("Loaded plugin '%s'", name)
    return plugin


__all__ = ["Plugin", "load_plugins", "PLUGIN_CLASS_ATTR"]
