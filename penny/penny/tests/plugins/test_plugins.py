"""Tests for the plugin system."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from penny.plugins import Plugin, load_plugins


class FakePlugin(Plugin):
    """A test plugin that is always configured."""

    name = "fake"
    capabilities = ["test"]

    @classmethod
    def is_configured(cls, config) -> bool:  # noqa: ARG003
        return True

    def get_tools(self):
        return []


class UnconfiguredPlugin(Plugin):
    """A plugin that is never configured."""

    name = "unconfigured"
    capabilities = ["test"]

    @classmethod
    def is_configured(cls, config) -> bool:  # noqa: ARG003
        return False

    def get_tools(self):
        return []


def test_load_plugins_skips_missing_module():
    config = MagicMock()
    config.plugins = ["nonexistent"]
    plugins = load_plugins(config, MagicMock())
    assert plugins == []


def test_load_plugins_skips_unconfigured():
    config = MagicMock()
    config.plugins = ["unconfigured"]
    module = MagicMock(PLUGIN_CLASS=UnconfiguredPlugin)
    with patch.dict("sys.modules", {"penny.plugins.unconfigured": module}):
        plugins = load_plugins(config, MagicMock())
    assert plugins == []


def test_load_plugins_instantiates_configured():
    config = MagicMock()
    config.plugins = ["fake"]
    db = MagicMock()
    fake_module = MagicMock(PLUGIN_CLASS=FakePlugin)
    with patch.dict("sys.modules", {"penny.plugins.fake": fake_module}):
        plugins = load_plugins(config, db)
    assert len(plugins) == 1
    assert isinstance(plugins[0], FakePlugin)
    # db is injected explicitly through construction, not read from config.runtime.
    assert plugins[0]._db is db


def test_load_plugins_missing_plugin_class():
    config = MagicMock()
    config.plugins = ["bad"]
    bad_module = MagicMock(spec=["__name__"])
    bad_module.PLUGIN_CLASS = None
    with patch.dict("sys.modules", {"penny.plugins.bad": bad_module}):
        plugins = load_plugins(config, MagicMock())
    assert plugins == []
