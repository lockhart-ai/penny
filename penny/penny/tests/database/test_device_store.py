"""Tests for DeviceStore CRUD operations."""

from penny.constants import ChannelType
from penny.database import Database
from penny.tests.schema_template import migrated_db


class TestDeviceStore:
    """DeviceStore register, lookup, and default management."""

    def _make_db(self, tmp_path) -> Database:
        db_path = str(tmp_path / "test.db")
        db = migrated_db(db_path)
        return db

    def test_register_creates_device(self, tmp_path):
        db = self._make_db(tmp_path)
        device = db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal Phone")
        assert device.id is not None
        assert device.channel_type == ChannelType.SIGNAL
        assert device.identifier == "+15551234567"
        assert device.label == "Signal Phone"
        assert device.is_default is False

    def test_register_with_default(self, tmp_path):
        db = self._make_db(tmp_path)
        device = db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)
        assert device.is_default is True
        default = db.devices.get_default()
        assert default is not None
        assert default.id == device.id

    def test_register_default_clears_prior_default(self, tmp_path):
        db = self._make_db(tmp_path)
        signal = db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)
        ios = db.devices.register(ChannelType.IOS, "ios-keychain-id", "iPhone", is_default=True)
        assert signal.id is not None
        # Only one default row remains — the newest registration wins.
        default = db.devices.get_default()
        assert default is not None
        assert default.id == ios.id
        refreshed_signal = db.devices.get_by_id(signal.id)
        assert refreshed_signal is not None
        assert refreshed_signal.is_default is False

    def test_register_upsert_returns_existing(self, tmp_path):
        db = self._make_db(tmp_path)
        first = db.devices.register(ChannelType.SIGNAL, "+15551234567", "Original Label")
        second = db.devices.register(ChannelType.SIGNAL, "+15551234567", "New Label")
        assert first.id == second.id
        assert second.label == "Original Label"  # not overwritten

    def test_get_by_identifier(self, tmp_path):
        db = self._make_db(tmp_path)
        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal")
        found = db.devices.get_by_identifier("+15551234567")
        assert found is not None
        assert found.identifier == "+15551234567"

    def test_get_by_identifier_not_found(self, tmp_path):
        db = self._make_db(tmp_path)
        assert db.devices.get_by_identifier("nonexistent") is None

    def test_get_by_id(self, tmp_path):
        db = self._make_db(tmp_path)
        device = db.devices.register(ChannelType.BROWSER, "firefox-laptop", "Firefox Laptop")
        assert device.id is not None
        found = db.devices.get_by_id(device.id)
        assert found is not None
        assert found.identifier == "firefox-laptop"

    def test_get_all(self, tmp_path):
        db = self._make_db(tmp_path)
        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal")
        db.devices.register(ChannelType.BROWSER, "firefox-laptop", "Firefox Laptop")
        all_devices = db.devices.get_all()
        assert len(all_devices) == 2
        identifiers = {d.identifier for d in all_devices}
        assert identifiers == {"+15551234567", "firefox-laptop"}

    def test_get_default_returns_none_when_no_default(self, tmp_path):
        db = self._make_db(tmp_path)
        db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal")
        assert db.devices.get_default() is None

    def test_set_default_clears_others(self, tmp_path):
        db = self._make_db(tmp_path)
        signal = db.devices.register(ChannelType.SIGNAL, "+15551234567", "Signal", is_default=True)
        browser = db.devices.register(ChannelType.BROWSER, "firefox", "Firefox")
        assert signal.id is not None
        assert browser.id is not None

        db.devices.set_default(browser.id)

        default = db.devices.get_default()
        assert default is not None
        assert default.id == browser.id
        refreshed_signal = db.devices.get_by_id(signal.id)
        assert refreshed_signal is not None
        assert refreshed_signal.is_default is False
