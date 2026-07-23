"""Fastmail JMAP client for email search."""

from penny.plugins.fastmail.client import JmapClient
from penny.plugins.fastmail.models import JmapSession

__all__ = [
    "JmapClient",
    "JmapSession",
]
