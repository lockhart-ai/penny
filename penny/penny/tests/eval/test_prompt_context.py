"""Prompt-context contracts — the volatile bits the base agent prepends to every
system prompt, driven against the REAL chat loop and scored on the PERSISTED
promptlog (the exact system message the model saw).

The one covered here is the ``Current date and time:`` anchor.  The profile
advertises the user's IANA timezone, so the clock the model reasons from must be
rendered in THAT zone — otherwise Penny is told a UTC time under a non-UTC
profile and, for the hours around local midnight, the wrong calendar day.  The
eval user is seeded in ``America/Los_Angeles`` (see ``seed_user``), so the anchor
must carry that zone's label (PST/PDT), never ``UTC``.

The rendering is deterministic Python, but it lives in the shipped prompt path,
so this contract runs the real chat flow and reads the anchor off the persisted
promptlog — a future refactor of the prompt scaffolding can't silently drop the
timezone conversion without tripping it.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from penny.database import Database
from penny.tests.eval.conftest import ChatEval, Check

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module.
_FAMILY = "prompt-render"

# The zone the eval user is seeded in (``seed_user``).
_PROFILE_TIMEZONE = "America/Los_Angeles"
_ANCHOR_PREFIX = "Current date and time: "


def _datetime_anchor(db: Database) -> str | None:
    """The ``Current date and time:`` line from the earliest system message the
    model was sent this run (read off the persisted promptlog), or None if no
    prompt was logged."""
    rows = db.messages.recent_prompts(limit=200)
    for row in sorted(rows, key=lambda r: r.timestamp):
        for message in json.loads(row.messages):
            content = message.get("content") or ""
            if message.get("role") == "system" and content.startswith(_ANCHOR_PREFIX):
                return content.split("\n", 1)[0]
    return None


def _score_datetime_anchor(db: Database, before: set[str], reply: str) -> list[Check]:
    # The profile zone's own label (PST/PDT), never a hardcoded UTC.
    local_abbrev = datetime.now(ZoneInfo(_PROFILE_TIMEZONE)).strftime("%Z")
    rendered = _datetime_anchor(db)
    if rendered is None:
        # No prompt logged — nothing to inspect for the two timezone checks (not-applicable).
        return [
            Check("date/time anchor logged in the system prompt", False, kind="proc"),
            Check.na(f"anchor rendered in the profile timezone ({local_abbrev})", kind="proc"),
            Check.na("anchor not rendered in UTC", kind="proc"),
        ]
    in_zone = rendered.endswith(f" {local_abbrev}")
    not_utc = "UTC" not in rendered
    return [
        Check("date/time anchor logged in the system prompt", True, kind="proc"),
        Check(
            f"anchor rendered in the profile timezone ({local_abbrev})",
            in_zone,
            rationale=None if in_zone else f"{rendered!r}",
            kind="proc",
        ),
        Check(
            "anchor not rendered in UTC",
            not_utc,
            rationale=None if not_utc else f"{rendered!r}",
            kind="proc",
        ),
    ]


async def test_datetime_anchor_in_profile_timezone(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="prompt-datetime-timezone",
        family=_FAMILY,
        message="hey! what's up?",
        score=_score_datetime_anchor,
    )
