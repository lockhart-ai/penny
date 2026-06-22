"""Skills-collector contracts — the background loop that grounds skills in the
real collections that exist.

The skills collector (migration 0069) no longer reads chat.  Each cycle it calls
``collection_catalog`` to see every user-built collection's gather recipe, distils
the reusable, topic-agnostic *pattern* behind each, and reconciles against the
existing skills: create a skill for a new kind of collection, fold a
*generalizable* recipe improvement into the matching skill, leave
collection-specific quirks in the collection's own prompt, and never delete.

These cases seed COLLECTIONS (not chat) and run ``collector.run_for("skills")``
over the migration-seeded skill set, diffing the skills collection before/after:

  new-pattern   — a kind of collection no skill covers      → a new skill written
  drift-update  — a recipe gained a GENERALIZABLE step       → the matching skill edited
  quirk-left    — a recipe gained a collection-specific quirk → skills untouched by it
  consolidate   — two collections of one kind                → still ONE skill, no fork
  healthy-noop  — a recipe already matches its skill          → skills unchanged
  operate-keep  — operate-pattern seeds (no source collection)→ left untouched
  quiet         — no user collections                         → no-op

``drift-update`` vs ``quirk-left`` is the experiment: can gpt-oss tell a general
improvement to a KIND of task (fold into the shared skill) from a one-off quirk of
a single collection (leave it in that collection's prompt)?  Both are report-only
(``min_pass_rate=None``) — the printed X/Y rate is the signal while we tune the
prompt.  The guard cases (healthy / operate-keep / quiet) gate, since "don't
over-correct" must hold.
"""

from __future__ import annotations

from typing import cast

import pytest

from penny.database import Database
from penny.tests.eval.conftest import collection_entries, seed_collection
from penny.tests.eval.fixtures import (
    BOARD_GAMES,
    RESEARCH_WATCHER,
    RESEARCH_WATCHER_EXTRACTION_PROMPT,
    RESEARCH_WATCHER_INTENT,
    SynthCollection,
)

pytestmark = pytest.mark.eval

_SKILLS = "skills"

# ── Case-specific collection recipes (pub/sub-correct: notify = published flag,
#    never a send_message step in a collector body) ────────────────────────────

# Healthy research+notify recipe — matches the seeded "Research — notify" skill.
_RESEARCH_HEALTHY = RESEARCH_WATCHER_EXTRACTION_PROMPT

# Drift: the same recipe gained a GENERALIZABLE method — consult curated lists,
# then cross-check a reference source before writing.  Any research collection
# would benefit, so the matching skill should fold a topic-agnostic version in.
_RESEARCH_DRIFT = (
    "Collect newly released indie metroidvania games worth playing.\n"
    "1. browse curated 'best new metroidvania' lists and aggregator roundups first.\n"
    "2. For each candidate, cross-check its details against a reference source "
    "(its store page or Wikipedia) before recording it.\n"
    "3. Each entry: key = game name; content = name + description + URL.\n"
    '4. collection_write("indie-metroidvanias", entries=[...]).\n'
    "5. done()."
)

# Quirk: the same recipe gained a one-off specific to THIS collection's formatting
# — a tag prefix.  It belongs in this collection's prompt, never the shared skill.
_RESEARCH_QUIRK = (
    "Collect newly released indie metroidvania games worth playing.\n"
    "1. browse the web for new indie metroidvania releases; read actual pages.\n"
    "2. Each entry: key = game name; content = name + description + URL.\n"
    '3. Prefix every entry\'s content with the literal tag "[MV]" so they stand out.\n'
    '4. collection_write("indie-metroidvanias", entries=[...]).\n'
    "5. done()."
)

# A second research+notify collection (different topic, same KIND) for consolidate.
_BOARDGAMES_NOTIFY = (
    "Collect heavier euro-style strategy board games worth buying.\n"
    "1. browse the web for new strategy board games; read actual pages.\n"
    "2. Each entry: key = game name; content = name + description + player count + URL.\n"
    '3. collection_write("board-games", entries=[...]).\n'
    "4. done()."
)

# A genuinely new KIND of collection — watch a page and record changes — that no
# seeded skill covers.
_PAGE_WATCH = SynthCollection(
    "release-watch",
    "Watches a software release page and records each new version as it appears.",
    inclusion="relevant",
    entries=(),
)
_PAGE_WATCH_INTENT = "Keep an eye on that release page and log every new version that shows up."
_PAGE_WATCH_PROMPT = (
    "Watch a release page and record new versions.\n"
    "1. browse the page at https://releases.example.com/app.\n"
    '2. collection_read_latest("release-watch", k=1) — get the last recorded version.\n'
    "3. If the page shows a newer version than the last entry, collection_write a new "
    "entry with the version and what changed.\n"
    "4. done()."
)


def _seed_research(prompt: str):
    """Seeder: one research+notify collection (indie-metroidvanias) with ``prompt``."""

    def _apply(db: Database) -> None:
        seed_collection(
            db,
            RESEARCH_WATCHER,
            extraction_prompt=prompt,
            intent=RESEARCH_WATCHER_INTENT,
            interval=3600,
            published=True,
        )

    return _apply


def _snapshot(db: Database) -> dict[str, str]:
    return collection_entries(db, _SKILLS)


def _find_key(entries: dict[str, str], *needles: str) -> str | None:
    """The first skill key containing all ``needles`` (case-insensitive)."""
    for key in entries:
        low = key.lower()
        if all(needle.lower() in low for needle in needles):
            return key
    return None


def _research_keys(entries: dict[str, str]) -> list[str]:
    """Skill keys naming the research-and-notify pattern.

    Matches on the KEY only — not the content — so the seeded "Research
    collection — silent" skill (whose body mentions "notifies you" in the
    negative) isn't miscounted as a research-notify duplicate.
    """
    return [key for key in entries if "research" in key.lower() and "notif" in key.lower()]


# ── Cases ─────────────────────────────────────────────────────────────────────


async def test_new_pattern(collector_eval) -> None:
    """A kind of collection no seeded skill covers → a new, generalized skill."""

    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        before_entries = cast("dict[str, str]", before)
        after = collection_entries(db, _SKILLS)
        new_keys = set(after) - set(before_entries)
        if not new_keys:
            return ["no new skill written for the page-watch pattern"]
        watch = [
            k
            for k in new_keys
            if any(
                w in (k + after[k]).lower()
                for w in ("watch", "page", "change", "monitor", "snapshot", "diff")
            )
        ]
        if not watch:
            return [f"new skill(s) don't describe page-watching: {sorted(new_keys)}"]
        # A topic-free key is the generalization signal; the seed's specific host
        # (releases.example.com) must not appear — a bare example URL in an
        # example phrasing is fine.
        leaked = [k for k in watch if "releases.example.com" in after[k].lower()]
        return (
            [f"new skill leaked the source host instead of generalizing: {leaked}"]
            if leaked
            else []
        )

    def _seed(db: Database) -> None:
        seed_collection(
            db,
            _PAGE_WATCH,
            extraction_prompt=_PAGE_WATCH_PROMPT,
            intent=_PAGE_WATCH_INTENT,
            interval=86400,
            published=True,
        )

    await collector_eval(
        case_id="skills-new-pattern",
        collection=_SKILLS,
        seed=_seed,
        snapshot=_snapshot,
        score=_score,
        min_pass_rate=None,
    )


async def test_drift_update(collector_eval) -> None:
    """A recipe gained a GENERALIZABLE step → the matching skill is edited to fold
    it in, NOT forked into a contradictory second skill.  The recipe-drift case."""

    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        before_entries = cast("dict[str, str]", before)
        after = collection_entries(db, _SKILLS)
        rkey = _find_key(before_entries, "research", "notif")
        if rkey is None:
            return ["no seeded research-notify skill present to update"]
        fails: list[str] = []
        folded = any(
            term in after.get(rkey, "").lower()
            for term in (
                "cross-check",
                "cross check",
                "reference",
                "curated list",
                "aggregator",
                "verify",
            )
        )
        if not folded:
            fails.append("research skill did not absorb the cross-check/curated-list step")
        forks = [k for k in set(after) - set(before_entries) if k in _research_keys(after)]
        if forks:
            fails.append(f"forked a new research skill instead of editing in place: {forks}")
        return fails

    await collector_eval(
        case_id="skills-drift-update",
        collection=_SKILLS,
        seed=_seed_research(_RESEARCH_DRIFT),
        snapshot=_snapshot,
        score=_score,
        min_pass_rate=None,
    )


async def test_quirk_left(collector_eval) -> None:
    """A recipe gained a collection-specific quirk → no skill absorbs it."""

    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        after = collection_entries(db, _SKILLS)
        offenders = [key for key, content in after.items() if "[mv]" in (key + content).lower()]
        return (
            [f"absorbed a collection-specific quirk into skill(s): {offenders}"]
            if offenders
            else []
        )

    await collector_eval(
        case_id="skills-quirk-left",
        collection=_SKILLS,
        seed=_seed_research(_RESEARCH_QUIRK),
        snapshot=_snapshot,
        score=_score,
        min_pass_rate=None,
    )


async def test_consolidate(collector_eval) -> None:
    """Two collections of the same kind → still ONE research skill, no duplicate."""

    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        before_entries = cast("dict[str, str]", before)
        after = collection_entries(db, _SKILLS)
        # Both collections are already covered by one seeded skill — the loop must
        # not fork a second skill for the kind.  Any NEW key is a duplicate.
        forked = sorted(set(after) - set(before_entries))
        return [f"forked new skill(s) for an already-covered kind: {forked}"] if forked else []

    def _seed(db: Database) -> None:
        seed_collection(
            db,
            RESEARCH_WATCHER,
            extraction_prompt=_RESEARCH_HEALTHY,
            intent=RESEARCH_WATCHER_INTENT,
            interval=3600,
            published=True,
        )
        seed_collection(
            db,
            BOARD_GAMES,
            extraction_prompt=_BOARDGAMES_NOTIFY,
            intent="Tell me when a good new strategy board game shows up.",
            interval=3600,
            published=True,
        )

    await collector_eval(
        case_id="skills-consolidate",
        collection=_SKILLS,
        seed=_seed,
        snapshot=_snapshot,
        score=_score,
        min_pass_rate=None,
    )


async def test_healthy_noop(collector_eval) -> None:
    """A recipe that already matches its skill → skills unchanged (no over-correction)."""

    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        before_entries = cast("dict[str, str]", before)
        after = collection_entries(db, _SKILLS)
        if after != before_entries:
            changed = {
                k for k in set(after) | set(before_entries) if after.get(k) != before_entries.get(k)
            }
            return [f"mutated skills on a healthy cycle: {sorted(changed)}"]
        return []

    await collector_eval(
        case_id="skills-healthy-noop",
        collection=_SKILLS,
        seed=_seed_research(_RESEARCH_HEALTHY),
        snapshot=_snapshot,
        score=_score,
        min_pass_rate=0.6,
    )


async def test_operate_patterns_kept(collector_eval) -> None:
    """Operate-the-system skills (archive/cadence/flip) have no source collection,
    so the loop must leave them byte-unchanged."""

    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        before_entries = cast("dict[str, str]", before)
        after = collection_entries(db, _SKILLS)
        fails: list[str] = []
        for needle in ("archive", "cadence", "flip"):
            key = _find_key(before_entries, needle)
            if key is not None and after.get(key) != before_entries.get(key):
                fails.append(f"operate-pattern skill {key!r} was modified")
        return fails

    await collector_eval(
        case_id="skills-operate-kept",
        collection=_SKILLS,
        seed=_seed_research(_RESEARCH_DRIFT),
        snapshot=_snapshot,
        score=_score,
        min_pass_rate=0.8,
    )


async def test_quiet(collector_eval) -> None:
    """No user-built collections → catalog empty → no-op, skills unchanged."""

    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        before_entries = cast("dict[str, str]", before)
        after = collection_entries(db, _SKILLS)
        return (
            ["mutated skills with no collections to learn from"] if after != before_entries else []
        )

    await collector_eval(
        case_id="skills-quiet",
        collection=_SKILLS,
        seed=lambda db: None,
        snapshot=_snapshot,
        score=_score,
        min_pass_rate=0.8,
    )
