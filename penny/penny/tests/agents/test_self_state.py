"""Whole-render tests for the self-state header (#1555).

The header is a deterministic projection of the registry (``memory`` rows) + the
ledger (``promptlog`` runs + ``mutation_event``), so it is tested by asserting the
ENTIRE rendered block against a frozen literal — the review guide's whole-render
assertion discipline.  Fixtures seed rows directly (fixed timestamps + ids, no
incidental mutations) so every literal is byte-stable.

Cases: a kitchen-sink folding every shape (healthy + failed run, user-run +
system-actor mutation, a mechanism nearing expiry, a one-shot, an archived
tombstone), the sub-cases (empty deployment, activity overflow, archived-heavy),
the activity-block shape matrix (every run-outcome line shape, the by-design
exclusions, the full mutation action × actor × detail cross-product), then the
full chat system-prompt composition around the header.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from sqlmodel import Session

from penny.agents.chat import ChatAgent
from penny.agents.self_state import SelfStateHeader
from penny.constants import MutationAction, MutationActor, MutationEntityType, PennyConstants
from penny.database.database import Database
from penny.database.models import (
    MemoryEntry,
    MemoryRow,
    MessageLog,
    MutationEvent,
    PromptLog,
    Skill,
    UserInfo,
)
from penny.database.mutation_store import MutationDetail
from penny.prompts import Prompt

USER = "+15550001111"


def _db(tmp_path) -> Database:
    """A fresh, empty database (schema only — no migration-seeded system rows)."""
    db = Database(str(tmp_path / "self_state.db"))
    db.create_tables()
    return db


def _add_collection(
    session: Session,
    name: str,
    *,
    description: str,
    extraction_prompt: str | None = None,
    interval: int | None = None,
    expires_at: datetime | None = None,
    run_at: datetime | None = None,
    max_runs: int | None = None,
    source_log: str | None = None,
    archived: bool = False,
    created_at: datetime,
    updated_at: datetime,
    last_collected_at: datetime | None = None,
) -> None:
    session.add(
        MemoryRow(
            name=name,
            type="collection",
            description=description,
            extraction_prompt=extraction_prompt,
            collector_interval_seconds=interval,
            expires_at=expires_at,
            run_at=run_at,
            max_runs=max_runs,
            source_log=source_log,
            archived=archived,
            created_at=created_at,
            updated_at=updated_at,
            last_collected_at=last_collected_at,
        )
    )


def _add_log(session: Session, name: str, description: str, *, when: datetime) -> None:
    session.add(
        MemoryRow(
            name=name,
            type="log",
            description=description,
            created_at=when,
            updated_at=when,
        )
    )


def _add_entries(session: Session, name: str, count: int, *, when: datetime) -> None:
    for i in range(count):
        session.add(
            MemoryEntry(
                memory_name=name, key=f"k{i}", content=f"c{i}", author="user", created_at=when
            )
        )


def _add_written_entry(
    session: Session,
    *,
    memory_name: str,
    key: str,
    run_id: str,
    when: datetime,
    created_run_id: str | None = None,
) -> None:
    """A collection entry whose CURRENT value was written by ``run_id`` — the
    #1560 ``last_written_by_run_id`` stamp the run-line writes clause joins on
    (#1641).  ``created_run_id`` defaults to ``run_id`` (a fresh write); pass a
    different value to model a rewrite (created by one run, last written by
    another), so the clause is shown to report the LAST writer, not the creator."""
    session.add(
        MemoryEntry(
            memory_name=memory_name,
            key=key,
            content=f"value for {key}",
            author="collector",
            created_at=when,
            created_by_run_id=created_run_id if created_run_id is not None else run_id,
            last_written_by_run_id=run_id,
        )
    )


def _add_skill(session: Session, *, name: str, intent: str, when: datetime) -> None:
    """A taught skill (the ``skill`` registry, #1590) — the taught-skill feed of
    the self-state Skills-and-rules section.  ``steps``/``holes`` are irrelevant
    to the render (it shows name + intent), so they're seeded empty."""
    session.add(
        Skill(
            name=name,
            steps="[]",
            holes="[]",
            intent=intent,
            description=intent,
            author="chat",
            created_at=when,
            updated_at=when,
        )
    )


def _add_run(
    session: Session,
    *,
    run_id: str,
    target: str,
    outcome: str,
    calls: int,
    finished_at: datetime,
    reason: str = "",
) -> None:
    response = {"choices": [{"message": {"tool_calls": [{"id": str(i)} for i in range(calls)]}}]}
    session.add(
        PromptLog(
            model="test-model",
            messages="[]",
            response=json.dumps(response),
            agent_name="collector",
            run_id=run_id,
            run_outcome=outcome,
            run_reason=reason,
            run_target=target,
            timestamp=finished_at,
        )
    )


def _add_emission(session: Session, *, mechanism: str, content: str, sent_at: datetime) -> None:
    """A delivered autonomous send: an OUTGOING messagelog row stamped with the
    mechanism that produced it (#1568) — the shape the activity block renders as a
    ``sent · …`` line.  A direct reply (``mechanism=None``) is excluded, so these
    seed a non-NULL mechanism."""
    session.add(
        MessageLog(
            direction=PennyConstants.MessageDirection.OUTGOING,
            sender="penny",
            content=content,
            mechanism=mechanism,
            timestamp=sent_at,
        )
    )


def _add_direct_reply(session: Session, *, content: str, when: datetime) -> None:
    """An OUTGOING messagelog row with NO mechanism — a direct reply, which the
    activity block excludes by construction (it is the conversation, already in
    context)."""
    session.add(
        MessageLog(
            direction=PennyConstants.MessageDirection.OUTGOING,
            sender="penny",
            content=content,
            timestamp=when,
        )
    )


def _add_chat_run(session: Session, *, run_id: str, when: datetime) -> None:
    """A conversational chat-agent run: stamps NO ``run_outcome`` and NO
    ``run_target`` — the structural shape the activity block excludes (chat turns
    are already the conversation; the block renders its complement)."""
    session.add(
        PromptLog(
            model="test-model",
            messages="[]",
            response=json.dumps({"choices": []}),
            agent_name="chat",
            run_id=run_id,
            timestamp=when,
        )
    )


def _add_mutation(
    session: Session,
    *,
    entity_name: str,
    action: MutationAction,
    actor: MutationActor,
    created_at: datetime,
    run_id: str | None = None,
    detail: MutationDetail | None = None,
) -> None:
    session.add(
        MutationEvent(
            entity_type=MutationEntityType.COLLECTION.value,
            entity_name=entity_name,
            action=action.value,
            actor=actor.value,
            run_id=run_id,
            detail=detail.model_dump_json() if detail is not None else None,
            created_at=created_at,
        )
    )


def _add_user(db: Database) -> None:
    with Session(db.engine) as session:
        session.add(
            UserInfo(
                sender=USER,
                name="Alex",
                location="Toronto, Canada",
                timezone="America/Toronto",
                date_of_birth="1990-01-01",
            )
        )
        session.commit()


def _t(hour: int, minute: int = 0, day: int = 11) -> datetime:
    return datetime(2026, 7, day, hour, minute, tzinfo=UTC)


def _seed_kitchen_sink(db: Database) -> None:
    """One collection of every renderable shape + a healthy and a failed run + a
    user-run and a system-actor mutation + a log and a plain (non-collector)
    collection for the map."""
    with Session(db.engine) as session:
        _add_collection(
            session,
            "price-watch",
            description="watch a product price",
            extraction_prompt="1. browse the page",
            interval=21600,
            expires_at=_t(0, 0, day=20),
            created_at=_t(6),
            updated_at=_t(9, 20),
            last_collected_at=_t(9, 14),
        )
        _add_collection(
            session,
            "news-digest",
            description="gather headlines",
            extraction_prompt="1. read the feed",
            interval=3600,
            created_at=_t(6),
            updated_at=_t(6),
            last_collected_at=_t(8),
        )
        _add_collection(
            session,
            "reminder",
            description="one-off reminder",
            extraction_prompt="1. remind",
            run_at=_t(12),
            max_runs=1,
            created_at=_t(6),
            updated_at=_t(6),
        )
        _add_collection(
            session,
            "old-watch",
            description="a retired watch",
            extraction_prompt="1. old",
            interval=3600,
            archived=True,
            created_at=_t(5, 0, day=1),
            updated_at=_t(8, 30),
        )
        _add_log(session, "chat-log", "shared conversation log", when=_t(6))
        _add_collection(
            session,
            "favorites",
            description="things the user likes",
            created_at=_t(6),
            updated_at=_t(6),
        )
        _add_entries(session, "favorites", 2, when=_t(7))
        # The Skills-and-rules section's one feed (#1471/#1624): the taught-skill
        # registry (the sole skills store — the legacy ``skills`` collection's
        # standing-rules feed retired with the collection, migration 0092).
        _add_skill(
            session,
            name="Track a shipment",
            intent="track my package from acme and tell me when it moves",
            when=_t(7),
        )
        _add_skill(
            session,
            name="Watch a page field",
            intent="watch the price on a product page and ping me when it drops",
            when=_t(7),
        )
        _add_run(
            session,
            run_id="7f3a1b2c",
            target="price-watch",
            outcome="worked",
            calls=3,
            finished_at=_t(9, 14),
        )
        _add_run(
            session,
            run_id="88d14e5f",
            target="news-digest",
            outcome="failed",
            calls=2,
            finished_at=_t(8),
        )
        # The price-watch cycle (run 7f3a1b2c) wrote one entry — its run line
        # grows a writes clause naming the key + collection (#1641).  Created by
        # an earlier run, last written by this one, so the clause reports the
        # current-value writer; the failed news-digest run wrote nothing (no
        # clause — byte-identical to the pre-#1641 line).
        _add_written_entry(
            session,
            memory_name="price-watch",
            key="aurora deck 2 price",
            run_id="7f3a1b2c",
            when=_t(9, 14),
            created_run_id="5c0dd001",
        )
        _add_mutation(
            session,
            entity_name="price-watch",
            action=MutationAction.UPDATED,
            actor=MutationActor.USER_RUN,
            created_at=_t(9, 20),
            run_id="66aa0099",
            detail=MutationDetail(changed_fields=["collector_interval_seconds"]),
        )
        _add_mutation(
            session,
            entity_name="old-watch",
            action=MutationAction.ARCHIVED,
            actor=MutationActor.SYSTEM,
            created_at=_t(8, 30),
            detail=MutationDetail(note="max_runs reached (1 of 1)"),
        )
        # A delivered autonomous send (#1568) interleaves into the activity block;
        # a direct reply (no mechanism) seeded alongside must NOT appear.
        _add_emission(
            session,
            mechanism="price-watch",
            content="Heads up: the price dropped to $42!",
            sent_at=_t(9, 5),
        )
        _add_direct_reply(session, content="sure, happy to help!", when=_t(9, 18))
        session.commit()
    _add_user(db)


# ── 1. Kitchen sink — every shape in one whole-render literal ─────────────


def test_self_state_kitchen_sink_render(tmp_path):
    db = _db(tmp_path)
    _seed_kitchen_sink(db)
    actual = SelfStateHeader(db, USER).render()
    assert actual == _KITCHEN_SINK


# ── 1b. on_advance trigger — the mechanisms line reads the source clause ───


def test_self_state_on_advance_mechanism_render(tmp_path):
    """A source-driven (on_advance) collection renders ``on advance of <log>`` in
    place of a cadence clause on the mechanisms line (#1604), so the trigger reads
    at a glance without a memory_metadata hop."""
    db = _db(tmp_path)
    with Session(db.engine) as session:
        _add_log(session, "events-log", "an event stream", when=_t(6))
        _add_collection(
            session,
            "chained-watch",
            description="digest upstream events",
            extraction_prompt="1. digest the events.",
            interval=30,  # only the min floor — the trigger clause wins the render
            source_log="events-log",
            created_at=_t(6),
            updated_at=_t(6),
        )
        session.commit()
    _add_user(db)
    assert SelfStateHeader(db, USER).render() == _ON_ADVANCE


# ── 2. Empty state — a fresh deployment ───────────────────────────────────


def test_self_state_empty_render(tmp_path):
    db = _db(tmp_path)
    actual = SelfStateHeader(db, None).render()
    assert actual == _EMPTY


# ── 3. Budget overflow — the cap line is visible ──────────────────────────


def test_self_state_activity_overflow_render(tmp_path):
    db = _db(tmp_path)
    with Session(db.engine) as session:
        _add_collection(
            session,
            "watcher",
            description="a watcher",
            extraction_prompt="1. go",
            interval=3600,
            created_at=_t(6),
            updated_at=_t(6),
        )
        # One more run than the activity cap so the "+ older activity" tail shows.
        for i in range(PennyConstants.SELF_STATE_ACTIVITY_LIMIT + 1):
            _add_run(
                session,
                run_id=f"run{i:02d}",
                target="watcher",
                outcome="worked",
                calls=1,
                finished_at=_t(9, i),
            )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    assert actual == _OVERFLOW


# ── 4. Archived-heavy — retired tombstones stay enumerable ────────────────


def test_self_state_archived_heavy_render(tmp_path):
    db = _db(tmp_path)
    with Session(db.engine) as session:
        _add_collection(
            session,
            "live-watch",
            description="still running",
            extraction_prompt="1. go",
            interval=3600,
            created_at=_t(6),
            updated_at=_t(6),
        )
        for i in range(3):
            _add_collection(
                session,
                f"retired-{i}",
                description=f"retired watch {i}",
                extraction_prompt="1. old",
                interval=3600,
                archived=True,
                created_at=_t(5, 0, day=1),
                updated_at=_t(8, i),
            )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    assert actual == _ARCHIVED_HEAVY


# ── 4c. Skills-and-rules section — the taught-skill feed (#1471/#1624) ────
#
# The populated shape is pinned here and in the kitchen sink; the empty shape
# (the fresh-install default — migration 0084 ships the taught-skill table
# empty) is the empty deployment's placeholder.  The taught-skill registry is
# the section's ONLY feed: the legacy ``skills`` collection's standing-rules
# feed retired with the collection (#1624, migration 0092).


def test_self_state_taught_skills_only_render(tmp_path):
    """The taught-skill registry renders under its drill-down label (name order,
    intent collapsed to one line) — the section's sole feed."""
    db = _db(tmp_path)
    with Session(db.engine) as session:
        _add_skill(
            session,
            name="Track a shipment",
            intent="track my package from acme and tell me when it moves",
            when=_t(7),
        )
        _add_skill(
            session,
            name="Watch a page field",
            intent="watch the price on a product page and ping me when it drops",
            when=_t(7),
        )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    assert actual == _TAUGHT_SKILLS_ONLY


# ── 4b. Activity-block shape matrix ───────────────────────────────────────
#
# The activity block is a render with an enumerable input space, so every line
# shape renderable from today's substrate is pinned as a whole-render literal:
# run lines across every RunOutcome value (plus the zero-call and singular-call
# forms), the exclusions (cancelled runs; chat runs, per the complement-of-
# context rule), and the full mutation cross-product — action (created /
# updated / archived / unarchived) × actor (user-run WITH its run id / system
# WITHOUT) × detail (changed-fields list / cause note / no detail), grouped one
# test per detail variant with all eight action×actor cells in each literal.
#
# These literals are the template two later tickets grow: #1568 extends the
# SAME renders with emission/send lines, and #1562's STOP enums replace the
# RunOutcome vocabulary in the run lines when they land.


def test_activity_run_lines_every_rendered_outcome(tmp_path):
    """One run line per rendering RunOutcome — WORKED / FAILED / NO_WORK /
    INCOMPLETE — plus the zero-call form and the singular '1 call' form.
    (CANCELLED is excluded by design; pinned in the exclusion test below.)"""
    db = _db(tmp_path)
    with Session(db.engine) as session:
        _add_run(
            session,
            run_id="aa11worked",
            target="alpha-watch",
            outcome="worked",
            calls=3,
            finished_at=_t(9, 57),
        )
        _add_run(
            session,
            run_id="bb22failed",
            target="beta-watch",
            outcome="failed",
            calls=2,
            finished_at=_t(9, 56),
        )
        _add_run(
            session,
            run_id="cc33nowork",
            target="gamma-watch",
            outcome="no_work",
            calls=1,
            finished_at=_t(9, 55),
        )
        _add_run(
            session,
            run_id="dd44incomp",
            target="delta-watch",
            outcome="incomplete",
            calls=4,
            finished_at=_t(9, 54),
        )
        # A failed run that made NO tool call at all (the exhausted-no-call bail).
        _add_run(
            session,
            run_id="ee55nocall",
            target="alpha-watch",
            outcome="failed",
            calls=0,
            finished_at=_t(9, 53),
        )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    assert actual == _RUN_OUTCOME_MATRIX


# ── 4d. Run-line writes clause — what a run wrote is ambient (#1641) ───────
#
# A run's line grows a ``· wrote …`` clause from the #1560 entry stamp
# ``last_written_by_run_id``: the single-write form names the key, the
# multi-write form compacts to a count + a bounded key sample with an ellipsis,
# and a no-write run renders byte-identical to the pre-#1641 line.  Keys render
# invocation-form (``'<key>'`` — the value a read tool's ``key=`` receives).


def test_activity_run_lines_carry_writes(tmp_path):
    """The four writes-clause shapes in one render: a single-write run names its
    key + collection; a two-write run (exactly the key-sample cap) lists both keys
    with NO ellipsis; a three-write run compacts to a count + a two-key sample WITH
    an ellipsis; a no-write run is byte-identical to today.  The single-write entry
    was CREATED by an earlier run but LAST WRITTEN by this one, proving the clause
    reports the current-value writer (``last_written_by_run_id``), not the
    creator."""
    db = _db(tmp_path)
    with Session(db.engine) as session:
        _add_collection(
            session, "knowledge", description="web-page facts", created_at=_t(6), updated_at=_t(6)
        )
        _add_collection(
            session, "board-games", description="games to try", created_at=_t(6), updated_at=_t(6)
        )
        _add_collection(
            session, "watchlist", description="pages watched", created_at=_t(6), updated_at=_t(6)
        )
        _add_collection(
            session,
            "news-digest",
            description="gather headlines",
            created_at=_t(6),
            updated_at=_t(6),
        )
        _add_run(
            session,
            run_id="a1knowrun",
            target="knowledge",
            outcome="worked",
            calls=2,
            finished_at=_t(9, 30),
        )
        _add_written_entry(
            session,
            memory_name="knowledge",
            key="aurora deck 2 price",
            run_id="a1knowrun",
            when=_t(9, 30),
            created_run_id="0old0run",
        )
        _add_run(
            session,
            run_id="b2boardrun",
            target="board-games",
            outcome="worked",
            calls=4,
            finished_at=_t(9, 20),
        )
        for index in range(3):
            _add_written_entry(
                session,
                memory_name="board-games",
                key=f"k{index + 1}",
                run_id="b2boardrun",
                when=_t(9, 20 + index),
            )
        # Exactly the key-sample cap (2): both keys render, NO ellipsis — the
        # ``== cap`` boundary of the count-vs-sample split (#1641).
        _add_run(
            session,
            run_id="e5pairrun",
            target="watchlist",
            outcome="worked",
            calls=2,
            finished_at=_t(9, 15),
        )
        for index in range(2):
            _add_written_entry(
                session,
                memory_name="watchlist",
                key=f"p{index + 1}",
                run_id="e5pairrun",
                when=_t(9, 15 + index),
            )
        _add_run(
            session,
            run_id="c3newsrun",
            target="news-digest",
            outcome="worked",
            calls=1,
            finished_at=_t(9, 10),
        )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    assert actual == _RUN_WRITES


def test_activity_run_line_multiple_collections(tmp_path):
    """A run that wrote keyed entries to two collections renders one ``· wrote …``
    clause per collection, grouped by name (the join is run-type agnostic — a chat
    turn writing likes + dislikes would render the same way).  Groups render in
    name order (``dislikes`` before ``likes``)."""
    db = _db(tmp_path)
    with Session(db.engine) as session:
        _add_collection(
            session, "likes", description="things liked", created_at=_t(6), updated_at=_t(6)
        )
        _add_collection(
            session, "dislikes", description="things disliked", created_at=_t(6), updated_at=_t(6)
        )
        _add_run(
            session,
            run_id="d4dualrun",
            target="likes",
            outcome="worked",
            calls=3,
            finished_at=_t(9, 40),
        )
        _add_written_entry(
            session, memory_name="likes", key="hiking", run_id="d4dualrun", when=_t(9, 40)
        )
        _add_written_entry(
            session, memory_name="dislikes", key="rain", run_id="d4dualrun", when=_t(9, 41)
        )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    assert actual == _RUN_WRITES_MULTI_COLLECTION


def test_activity_excludes_cancelled_and_chat_runs(tmp_path):
    """The two by-design exclusions, proven against a rendering sibling: a
    CANCELLED collector run (not a real cycle) and a chat run (no outcome, no
    target — already the conversation) are seeded NEWER than a worked sibling,
    yet only the sibling renders."""
    db = _db(tmp_path)
    with Session(db.engine) as session:
        _add_run(
            session,
            run_id="ff66cancel",
            target="alpha-watch",
            outcome="cancelled",
            calls=2,
            finished_at=_t(9, 50),
        )
        _add_chat_run(session, run_id="99chatturn", when=_t(9, 45))
        _add_run(
            session,
            run_id="aa77worked",
            target="alpha-watch",
            outcome="worked",
            calls=1,
            finished_at=_t(9, 30),
        )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    assert "ff66cancel" not in actual
    assert "99chatturn" not in actual
    assert actual == _EXCLUSION_RENDER


def test_activity_mutation_lines_changed_fields_matrix(tmp_path):
    """Every action × actor cell with a changed-fields detail: the multi-field
    'changed a, b' tail, user-run rows naming their run id, system rows none."""
    db = _db(tmp_path)
    _seed_mutation_matrix(db, detail_factory=_changed_fields_detail)
    actual = SelfStateHeader(db, None).render()
    assert actual == _MUTATION_CHANGED_FIELDS_MATRIX


def test_activity_mutation_lines_note_matrix(tmp_path):
    """Every action × actor cell with a cause-note detail (the system-archive
    policy-reason shape, e.g. a max_runs retire)."""
    db = _db(tmp_path)
    _seed_mutation_matrix(db, detail_factory=_note_detail)
    actual = SelfStateHeader(db, None).render()
    assert actual == _MUTATION_NOTE_MATRIX


def test_activity_mutation_lines_no_detail_matrix(tmp_path):
    """Every action × actor cell with NO detail payload: the bare line — no
    tail, no dash."""
    db = _db(tmp_path)
    _seed_mutation_matrix(db, detail_factory=_no_detail)
    actual = SelfStateHeader(db, None).render()
    assert actual == _MUTATION_BARE_MATRIX


def test_activity_emission_lines_and_exclusions(tmp_path):
    """Delivered autonomous sends (#1568) render as ``sent · …`` lines, interleaved
    by time with a run: a short body renders whole, a long one is snippet-truncated
    with an ellipsis, and a direct reply (no mechanism) seeded NEWER than an
    emission is excluded (it is the conversation, not its complement)."""
    db = _db(tmp_path)
    with Session(db.engine) as session:
        _add_run(
            session,
            run_id="rr11worked",
            target="alpha-watch",
            outcome="worked",
            calls=2,
            finished_at=_t(9, 50),
        )
        _add_emission(session, mechanism="alpha-watch", content="A short ping.", sent_at=_t(9, 40))
        # A direct reply NEWER than the beta emission below — must NOT appear.
        _add_direct_reply(session, content="you got it!", when=_t(9, 35))
        _add_emission(
            session,
            mechanism="beta-watch",
            content=(
                "The morning digest is ready with today's five top headlines "
                "for you to skim over coffee"
            ),
            sent_at=_t(9, 20),
        )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    assert "you got it!" not in actual
    assert actual == _EMISSION_MATRIX


_MATRIX_ACTIONS = [
    MutationAction.CREATED,
    MutationAction.UPDATED,
    MutationAction.ARCHIVED,
    MutationAction.UNARCHIVED,
]


def _changed_fields_detail() -> MutationDetail:
    return MutationDetail(changed_fields=["cadence", "expiry"])


def _note_detail() -> MutationDetail:
    return MutationDetail(note="max_runs reached (2 of 2)")


def _no_detail() -> None:
    return None


def _seed_mutation_matrix(db: Database, *, detail_factory) -> None:
    """All eight action × actor cells on one entity, one minute apart (newest
    first = created/user-run), each cell carrying ``detail_factory()``'s detail
    variant.  User-run cells carry a distinct run id; system cells carry none."""
    with Session(db.engine) as session:
        minute = 57
        for index, action in enumerate(_MATRIX_ACTIONS):
            _add_mutation(
                session,
                entity_name="demo-watch",
                action=action,
                actor=MutationActor.USER_RUN,
                run_id=f"aa{index}0run",
                created_at=_t(9, minute),
                detail=detail_factory(),
            )
            minute -= 1
            _add_mutation(
                session,
                entity_name="demo-watch",
                action=action,
                actor=MutationActor.SYSTEM,
                created_at=_t(9, minute),
                detail=detail_factory(),
            )
            minute -= 1
        session.commit()


# ── 5. Full chat system-prompt composition ────────────────────────────────


@pytest.mark.asyncio
async def test_full_chat_system_prompt_composition(tmp_path):
    """The whole chat system prompt: Identity + Instructions + self-state header
    (the dynamic tail), with the speculative recall / profile / inventory gone."""
    db = _db(tmp_path)
    _seed_kitchen_sink(db)
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    agent._pending_page_context = None
    prompt = await agent._build_system_prompt(USER)
    expected = (
        f"## Identity\n{Prompt.PENNY_IDENTITY}\n\n"
        f"## Instructions\n{Prompt.CONVERSATION_PROMPT}\n\n"
        f"{_KITCHEN_SINK}"
    )
    assert prompt == expected


# ── Frozen render literals (filled from the captured actual) ──────────────

_KITCHEN_SINK = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "- news-digest — active · every 3600 · last run FAILED 2026-07-11 08:00 UTC\n"
    "- old-watch — archived 2026-07-11 08:30 UTC · no runs yet\n"
    "- price-watch — active · every 21600 · expires 2026-07-20 00:00 UTC · last run WORKED "
    "2026-07-11 09:14 UTC\n"
    "- reminder — active · once at 2026-07-11T12:00:00+00:00 · one-shot · no runs yet\n"
    "\n"
    "### Recent activity\n"
    "change · 2026-07-11 09:20 UTC · price-watch updated by user-run (run 66aa0099) — "
    "changed collector_interval_seconds\n"
    "run 7f3a1b2c · 2026-07-11 09:14 UTC · price-watch → WORKED (3 calls) · "
    "wrote 'aurora deck 2 price' → `price-watch`\n"
    'sent · 2026-07-11 09:05 UTC · price-watch — "Heads up: the price dropped to $42!"\n'
    "change · 2026-07-11 08:30 UTC · old-watch archived by system — max_runs reached (1 "
    "of 1)\n"
    "run 88d14e5f · 2026-07-11 08:00 UTC · news-digest → FAILED (2 calls)\n"
    "\n"
    "### Your memory\n"
    "- chat-log (log, 0 entries) — shared conversation log\n"
    "- favorites (collection, 2 entries) — things the user likes\n"
    "- news-digest (collection, 0 entries) — gather headlines\n"
    "- price-watch (collection, 1 entries) — watch a product price\n"
    "- reminder (collection, 0 entries) — one-off reminder\n"
    "\n"
    "### Skills and rules\n"
    "Skills you've been taught — skill_read(<name>) for the full recipe:\n"
    "- Track a shipment — track my package from acme and tell me when it moves\n"
    "- Watch a page field — watch the price on a product page and ping me when it drops\n"
    "\n"
    "### About the user\n"
    "- name: Alex\n"
    "- timezone: America/Toronto\n"
    "- location: Toronto, Canada\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)

_ON_ADVANCE = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "- chained-watch — active · on advance of events-log · no runs yet\n"
    "\n"
    "### Recent activity\n"
    "(no recent activity)\n"
    "\n"
    "### Your memory\n"
    "- chained-watch (collection, 0 entries) — digest upstream events\n"
    "- events-log (log, 0 entries) — an event stream\n"
    "\n"
    "### Skills and rules\n"
    "(no skills yet — when a task needs one, ask the user to walk you through it "
    "once and you'll learn it automatically)\n"
    "\n"
    "### About the user\n"
    "- name: Alex\n"
    "- timezone: America/Toronto\n"
    "- location: Toronto, Canada\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)

_EMPTY = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "(no mechanisms yet)\n"
    "\n"
    "### Recent activity\n"
    "(no recent activity)\n"
    "\n"
    "### Your memory\n"
    "(no stores yet)\n"
    "\n"
    "### Skills and rules\n"
    "(no skills yet — when a task needs one, ask the user to walk you through it "
    "once and you'll learn it automatically)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)

_OVERFLOW = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "- watcher — active · every 3600 · last run WORKED 2026-07-11 09:08 UTC\n"
    "\n"
    "### Recent activity\n"
    "run run08 · 2026-07-11 09:08 UTC · watcher → WORKED (1 call)\n"
    "run run07 · 2026-07-11 09:07 UTC · watcher → WORKED (1 call)\n"
    "run run06 · 2026-07-11 09:06 UTC · watcher → WORKED (1 call)\n"
    "run run05 · 2026-07-11 09:05 UTC · watcher → WORKED (1 call)\n"
    "run run04 · 2026-07-11 09:04 UTC · watcher → WORKED (1 call)\n"
    "run run03 · 2026-07-11 09:03 UTC · watcher → WORKED (1 call)\n"
    "run run02 · 2026-07-11 09:02 UTC · watcher → WORKED (1 call)\n"
    "run run01 · 2026-07-11 09:01 UTC · watcher → WORKED (1 call)\n"
    "+ older activity — read_run_calls(<target>)\n"
    "\n"
    "### Your memory\n"
    "- watcher (collection, 0 entries) — a watcher\n"
    "\n"
    "### Skills and rules\n"
    "(no skills yet — when a task needs one, ask the user to walk you through it "
    "once and you'll learn it automatically)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)

_ARCHIVED_HEAVY = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "- live-watch — active · every 3600 · no runs yet\n"
    "- retired-0 — archived 2026-07-11 08:00 UTC · no runs yet\n"
    "- retired-1 — archived 2026-07-11 08:01 UTC · no runs yet\n"
    "- retired-2 — archived 2026-07-11 08:02 UTC · no runs yet\n"
    "\n"
    "### Recent activity\n"
    "(no recent activity)\n"
    "\n"
    "### Your memory\n"
    "- live-watch (collection, 0 entries) — still running\n"
    "\n"
    "### Skills and rules\n"
    "(no skills yet — when a task needs one, ask the user to walk you through it "
    "once and you'll learn it automatically)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)

_TAUGHT_SKILLS_ONLY = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "(no mechanisms yet)\n"
    "\n"
    "### Recent activity\n"
    "(no recent activity)\n"
    "\n"
    "### Your memory\n"
    "(no stores yet)\n"
    "\n"
    "### Skills and rules\n"
    "Skills you've been taught — skill_read(<name>) for the full recipe:\n"
    "- Track a shipment — track my package from acme and tell me when it moves\n"
    "- Watch a page field — watch the price on a product page and ping me when it drops\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)


# ── Shape-matrix literals (filled from the captured actual) ──────────────

_RUN_OUTCOME_MATRIX = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "(no mechanisms yet)\n"
    "\n"
    "### Recent activity\n"
    "run aa11worked · 2026-07-11 09:57 UTC · alpha-watch → WORKED (3 calls)\n"
    "run bb22failed · 2026-07-11 09:56 UTC · beta-watch → FAILED (2 calls)\n"
    "run cc33nowork · 2026-07-11 09:55 UTC · gamma-watch → NO_WORK (1 call)\n"
    "run dd44incomp · 2026-07-11 09:54 UTC · delta-watch → INCOMPLETE (4 calls)\n"
    "run ee55nocall · 2026-07-11 09:53 UTC · alpha-watch → FAILED (0 calls)\n"
    "\n"
    "### Your memory\n"
    "(no stores yet)\n"
    "\n"
    "### Skills and rules\n"
    "(no skills yet — when a task needs one, ask the user to walk you through it "
    "once and you'll learn it automatically)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)

_RUN_WRITES = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "(no mechanisms yet)\n"
    "\n"
    "### Recent activity\n"
    "run a1knowrun · 2026-07-11 09:30 UTC · knowledge → WORKED (2 calls) · "
    "wrote 'aurora deck 2 price' → `knowledge`\n"
    "run b2boardrun · 2026-07-11 09:20 UTC · board-games → WORKED (4 calls) · "
    "wrote 3 entries → `board-games` ('k1', 'k2', …)\n"
    "run e5pairrun · 2026-07-11 09:15 UTC · watchlist → WORKED (2 calls) · "
    "wrote 2 entries → `watchlist` ('p1', 'p2')\n"
    "run c3newsrun · 2026-07-11 09:10 UTC · news-digest → WORKED (1 call)\n"
    "\n"
    "### Your memory\n"
    "- board-games (collection, 3 entries) — games to try\n"
    "- knowledge (collection, 1 entries) — web-page facts\n"
    "- news-digest (collection, 0 entries) — gather headlines\n"
    "- watchlist (collection, 2 entries) — pages watched\n"
    "\n"
    "### Skills and rules\n"
    "(no skills yet — when a task needs one, ask the user to walk you through it "
    "once and you'll learn it automatically)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)

_RUN_WRITES_MULTI_COLLECTION = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "(no mechanisms yet)\n"
    "\n"
    "### Recent activity\n"
    "run d4dualrun · 2026-07-11 09:40 UTC · likes → WORKED (3 calls) · "
    "wrote 'rain' → `dislikes` · wrote 'hiking' → `likes`\n"
    "\n"
    "### Your memory\n"
    "- dislikes (collection, 1 entries) — things disliked\n"
    "- likes (collection, 1 entries) — things liked\n"
    "\n"
    "### Skills and rules\n"
    "(no skills yet — when a task needs one, ask the user to walk you through it "
    "once and you'll learn it automatically)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)

_EXCLUSION_RENDER = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "(no mechanisms yet)\n"
    "\n"
    "### Recent activity\n"
    "run aa77worked · 2026-07-11 09:30 UTC · alpha-watch → WORKED (1 call)\n"
    "\n"
    "### Your memory\n"
    "(no stores yet)\n"
    "\n"
    "### Skills and rules\n"
    "(no skills yet — when a task needs one, ask the user to walk you through it "
    "once and you'll learn it automatically)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)

_MUTATION_CHANGED_FIELDS_MATRIX = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "(no mechanisms yet)\n"
    "\n"
    "### Recent activity\n"
    "change · 2026-07-11 09:57 UTC · demo-watch created by user-run (run aa00run) — "
    "changed cadence, expiry\n"
    "change · 2026-07-11 09:56 UTC · demo-watch created by system — changed cadence, "
    "expiry\n"
    "change · 2026-07-11 09:55 UTC · demo-watch updated by user-run (run aa10run) — "
    "changed cadence, expiry\n"
    "change · 2026-07-11 09:54 UTC · demo-watch updated by system — changed cadence, "
    "expiry\n"
    "change · 2026-07-11 09:53 UTC · demo-watch archived by user-run (run aa20run) — "
    "changed cadence, expiry\n"
    "change · 2026-07-11 09:52 UTC · demo-watch archived by system — changed cadence, "
    "expiry\n"
    "change · 2026-07-11 09:51 UTC · demo-watch unarchived by user-run (run aa30run) — "
    "changed cadence, expiry\n"
    "change · 2026-07-11 09:50 UTC · demo-watch unarchived by system — changed cadence, "
    "expiry\n"
    "\n"
    "### Your memory\n"
    "(no stores yet)\n"
    "\n"
    "### Skills and rules\n"
    "(no skills yet — when a task needs one, ask the user to walk you through it "
    "once and you'll learn it automatically)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)

_MUTATION_NOTE_MATRIX = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "(no mechanisms yet)\n"
    "\n"
    "### Recent activity\n"
    "change · 2026-07-11 09:57 UTC · demo-watch created by user-run (run aa00run) — "
    "max_runs reached (2 of 2)\n"
    "change · 2026-07-11 09:56 UTC · demo-watch created by system — max_runs reached (2 "
    "of 2)\n"
    "change · 2026-07-11 09:55 UTC · demo-watch updated by user-run (run aa10run) — "
    "max_runs reached (2 of 2)\n"
    "change · 2026-07-11 09:54 UTC · demo-watch updated by system — max_runs reached (2 "
    "of 2)\n"
    "change · 2026-07-11 09:53 UTC · demo-watch archived by user-run (run aa20run) — "
    "max_runs reached (2 of 2)\n"
    "change · 2026-07-11 09:52 UTC · demo-watch archived by system — max_runs reached (2 "
    "of 2)\n"
    "change · 2026-07-11 09:51 UTC · demo-watch unarchived by user-run (run aa30run) — "
    "max_runs reached (2 of 2)\n"
    "change · 2026-07-11 09:50 UTC · demo-watch unarchived by system — max_runs reached "
    "(2 of 2)\n"
    "\n"
    "### Your memory\n"
    "(no stores yet)\n"
    "\n"
    "### Skills and rules\n"
    "(no skills yet — when a task needs one, ask the user to walk you through it "
    "once and you'll learn it automatically)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)

_EMISSION_MATRIX = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "(no mechanisms yet)\n"
    "\n"
    "### Recent activity\n"
    "run rr11worked · 2026-07-11 09:50 UTC · alpha-watch → WORKED (2 calls)\n"
    'sent · 2026-07-11 09:40 UTC · alpha-watch — "A short ping."\n'
    'sent · 2026-07-11 09:20 UTC · beta-watch — "The morning digest is ready with '
    "today's five top…\"\n"
    "\n"
    "### Your memory\n"
    "(no stores yet)\n"
    "\n"
    "### Skills and rules\n"
    "(no skills yet — when a task needs one, ask the user to walk you through it "
    "once and you'll learn it automatically)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)

_MUTATION_BARE_MATRIX = (
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "(no mechanisms yet)\n"
    "\n"
    "### Recent activity\n"
    "change · 2026-07-11 09:57 UTC · demo-watch created by user-run (run aa00run)\n"
    "change · 2026-07-11 09:56 UTC · demo-watch created by system\n"
    "change · 2026-07-11 09:55 UTC · demo-watch updated by user-run (run aa10run)\n"
    "change · 2026-07-11 09:54 UTC · demo-watch updated by system\n"
    "change · 2026-07-11 09:53 UTC · demo-watch archived by user-run (run aa20run)\n"
    "change · 2026-07-11 09:52 UTC · demo-watch archived by system\n"
    "change · 2026-07-11 09:51 UTC · demo-watch unarchived by user-run (run aa30run)\n"
    "change · 2026-07-11 09:50 UTC · demo-watch unarchived by system\n"
    "\n"
    "### Your memory\n"
    "(no stores yet)\n"
    "\n"
    "### Skills and rules\n"
    "(no skills yet — when a task needs one, ask the user to walk you through it "
    "once and you'll learn it automatically)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, get_event(run <id>) for one run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find(query=<text>) to find anything of yours by meaning "
    "(a collection, a skill, or a stored entry), and collection_catalog() for "
    "every collection."
)
