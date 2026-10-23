"""Whole-render tests for the self-state header (#1555).

The header is a deterministic projection of the registry (``memory`` rows) + the
ledger (``promptlog`` runs + ``mutation_event``), so it is tested by asserting the
ENTIRE rendered block against a frozen literal — the review guide's whole-render
assertion discipline.  Fixtures seed rows directly (fixed timestamps + ids, no
incidental mutations) so every literal is byte-stable.

Cases: a kitchen-sink folding every shape (healthy + failed run, user-run +
system-actor mutation, a mechanism nearing expiry, a one-shot, an archived
tombstone), the sub-cases (empty deployment, activity overflow, archived-heavy,
past-the-old-cap mechanisms/store-map renders, the archived-names line),
the activity-block shape matrix (every run-outcome line shape, the by-design
exclusions, the full mutation action × actor × detail cross-product), then the
full chat system-prompt composition around the header.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from sqlmodel import Session

from penny.agents.chat import ChatAgent
from penny.agents.self_state import SelfStateHeader
from penny.config import Config
from penny.config_params import RuntimeParams
from penny.constants import MutationAction, MutationActor, MutationEntityType, PennyConstants
from penny.database.database import Database
from penny.database.models import (
    MemoryEntry,
    MemoryRow,
    MessageLog,
    MutationEvent,
    PromptLog,
    UserInfo,
)
from penny.database.mutation_store import MutationDetail
from penny.prompts import Prompt
from penny.tests.conftest import DEFAULT_TEST_CONFIG

USER = "+15550001111"


pytestmark = pytest.mark.bare_db


def _add_collection(
    session: Session,
    name: str,
    *,
    description: str,
    extraction_prompt: str | None = None,
    schedule: str | None = None,
    expires_at: datetime | None = None,
    max_runs: int | None = None,
    archived: bool = False,
    notify: bool = False,
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
            schedule=schedule,
            expires_at=expires_at,
            max_runs=max_runs,
            archived=archived,
            notify=notify,
            created_at=created_at,
            updated_at=updated_at,
            last_collected_at=last_collected_at,
        )
    )


def _add_log(
    session: Session, name: str, description: str, *, when: datetime, archived: bool = False
) -> None:
    session.add(
        MemoryRow(
            name=name,
            type="log",
            description=description,
            archived=archived,
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


def _add_rule(session: Session, key: str, content: str, when: datetime) -> None:
    """One standing rule — an entry in the ``rules`` collection the way a user's
    ``collection_write`` leaves one (a short key, the rule in their words)."""
    session.add(
        MemoryEntry(memory_name="rules", key=key, content=content, author="user", created_at=when)
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


def _add_failed_chat_run(session: Session, *, run_id: str, reason: str, when: datetime) -> None:
    """A chat run that ended in a failure (#2355 part D) — ``ChatAgent._record_turn_
    failure``'s own shape: ``run_outcome = "failed"``, a THIRD-PERSON structural
    ``reason``, and NO ``run_target`` (a chat run names no collection — that is what
    keeps it out of ``recent_collector_runs`` and routes it through
    ``recent_failed_chat_turns`` instead)."""
    session.add(
        PromptLog(
            model="test-model",
            messages="[]",
            response=json.dumps({"choices": []}),
            agent_name="chat",
            run_id=run_id,
            run_outcome="failed",
            run_reason=reason,
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
            schedule="FREQ=HOURLY;INTERVAL=6",
            expires_at=_t(0, 0, day=20),
            # The one mechanism here with `notify` ON, for two reasons: it is what makes
            # the activity block's `sent · … · price-watch` line a world that CAN happen
            # (a job flagged not to notify cannot have sent the user anything), and it is
            # what pins the notify clause's OTHER direction beside three `notify: off`
            # rows — both states, always, on every live row.
            notify=True,
            created_at=_t(6),
            updated_at=_t(9, 20),
            last_collected_at=_t(9, 14),
        )
        _add_collection(
            session,
            "news-digest",
            description="gather headlines",
            extraction_prompt="1. read the feed",
            schedule="FREQ=HOURLY",
            created_at=_t(6),
            updated_at=_t(6),
            last_collected_at=_t(8),
        )
        _add_collection(
            session,
            "reminder",
            description="one-off reminder",
            extraction_prompt="1. remind",
            schedule="DTSTART:20260711T120000Z\nFREQ=DAILY;COUNT=1",
            max_runs=1,
            created_at=_t(6),
            updated_at=_t(6),
        )
        _add_collection(
            session,
            "old-watch",
            description="a retired watch",
            extraction_prompt="1. old",
            schedule="FREQ=HOURLY",
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
        # The seeded rules collection (#2633) holding two rules — the WITH-rules form:
        # the chat prompt's instruction section renders them (below), and the
        # self-state header itself renders NO standing-rules section at all; every
        # other literal in this file has an empty rules collection, so no
        # standing-rules section renders there either (#2686).
        _add_collection(
            session,
            "rules",
            description=("Standing instructions the user has given: things to always or never do"),
            created_at=_t(6),
            updated_at=_t(6),
        )
        _add_rule(session, "page links", "always include a link when you mention a page", _t(7))
        _add_rule(session, "no metric units", "never use metric units", _t(7, 30))
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
            detail=MutationDetail(changed_fields=["schedule"]),
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


def test_self_state_kitchen_sink_render(db):
    _seed_kitchen_sink(db)
    actual = SelfStateHeader(db, USER).render()
    assert actual == _KITCHEN_SINK


# ── 1b. The mechanisms line renders the stored rule verbatim ──────────────


def test_self_state_schedule_mechanism_render(db):
    """A collection's schedule renders VERBATIM as its cadence clause on the
    mechanisms line (#1857, display form == invocation form), so a stated schedule
    reads at a glance and copies straight back as the ``schedule`` arg — including a
    two-line rule, whose newline renders as ``\\n`` so the line stays a line."""
    with Session(db.engine) as session:
        _add_collection(
            session,
            "twice-daily",
            description="check the peak morning and evening",
            extraction_prompt="1. check the peak.",
            schedule="FREQ=DAILY;BYHOUR=8,20",
            created_at=_t(6),
            updated_at=_t(6),
        )
        _add_collection(
            session,
            "one-off",
            description="check it once, at nine",
            extraction_prompt="1. check it.",
            schedule="DTSTART:20260711T090000Z\nFREQ=DAILY;COUNT=1",
            max_runs=1,
            created_at=_t(6),
            updated_at=_t(6),
        )
        session.commit()
    _add_user(db)
    assert SelfStateHeader(db, USER).render() == _SCHEDULE_MECH


# ── 2. Empty state — a fresh deployment ───────────────────────────────────


def test_self_state_empty_render(db):
    actual = SelfStateHeader(db, None).render()
    assert actual == _EMPTY


def test_self_state_renders_the_mute_state_in_both_directions(db):
    """Whether proactive notifications are going out is ALWAYS stated, either way (#1919).

    Both directions matter for the same reason: the mute tools' descriptions used to be
    the only thing carrying this state, and they conditioned every trigger on "an earlier
    pause" nothing rendered — so the model correctly declined to assume one. A visible
    "on" is as load-bearing as a visible "muted": with no line at all there is a hidden
    state to imagine, and what stops it being imagined is that the real one is always
    there.  Read off the SAME key the notifier gates on, so the render and the delivery
    decision cannot disagree."""
    sender = "+15550000001"

    assert SelfStateHeader.NOTIFICATIONS_ON in SelfStateHeader(db, sender).render()
    assert SelfStateHeader.NOTIFICATIONS_MUTED not in SelfStateHeader(db, sender).render()

    db.users.set_muted(sender)
    muted = SelfStateHeader(db, sender).render()
    assert SelfStateHeader.NOTIFICATIONS_MUTED in muted
    assert SelfStateHeader.NOTIFICATIONS_ON not in muted
    # It says WHICH channel is paused — a mute silences the jobs, never the replies, and
    # a line reading only "notifications are paused" would say Penny had been told to
    # stop talking.
    assert "replies to the user's own messages are unaffected" in muted

    db.users.set_unmuted(sender)
    assert SelfStateHeader.NOTIFICATIONS_ON in SelfStateHeader(db, sender).render()


# ── 3. Budget overflow — the cap line is visible ──────────────────────────


def test_self_state_activity_overflow_render(db):
    with Session(db.engine) as session:
        _add_collection(
            session,
            "watcher",
            description="a watcher",
            extraction_prompt="1. go",
            schedule="FREQ=HOURLY",
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


def test_self_state_archived_heavy_render(db):
    with Session(db.engine) as session:
        _add_collection(
            session,
            "live-watch",
            description="still running",
            extraction_prompt="1. go",
            schedule="FREQ=HOURLY",
            created_at=_t(6),
            updated_at=_t(6),
        )
        for i in range(3):
            _add_collection(
                session,
                f"retired-{i}",
                description=f"retired watch {i}",
                extraction_prompt="1. old",
                schedule="FREQ=HOURLY",
                archived=True,
                created_at=_t(5, 0, day=1),
                updated_at=_t(8, i),
            )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    assert actual == _ARCHIVED_HEAVY


# ── 4a. The instructions line — what a live job reads and follows (#2495) ──


# First, test for a sentence longer than the cap (single-sentence prompt)
def test_self_state_job_instructions_first_sentence_over_cap(db):
    """When the FIRST sentence itself is cut because it's longer than the cap,
    the note reads '(the rest is cut — memory_get(...) shows the whole prompt)'
    instead of '(0 more sentences cut — ...)'."""
    # Create a prompt with a single sentence much longer than the cap
    # (no periods to avoid creating sentence boundaries)
    long_words = "this is a very long instruction with many words that goes on and on "
    long_sentence = "1. " + (long_words * 5) + "done"
    with Session(db.engine) as session:
        _add_collection(
            session,
            "long-instruction",
            description="job with long single sentence",
            extraction_prompt=long_sentence,
            schedule="FREQ=HOURLY",
            created_at=_t(6),
            updated_at=_t(6),
        )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    # Should render with "the rest is cut" not "0 more sentences cut"
    assert "the rest is cut" in actual
    assert "0 more sentences cut" not in actual


def test_self_state_job_instructions_url_with_trailing_punctuation(db):
    """The page URL is stripped of trailing punctuation so 'https://example.com.'
    renders as 'https://example.com' not with the period."""
    with Session(db.engine) as session:
        _add_collection(
            session,
            "url-with-period",
            description="instruction ending with URL followed by period",
            extraction_prompt="1. read the page at https://example.com. then analyze it.",
            schedule="FREQ=HOURLY",
            created_at=_t(6),
            updated_at=_t(6),
        )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    # The URL should appear without the trailing period
    assert "reads https://example.com —" in actual
    assert "reads https://example.com. —" not in actual


def test_self_state_job_instructions_line_shapes(db):
    """Every shape of the hung instructions line (#2495) in one render: a job whose
    prompt names a page renders ``reads <first-url> — "<instructions>"``; a job whose
    prompt fits the character budget renders it whole with no cut note; a job whose
    prompt runs past the budget renders its first sentence plus the cut note, which
    counts the sentences dropped and names the ``memory_get`` hop to the whole prompt —
    while the ``reads`` clause still shows the page that dropped sentence named, since
    the URL is read off the FULL prompt; a prompt with no URL renders the quoted
    instructions alone; and an inert collection (no prompt) renders no second line at
    all, here and nowhere."""
    with Session(db.engine) as session:
        _add_collection(
            session,
            "feed-check",
            description="check the feed",
            extraction_prompt=(
                "1. read https://harborferries.example/feed each morning and write the headlines."
            ),
            schedule="FREQ=HOURLY",
            created_at=_t(6),
            updated_at=_t(6),
        )
        _add_collection(
            session,
            "harbour-watch",
            description="watch the harbour notices",
            extraction_prompt=(
                "1. read the harbour notice board each morning before the first "
                "sailing. 2. find the night sailing's departure time on "
                "https://harborferries.example/seasonal-sailings and write it with the "
                "date you read it on. 3. if the page carries no night sailing at all, "
                "write that instead — an empty result is a result."
            ),
            schedule="FREQ=HOURLY",
            created_at=_t(6),
            updated_at=_t(6),
        )
        _add_collection(
            session,
            "plain-note",
            description="count the boats",
            extraction_prompt="1. count the boats in the harbour.",
            schedule="FREQ=HOURLY",
            created_at=_t(6),
            updated_at=_t(6),
        )
        _add_collection(
            session,
            "keepsakes",
            description="things worth keeping",
            created_at=_t(6),
            updated_at=_t(6),
        )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    assert actual == _INSTRUCTION_SHAPES


# ── 4b. No cap — the whole registry renders, archived names index (#2288) ─


def test_self_state_mechanisms_render_past_the_old_twelve_row_cap(db):
    """More than the old 12-row mechanisms cap all render — nothing hidden
    behind a count, no overflow tail (#2288, code-owner ruling: "list
    everything in her state")."""
    with Session(db.engine) as session:
        for i in range(15):
            _add_collection(
                session,
                f"watch-{i:02d}",
                description=f"watch {i}",
                extraction_prompt="1. go",
                schedule="FREQ=HOURLY",
                created_at=_t(6),
                updated_at=_t(6),
            )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    mechanisms = actual.split("### Jobs\n", 1)[1].split("\n\n", 1)[0]
    rows = [line for line in mechanisms.splitlines() if line.startswith("- ")]
    assert len(rows) == 15
    assert "more" not in mechanisms


def test_self_state_store_map_renders_past_the_old_twenty_row_cap(db):
    """More than the old 20-row store-map cap of live collections all render
    (#2288) — no overflow tail naming a count."""
    with Session(db.engine) as session:
        for i in range(25):
            _add_collection(
                session,
                f"box-{i:02d}",
                description=f"box {i}",
                created_at=_t(6),
                updated_at=_t(6),
            )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    store_map = actual.split("### Your memory\n", 1)[1].split("\n\n", 1)[0]
    rows = [line for line in store_map.splitlines() if line.startswith("- ")]
    assert len(rows) == 25
    assert "more" not in store_map


def test_self_state_archived_line_names_a_collection_and_a_log(db):
    """The store map's compact archived line (#2288, "one compact line naming
    them") names BOTH an archived collection and an archived log, by name —
    the complete index of archived stores, not a remainder."""
    with Session(db.engine) as session:
        _add_collection(
            session,
            "recipe-box",
            description="old recipes",
            archived=True,
            created_at=_t(6),
            updated_at=_t(6),
        )
        _add_log(session, "old-trips", "old trip logs", when=_t(6), archived=True)
        session.commit()
    actual = SelfStateHeader(db, None).render()
    assert "archived: `old-trips`, `recipe-box`" in actual


def test_self_state_no_archived_line_when_nothing_archived(db):
    """No archived collections or logs → no archived line at all — the
    unmarked case stays quiet, exactly like every other empty section."""
    with Session(db.engine) as session:
        _add_collection(
            session,
            "live-box",
            description="still live",
            created_at=_t(6),
            updated_at=_t(6),
        )
        session.commit()
    actual = SelfStateHeader(db, None).render()
    assert "archived:" not in actual


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


def test_activity_run_lines_every_rendered_outcome(db):
    """One run line per rendering RunOutcome — WORKED / FAILED / NO_WORK /
    INCOMPLETE — plus the zero-call form and the singular '1 call' form.
    (CANCELLED is excluded by design; pinned in the exclusion test below.)"""
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


def test_activity_run_lines_carry_writes(db):
    """The four writes-clause shapes in one render: a single-write run names its
    key + collection; a two-write run (exactly the key-sample cap) lists both keys
    with NO ellipsis; a three-write run compacts to a count + a two-key sample WITH
    an ellipsis; a no-write run is byte-identical to today.  The single-write entry
    was CREATED by an earlier run but LAST WRITTEN by this one, proving the clause
    reports the current-value writer (``last_written_by_run_id``), not the
    creator."""
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


def test_activity_run_line_multiple_collections(db):
    """A run that wrote keyed entries to two collections renders one ``· wrote …``
    clause per collection, grouped by name (the join is run-type agnostic — a chat
    turn writing likes + dislikes would render the same way).  Groups render in
    name order (``dislikes`` before ``likes``)."""
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


def test_activity_excludes_cancelled_and_chat_runs(db):
    """The two by-design exclusions, proven against a rendering sibling: a
    CANCELLED collector run (not a real cycle) and a chat run (no outcome, no
    target — already the conversation) are seeded NEWER than a worked sibling,
    yet only the sibling renders."""
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


def test_activity_failed_chat_turn_renders_its_cause_and_run_id(db):
    """The one exception to "chat turns are excluded" (#2355 part D): a chat turn
    that fell back to the honest-account machinery stamps a run_outcome too, and
    gets its own line — interleaved by time with an ordinary collector run, never
    confused with one (no target, no call count — just the typed run id and the
    third-person reason)."""
    with Session(db.engine) as session:
        _add_failed_chat_run(
            session,
            run_id="cc88failed",
            reason="the turn captured no routine to learn",
            when=_t(9, 50),
        )
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
    assert (
        "run cc88failed · 2026-07-11 09:50 UTC · "
        "chat turn failed: the turn captured no routine to learn"
    ) in actual
    # Newest first, same as every other activity source.
    failed_line = actual.index("cc88failed")
    worked_line = actual.index("aa77worked")
    assert failed_line < worked_line


def test_activity_mutation_lines_changed_fields_matrix(db):
    """Every action × actor cell with a changed-fields detail: the multi-field
    'changed a, b' tail, user-run rows naming their run id, system rows none."""
    _seed_mutation_matrix(db, detail_factory=_changed_fields_detail)
    actual = SelfStateHeader(db, None).render()
    assert actual == _MUTATION_CHANGED_FIELDS_MATRIX


def test_activity_mutation_lines_note_matrix(db):
    """Every action × actor cell with a cause-note detail (the system-archive
    policy-reason shape, e.g. a max_runs retire)."""
    _seed_mutation_matrix(db, detail_factory=_note_detail)
    actual = SelfStateHeader(db, None).render()
    assert actual == _MUTATION_NOTE_MATRIX


def test_activity_mutation_lines_no_detail_matrix(db):
    """Every action × actor cell with NO detail payload: the bare line — no
    tail, no dash."""
    _seed_mutation_matrix(db, detail_factory=_no_detail)
    actual = SelfStateHeader(db, None).render()
    assert actual == _MUTATION_BARE_MATRIX


def test_activity_emission_lines_and_exclusions(db):
    """Delivered autonomous sends (#1568) render as ``sent · …`` lines, interleaved
    by time with a run: a body renders whole whatever its length — no cap (#2642) —
    and a direct reply (no mechanism) seeded NEWER than an
    emission is excluded (it is the conversation, not its complement)."""
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
async def test_full_chat_system_prompt_composition(db, tmp_path):
    """The whole chat request's system side (#2745): the STATIC message —
    Identity + the standing rules (the instruction part, right after the voice)
    + Instructions + the conversation map (#2752, empty here since the kitchen
    sink logs no messages), the prefix the server reuses across turns — and
    the DYNAMIC message — the clock, then the self-state header.  The rules
    render in the static message, not the header: where she reads
    instructions rather than state.  The speculative recall / profile /
    inventory are gone."""
    _seed_kitchen_sink(db)
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    agent.config = Config(
        **cast(
            Any,
            {
                **DEFAULT_TEST_CONFIG,
                "signal_api_url": "http://localhost:1",
                "db_path": str(tmp_path / "config.db"),
                "runtime": RuntimeParams(db),
            },
        )
    )
    agent._pending_page_context = None
    agent._current_user = USER
    static = await agent._build_system_prompt(USER)
    expected_static = (
        f"## Identity\n{Prompt.PENNY_IDENTITY}\n\n"
        f"{_STANDING_RULES_INSTRUCTION}\n\n"
        f"## Instructions\n{Prompt.CONVERSATION_PROMPT}"
    )
    assert static == expected_static
    dynamic = agent._dynamic_system_section()
    clock_line, _, state = dynamic.partition("\n\n")
    assert clock_line.startswith("Current date and time: "), clock_line
    assert state == _KITCHEN_SINK


# ── Frozen render literals (filled from the captured actual) ──────────────

_STANDING_RULES_INSTRUCTION = (
    "### Standing rules\n"
    "The user's standing rules: follow each one in everything you send, to the very "
    "end of the message. Where a rule asks for something different from any other "
    "guidance in this prompt — how you look things up, what you do, or how you "
    "write — the rule is what they want.\n"
    "\n"
    "- always include a link when you mention a page\n"
    "- never use metric units"
)

_KITCHEN_SINK = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "- news-digest — active · every hour (rule: `FREQ=HOURLY`) · notify: off · last run "
    "FAILED 2026-07-11 08:00 UTC\n"
    '  "1. read the feed"\n'
    "- old-watch — archived 2026-07-11 08:30 UTC · no runs yet\n"
    "- price-watch — active · every 6 hours (rule: `FREQ=HOURLY;INTERVAL=6`) · notify: on "
    "· expires Sunday, July 19, 2026 at 08:00 PM EDT · last run WORKED 2026-07-11 09:14 "
    "UTC\n"
    '  "1. browse the page"\n'
    "- reminder — active · once, at Saturday, July 11, 2026 at 08:00 AM EDT, then it "
    "retires (rule: `DTSTART:20260711T120000Z FREQ=DAILY;COUNT=1`) · notify: off · "
    "one-shot · no runs yet\n"
    '  "1. remind"\n'
    "\n"
    "### Recent activity\n"
    "change · 2026-07-11 09:20 UTC · price-watch updated by user-run (run 66aa0099) — "
    "changed schedule\n"
    "run 7f3a1b2c · 2026-07-11 09:14 UTC · price-watch → WORKED (3 calls) · wrote 'aurora "
    "deck 2 price' → `price-watch`\n"
    'sent · 2026-07-11 09:05 UTC · price-watch — "Heads up: the price dropped to $42!"\n'
    "change · 2026-07-11 08:30 UTC · old-watch archived by system — max_runs reached (1 of "
    "1)\n"
    "run 88d14e5f · 2026-07-11 08:00 UTC · news-digest → FAILED (2 calls)\n"
    "\n"
    "### Your memory\n"
    "- chat-log (log, 0 entries) — shared conversation log\n"
    "- favorites (collection, 2 entries) — things the user likes\n"
    "- news-digest (collection, 0 entries) — gather headlines\n"
    "- price-watch (collection, 1 entries) — watch a product price\n"
    "- reminder (collection, 0 entries) — one-off reminder\n"
    "- rules (collection, 2 entries) — Standing instructions the user has given: things "
    "to always or never do\n"
    "archived: `old-watch`\n"
    "\n"
    "### About the user\n"
    "- name: Alex\n"
    "- timezone: America/Toronto\n"
    "- location: Toronto, Canada\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and change history, "
    "run_get(run <id>) for one run's tool calls, collection_list(<name>) reads a store's "
    "entries, and find(query=<text>) searches them all by meaning — a collection, a stored "
    "entry, or something said in the conversation."
)

_SCHEDULE_MECH = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "- one-off — active · once, at Saturday, July 11, 2026 at 05:00 AM EDT, then it "
    "retires (rule: `DTSTART:20260711T090000Z FREQ=DAILY;COUNT=1`) · notify: off · "
    "one-shot · no runs yet\n"
    '  "1. check it."\n'
    "- twice-daily — active · every day at 08:00 AM and 08:00 PM (rule: "
    "`FREQ=DAILY;BYHOUR=8,20`) · notify: off · no runs yet\n"
    '  "1. check the peak."\n'
    "\n"
    "### Recent activity\n"
    "(no recent activity)\n"
    "\n"
    "### Your memory\n"
    "- one-off (collection, 0 entries) — check it once, at nine\n"
    "- twice-daily (collection, 0 entries) — check the peak morning and evening\n"
    "\n"
    "### About the user\n"
    "- name: Alex\n"
    "- timezone: America/Toronto\n"
    "- location: Toronto, Canada\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and change history, "
    "run_get(run <id>) for one run's tool calls, collection_list(<name>) reads a store's "
    "entries, and find(query=<text>) searches them all by meaning — a collection, a stored "
    "entry, or something said in the conversation."
)

_EMPTY = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "(no jobs yet)\n"
    "\n"
    "### Recent activity\n"
    "(no recent activity)\n"
    "\n"
    "### Your memory\n"
    "(no stores yet)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and "
    "change history, run_get(run <id>) for one run's tool calls, "
    "collection_list(<name>) reads a store's entries, and find(query=<text>) "
    "searches them all by meaning — a collection, a stored entry, or "
    "something said in the conversation."
)

_OVERFLOW = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "- watcher — active · every hour (rule: `FREQ=HOURLY`) · notify: off · last run WORKED "
    "2026-07-11 09:08 UTC\n"
    '  "1. go"\n'
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
    "+ older activity — run_list(<source>)\n"
    "\n"
    "### Your memory\n"
    "- watcher (collection, 0 entries) — a watcher\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and change history, "
    "run_get(run <id>) for one run's tool calls, collection_list(<name>) reads a store's "
    "entries, and find(query=<text>) searches them all by meaning — a collection, a stored "
    "entry, or something said in the conversation."
)

_ARCHIVED_HEAVY = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "- live-watch — active · every hour (rule: `FREQ=HOURLY`) · notify: off · no runs yet\n"
    '  "1. go"\n'
    "- retired-0 — archived 2026-07-11 08:00 UTC · no runs yet\n"
    "- retired-1 — archived 2026-07-11 08:01 UTC · no runs yet\n"
    "- retired-2 — archived 2026-07-11 08:02 UTC · no runs yet\n"
    "\n"
    "### Recent activity\n"
    "(no recent activity)\n"
    "\n"
    "### Your memory\n"
    "- live-watch (collection, 0 entries) — still running\n"
    "archived: `retired-0`, `retired-1`, `retired-2`\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and change history, "
    "run_get(run <id>) for one run's tool calls, collection_list(<name>) reads a store's "
    "entries, and find(query=<text>) searches them all by meaning — a collection, a stored "
    "entry, or something said in the conversation."
)


_INSTRUCTION_SHAPES = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "- feed-check — active · every hour (rule: `FREQ=HOURLY`) · notify: off · no runs yet\n"
    '  reads https://harborferries.example/feed — "1. read '
    'https://harborferries.example/feed each morning and write the headlines."\n'
    "- harbour-watch — active · every hour (rule: `FREQ=HOURLY`) · notify: off · no runs "
    "yet\n"
    '  reads https://harborferries.example/seasonal-sailings — "1. read the harbour notice '
    "board each morning before the first sailing. (2 more sentences cut — "
    "memory_get('harbour-watch') shows the whole prompt)\"\n"
    "- plain-note — active · every hour (rule: `FREQ=HOURLY`) · notify: off · no runs yet\n"
    '  "1. count the boats in the harbour."\n'
    "\n"
    "### Recent activity\n"
    "(no recent activity)\n"
    "\n"
    "### Your memory\n"
    "- feed-check (collection, 0 entries) — check the feed\n"
    "- harbour-watch (collection, 0 entries) — watch the harbour notices\n"
    "- keepsakes (collection, 0 entries) — things worth keeping\n"
    "- plain-note (collection, 0 entries) — count the boats\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and change history, "
    "run_get(run <id>) for one run's tool calls, collection_list(<name>) reads a store's "
    "entries, and find(query=<text>) searches them all by meaning — a collection, a stored "
    "entry, or something said in the conversation."
)


# ── Shape-matrix literals (filled from the captured actual) ──────────────

_RUN_OUTCOME_MATRIX = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "(no jobs yet)\n"
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
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and "
    "change history, run_get(run <id>) for one run's tool calls, "
    "collection_list(<name>) reads a store's entries, and find(query=<text>) "
    "searches them all by meaning — a collection, a stored entry, or "
    "something said in the conversation."
)

_RUN_WRITES = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "(no jobs yet)\n"
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
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and "
    "change history, run_get(run <id>) for one run's tool calls, "
    "collection_list(<name>) reads a store's entries, and find(query=<text>) "
    "searches them all by meaning — a collection, a stored entry, or "
    "something said in the conversation."
)

_RUN_WRITES_MULTI_COLLECTION = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "(no jobs yet)\n"
    "\n"
    "### Recent activity\n"
    "run d4dualrun · 2026-07-11 09:40 UTC · likes → WORKED (3 calls) · "
    "wrote 'rain' → `dislikes` · wrote 'hiking' → `likes`\n"
    "\n"
    "### Your memory\n"
    "- dislikes (collection, 1 entries) — things disliked\n"
    "- likes (collection, 1 entries) — things liked\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and "
    "change history, run_get(run <id>) for one run's tool calls, "
    "collection_list(<name>) reads a store's entries, and find(query=<text>) "
    "searches them all by meaning — a collection, a stored entry, or "
    "something said in the conversation."
)

_EXCLUSION_RENDER = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "(no jobs yet)\n"
    "\n"
    "### Recent activity\n"
    "run aa77worked · 2026-07-11 09:30 UTC · alpha-watch → WORKED (1 call)\n"
    "\n"
    "### Your memory\n"
    "(no stores yet)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and "
    "change history, run_get(run <id>) for one run's tool calls, "
    "collection_list(<name>) reads a store's entries, and find(query=<text>) "
    "searches them all by meaning — a collection, a stored entry, or "
    "something said in the conversation."
)

_MUTATION_CHANGED_FIELDS_MATRIX = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "(no jobs yet)\n"
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
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and "
    "change history, run_get(run <id>) for one run's tool calls, "
    "collection_list(<name>) reads a store's entries, and find(query=<text>) "
    "searches them all by meaning — a collection, a stored entry, or "
    "something said in the conversation."
)

_MUTATION_NOTE_MATRIX = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "(no jobs yet)\n"
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
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and "
    "change history, run_get(run <id>) for one run's tool calls, "
    "collection_list(<name>) reads a store's entries, and find(query=<text>) "
    "searches them all by meaning — a collection, a stored entry, or "
    "something said in the conversation."
)

_EMISSION_MATRIX = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "(no jobs yet)\n"
    "\n"
    "### Recent activity\n"
    "run rr11worked · 2026-07-11 09:50 UTC · alpha-watch → WORKED (2 calls)\n"
    'sent · 2026-07-11 09:40 UTC · alpha-watch — "A short ping."\n'
    'sent · 2026-07-11 09:20 UTC · beta-watch — "The morning digest is ready with '
    "today's five top headlines for you to skim over coffee\"\n"
    "\n"
    "### Your memory\n"
    "(no stores yet)\n"
    "\n"
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and "
    "change history, run_get(run <id>) for one run's tool calls, "
    "collection_list(<name>) reads a store's entries, and find(query=<text>) "
    "searches them all by meaning — a collection, a stored entry, or "
    "something said in the conversation."
)

_MUTATION_BARE_MATRIX = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "(no jobs yet)\n"
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
    "### About the user\n"
    "(no profile set yet)\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and "
    "change history, run_get(run <id>) for one run's tool calls, "
    "collection_list(<name>) reads a store's entries, and find(query=<text>) "
    "searches them all by meaning — a collection, a stored entry, or "
    "something said in the conversation."
)
