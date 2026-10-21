"""Shared machinery for the end-to-end job cases: the one case-local variance feature over the
whole sample's walk, the small readers over `SampleObservation.cycles` every case's STORE
claims use to read the created job's cycles as a union (#2375) — never by a container name, per
the ruling that drops every claim on which collection a job lands on — and the shared seed for
"a routine Penny already knows" (#2455 naive baseline, code-owner ruling).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import timedelta
from typing import NamedTuple

from sqlalchemy import update
from sqlmodel import col

from penny.constants import ChatPromptType, PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.models import MemoryEntry, PromptLog
from penny.datetime_utils import user_timezone_name
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, Seeder, seeded_run_id
from penny.tests.eval.utils.cohort import CycleSnapshot, Feature
from penny.tools.base import Tool
from penny.tools.collection_instantiation import render_creation_echo
from penny.tools.models import ToolResult

# The sample's WHOLE walk: the conversation machine's moves through the scripted turns, then
# every created job's own ordered cycle script ("wrote", "quiet", "…+told") — one feature
# because an end-to-end sample's walk is neither a chat sample's alone nor a collector's
# alone, and no other feature in the suite reads this field.
END_TO_END_WALK = Feature("end-to-end walk", lambda o: o.walk, absent="no move")

# The routine's own name and shape are MEASURED nowhere by these cases, and this is the one
# place that is worth saying rather than leaving as a silent omission: an end-to-end case's
# claims and features never read which collection the job lands on or what the routine is
# called, so there is nothing here that reading the registry would answer differently from
# reading the store the cases already read.


def held_text_at(snapshots: Sequence[CycleSnapshot], index: int) -> str:
    """Every entry held by the created job(s) after cycle ``index`` — the UNION across however
    many jobs the script created, key and content together.

    ``index`` is the arm's own per-cycle position (0 for the first cycle the script's job ran,
    1 for the second, …), not a count: a case names what the store holds AFTER one particular
    cycle, so a reader can tell "the first reading landed" from "the moved reading replaced
    it" without having to recompute an offset."""
    return " ".join(
        entry.text for snapshot in snapshots if snapshot.index == index for entry in snapshot.held
    )


def ran_at(snapshots: Sequence[CycleSnapshot], index: int) -> bool:
    """Whether ANY created job ran a cycle at this position — the guard a negative claim
    needs so it is only answered against a sample that actually reached that cycle.

    Positive claims (``held_text_at`` names a value, ``sent_at`` names a message) read FALSE
    on their own when no snapshot exists at ``index``, because an empty string contains no
    amount and an empty list has no length to equal.  A claim shaped "nothing happened" has
    no such floor — ``not sent_at(..., index)`` is vacuously TRUE of a sample that never got
    there — so it reads this first (#2375's review)."""
    return any(snapshot.index == index for snapshot in snapshots)


def sent_at(snapshots: Sequence[CycleSnapshot], index: int) -> list[str]:
    """Every message the created job(s) put on the send queue DURING cycle ``index`` — the
    union across however many jobs the script created.

    Per-cycle rather than cumulative, because "told once" and "told once more two cycles
    later" are two different findings a running total erases: a watch that notifies on cycle
    one and stays quiet on cycle two must show nothing sent on cycle two, not two messages
    carried forward from the first."""
    return [
        message for snapshot in snapshots if snapshot.index == index for message in snapshot.sent
    ]


# ── "A routine Penny already knows" (#2455 naive baseline, code-owner ruling) ──────────────
#
# There is no skill registry left to teach a routine into, so a job she already knows is
# represented by two things, both written through the SAME chokepoints the runtime itself
# writes through — never a second, harness-only shape:
#
#   (a) the job ALREADY RUNNING — the collection, scheduled, holding the entry it was set
#       up to watch; and
#   (b) the EARLIER CONVERSATION that set it up, in her own logs — the user's ask, her own
#       turn's tool calls (the read, the write, the `collection_configure` naming its
#       prompt), and her reply — so her self-state's recent activity, `log_read`, and
#       `run_get` all read it back exactly as if that turn had really happened.
#
# The seeded run's own model name is cosmetic (no claim reads it) but kept honest rather
# than invented, so a promptlog row a reader opens names the model the suite actually runs.
_SEEDED_MODEL = EVAL_MODELS[0]


def _seeded_response(content: str = "", tool_calls: list[dict] | None = None) -> dict:
    """One model response in the shape ``raw.model_dump()`` persists — the envelope every
    ledger reader walks (``choices[].message``), with ``tool_calls`` present only when the
    draw made calls, exactly as a text reply's dump has it."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"model": _SEEDED_MODEL, "choices": [{"index": 0, "message": message}]}


def _wire_tool_call(call_id: str, tool: str, arguments: dict) -> dict:
    """One tool call as the wire stores it.  ``arguments`` is a JSON STRING — the shape
    ``LoggedToolCall.from_function`` decodes, and the one a dict would silently lose (it
    returns an empty mapping for a non-string, so the call would render argument-less)."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": tool, "arguments": json.dumps(arguments)},
    }


def _tool_result_turn(call_id: str, content: str) -> dict:
    """The tool turn the loop appended for one call — the framed result keyed by the call's
    own id, carrying the framework's structural success stamp, which is how a run's calls
    pair with their outcomes rather than by position."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": content,
        PennyConstants.TOOL_RESULT_SUCCESS_KEY: True,
    }


def _log_chat_step(db: Database, *, run_id: str, messages: list[dict], response: dict) -> None:
    """One step of a seeded chat run — the accumulated conversation as it stood when the
    call was made, and what the model returned."""
    db.messages.log_prompt(
        model=_SEEDED_MODEL,
        messages=messages,
        response=response,
        agent_name=PennyConstants.CHAT_AGENT_NAME,
        prompt_type=ChatPromptType.USER_MESSAGE,
        run_id=run_id,
    )


# Phrases that mark a seeded ask as wanting to be told when the routine's cycles find
# something — the same family the live scripts' own ``_TELLING_CLAUSES`` test for.  The
# seeded job's own ``notify`` must agree with this, or a later turn reads a history that
# asked to be told sitting beside a job that quietly wouldn't, and rationally "fixes" it
# (#2508: the finding behind samples 4 and 13 of ``end-to-end-new-page``, where the model
# flipped the seeded Analogue Pocket job's ``notify`` unprompted because its own earlier ask —
# "let me know if it changes" — contradicted the row).
_TELLING_PHRASES = ("let me know", "tell me", "notify")


def _asks_to_be_told(ask: str) -> bool:
    """Whether this seeded ask asked to be told when something changes — the one fact the
    seeded job's ``notify`` flag must agree with, so the seed never hands the model a
    history that contradicts its own store."""
    lowered = ask.lower()
    return any(phrase in lowered for phrase in _TELLING_PHRASES)


def seed_known_routine(
    db: Database,
    *,
    collection: str,
    description: str,
    url: str,
    extract: str,
    entry_key: str,
    entry_value: str,
    schedule: str,
    ask: str,
    reply: str,
) -> None:
    """Stand up ONE job she already knows, both ways at once (see the module note above):
    the collection already running and holding its entry, and the earlier turn that set it
    up durable in her own logs — the user's ask, her read → write → ``collection_configure``
    calls (each with its real result turn), and her closing reply, all under one run id, so
    a reader of her recent activity, ``log_read``, or ``run_get`` sees exactly what a real
    turn would have left behind.

    The job's prompt is the plain naive shape (#2455): free text naming the page and what
    to find on it, written straight into ``extraction_prompt`` — what the seeded
    ``collection_configure`` call itself names, so the logged call and the row it produced
    agree.

    Each logged tool turn's result text is the REAL render (``Tool.format_result`` over the
    narration + tag every live call gets, bodied with the exact text the create/write path
    itself produces — ``render_creation_echo`` for the configure call), not a hand-written
    approximation — so a reader can't tell this history apart from a live turn's by its
    shape (code-owner review, #2455)."""
    ask_id = db.messages.log_message(PennyConstants.MessageDirection.INCOMING, TEST_SENDER, ask)
    assert ask_id is not None, "the seeded ask failed to log — nothing for the run to link to"
    # The harness's own seeded-run marker (``is_seeded_run``) — every "what did THIS sample
    # do" reader excludes rows under this prefix, which is what keeps the already-running
    # job's entry from reading as something the live turn just stored.
    run_id = seeded_run_id(collection)
    prompt = f"Check {url} for {extract} and save it."
    # The job's own notify must agree with what its seeded ask asked for (#2508) — never a
    # bare default — so the seeded history and the seeded store never contradict each other.
    notify = _asks_to_be_told(ask)

    # The durable state first, through the SAME store methods a live turn calls, so the row
    # and the entry carry the real provenance (`created_by_run_id`, the mutation event, the
    # entry's own `created_by_run_id`/`last_written_by_run_id`) a reader's `memory_get`/
    # recent-activity render reads — and so the logged turns below can quote back the exact
    # row and entry this produced, rather than a description of what they WOULD produce.
    row = db.memories.create_collection(
        collection,
        description,
        extraction_prompt=prompt,
        schedule=schedule,
        notify=notify,
        created_by_run_id=run_id,
    )
    require_memory(db, collection).write_rule_decided(
        [EntryInput(key=entry_key, content=entry_value)], author="chat", run_id=run_id
    )
    db.memories.link_source_message(run_id, ask_id)

    conversation: list[dict] = [{"role": "user", "content": ask}]

    browse_args = {"queries": [url], "extract": extract}
    browse_id = "call-seeded-browse"
    _log_chat_step(
        db,
        run_id=run_id,
        messages=conversation,
        response=_seeded_response(tool_calls=[_wire_tool_call(browse_id, "browse", browse_args)]),
    )
    browse_result = Tool.format_result(
        "browse", browse_args, ToolResult(message=entry_value, success=True, source_urls=[url])
    )
    conversation = [
        *conversation,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_wire_tool_call(browse_id, "browse", browse_args)],
        },
        _tool_result_turn(browse_id, browse_result),
    ]

    write_id = "call-seeded-write"
    write_args = {"collection": collection, "entries": [{"key": entry_key, "content": entry_value}]}
    _log_chat_step(
        db,
        run_id=run_id,
        messages=conversation,
        response=_seeded_response(
            tool_calls=[_wire_tool_call(write_id, "collection_write", write_args)]
        ),
    )
    write_message = f"Wrote 1 entry to '{collection}': {entry_key}."
    write_result = Tool.format_result(
        "collection_write", write_args, ToolResult(message=write_message, mutated=True)
    )
    conversation = [
        *conversation,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_wire_tool_call(write_id, "collection_write", write_args)],
        },
        _tool_result_turn(write_id, write_result),
    ]

    configure_id = "call-seeded-configure"
    configure_args = {
        "collection": collection,
        "description": description,
        "prompt": prompt,
        "schedule": schedule,
        "notify": notify,
    }
    _log_chat_step(
        db,
        run_id=run_id,
        messages=conversation,
        response=_seeded_response(
            tool_calls=[_wire_tool_call(configure_id, "collection_configure", configure_args)]
        ),
    )
    echo = render_creation_echo(row, user_timezone_name(db))
    configure_result = Tool.format_result(
        "collection_configure", configure_args, ToolResult(message=echo, mutated=True)
    )
    conversation = [
        *conversation,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_wire_tool_call(configure_id, "collection_configure", configure_args)],
        },
        _tool_result_turn(configure_id, configure_result),
    ]

    _log_chat_step(db, run_id=run_id, messages=conversation, response=_seeded_response(reply))
    db.messages.log_message(PennyConstants.MessageDirection.OUTGOING, "penny", reply)


# ── A standing job that operates on, rather than a turn that stands one up (#2519) ──────────
#
# ``seed_known_routine`` above is the full shape — a job plus the earlier conversation that
# set it up — for a case whose claims read that conversation back.  A case that only operates
# on an already-running job (switches a notify flag, re-times it, lists several at once, reads
# one's record) needs none of that history: it needs the job's own row, exactly as the
# production chokepoints (``create_collection`` + ``update_collection_metadata``) leave it.
# ``StandingJob`` is that lighter shape, naive throughout: no skill or routine behind it, just
# the job's own plain-text ``program`` — the same free-text shape ``seed_known_routine`` writes
# into ``extraction_prompt`` — so a job built either way reads as one production shape.


class StandingJob(NamedTuple):
    """One collection as a standing job, naive-shape (#2455 naive baseline): no routine
    distilled through a skill pipeline, just the job's own prompt naming the page and what to
    find on it.

    ``container`` is explicit rather than derived — there is no skill name and no bound-values
    tuple left to derive one from, so a case names the collection a job lands on directly."""

    container: str
    url: str
    watched_for: str
    description: str
    schedule: str
    notify: bool
    holdings: tuple[tuple[str, str], ...] = ()

    @property
    def program(self) -> str:
        """The job's own prompt — the plain naive shape ``seed_known_routine`` writes, so a
        job built here and one built there read as the same production shape."""
        return f"Check {self.url} for {self.watched_for} and save it."


_STOOD_UP_RUN = seeded_run_id("stood-up-the-job")


def seed_standing_jobs(*jobs: StandingJob) -> Seeder:
    """Stand each job up through the store's own chokepoints: the container created, then
    configured — its prompt, its cadence, its switch — then whatever it has already gathered.
    Both writes cite ``_STOOD_UP_RUN``, so a later "who changed this, and when" read finds a
    job with a history rather than a row that appeared from nowhere."""

    def seed(db: Database) -> None:
        for job in jobs:
            _stand_up(db, job)

    return seed


def _stand_up(db: Database, job: StandingJob) -> None:
    db.memories.create_collection(job.container, job.description, created_by_run_id=_STOOD_UP_RUN)
    db.memories.update_collection_metadata(
        job.container,
        extraction_prompt=job.program,
        schedule=job.schedule,
        replace_schedule=True,
        notify=job.notify,
        run_id=_STOOD_UP_RUN,
    )
    if job.holdings:
        require_memory(db, job.container).write_rule_decided(
            [EntryInput(key=key, content=content) for key, content in job.holdings],
            author=PennyConstants.MessageAuthor.COLLECTOR,
            run_id=_STOOD_UP_RUN,
        )


def seed_job_cycle(
    db: Database,
    *,
    container: str,
    run_name: str,
    calls: list[dict],
    outcome: RunOutcome,
    summary: str,
    wrote: tuple[str, str] | None = None,
    days_ago: int = 0,
) -> str:
    """Seed one completed collector cycle behind an already-standing job: one prompt-log row
    carrying its own tool calls (built with ``_wire_tool_call``, the same wire shape
    ``seed_known_routine``'s own turns use), its outcome, and — when it wrote — the entry it
    left behind, attributed to this run.  Returns the run id, so a case can assert against the
    exact row this wrote.

    ``days_ago`` backdates the run's prompt rows (and any entry it wrote) — true of the seeded
    world, not of anything the model did — so "last time" means something earlier than the
    sample's own turn rather than the instant the seed runs."""
    run_id = seeded_run_id(run_name)
    db.messages.log_prompt(
        model=_SEEDED_MODEL,
        messages=[],
        response=_seeded_response(tool_calls=calls),
        agent_name=PennyConstants.MessageAuthor.COLLECTOR,
        run_id=run_id,
        run_target=container,
    )
    db.messages.set_run_outcome(run_id, outcome.value, summary)
    if wrote is not None:
        key, content = wrote
        require_memory(db, container).write_rule_decided(
            [EntryInput(key=key, content=content)],
            author=PennyConstants.MessageAuthor.COLLECTOR,
            run_id=run_id,
        )
    if days_ago:
        _backdate_run(db, run_id, days_ago)
    return run_id


def _backdate_run(db: Database, run_id: str, days_ago: int) -> None:
    """Push one run's prompt rows (and the entry it wrote) back by ``days_ago`` — true of the
    seeded world, not of anything the model did.  A raw update because no public store method
    takes an explicit timestamp for a logged run."""
    stamp = db.clock.now() - timedelta(days=days_ago)
    with db.get_session() as session:
        session.exec(
            update(PromptLog).where(col(PromptLog.run_id) == run_id).values(timestamp=stamp)
        )
        session.exec(
            update(MemoryEntry)
            .where(col(MemoryEntry.last_written_by_run_id) == run_id)
            .values(created_at=stamp)
        )
        session.commit()


__all__ = [
    "END_TO_END_WALK",
    "StandingJob",
    "held_text_at",
    "ran_at",
    "seed_job_cycle",
    "seed_known_routine",
    "seed_standing_jobs",
    "sent_at",
]
