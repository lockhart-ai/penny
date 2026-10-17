"""Shared machinery for the end-to-end job cases: the one case-local variance feature over the
whole sample's walk, the small readers over `SampleObservation.cycles` every case's STORE
claims use to read the created job's cycles as a union (#2375) — never by a container name, per
the ruling that drops every claim on which collection a job lands on — and the shared seed for
"a routine Penny already knows" (#2455 naive baseline, code-owner ruling).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from penny.constants import ChatPromptType, PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput
from penny.datetime_utils import user_timezone_name
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, seeded_run_id
from penny.tests.eval.utils.cohort import CycleSnapshot, Feature
from penny.tools.base import Tool
from penny.tools.collection_instantiation import render_creation_echo
from penny.tools.models import ToolResult

# The sample's WHOLE walk: the conversation machine's moves through the scripted turns, then
# every created job's own ordered cycle script ("wrote", "quiet", "…+told") — one feature
# because ``TRANSITIONS`` and ``CYCLE_SCRIPT`` both read the identical field under two labels
# that fit neither half of an end-to-end sample on its own.
END_TO_END_WALK = Feature("end-to-end walk", lambda o: o.walk, absent="no move")

# The routine's own name and shape are MEASURED nowhere by these cases, and this is the one
# place that is worth saying rather than leaving as a silent omission: a round that teaches a
# routine from scratch mints a fresh name and shape every sample, the way ``test_teach_arrives_
# whole.py`` reports FRAMER naming spread with ``ROUTINE_NAME`` — but an end-to-end case's
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
        created_by_run_id=run_id,
    )
    require_memory(db, collection).write(
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


__all__ = [
    "END_TO_END_WALK",
    "held_text_at",
    "ran_at",
    "sent_at",
    "seed_known_routine",
]
