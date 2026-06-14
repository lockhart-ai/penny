"""Quality-correction contracts — the self-correcting collector (Phase-3 prototype).

A collector whose job is to review Penny's own recent behaviour (the
``collector-runs`` and ``penny-messages`` logs) against each collection's
``intent`` — the user's own words for what it should do — and rewrite the
``extraction_prompt`` of whichever collection has drifted, then tell the user.

The candidate quality prompt lives inline here (iterate until the pass-rate is
solid, then graduate it into a migration that seeds a real ``quality``
collection).  This drives it through the REAL collector: seed a synthetic
``quality`` collection whose extraction_prompt IS the candidate, seed one
synthetic suspect collection (intent + a drifted prompt) plus the logs of its
misbehaviour, then ``run_for("quality")`` and check the persisted correction.

  rebroadcast  — intent "one fresh thought, never repeat"; a digest re-sent →
                 rewrite the prompt (loose bar: any material corrective change).
  silent-drift — intent "never ping me"; still sends updates → drop send_message.
  healthy      — behaviour matches intent → change nothing.
"""

from __future__ import annotations

from typing import cast

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory_store import Inclusion, LogEntryInput, RecallMode
from penny.tests.eval.conftest import CollectorScorer

pytestmark = pytest.mark.eval

_QUALITY = "quality"

# The quality flow is the hardest multi-hop cycle (read two logs → diagnose →
# read metadata → rewrite → notify), and the candidate prompt is an unshipped
# PROTOTYPE.  gpt-oss clears it only some of the time per sample, so the two
# corrective cases use a looser bar than the suite default — this is the surface
# you iterate the candidate prompt against (raise it as the prompt improves).
_PROTOTYPE_PASS_RATE = 0.6

# ── Candidate quality-collector prompt (iterate here, then graduate) ─────────

_CANDIDATE_PROMPT = (
    "You are Penny's quality agent.  Each cycle you review your own recent "
    "behavior and fix the ONE collection that has drifted most from what the "
    "user asked of it — then tell the user what you changed.\n\n"
    "A collection's `intent` is the user's own words for what it should do — "
    "the spec.  Its `extraction_prompt` is how it tries to do it.  Your job is "
    "to catch where the prompt (or the behavior it produces) no longer serves "
    "the intent, and rewrite the prompt to match.  The intent is fixed — you "
    "can never change it; you change the prompt to honour it.\n\n"
    "Sequence:\n"
    '1. log_read_recent("collector-runs", window_seconds=86400) — what your '
    "collectors actually did.\n"
    '2. log_read_recent("penny-messages", window_seconds=86400) — what you '
    "actually sent the user.\n"
    "3. Look for ONE concrete problem in that behavior: a message the user "
    "didn't ask for, the same thing sent twice, a collection acting against "
    "its stated intent.  If nothing looks wrong, call done() and change "
    "nothing.\n"
    "4. collection_metadata(<the suspect collection>) — read its intent and "
    "its current extraction_prompt.\n"
    "5. Find the step in the extraction_prompt that causes the problem, then "
    "call collection_update(name=<collection>, extraction_prompt=<the full "
    "corrected body>) — remove or fix that one step and keep every other step "
    "intact.  Match the prompt to the intent; never weaken the intent to "
    "excuse the prompt.  Diagnosing which step is to blame:\n"
    "   - Repeats (the same thing sent more than once): look at any step that "
    "reads your OWN past output — `penny-messages`, your recent replies, things "
    "you already sent.  That read exists only to CHECK what you've already said "
    "so you can AVOID repeating it — its result is never a source of things to "
    "send.  If the cycle turned what it read there into a new outgoing message, "
    "that is the bug.  Fix it either way: drop the read step if nothing needs "
    "it, OR keep it but make the prompt explicit that its result is only for "
    "avoiding repeats and must never itself be sent to the user.\n"
    "   - Unwanted pings (intent says stay silent / never notify): delete the "
    "`send_message` step from the body.\n"
    "6. send_message(content=...) — REQUIRED whenever you changed a prompt: "
    "one or two sentences telling the user which collection you fixed, what was "
    "going wrong, and what you changed, so they can correct you if needed.\n"
    "7. done().\n\n"
    "Only act on a clear, current contradiction between behavior and intent.  "
    "If the recent behavior already matches every collection's intent, change "
    "nothing and call done()."
)

# ── Synthetic suspect collections (intent + a drifted extraction_prompt) ─────

_DIGEST_PROMPT = (
    "Share one fresh daily digest thought.\n"
    '1. log_read_recent("penny-messages") — re-read what you sent so you do not '
    "repeat yourself.\n"
    "2. Compose a short digest of the latest items.\n"
    "3. send_message the digest.\n"
    "4. done()."
)
_SILENT_DRIFT_PROMPT = (
    "Collect espresso equipment worth considering.\n"
    "1. browse(...) for new espresso gear; read actual pages.\n"
    '2. collection_write("espresso-gear", entries=[...]).\n'
    "3. If the write succeeded, send_message: one-sentence 'found a new item' + URL.\n"
    "4. done()."
)
_HEALTHY_PROMPT = (
    "Collect houseplant care tips.\n"
    "1. browse(...) for fresh houseplant-care advice; read pages.\n"
    '2. collection_write("houseplant-care", entries=[...]).\n'
    "3. If a genuinely new tip was written, send_message one sentence + URL.\n"
    "4. done()."
)


def _seed_quality(db: Database) -> None:
    db.memories.create_collection(
        _QUALITY,
        "Reviews Penny's own runs and messages and corrects collection prompts "
        "that have drifted from their intent",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt=_CANDIDATE_PROMPT,
    )


def _seed(*, suspect: str, description: str, intent: str, prompt: str, runs, penny_msgs):
    """Seeder: the quality collection + one suspect collection + its log trail."""

    def _apply(db: Database) -> None:
        _seed_quality(db)
        db.memories.create_collection(
            suspect,
            description,
            Inclusion.RELEVANT,
            RecallMode.RECENT,
            extraction_prompt=prompt,
            intent=intent,
        )
        db.memories.append(
            PennyConstants.MEMORY_COLLECTOR_RUNS_LOG,
            [LogEntryInput(content=entry) for entry in runs],
            author="collector",
        )
        db.memories.append(
            PennyConstants.MEMORY_PENNY_MESSAGES_LOG,
            [LogEntryInput(content=entry) for entry in penny_msgs],
            author="penny",
        )

    return _apply


def _snapshot(suspect: str):
    def _take(db: Database) -> str:
        memory = db.memories.get(suspect)
        return (memory.extraction_prompt or "") if memory else ""

    return _take


def _score_update(suspect: str, forbidden: str | None) -> CollectorScorer:
    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        original = cast(str, before)
        memory = db.memories.get(suspect)
        new_prompt = (memory.extraction_prompt or "") if memory else ""
        fails = []
        if new_prompt == original:
            fails.append(f"did not change {suspect!r}'s extraction_prompt")
        elif forbidden is not None and forbidden in new_prompt:
            fails.append(f"corrected prompt still contains the offending {forbidden!r} step")
        elif len(new_prompt) < 80:
            fails.append(f"corrected prompt looks gutted ({len(new_prompt)} chars)")
        if not sent:
            fails.append("did not message the user about the change")
        return fails

    return _score


def _score_no_op(suspect: str) -> CollectorScorer:
    def _score(db: Database, before: object, sent: list[str]) -> list[str]:
        original = cast(str, before)
        memory = db.memories.get(suspect)
        new_prompt = (memory.extraction_prompt or "") if memory else ""
        if new_prompt != original:
            return [f"over-corrected a healthy collection ({suspect!r})"]
        return []

    return _score


# ── Cases ───────────────────────────────────────────────────────────────────


async def test_rebroadcast(collector_eval) -> None:
    suspect = "daily-digest"
    await collector_eval(
        case_id="quality-rebroadcast",
        collection=_QUALITY,
        seed=_seed(
            suspect=suspect,
            description="A once-daily digest of fresh items worth a heads-up.",
            intent="Once per cycle, share exactly one fresh thought I haven't seen "
            "before, and never resend something you've already sent me.",
            prompt=_DIGEST_PROMPT,
            runs=[
                "[daily-digest] Sent the daily digest and marked it shared.",
                "[daily-digest] Sent the daily digest.",
            ],
            penny_msgs=[
                "Daily digest — new co-op title announced, a reprint, and a sale.",
                "Daily digest — new co-op title announced, a reprint, and a sale.",
            ],
        ),
        snapshot=_snapshot(suspect),
        score=_score_update(suspect, forbidden=None),
        min_pass_rate=_PROTOTYPE_PASS_RATE,
    )


async def test_silent_drift(collector_eval) -> None:
    suspect = "espresso-gear"
    await collector_eval(
        case_id="quality-silent-drift",
        collection=_QUALITY,
        seed=_seed(
            suspect=suspect,
            description="A quiet running list of espresso equipment worth considering.",
            intent="Keep a quiet running list of espresso equipment worth considering "
            "— never ping me about it, I'll check the list myself.",
            prompt=_SILENT_DRIFT_PROMPT,
            runs=["[espresso-gear] wrote 2 entries and sent an update about a new grinder."],
            penny_msgs=["Found a new espresso grinder: the Niche Zero clone, $300."],
        ),
        snapshot=_snapshot(suspect),
        score=_score_update(suspect, forbidden="send_message"),
        min_pass_rate=_PROTOTYPE_PASS_RATE,
    )


async def test_healthy(collector_eval) -> None:
    suspect = "houseplant-care"
    await collector_eval(
        case_id="quality-healthy",
        collection=_QUALITY,
        seed=_seed(
            suspect=suspect,
            description="A list of houseplant-care tips, with a ping on genuinely new ones.",
            intent="Keep a list of houseplant-care tips and ping me when you find a "
            "genuinely new one.",
            prompt=_HEALTHY_PROMPT,
            runs=["[houseplant-care] wrote 1 new entry and sent one update about watering."],
            penny_msgs=["New houseplant tip: bottom-water pothos weekly to avoid root rot."],
        ),
        snapshot=_snapshot(suspect),
        score=_score_no_op(suspect),
    )
