"""What the memory stories read back, and the two-source world three of them share.

Two kinds of thing, both used from more than one file since story 15's teach cases
moved to ``chat/learn/``:

  * the readers a story scores on — which entries THIS run wrote, the pages it
    fetched, the state the machine landed in, and the advisories that report the walk
    and the routing without scoring them; and
  * the two-source vocabulary — the tokens each source's page is recognised by, and
    the ask the learn round closes on.
"""

from __future__ import annotations

from penny.constants import PennyConstants
from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.memory import MemoryType
from penny.database.models import MemoryEntry
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import (
    Check,
    is_seeded_run,
    routing_clean,
)

# The enacting-tool set is read from the suite's shared fixtures, not restated here: the
# state machine's elicitation edge asks the same question of a turn (nothing acted on
# before it was taught), and one policy in two copies is two contracts.
# Standing a ROUND up before the measured turn is the transition suite's idiom, read from
# where that suite declares it rather than restated here: a seeded machine state, a seeded
# conversation turn and a seeded ledger row are one shape, and a second copy of it would be
# a second contract free to drift from the one every edge case is measured against.
#
# Its PROBES are deliberately restated instead (``_assert_parked_in_elicit`` below): that
# suite's are keyed to its own ``_ElicitRound`` case type, which this file has no shape
# for.  What a probe asserts is the seed it stands beside, so the honest cost of not
# widening a neighbour's fixture type is one restated probe, named here rather than left
# for a reader to notice.

_FAMILY = "chat-memory"


# ── Reading what the sample did ──────────────────────────────────────────────


def _written_by_this_run(entry: MemoryEntry) -> bool:
    """Whether THIS sample put an entry's current value there — created by a live run,
    or last rewritten by one.  Both stamps, because an edit of a seeded entry moves
    only ``last_written_by_run_id``."""
    stamps = (entry.created_by_run_id, entry.last_written_by_run_id)
    return any(stamp is not None and not is_seeded_run(stamp) for stamp in stamps)


def _entries_this_run_wrote(db: Database) -> list[tuple[str, MemoryEntry]]:
    """``(collection, entry)`` for every COLLECTION entry this sample wrote, wherever
    it landed — the run-id stamp answering "what did she store", so a case never has
    to guess which container a framed round used.

    Collections only: the browse log carries the fetched page, and counting that as a
    stored fact would let "she wrote it down" pass on a run that only read a page."""
    written: list[tuple[str, MemoryEntry]] = []
    for row in db.memories.list_all():
        if row.type != MemoryType.COLLECTION:
            continue
        memory = db.memory(row.name)
        entries = memory.read_all() if memory is not None else []
        written += [(row.name, entry) for entry in entries if _written_by_this_run(entry)]
    return written


def _normalize(text: str) -> str:
    """Fold the typography gpt-oss sprinkles into its output so a SEMANTIC substring
    probe isn't defeated by cosmetics: unicode hyphens → '-', nbsp/zero-width/narrow
    spaces → ' ', bold markers stripped, curly quotes straightened, lowercased.  (A
    0/N from an un-normalized probe is a scorer bug — the model wrote 'co‑op' /
    'Mist​forge', semantically right.)"""
    folded = text.lower()
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        folded = folded.replace(dash, "-")
    for space in ("\xa0", "​", " ", " "):
        folded = folded.replace(space, " ")
    for source, target in (("’", "'"), ("“", '"'), ("”", '"'), ("*", "")):
        folded = folded.replace(source, target)
    return folded


def _entry_text(entry: MemoryEntry) -> str:
    """An entry's KEY and CONTENT, normalized and joined — the probe for "did this
    fact land here", robust to which half the model put the fact in (one measured
    sample keyed ``mistforge_tactics`` and stylized the body, so contents alone
    missed it)."""
    return _normalize(" ".join(text for text in (entry.key, entry.content) if text))


def _pages_fetched(db: Database) -> list[MemoryEntry]:
    """Every page this sample read — the browse log's recent window."""
    return require_memory(db, PennyConstants.MEMORY_BROWSE_RESULTS_LOG).read_recent(
        window_seconds=3600, cap=None
    )


# ── The advisory rows every case carries ─────────────────────────────────────


def _landed_state(db: Database) -> str | None:
    latest = db.machine.latest_transition()
    return latest.to_state if latest else None


def _walked(db: Database) -> str:
    """The machine's walk this sample, oldest move first — ``idle→learn, learn→apply``."""
    moves = reversed(db.machine.recent_transitions(limit=20))
    return ", ".join(f"{move.from_state}→{move.to_state}" for move in moves) or "no move"


def _landing_advisory(
    db: Database, expected: ConversationState | None = None, *, scored: bool = False
) -> Check:
    """Where the machine ended up, reported beside the story's own checks.

    A turn carrying an instruction lands in ``learn`` and mints a routine at run end;
    a question lands in idle.  Which of those a given phrasing is belongs to the state
    definitions rather than to a memory story, so this row REPORTS the landing (and
    the whole walk) and only ``expected`` — set where the story does turn on it —
    makes it a verdict.

    ``scored`` promotes that verdict into the graded denominator, for the one story whose
    contract only EXISTS in the state it names (#1989): the learn close is a claim about
    the reply that ends a learn round, so where the machine landed is that story's
    precondition, and reporting it unscored beside scored reply checks put the accurate
    signal out of the score and the misleading one in it."""
    landed = _landed_state(db)
    ok = landed == expected.value if expected is not None else landed is not None
    return Check(
        "calls: where the machine landed",
        ok,
        rationale=f"walked {_walked(db)}",
        scored=scored,
        kind="spine",
    )


def _routing_advisory(db: Database) -> Check:
    return Check(
        "calls: clean routing (no re-rolled draw or continue nudge)",
        routing_clean(db),
        scored=False,
        kind="proc",
    )


# Tokens that exist ONLY on one page, so a stored copy names which source it came from
# and a fabricated entry matches neither.
_FOXES_TOKENS = ("brandt", "aurelio", "goalie")
_SEALS_TOKENS = ("volk", "petra", "player development")


def _carries(db: Database, tokens: tuple[str, ...]) -> bool:
    """Whether any entry this run wrote carries one of a page's own tokens."""
    written = [_entry_text(entry) for _, entry in _entries_this_run_wrote(db)]
    return any(token in text for text in written for token in tokens)


# ONE demonstration in five wordings.  They pool: phrasing contributes ~0.05 of the spread while
# model stochasticity carries the rest.  They are still five because phrasings are a COVERAGE
# mechanism — measured, four scored H = 0.00, 0.52, 0.00, 0.00, which pools to 0.18 and hides
# the one that came apart.
#
# A DEMONSTRATION, not a request, and that distinction is the whole case.  What stood here was a
# flowing sentence with four requirements embedded in it — "go to X and Y, pull out the trades
# and signings from each, and keep the headline plus a short blurb in a team news list for me" —
# which asks the model to decide what the STEPS even are before it can enact them, and it decided
# differently nearly every time: 13 distinct tool sequences in 18 samples against 60-80% modal
# share on every one-source learn case in the suite.
#
# The turn is what a user actually says when asked to walk through one pass, and it answers the
# three questions the seeded round asked, in order: which pages, what counts, what to keep.  It
# references "those two news pages" rather than retyping the URLs, because a real user does not
# repeat themselves and the referent is right there in the turn before.  Same shape as the
# sibling two-source case's turn 2, which is where it came from.
LEARN_CLOSE_ASK = (
    "sure: 1. go to those two news pages 2. pull out any trades, signings, "
    "or injuries — skip game scores 3. remember the title plus a short "
    "blurb for each"
)
