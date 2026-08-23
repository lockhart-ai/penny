"""The world the NL-dispatch stories are answered in, and the probe that asserts it.

The three dispatch modules (email, generate_image, choose) make the SAME two claims about
the world before a turn is driven, so the claims live here once rather than three times:

  * the tool the case is about is REGISTERED on the chat surface — for the config-gated
    tools the hook that mocks the boundary is the same hook that registers them, so a hook
    that silently failed would score every sample a dispatch miss the model was never
    offered; and
  * the registry holds NO COLLECTION — nothing has been pre-seeded since migration 0108,
    which is what makes each module's "nothing was created" check a TOTAL reading of what
    the turn touched rather than a sample of it.

**The second claim is about COLLECTIONS, never about every memory row.**  Migration 0026
seeds four system LOG markers — ``browse-results``, ``collector-runs``, ``penny-messages``,
``user-messages`` — into every database that has ever migrated, and they are permanent
furniture: a fresh deployment has them, so a probe that demanded an empty ``memory`` table
was asserting something no Penny has ever been true of.  It failed all eight dispatch cases
at the probe on their first live run, before a single sample was driven, which is why the
shape read lives in ONE function here and why the pins in ``tests/test_eval_harness.py``
run it against a real migrated database inside ``make check``.

Nothing in here is eval-marked: it is plain helper code the ``eval`` modules and the
deterministic harness pins both import, the same way ``fixtures.py`` and ``report.py`` are.
"""

from __future__ import annotations

from collections.abc import Iterable

from penny.database import Database
from penny.database.memory import MemoryType
from penny.database.models import MemoryRow
from penny.penny import Penny


def collection_rows(db: Database) -> list[MemoryRow]:
    """Every COLLECTION-shaped memory — the registry as a dispatch story means it.

    The ``memory`` table holds both shapes, so ``list_all`` alone answers a different
    question: it includes the migration-0026 system log markers, which are permanent and
    present in every migrated database.  Reading the shape off the row's own ``type``
    column keeps this the runtime's definition rather than a list of names to maintain — a
    new system log would join the table without anything here needing to know."""
    return [row for row in db.memories.list_all() if row.type == MemoryType.COLLECTION]


def collection_names(db: Database) -> list[str]:
    """The names of every collection-shaped memory, sorted — what a probe or a rationale
    prints when it has to say what the registry holds."""
    return sorted(row.name for row in collection_rows(db))


def assert_no_collections(db: Database, case_id: str) -> None:
    """The registry holds no COLLECTION.

    The system log markers are deliberately not counted: they are seeded by migration 0026
    into every database, so counting them would make this assertion unsatisfiable — which
    is exactly the bug it now pins."""
    held = collection_names(db)
    assert not held, f"{case_id}: the world must hold no collection, it holds {held}"


def assert_surface_carries(penny: Penny, case_id: str, tools: Iterable[str]) -> None:
    """The chat surface carries every tool the case's story dispatches to.

    Read off the real ``get_tools`` surface rather than off the hook that installed it, so
    a builder that was wired but produced nothing is caught here rather than read as the
    model declining to call it."""
    surface = {tool.name for tool in penny.chat_agent.get_tools()}
    missing = sorted(set(tools) - surface)
    assert not missing, (
        f"{case_id}: the chat surface must carry {missing} — it carries {sorted(surface)}"
    )


def assert_dispatch_world(penny: Penny, case_id: str, tools: Iterable[str]) -> None:
    """Both claims, in the order a failure is most legible: the surface the case needs,
    then the registry the case's state checks are read against."""
    assert_surface_carries(penny, case_id, tools)
    assert_no_collections(penny.db, case_id)
