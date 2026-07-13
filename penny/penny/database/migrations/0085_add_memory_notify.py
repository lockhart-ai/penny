"""Add the ``memory.notify`` emission flag and seed it from ``published``.

Type: schema + data

Emission-as-property (#1557, exposed by #1591's ``collection_create``): a
collection's ``notify`` flag declares that new/changed entries should reach the
user.  It supersedes the ``published`` pub/sub side-channel — but the notifier
consumer that drains ``published`` collections stays alive until #1557 retires it
and wires the run-time notify suffix off this column, so BOTH columns coexist for
now: ``notify`` is the new source of truth, and ``collection_create`` mirrors it
into ``published`` so the live delivery path keeps working in the interim.

This migration adds ``notify`` (default 0) and copies every existing collection's
``published`` value into it, so a collection that already notifies keeps
notifying under the new flag.  Generic criteria only (a column-wide copy) — no
deployment-specific rows are touched, so it is universal.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()]
    if "notify" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN notify INTEGER NOT NULL DEFAULT 0")
    # Seed the new flag from the interim delivery flag: a collection that was
    # published (drained by the notifier) becomes one that notifies.
    conn.execute("UPDATE memory SET notify = published")
    conn.commit()
