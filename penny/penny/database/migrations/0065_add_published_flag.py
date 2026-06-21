"""Add the ``published`` pub/sub flag to ``memory``.

Type: schema

``published`` marks a collection as a consumable stream: its new entries are
drained by a downstream consumer (the notifier, and later others) via
``read_published_latest``, each consumer tracking its own cursor.  It is
orthogonal to ``inclusion``/``recall`` (those govern the chat agent's ambient
recall, not notification) — a collection can be silent in chat yet published, or
surfaced in chat yet not published.

Defaults to 0 (opt-in).  The data backfill that flips specific collections to
published (and strips their in-prompt ``send_message`` steps) lives in a later
data migration, once the consumer side exists.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()]
    if "published" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN published INTEGER NOT NULL DEFAULT 0")
    conn.commit()
