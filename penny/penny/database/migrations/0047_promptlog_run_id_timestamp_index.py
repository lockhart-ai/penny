"""Index promptlog for the addon's prompt-log queries.

The Prompts tab pages runs newest-first
(``GROUP BY run_id ORDER BY MAX(timestamp)``), loads the rows for a page
of runs (``WHERE run_id IN (...)``), and stamps run outcomes
(``WHERE run_id = ? ORDER BY timestamp DESC LIMIT 1``).  A composite
``(run_id, timestamp)`` index serves all three, so they stop
full-scanning the (100k+ row) table — on a real DB this took the run
pagination from ~1-7 s to ~15 ms.

Supersedes the single-column ``ix_promptlog_run_id`` from 0021: the
composite covers every ``run_id`` lookup via its leftmost prefix, so the
old index is redundant write overhead and is dropped.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "promptlog" not in tables:
        return
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_promptlog_run_id_timestamp ON promptlog (run_id, timestamp)"
    )
    conn.execute("DROP INDEX IF EXISTS ix_promptlog_run_id")
