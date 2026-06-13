"""Index promptlog for the addon's per-agent prompt-log filter.

Clicking an agent tab runs
``WHERE agent_name = ? GROUP BY run_id ORDER BY MAX(timestamp) DESC``.
Without an ``agent_name`` index this full-scans the (130k+ row) table —
7-15 s on a real DB — and because the query is synchronous SQLite on the
asyncio loop, that scan freezes the whole process (browser, Signal,
scheduler) until it finishes.

``(agent_name, run_id, timestamp)`` lets the filter seek straight to one
agent's rows, group by run_id, and read ``MAX(timestamp)`` as the last
entry per group: ~15 s -> ~1 ms.  The unfiltered query keeps using
``(run_id, timestamp)`` from 0047 (and that index still serves the
``run_id IN (...)`` row fetch and the run-outcome stamp), so both indexes
are kept.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "promptlog" not in tables:
        return
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_promptlog_agent_run_timestamp "
        "ON promptlog (agent_name, run_id, timestamp)"
    )
