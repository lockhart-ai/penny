"""Add run_id to messagelog.

messagelog.run_id: the run whose reply an outgoing row carries (#2630) — stamped
by the framework on the direct-reply path, so the conversation history's per-turn
tool line pairs a chat run with its reply by identity rather than by the
timestamp guess "the first send at or after the run's last logged call".  NULL
for every row already logged (older replies get no tool line, not a guessed one)
and for autonomous sends, which have no run behind them.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }

    if "messagelog" in tables:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(messagelog)").fetchall()]
        if "run_id" not in columns:
            conn.execute("ALTER TABLE messagelog ADD COLUMN run_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_messagelog_run_id ON messagelog (run_id)")
