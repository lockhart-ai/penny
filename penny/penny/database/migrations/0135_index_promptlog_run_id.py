"""Index promptlog.run_id.

The column has existed since the run-id ledger landed; it gains an index here
(#2915) because chat history rendering now looks up a past reply's own run by
this id on every turn (``MessageStore.prompts_for_run``) — an unindexed scan
over the whole table on every chat turn would get slower as it grows.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "promptlog" in tables:
        conn.execute("CREATE INDEX IF NOT EXISTS ix_promptlog_run_id ON promptlog (run_id)")
