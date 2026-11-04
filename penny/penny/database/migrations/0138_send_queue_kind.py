"""Add the kind column to send_queue.

Type: schema

Autonomous events beyond a notifying cycle's report reach Penny as system
messages too (#2975): a job retiring (run limit spent or expiry passed) and a
cadence job's run failing for good enqueue a composing row the same way a job
report does — the drainer runs ONE delivery chat turn on the event's facts and
delivers what she writes.  The drainer needs a structural mark for WHICH fact
set to build, and today's row shape cannot express it: ``origin`` is the
delivery lane (a ``CycleTrigger``), and a run-limit one-shot's last run looks
exactly like an ordinary report's.

* ``kind`` — which fact set the composing row's message is built from:
  ``report`` (a notifying cycle's JobReport/SandboxReport, every row that
  exists before this migration), ``job_retired`` (the system archive of a
  run-limit or expired job), or ``job_failed`` (a CADENCE run whose retry
  budget spent out).  NOT NULL DEFAULT 'report', so every existing row is a
  report without a backfill and nothing else reads the column.

Universal: a defaulted column on a table every deployment has; no row is
targeted by key.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }

    if "send_queue" in tables:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(send_queue)").fetchall()]
        if "kind" not in columns:
            conn.execute("ALTER TABLE send_queue ADD COLUMN kind TEXT NOT NULL DEFAULT 'report'")
