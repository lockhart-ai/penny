"""Add the job-report columns to send_queue.

Type: schema

A notifying cycle no longer composes its message in a micro-context (#2656):
``_notify_if_due`` enqueues a row that carries WHERE the report comes from and
the drainer composes the message from it as one chat turn before delivery.

* ``report_run_id`` — the collector cycle's own ``promptlog.run_id`` whose
  written entries the report is built from.  NULL for a row that already
  carries its content (the ordinary queued message); a row with
  ``report_run_id`` set and empty content is a report awaiting composition.
* ``attempts`` — how many times composing this row's report has failed, so the
  retry is bounded and the bounded end is visible.  0 for every existing row.
* ``failed_at`` — stamped when the retry budget is spent without a sendable
  message: a VISIBLE terminal state (the row is kept, never delivered),
  excluded from the pending query structurally, and distinct from
  ``cancelled_at`` — a failed report is a compose that never produced a
  message, where a cancelled one was withheld on purpose (an archived
  collector, a mute).  ``sent_at`` stays NULL on both: neither was ever sent.

Universal: all three are nullable/defaulted columns on a table every
deployment has; no row is targeted by key.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }

    if "send_queue" in tables:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(send_queue)").fetchall()]
        if "report_run_id" not in columns:
            conn.execute("ALTER TABLE send_queue ADD COLUMN report_run_id TEXT")
        if "attempts" not in columns:
            conn.execute("ALTER TABLE send_queue ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
        if "failed_at" not in columns:
            conn.execute("ALTER TABLE send_queue ADD COLUMN failed_at DATETIME")
