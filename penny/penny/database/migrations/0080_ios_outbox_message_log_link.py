"""Link iOS outbox rows to their canonical message-log records."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if not {"ios_outbox", "messagelog", "device"}.issubset(tables):
        return

    columns = {row[1] for row in conn.execute("PRAGMA table_info(ios_outbox)").fetchall()}
    if "message_log_id" not in columns:
        conn.execute(
            "ALTER TABLE ios_outbox ADD COLUMN message_log_id INTEGER REFERENCES messagelog(id)"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_ios_outbox_message_log_id ON ios_outbox (message_log_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_messagelog_device_timestamp_id "
        "ON messagelog (device_id, timestamp, id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_messagelog_sender_timestamp_id "
        "ON messagelog (sender, timestamp, id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_messagelog_recipient_timestamp_id "
        "ON messagelog (recipient, timestamp, id)"
    )
    # Older iOS sends stored the outbox ID in messagelog.external_id. Recover
    # that relationship before the client starts using message-log IDs for
    # history deduplication.
    conn.execute(
        """
        UPDATE ios_outbox
        SET message_log_id = (
            SELECT ml.id
            FROM messagelog AS ml
            WHERE ml.direction = 'outgoing'
              AND ml.external_id = CAST(ios_outbox.id AS TEXT)
              AND (
                  ml.device_id = ios_outbox.device_id
                  OR ml.recipient = (
                      SELECT d.identifier FROM device AS d WHERE d.id = ios_outbox.device_id
                  )
              )
            ORDER BY ml.id DESC
            LIMIT 1
        )
        WHERE ios_outbox.message_log_id IS NULL
        """
    )
    conn.commit()
