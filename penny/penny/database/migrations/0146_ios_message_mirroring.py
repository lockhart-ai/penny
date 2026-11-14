"""Support early user-message sync and a single delivery per iOS device.

Type: schema
"""

import sqlite3

from penny.database.ios_history import SEED_IOS_HISTORY


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(messagelog)")}
    if not columns:
        return
    if "client_message_id" not in columns:
        conn.execute("ALTER TABLE messagelog ADD COLUMN client_message_id VARCHAR")
    if "turn_pending" not in columns:
        conn.execute("ALTER TABLE messagelog ADD COLUMN turn_pending BOOLEAN NOT NULL DEFAULT 0")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_messagelog_device_client_message "
        "ON messagelog (device_id, client_message_id)"
    )
    outbox_columns = {row[1] for row in conn.execute("PRAGMA table_info(ios_outbox)")}
    if not outbox_columns:
        return
    if "legacy_id" not in outbox_columns:
        conn.execute("ALTER TABLE ios_outbox ADD COLUMN legacy_id BOOLEAN NOT NULL DEFAULT 1")
    # Preserve legacy IDs (including IDs cached by old clients) and retire
    # duplicate deliveries while retaining the most useful delivery record.
    conn.execute(
        """
        WITH duplicates AS (
            SELECT id, ROW_NUMBER() OVER (
                PARTITION BY device_id, message_log_id
                ORDER BY (acked_at IS NULL), (push_sent_at IS NULL), id
            ) AS position
            FROM ios_outbox WHERE message_log_id IS NOT NULL
        )
        UPDATE ios_outbox
        SET message_log_id = NULL, acked_at = COALESCE(acked_at, CURRENT_TIMESTAMP)
        WHERE id IN (SELECT id FROM duplicates WHERE position > 1)
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_ios_outbox_device_message "
        "ON ios_outbox (device_id, message_log_id)"
    )
    if conn.execute("PRAGMA table_info(ios_device_registration)").fetchone() is not None:
        conn.execute(SEED_IOS_HISTORY, {"device_id": None})
