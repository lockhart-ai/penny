"""Add iOS channel registration and outbox tables."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }

    if "ios_device_registration" not in tables:
        conn.execute("""
            CREATE TABLE ios_device_registration (
                device_id INTEGER PRIMARY KEY,
                apns_token TEXT,
                apns_environment TEXT NOT NULL DEFAULT 'sandbox',
                app_version TEXT,
                device_secret_hash TEXT,
                push_enabled INTEGER NOT NULL DEFAULT 1,
                last_seen_at TIMESTAMP NOT NULL,
                token_updated_at TIMESTAMP,
                FOREIGN KEY(device_id) REFERENCES device(id)
            )
        """)
        conn.execute(
            "CREATE INDEX ix_ios_device_registration_apns_token "
            "ON ios_device_registration (apns_token)"
        )

    if "ios_outbox" not in tables:
        conn.execute("""
            CREATE TABLE ios_outbox (
                id INTEGER PRIMARY KEY,
                device_id INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                content TEXT NOT NULL,
                attachments_json TEXT,
                source_type TEXT,
                source_name TEXT,
                source_hint TEXT,
                push_title TEXT NOT NULL,
                push_summary TEXT NOT NULL,
                push_sent_at TIMESTAMP,
                push_error TEXT,
                acked_at TIMESTAMP,
                FOREIGN KEY(device_id) REFERENCES device(id)
            )
        """)
        conn.execute(
            "CREATE INDEX ix_ios_outbox_pending "
            "ON ios_outbox (device_id, created_at) WHERE acked_at IS NULL"
        )
        conn.execute("CREATE INDEX ix_ios_outbox_created_at ON ios_outbox (created_at)")
        conn.execute("CREATE INDEX ix_ios_outbox_acked_at ON ios_outbox (acked_at)")
        conn.execute("CREATE INDEX ix_ios_outbox_source_type ON ios_outbox (source_type)")
        conn.execute("CREATE INDEX ix_ios_outbox_source_name ON ios_outbox (source_name)")
