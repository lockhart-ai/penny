"""Add updated_at column to the memory table."""


def up(conn) -> None:
    columns = [row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()]
    if "updated_at" not in columns:
        default = "'1970-01-01 00:00:00'"
        conn.execute(
            f"ALTER TABLE memory ADD COLUMN updated_at TIMESTAMP NOT NULL DEFAULT {default}"
        )
        # Backfill: use last_collected_at where available, otherwise created_at.
        conn.execute("""
            UPDATE memory
            SET updated_at = COALESCE(last_collected_at, created_at)
        """)
