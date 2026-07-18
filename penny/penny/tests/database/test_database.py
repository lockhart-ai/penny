"""Tests for the database facade's startup maintenance operations."""

from penny.database import Database


def test_analyze_refreshes_sqlite_statistics_and_logs_completion(tmp_path, caplog):
    caplog.set_level("INFO", logger="penny.database.database")
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    with db.engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO messagelog "
            "(timestamp, direction, sender, content, is_reaction, processed) "
            "VALUES ('2026-01-01T00:00:00Z', 'incoming', 'user', 'one', 0, 0), "
            "('2026-01-01T00:00:01Z', 'incoming', 'user', 'two', 0, 0)"
        )

    db.analyze()

    with db.engine.connect() as connection:
        stat = connection.exec_driver_sql(
            "SELECT 1 FROM sqlite_stat1 "
            "WHERE tbl = 'messagelog' AND idx = 'ix_messagelog_timestamp'"
        ).first()

    assert stat is not None
    assert any(
        record.message.startswith("Database query completed: ANALYZE") for record in caplog.records
    )
