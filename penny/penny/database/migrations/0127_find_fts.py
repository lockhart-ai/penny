"""Full-text indexes over stored entries and conversation messages, for find's
keyword leg.

Type: schema

#2774: ``find`` matched by meaning (embedding cosine) only, so a rare word —
above all a name — was often invisible even though it appears verbatim in the
store or the conversation: one query vector blends several names into "pets",
and a message literally containing a specific name can score below an
unrelated generic fact. This adds two FTS5 full-text indexes and the triggers
that keep them in sync, then rebuilds each from the existing rows — the
identical pattern ``0051_promptlog_fts.py`` already uses for the addon's
prompt search.

``memory_entry_fts`` covers a stored entry's ``key`` + ``content`` (collections
and logs). ``messagelog_fts`` covers a conversation message's ``content``.
Collection/log DESCRIPTIONS are deliberately NOT indexed here — #2774 scopes
the keyword leg to stored entries and messages; an object's own resolve-by-
meaning stays embedding-only.

External-content mode (``content='<table>'``) means each index stores only
the inverted index, not a second copy of the text. The one-time ``rebuild``
reads every existing row once; later writes are maintained by the AFTER
INSERT/DELETE/UPDATE triggers — every ``memory_entry``/``messagelog`` write
path (the ORM's ``session.add``/``session.delete`` calls across
``MemoryStore``/``MessageStore``) issues the same underlying SQL regardless of
call site, so the triggers need no corresponding Python-side sync call and
can't be forgotten by a future write path. ``messagelog``'s UPDATE trigger is
defensive (content is never rewritten post-insert today — only ``embedding``/
``mechanism``/``run_id`` are set after the fact) but is guarded on content
change exactly like ``memory_entry``'s and ``promptlog``'s, so it costs
nothing when it never fires.
"""


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }

    if "memory_entry" in tables and "memory_entry_fts" not in tables:
        conn.execute(
            "CREATE VIRTUAL TABLE memory_entry_fts USING fts5("
            "key, content, content='memory_entry', content_rowid='id')"
        )
        conn.execute(
            "CREATE TRIGGER memory_entry_fts_ai AFTER INSERT ON memory_entry BEGIN "
            "INSERT INTO memory_entry_fts(rowid, key, content) "
            "VALUES (new.id, new.key, new.content); END"
        )
        conn.execute(
            "CREATE TRIGGER memory_entry_fts_ad AFTER DELETE ON memory_entry BEGIN "
            "INSERT INTO memory_entry_fts(memory_entry_fts, rowid, key, content) "
            "VALUES ('delete', old.id, old.key, old.content); END"
        )
        conn.execute(
            "CREATE TRIGGER memory_entry_fts_au AFTER UPDATE ON memory_entry "
            "WHEN old.key IS NOT new.key OR old.content IS NOT new.content BEGIN "
            "INSERT INTO memory_entry_fts(memory_entry_fts, rowid, key, content) "
            "VALUES ('delete', old.id, old.key, old.content); "
            "INSERT INTO memory_entry_fts(rowid, key, content) "
            "VALUES (new.id, new.key, new.content); END"
        )
        conn.execute("INSERT INTO memory_entry_fts(memory_entry_fts) VALUES ('rebuild')")

    if "messagelog" in tables and "messagelog_fts" not in tables:
        conn.execute(
            "CREATE VIRTUAL TABLE messagelog_fts USING fts5("
            "content, content='messagelog', content_rowid='id')"
        )
        conn.execute(
            "CREATE TRIGGER messagelog_fts_ai AFTER INSERT ON messagelog BEGIN "
            "INSERT INTO messagelog_fts(rowid, content) VALUES (new.id, new.content); END"
        )
        conn.execute(
            "CREATE TRIGGER messagelog_fts_ad AFTER DELETE ON messagelog BEGIN "
            "INSERT INTO messagelog_fts(messagelog_fts, rowid, content) "
            "VALUES ('delete', old.id, old.content); END"
        )
        conn.execute(
            "CREATE TRIGGER messagelog_fts_au AFTER UPDATE ON messagelog "
            "WHEN old.content IS NOT new.content BEGIN "
            "INSERT INTO messagelog_fts(messagelog_fts, rowid, content) "
            "VALUES ('delete', old.id, old.content); "
            "INSERT INTO messagelog_fts(rowid, content) VALUES (new.id, new.content); END"
        )
        conn.execute("INSERT INTO messagelog_fts(messagelog_fts) VALUES ('rebuild')")

    conn.commit()
