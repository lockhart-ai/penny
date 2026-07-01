"""Delete memory entries corrupted by a gpt-oss degeneration collapse.

Type: data

gpt-oss occasionally collapses mid-generation into a run of ``.`` / ``…`` / ``?``
("...??…?..?????") when its context grows large.  Before the agent loop learned to
discard + re-roll that output, a collapse could land inside a ``collection_write`` /
``update_entry`` argument and get stored — a poisoned entry that reads as garbage
and, when later loaded into context, nudges the next run toward collapsing too.

This is a one-time cleanup of entries already written that way.  It's a *generic
content-shape* deletion (any gpt-oss deployment can accumulate these — it targets
no deployment-specific key), so it belongs in a universal migration.  Fresh
installs run it against an empty ``memory_entry`` and delete nothing; going
forward the loop guard and the corpus write gate keep new poison out, so this
never needs to run again.

The regex is a frozen copy of ``is_degenerate_run`` in ``penny/text_validity.py``
(migrations stay self-contained — they can't couple to evolving app code — so the
canonical detector is duplicated here deliberately, not imported).
"""

import re

# NBSP / narrow-NBSP are already matched by ``\s``; the rest are the zero-width
# space, soft hyphen, and hyphen/dash family gpt-oss laces through a collapse.
_DEGEN_SEP = r"\s\u200b\u00ad\u2010\u2011\u2013\u2014"

_DEGENERATE_RUN_RE = re.compile(
    r"…{3,}"
    r"|(?:…|\.\.\.)[" + _DEGEN_SEP + r"?!.]*(?:…|\.\.\.)"
    r"|[.?!]{2,}[" + _DEGEN_SEP + r"]+[.?!…]{2,}"
    r"|[.…]{2,}[?!]{2,}|[?!]{2,}[.…]{2,}"
    r"|[.…?]{5,}"
)


def _is_degenerate(text):
    return bool(text) and bool(_DEGENERATE_RUN_RE.search(text))


def up(conn):
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory_entry" not in tables:
        return
    doomed = [
        row[0]
        for row in conn.execute("SELECT id, key, content FROM memory_entry").fetchall()
        if _is_degenerate(row[2]) or _is_degenerate(row[1])
    ]
    if doomed:
        conn.executemany(
            "DELETE FROM memory_entry WHERE id = ?", [(entry_id,) for entry_id in doomed]
        )
    conn.commit()
