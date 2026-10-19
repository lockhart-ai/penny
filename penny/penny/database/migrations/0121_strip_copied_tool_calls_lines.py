"""Strip the ``[Tool calls: …]`` lines copied into stored outgoing replies.

Type: data

While the conversation history annotated each past reply with a
``[Tool calls: …]`` line (#2634, reverted by #2651), Penny began writing the
line into her own replies — the stored outgoing rows carry one or more copies,
and they sit in the history the chat agent reads every turn. This removes
every whole ``[Tool calls: …]`` line from a stored outgoing row's content.

Shape match (generic content criteria, not known keys — universal):

* the row is OUTGOING (an incoming row quoting the text is the user's words);
* the line is nothing but the annotation — ``[Tool calls: <anything>]``
  (``[Tool calls: none]`` included), optionally with trailing whitespace —
  followed by the line's end.

Anything else on the row stays byte-identical, including a paragraph after a
stripped line. ``promptlog`` is untouched: it is the record of what the model
emitted. Idempotent: a second application finds nothing left to strip.
"""

from __future__ import annotations

import re
import sqlite3

# A whole line that is nothing but the copied annotation, plus its newline.
# ``[ \t]`` only (never ``\s``): under MULTILINE a ``\s`` could swallow a
# newline and match across lines — the shape is strictly one line.
_TOOL_CALLS_LINE = re.compile(r"^\[Tool calls: [^\]]*\][ \t]*\n?", re.MULTILINE)

_OUTGOING = "outgoing"


def strip_tool_call_lines(content: str) -> str:
    """``content`` without its whole ``[Tool calls: …]`` lines, end-trimmed."""
    return _TOOL_CALLS_LINE.sub("", content).rstrip()


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "messagelog" not in tables:
        return
    rows = conn.execute(
        "SELECT id, content FROM messagelog WHERE direction = ? AND content LIKE '%[Tool calls:%'",
        (_OUTGOING,),
    ).fetchall()
    for row_id, content in rows:
        cleaned = strip_tool_call_lines(content)
        if cleaned != content:
            conn.execute("UPDATE messagelog SET content = ? WHERE id = ?", (cleaned, row_id))
    conn.commit()
