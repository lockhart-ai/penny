"""Strip the terminal bare-``done()`` step from stored extraction_prompts.

Type: data

The terminal ``done()`` is assembly's now (#1557 revision): ``Collector`` injects
it as the final numbered step of every composed prompt (after the notify steps
when the collection notifies), so a stored ``extraction_prompt`` must not carry
its own — a leftover mid-list ``8. done().`` would tell the model to end the run
before the injected steps, and a trailing one would double the terminal.  A skill
render can never produce a ``done()`` step (the chat ledger has no ``done`` tool —
a chat turn ends in text); this migration brings the legacy stored prompts to the
same invariant: **stored prompt = steps 1..A, no terminal**.

Shape match (generic content criteria, not known keys — universal):

* the line is a numbered step that is NOTHING but a done call —
  ``N. done().`` / ``N. Call done().`` / ``N. done(success=true, summary=<...>).``
  (optionally with args, optional ``Call``, optional trailing period);
* AND no numbered step follows it (it is the prompt's terminal step — the seeded
  prompts' done steps are followed only by trailing prose paragraphs, which stay).

Verified against the actual post-0086 seeded texts: strips the bare terminal done
step from ``likes`` / ``dislikes`` (``6. Call done().``), ``quality``
(``5. done().``), ``thoughts`` (``8. done().``), and the archived ``notifier`` /
``unnotified-thoughts`` shells; leaves untouched — zero-false-positive discipline —
``skills`` (its step 8 is a compound conditional, not a bare done call),
``notified-thoughts`` (``4. done().  If there's nothing fresh...`` — compound),
``knowledge`` (no numbered done step), and every mid-prompt prose *description* of
done behaviour ("call done() without writing anything").

Single pass (the seeds have exactly one such line); re-running finds nothing left
to strip on them, so it is idempotent over seeded data.
"""

from __future__ import annotations

import re
import sqlite3

# A numbered step line that is nothing but a done call: "N. done().",
# "N. Call done().", "N. done(success=true, summary=<...>)." — optional args
# (paren-free, as every seeded done call's are), optional trailing period.
# ``[ \t]`` only (never ``\s``): under MULTILINE a ``\s`` could swallow a newline
# and match across lines — the shape is strictly one line.
_DONE_STEP_LINE = re.compile(r"^\d+\.[ \t]*(?:Call[ \t]+)?done\([^()]*\)\.?[ \t]*$", re.MULTILINE)
# Any numbered step line — used to check nothing follows the done step.
_STEP_LINE = re.compile(r"^\d+\.", re.MULTILINE)


def strip_terminal_done(prompt: str) -> str:
    """``prompt`` without its terminal bare-``done()`` step line, or unchanged.

    Strips only when the last bare-done step line has no numbered step after it
    (a bare done mid-list, with later steps, is left alone — stripping it would
    corrupt the program).  Removes exactly one adjacent newline with the line.
    """
    matches = list(_DONE_STEP_LINE.finditer(prompt))
    if not matches:
        return prompt
    done_step = matches[-1]
    if any(step.start() > done_step.end() for step in _STEP_LINE.finditer(prompt)):
        return prompt
    start, end = done_step.start(), done_step.end()
    if end < len(prompt) and prompt[end] == "\n":
        end += 1
    elif start > 0 and prompt[start - 1] == "\n":
        start -= 1
    return prompt[:start] + prompt[end:]


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "memory" not in tables:
        return
    rows = conn.execute(
        "SELECT name, extraction_prompt FROM memory WHERE extraction_prompt IS NOT NULL"
    ).fetchall()
    for name, prompt in rows:
        stripped = strip_terminal_done(prompt)
        if stripped != prompt:
            conn.execute("UPDATE memory SET extraction_prompt = ? WHERE name = ?", (stripped, name))
    conn.commit()
