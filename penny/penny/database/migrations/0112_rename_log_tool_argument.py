"""Rename the log tools' first argument to ``log`` (#2304).

Type: data

``log_read`` and ``log_append`` take their log name as ``log`` now, where they
took ``memory`` (the tool names themselves don't change; ``log_create`` takes
``log`` too, but it is a lifecycle tool a skill never holds as a step, so no
stored step or rendered line can carry it).

UNIVERSAL: a generic, content-shape rewrite over every row of the ``skill``
table (the 0096/0101/0111 precedent) and over every ``memory.extraction_prompt``,
never a skill or log by name.  IDEMPOTENT: a second application matches nothing —
a rewritten step already names its log under ``log``, and a rewritten line
already opens with ``log=``.

Skill steps: on a step whose ``tool`` is one of the two log tools, the
``arguments`` key ``memory`` becomes ``log``, and every substitution whose
``path`` starts with ``"memory"`` is remapped to start with ``"log"`` instead —
so a placeholder, a parameter or an attachment mark travels with the leaf it
sits on.  The ``content`` argument is untouched.

Rendered ``extraction_prompt`` lines are rewritten line by line, by parsing each
``N. tool(args)`` line's argument list (the same top-level-comma splitter 0111
uses) and re-emitting the call with ``log=`` in place of ``memory=`` — every
leaf's rendered text carried over VERBATIM, and the rewrite applies to that
line's own call only.  A line whose shape this migration does not recognize is
left standing, unchanged (never guessed at).

Prompt-log rows are NOT touched: history is never rewritten.  A logged call's
old argument key is read at the read projection (``render_tool_call``) through
the same ``log``-or-``memory`` fallback the live object uses.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3

logger = logging.getLogger(__name__)

# The two log tools a skill can hold as a stored step, by their CURRENT name —
# both take their log name as ``log`` now.
_LOG_TOOLS = frozenset({"log_read", "log_append"})


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    steps_skills = prompts = 0
    if "skill" in tables:
        steps_skills = _rewrite_skill_steps(conn)
    if "memory" in tables:
        prompts = _rewrite_extraction_prompts(conn)
    logger.info(
        "0112: renamed the log argument in %d skill row(s) and %d extraction_prompt row(s)",
        steps_skills,
        prompts,
    )
    conn.commit()


def _rewrite_skill_steps(conn: sqlite3.Connection) -> int:
    changed_rows = 0
    for name, steps_json in conn.execute("SELECT name, steps FROM skill").fetchall():
        steps = json.loads(steps_json)
        changed = [_rewrite_step(step) for step in steps]
        if not any(changed):
            continue
        conn.execute("UPDATE skill SET steps = ? WHERE name = ?", (json.dumps(steps), name))
        changed_rows += 1
    return changed_rows


def _rewrite_step(step: dict) -> bool:
    """Rename this log-tool step's ``memory`` argument to ``log``, and remap every
    substitution path that starts with ``"memory"`` to start with ``"log"``
    instead.  A no-op (returns ``False``) on a step of any other tool, or one
    with no ``memory`` key — already migrated, or never had one."""
    if step.get("tool") not in _LOG_TOOLS:
        return False
    args = step.get("arguments")
    if not isinstance(args, dict) or "memory" not in args:
        return False
    step["arguments"] = {("log" if key == "memory" else key): value for key, value in args.items()}
    for sub in step.get("substitutions") or []:
        path = sub.get("path")
        if isinstance(path, list) and path and path[0] == "memory":
            sub["path"] = ["log"] + list(path[1:])
    return True


# The rendered step dialect: ``N. tool(args)`` — the one shape ``render_skill``
# writes and ``program_calls`` reads.
_STEP_LINE = re.compile(r"^(?P<ordinal>\d+)\.\s*(?P<tool>[a-z_][a-z0-9_]*)\((?P<args>.*)\)$")


def _rewrite_extraction_prompts(conn: sqlite3.Connection) -> int:
    changed_rows = 0
    for name, prompt in conn.execute(
        "SELECT name, extraction_prompt FROM memory WHERE extraction_prompt IS NOT NULL"
    ).fetchall():
        lines = [_rewrite_line(line) for line in prompt.splitlines()]
        rewritten = "\n".join(lines)
        if rewritten == prompt:
            continue
        conn.execute("UPDATE memory SET extraction_prompt = ? WHERE name = ?", (rewritten, name))
        changed_rows += 1
    return changed_rows


def _rewrite_line(line: str) -> str:
    """One rendered step line: a call to a log tool carrying ``memory=`` is
    re-emitted with ``log=``; every other line — another tool's call, a positional
    ``log_read('x')`` form, or a shape this migration does not recognize — passes
    through unchanged."""
    match = _STEP_LINE.match(line.strip())
    if match is None:
        return line
    tool = match.group("tool")
    if tool not in _LOG_TOOLS:
        return line
    fields = dict(_split_args(match.group("args")))
    if "memory" not in fields:
        return line
    fields = {("log" if key == "memory" else key): value for key, value in fields.items()}
    return _render_line(match.group("ordinal"), tool, fields)


def _render_line(ordinal: str, tool: str, fields: dict) -> str:
    args = ", ".join(f"{key}={value}" for key, value in fields.items())
    return f"{ordinal}. {tool}({args})"


def _split_args(text: str) -> list[tuple[str, str]]:
    """Split ``name=value`` pairs on top-level commas — commas inside a quoted
    literal, a ``{placeholder}`` span or a list stay inside their value."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    start = 0
    for index, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    out: list[tuple[str, str]] = []
    for part in parts:
        key, _, value = part.strip().partition("=")
        out.append((key.strip(), value.strip()))
    return out
