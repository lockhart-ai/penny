"""Rename the two run-record read tools and their arguments (#2306).

Type: data

Two tool renames, each carrying its argument: ``read_run_calls`` -> ``run_list``
(argument ``target`` -> ``source``) and ``get_event`` -> ``run_get`` (argument
``event_id`` -> ``run``).  One execution of a collector or a chat turn is a
``run`` everywhere the model reads, so the reads of the run record say so too.

UNIVERSAL: a generic, content-shape rewrite over every row of the ``skill``
table (the 0096/0101/0111/0112 precedent) and over every ``memory.extraction_prompt``,
never a skill or collection by name.  IDEMPOTENT: a second application matches
nothing — a rewritten step already names a current tool with the new argument
key, and a rewritten line already opens with the current tool name.

Skill steps: on a step whose ``tool`` is one of the two run tools, the tool is
renamed and the ``arguments`` key moves with it (``target`` -> ``source`` on a
``run_list`` step, ``event_id`` -> ``run`` on a ``run_get`` step), and every
substitution whose ``path`` starts with the old key is remapped to start with
the new one instead — so a placeholder, a parameter or an attachment mark
travels with the leaf it sits on.  Every other argument is untouched.

Rendered ``extraction_prompt`` lines are rewritten line by line, by parsing each
``N. tool(args)`` line's argument list (the same top-level-comma splitter 0112
uses) and re-emitting the call under the new tool name / argument name — every
leaf's rendered text carried over VERBATIM, and the rewrite applies to that
line's own call only.  A line whose shape this migration does not recognize is
left standing, unchanged (never guessed at).

Prompt-log rows are NOT touched: history is never rewritten.  A logged call
keeps its old argument key, and nothing re-reads the key — the run-record
projection renders whatever keys the logged row carries.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3

logger = logging.getLogger(__name__)

# The two run tools a skill can hold as a stored step, keyed OLD -> CURRENT,
# with the argument key each rename moves.
_RUN_TOOL_RENAMES = {
    "read_run_calls": ("run_list", "target", "source"),
    "get_event": ("run_get", "event_id", "run"),
}


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
        "0113: renamed run-tool steps in %d skill row(s) and %d extraction_prompt row(s)",
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
    """Rename this run-tool step's tool and argument key, and remap every
    substitution path that starts with the old key to start with the new one
    instead.  A no-op (returns ``False``) on a step of any other tool."""
    rename = _RUN_TOOL_RENAMES.get(step.get("tool"))
    if rename is None:
        return False
    new_tool, old_key, new_key = rename
    step["tool"] = new_tool
    args = step.get("arguments")
    if isinstance(args, dict):
        step["arguments"] = {
            new_key if key == old_key else key: value for key, value in args.items()
        }
    for sub in step.get("substitutions") or []:
        path = sub.get("path")
        if isinstance(path, list) and path and path[0] == old_key:
            sub["path"] = [new_key] + list(path[1:])
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
    """One rendered step line: a call to one of the run tools is re-emitted under
    the new tool name / argument name; every other line — another tool's call, a
    positional form, or a shape this migration does not recognize — passes
    through unchanged."""
    match = _STEP_LINE.match(line.strip())
    if match is None:
        return line
    rename = _RUN_TOOL_RENAMES.get(match.group("tool"))
    if rename is None:
        return line
    new_tool, old_key, new_key = rename
    fields = dict(_split_args(match.group("args")))
    if old_key not in fields:
        return line
    fields = {new_key if key == old_key else key: value for key, value in fields.items()}
    return _render_line(match.group("ordinal"), new_tool, fields)


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
