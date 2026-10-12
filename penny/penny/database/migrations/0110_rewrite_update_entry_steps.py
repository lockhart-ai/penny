"""Rewrite every stored ``update_entry`` step into a ``collection_write`` step (#2281).

Type: data

"Write this value under this key" is ONE call: ``collection_write``.  The change
gate reads the written content against what the key holds, so a correction under an
existing key lands as ``KEY_EXISTS_CHANGED`` (the stored baseline is refreshed in
place) and an unchanged value reads ``KEY_EXISTS_UNCHANGED``.  ``update_entry`` ran
no such gate — it rewrote unconditionally, stamped the entry as written, and
reported success — and it is off the tool surface now, so a stored step naming it
would leave a routine with a call it can never run.

UNIVERSAL: a generic, content-shape rewrite over every row of the ``skill`` table
(the 0096/0101/0103 precedent) and over every ``memory.extraction_prompt``, never a
skill or collection by name.  IDEMPOTENT: a second application matches nothing —
the rewritten step names ``collection_write`` and the rewritten line starts with
``N. collection_write(``.

Skill steps: ``tool`` becomes ``collection_write`` and the arguments reshaped to
``{memory, entries: [{key, content}]}`` built from the step's own leaves.  Every
substitution path remaps with its leaf — ``["memory"]`` stays, ``["key"]`` becomes
``["entries", 0, "key"]`` and ``["content"]`` becomes ``["entries", 0,
"content"]`` — so a placeholder, a parameter or an attachment mark travels with the
leaf it sits on and the skill renders exactly as before, under the one write call.
Rendered ``extraction_prompt`` lines are rewritten line by line: the parsed
``memory=``/``key=``/``content=`` leaf values (a bound literal, an unbound
``{name}``, a ``{<what belongs there>}`` slot or a ``the value from step N``
binding) re-emit verbatim inside the ``collection_write`` form, so the rendered
program means what it meant under one call shape.  A line whose arguments do not
parse as that shape is left standing and logged loudly per row — visible
degradation, never a silent mangle.

Prompt-log rows are NOT touched: history is never rewritten.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3

logger = logging.getLogger(__name__)

_OLD_TOOL = "update_entry"
_NEW_TOOL = "collection_write"
_MEMORY_PATH = ["memory"]
_KEY_PATH = ["entries", 0, "key"]
_CONTENT_PATH = ["entries", 0, "content"]
_HEAD_REMAP = {"memory": _MEMORY_PATH, "key": _KEY_PATH, "content": _CONTENT_PATH}

# The rendered step dialect: ``N. update_entry(memory=…, key=…, content=…)`` —
# the one shape ``render_skill`` writes and ``program_calls`` reads.
_STEP_LINE = re.compile(r"^(?P<ordinal>\d+)\. update_entry\((?P<args>.*)\)$", re.DOTALL)


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
        "0110: rewrote update_entry steps in %d skill row(s) and %d extraction_prompt row(s)",
        steps_skills,
        prompts,
    )
    conn.commit()


def _rewrite_skill_steps(conn: sqlite3.Connection) -> int:
    changed_rows = 0
    for name, steps_json in conn.execute("SELECT name, steps FROM skill").fetchall():
        steps = json.loads(steps_json)
        # A list, not a short-circuiting ``any(...)`` generator: EVERY step of the
        # skill must be visited, not just up to the first one that changed.
        changed = [_rewrite_step(step) for step in steps]
        if not any(changed):
            continue
        conn.execute("UPDATE skill SET steps = ? WHERE name = ?", (json.dumps(steps), name))
        changed_rows += 1
    return changed_rows


def _rewrite_step(step: dict) -> bool:
    if step.get("tool") != _OLD_TOOL:
        return False
    args = step.get("arguments") or {}
    step["tool"] = _NEW_TOOL
    step["arguments"] = {
        "memory": args.get("memory"),
        "entries": [{"key": args.get("key"), "content": args.get("content")}],
    }
    for sub in step.get("substitutions") or []:
        path = sub.get("path")
        if isinstance(path, list) and path and path[0] in _HEAD_REMAP:
            sub["path"] = list(_HEAD_REMAP[path[0]]) + list(path[1:])
    return True


def _rewrite_extraction_prompts(conn: sqlite3.Connection) -> int:
    changed_rows = 0
    for name, prompt in conn.execute(
        "SELECT name, extraction_prompt FROM memory WHERE extraction_prompt IS NOT NULL"
    ).fetchall():
        if "update_entry(" not in prompt:
            continue
        lines = [_rewrite_line(line) for line in prompt.splitlines()]
        rewritten = "\n".join(lines)
        if rewritten == prompt:
            continue
        if _OLD_TOOL + "(" in rewritten:
            logger.warning(
                "0110: '%s' still names update_entry after the rewrite — its step lines do "
                "not parse as the rendered memory/key/content shape; rewrite it with "
                "collection_set(name='%s', skill=<the routine>)",
                name,
                name,
            )
        conn.execute("UPDATE memory SET extraction_prompt = ? WHERE name = ?", (rewritten, name))
        changed_rows += 1
    return changed_rows


def _rewrite_line(line: str) -> str:
    match = _STEP_LINE.match(line.strip())
    if match is None:
        return line
    args = _split_args(match.group("args"))
    fields = dict(args)
    if set(fields) != {"memory", "key", "content"}:
        return line
    return (
        f"{match.group('ordinal')}. {_NEW_TOOL}("
        f"memory={fields['memory']}, "
        f"entries=[{{'key': {fields['key']}, 'content': {fields['content']}}}])"
    )


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
