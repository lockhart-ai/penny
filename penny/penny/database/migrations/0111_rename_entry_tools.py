"""Rename the collection-entry tools and their first argument (#2307).

Type: data

Three tool renames: ``collection_read_latest`` -> ``collection_list``,
``collection_delete_entry`` -> ``collection_remove``, ``collection_read_random`` ->
``collection_sample`` (the last already a legacy name nobody could call live, but a
skill taught before #2284 can still hold it as a stored step).  One argument
rename, on all six entry tools (the three above plus ``collection_get`` /
``collection_keys`` / ``collection_write``, whose own names don't change): the
first argument becomes ``collection`` wherever it was ``memory``.  Two exact
swaps left from #2290/#2302: a stored ``exists`` step becomes ``collection_get``,
and a stored ``read_similar`` step becomes ``find`` (which keeps ``memory`` as
its own scoping argument — only ``anchor`` moves, to ``query``, and ``k`` has no
home in ``find`` so it is dropped).

UNIVERSAL: a generic, content-shape rewrite over every row of the ``skill`` table
(the 0096/0101/0103/0110 precedent) and over every ``memory.extraction_prompt``,
never a skill or collection by name.  IDEMPOTENT: a second application matches
nothing — a rewritten step already names a current tool with a ``collection``
argument, and a rewritten line already opens with the current tool name.

Skill steps: each step's ``tool`` is renamed per ``_TOOL_RENAMES`` first; then, on
a step whose (possibly just-renamed) tool is one of the six entry tools, the
``arguments`` key ``memory`` becomes ``collection`` and every substitution whose
``path`` starts with ``"memory"`` is remapped to start with ``"collection"`` instead
— so a placeholder, a parameter or an attachment mark travels with the leaf it
sits on.  The two swaps reshape the argument dict itself (see
``_rewrite_exists_step`` / ``_rewrite_read_similar_step`` below) and remap their
substitutions the same way.

A step that cannot be mapped unambiguously is LEFT STANDING and logged loudly —
visible degradation, never a silent mangle or a guess:
- ``exists`` with other than exactly one ``memories`` entry (which collection it
  meant is genuinely ambiguous).
- ``exists`` whose ``content`` leaf carries its own substitution AND an explicit
  ``key`` is also present — the new shape has no field for ``content`` once
  ``key`` already answers "which key", so keeping that substitution would silently
  drop a parameter/binding. (A ``content`` substitution that supplies the KEY —
  the stored step had no explicit ``key`` — moves onto ``key`` instead, which is
  not a drop.)

Rendered ``extraction_prompt`` lines are rewritten line by line, by parsing each
``N. tool(args)`` line's argument list (the same top-level-comma splitter 0110
uses) and re-emitting it under the new tool name / argument name, with every
leaf's rendered text — a bound literal, an unbound ``{name}``, a
``{<what belongs there>}`` slot, or a ``the value from step N`` binding — carried
over VERBATIM.  A line whose shape this migration does not recognize (the same
ambiguity rules as the steps above) is left standing and logged.

Prompt-log rows are NOT touched: history is never rewritten.  (The LIVE readers'
own seam for an old logged NAME is ``LoggedToolCall.tool``, resolved through
``LEGACY_TOOL_RENAMES`` — unrelated to this migration, which only rewrites the
stored skill/collection registry.)
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3

logger = logging.getLogger(__name__)

# Tool renames (steps + rendered lines).  ``collection_read_random`` was already
# off the live surface by the time #2302 restored it as ``collection_sample``, but
# a skill taught before #2284 can still hold the old name as a stored step.
_TOOL_RENAMES = {
    "collection_read_latest": "collection_list",
    "collection_delete_entry": "collection_remove",
    "collection_read_random": "collection_sample",
}

# The six entry tools, by their CURRENT name — every one takes its first argument
# as ``collection`` now.  Checked AFTER ``_TOOL_RENAMES`` is applied, so a step
# renamed above is caught here too.
_ENTRY_TOOLS = frozenset(
    {
        "collection_get",
        "collection_keys",
        "collection_list",
        "collection_sample",
        "collection_write",
        "collection_remove",
    }
)


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
        "0111: renamed entry-tool steps in %d skill row(s) and %d extraction_prompt row(s)",
        steps_skills,
        prompts,
    )
    conn.commit()


def _rewrite_skill_steps(conn: sqlite3.Connection) -> int:
    changed_rows = 0
    for name, steps_json in conn.execute("SELECT name, steps FROM skill").fetchall():
        steps = json.loads(steps_json)
        changed = [_rewrite_step(name, step) for step in steps]
        if not any(changed):
            continue
        conn.execute("UPDATE skill SET steps = ? WHERE name = ?", (json.dumps(steps), name))
        changed_rows += 1
    return changed_rows


def _rewrite_step(skill_name: str, step: dict) -> bool:
    """Dispatch one step to its rewrite, trying each kind in turn.  Returns
    whether the step changed."""
    tool = step.get("tool")
    if tool == "exists":
        return _rewrite_exists_step(skill_name, step)
    if tool == "read_similar":
        return _rewrite_read_similar_step(step)
    renamed_tool = _TOOL_RENAMES.get(tool, tool)
    tool_changed = renamed_tool != tool
    if tool_changed:
        step["tool"] = renamed_tool
    if renamed_tool in _ENTRY_TOOLS:
        return _rename_memory_argument(step) or tool_changed
    return tool_changed


def _rename_memory_argument(step: dict) -> bool:
    """Rename this entry-tool step's ``memory`` argument to ``collection``, and
    remap every substitution path that starts with ``"memory"`` to start with
    ``"collection"`` instead.  A no-op (returns ``False``) on a step with no
    ``memory`` key — already migrated, or never had one."""
    args = step.get("arguments")
    if not isinstance(args, dict) or "memory" not in args:
        return False
    step["arguments"] = {
        ("collection" if key == "memory" else key): value for key, value in args.items()
    }
    for sub in step.get("substitutions") or []:
        path = sub.get("path")
        if isinstance(path, list) and path and path[0] == "memory":
            sub["path"] = ["collection"] + list(path[1:])
    return True


def _rewrite_exists_step(skill_name: str, step: dict) -> bool:
    """``exists(memories=[...], content=..., key=...)`` -> ``collection_get(collection=<the
    one memory>, key=<key>)`` (#2290/#2302) — left standing, logged, when the
    memory is ambiguous or a substitution would be silently dropped."""
    args = step.get("arguments") or {}
    memories = args.get("memories")
    if not isinstance(memories, list) or len(memories) != 1:
        logger.warning(
            "0111: '%s' still names exists() after the rewrite — its memories list is not "
            "exactly one name, so which collection it meant is ambiguous; rewrite it by hand "
            "with collection_get(collection=<name>, key=<key>)",
            skill_name,
        )
        return False
    has_key = args.get("key") is not None
    subs = step.get("substitutions") or []
    if has_key and any(_path_head(sub) == "content" for sub in subs):
        logger.warning(
            "0111: '%s' still names exists() after the rewrite — its content leaf carries a "
            "parameter/binding that collection_get has no field for once key is also present; "
            "rewrite it by hand with collection_get(collection=<name>, key=<key>)",
            skill_name,
        )
        return False
    key_value = args.get("key") if has_key else args.get("content")
    step["tool"] = "collection_get"
    step["arguments"] = {"collection": memories[0], "key": key_value}
    new_subs = []
    for sub in subs:
        head = _path_head(sub)
        path = sub["path"]
        if head == "memories" and len(path) > 1 and path[1] == 0:
            sub["path"] = ["collection"] + list(path[2:])
        elif head == "key":
            pass  # stays
        elif head == "content" and not has_key:
            # content supplied the key (no explicit key was given) — its leaf IS the key leaf.
            sub["path"] = ["key"] + list(path[1:])
        else:
            continue  # unreachable given the guards above; drop defensively rather than guess
        new_subs.append(sub)
    step["substitutions"] = new_subs
    return True


def _rewrite_read_similar_step(step: dict) -> bool:
    """``read_similar(memory=M, anchor=A[, k=K])`` -> ``find(query=A, memory=M)`` (#2290):
    ``find`` keeps ``memory`` as its own scoping argument untouched; ``anchor`` becomes
    ``query``; ``k`` has no home in ``find`` and is dropped."""
    args = step.get("arguments") or {}
    step["tool"] = "find"
    step["arguments"] = {"query": args.get("anchor"), "memory": args.get("memory")}
    new_subs = []
    for sub in step.get("substitutions") or []:
        head = _path_head(sub)
        path = sub["path"]
        if head == "anchor":
            sub["path"] = ["query"] + list(path[1:])
        elif head == "memory":
            pass  # stays
        else:
            continue  # a "k" substitution (or anything else) has nowhere to go — dropped
        new_subs.append(sub)
    step["substitutions"] = new_subs
    return True


def _path_head(sub: dict) -> str | None:
    path = sub.get("path")
    if isinstance(path, list) and path and isinstance(path[0], str):
        return path[0]
    return None


# The rendered step dialect: ``N. tool(args)`` — the one shape ``render_skill``
# writes and ``program_calls`` reads.
_STEP_LINE = re.compile(r"^(?P<ordinal>\d+)\.\s*(?P<tool>[a-z_][a-z0-9_]*)\((?P<args>.*)\)$")


def _rewrite_extraction_prompts(conn: sqlite3.Connection) -> int:
    changed_rows = 0
    for name, prompt in conn.execute(
        "SELECT name, extraction_prompt FROM memory WHERE extraction_prompt IS NOT NULL"
    ).fetchall():
        lines = [_rewrite_line(name, line) for line in prompt.splitlines()]
        rewritten = "\n".join(lines)
        if rewritten == prompt:
            continue
        conn.execute("UPDATE memory SET extraction_prompt = ? WHERE name = ?", (rewritten, name))
        changed_rows += 1
    return changed_rows


def _rewrite_line(collection_name: str, line: str) -> str:
    match = _STEP_LINE.match(line.strip())
    if match is None:
        return line
    tool = match.group("tool")
    ordinal = match.group("ordinal")
    fields = dict(_split_args(match.group("args")))
    if tool == "exists":
        return _rewrite_exists_line(collection_name, ordinal, fields) or line
    if tool == "read_similar":
        return _rewrite_read_similar_line(ordinal, fields) or line
    renamed_tool = _TOOL_RENAMES.get(tool, tool)
    if renamed_tool in _ENTRY_TOOLS and "memory" in fields:
        fields = {
            ("collection" if key == "memory" else key): value for key, value in fields.items()
        }
        return _render_line(ordinal, renamed_tool, fields)
    if renamed_tool != tool:
        return _render_line(ordinal, renamed_tool, fields)
    return line


def _rewrite_exists_line(collection_name: str, ordinal: str, fields: dict) -> str | None:
    memories_text = fields.get("memories")
    if memories_text is None:
        return None
    elements = _split_list(memories_text)
    if elements is None or len(elements) != 1:
        logger.warning(
            "0111: '%s' still names exists() after the rewrite — its memories list is not "
            "exactly one name; rewrite it by hand with "
            "collection_get(collection=<name>, key=<key>)",
            collection_name,
        )
        return None
    key_text = fields.get("key", fields.get("content"))
    if key_text is None:
        return None
    return _render_line(ordinal, "collection_get", {"collection": elements[0], "key": key_text})


def _rewrite_read_similar_line(ordinal: str, fields: dict) -> str | None:
    if "anchor" not in fields:
        return None
    new_fields = {"query": fields["anchor"], "memory": fields.get("memory", "?")}
    return _render_line(ordinal, "find", new_fields)


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


def _split_list(text: str) -> list[str] | None:
    """Parse a rendered list literal's elements (``['a', 'b']`` -> ``["'a'", "'b'"]``),
    on the same top-level-comma rule as ``_split_args``.  ``None`` if ``text`` is not a
    ``[...]`` literal."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    inner = text[1:-1]
    if not inner.strip():
        return []
    depth = 0
    quote: str | None = None
    start = 0
    parts: list[str] = []
    for index, char in enumerate(inner):
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
            parts.append(inner[start:index])
            start = index + 1
    parts.append(inner[start:])
    return [part.strip() for part in parts]
