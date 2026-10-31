"""Guard: no model-facing string writes a tool call in Python's ``tool(args)``
notation (#2859).

Gemma's real tool calls, and the tool declarations her own chat template shows
her, are written ``call:name{key:value, ...}`` — never with parentheses. A
prompt, a tool description, a refusal, or a run render that showed her
``tool(args)`` taught a body her raw-template parser cannot read; measured in
prod she sometimes wrote exactly that hybrid (her template's ``call:`` opener
around a Python-parenthesised body), and every one of those failed to parse.
The fix is the brace form, ``tool{args}`` — this test keeps the parens form
from creeping back into the files that write what she reads.

Scoped to the model-facing modules named in #2859 (prompts, tool descriptions
and refusals, run renders, the self-state header, the conversation map, the
behaviour-condition catalog, and the changelog), and keyed to the LIVE tool
name list — every ``Tool`` subclass's own ``name`` attribute, read off the
class rather than restated by hand, so a renamed or retired tool can't leave
a stale entry behind. ``text_validity.py`` (the call-as-text detectors, which
catch the notation rather than teach it) and the test suite itself are out of
scope, per #2859.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
from pathlib import Path

import penny.plugins as _plugins_pkg
import penny.tools as _tools_pkg
from penny.tools.base import Tool

_PENNY_ROOT = Path(__file__).resolve().parents[1]

# Every model-facing module #2859 named: prompts, tool descriptions/refusals,
# run renders, the self-state header, the conversation map's anchor, the
# behaviour-condition catalog (rendered on a run record), and the changelog
# entries written for Penny to read with ``log_read``.
_MODEL_FACING_PY_FILES = (
    "prompts.py",
    "agents/self_state.py",
    "agents/collector.py",
    "agents/conversation_map.py",
    "database/memory/types.py",
    "database/memory/objects.py",
    "tools/memory_tools.py",
    "tools/memory_args.py",
    "tools/browse.py",
    "tools/collection_instantiation.py",
    "tools/search_emails.py",
    "tools/read_emails.py",
    "tools/draft_email.py",
    "tools/list_folders.py",
    "tools/list_emails.py",
    "tools/job_summary.py",
    "validation/conditions.py",
)
_MODEL_FACING_YAML_FILES = ("changelog.yaml",)


def _all_tool_names() -> set[str]:
    """Every live tool's ``name`` — a recursive walk of ``Tool`` subclasses
    after importing every module under ``penny.tools`` and ``penny.plugins``,
    so a tool added, renamed, or retired anywhere never needs this test
    hand-updated to match (the reachability doctrine: read the live
    structure, never a restated list)."""
    for package in (_tools_pkg, _plugins_pkg):
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            importlib.import_module(info.name)

    names: set[str] = set()

    def _walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            tool_name = getattr(sub, "name", None)
            if isinstance(tool_name, str) and tool_name:
                names.add(tool_name)
            _walk(sub)

    _walk(Tool)
    return names


def _docstring_string_ids(tree: ast.AST) -> set[int]:
    """The ``id()`` of every docstring ``Constant`` node (module/class/def's
    first statement) — excluded, since a docstring explains code to a
    developer and is never shown to the model (#2859's own scope)."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            ids.add(id(node.body[0].value))
    return ids


def _call_notation_pattern(tool_names: set[str]) -> re.Pattern[str]:
    alternation = "|".join(re.escape(name) for name in sorted(tool_names))
    return re.compile(rf"\b(?:{alternation})\(")


def _violations_in_py_file(path: Path, pattern: re.Pattern[str]) -> list[str]:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    skip = _docstring_string_ids(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skip
            and pattern.search(node.value)
        ):
            match = pattern.search(node.value)
            assert match is not None
            found.append(f"{path.name}:{node.lineno}: {match.group(0)}...")
    return found


def _violations_in_yaml_file(path: Path, pattern: re.Pattern[str]) -> list[str]:
    found: list[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if line.strip().startswith("#"):
            continue
        match = pattern.search(line)
        if match:
            found.append(f"{path.name}:{lineno}: {match.group(0)}...")
    return found


def test_no_model_facing_string_writes_a_call_in_python_parens_notation() -> None:
    tool_names = _all_tool_names()
    assert len(tool_names) > 10, (
        "the live tool-name walk found suspiciously few tools — it likely failed "
        "to import a tools/plugins module rather than the fleet actually shrinking"
    )
    pattern = _call_notation_pattern(tool_names)

    violations: list[str] = []
    for rel in _MODEL_FACING_PY_FILES:
        violations += _violations_in_py_file(_PENNY_ROOT / rel, pattern)
    for rel in _MODEL_FACING_YAML_FILES:
        violations += _violations_in_yaml_file(_PENNY_ROOT / rel, pattern)

    assert not violations, (
        "Python call notation tool(args) found in model-facing text — gemma's "
        "parser only reads the brace form tool{args} (#2859):\n" + "\n".join(violations)
    )
