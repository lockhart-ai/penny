"""No rendered text may name a tool the surface does not carry (#1911).

A collector cycle's tool surface is SCOPED to its program's own calls now, so "is the
tool this message tells the model to call actually here?" stopped being a theoretical
question.  A result that says ``call update_entry(key=…)`` on a surface with no
``update_entry`` is a rendered instruction that cannot be followed — an n≤1 reachability
bug, and exactly the guessing the anchor discipline exists to remove.

The resolution is a DECLARED RELATION rather than a swept-once list: each tool declares
the tools its own text may point at (``Tool.advises``), and a scoped surface is the
CLOSURE of the program's calls under it.  This module is what makes the declaration
trustworthy — it reads each tool's model-facing strings (its ``description``, its message
constants, the f-string literals inside its methods, and the module-level helpers those
methods compose results from) for references to any REGISTERED tool name, and fails when
one is neither the tool itself nor declared.

Docstrings are excluded: a class or method docstring is read by maintainers, never
rendered to the model, so a `` ``done()`` `` in one is prose about the system rather than
an instruction.

The invariant is therefore maintained by construction: a new cross-reference fails here
until it is declared, and once declared the closure carries it onto every surface that
could render it.  The other way to satisfy it is to stop naming the tool literally — the
all-duplicates close interpolates the surface's OWN terminator instead of writing
``done()``, so it cannot name a terminator the surface lacks.
"""

from __future__ import annotations

import ast
import inspect
import re
from collections.abc import Callable
from typing import Any

from penny.agents.collector import Collector
from penny.tests.agents.test_collector import _make_collector
from penny.tools.base import Tool
from penny.tools.browse import BrowseTool  # noqa: F401  (registers browse)
from penny.tools.choose import ChooseTool  # noqa: F401  (registers choose)
from penny.tools.memory_tools import build_memory_tools  # noqa: F401  (registers memory)

# A call reference in rendered text: a registered tool's name followed by ``(``, or the
# name in backticks.  Both are how these messages name a tool the model should call.
#
# Matched on a WORD BOUNDARY, because tool names nest: ``collection_read_latest(``
# contains ``read_latest(``, and a plain substring test reads one call as two.  The
# paren is ATTACHED, because a call is written ``tool(`` and never ``tool (`` — the
# space-separated form is ordinary prose ("search (the meaning leg)") and reading it as
# a call would make the invariant fail on sentences.
_CALL_REFERENCE = r"\b{name}\("
_QUOTED_REFERENCE = r"`{name}`"


def _names_tool(text: str, name: str) -> bool:
    """Whether ``text`` points the model at the tool called ``name``."""
    return bool(
        re.search(_CALL_REFERENCE.format(name=re.escape(name)), text)
        or re.search(_QUOTED_REFERENCE.format(name=re.escape(name)), text)
    )


_PROGRAM = (
    "1. browse(queries=['https://ex.example/t'], extract='the dawn sailing')\n"
    "2. collection_write(memory='ferry-departures', entries=[{'key': 'x'}])"
)


def _registered_tools() -> dict[str, type[Tool]]:
    """Every tool class the registry knows, by name — the vocabulary a reference is
    matched against, so nothing here enumerates tools by hand."""
    return dict(Tool._registry)


def _dedent(source: str) -> str:
    """Source lifted to column zero so ``ast.parse`` accepts a class or method body."""
    lines = source.splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[indent:] if len(line) > indent else line.lstrip() for line in lines)


def _parse(obj: type | Callable[..., Any]) -> ast.Module | None:
    try:
        return ast.parse(_dedent(inspect.getsource(obj)))
    except OSError, TypeError, SyntaxError:  # pragma: no cover — no readable source
        return None


def _model_facing_strings(parsed: ast.Module) -> list[str]:
    """Every string literal EXCEPT docstrings — the text that can reach the model.

    Read from the source rather than by executing anything, because most of these
    messages are composed inside ``_run`` at call time and never exist as an
    inspectable value.  An f-string's literal segments are what name a sibling tool;
    its interpolated values are runtime data and are skipped by construction."""
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(parsed)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        node.value
        for node in ast.walk(parsed)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _rendered_text(tool: type[Tool]) -> str:
    """Everything this tool can render: its own body, plus the module-level constants
    and helper functions its body composes results from — followed TRANSITIVELY.

    The helpers matter as much as the class — ``collection_write``'s duplicate rejection
    is built by a module-level ``_format_duplicate``, which is exactly where it names
    ``update_entry``, so a scan of the class alone would miss the reference that started
    all this.  And a helper usually names a CONSTANT rather than carrying the text
    inline, because that is the house style — so a scan that stopped at the helper would
    read it as having nothing to say and the invariant would quietly enforce nothing on
    exactly the messages it exists for (#1919: this went from passing to empty the moment
    that rejection moved into a constant, caught only by the guard-on-the-guard below)."""
    parsed = _parse(tool)
    if parsed is None:  # pragma: no cover
        return ""
    strings: list[str] = []
    _collect_rendered(parsed, inspect.getmodule(tool), strings, set())
    return " ".join(strings)


def _collect_rendered(parsed: ast.Module, module, strings: list[str], seen: set[str]) -> None:
    """This node's own model-facing strings, then every module-level name it references:
    a constant contributes its text, a helper contributes its strings AND whatever it
    references in turn.  ``seen`` bounds the walk, so a cycle terminates."""
    strings.extend(_model_facing_strings(parsed))
    for name in {node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)}:
        if name in seen:
            continue
        seen.add(name)
        value = getattr(module, name, None)
        if isinstance(value, str):
            strings.append(value)
        elif inspect.isfunction(value) and inspect.getmodule(value) is module:
            helper = _parse(value)
            if helper is not None:
                _collect_rendered(helper, module, strings, seen)


def _referenced_tools(tool: type[Tool], vocabulary: dict[str, type[Tool]]) -> set[str]:
    """The tool names this tool's own rendered text points the model at."""
    text = _rendered_text(tool)
    return {name for name in vocabulary if name != tool.name and _names_tool(text, name)}


def test_every_tool_declares_the_siblings_its_text_names():
    """THE COHERENCE INVARIANT (#1911): every tool a tool's rendered text points at is
    declared on ``advises``, so a scoped surface's closure carries it.

    Failing here means a message names a call the model may not have.  Two ways to fix
    it: declare the sibling (so it joins the surface wherever that text can render), or
    stop naming it literally — interpolate what the surface actually gave you, the way
    the all-duplicates close names its terminator."""
    vocabulary = _registered_tools()
    undeclared = {
        tool.name: sorted(referenced - set(tool.advises))
        for tool in vocabulary.values()
        if (referenced := _referenced_tools(tool, vocabulary)) - set(tool.advises)
    }
    assert undeclared == {}, (
        "These tools' rendered text names siblings they do not declare on `advises` — "
        "on a program-scoped collector surface that text could point at a tool the "
        f"model does not have:\n{undeclared}"
    )


def test_the_repo_declares_no_advice_for_a_tool_that_does_not_exist():
    """A declared sibling has to BE a tool — a typo in an ``advises`` entry would widen
    nothing and silently hide the reference it was meant to cover."""
    vocabulary = _registered_tools()
    for tool in vocabulary.values():
        unknown = sorted(set(tool.advises) - set(vocabulary))
        assert unknown == [], f"`{tool.name}` declares advice for unknown tools: {unknown}"


def test_the_scanner_reads_the_real_message_sources():
    """A guard on the guard: the scan has to actually be reading text, since a scan that
    read nothing would pass every assertion here and enforce nothing.

    ``collection_write`` is the canary — its duplicate rejection is the reference that
    motivated the whole relation, and it lives in a module-level helper, so seeing it
    proves the class scan, the constant scan and the helper scan are all live."""
    vocabulary = _registered_tools()
    write = vocabulary["collection_write"]
    assert "update_entry" in _referenced_tools(write, vocabulary)
    assert len(vocabulary) > 20, "the registry looks unpopulated — imports may be missing"


def test_a_scoped_surface_renders_no_off_surface_tool(test_config, tmp_path):
    """THE END-TO-END READ: a real program-scoped cycle's surface is closed over the
    advice relation, so nothing it can render names a tool it lacks — neither a tool
    result nor the composed prompt's own runtime rules.

    The same invariant as the declaration test, read from the other end: over the surface
    a live cycle actually builds rather than over what the classes declare."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "ferry-departures", "the dawn sailing", extraction_prompt=_PROGRAM
    )
    collector._bind(db.memories.get("ferry-departures"))
    vocabulary = _registered_tools()

    surface = {tool.name for tool in collector.get_tools()}

    for name in surface:
        referenced = _referenced_tools(vocabulary[name], vocabulary)
        assert referenced <= surface, (
            f"`{name}` can render text naming {sorted(referenced - surface)}, "
            f"which this scoped surface does not carry: {sorted(surface)}"
        )
    target = db.memories.get("ferry-departures")
    assert target is not None
    composed = Collector._compose_prompt(target, None, frozenset(surface))
    off_surface = [
        name for name in vocabulary if name not in surface and _names_tool(composed, name)
    ]
    assert off_surface == [], f"the composed prompt names off-surface tools: {off_surface}"
