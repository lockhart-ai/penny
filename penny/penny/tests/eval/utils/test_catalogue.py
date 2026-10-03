"""The catalogue of supported behaviours (#2214): its render, its staleness check, its CLI.

Deterministic — fixture entries in, a document out.  That a driver records its case and
stops in catalogue mode is pinned through the real drivers in ``test_run_health.py``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from penny.conversation_machine import ConversationState
from penny.tests.eval.utils import catalogue
from penny.tests.eval.utils.catalogue import (
    CATALOGUE_ENV,
    Area,
    CatalogueEntry,
    CatalogueError,
    Layer,
    collect,
    load,
    main,
    record,
    render,
    stale_reason,
)

_IDLE = ConversationState.IDLE
_ELICIT = ConversationState.ELICIT
_NODE = "penny/tests/eval/chat/idle/test_a_file.py::test_a_case"

_HOLDS_IDLE = CatalogueEntry(
    case_id="classifier-holds-idle",
    area=Area.CONVERSATION_MACHINE,
    edge=(_IDLE, _IDLE),
    layer=Layer.CLASSIFIER_DRAW,
    sentence="In the state-classifier micro-context, when a remark asks nothing, Penny holds idle.",
    wording="lovely morning out there",
    node_id=_NODE,
)
_BANTER = CatalogueEntry(
    case_id="transition-idle-to-idle",
    area=Area.CONVERSATION_MACHINE,
    edge=(_IDLE, _IDLE),
    layer=Layer.WHOLE_TURN,
    sentence="In the chat agent, when a remark asks nothing, Penny answers and changes nothing.",
    wording="lovely morning out there",
    node_id=_NODE,
)
_COLD_ASK = CatalogueEntry(
    case_id="transition-idle-to-elicit",
    area=Area.CONVERSATION_MACHINE,
    edge=(_IDLE, _ELICIT),
    layer=Layer.WHOLE_TURN,
    sentence="In the chat agent, when no routine covers a standing ask, Penny asks to be taught.",
    wording="watch this listing\nfor me | daily",
    node_id=_NODE,
)
_RECALL = CatalogueEntry(
    case_id="memory-cold-recall",
    area=Area.MEMORY,
    edge=None,
    layer=Layer.WHOLE_TURN,
    sentence="In the chat agent, when asked for a stored fact, Penny states it.",
    wording="what was the deck listed at?",
    node_id=_NODE,
)
# Handed over out of order: the document sorts its own rows.
_ENTRIES = (_BANTER, _RECALL, _HOLDS_IDLE, _COLD_ASK)

_RENDERED = """\
# Eval catalogue

The behaviours Penny supports, one row per eval case, grouped by the area of Penny each case \
is about. This file is generated from the cases: `make fix` writes it and `make check` fails \
when it is out of date. It is not edited by hand.

A case declares its area and, where it covers one, the edge of the conversation machine it \
moves along. The first wording is the first of the wordings the case is driven in.

## Cases by area

| Area | Cases |
|---|---|
| Conversation machine | 3 |
| Teaching a routine | 0 |
| Memory | 1 |
| Answering from the web | 0 |
| Standing jobs | 0 |
| Notifications | 0 |
| Background collectors | 0 |
| Reading a page | 0 |
| Chat tools | 0 |
| Total | 4 |

## Conversation machine

| Case | Behaviour | First wording | Edge |
|---|---|---|---|
| `classifier-holds-idle` | In the state-classifier micro-context, when a remark asks nothing, \
Penny holds idle. | lovely morning out there | idle → idle |
| `transition-idle-to-elicit` | In the chat agent, when no routine covers a standing ask, \
Penny asks to be taught. | watch this listing for me \\| daily | idle → elicit |
| `transition-idle-to-idle` | In the chat agent, when a remark asks nothing, Penny answers \
and changes nothing. | lovely morning out there | idle → idle |

## Teaching a routine

No cases.

## Memory

| Case | Behaviour | First wording | Edge |
|---|---|---|---|
| `memory-cold-recall` | In the chat agent, when asked for a stored fact, Penny states it. \
| what was the deck listed at? |  |

## Answering from the web

No cases.

## Standing jobs

No cases.

## Notifications

No cases.

## Background collectors

No cases.

## Reading a page

No cases.

## Chat tools

No cases.

## Machine edge coverage

One row per edge of the conversation machine. A classifier draw covers an edge when a case \
draws that decision on its own; a whole turn covers it when a case drives the chat turn that \
makes the move.

| Edge | Classifier draw | Whole turn |
|---|---|---|
| idle → apply | — | — |
| idle → request | — | — |
| idle → learn | — | — |
| idle → elicit | — | `transition-idle-to-elicit` |
| idle → idle | `classifier-holds-idle` | `transition-idle-to-idle` |
| elicit → learn | — | — |
| elicit → elicit | — | — |
| elicit → idle | — | — |
| learn → apply | — | — |
| learn → learn | — | — |
| learn → idle | — | — |
| request → apply | — | — |
| request → elicit | — | — |
| request → idle | — | — |
"""


def test_the_catalogue_renders_whole() -> None:
    """Every area in declaration order whether or not it has a case, each area's cases by id,
    a wording on one line with the column separator escaped, and one coverage row per edge of
    the machine with a dash where no case covers it at that layer."""
    assert render(_ENTRIES) == _RENDERED


def test_a_collection_is_read_back_and_a_shared_case_id_is_refused(tmp_path: Path) -> None:
    """What the drivers append is what ``load`` returns, and two cases under one id are
    refused naming the id and both tests — the catalogue is keyed by it."""
    gathered = tmp_path / "entries.jsonl"
    assert load(gathered) == []
    for entry in _ENTRIES:
        record(gathered, entry)
    assert load(gathered) == list(_ENTRIES)

    record(gathered, _RECALL.model_copy(update={"node_id": "another_file.py::test_again"}))
    with pytest.raises(CatalogueError, match="memory-cold-recall: declared by more than one"):
        load(gathered)


def test_collecting_runs_the_cases_in_catalogue_mode_and_refuses_a_failed_run(monkeypatch) -> None:
    """``collect`` asks a child pytest for the eval-marked cases — the machinery folder left
    out — with the catalogue file named in its environment, and reads back what was appended.
    A child that failed is refused with its own output, which names the case."""
    asked: list[list[str]] = []

    def _child(command: list[str], *, env: dict[str, str], **_: object):
        asked.append(command)
        record(Path(env[CATALOGUE_ENV]), _RECALL)
        return subprocess.CompletedProcess(command, 0, stdout="1 skipped", stderr="")

    monkeypatch.setattr(catalogue.subprocess, "run", _child)
    assert collect() == [_RECALL]
    [command] = asked
    suite = Path(catalogue.__file__).resolve().parent.parent
    assert command[1:] == [
        "-m",
        "pytest",
        str(suite),
        f"--ignore={suite / 'utils'}",
        "-m",
        "eval",
        "-q",
        "--tb=line",
        "-p",
        "no:cacheprovider",
    ]

    refusal = "ValueError: a-case: a case must declare its area"
    monkeypatch.setattr(
        catalogue.subprocess,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 1, stdout=refusal, stderr=""),
    )
    with pytest.raises(CatalogueError, match=refusal):
        collect()


def test_a_doctored_catalogue_is_stale_and_the_check_says_to_run_make_fix(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """``write`` leaves the file ``check`` accepts; a file edited by hand, or left behind when
    a case changed, fails the check with the fix named and the difference shown — and so does
    a missing file, a suite that cannot be collected, and a bad invocation."""
    path = tmp_path / "eval-catalogue.md"
    monkeypatch.setattr(catalogue, "collect", lambda: list(_ENTRIES))

    assert stale_reason(path, _RENDERED) == (
        f"{path} is missing — run `make fix` to generate it from the eval cases"
    )
    assert main(["check", str(path)]) == 1
    assert main(["write", str(path)]) == 0
    assert path.read_text() == _RENDERED
    assert stale_reason(path, _RENDERED) is None
    assert main(["check", str(path)]) == 0
    capsys.readouterr()

    path.write_text(_RENDERED.replace("| Memory | 1 |", "| Memory | 2 |"))
    assert main(["check", str(path)]) == 1
    assert capsys.readouterr().err == (
        f"{path} is stale — the eval cases no longer render to it; run `make fix` to "
        "regenerate it, and commit the result\n"
        "--- committed\n"
        "+++ generated from the cases\n"
        "@@ -10,7 +10,7 @@\n"
        " |---|---|\n"
        " | Conversation machine | 3 |\n"
        " | Teaching a routine | 0 |\n"
        "-| Memory | 2 |\n"
        "+| Memory | 1 |\n"
        " | Answering from the web | 0 |\n"
        " | Standing jobs | 0 |\n"
        " | Notifications | 0 |\n"
    )

    def _uncollectable() -> list[CatalogueEntry]:
        raise CatalogueError("catalogue: the eval cases could not be collected")

    monkeypatch.setattr(catalogue, "collect", _uncollectable)
    assert main(["check", str(path)]) == 1
    assert capsys.readouterr().err == "catalogue: the eval cases could not be collected\n"
    assert main(["check"]) == 2
    assert main(["publish", str(path)]) == 2
