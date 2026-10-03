"""The catalogue of supported behaviours: every eval case, grouped by area (#2214).

The eval suite is the list of what Penny is expected to do, and ``docs/eval-catalogue.md`` is
that list as a document.  It is generated from the cases, so it cannot say something the
cases do not.

A case declares two things to its driver: the ``area`` of Penny it is about (:class:`Area`,
a closed set) and, where it covers one, the machine ``edge`` it moves along.  Its
:class:`Layer` is not declared — it is which driver the case uses.

Collecting the cases runs no model.  With ``EVAL_CATALOGUE`` naming a file, every driver
validates the case it was handed, appends it to that file as a :class:`CatalogueEntry`, and
skips the test before any sample, endpoint check or database is stood up.  :func:`collect`
runs the suite that way in a child pytest and reads the file back.

CLI (run in the test container by ``make fix`` and ``make check``):

  ``python -m penny.tests.eval.utils.catalogue write <path>``
      → collects the cases and writes the catalogue to ``<path>``.
  ``python -m penny.tests.eval.utils.catalogue check <path>``
      → collects the cases and exits 1, saying to run ``make fix``, when ``<path>`` is not
      what they render to.

Both exit 1, naming the case, when a case states no behaviour, declares no area, names an
edge the machine does not have, or shares its id with another case.
"""

from __future__ import annotations

import difflib
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from penny.conversation_machine import OUT_EDGES, ConversationState


class Area(StrEnum):
    """The part of Penny a case is about.  Closed: a case names one of these or is refused.
    Declaration order is the order the catalogue renders them in."""

    CONVERSATION_MACHINE = "conversation machine"
    TEACHING_A_ROUTINE = "teaching a routine"
    MEMORY = "memory"
    ANSWERING_FROM_THE_WEB = "answering from the web"
    STANDING_JOBS = "standing jobs"
    NOTIFICATIONS = "notifications"
    BACKGROUND_COLLECTORS = "background collectors"
    READING_A_PAGE = "reading a page"
    CHAT_TOOLS = "chat tools"


class Layer(StrEnum):
    """How much of Penny a case drives.  Each driver states its own; a case never does."""

    CLASSIFIER_DRAW = "classifier draw"
    WHOLE_TURN = "whole turn"
    MICRO_CONTEXT = "micro-context"
    COLLECTOR_CYCLE = "collector cycle"


# One move of the conversation machine: the state it is drawn from, and the state it lands in.
Edge = tuple[ConversationState, ConversationState]


class CatalogueEntry(BaseModel):
    """One case as the catalogue lists it."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    area: Area
    edge: Edge | None
    layer: Layer
    sentence: str
    wording: str
    node_id: str


class CatalogueError(Exception):
    """The cases could not be collected into a catalogue."""


# The environment variable that puts the drivers in catalogue mode: the file each case is
# appended to.
CATALOGUE_ENV = "EVAL_CATALOGUE"

# Why a test is skipped in catalogue mode.
CATALOGUED = "catalogued — no sample runs in catalogue mode"

_NO_AREA = (
    "{case_id}: a case must declare its area — area=Area.<NAME>, one of: {areas} "
    "(penny.tests.eval.utils.catalogue)"
)
_NOT_AN_EDGE = (
    "{case_id}: edge=({origin}, {landing}) is not an edge of the conversation machine — "
    "from {origin} it offers {offered}"
)
_NO_OUT_EDGES = "nothing"
_DUPLICATE_ID = "{case_id}: declared by more than one case — {nodes}; a case id names one case"
_COLLECTION_FAILED = (
    "catalogue: the eval cases could not be collected — every case states its behaviour, "
    "declares its area, and names only an edge the machine has:\n{output}"
)


def declare(
    *,
    case_id: str,
    layer: Layer,
    sentence: str,
    area: Area | None,
    edge: Edge | None,
    wording: str,
    node_id: str,
) -> CatalogueEntry:
    """The case as the catalogue lists it, or a refusal naming what it left out."""
    if not isinstance(area, Area):
        areas = ", ".join(member.value for member in Area)
        raise ValueError(_NO_AREA.format(case_id=case_id, areas=areas))
    if edge is not None:
        _require_machine_edge(case_id, edge)
    return CatalogueEntry(
        case_id=case_id,
        area=area,
        edge=edge,
        layer=layer,
        sentence=sentence,
        wording=wording,
        node_id=node_id,
    )


def _require_machine_edge(case_id: str, edge: Edge) -> None:
    """Refuse an edge that is not in the machine's own table."""
    origin, landing = edge
    offered = OUT_EDGES.get(origin, ())
    if landing in offered:
        return
    raise ValueError(
        _NOT_AN_EDGE.format(
            case_id=case_id,
            origin=str(origin),
            landing=str(landing),
            offered=", ".join(offered) or _NO_OUT_EDGES,
        )
    )


def catalogue_path() -> Path | None:
    """Where catalogue mode appends its entries, or ``None`` outside catalogue mode."""
    destination = os.environ.get(CATALOGUE_ENV)
    return Path(destination) if destination else None


def record(destination: Path, entry: CatalogueEntry) -> None:
    """Append one case to the file a collection is being gathered in."""
    with destination.open("a") as lines:
        lines.write(entry.model_dump_json() + "\n")


def load(source: Path) -> list[CatalogueEntry]:
    """Every case a collection gathered, refused when two share an id."""
    lines = source.read_text().splitlines() if source.exists() else []
    entries = [CatalogueEntry.model_validate_json(line) for line in lines if line.strip()]
    _refuse_duplicate_ids(entries)
    return entries


def _refuse_duplicate_ids(entries: Sequence[CatalogueEntry]) -> None:
    counts = Counter(entry.case_id for entry in entries)
    for case_id, count in counts.items():
        if count > 1:
            nodes = ", ".join(entry.node_id for entry in entries if entry.case_id == case_id)
            raise CatalogueError(_DUPLICATE_ID.format(case_id=case_id, nodes=nodes))


# Where the case files live, and how the child pytest is asked for them: every eval-marked
# test under the suite, one line per failure.  This folder is left out — it holds the shared
# machinery and no case, and its one eval-marked test is a probe of a live endpoint.
_MACHINERY_DIR = Path(__file__).resolve().parent
_CASES_DIR = _MACHINERY_DIR.parent
_PYTEST_MODULE = ("-m", "pytest")
_PYTEST_ARGS = ("-m", "eval", "-q", "--tb=line", "-p", "no:cacheprovider")
_IGNORE = "--ignore={directory}"
_GATHERED_FILENAME = "entries.jsonl"


def collect() -> list[CatalogueEntry]:
    """Every case in the suite, gathered by running it in catalogue mode — no model call."""
    command = [
        sys.executable,
        *_PYTEST_MODULE,
        str(_CASES_DIR),
        _IGNORE.format(directory=_MACHINERY_DIR),
        *_PYTEST_ARGS,
    ]
    with tempfile.TemporaryDirectory() as scratch:
        gathered = Path(scratch) / _GATHERED_FILENAME
        ran = subprocess.run(
            command,
            env={**os.environ, CATALOGUE_ENV: str(gathered)},
            capture_output=True,
            text=True,
            check=False,
        )
        if ran.returncode != 0:
            raise CatalogueError(_COLLECTION_FAILED.format(output=ran.stdout + ran.stderr))
        return load(gathered)


# ── Rendering ────────────────────────────────────────────────────────────────

_HEAD = """\
# Eval catalogue

The behaviours Penny supports, one row per eval case, grouped by the area of Penny each case \
is about. This file is generated from the cases: `make fix` writes it and `make check` fails \
when it is out of date. It is not edited by hand.

A case declares its area and, where it covers one, the edge of the conversation machine it \
moves along. The first wording is the first of the wordings the case is driven in."""

_SUMMARY_HEAD = "## Cases by area\n\n| Area | Cases |\n|---|---|"
_SUMMARY_TOTAL = "Total"
_AREA_TABLE_HEAD = "| Case | Behaviour | First wording | Edge |\n|---|---|---|---|"
_NO_CASES = "No cases."
_COVERAGE_HEAD = f"""\
## Machine edge coverage

One row per edge of the conversation machine. A classifier draw covers an edge when a case \
draws that decision on its own; a whole turn covers it when a case drives the chat turn that \
makes the move.

| Edge | {Layer.CLASSIFIER_DRAW.capitalize()} | {Layer.WHOLE_TURN.capitalize()} |
|---|---|---|"""
_UNCOVERED = "—"
_ARROW = " → "


def render(entries: Sequence[CatalogueEntry]) -> str:
    """The whole catalogue document for these cases."""
    ordered = sorted(entries, key=lambda entry: entry.case_id)
    sections = [
        _HEAD,
        _summary(ordered),
        *(_area_section(area, ordered) for area in Area),
        _edge_coverage(ordered),
    ]
    return "\n\n".join(sections) + "\n"


def _summary(entries: Sequence[CatalogueEntry]) -> str:
    """How many cases each area has."""
    counts = Counter(entry.area for entry in entries)
    rows = [f"| {area.capitalize()} | {counts[area]} |" for area in Area]
    return "\n".join([_SUMMARY_HEAD, *rows, f"| {_SUMMARY_TOTAL} | {len(entries)} |"])


def _area_section(area: Area, entries: Sequence[CatalogueEntry]) -> str:
    """One area's cases as a table."""
    rows = [_case_row(entry) for entry in entries if entry.area is area]
    body = "\n".join([_AREA_TABLE_HEAD, *rows]) if rows else _NO_CASES
    return f"## {area.capitalize()}\n\n{body}"


def _case_row(entry: CatalogueEntry) -> str:
    edge = _edge_label(entry.edge) if entry.edge is not None else ""
    cells = [_code(entry.case_id), _cell(entry.sentence), _cell(entry.wording), edge]
    return f"| {' | '.join(cells)} |"


def _edge_coverage(entries: Sequence[CatalogueEntry]) -> str:
    """Which cases cover each machine edge, by the layer they drive."""
    rows = [
        f"| {_edge_label((origin, landing))} "
        f"| {_covering(entries, (origin, landing), Layer.CLASSIFIER_DRAW)} "
        f"| {_covering(entries, (origin, landing), Layer.WHOLE_TURN)} |"
        for origin, landings in OUT_EDGES.items()
        for landing in landings
    ]
    return "\n".join([_COVERAGE_HEAD, *rows])


def _covering(entries: Sequence[CatalogueEntry], edge: Edge, layer: Layer) -> str:
    """The cases covering one edge at one layer, or the dash that says there are none."""
    covering = [
        _code(entry.case_id) for entry in entries if entry.edge == edge and entry.layer is layer
    ]
    return ", ".join(covering) or _UNCOVERED


def _edge_label(edge: Edge) -> str:
    origin, landing = edge
    return f"{origin}{_ARROW}{landing}"


def _code(text: str) -> str:
    return f"`{text}`"


def _cell(text: str) -> str:
    """Text as one table cell: on one line, with the column separator escaped."""
    return " ".join(text.split()).replace("|", "\\|")


# ── Staleness ────────────────────────────────────────────────────────────────

_MISSING = "{path} is missing — run `make fix` to generate it from the eval cases"
_STALE = (
    "{path} is stale — the eval cases no longer render to it; run `make fix` to regenerate "
    "it, and commit the result\n{diff}"
)
_COMMITTED = "committed"
_GENERATED = "generated from the cases"


def stale_reason(path: Path, rendered: str) -> str | None:
    """Why the committed catalogue is not what the cases render to, or ``None`` when it is."""
    if not path.exists():
        return _MISSING.format(path=path)
    committed = path.read_text()
    if committed == rendered:
        return None
    diff = difflib.unified_diff(
        committed.splitlines(), rendered.splitlines(), _COMMITTED, _GENERATED, lineterm=""
    )
    return _STALE.format(path=path, diff="\n".join(diff))


# ── CLI: catalogue write <path> | catalogue check <path> ─────────────────────

WRITE_CMD = "write"
CHECK_CMD = "check"
USAGE = (
    f"usage: python -m penny.tests.eval.utils.catalogue {WRITE_CMD} <path>\n"
    f"       python -m penny.tests.eval.utils.catalogue {CHECK_CMD} <path>"
)
_WROTE = "catalogue: {count} cases → {path}"
_CURRENT = "catalogue: {path} is current ({count} cases)"


def main(argv: list[str]) -> int:
    """Dispatch the two verbs.  ``write`` regenerates the file; ``check`` exits 1 when it is
    stale.  Either exits 1 when the cases cannot be collected.  Bad args → 2."""
    if len(argv) != 2 or argv[0] not in (WRITE_CMD, CHECK_CMD):
        print(USAGE, file=sys.stderr)
        return 2
    path = Path(argv[1])
    try:
        entries = collect()
    except CatalogueError as error:
        print(error, file=sys.stderr)
        return 1
    rendered = render(entries)
    if argv[0] == WRITE_CMD:
        path.write_text(rendered)
        print(_WROTE.format(count=len(entries), path=path))
        return 0
    reason = stale_reason(path, rendered)
    if reason is not None:
        print(reason, file=sys.stderr)
        return 1
    print(_CURRENT.format(count=len(entries), path=path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
