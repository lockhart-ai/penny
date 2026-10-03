"""What a round was GIVEN, read off its prompt log and split by who put it there (#2203).

A provenance claim asks whether a value the round stored or said traces to something it was
given.  Weighed against one undivided blob, a number was sourced by anything that happened to
print it: a prompt's numbered list, the count a read is headed by, the number an entry is
listed under, the day of a stamp.  MEASURED: a stored recipe entry with an invented `2 tbsp`
and `1 tsp` read as fully sourced.

So the split is made HERE, where the turns still have their structure, and never by looking
for scaffolding-shaped text in a blob afterwards:

  CONTENT       what somebody or something stated to the round
                · every ``user`` turn — the user's message, a seeded turn, the document a
                  micro-context is handed (a page, the round's turns, a slice of the registry)
                · every ``tool`` turn's PAYLOAD — the tool's own message, without the line the
                  framework narrates it with
                · the entries a store read laid out, as the entries they are: key and content

  SCAFFOLDING   what the framework wrapped that in
                · every ``system`` prompt — instructions, the self-state header, a run's record
                · the narration line a result opens with, ending in its ``(<tool> result)`` tag
                · the frame the entry render lays entries out in: the count line, each entry's
                  list number, each entry's stamp

  MOMENTS       the timestamps the framework rendered, wherever they stand — the date and time
                the round was told, the stamp on an entry, a run, a change.  A stamp is lifted
                out of the content as the moment it is, so its digits are no quantity.

Nothing here knows a tool's name.  The line between a result's payload and its frame is the
one ``Tool.format_result`` draws for every tool, so a plugin's verb nobody has heard of is
split exactly as a built-in is; and what a tool words into its own message — a count, an id —
is its payload, taken as it comes.  The one render read INSIDE a payload is the store's own
entry render (``format_entries``), because it is the framework's, shared by every read and by
the collector's prompt, and its count line and list numbers are the measured leak.

The cost, stated: a system prompt is scaffolding whole apart from the entries it lays out, so
a job's stored terms are content only where a content turn carries them — the classifier's
document lists every running job with its schedule, and a read returns the rest.  Assistant
turns stay out entirely, as they always have: a value Penny invents early in a turn rides
into the history and would then source itself.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, NamedTuple

from penny.constants import PennyConstants
from penny.datetime_utils import format_log_timestamp
from penny.llm.models import strip_harmony_control_tokens
from penny.tests.eval.utils.cohort import Given
from penny.tools.base import RESULT_TAG


class Role(StrEnum):
    """The turn roles a round is GIVEN.  Assistant turns are absent by design."""

    USER = "user"
    TOOL = "tool"
    SYSTEM = "system"


_GIVEN_ROLES = frozenset(Role)
_LINE_BREAK = "\n"

# The stamp ``format_log_timestamp`` renders.  Spelled here to PARSE one back, and every
# candidate is re-rendered through that function before it counts, so a stamp is only ever
# what production would have written; ``test_given`` pins the two against each other.
_LOG_STAMP_FORMAT = "%Y-%m-%d %H:%M UTC"
_LOG_STAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC")
# The told date ends in the zone's abbreviation, which ``strptime`` reads for the local zone
# alone — so the abbreviation is cut off and the rest parsed by production's own format.
_ZONE_SUFFIX = " %Z"
_TOLD_FORMAT = PennyConstants.CURRENT_DATETIME_FORMAT.removesuffix(_ZONE_SUFFIX)
_WORD_GAP = " "

# The entry render, as ``format_entries`` lays it out: `<n> entries from `<source>`…:` over one
# `<n>. [<stamp>] <key and content>` line per entry, numbered from one.
_ENTRY_NUMBERED = "{number}. ["
_STAMP_CLOSED = "] "
_COUNT_LINE_OPENS = "{count} {noun} from `"
_COUNT_LINE_CLOSES = ":"
_ONE_ENTRY, _MANY_ENTRIES = "entry", "entries"


class _Kind(StrEnum):
    ENTRY = "entry"
    CONTINUES = "continues"
    FRAME = "frame"
    OTHER = "other"


class _Line(NamedTuple):
    kind: _Kind
    text: str


_AN_ENTRY = frozenset({_Kind.ENTRY, _Kind.CONTINUES})
_STATED_BY_ROLE = {
    Role.USER: _AN_ENTRY | {_Kind.OTHER},
    Role.TOOL: _AN_ENTRY | {_Kind.OTHER},
    Role.SYSTEM: _AN_ENTRY,
}


def read_given(turns: Iterable[Mapping[str, Any]]) -> Given:
    """Everything ``turns`` handed the round — the whole text, what in it was stated, and the
    moments it was told."""
    everything: list[str] = []
    stated: list[str] = []
    moments: set[datetime] = set()
    calls: dict[str, str] = {}
    for turn in turns:
        calls.update(_calls_made(turn))
        if turn.get("role") not in _GIVEN_ROLES:
            continue
        text = str(turn.get("content") or "")
        everything.append(text)
        moments.update(_moments_in(text))
        stated.append(_stated(Role(turn["role"]), text, calls.get(turn.get("tool_call_id", ""))))
    return Given(
        _LINE_BREAK.join(everything),
        content=_LINE_BREAK.join(stated),
        moments=sorted(moments),
    )


def _calls_made(turn: Mapping[str, Any]) -> dict[str, str]:
    """The tool each call in an assistant turn named, by the call's id — what pairs a result
    with the tag it was framed under.  The name is read the way production reads it."""
    return {
        call["id"]: strip_harmony_control_tokens((call.get("function") or {}).get("name") or "")
        for call in turn.get("tool_calls") or []
        if call.get("id")
    }


def _stated(role: Role, text: str, tool_name: str | None) -> str:
    """The part of one turn that was STATED to the round."""
    payload = _tool_payload(text, tool_name) if role == Role.TOOL else text
    lines = [line for line in payload.split(_LINE_BREAK) if _told_moment(line) is None]
    kinds = _STATED_BY_ROLE[role]
    kept = [line.text for line in _read_entry_renders(lines) if line.kind in kinds]
    return _without_stamps(_LINE_BREAK.join(kept))


def _tool_payload(text: str, tool_name: str | None) -> str:
    """A tool turn's PAYLOAD: what the tool returned, without the narration the framework
    opens it with.

    ``Tool.format_result`` writes every result as ``<narration> <tag>\\n<message>``, the tag
    naming the call's own tool, so the payload is what follows the tag.  A turn that carries
    no such tag was not framed, and is payload whole."""
    if not tool_name:
        return text
    tag = RESULT_TAG.format(tool_name=tool_name)
    _narration, framed, payload = text.partition(f"{tag}{_LINE_BREAK}")
    if framed:
        return payload
    return "" if text.endswith(tag) else text


def _read_entry_renders(lines: Sequence[str]) -> list[_Line]:
    """``lines`` with every block the entry render laid out read as its entries."""
    return _EntryRenders().read(lines)


class _EntryRenders:
    """One text's lines, read for the blocks the entry render laid out in it.

    An entry opens on the line carrying its number and its stamp, and runs until the next one
    opens or a blank line.  The count line is settled when its block ends, because only then
    is the count it has to state known."""

    def __init__(self) -> None:
        self._read: list[_Line] = []
        self._expected = 1
        self._headed_at: int | None = None

    def read(self, lines: Sequence[str]) -> list[_Line]:
        for line in lines:
            self._take(line)
        self._settle_count_line()
        return self._read

    def _take(self, line: str) -> None:
        opened = self._opens(line)
        if opened is None:
            self._read.append(_Line(self._continues(line), line))
            return
        number, text = opened
        if number == 1:
            self._settle_count_line()
            self._headed_at = len(self._read) - 1
        self._read.append(_Line(_Kind.ENTRY, text))
        self._expected = number + 1

    def _opens(self, line: str) -> tuple[int, str] | None:
        """The entry ``line`` opens — the next one of the block being read, or the first of a
        new block — with what it says."""
        for number in (self._expected, 1):
            text = _entry_text(line, number)
            if text is not None:
                return number, text
        return None

    def _continues(self, line: str) -> _Kind:
        """A line under an entry is more of that entry, until a blank one."""
        running = bool(self._read) and self._read[-1].kind in _AN_ENTRY
        return _Kind.CONTINUES if running and line.strip() else _Kind.OTHER

    def _settle_count_line(self) -> None:
        """Mark the line above the block just read as FRAME when it is that block's count."""
        if self._headed_at is None or self._headed_at < 0:
            return
        count = self._expected - 1
        noun = _ONE_ENTRY if count == 1 else _MANY_ENTRIES
        line = self._read[self._headed_at].text
        opens = line.startswith(_COUNT_LINE_OPENS.format(count=count, noun=noun))
        if opens and line.endswith(_COUNT_LINE_CLOSES):
            self._read[self._headed_at] = _Line(_Kind.FRAME, line)


def _entry_text(line: str, number: int) -> str | None:
    """What entry ``number`` says on the line that opens it — its key and content, after the
    number and the stamp — or ``None`` when the line does not open that entry."""
    opened = _ENTRY_NUMBERED.format(number=number)
    if not line.startswith(opened):
        return None
    stamp, closed, text = line[len(opened) :].partition(_STAMP_CLOSED)
    return text if closed and _stamp_moment(stamp) is not None else None


def _moments_in(text: str) -> list[datetime]:
    """Every moment ``text`` renders: the date and time the round was told, and each stamp."""
    told = (_told_moment(line) for line in text.split(_LINE_BREAK))
    stamped = (_stamp_moment(stamp) for stamp in _LOG_STAMP.findall(text))
    return [moment for moment in (*told, *stamped) if moment is not None]


def _told_moment(line: str) -> datetime | None:
    """The moment a ``Current date and time:`` line tells, or ``None`` for any other line."""
    prefix = PennyConstants.CURRENT_DATETIME_PREFIX
    if not line.startswith(prefix):
        return None
    stamp, _, _zone = line[len(prefix) :].rpartition(_WORD_GAP)
    try:
        return datetime.strptime(stamp, _TOLD_FORMAT)
    except ValueError:
        return None


def _stamp_moment(stamp: str) -> datetime | None:
    """The moment a log stamp says, or ``None`` when production would not have written it."""
    try:
        moment = datetime.strptime(stamp, _LOG_STAMP_FORMAT)
    except ValueError:
        return None
    return moment if format_log_timestamp(moment) == stamp else None


def _without_stamps(text: str) -> str:
    """``text`` with each log stamp taken out — it was read as a moment, not as digits."""
    return _LOG_STAMP.sub(lambda stamp: "" if _stamp_moment(stamp.group()) else stamp.group(), text)
