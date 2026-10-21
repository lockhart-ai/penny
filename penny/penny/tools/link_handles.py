"""Link handles — a run's numbered link table (#2717).

Links travel as numbered handles end to end: the model reads ``[12]`` instead of
a URL and writes ``[12]`` back, and Python expands the handle when a message is
finalised, so the model never types a URL.  Measured on 7 logged page reads
(gemma, thinking off): reader input dropped 42%, and no link that wasn't on the
page and no handle that didn't exist ever appeared in a draw.

One :class:`HandleTable` per run — a chat turn, a collector cycle, or a job-report
delivery turn — owned by Python and passed as an explicit parameter, never ambient
state.  It is assigned on first sight: the first time a URL is rendered to the
model it gets the next number, and the same URL keeps its number for the whole
run, so a handle the model read from one tool result is the same number it cites
back.

Handles are a RENDERING.  Stored state keeps real URLs: memory entries are
expanded back to their URLs before they are written (``collection_write``), and
the browse-results log stores the un-converted sections.  Conversion happens only
at each seam where the model READS — page sections, search results, history,
memory read renders, the job-report render.

The table is mutable only by registration (a URL's number never changes once
assigned); everything else is a pure read.  An empty table is honest: it means
this run rendered no handles, so no citation can dangle and no prose ``[12]``
is ever mistaken for one.
"""

from __future__ import annotations

import re

# A markdown link ``[text](url)`` — the form the page reader renders links in.
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
# An image ``![alt](url)`` — dropped before the link pass, so an image is never
# registered as a link handle.
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# A line that is nothing but a URL — the form the reply-side source fallback
# appends and some pages carry.  Only a STANDALONE line converts: a URL inside
# prose is left alone (the prototype converted markdown links only and measured
# clean, so the conservative form is what shipped).
_BARE_URL_LINE_RE = re.compile(r"^(https?://\S+)[ \t]*$", re.MULTILINE)
# A handle citation — bare ``[12]`` or the ``[text][12]`` form the page reader
# itself renders.  ``[12]`` alone is ambiguous with prose ("[12] people said"),
# so expansion rewrites only numbers the table actually holds, and the dangling
# check only fires while the table holds at least one link.
_HANDLE_RE = re.compile(r"\[([^\]]*)\]\[(\d+)\]|\[(\d+)\]")
# A browse section's own header line — the page's address (or an error's failed
# one) after the header text.
_BROWSE_HEADER_URL_RE = re.compile(r"^(## browse(?: error| dropped)?: )(https?://\S+)\s*$")

# The one plain line the page reader renders where handles appear — the exact
# wording the prototype measured (no dangling handles, zero unsourced URLs).
PAGE_CITATION_LINE = (
    "Links on this page are numbered, like [12]. When you give a link, "
    "write its number in brackets instead of the address."
)
# The same instruction for seams that are not a page: history, memory read
# renders, the job-report render.
CITATION_LINE = (
    "Links are numbered, like [12]. When you give a link, write its number "
    "in brackets instead of the address."
)


class HandleTable:
    """One run's numbered link table: number → URL, assigned on first sight."""

    def __init__(self) -> None:
        self._by_number: dict[int, str] = {}
        self._by_url: dict[str, int] = {}

    def __bool__(self) -> bool:
        """Whether any link has been rendered as a handle this run."""
        return bool(self._by_number)

    def __len__(self) -> int:
        """How many links this run has rendered as handles — the growth a render
        reads to decide whether its own citation line is new information."""
        return len(self._by_number)

    def register(self, url: str) -> int:
        """The handle for ``url`` — assigned on first sight, stable after."""
        existing = self._by_url.get(url)
        if existing is not None:
            return existing
        number = len(self._by_number) + 1
        self._by_number[number] = url
        self._by_url[url] = number
        return number

    def url_for(self, number: int) -> str | None:
        """The URL a handle stands for, or ``None`` when it dangles."""
        return self._by_number.get(number)

    def convert(self, text: str) -> str:
        """``text`` with every link replaced by its handle — the model-facing form.

        Images are dropped first (an image is not a link the model can cite);
        then ``[text](url)`` becomes ``[text][n]``, and a line that is nothing
        but a URL becomes ``[n]``.  Every URL converted is registered, so the
        numbers are the run's."""
        text = _IMAGE_RE.sub("", text)
        text = _MARKDOWN_LINK_RE.sub(self._link_sub, text)
        return _BARE_URL_LINE_RE.sub(self._bare_sub, text)

    def convert_browse_section(self, section: str) -> str:
        """One browse section (a ``## browse …:`` header plus its body) in the
        model-facing form: the section header's own URL becomes its handle too —
        the model never reads a raw URL anywhere in the section."""
        header, _, body = section.partition("\n")
        converted = _BROWSE_HEADER_URL_RE.sub(self._header_sub, header)
        return f"{converted}\n{self.convert(body)}" if body else converted

    def expand(self, text: str) -> str:
        """``text`` with every handle the table holds replaced by its URL — the
        form a message is finalised in, so the conversation record keeps real
        URLs.  A ``[text][n]`` citation keeps its text as a markdown link; a
        bare ``[n]`` becomes the bare URL.  A handle not in the table is left
        exactly as written (the validator rejects it before sending)."""
        return _HANDLE_RE.sub(self._expand_sub, text)

    def dangling(self, text: str) -> list[int]:
        """The handle numbers ``text`` cites that the table does not hold, in
        order, without repeats.  Empty while the table itself is empty — with
        no links rendered, a bracketed number is prose, not a failed citation."""
        if not self._by_number:
            return []
        seen: list[int] = []
        for match in _HANDLE_RE.finditer(text):
            number = int(match.group(2) or match.group(3))
            if number not in self._by_number and number not in seen:
                seen.append(number)
        return seen

    def dangling_note(self, numbers: list[int]) -> str:
        """The honest rejection a dangling citation draws: which handles don't
        exist and which do — the action-space fact the re-draw works from."""
        cited = ", ".join(f"[{n}]" for n in numbers)
        if self._by_number:
            low, high = min(self._by_number), max(self._by_number)
            return (
                f"There is no link {cited} in this turn; the links you have are [{low}]–[{high}]."
            )
        return f"There is no link {cited} in this turn; there are no numbered links."

    # ── substitution helpers ──────────────────────────────────────────────

    def _link_sub(self, match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        return f"[{label}][{self.register(url)}]"

    def _bare_sub(self, match: re.Match) -> str:
        return f"[{self.register(match.group(1))}]"

    def _header_sub(self, match: re.Match) -> str:
        prefix, url = match.group(1), match.group(2)
        return f"{prefix}[{self.register(url)}]"

    def _expand_sub(self, match: re.Match) -> str:
        text, linked, bare = match.group(1), match.group(2), match.group(3)
        url = self._by_number.get(int(linked or bare))
        if url is None:
            return match.group(0)
        return f"[{text}]({url})" if text else url
