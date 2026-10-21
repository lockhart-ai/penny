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
# A bare URL anywhere in the text — the run of non-space characters a scheme
# opens.  The stored-digest path is why this exists: a collector entry reads
# ``Headline text - https://www.example.com/news/story-9.7387777``, and a URL
# the model READS must arrive as a handle or the next turn retypes it.
_BARE_URL_RE = re.compile(r"https?://[^\s]+")
# Marks that, at a bare URL's END, belong to the prose around it rather than to
# the address — a sentence's full stop, a comma, emphasis — and the closers
# decided by PAIRING: a ``)`` stays in the URL when the URL itself opened one
# (a Wikipedia link like ``/wiki/Dune_(novel)``) and leaves when it did not
# (``(see https://x.com/a)``).  The same address-reading the hallucinated-URL
# validator applies to a reply's URLs, declared here for the bare pass.
_PROSE_TAIL_MARKS = ".,;:!?*_"
_PAIR_CLOSER = {")": "(", "]": "[", "}": "{"}
# A handle citation — bare ``[12]`` or the ``[text][12]`` form the page reader
# itself renders.  ``[12]`` alone is ambiguous with prose ("[12] people said"),
# so expansion rewrites only numbers the table actually holds, and the dangling
# check only fires while the table holds at least one link.
_HANDLE_RE = re.compile(r"\[([^\]]*)\]\[(\d+)\]|\[(\d+)\]")
# A browse section's own header line — the page's address (or an error's failed
# one) after the header text.
_BROWSE_HEADER_URL_RE = re.compile(r"^(## browse(?: error| dropped)?: )(https?://\S+)\s*$")


def _address_only(url: str) -> str:
    """A bare URL's address without the prose marks it ran into at its end."""
    while url:
        last = url[-1]
        if last in _PROSE_TAIL_MARKS:
            url = url[:-1]
            continue
        opener = _PAIR_CLOSER.get(last)
        if opener is not None and url.count(last) > url.count(opener):
            url = url[:-1]
            continue
        break
    return url


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
        then every markdown ``[text](url)`` becomes ``[text][n]``, and every bare
        ``http(s)://`` URL that remains becomes ``[n]`` — anywhere in a line,
        not only standing alone, with the prose punctuation it ran into (a
        sentence's full stop, a closing paren the URL never opened) left around
        the handle.  A URL already inside a markdown link's ``(...)`` is gone
        with that link before the bare pass runs, so nothing double-converts.
        Every URL converted is registered, so the numbers are the run's."""
        text = _IMAGE_RE.sub("", text)
        text = _MARKDOWN_LINK_RE.sub(self._link_sub, text)
        return self._convert_bare(text)

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
        number = self.register(url)
        if not label.strip() or label.strip() == url:
            # An empty label, or a link whose text is its own URL — the handle
            # alone: the URL must not survive as readable text.
            return f"[{number}]"
        return f"[{self._convert_bare(label)}][{number}]"

    def _convert_bare(self, text: str) -> str:
        """The bare-URL pass over ``text`` — every scheme-opening run becomes its
        handle, with prose tail marks left around it."""
        return _BARE_URL_RE.sub(self._bare_sub, text)

    def _bare_sub(self, match: re.Match) -> str:
        url = _address_only(match.group(0))
        if not url:
            return match.group(0)
        return f"[{self.register(url)}]{match.group(0)[len(url) :]}"

    def _header_sub(self, match: re.Match) -> str:
        prefix, url = match.group(1), match.group(2)
        return f"{prefix}[{self.register(url)}]"

    def _expand_sub(self, match: re.Match) -> str:
        text, linked, bare = match.group(1), match.group(2), match.group(3)
        url = self._by_number.get(int(linked or bare))
        if url is None:
            return match.group(0)
        return f"[{text}]({url})" if text else url
