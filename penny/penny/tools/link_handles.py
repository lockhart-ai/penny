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
from collections.abc import Mapping
from urllib.parse import urlparse

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
# A handle citation — five forms: bare ``[12]``; the ``[text][12]`` form the
# page reader itself renders; the two hybrids a model composes when it
# treats the handle as a markdown link — ``[12](https://site.example)`` (the
# number as link text, a site root it knows as the target — the stored-digest
# sample that stored ``story-url(site-root)``) and ``[text](12)`` (the number
# written as the target); and a list form ``[4, 6]`` citing several handles in
# one bracket.  The handle IS the address, so a parenthesized or
# bracketed rest glued to a citation is part of the citation and expands away
# with it.  ``[12]`` alone is ambiguous with prose ("[12] people said"), so
# expansion rewrites only numbers the table actually holds, and the dangling
# check only fires while the table holds at least one link.
_HANDLE_RE = re.compile(
    r"\[([^\]]*)\]\[(\d+)\]"
    r"|\[(\d+)\]\(([^)]*)\)"
    r"|\[([^\]]+)\]\((\d+)\)"
    r"|\[(\d+(?:,\s*\d+)+)\]"
    r"|\[(\d+)\]"
)
_MULTI_CITATION_SEPARATOR = ", "
# A browse section's own header line — the page's address (or an error's failed
# one) after the header text.  Looked up, never registered (#2735): the page
# itself takes no number, so its first content link keeps the next one.
_BROWSE_HEADER_URL_RE = re.compile(r"^(## browse(?: error| dropped)?: )(https?://\S+)\s*$")


def _cited_numbers(match: re.Match) -> list[int]:
    """All handle numbers a citation form cites — a list of numbers, one for
    single citation forms, multiple for the list form ``[4, 6]``."""
    text, linked, hybrid_num, _target, label, target_num, multi_numbers, bare = match.groups()
    if multi_numbers is not None:
        return [int(n) for n in multi_numbers.split(",")]
    num = int(next(g for g in (linked, hybrid_num, target_num, bare) if g is not None))
    return [num]


def _registrable_domain(url: str) -> str:
    """``url``'s own host, scheme and path dropped — the shape a bare handle
    is annotated with (#2832): a domain names what a link IS without inviting
    anyone to copy a path out of it.  Falls back to the whole ``url`` on a
    scheme-less or otherwise unparseable address, which handle registration
    never actually hands this (every registered URL opens ``https?://``), but
    a fallback that degrades to the input is safer than one that raises."""
    return urlparse(url).hostname or url


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


# The one plain line rendered where handles appear.  It states the number's
# COMPLETENESS, not just the form: a stored-digest sample read "[2]" as a
# reference without the address the ask demanded and browsed one site root
# eighteen times hunting for literal URLs that the write-side expansion would
# have supplied — the state never told her the number carries the address.
PAGE_CITATION_LINE = (
    "Links on this page are numbered, like [12]. The number is the link "
    "itself: when your message is sent or an entry is stored, [12] becomes "
    "the full address. So write [12] wherever the link belongs, and never "
    "look for the address."
)
# The same line for seams that are not a page: history, memory read renders,
# the job-report render.
CITATION_LINE = (
    "Links are numbered, like [12]. The number is the link itself: when your "
    "message is sent or an entry is stored, [12] becomes the full address. So "
    "write [12] wherever the link belongs, and never look for the address."
)


class HandleTable:
    """One run's numbered link table: number → URL, assigned on first sight."""

    def __init__(self) -> None:
        self._by_number: dict[int, str] = {}
        self._by_url: dict[str, int] = {}
        # A title is known only when a caller that actually read the page
        # (browse) supplies one at registration — never backfilled from
        # anywhere else, so a handle this run never fetched just renders its
        # domain alone rather than a guessed name (#2832).
        self._titles: dict[int, str] = {}

    def __bool__(self) -> bool:
        """Whether any link has been rendered as a handle this run."""
        return bool(self._by_number)

    def __len__(self) -> int:
        """How many links this run has rendered as handles — the growth a render
        reads to decide whether its own citation line is new information."""
        return len(self._by_number)

    def register(self, url: str, *, title: str | None = None) -> int:
        """The handle for ``url`` — assigned on first sight, stable after.

        ``title`` backfills the handle's title the first time one is known —
        a URL is often registered before its page is read (the pick-time
        citation a run assigns itself) and again once it is, so a later call
        naming the title fills the gap rather than needing to be first. An
        already-known title is never overwritten by a second, possibly
        different one."""
        existing = self._by_url.get(url)
        if existing is not None:
            if title and existing not in self._titles:
                self._titles[existing] = title
            return existing
        number = len(self._by_number) + 1
        self._by_number[number] = url
        self._by_url[url] = number
        if title:
            self._titles[number] = title
        return number

    def _describe(self, number: int) -> str:
        """``[n]`` with its registrable domain, and its page title when one is
        known, beside it — the render that makes a bare handle legible as a
        real address rather than an opaque reference worth resolving (#2832):
        ``[1] store.steampowered.com`` or, once a title is known,
        ``[1] store.steampowered.com — Big Pharma``. The full path never
        renders, so nothing invites copying one out by hand."""
        url = self._by_number.get(number)
        if url is None:
            return f"[{number}]"
        domain = _registrable_domain(url)
        title = self._titles.get(number)
        return f"[{number}] {domain} — {title}" if title else f"[{number}] {domain}"

    def _describe_full(self, number: int) -> str:
        """``[n]`` with its FULL address beside it — the render for the shape
        where the link IS the whole answer (#2832 arm C): a stored entry's own
        ``Source:`` line, and a page reader's answer that is entirely a link.
        Everywhere else a bare handle renders (a page body, a browse header),
        the number plus domain stays as ``_describe`` already has it."""
        url = self._by_number.get(number)
        return f"[{number}] {url}" if url is not None else f"[{number}]"

    def annotate(self, text: str) -> str:
        """``text`` described by its FULL address when ``text``, in its
        entirety, IS one handle citation — the shape a page reader's answer
        takes when the one fact asked for is a link (#2832 arm C): a bare
        ``[1]`` with nothing else around it IS the answer, so it carries the
        real address rather than a domain that still reads as partial. Text
        that carries anything besides the citation — a sentence, a second
        fact — is left exactly as it is: this names what a BARE answer stands
        for, it is not a second citation form stacked onto normal prose."""
        match = _HANDLE_RE.fullmatch(text.strip())
        if match is None:
            return text
        described = (self._describe_full(number) for number in _cited_numbers(match))
        return _MULTI_CITATION_SEPARATOR.join(described)

    def convert_entry(self, text: str) -> str:
        """``text`` with every link replaced by its handle, FULL-address form
        — a stored entry's own render (#2832 arm C): the citation on an
        entry's ``Source:`` line IS the fact the ask wants, so a bare or
        self-referential link shows its full address beside the number
        instead of just the domain.  A LABELLED markdown link keeps its
        ordinary ``[text][n]`` form unchanged — it already carries readable
        text, so this only touches the shape where the link is the whole
        value."""
        text = _IMAGE_RE.sub("", text)
        text = _MARKDOWN_LINK_RE.sub(self._link_sub_full, text)
        return _BARE_URL_RE.sub(self._bare_sub_full, text)

    def _link_sub_full(self, match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        # Registering here (not only in the bare pass below) keeps "assigned
        # on first sight" true in text order even for the collapsed branch,
        # which defers its own text to that bare pass rather than describing
        # the url twice (#2832 arm C) — register() is idempotent, so the bare
        # pass re-registering the same url is a no-op that returns this
        # number back.
        number = self.register(url)
        if not label.strip() or label.strip() == url:
            return url
        return f"[{self._convert_bare(label)}][{number}]"

    def _bare_sub_full(self, match: re.Match) -> str:
        url = _address_only(match.group(0))
        if not url:
            return match.group(0)
        number = self.register(url)
        return f"{self._describe_full(number)}{match.group(0)[len(url) :]}"

    @classmethod
    def from_mapping(cls, mapping: Mapping[int, str]) -> HandleTable:
        """A table rebuilt from a recorded ``{number: url}`` mapping — the shape
        ``mapping`` snapshots and the run record stores.  The harness expands a
        run's given text through the run's own recorded table, so a handle the
        model read from one surface expands to the URL it stood for."""
        table = cls()
        for number, url in sorted(mapping.items()):
            table._by_number[int(number)] = url
            table._by_url[url] = int(number)
        return table

    def mapping(self) -> dict[int, str]:
        """The table as ``{number: url}`` — a snapshot read for a run-record
        stamp (``MessageStore.set_run_link_handles``) and the harness's given
        expansion; mutating the copy changes nothing here."""
        return dict(self._by_number)

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
        model-facing form.  The header shows the page as the model gave it
        (#2735): the address it typed stays written out, and a page it cited by
        number comes back as the handle it already holds.  The page itself never
        takes a NEW number — one would sit where the page's first content link
        belongs, and the model pairs what it reads in order, so a headline's
        link lands on its neighbour.  Only links found INSIDE the page content
        get handles."""
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
        order, without repeats.  Every citation form counts — including the
        hybrids, where the number stands in the brackets or the target alike,
        and every number in a list form.  Empty while the table itself is empty
        — with no links rendered, a bracketed number is prose, not a failed
        citation."""
        if not self._by_number:
            return []
        seen: list[int] = []
        for match in _HANDLE_RE.finditer(text):
            for number in _cited_numbers(match):
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

    def drop_citations(self, text: str, numbers: list[int]) -> str:
        """``text`` with every citation of a number in ``numbers`` removed — the
        repair for a handle still dangling after the hallucinated-URL
        validator's one retry (#2721): the model gets no second nudge, so the
        still-bad citation is dropped instead of reaching the user.  A labelled
        form (``[text][n]``, ``[text](n)``) keeps its text; an unlabelled one
        (``[n]``, ``[n](target)``) drops outright; a list form (``[4, 6]``)
        keeps whichever numbers aren't in ``numbers``."""
        bad = set(numbers)

        def sub(match: re.Match) -> str:
            text_label, linked, hybrid_num, _target, label, target_num, multi_numbers, bare = (
                match.groups()
            )
            if multi_numbers is not None:
                kept = [n.strip() for n in multi_numbers.split(",") if int(n.strip()) not in bad]
                return f"[{', '.join(kept)}]" if kept else ""
            if linked is not None:
                return text_label if int(linked) in bad else match.group(0)
            if target_num is not None:
                return label if int(target_num) in bad else match.group(0)
            num = hybrid_num if hybrid_num is not None else bare
            return "" if int(num) in bad else match.group(0)

        return _HANDLE_RE.sub(sub, text)

    # ── substitution helpers ──────────────────────────────────────────────

    def _link_sub(self, match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        number = self.register(url)
        if not label.strip() or label.strip() == url:
            # An empty label, or a link whose text is its own URL — the handle
            # alone, described (#2832): the URL must not survive as readable
            # text, but the handle on its own carries nothing to trust either.
            return self._describe(number)
        return f"[{self._convert_bare(label)}][{number}]"

    def _convert_bare(self, text: str) -> str:
        """The bare-URL pass over ``text`` — every scheme-opening run becomes its
        handle, with prose tail marks left around it."""
        return _BARE_URL_RE.sub(self._bare_sub, text)

    def _bare_sub(self, match: re.Match) -> str:
        url = _address_only(match.group(0))
        if not url:
            return match.group(0)
        number = self.register(url)
        return f"{self._describe(number)}{match.group(0)[len(url) :]}"

    def _header_sub(self, match: re.Match) -> str:
        prefix, url = match.group(1), match.group(2)
        number = self._by_url.get(url)
        # A page the model cited by number already has its handle; one it asked
        # for by address stays written out.  Never a new registration.
        return f"{prefix}{self._describe(number)}" if number is not None else match.group(0)

    def _expand_list(self, match: re.Match) -> str:
        """Expand a list-form citation ``[4, 6]`` to its held URLs joined by
        the separator, or leave it unchanged when none are held."""
        expanded_parts = []
        has_any_held = False
        for num in _cited_numbers(match):
            url = self._by_number.get(num)
            if url is not None:
                expanded_parts.append(url)
                has_any_held = True
            else:
                expanded_parts.append(f"[{num}]")
        if not has_any_held:
            return match.group(0)
        return _MULTI_CITATION_SEPARATOR.join(expanded_parts)

    def _expand_sub(self, match: re.Match) -> str:
        text, linked, hybrid_num, _hybrid_target, label, target_num, multi_numbers, bare = (
            match.groups()
        )
        if multi_numbers is not None:
            return self._expand_list(match)
        if hybrid_num is not None:
            # ``[12](target)`` — the whole citation expands to the URL: the
            # handle is the address, and a target glued after it (a site root
            # the model knew) must not survive beside the expansion.
            return self._by_number.get(int(hybrid_num)) or match.group(0)
        if target_num is not None:
            # ``[text](12)`` — the number written as the target; same rule.
            return self._by_number.get(int(target_num)) or match.group(0)
        url = self._by_number.get(int(linked or bare))
        if url is None:
            return match.group(0)
        return f"[{text}]({url})" if text else url
