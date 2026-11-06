"""Link handles — a run's numbered link table (#2717, token form #3074).

Links travel as typed tokens end to end: the model reads ``{link 101}`` instead of
a URL and writes ``{link 101}`` back, and Python expands the token when a message
is finalised, so the model never types a URL.  Measured on 7 logged page reads
(gemma, thinking off): reader input dropped 42%, and no link that wasn't on the
page and no handle that didn't exist ever appeared in a draw.  The token form and
the 101+ numbering are the #3074 fix for the prod write step: a stored entry's
rendered ``[1] <full url>`` pair taught "number + URL" and a bare bracketed number
read as a footnote, so the cycle wrote a wrong source address; with entries
rendered ``{link 118}`` alone and numbering starting at 101, the same step wrote
the right page 12/12.

One :class:`HandleTable` per run — a chat turn, a collector cycle, or a job-report
delivery turn — owned by Python and passed as an explicit parameter, never ambient
state.  It is assigned on first sight: the first time a URL is rendered to the
model it gets the next number (from :data:`PennyConstants.HANDLE_FIRST_NUMBER`),
and the same URL keeps its number for the whole run, so a handle the model read
from one tool result is the same number it cites back.  Because numbering starts
at 101, a small bracketed number in a reply is prose by construction, never a
failed citation.

Tokens are a RENDERING.  Stored state keeps real URLs: memory entries are
expanded back to their URLs before they are written (``collection_write``), and
the browse-results log stores the un-converted sections.  Conversion happens only
at each seam where the model READS — page sections, search results, history,
memory read renders, the job-report render.

The table is mutable only by registration (a URL's number never changes once
assigned); everything else is a pure read.  An empty table is honest: it means
this run rendered no handles, so no citation can dangle and no prose ``{link
12}`` is ever mistaken for one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from urllib.parse import urlparse

from penny.constants import PennyConstants

# A markdown link ``[text](url)`` — the form the page reader renders links in.
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
# An image ``![alt](url)`` — dropped before the link pass, so an image is never
# registered as a link handle.
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# A bare URL anywhere in the text — the run of non-space characters a scheme
# opens.  The stored-digest path is why this exists: a collector entry reads
# ``Headline text - https://www.example.com/news/story-9.7387777``, and a URL
# the model READS must arrive as a token or the next turn retypes it.
_BARE_URL_RE = re.compile(r"https?://[^\s]+")
# Marks that, at a bare URL's END, belong to the prose around it rather than to
# the address — a sentence's full stop, a comma, emphasis — and the closers
# decided by PAIRING: a ``)`` stays in the URL when the URL itself opened one
# (a Wikipedia link like ``/wiki/Dune_(novel)``) and leaves when it did not
# (``(see https://x.com/a)``).  The same address-reading the hallucinated-URL
# validator applies to a reply's URLs, declared here for the bare pass.
_PROSE_TAIL_MARKS = ".,;:!?*_"
_PAIR_CLOSER = {")": "(", "]": "[", "}": "{"}
# A handle citation (#3074) — two forms, both in the shared grammar
# (``PennyConstants.HANDLE_TOKEN_RE``): the bare token ``{link 12}``, and the
# labelled form the page reader itself renders for a markdown link,
# ``[text] {link 12}``.  The handle IS the address, so a labelled citation
# carries its text beside the token and expands back to a markdown link.  A
# bare bracketed number (``[12]``, ``{12}``) matches neither form — it is plain
# text, and the old bracket hybrids are gone with the numbering that starts at
# 101: no token the model is ever shown can carry a number that low.
_HANDLE_RE = re.compile(
    r"\[([^\]\[]+)\] ?\{link (\d+)\}"
    r"|\{link (\d+)\}"
)
# A browse section's own header line — the page's address (or an error's failed
# one) after the header text.  Looked up, never registered (#2735): the page
# itself takes no number from the header, so its first content link keeps the
# next one.
_BROWSE_HEADER_URL_RE = re.compile(r"^(## browse(?: error| dropped)?: )(https?://\S+)\s*$")


def _cited_number(match: re.Match) -> int:
    """The one handle number a citation form cites."""
    _label, labelled, bare = match.groups()
    return int(labelled if labelled is not None else bare)


def _registrable_domain(url: str) -> str:
    """``url``'s own host, scheme and path dropped — the shape a bare handle
    is annotated with (#2832): a domain names what a link IS without inviting
    anyone to copy a path out of it.  Falls back to the whole ``url`` on a
    scheme-less or otherwise unparseable address, which handle registration
    never actually hands this (every registered URL opens ``https?://``), but
    a fallback that degrades to the input is safer than one that raises."""
    return urlparse(url).hostname or url


def normalise_link_address(url: str) -> str:
    """A URL's address without the prose marks it ran into at its end — the one
    normalisation a link goes through before the handle table registers it, and
    the one every consumer that must match what a handle resolves to applies
    too (#3072: a recorded link card thumbnail is keyed by this)."""
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


def _plain(number: int) -> str:
    """The bare token ``{link n}`` — the render a stored entry's own source
    line takes (#3074): the token ALONE, never a token+URL pair.  The pair is
    what taught a cycle to write "number + URL" and offered ``[1]`` as an
    example; the token-only render is the arm the prod replay measured at
    12/12."""
    return f"{PennyConstants.LINK_TOKEN_PREFIX}{number}{PennyConstants.HANDLE_TOKEN_CLOSE}"


# The one plain line rendered where handles appear.  It states the token's
# COMPLETENESS, not just the form: a stored-digest sample read "[2]" as a
# reference without the address the ask demanded and browsed one site root
# eighteen times hunting for literal URLs that the write-side expansion would
# have supplied — the state never told her the token carries the address.
PAGE_CITATION_LINE = (
    "Links on this page carry tokens, like {link 101}. The token is the link "
    "itself: when your message is sent or an entry is stored, {link 101} becomes "
    "the full address. So write {link 101} wherever the link belongs, and never "
    "look for the address."
)
# The same line for seams that are not a page: history, memory read renders,
# the job-report render.
CITATION_LINE = (
    "Links carry tokens, like {link 101}. The token is the link itself: when "
    "your message is sent or an entry is stored, {link 101} becomes the full "
    "address. So write {link 101} wherever the link belongs, and never look for "
    "the address."
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
        """Whether any link has been rendered as a token this run."""
        return bool(self._by_number)

    def __len__(self) -> int:
        """How many links this run has rendered as tokens — the growth a render
        reads to decide whether its own citation line is new information."""
        return len(self._by_number)

    def register(self, url: str, *, title: str | None = None) -> int:
        """The number ``url``'s token carries — assigned on first sight, stable
        after, starting at :data:`PennyConstants.HANDLE_FIRST_NUMBER` (#3074).

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
        number = len(self._by_number) + PennyConstants.HANDLE_FIRST_NUMBER
        self._by_number[number] = url
        self._by_url[url] = number
        if title:
            self._titles[number] = title
        return number

    def plain(self, number: int) -> str:
        """The bare token ``{link n}`` — the public form of :func:`_plain`, for
        render sites that hold a number and no context of their own."""
        return _plain(number)

    def _describe(self, number: int) -> str:
        """``{link n}`` with its registrable domain, and its page title when one
        is known, beside it — the render that makes a bare token legible as a
        real address rather than an opaque reference worth resolving (#2832):
        ``{link 101} store.steampowered.com`` or, once a title is known,
        ``{link 101} store.steampowered.com — Big Pharma``. The full path never
        renders, so nothing invites copying one out by hand.  A number the
        table does not hold renders as the bare token, so a render can never
        invent a domain."""
        url = self._by_number.get(number)
        if url is None:
            return _plain(number)
        domain = _registrable_domain(url)
        title = self._titles.get(number)
        return f"{_plain(number)} {domain} — {title}" if title else f"{_plain(number)} {domain}"

    @classmethod
    def from_mapping(cls, mapping: Mapping[int, str]) -> HandleTable:
        """A table rebuilt from a recorded ``{number: url}`` mapping — the shape
        ``mapping`` snapshots and the run record stores.  The harness expands a
        run's given text through the run's own recorded table, so a token the
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
        """The URL a token stands for, or ``None`` when it dangles."""
        return self._by_number.get(number)

    def convert_entry(self, text: str) -> str:
        """``text`` with every link replaced by its bare token — a stored
        entry's own render (#3074): the citation on an entry's ``Source:`` line
        renders as the token ALONE, never a token+URL pair.  A LABELLED
        markdown link keeps its readable text beside the token — ``[text]
        {link n}`` — since the text is not the URL; only the shape where the
        link is the whole value collapses to the bare token."""
        text = _IMAGE_RE.sub("", text)
        text = _MARKDOWN_LINK_RE.sub(self._entry_link_sub, text)
        return _BARE_URL_RE.sub(self._entry_bare_sub, text)

    def _entry_link_sub(self, match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        number = self.register(url)
        if not label.strip() or label.strip() == url:
            return _plain(number)
        return f"[{self._convert_bare(label)}] {_plain(number)}"

    def _entry_bare_sub(self, match: re.Match) -> str:
        url = normalise_link_address(match.group(0))
        if not url:
            return match.group(0)
        number = self.register(url)
        return f"{_plain(number)}{match.group(0)[len(url) :]}"

    def convert(self, text: str) -> str:
        """``text`` with every link replaced by its token — the model-facing form.

        Images are dropped first (an image is not a link the model can cite);
        then every markdown ``[text](url)`` becomes ``[text] {link n}`` — or,
        when the link's text is empty or its own URL, the token described with
        its domain (#2832) — and every bare ``http(s)://`` URL that remains
        becomes ``{link n} domain``, anywhere in a line, not only standing
        alone, with the prose punctuation it ran into (a sentence's full stop,
        a closing paren the URL never opened) left around the token.  A URL
        already inside a markdown link's ``(...)`` is gone with that link
        before the bare pass runs, so nothing double-converts.  Every URL
        converted is registered, so the numbers are the run's."""
        text = _IMAGE_RE.sub("", text)
        text = _MARKDOWN_LINK_RE.sub(self._link_sub, text)
        return self._convert_bare(text)

    def convert_browse_section(self, section: str) -> str:
        """One browse section (a ``## browse …:`` header plus its body) in the
        model-facing form.  The header shows the page as the model gave it
        (#2735): the address it typed stays written out, and a page it cited by
        number comes back as the token it already holds.  The page itself never
        takes a NEW number from the header — one would sit where the page's
        first content link belongs, and the model pairs what it reads in order,
        so a headline's link lands on its neighbour.  Only links found INSIDE
        the page content get tokens from the body pass (a page fetched whole is
        registered at extract time, with its title)."""
        header, _, body = section.partition("\n")
        converted = _BROWSE_HEADER_URL_RE.sub(self._header_sub, header)
        return f"{converted}\n{self.convert(body)}" if body else converted

    def expand(self, text: str) -> str:
        """``text`` with every token the table holds replaced by its URL — the
        form a message is finalised in, so the conversation record keeps real
        URLs.  A ``[text] {link n}`` citation keeps its text as a markdown
        link; a bare ``{link n}`` becomes the bare URL.  A token not in the
        table is left exactly as written (the validator rejects it before
        sending)."""
        return _HANDLE_RE.sub(self._expand_sub, text)

    def dangling(self, text: str) -> list[int]:
        """The token numbers ``text`` cites that the table does not hold, in
        order, without repeats.  Both citation forms count — the labelled form
        and the bare token.  Empty while the table itself is empty — with no
        links rendered, a bracketed number is prose, not a failed citation."""
        if not self._by_number:
            return []
        seen: list[int] = []
        for match in _HANDLE_RE.finditer(text):
            number = _cited_number(match)
            if number not in self._by_number and number not in seen:
                seen.append(number)
        return seen

    def dangling_note(self, numbers: list[int]) -> str:
        """The honest rejection a dangling citation draws: which tokens don't
        exist and which do — the action-space fact the re-draw works from."""
        cited = ", ".join(_plain(n) for n in numbers)
        if self._by_number:
            low, high = min(self._by_number), max(self._by_number)
            return (
                f"There is no link {cited} in this turn; the links you have are "
                f"{_plain(low)}–{_plain(high)}."
            )
        return f"There is no link {cited} in this turn; there are no numbered links."

    def drop_citations(self, text: str, numbers: list[int]) -> str:
        """``text`` with every citation of a number in ``numbers`` removed — the
        repair for a token still dangling after the hallucinated-URL
        validator's one retry (#2721): the model gets no second nudge, so the
        still-bad citation is dropped instead of reaching the user.  A labelled
        form (``[text] {link n}``) keeps its text; a bare one drops outright."""
        bad = set(numbers)

        def sub(match: re.Match) -> str:
            if _cited_number(match) not in bad:
                return match.group(0)
            return match.group(1) or ""

        return _HANDLE_RE.sub(sub, text)

    # ── substitution helpers ──────────────────────────────────────────────

    def _link_sub(self, match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        number = self.register(url)
        if not label.strip() or label.strip() == url:
            # An empty label, or a link whose text is its own URL — the token
            # alone, described (#2832): the URL must not survive as readable
            # text, but the bare token on its own carries nothing to trust
            # either, so the domain beside it names what the link IS.
            return self._describe(number)
        return f"[{self._convert_bare(label)}] {_plain(number)}"

    def _convert_bare(self, text: str) -> str:
        """The bare-URL pass over ``text`` — every scheme-opening run becomes its
        token, with prose tail marks left around it."""
        return _BARE_URL_RE.sub(self._bare_sub, text)

    def _bare_sub(self, match: re.Match) -> str:
        url = normalise_link_address(match.group(0))
        if not url:
            return match.group(0)
        number = self.register(url)
        return f"{self._describe(number)}{match.group(0)[len(url) :]}"

    def _header_sub(self, match: re.Match) -> str:
        prefix, url = match.group(1), match.group(2)
        number = self._by_url.get(url)
        # A page the model cited by number already has its token; one it asked
        # for by address stays written out.  Never a new registration.
        return f"{prefix}{self._describe(number)}" if number is not None else match.group(0)

    def _expand_sub(self, match: re.Match) -> str:
        url = self._by_number.get(_cited_number(match))
        if url is None:
            return match.group(0)
        return f"[{match.group(1)}]({url})" if match.group(1) else url
