"""HandleTable — the run's numbered link table (#2717, token form #3074).

Pure-utility unit tests: the table is registration, conversion, expansion and
the dangling check — no database, no model.  The integration behaviour (a
browse result rendered as tokens, a reply expanded at send) is covered by the
browse and agentic-loop tests that drive those seams.

The token grammar (#3074): every handle renders, parses, expands and validates
as ``{link n}`` — a bare bracketed number (``[1]``, ``{1}``) is plain text, and
numbering starts at 101 per run, so a small bracketed number can never read as
a failed citation.  A stored entry's own render is the token ALONE (never the
old ``token + URL`` pair the prod replay measured at 1/12).
"""

from penny.constants import PennyConstants
from penny.tools.link_handles import CITATION_LINE, PAGE_CITATION_LINE, HandleTable

_FIRST = PennyConstants.HANDLE_FIRST_NUMBER


def test_register_assigns_on_first_sight_from_101_and_is_stable():
    table = HandleTable()
    assert table.register("https://a.example/x") == _FIRST
    assert table.register("https://b.example/y") == _FIRST + 1
    assert table.register("https://a.example/x") == _FIRST
    assert len(table) == 2
    assert table.url_for(_FIRST) == "https://a.example/x"


def test_convert_markdown_links_and_standalone_urls():
    table = HandleTable()
    text = "Read [the report](https://a.example/r) today.\nhttps://b.example/n\n"
    converted = table.convert(text)
    assert converted == (
        f"Read [the report] {{link {_FIRST}}} today.\n{{link {_FIRST + 1}}} b.example\n"
    )
    assert table.url_for(_FIRST) == "https://a.example/r"
    assert table.url_for(_FIRST + 1) == "https://b.example/n"


def test_convert_drops_images():
    table = HandleTable()
    converted = table.convert(
        "See ![chart](https://a.example/c.png) and [the page](https://a.example/p)"
    )
    assert converted == f"See  and [the page] {{link {_FIRST}}}"
    assert table.url_for(_FIRST) == "https://a.example/p"


def test_convert_bare_url_inside_a_line():
    """A bare URL ANYWHERE in a line converts — the stored-digest form
    ``Headline - https://…`` is exactly the production path that broke."""
    table = HandleTable()
    converted = table.convert("Headline - https://x.com/a.")
    assert converted == f"Headline - {{link {_FIRST}}} x.com."
    assert table.url_for(_FIRST) == "https://x.com/a"


def test_convert_bare_url_inside_parens():
    """A bare URL inside prose parens converts, and the paren the URL never
    opened stays around the token."""
    table = HandleTable()
    assert table.convert("(see https://x.com/a)") == f"(see {{link {_FIRST}}} x.com)"


def test_convert_keeps_parens_the_url_opened():
    """A ``)`` that closes a ``(`` the URL itself opened is part of the address —
    a Wikipedia link keeps its brackets."""
    table = HandleTable()
    url = "https://en.wikipedia.org/wiki/Dune_(novel)"
    converted = table.convert(f"The novel is at {url}, read it.")
    assert converted == f"The novel is at {{link {_FIRST}}} en.wikipedia.org, read it."
    assert table.url_for(_FIRST) == url


def test_convert_self_link_collapses_to_the_described_token():
    """A link whose text is its own URL carries no readable address at all —
    the token, described (#2832): a domain beside the token, so a reader has
    grounds to trust it is real."""
    table = HandleTable()
    assert table.convert("[https://x.com/a](https://x.com/a)") == f"{{link {_FIRST}}} x.com"


def test_convert_markdown_link_url_is_not_double_converted():
    """A URL already inside a markdown link's ``(...)`` is gone with that link
    before the bare pass runs — one token, no double registration."""
    table = HandleTable()
    converted = table.convert("see [the page](https://x.com/a) ok")
    assert converted == f"see [the page] {{link {_FIRST}}} ok"
    assert len(table) == 1


def test_convert_is_idempotent():
    table = HandleTable()
    text = "[a](https://a.example/x)\nhttps://b.example/y"
    assert table.convert(table.convert(text)) == table.convert(text)


def test_convert_browse_section_leaves_the_page_address_written_out():
    """The page itself takes no number from its header (#2735): its own address
    stays written out there, and the FIRST content link is what gets the next
    token — a new number for the page would sit where the first headline's link
    belongs and pair a digest's headlines with their neighbours."""
    table = HandleTable()
    section = (
        "## browse: https://a.example/page\n"
        "Headline one [link](https://b.example/l)\n"
        "Headline two [other](https://b.example/m)"
    )
    converted = table.convert_browse_section(section)
    assert converted.startswith("## browse: https://a.example/page\n")
    assert f"[link] {{link {_FIRST}}}" in converted
    assert f"[other] {{link {_FIRST + 1}}}" in converted
    assert len(table) == 2


def test_convert_browse_section_cites_a_registered_page_by_its_token():
    """A page the model asked for by number comes back by that number — the
    token it cited is the page as it gave it (#2735); the address never re-enters
    the model-facing text and no second number is assigned.  Its domain
    renders beside it (#2832); a title the page was never registered with
    (no caller supplied one here) just isn't there — this handle's number is
    the only thing carried over from the pick."""
    table = HandleTable()
    table.register("https://a.example/page")
    section = "## browse: https://a.example/page\nBody [link](https://b.example/l) text"
    converted = table.convert_browse_section(section)
    assert converted.startswith(f"## browse: {{link {_FIRST}}} a.example\n")
    assert f"[link] {{link {_FIRST + 1}}}" in converted
    assert "https://" not in converted
    assert len(table) == 2


def test_expand_bare_and_labelled_forms():
    table = HandleTable()
    table.register("https://a.example/x")
    table.register("https://b.example/y")
    assert table.expand(f"see {{link {_FIRST}}} and [the b page] {{link {_FIRST + 1}}}") == (
        "see https://a.example/x and [the b page](https://b.example/y)"
    )


def test_expand_leaves_dangling_tokens_as_written():
    table = HandleTable()
    table.register("https://a.example/x")
    assert table.expand(f"see {{link {_FIRST}}} and {{link 999}}") == (
        "see https://a.example/x and {link 999}"
    )


def test_expand_leaves_a_genuine_markdown_link_alone():
    """A markdown link whose target is a URL is a real link — expansion never
    touches it, whatever its text."""
    table = HandleTable()
    table.register("https://news.example/story-1")
    text = "read [the report](https://real.example/report) today"
    assert table.expand(text) == text


def test_expand_never_touches_a_bare_bracketed_number():
    """A bare ``[1]`` or ``{1}`` is plain text, never a link (#3074): with
    numbering from 101, no token the model is ever shown can carry a number
    that low, so prose that happens to bracket a number stays prose."""
    table = HandleTable()
    table.register("https://a.example/x")
    text = "[12] people said, and {3} of them meant it"
    assert table.expand(text) == text
    assert table.dangling(text) == []


def test_from_mapping_roundtrips():
    table = HandleTable()
    table.register("https://a.example/x")
    table.register("https://b.example/y")
    rebuilt = HandleTable.from_mapping(table.mapping())
    assert rebuilt.expand(f"see {{link {_FIRST}}} and [the y page] {{link {_FIRST + 1}}}") == (
        "see https://a.example/x and [the y page](https://b.example/y)"
    )
    assert rebuilt.dangling("{link 999}") == [999]


def test_dangling_reports_only_unknown_numbers_and_only_when_links_exist():
    empty = HandleTable()
    assert empty.dangling("{link 127} people said") == []
    table = HandleTable()
    table.register("https://a.example/x")
    table.register("https://b.example/y")
    assert table.dangling(
        f"cite {{link {_FIRST}}} then {{link {_FIRST + 1}}} then {{link 127}} again {{link 127}}"
    ) == [127]
    assert table.dangling("no citations here") == []


def test_dangling_reads_the_labelled_form_too():
    table = HandleTable()
    table.register("https://a.example/x")
    assert table.dangling("see [the page] {link 127}") == [127]


def test_dangling_note_names_the_range():
    table = HandleTable()
    for i in range(_FIRST, _FIRST + 24):
        table.register(f"https://a.example/{i}")
    note = table.dangling_note([127, 131])
    assert note == (
        f"There is no link {{link 127}}, {{link 131}} in this turn; the links you "
        f"have are {{link {_FIRST}}}–{{link {_FIRST + 23}}}."
    )


def test_dangling_note_on_an_empty_table():
    note = HandleTable().dangling_note([103])
    assert note == "There is no link {link 103} in this turn; there are no numbered links."


def test_drop_citations_labelled_form_keeps_its_text():
    table = HandleTable()
    assert table.drop_citations("see [Travel Guide] {link 127} here", [127]) == (
        "see Travel Guide here"
    )


def test_drop_citations_bare_form_drops_outright():
    table = HandleTable()
    assert table.drop_citations("see {link 127} here", [127]) == "see  here"


def test_drop_citations_leaves_numbers_not_named_alone():
    table = HandleTable()
    assert (
        table.drop_citations(f"see {{link {_FIRST}}} and {{link 127}} here", [127])
        == f"see {{link {_FIRST}}} and  here"
    )


def test_citation_lines():
    assert PAGE_CITATION_LINE.startswith("Links on this page carry tokens, like {link 101}.")
    assert CITATION_LINE.startswith("Links carry tokens, like {link 101}.")


def test_empty_table_is_falsy():
    table = HandleTable()
    assert not table
    table.register("https://a.example/x")
    assert table


# ── Domain + title beside a bare token (#2832) ────────────────────────────────
#
# A bare token with nothing else beside it would read as an opaque reference
# worth resolving rather than a citable address — production's turn searched
# five more times and drafted raw URLs after a collection entry's source token
# was right in front of it.  Every page-body render carries the handle's own
# registrable domain, and its page title once one is known.  A stored ENTRY's
# render is the exception, and the exception the #3074 replay measured: the
# token ALONE, never a token+URL pair.


def test_register_backfills_a_title_once_known_but_never_overwrites_it():
    """A URL is often registered before its page is read (the pick-time
    citation) and again once it is (the title now known) — the SECOND call
    backfills the title onto the SAME number, and a third call naming a
    different title changes nothing: the first title known wins."""
    table = HandleTable()
    assert table.register("https://a.example/x") == _FIRST
    assert table.register("https://a.example/x", title="Page One") == _FIRST
    assert table.register("https://a.example/x", title="A Different Title") == _FIRST
    assert table.convert("https://a.example/x") == f"{{link {_FIRST}}} a.example — Page One"


def test_convert_bare_url_with_no_known_title_shows_the_domain_alone():
    """A handle this run never fetched — a page-body link, say — carries no
    title to show, and the render is honest about that: the domain alone,
    never a guessed name."""
    table = HandleTable()
    converted = table.convert("Source: https://store.example/app/1/Big-Game/")
    assert converted == f"Source: {{link {_FIRST}}} store.example"


def test_convert_bare_url_with_a_known_title_shows_both():
    """Once a title is known (a page actually read this run), the bare token
    carries it beside the domain."""
    table = HandleTable()
    table.register("https://store.example/app/1/Big-Game/", title="Big Game")
    converted = table.convert("Source: https://store.example/app/1/Big-Game/")
    assert converted == f"Source: {{link {_FIRST}}} store.example — Big Game"


def test_header_sub_carries_the_domain_and_title_too():
    """The browse page header names the SAME token the same way — one
    mechanism for every bare render, not a special case per seam."""
    table = HandleTable()
    table.register("https://a.example/page", title="A Page")
    section = "## browse: https://a.example/page\nBody text"
    converted = table.convert_browse_section(section)
    assert converted.startswith(f"## browse: {{link {_FIRST}}} a.example — A Page\n")


# ── A stored entry's render is the token ALONE (#3074) ────────────────────────
#
# A stored entry's own ``Source:`` line rendered ``[1] <full url>`` — the pair
# that taught a cycle "number + URL" and offered ``[1]`` as an example, and the
# render the prod replay measured at 1/12 on the write step.  The token alone
# measured 12/12, so convert_entry() renders the bare token and nothing beside
# it; the domain+title form stays where it was measured to help (page bodies).


def test_convert_entry_shows_the_token_alone_on_a_bare_url():
    table = HandleTable()
    url = "https://store.example/app/1/Big-Game/"
    converted = table.convert_entry(f"Title: Big Game\nSource: {url}")
    assert converted == f"Title: Big Game\nSource: {{link {_FIRST}}}"
    assert "store.example" not in converted


def test_convert_entry_shows_the_token_alone_on_a_self_referential_link():
    """A markdown link whose text is its own URL — the collapsed shape — gets
    the bare token too, with no address of any kind beside it."""
    table = HandleTable()
    url = "https://store.example/app/1/Big-Game/"
    assert table.convert_entry(f"[{url}]({url})") == f"{{link {_FIRST}}}"


def test_convert_entry_keeps_a_labelled_link_text_beside_the_token():
    """A link with real anchor text keeps it — the text is not the URL, so it
    stays readable — while the citation itself renders as the bare token, and
    the URL is gone."""
    table = HandleTable()
    url = "https://store.example/app/1/Big-Game/"
    converted = table.convert_entry(f"loves this game — [the store page]({url})")
    assert converted == f"loves this game — [the store page] {{link {_FIRST}}}"
    assert url not in converted


def test_convert_entry_expands_back_at_write():
    """The write-side seam stays closed: the token an entry render put in front
    of the model is the token ``collection_write`` expands back to the stored
    URL — the same number, the same table."""
    table = HandleTable()
    url = "https://store.example/app/1/Big-Game/"
    rendered = table.convert_entry(f"Source: {url}")
    assert table.expand(rendered) == f"Source: {url}"


def test_plain_is_the_public_bare_token():
    table = HandleTable()
    number = table.register("https://a.example/x")
    assert table.plain(number) == f"{{link {_FIRST}}}"
