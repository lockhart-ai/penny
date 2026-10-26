"""HandleTable — the run's numbered link table (#2717).

Pure-utility unit tests: the table is registration, conversion, expansion and
the dangling check — no database, no model.  The integration behaviour (a
browse result rendered as handles, a reply expanded at send) is covered by the
browse and agentic-loop tests that drive those seams.
"""

from penny.tools.link_handles import CITATION_LINE, PAGE_CITATION_LINE, HandleTable


def test_register_assigns_on_first_sight_and_is_stable():
    table = HandleTable()
    assert table.register("https://a.example/x") == 1
    assert table.register("https://b.example/y") == 2
    assert table.register("https://a.example/x") == 1
    assert len(table) == 2
    assert table.url_for(1) == "https://a.example/x"


def test_convert_markdown_links_and_standalone_urls():
    table = HandleTable()
    text = "Read [the report](https://a.example/r) today.\nhttps://b.example/n\n"
    converted = table.convert(text)
    assert converted == "Read [the report][1] today.\n[2] b.example\n"
    assert table.url_for(1) == "https://a.example/r"
    assert table.url_for(2) == "https://b.example/n"


def test_convert_drops_images():
    table = HandleTable()
    converted = table.convert(
        "See ![chart](https://a.example/c.png) and [the page](https://a.example/p)"
    )
    assert converted == "See  and [the page][1]"
    assert table.url_for(1) == "https://a.example/p"


def test_convert_bare_url_inside_a_line():
    """A bare URL ANYWHERE in a line converts — the stored-digest form
    ``Headline - https://…`` is exactly the production path that broke."""
    table = HandleTable()
    converted = table.convert("Headline - https://x.com/a.")
    assert converted == "Headline - [1] x.com."
    assert table.url_for(1) == "https://x.com/a"


def test_convert_bare_url_inside_parens():
    """A bare URL inside prose parens converts, and the paren the URL never
    opened stays around the handle."""
    table = HandleTable()
    assert table.convert("(see https://x.com/a)") == "(see [1] x.com)"


def test_convert_keeps_parens_the_url_opened():
    """A ``)`` that closes a ``(`` the URL itself opened is part of the address —
    a Wikipedia link keeps its brackets."""
    table = HandleTable()
    url = "https://en.wikipedia.org/wiki/Dune_(novel)"
    converted = table.convert(f"The novel is at {url}, read it.")
    assert converted == "The novel is at [1] en.wikipedia.org, read it."
    assert table.url_for(1) == url


def test_convert_self_link_collapses_to_the_handle():
    """A link whose text is its own URL carries no readable address at all —
    the handle alone, described (#2832): a domain beside a bare number, so
    a reader has grounds to trust it is real."""
    table = HandleTable()
    assert table.convert("[https://x.com/a](https://x.com/a)") == "[1] x.com"


def test_convert_markdown_link_url_is_not_double_converted():
    """A URL already inside a markdown link's ``(...)`` is gone with that link
    before the bare pass runs — one handle, no double registration."""
    table = HandleTable()
    converted = table.convert("see [the page](https://x.com/a) ok")
    assert converted == "see [the page][1] ok"
    assert len(table) == 1


def test_convert_is_idempotent():
    table = HandleTable()
    text = "[a](https://a.example/x)\nhttps://b.example/y"
    assert table.convert(table.convert(text)) == table.convert(text)


def test_convert_browse_section_leaves_the_page_address_written_out():
    """The page itself takes no number (#2735): its own address stays written out
    in its section header, and the FIRST content link is what gets [1] — a new
    number for the page would sit where the first headline's link belongs and
    pair a digest's headlines with their neighbours."""
    table = HandleTable()
    section = (
        "## browse: https://a.example/page\n"
        "Headline one [link](https://b.example/l)\n"
        "Headline two [other](https://b.example/m)"
    )
    converted = table.convert_browse_section(section)
    assert converted.startswith("## browse: https://a.example/page\n")
    assert "[link][1]" in converted
    assert "[other][2]" in converted
    assert len(table) == 2


def test_convert_browse_section_cites_a_registered_page_by_its_handle():
    """A page the model asked for by number comes back by that number — the
    handle it cited is the page as it gave it (#2735); the address never re-enters
    the model-facing text and no second number is assigned.  Its domain
    renders beside it (#2832); a title the page was never registered with
    (no caller supplied one here) just isn't there — this handle's number is
    the only thing carried over from the pick."""
    table = HandleTable()
    table.register("https://a.example/page")
    section = "## browse: https://a.example/page\nBody [link](https://b.example/l) text"
    converted = table.convert_browse_section(section)
    assert converted.startswith("## browse: [1] a.example\n")
    assert "[link][2]" in converted
    assert "https://" not in converted
    assert len(table) == 2


def test_expand_bare_and_text_forms():
    table = HandleTable()
    table.register("https://a.example/x")
    table.register("https://b.example/y")
    assert table.expand("see [1] and [the b page][2]") == (
        "see https://a.example/x and [the b page](https://b.example/y)"
    )


def test_expand_leaves_dangling_handles_as_written():
    table = HandleTable()
    table.register("https://a.example/x")
    assert table.expand("see [1] and [9]") == "see https://a.example/x and [9]"


def test_expand_consumes_a_target_glued_after_the_handle():
    """``[12](https://site.example)`` — the hybrid the model composes when it
    treats the handle as a markdown link: the whole citation expands to the
    URL, and the site root it knew must not survive beside the expansion (the
    stored-digest sample that stored ``story-url(site-root)``)."""
    table = HandleTable()
    table.register("https://news.example/story-1")
    table.register("https://news.example/story-2")
    assert table.expand("see [2](https://site.example)") == "see https://news.example/story-2"
    assert table.dangling("cite [3](https://site.example)") == [3]


def test_expand_consumes_a_handle_written_as_the_target():
    """``[text](12)`` — the number written as the target — is the same citation
    in the other order: the whole form expands to the URL."""
    table = HandleTable()
    table.register("https://news.example/story-1")
    assert table.expand("the [harbour story](1) leads") == (
        "the https://news.example/story-1 leads"
    )
    assert table.dangling("the [harbour story](7) leads") == [7]


def test_expand_leaves_a_genuine_markdown_link_alone():
    """A markdown link whose bracket is not a number and whose target is not a
    number is a real link — expansion never touches it."""
    table = HandleTable()
    table.register("https://news.example/story-1")
    text = "read [the report](https://real.example/report) today"
    assert table.expand(text) == text


def test_from_mapping_roundtrips():
    table = HandleTable()
    table.register("https://a.example/x")
    table.register("https://b.example/y")
    rebuilt = HandleTable.from_mapping(table.mapping())
    assert rebuilt.expand("see [1] and [the y page][2]") == (
        "see https://a.example/x and [the y page](https://b.example/y)"
    )
    assert rebuilt.dangling("[3]") == [3]


def test_dangling_reports_only_unknown_numbers_and_only_when_links_exist():
    empty = HandleTable()
    assert empty.dangling("[27] people said") == []
    table = HandleTable()
    table.register("https://a.example/x")
    table.register("https://b.example/y")
    assert table.dangling("cite [1] then [2] then [27] again [27]") == [27]
    assert table.dangling("no citations here") == []


def test_dangling_note_names_the_range():
    table = HandleTable()
    for i in range(1, 25):
        table.register(f"https://a.example/{i}")
    note = table.dangling_note([27, 31])
    assert note == "There is no link [27], [31] in this turn; the links you have are [1]–[24]."


def test_dangling_note_on_an_empty_table():
    note = HandleTable().dangling_note([3])
    assert note == "There is no link [3] in this turn; there are no numbered links."


def test_drop_citations_labelled_forms_keep_their_text():
    table = HandleTable()
    assert table.drop_citations("see [Travel Guide][27] here", [27]) == "see Travel Guide here"
    assert table.drop_citations("see [Travel Guide](27) here", [27]) == "see Travel Guide here"


def test_drop_citations_unlabelled_forms_drop_outright():
    table = HandleTable()
    assert table.drop_citations("see [27] here", [27]) == "see  here"
    assert table.drop_citations("see [27](some-target) here", [27]) == "see  here"


def test_drop_citations_list_form_keeps_numbers_not_dropped():
    table = HandleTable()
    assert table.drop_citations("see [4, 6, 27] here", [27]) == "see [4, 6] here"
    assert table.drop_citations("see [4, 27] here", [27]) == "see [4] here"
    assert table.drop_citations("see [27] here", [27]) == "see  here"


def test_drop_citations_leaves_numbers_not_named_alone():
    table = HandleTable()
    assert table.drop_citations("see [1] and [27] here", [27]) == "see [1] and  here"


def test_citation_lines():
    assert PAGE_CITATION_LINE.startswith("Links on this page are numbered, like [12].")
    assert CITATION_LINE.startswith("Links are numbered, like [12].")


def test_empty_table_is_falsy():
    table = HandleTable()
    assert not table
    table.register("https://a.example/x")
    assert table


def test_expand_multi_cite_two_numbers_both_held():
    """A list citation [1, 2] with both numbers held expands each to its URL
    joined by the separator."""
    table = HandleTable()
    table.register("https://example.com/story-vancouver")
    table.register("https://example.com/story-transit")
    assert table.expand("The ferry strike in Vancouver [1, 2].") == (
        "The ferry strike in Vancouver https://example.com/story-vancouver, https://example.com/story-transit."
    )


def test_expand_multi_cite_three_numbers_all_held():
    """A three-number list citation with all numbers held expands each to its
    URL, joined by the separator."""
    table = HandleTable()
    table.register("https://example.com/article-background")
    table.register("https://example.com/article-context")
    table.register("https://example.com/article-analysis")
    assert table.expand("Read [1, 2, 3] for background.") == (
        "Read https://example.com/article-background, "
        "https://example.com/article-context, "
        "https://example.com/article-analysis for background."
    )


def test_expand_multi_cite_mixed_held_and_unheld():
    """A list citation with one held number and one unheld expands the held one
    and keeps the unheld as [n]."""
    table = HandleTable()
    table.register("https://example.com/story-details")
    assert (
        table.expand("See [1, 6] for details.")
        == "See https://example.com/story-details, [6] for details."
    )
    assert table.dangling("[1, 6]") == [6]


def test_expand_multi_cite_none_held_unchanged():
    """A list citation with no held numbers is left exactly as written."""
    table = HandleTable()
    table.register("https://example.com/story-transit")
    assert table.expand("The transit report [4, 6] shows trends.") == (
        "The transit report [4, 6] shows trends."
    )
    assert table.dangling("[4, 6]") == [4, 6]


def test_dangling_multi_cite_all_unheld_in_order():
    """dangling() reports all unheld numbers in a list, in order, without
    repeats."""
    table = HandleTable()
    table.register("https://example.com/story-transit")
    assert table.dangling("[4, 6, 8]") == [4, 6, 8]


def test_dangling_multi_cite_with_repeats_deduplicated():
    """dangling() on a list with repeats reports each unheld number only once,
    in first-appearance order."""
    table = HandleTable()
    table.register("https://example.com/story-transit")
    assert table.dangling("[4, 6, 4]") == [4, 6]


# ── Domain + title beside a bare handle (#2832) ───────────────────────────────
#
# A bare ``[n]`` with nothing else beside it read as an opaque reference worth
# resolving rather than a citable address — production's turn searched five
# more times and drafted raw URLs after a collection entry's ``Source: [209]``
# was right in front of it.  Every bare render now carries the handle's own
# registrable domain, and its page title once one is known.


def test_register_backfills_a_title_once_known_but_never_overwrites_it():
    """A URL is often registered before its page is read (the pick-time
    citation) and again once it is (the title now known) — the SECOND call
    backfills the title onto the SAME number, and a third call naming a
    different title changes nothing: the first title known wins."""
    table = HandleTable()
    assert table.register("https://a.example/x") == 1
    assert table.register("https://a.example/x", title="Page One") == 1
    assert table.register("https://a.example/x", title="A Different Title") == 1
    assert table.convert("https://a.example/x") == "[1] a.example — Page One"


def test_convert_bare_url_with_no_known_title_shows_the_domain_alone():
    """A handle this run never fetched — a collection entry's stored
    ``Source:`` link, say — carries no title to show, and the render is
    honest about that: the domain alone, never a guessed name."""
    table = HandleTable()
    converted = table.convert("Source: https://store.example/app/1/Big-Game/")
    assert converted == "Source: [1] store.example"


def test_convert_bare_url_with_a_known_title_shows_both():
    """Once a title is known (a page actually read this run), the bare handle
    carries it beside the domain — the shape the issue's own example gives:
    ``[1] store.steampowered.com — <title>``."""
    table = HandleTable()
    table.register("https://store.example/app/1/Big-Game/", title="Big Game")
    converted = table.convert("Source: https://store.example/app/1/Big-Game/")
    assert converted == "Source: [1] store.example — Big Game"


def test_header_sub_carries_the_domain_and_title_too():
    """The browse page header names the SAME handle the same way — one
    mechanism for every bare render, not a special case per seam."""
    table = HandleTable()
    table.register("https://a.example/page", title="A Page")
    section = "## browse: https://a.example/page\nBody text"
    converted = table.convert_browse_section(section)
    assert converted.startswith("## browse: [1] a.example — A Page\n")


def test_annotate_describes_an_answer_that_is_only_a_handle():
    """A page reader's answer that IS, in its entirety, the handle it was
    asked for — "the official store link: [1]" — carries its FULL address
    beside the number (#2832 arm C): the link IS the whole answer here, so a
    domain alone still reads as partial."""
    table = HandleTable()
    table.register("https://store.example/app/1/Big-Game/", title="Big Game")
    assert table.annotate("[1]") == "[1] https://store.example/app/1/Big-Game/"
    assert table.annotate(" [1] ") == "[1] https://store.example/app/1/Big-Game/"


def test_annotate_leaves_an_answer_that_carries_more_than_the_citation():
    """An answer that is a SENTENCE naming the handle, not just the bare
    number, is already legible — annotate() only describes a citation that
    stands completely alone, so it never double-renders a label the model
    already wrote."""
    table = HandleTable()
    table.register("https://store.example/app/1/Big-Game/", title="Big Game")
    text = "the store page is [1]"
    assert table.annotate(text) == text


def test_annotate_leaves_plain_text_and_dangling_handles_alone():
    """Text that is no citation at all, and a bare number the table never
    registered, are both left exactly as given — annotate() never invents a
    description for a handle it doesn't hold."""
    table = HandleTable()
    assert table.annotate("no link here") == "no link here"
    assert table.annotate("[7]") == "[7]"


# ── The full-address render for a stored entry (#2832 arm C) ─────────────────
#
# A stored entry's own ``Source:`` line is where the citation IS the fact an
# entry exists for, so convert_entry() shows the full address beside the
# number — unlike convert(), which shows the domain alone everywhere else.


def test_convert_entry_shows_the_full_address_on_a_bare_url():
    table = HandleTable()
    url = "https://store.example/app/1/Big-Game/"
    converted = table.convert_entry(f"Title: Big Game\nSource: {url}")
    assert converted == f"Title: Big Game\nSource: [1] {url}"


def test_convert_entry_shows_the_full_address_on_a_self_referential_link():
    """A markdown link whose text is its own URL — the same collapsed shape
    ``convert()`` reduces to a bare handle — gets the full address too."""
    table = HandleTable()
    url = "https://store.example/app/1/Big-Game/"
    assert table.convert_entry(f"[{url}]({url})") == f"[1] {url}"


def test_convert_entry_leaves_a_labelled_markdown_link_unchanged():
    """A link with real anchor text already reads fine — convert_entry()
    only touches the shape where the link is the whole value, same as
    convert() does for everything else."""
    table = HandleTable()
    url = "https://store.example/app/1/Big-Game/"
    converted = table.convert_entry(f"loves this game — [the store page]({url})")
    assert converted == "loves this game — [the store page][1]"
    assert url not in converted
