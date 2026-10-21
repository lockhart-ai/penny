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
    assert converted == "Read [the report][1] today.\n[2]\n"
    assert table.url_for(1) == "https://a.example/r"
    assert table.url_for(2) == "https://b.example/n"


def test_convert_drops_images():
    table = HandleTable()
    converted = table.convert(
        "See ![chart](https://a.example/c.png) and [the page](https://a.example/p)"
    )
    assert converted == "See  and [the page][1]"
    assert table.url_for(1) == "https://a.example/p"


def test_convert_is_idempotent():
    table = HandleTable()
    text = "[a](https://a.example/x)\nhttps://b.example/y"
    assert table.convert(table.convert(text)) == table.convert(text)


def test_convert_browse_section_includes_the_header_url():
    table = HandleTable()
    section = "## browse: https://a.example/page\nBody [link](https://b.example/l) text"
    converted = table.convert_browse_section(section)
    assert converted.startswith("## browse: [1]\n")
    assert "[link][2]" in converted
    assert "https://" not in converted


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


def test_citation_lines():
    assert PAGE_CITATION_LINE.startswith("Links on this page are numbered, like [12].")
    assert CITATION_LINE.startswith("Links are numbered, like [12].")


def test_empty_table_is_falsy():
    table = HandleTable()
    assert not table
    table.register("https://a.example/x")
    assert table
