"""Unit tests for Signal channel formatting (prepare_outgoing, _table_to_bullets)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from penny.channels.signal import SignalChannel


@pytest.fixture
def channel():
    """Minimal SignalChannel for testing formatting methods."""
    return SignalChannel(
        api_url="http://localhost:8080",
        phone_number="+15551234567",
        message_agent=MagicMock(),
        db=MagicMock(),
    )


class TestTableToBullets:
    """Tests for _table_to_bullets classmethod."""

    def test_simple_table_becomes_bullets(self):
        text = (
            "| Name | Price | Type |\n"
            "|------|-------|------|\n"
            "| Foo  | $100  | Basic |\n"
            "| Bar  | $200  | Pro   |"
        )
        assert SignalChannel._table_to_bullets(text) == (
            "**Foo**\n"
            "  \u2022 **Price**: $100\n"
            "  \u2022 **Type**: Basic\n"
            "\n"
            "**Bar**\n"
            "  \u2022 **Price**: $200\n"
            "  \u2022 **Type**: Pro\n"
        )

    def test_non_table_text_unchanged(self):
        text = "Hello world\nNo tables here"
        assert SignalChannel._table_to_bullets(text) == "Hello world\nNo tables here"

    def test_table_with_surrounding_text_preserved(self):
        text = "Results:\n\n| Name | Score |\n|------|-------|\n| Foo  | 95    |\n\nEnd."
        assert SignalChannel._table_to_bullets(text) == (
            "Results:\n\n**Foo**\n  \u2022 **Score**: 95\n\nEnd."
        )

    def test_bold_stripped_from_first_column(self):
        text = "| Name | Price |\n|------|-------|\n| **Foo** | $100 |"
        assert SignalChannel._table_to_bullets(text) == ("**Foo**\n  \u2022 **Price**: $100\n")

    def test_empty_cells_skipped(self):
        text = "| Name | A | B |\n|------|---|---|\n| Foo  | 1 |   |"
        assert SignalChannel._table_to_bullets(text) == ("**Foo**\n  \u2022 **A**: 1\n")


class TestPrepareOutgoing:
    """Tests for prepare_outgoing instance method."""

    # --- Headings become bold lines ---

    def test_converts_h1_to_bold(self, channel):
        assert channel.prepare_outgoing("# Foo") == "**Foo**"

    def test_converts_h2_to_bold(self, channel):
        assert channel.prepare_outgoing("## Foo") == "**Foo**"

    def test_converts_h3_to_bold(self, channel):
        assert channel.prepare_outgoing("### Foo bar") == "**Foo bar**"

    def test_converts_h4_to_bold(self, channel):
        assert channel.prepare_outgoing("#### Foo bar baz") == "**Foo bar baz**"

    def test_converts_h5_to_bold(self, channel):
        assert channel.prepare_outgoing("##### Foo") == "**Foo**"

    def test_converts_heading_with_emoji_to_bold(self, channel):
        assert channel.prepare_outgoing("### 1\ufe0f\u20e3 Foo Bar") == "**1\ufe0f\u20e3 Foo Bar**"

    # --- Bold / italic preservation ---

    def test_preserves_bold(self, channel):
        assert channel.prepare_outgoing("This is **bold** text") == "This is **bold** text"

    def test_preserves_italic(self, channel):
        assert channel.prepare_outgoing("This is *italic* text") == "This is *italic* text"

    def test_preserves_italic_field_label(self, channel):
        assert channel.prepare_outgoing("*Lorem:* ipsum dolor") == "*Lorem:* ipsum dolor"

    def test_preserves_bold_colon_list(self, channel):
        assert channel.prepare_outgoing("- **Foo:** $100\n- **Bar:** lorem ipsum") == (
            "• **Foo:** $100\n• **Bar:** lorem ipsum"
        )

    # --- Strikethrough / tilde handling ---

    def test_converts_double_tilde_to_single(self, channel):
        assert channel.prepare_outgoing("This is ~~struck~~ text") == "This is ~struck~ text"

    def test_escapes_stray_tilde_to_approx(self, channel):
        assert channel.prepare_outgoing("About ~50 items") == "About \u223c50 items"

    def test_preserves_tilde_in_bare_url(self, channel):
        assert (
            channel.prepare_outgoing("About ~50 items at https://example.com/~foo/page")
            == "About \u223c50 items at https://example.com/~foo/page"
        )

    def test_preserves_tilde_in_markdown_link_url(self, channel):
        assert (
            channel.prepare_outgoing("[Foo](https://example.com/~bar)")
            == "Foo (https://example.com/~bar)"
        )

    # --- Stray asterisk removal ---

    def test_removes_stray_footnote_asterisk(self, channel):
        assert channel.prepare_outgoing("Price: $100*\n\n- **Foo:** bar") == (
            "Price: $100\n\n• **Foo:** bar"
        )

    def test_stray_asterisk_does_not_break_bold(self, channel):
        assert channel.prepare_outgoing("Foo*\n- **Bar:** $100\n- **Baz:** lorem") == (
            "Foo\n• **Bar:** $100\n• **Baz:** lorem"
        )

    def test_stray_asterisk_preserves_italic(self, channel):
        assert channel.prepare_outgoing("Foo*\n\n*Lorem ipsum.*") == "Foo\n\n*Lorem ipsum.*"

    def test_multiple_stray_asterisks_removed(self, channel):
        assert channel.prepare_outgoing("Foo*\nBar*\n- **Baz:** value") == (
            "Foo\nBar\n• **Baz:** value"
        )

    def test_no_stray_asterisks_is_noop(self, channel):
        assert (
            channel.prepare_outgoing("**Bold** and *italic* text") == "**Bold** and *italic* text"
        )

    # --- Horizontal rules ---

    def test_removes_horizontal_rule(self, channel):
        assert channel.prepare_outgoing("Before\n\n---\n\nAfter") == "Before\n\nAfter"

    def test_removes_long_horizontal_rule(self, channel):
        assert channel.prepare_outgoing("Before\n\n----------\n\nAfter") == "Before\n\nAfter"

    def test_removes_multiple_horizontal_rules(self, channel):
        assert channel.prepare_outgoing("Foo\n\n---\n\nBar\n\n---\n\nBaz") == "Foo\n\nBar\n\nBaz"

    def test_table_separator_not_stripped_as_hr(self, channel):
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        assert channel.prepare_outgoing(text) == "**1**\n  \u2022 **B**: 2"

    # --- Blockquote stripping ---

    def test_strips_blockquote_marker(self, channel):
        assert channel.prepare_outgoing("> **Foo:** lorem ipsum.") == "**Foo:** lorem ipsum."

    def test_strips_multiline_blockquote(self, channel):
        assert channel.prepare_outgoing("> Foo bar\n> Baz qux") == "Foo bar\nBaz qux"

    # --- HTML br tags ---

    def test_converts_br_tag_to_newline(self, channel):
        assert channel.prepare_outgoing("Foo<br>Bar") == "Foo\nBar"

    def test_converts_br_self_closing_to_newline(self, channel):
        assert channel.prepare_outgoing("Foo<br/>Bar") == "Foo\nBar"

    # --- Markdown links ---

    def test_converts_markdown_link(self, channel):
        assert (
            channel.prepare_outgoing("[Click here](https://example.com)")
            == "Click here (https://example.com)"
        )

    def test_converts_markdown_link_with_path(self, channel):
        assert (
            channel.prepare_outgoing("[Foo](https://example.com/bar/baz-qux)")
            == "Foo (https://example.com/bar/baz-qux)"
        )

    # --- Bare URLs preserved ---

    def test_preserves_bare_url(self, channel):
        assert (
            channel.prepare_outgoing("See https://example.com/foo for details")
            == "See https://example.com/foo for details"
        )

    def test_heading_converted_bare_urls_preserved(self, channel):
        text = "## sources\nhttps://example.com\nhttps://other.com"
        assert (
            channel.prepare_outgoing(text) == "**sources**\nhttps://example.com\nhttps://other.com"
        )

    # --- Code blocks preserved ---

    def test_code_block_headings_not_stripped(self, channel):
        text = "```\n# comment\nprint('hello')\n```"
        assert channel.prepare_outgoing(text) == "```\n# comment\nprint('hello')\n```"

    def test_code_block_formatting_not_mangled(self, channel):
        text = "Before\n\n```\n# heading\n**foo** ~bar~\n---\n```\n\nAfter"
        assert channel.prepare_outgoing(text) == (
            "Before\n\n```\n# heading\n**foo** ~bar~\n---\n```\n\nAfter"
        )

    # --- Other ---

    def test_preserves_inline_code(self, channel):
        assert channel.prepare_outgoing("Use `/foo bar` to change") == "Use `/foo bar` to change"

    def test_preserves_numbered_list(self, channel):
        text = "1. Foo\n2. Bar\n3. Baz"
        assert channel.prepare_outgoing(text) == text

    def test_collapses_blank_lines(self, channel):
        assert channel.prepare_outgoing("A\n\n\n\n\nB") == "A\n\nB"

    # --- Integration ---

    def test_full_report_with_mixed_formatting(self, channel):
        text = (
            "## Top Items\n\n"
            "### 1. Foo\n"
            "- **Price:** $100\n"
            "- **Pros**\n"
            "  - Lorem ipsum\n\n"
            "---\n\n"
            "### 2. Bar*\n"
            "- **Price:** $200\n\n"
            "| Name | Price |\n"
            "|------|-------|\n"
            "| Foo  | $100  |\n"
            "| Bar  | $200  |\n\n"
            "*All prices approximate.*"
        )
        assert channel.prepare_outgoing(text) == (
            "**Top Items**\n\n"
            "**1. Foo**\n"
            "\u2022 **Price:** $100\n"
            "\u2022 **Pros**\n"
            "  \u2022 Lorem ipsum\n\n"
            "**2. Bar**\n"
            "\u2022 **Price:** $200\n\n"
            "**Foo**\n"
            "  \u2022 **Price**: $100\n\n"
            "**Bar**\n"
            "  \u2022 **Price**: $200\n\n"
            "*All prices approximate.*"
        )


class TestGemmaShapes:
    """Whole-output shapes gemma writes that the converter used to mangle.

    One literal per shape (#2659); all content invented.
    """

    def test_asterisk_rule_line_dropped(self, channel):
        assert channel.prepare_outgoing("Before\n\n***\n\nAfter") == "Before\n\nAfter"

    def test_underscore_rule_line_dropped(self, channel):
        assert channel.prepare_outgoing("Before\n\n___\n\nAfter") == "Before\n\nAfter"

    def test_spaced_asterisk_rule_line_dropped(self, channel):
        assert channel.prepare_outgoing("Before\n\n* * *\n\nAfter") == "Before\n\nAfter"

    def test_star_bullet_with_bold_link_stays_a_bullet(self, channel):
        assert (
            channel.prepare_outgoing(
                "* [**Example headline about a transit delay**](https://www.example.com/one)"
            )
            == "\u2022 **Example headline about a transit delay** (https://www.example.com/one)"
        )

    def test_star_bullet_with_wide_spacing_stays_a_bullet(self, channel):
        assert channel.prepare_outgoing("*   **Protein Levels:** Every cat is different.") == (
            "\u2022 **Protein Levels:** Every cat is different."
        )

    def test_indented_dash_bullet_becomes_bullet(self, channel):
        assert channel.prepare_outgoing("Intro:\n  - nested point\n- top point") == (
            "Intro:\n  \u2022 nested point\n\u2022 top point"
        )

    def test_trailing_stray_underscore_after_emoji_removed(self, channel):
        assert (
            channel.prepare_outgoing("Hope that helps! \ud83d\udc3e_")
            == "Hope that helps! \ud83d\udc3e"
        )

    def test_word_internal_underscores_kept(self, channel):
        text = "Read my_var_name and foo_bar_baz carefully."
        assert channel.prepare_outgoing(text) == text

    def test_italic_date_line_preserved(self, channel):
        assert channel.prepare_outgoing("*October 19, 2026*") == "*October 19, 2026*"

    def test_daily_brief_shape_end_to_end(self, channel):
        text = (
            "# \ud83d\udcf0 Daily News Brief\n"
            "*October 19, 2026*\n"
            "***\n"
            "\n"
            "### \ud83c\udde8\ud83c\udde6 World News\n"
            "* [**Example headline one about a library opening**](https://www.example.com/one)\n"
            "* [**Example headline two about a transit delay**](https://www.example.com/two)\n"
            "\n"
            "### \ud83c\udf0d More News\n"
            "* [**Example headline three about community gardens**](https://www.example.com/three)\n"
            "\n"
            "***\n"
            "\n"
            "How does that look? \u2728"
        )
        assert channel.prepare_outgoing(text) == (
            "**\ud83d\udcf0 Daily News Brief**\n"
            "*October 19, 2026*\n"
            "\n"
            "**\ud83c\udde8\ud83c\udde6 World News**\n"
            "\u2022 **Example headline one about a library opening** (https://www.example.com/one)\n"
            "\u2022 **Example headline two about a transit delay** (https://www.example.com/two)\n"
            "\n"
            "**\ud83c\udf0d More News**\n"
            "\u2022 **Example headline three about community gardens** (https://www.example.com/three)\n"
            "\n"
            "How does that look? \u2728"
        )
