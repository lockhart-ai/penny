"""``clean_browser_content`` (penny/tools/content_cleaning.py): a pure text
transform that strips structural cruft — Kagi search-result furniture and
cross-site boilerplate — from browser-extracted markdown before it reaches
the model. No external state; every case is a small inline markdown input
checked against its whole cleaned output.
"""

from __future__ import annotations

from penny.tools.content_cleaning import clean_browser_content


def test_leaves_ordinary_content_untouched():
    """A page with no structural cruft comes back byte-identical."""
    text = "Title: A plain article\nThis is the first paragraph.\nThis is the second."
    assert clean_browser_content(text) == text


def test_removes_kagi_proxy_images_and_domain_attribution():
    """Kagi's thumbnail proxy images and the bare-domain attribution line
    underneath them are dropped; ordinary content around them survives."""
    text = (
        "Title: Some article\n"
        "![Favicon](https://example.com/favicon.ico)\n"
        "[![](https://p.kagi.com/proxy/abc123)](https://example.com/photo.jpg)\n"
        "[www.example.com](https://example.com/photo.jpg)\n"
        "Real content line."
    )
    assert clean_browser_content(text) == "Title: Some article\nReal content line."


def test_removes_kagi_metadata_labels_and_openverse_boilerplate():
    """Openverse lightbox labels/captions are cruft; the content around them
    is kept."""
    text = (
        "Before.\n"
        "Size\n"
        "Type\n"
        "Image Made with [Openverse] under CC0.\n"
        "Upload time and license details here.\n"
        "Loading source...\n"
        "Loading license...\n"
        "Report to [OpenVerse] for a licensing issue.\n"
        "[View Image](https://example.com/full.jpg)\n"
        "[Download](https://example.com/dl)\n"
        "[Visit Page](https://example.com/page)\n"
        "After."
    )
    assert clean_browser_content(text) == "Before.\nAfter."


def test_kagi_image_grid_is_skipped_until_the_next_heading():
    """A ``[Images](https://kagi.com/...)`` header opens an image/video grid;
    every line until the next ``### `` heading is dropped, including lines
    that wouldn't otherwise be matched as cruft on their own — the heading
    line itself ends the skip and is kept."""
    text = (
        "Before text.\n"
        "[Images](https://kagi.com/images?q=foo)\n"
        "[www.site1.com](https://site1.com/thumb.jpg)\n"
        "Size\n"
        "some grid caption that matches no other rule\n"
        "### Next heading\n"
        "After text."
    )
    assert clean_browser_content(text) == "Before text.\n### Next heading\nAfter text."


def test_kagi_videos_grid_is_also_skipped():
    """The grid-skip trigger covers ``[Videos]`` the same as ``[Images]``."""
    text = (
        "[Videos](https://kagi.com/videos?q=foo)\nsome caption nobody else matches\n### Done\nKept."
    )
    assert clean_browser_content(text) == "### Done\nKept."


def test_removes_empty_markdown_links():
    """A link with no display text — ``[](https://...)`` — is cruft."""
    text = "A line.\n[](https://example.com/tracker.gif)\nAnother line."
    assert clean_browser_content(text) == "A line.\nAnother line."


def test_removes_bare_image_dimension_lines():
    """A bare ``WIDTH x HEIGHT`` line under a thumbnail is cruft; a line that
    merely contains dimensions as part of a sentence is kept."""
    text = "A photo.\n1920 x 1080\nThe photo is 1920 x 1080 and lovely."
    assert clean_browser_content(text) == "A photo.\nThe photo is 1920 x 1080 and lovely."


def test_removes_tracking_pixel_images():
    """An image whose URL names a transparent tracking pixel is cruft."""
    text = (
        "Welcome.\n"
        "![pixel](https://ads.example/transparent-pixel.gif)\n"
        "![a real photo](https://example.com/photo.jpg)"
    )
    assert clean_browser_content(text) == (
        "Welcome.\n![a real photo](https://example.com/photo.jpg)"
    )


def test_removes_inline_json_blobs():
    """A line that is itself a raw JSON object (structured data dumped
    inline, e.g. pricing widgets) is cruft."""
    json_line = '{"' + ("a" * 60) + '"}'
    text = f"Price: $19.99\n{json_line}\nIn stock."
    assert clean_browser_content(text) == "Price: $19.99\nIn stock."


def test_removes_links_with_extremely_long_urls():
    """A markdown link whose URL exceeds 300 characters (auth/signin/tracking
    links), standing alone on its own line, is cruft; an ordinary short link
    is kept."""
    long_url = "https://example.com/auth?" + ("x" * 300)
    text = f"[sign in]({long_url})\n[a normal link](https://example.com/page)"
    assert clean_browser_content(text) == "[a normal link](https://example.com/page)"


def test_removes_navigation_boilerplate_lines():
    """Exact-match navigation boilerplate lines are cruft; everything else on
    the page is kept verbatim."""
    text = (
        "Skip to main content\n"
        "Skip to search results\n"
        "Skip to content\n"
        "Skip to navigation\n"
        "Hide Filters\n"
        "Reset filters\n"
        "Show Filters\n"
        "Article continues below\n"
        "View Image Gallery\n"
        "The actual article text."
    )
    assert clean_browser_content(text) == "The actual article text."


def test_removes_repeated_category_label_runs():
    """A run of two or more ``WordCategory: Word.`` labels (listing-page
    furniture) is cruft."""
    text = "A headline.\nNewsCategory: News.|TechnologyCategory: Technology.\nThe body text."
    assert clean_browser_content(text) == "A headline.\nThe body text."


def test_collapses_three_or_more_blank_lines_to_one():
    """Runs of three or more consecutive blank lines collapse to a single
    blank line; a single blank line between paragraphs is untouched."""
    text = "Paragraph one.\n\n\n\nParagraph two.\n\nParagraph three."
    assert clean_browser_content(text) == ("Paragraph one.\n\nParagraph two.\n\nParagraph three.")


def test_empty_input_returns_empty_output():
    """The degenerate case: nothing in, nothing out, no error."""
    assert clean_browser_content("") == ""
