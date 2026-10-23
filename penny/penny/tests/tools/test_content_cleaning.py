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
    """An image whose URL names a transparent tracking pixel is cruft; a
    standalone image-only line is now ALSO cruft regardless of its URL (the
    guarded image-markup rule, #2759) — only an image mixed inline with
    prose on the same line survives."""
    text = (
        "Welcome.\n"
        "![pixel](https://ads.example/transparent-pixel.gif)\n"
        "![a real photo](https://example.com/photo.jpg)\n"
        "See the photo ![inline](https://example.com/inline.jpg) above."
    )
    assert clean_browser_content(text) == (
        "Welcome.\nSee the photo ![inline](https://example.com/inline.jpg) above."
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


def test_drops_repeated_lines_but_keeps_short_repeated_table_cells():
    """A line of 20+ chars that already appeared earlier (a re-rendered nav
    block) is dropped on its second occurrence; a short repeated line (a
    table cell) survives every repeat (#2759 rule 1)."""
    text = (
        "Home About Contact Pricing Support\n"
        "The first paragraph of the real article.\n"
        "Home About Contact Pricing Support\n"
        "Yes\n"
        "No\n"
        "Yes\n"
        "No\n"
        "The second paragraph of the real article."
    )
    assert clean_browser_content(text) == (
        "Home About Contact Pricing Support\n"
        "The first paragraph of the real article.\n"
        "Yes\n"
        "No\n"
        "Yes\n"
        "No\n"
        "The second paragraph of the real article."
    )


def test_drops_image_only_lines_of_any_url():
    """A line that is ONLY markdown image syntax is cruft, whatever its URL —
    an image mixed inline with prose on the same line survives (#2759 rule 2)."""
    text = (
        "A heading.\n"
        "![a scenic photo](https://example.com/scenic.jpg)\n"
        "The caption text continues here, with ![an inline image](https://example.com/x.jpg) in it."
    )
    assert clean_browser_content(text) == (
        "A heading.\n"
        "The caption text continues here, with ![an inline image](https://example.com/x.jpg) in it."
    )


def test_drops_short_boilerplate_chrome_lines():
    """SHORT lines matching known UI chrome phrasing (consent, subscribe,
    sign-in/up, related-stories, share/follow, advertisement, rights/
    copyright, skip-to) are dropped; ordinary content around them survives
    (#2759 rule 3)."""
    text = (
        "We use cookies to improve your experience.\n"
        "Subscribe to our newsletter today.\n"
        "Sign in to continue reading.\n"
        "Sign up for more like this.\n"
        "Related stories\n"
        "More stories\n"
        "You might also like\n"
        "Share this article\n"
        "Follow us on social media\n"
        "Advertisement\n"
        "All rights reserved\n"
        "Copyright 2026 Example Corp\n"
        "Skip to comments\n"
        "This is the real paragraph a reader would want to keep."
    )
    assert clean_browser_content(text) == (
        "This is the real paragraph a reader would want to keep."
    )


def test_long_prose_mentioning_chrome_words_survives_boilerplate_rule():
    """A long sentence that happens to mention a chrome phrase (e.g. 'privacy
    policy') is prose, not chrome — the 80-char guard keeps it (#2759 rule 3)."""
    text = (
        "Before.\n"
        "Our editorial team wrote at length about how the new privacy policy "
        "affects readers across the region, and why it matters for consent.\n"
        "After."
    )
    assert clean_browser_content(text) == text


def test_drops_blob_lines_over_500_chars_json_or_long_url():
    """A line over 500 chars that is a JSON-like blob, or carries a URL over
    200 chars, is a spec/price dump or an encoded tracking URL — dropped
    whole, never truncated (#2759 rule 4). Both fixtures below are shaped so
    only the new rule can catch them — the JSON blob starts with ``[``, not
    the existing ``_JSON_BLOB`` rule's ``{"``, and the URL line isn't a bare
    ``[text](url)`` link, which the existing 300-char ``_LONG_URL_LINK`` rule
    already covers on its own."""
    json_blob = "[" + ", ".join(str(i) for i in range(200)) + "]"
    assert len(json_blob) > 500
    long_url_line = "Tracking: https://example.com/track?" + ("a" * 500) + " end"
    assert len(long_url_line) > 500
    text = f"Intro.\n{json_blob}\n{long_url_line}\nOutro."
    assert clean_browser_content(text) == "Intro.\nOutro."


def test_long_prose_paragraph_over_500_chars_is_never_truncated():
    """A 500+ character prose paragraph — no JSON shape, no long URL — is
    left byte-for-byte as is: the blob rule never truncates prose, it only
    ever drops a matching line whole (#2759 rule 4)."""
    paragraph = (
        "The committee spent the better part of the afternoon debating the "
        "merits of the proposal, weighing the long-term costs against the "
        "short-term benefits, and ultimately decided that further study was "
        "warranted before any formal recommendation could be made to the "
        "full board, which would not reconvene until the following month "
        "at the earliest, pending the availability of a quorum of members, "
        "several of whom were traveling for the remainder of the fiscal "
        "quarter and could not be reached for comment before the deadline."
    )
    assert len(paragraph) > 500  # past the blob rule's own length guard
    text = f"Start.\n{paragraph}\nEnd."
    assert clean_browser_content(text) == text


def test_guarded_trims_combine_on_an_invented_fixture_page():
    """Whole-string before/after: an invented fixture page carrying repeated
    nav, a standalone photo, a consent banner, and a long tracking-URL link
    comes back with only the article text, verbatim (#2759)."""
    long_tracking_url = "https://shop.example.com/redirect?ref=" + ("z" * 470)
    before = (
        "Riverside Outfitters | Home | Shop | Support\n"
        "Riverside Outfitters | Home | Shop | Support\n"
        "![storefront photo](https://example.com/storefront.jpg)\n"
        "We use cookies to personalize your experience.\n"
        "Amara Collins reviewed the new Fjallraven Kaipak 28 backpack "
        "during a three-day trip through Banff National Park, and found "
        "the hip belt noticeably more comfortable than last year's model.\n"
        f"[track]({long_tracking_url})\n"
        "Riverside Outfitters | Home | Shop | Support\n"
        "The backpack retails for $149 and ships from Calgary within two "
        "business days of ordering."
    )
    after = clean_browser_content(before)
    assert after == (
        "Riverside Outfitters | Home | Shop | Support\n"
        "Amara Collins reviewed the new Fjallraven Kaipak 28 backpack "
        "during a three-day trip through Banff National Park, and found "
        "the hip belt noticeably more comfortable than last year's model.\n"
        "The backpack retails for $149 and ships from Calgary within two "
        "business days of ordering."
    )


def test_collapses_three_or_more_blank_lines_to_one():
    """Runs of three or more consecutive blank lines collapse to a single
    blank line; a single blank line between paragraphs is untouched."""
    text = "Paragraph one.\n\n\n\nParagraph two.\n\nParagraph three."
    assert clean_browser_content(text) == ("Paragraph one.\n\nParagraph two.\n\nParagraph three.")


def test_empty_input_returns_empty_output():
    """The degenerate case: nothing in, nothing out, no error."""
    assert clean_browser_content("") == ""
