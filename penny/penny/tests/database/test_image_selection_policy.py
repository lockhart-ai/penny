"""Automatic-image settings partition candidates before the existing ranking ladder."""

from itertools import product

import pytest

from penny.database.media_store import ImageSelectionPolicy
from penny.llm.embeddings import serialize_embedding


@pytest.mark.parametrize(
    "automatic,cited_page,same_site,related", tuple(product((False, True), repeat=4))
)
@pytest.mark.parametrize(
    "source_url,category",
    [
        ("https://www.example.test/article", "cited_page"),
        ("https://example.test/other", "same_site"),
        ("https://elsewhere.test/page", "related"),
        (None, "related"),
    ],
)
def test_each_candidate_keeps_its_strongest_category(
    db, automatic, cited_page, same_site, related, source_url, category
):
    policy = ImageSelectionPolicy(
        automatic=automatic, cited_page=cited_page, same_site=same_site, related=related
    )
    media_id = db.media.put(
        b"image",
        "image/png",
        source_url=source_url,
        embedding=serialize_embedding([1.0, 0.0]),
    )
    match = db.media.select_image(["https://www.example.test/article."], [1.0, 0.0], policy=policy)
    assert (match is not None) == (automatic and getattr(policy, category))
    if match:
        assert match.id == media_id


def test_enabled_lower_categories_still_match_and_generated_media_can_be_reused(db):
    cited = db.media.put(b"cited", "image/png", source_url="https://site.test/p")
    same = db.media.put(
        b"same",
        "image/webp",
        source_url="https://site.test/other",
        embedding=serialize_embedding([1.0, 0.0]),
    )
    generated = db.media.put(b"drawn", "image/png", embedding=serialize_embedding([1.0, 0.0]))
    urls = ["https://site.test/p"]
    match = db.media.select_image(urls, None)
    assert match is not None and match.id == cited
    assert db.media.select_image(urls, None, policy=ImageSelectionPolicy(cited_page=False)) is None
    match = db.media.select_image(urls, [1.0, 0.0], policy=ImageSelectionPolicy(cited_page=False))
    assert match is not None and match.id == same
    match = db.media.select_image(
        urls, [1.0, 0.0], policy=ImageSelectionPolicy(cited_page=False, same_site=False)
    )
    assert match is not None and match.id == generated
    assert (
        db.media.select_image(urls, [1.0, 0.0], policy=ImageSelectionPolicy(automatic=False))
        is None
    )
    # Default policy remains permissive for channels that do not supply settings.
    match = db.media.select_image(urls, [1.0, 0.0])
    assert match is not None and match.id == cited
