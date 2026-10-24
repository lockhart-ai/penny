"""Integration test for ``MediaStore.put`` normalizing every stored image (#2780).

The unit-level rules themselves live in ``tests/test_image_utils.py``; this
only checks that ``put`` actually routes through ``normalize_image`` before
the row is written, so a browse capture or a generated image lands already
within bounds — the same row egress later reads back unchanged.
"""

from __future__ import annotations

import io

from PIL import Image


def _oversized_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2000, 1000), color=(10, 200, 80)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def test_put_normalizes_an_oversized_image_before_storing(db):
    data = _oversized_jpeg()
    media_id = db.media.put(data, "image/jpeg")

    stored = db.media.get(media_id)
    assert stored is not None
    assert len(stored.data) < len(data)
    out = Image.open(io.BytesIO(stored.data))
    assert max(out.size) == 1024


def test_put_leaves_an_undecodable_blob_unchanged(db):
    """An already-existing caller shape (not a real image) must still work —
    normalize_image's undecodable-input passthrough keeps this backward
    compatible."""
    media_id = db.media.put(b"not-an-image", "image/png")
    stored = db.media.get(media_id)
    assert stored is not None
    assert stored.data == b"not-an-image"
    assert stored.mime_type == "image/png"
