"""Unit tests for ``penny.image_utils.normalize_image`` (#2780).

Every real image used here is generated in-process with Pillow — never a
real captured page image or a real user photo (see ``docs/`` privacy rules
on eval fixtures, which apply equally here: no real image content in the
repo).
"""

from __future__ import annotations

import base64
import io

from PIL import Image

from penny.constants import PennyConstants
from penny.image_utils import normalize_data_uri, normalize_image

_MIME_JPEG = "image/jpeg"
_MIME_PNG = "image/png"
_MIME_GIF = "image/gif"


def _solid_jpeg(size: tuple[int, int], color=(10, 200, 80)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _solid_png(size: tuple[int, int], color=(10, 200, 80)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _random_noise_png(size: tuple[int, int]) -> bytes:
    """A PNG that compresses poorly — reliably over the byte budget even at
    small pixel dimensions, so it exercises the over-budget-but-under-cap path."""
    import random

    rng = random.Random(0)
    w, h = size
    img = Image.new("RGB", size)
    img.putdata(
        [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(w * h)]
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _alpha_png(size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGBA", size, color=(10, 200, 80, 128))
    img.save(buf, format="PNG")
    return buf.getvalue()


def _exif_rotated_jpeg(size: tuple[int, int], orientation: int) -> bytes:
    img = Image.new("RGB", size, color=(10, 200, 80))
    exif = img.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


def _animated_gif(size: tuple[int, int], frame_count: int, duration_ms: int = 100) -> bytes:
    frames = [Image.new("RGBA", size, color=(i * 40 % 256, 10, 200)) for i in range(frame_count)]
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return buf.getvalue()


def test_small_opaque_image_passes_through_byte_identical():
    data = _solid_jpeg((50, 50))
    result, mime = normalize_image(data, _MIME_JPEG)
    assert (result, mime) == (data, _MIME_JPEG)


def test_undecodable_bytes_pass_through_unchanged():
    garbage = b"not an image, just bytes" * 10
    result, mime = normalize_image(garbage, "image/jpeg")
    assert (result, mime) == (garbage, "image/jpeg")


def test_svg_passes_through_unchanged():
    svg = b"<svg xmlns='http://www.w3.org/2000/svg'><rect width='10' height='10'/></svg>"
    result, mime = normalize_image(svg, "image/svg+xml")
    assert (result, mime) == (svg, "image/svg+xml")


def test_oversized_opaque_jpeg_is_resized_and_stays_jpeg():
    data = _solid_jpeg((2000, 1000))
    result, mime = normalize_image(data, _MIME_JPEG)
    assert mime == _MIME_JPEG
    out = Image.open(io.BytesIO(result))
    assert max(out.size) == PennyConstants.IMAGE_LONG_EDGE_CAP
    assert len(result) <= len(data)


def test_oversized_alpha_png_is_resized_and_keeps_alpha():
    data = _alpha_png((2000, 1000))
    result, mime = normalize_image(data, _MIME_PNG)
    assert mime == _MIME_PNG
    out = Image.open(io.BytesIO(result))
    assert max(out.size) == PennyConstants.IMAGE_LONG_EDGE_CAP
    assert out.mode in ("RGBA", "LA") or "transparency" in out.info


def test_within_cap_over_budget_static_is_reencoded_smaller_at_same_size():
    data = _random_noise_png((512, 512))
    assert len(data) > PennyConstants.IMAGE_PASSTHROUGH_MAX_BYTES
    result, mime = normalize_image(data, _MIME_PNG)
    out = Image.open(io.BytesIO(result))
    assert out.size == (512, 512)  # no resize — only the byte budget was exceeded
    assert mime == _MIME_JPEG  # opaque -> target format
    assert len(result) < len(data)


def test_within_cap_and_budget_passes_through_even_if_dims_identical_after_check():
    # A small, already-optimal image under the byte budget never gets re-encoded.
    data = _solid_png((100, 100))
    assert len(data) <= PennyConstants.IMAGE_PASSTHROUGH_MAX_BYTES
    result, mime = normalize_image(data, _MIME_PNG)
    assert (result, mime) == (data, _MIME_PNG)


def test_oversized_result_never_exceeds_original_falls_back_to_resized_original_format():
    """A tiny solid-colour PNG, oversized only in pixels: the target-format
    (JPEG) re-encode of such a simple image is bigger than the tiny original
    PNG, so normalize_image must fall back to the resized image saved in the
    ORIGINAL format rather than returning the larger JPEG — and must never
    return something bigger than the input it started with."""
    data = _solid_png((2000, 10))
    result, mime = normalize_image(data, _MIME_PNG)
    out = Image.open(io.BytesIO(result))
    assert max(out.size) == PennyConstants.IMAGE_LONG_EDGE_CAP
    assert mime == _MIME_PNG  # fell back to the original format, not JPEG
    assert len(result) <= len(data)


def test_over_budget_reencode_that_does_not_help_keeps_the_original(monkeypatch):
    """A tiny image already under the pixel cap: lowering the byte budget
    forces a re-encode attempt even though the image is minuscule. A JPEG
    re-encode of a 50x50 solid colour costs more in container overhead than
    the image itself, so the result must be the untouched original."""
    data = _solid_png((50, 50))
    monkeypatch.setattr(PennyConstants, "IMAGE_PASSTHROUGH_MAX_BYTES", 100)
    result, mime = normalize_image(data, _MIME_PNG)
    assert (result, mime) == (data, _MIME_PNG)


def test_animated_gif_under_cap_passes_through_untouched():
    data = _animated_gif((40, 40), frame_count=3)
    result, mime = normalize_image(data, _MIME_GIF)
    assert (result, mime) == (data, _MIME_GIF)


def test_oversized_animated_gif_keeps_every_frame_and_duration():
    frame_count = 3
    data = _animated_gif((2000, 500), frame_count=frame_count, duration_ms=120)
    result, mime = normalize_image(data, _MIME_GIF)
    assert mime == _MIME_GIF
    out = Image.open(io.BytesIO(result))
    assert getattr(out, "n_frames", 1) == frame_count  # never flattened
    assert max(out.size) <= PennyConstants.IMAGE_LONG_EDGE_CAP
    out.seek(1)
    assert out.info.get("duration") == 120


def test_exif_orientation_is_applied_before_resizing():
    """Orientation 6 means the stored pixels are landscape but must display
    portrait. If exif_transpose ran before the resize, the output comes out
    portrait (height > width); if it ran after (or not at all), the output
    would wrongly stay landscape."""
    data = _exif_rotated_jpeg((2000, 1000), orientation=6)
    result, mime = normalize_image(data, _MIME_JPEG)
    out = Image.open(io.BytesIO(result))
    assert out.height > out.width
    assert max(out.size) == PennyConstants.IMAGE_LONG_EDGE_CAP


def test_normalize_data_uri_round_trips_a_normalized_image():
    data = _solid_jpeg((2000, 1000))
    uri = f"data:{_MIME_JPEG};base64,{base64.b64encode(data).decode()}"
    result = normalize_data_uri(uri)
    header, encoded = result.split(";base64,", 1)
    assert header == f"data:{_MIME_JPEG}"
    out = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert max(out.size) == PennyConstants.IMAGE_LONG_EDGE_CAP


def test_normalize_data_uri_passes_through_a_non_data_uri():
    assert normalize_data_uri("https://example.test/image.png") == "https://example.test/image.png"
