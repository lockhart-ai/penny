"""Shared image-size normalization (#2780) — the one place that decides how
large a stored or sent image is allowed to be.

Two call sites feed every image into Penny: ``MediaStore.put`` (browse
captures, generated images) and the chat ingress boundary (an incoming
photo, before it rides the model request). Both call ``normalize_image``
so an image is capped once, consistently, regardless of which side it
entered from — "resize in both directions" from one implementation.

Rules, in order:
  - Undecodable input (corrupt bytes, SVG, a format Pillow has no codec
    for) passes through byte-for-byte — never a crash, same best-effort
    shape ``browse._decode_data_uri`` already uses for an unreadable image.
  - An animated image (today: GIF) whose long edge is already within the
    pixel cap passes through untouched, no byte budget applied — staying
    animated matters more than trimming an already-small GIF.
  - An animated image over the pixel cap has every frame downscaled,
    never flattened to one frame; its duration and loop count are kept.
  - A static image within both the pixel cap and the byte budget passes
    through untouched.
  - A static image within the pixel cap but over the byte budget is
    re-encoded at its current size.
  - A static image over the pixel cap is resized (EXIF orientation
    applied first, so a phone photo doesn't come out rotated) and
    re-encoded: JPEG ~q85 if opaque, PNG if it carries alpha.
  - Whatever gets re-encoded, the result is never larger than the
    input — the original bytes are always one of the candidates
    ``normalize_image`` picks from, so a re-encode that didn't pay off
    (or a resize whose encoding overhead ate the saving) falls back to
    the original rather than growing the row.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging

from PIL import Image, ImageOps, ImageSequence

from penny.constants import PennyConstants

logger = logging.getLogger(__name__)

# Pillow's save() format name for each mime type Penny already restricts
# incoming attachments to (``PennyConstants.VISION_SUPPORTED_CONTENT_TYPES``).
_FORMAT_BY_MIME = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/gif": "GIF",
    "image/webp": "WEBP",
}

_MIME_JPEG = "image/jpeg"
_MIME_PNG = "image/png"


def normalize_image(data: bytes, mime_type: str) -> tuple[bytes, str]:
    """Cap an image's pixel dimensions and byte size per the rules above.

    Returns ``(data, mime_type)`` unchanged whenever Pillow can't decode the
    input, or whenever no re-encode beats the original on size.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        logger.debug("normalize_image: undecodable input (%s); passing through", mime_type)
        return data, mime_type

    if getattr(img, "is_animated", False):
        return _normalize_animated(img, data, mime_type)
    return _normalize_static(img, data, mime_type)


def normalize_data_uri(uri: str) -> str:
    """Normalize the image payload of a ``data:<mime>;base64,<data>`` URI.

    Any string that isn't a decodable base64 data URI passes through
    unchanged — the ingress boundary's images are always built this way, but
    this stays defensive rather than assuming it.
    """
    if not uri.startswith("data:") or ";base64," not in uri:
        return uri
    header, encoded = uri.split(";base64,", 1)
    mime_type = header[len("data:") :]
    try:
        data = base64.b64decode(encoded)
    except binascii.Error:
        return uri
    normalized, normalized_mime = normalize_image(data, mime_type)
    return f"data:{normalized_mime};base64,{base64.b64encode(normalized).decode()}"


def _normalize_static(img: Image.Image, data: bytes, mime_type: str) -> tuple[bytes, str]:
    """A single-frame image: resize/re-encode only when the pixel cap or the
    byte budget is exceeded, and never return something bigger than ``data``."""
    long_edge = max(img.size)
    oversized = long_edge > PennyConstants.IMAGE_LONG_EDGE_CAP
    over_budget = len(data) > PennyConstants.IMAGE_PASSTHROUGH_MAX_BYTES
    if not oversized and not over_budget:
        return data, mime_type

    oriented = ImageOps.exif_transpose(img) or img
    target = oriented
    if oversized:
        target = _resize_to_cap(oriented, max(oriented.size))

    candidates: list[tuple[bytes, str]] = [(data, mime_type), _encode_target(target)]
    if oversized:
        same_format = _encode_same_format(target, mime_type)
        if same_format is not None:
            candidates.append((same_format, mime_type))
    return min(candidates, key=lambda candidate: len(candidate[0]))


def _normalize_animated(img: Image.Image, data: bytes, mime_type: str) -> tuple[bytes, str]:
    """An animated image: pass through untouched under the pixel cap (no byte
    budget applied to animation), otherwise downscale every frame and keep
    the frame count, durations, and loop — never flatten to one frame."""
    long_edge = max(img.size)
    if long_edge <= PennyConstants.IMAGE_LONG_EDGE_CAP:
        return data, mime_type

    scale = PennyConstants.IMAGE_LONG_EDGE_CAP / long_edge
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    frames: list[Image.Image] = []
    durations: list[int] = []
    disposals: list[int] = []
    for frame in ImageSequence.Iterator(img):
        durations.append(frame.info.get("duration", 0))
        disposals.append(frame.info.get("disposal", 0))
        frames.append(frame.convert("RGBA").resize(new_size, Image.Resampling.LANCZOS))

    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        disposal=disposals,
        loop=img.info.get("loop", 0),
    )
    resized = buf.getvalue()
    return min([(data, mime_type), (resized, mime_type)], key=lambda candidate: len(candidate[0]))


def _resize_to_cap(img: Image.Image, long_edge: int) -> Image.Image:
    scale = PennyConstants.IMAGE_LONG_EDGE_CAP / long_edge
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _has_alpha(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA"):
        return True
    return img.mode == "P" and "transparency" in img.info


def _encode_target(img: Image.Image) -> tuple[bytes, str]:
    """The format the normalized rules pick: PNG if the image carries alpha,
    JPEG q85 otherwise."""
    buf = io.BytesIO()
    if _has_alpha(img):
        img.convert("RGBA").save(buf, format="PNG")
        return buf.getvalue(), _MIME_PNG
    img.convert("RGB").save(buf, format="JPEG", quality=PennyConstants.IMAGE_JPEG_QUALITY)
    return buf.getvalue(), _MIME_JPEG


def _encode_same_format(img: Image.Image, mime_type: str) -> bytes | None:
    """Re-save ``img`` in its ORIGINAL format — the fallback candidate for an
    oversized image whose target-format re-encode didn't save bytes; a
    resize alone, kept in the source codec, usually still does. Returns
    ``None`` when the original format can't be resolved or re-saved."""
    fmt = _FORMAT_BY_MIME.get(mime_type)
    if fmt is None:
        return None
    buf = io.BytesIO()
    try:
        if fmt == "JPEG":
            img.convert("RGB").save(buf, format=fmt, quality=PennyConstants.IMAGE_JPEG_QUALITY)
        else:
            img.save(buf, format=fmt)
    except Exception:
        logger.debug("normalize_image: same-format fallback encode failed for %s", mime_type)
        return None
    return buf.getvalue()
