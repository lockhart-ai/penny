"""Post-processing for web page content returned by the browser extension.

Strips structural cruft (proxy images, navigation, licensing boilerplate)
that Turndown faithfully converts to markdown but that wastes model context.

Patterns are grouped by source:
  - Kagi search result pages (proxy images, image grids, Openverse)
  - Cross-site (tracking pixels, JSON blobs, nav headers, long auth URLs)

To find new patterns: check the logs for large browse_url results, compare
the raw and cleaned sizes, and look for repetitive non-content lines.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Kagi-specific ─────────────────────────────────────────────────────────

# Kagi image grid section starts with [Images] or [Videos] links
_KAGI_IMAGE_VIDEO_HEADER = re.compile(r"^\[(Images|Videos)]\(https://kagi\.com/")

# Bare domain attribution under Kagi thumbnail images: [www.example.com](https://...)
_KAGI_DOMAIN_ATTRIBUTION = re.compile(r"^\[www\.\S+]\(https://")

# Openverse metadata labels in the Kagi image lightbox
_KAGI_METADATA_LABELS = frozenset({"Size", "Type", "Uploaded", "Aspect"})

# ── Cross-site ────────────────────────────────────────────────────────────

# Empty markdown links with no display text: [](https://...)
_EMPTY_LINK = re.compile(r"^\[]\(https?://")

# Image dimension lines: "1920 x 1080", "980 x 980"
_DIMENSIONS = re.compile(r"^\d{2,5}\s*x\s*\d{2,5}$")

# Tracking pixel images (Amazon, analytics, etc.)
_TRACKING_PIXEL = re.compile(r"!\[.*?]\(https?://[^\)]*transparent-pixel[^\)]*\)")

# Inline JSON blobs (Amazon pricing, structured data) — line is >{...}<
_JSON_BLOB = re.compile(r'^["\']?\{".{50,}\}["\']?$')

# Links where the URL portion exceeds 300 chars (auth/signin, tracking)
_LONG_URL_LINK = re.compile(r"^\[.*?\]\((https?://[^\)]{300,})\)$")

# Navigation boilerplate lines
_NAV_BOILERPLATE = frozenset(
    {
        "Skip to main content",
        "Skip to search results",
        "Skip to content",
        "Skip to navigation",
        "Hide Filters",
        "Reset filters",
        "Article continues below",
        "Show Filters",
        "View Image Gallery",
    }
)

# Repeated category label pattern from listing pages
# e.g. "NewsCategory: News.|TechnologyCategory: Technology."
_CATEGORY_LABELS = re.compile(r"^(\w+Category:\s*\w+\.\|?){2,}$")

# ── Guarded page-chrome trims (#2759) ───────────────────────────────────────
# Each rule below is bounded by a length guard so it can only touch chrome,
# never prose: a long paragraph that happens to repeat, picture, or mention a
# chrome phrase survives untouched.

# Rule 1 (repeated lines): a line at least this long that already appeared
# earlier on the page is a re-rendered nav/footer block, not prose; a short
# repeated line (a table cell, a label) stays.
_MIN_REPEATED_LINE_LEN = 20

# Rule 2 (image markup): a line that is ONLY a markdown image carries no text
# the reader uses, whatever its URL.
_IMAGE_ONLY_LINE = re.compile(r"^!\[[^\]]*]\([^)]*\)$")

# Rule 3 (boilerplate): SHORT lines (guard: at most this many chars) matching
# known UI chrome phrasing; a long prose line that happens to mention the
# same words (e.g. "...read our privacy policy before you...") stays.
_MAX_BOILERPLATE_LINE_LEN = 80
_BOILERPLATE_PHRASES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"cookie",
        r"consent",
        r"privacy policy",
        r"subscribe",
        r"sign in",
        r"sign up",
        r"related (articles?|stories)",
        r"more stories",
        r"you might also like",
        r"share this",
        r"follow us",
        r"advertisement",
        r"all rights reserved",
        r"copyright \d{4}",
        r"^skip to",
    )
]

# Rule 4 (blob lines): a line over this many chars that is a JSON-like blob
# or carries a URL over 200 chars is a spec/price dump or an encoded URL, not
# prose; a prose line this long is never truncated, only left as is.
_MAX_BLOB_LINE_LEN = 500
_JSON_LIKE_LINE = re.compile(r"^[\[{].*[\]}]$")
_LONG_URL_IN_LINE = re.compile(r"https?://\S{200,}")


def clean_browser_content(text: str) -> str:
    """Remove structural cruft from browser-extracted markdown content."""
    lines = text.split("\n")
    cleaned: list[str] = []
    skipping_kagi_images = False

    for line in lines:
        stripped = line.strip()

        if _should_skip_line(stripped):
            continue

        # Kagi image/video grid — skip everything until next search result
        if _KAGI_IMAGE_VIDEO_HEADER.match(stripped):
            skipping_kagi_images = True
            continue

        if skipping_kagi_images:
            if stripped.startswith("### ") or stripped.startswith("![Favicon"):
                skipping_kagi_images = False
                if stripped.startswith("![Favicon"):
                    continue
            else:
                continue

        cleaned.append(line)

    cleaned = _apply_guarded_trims(cleaned)

    result = "\n".join(cleaned)
    # Collapse runs of 3+ blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)

    removed = len(text) - len(result)
    if removed > 0:
        logger.debug(
            "content_cleaning: %d -> %d chars (removed %d, %.0f%%)",
            len(text),
            len(result),
            removed,
            removed / len(text) * 100,
        )

    return result


def _should_skip_line(stripped: str) -> bool:
    """Return True if this line is structural cruft that should be removed."""
    # ── Kagi-specific ──
    if stripped.startswith("![Favicon"):
        return True
    if stripped.startswith(("[![](https://p.kagi.com/proxy/", "![](https://p.kagi.com/proxy/")):
        return True
    if _KAGI_DOMAIN_ATTRIBUTION.match(stripped):
        return True
    if stripped in _KAGI_METADATA_LABELS:
        return True
    if "Made with [Openverse]" in stripped:
        return True
    if "Upload time and" in stripped:
        return True
    if stripped.startswith(("[View Image]", "[Download]", "[Visit Page]")):
        return True
    if "Loading source..." in stripped or "Loading license..." in stripped:
        return True
    if stripped.startswith("Report to [OpenVerse]"):
        return True

    # ── Cross-site ──

    # Empty links
    if _EMPTY_LINK.match(stripped):
        return True

    # Image dimension lines
    if _DIMENSIONS.match(stripped):
        return True

    # Tracking pixel images
    if _TRACKING_PIXEL.match(stripped):
        return True

    # Inline JSON blobs
    if _JSON_BLOB.match(stripped):
        return True

    # Links with extremely long URLs (auth/signin/tracking)
    if _LONG_URL_LINK.match(stripped):
        return True

    # Navigation boilerplate
    if stripped in _NAV_BOILERPLATE:
        return True

    # Repeated category labels from listing pages
    return bool(_CATEGORY_LABELS.match(stripped))


def _apply_guarded_trims(lines: list[str]) -> list[str]:
    """Rules 1-4 (#2759), each guarded so it can only touch chrome, never
    prose: repeated lines, then image markup, boilerplate, and blob lines."""
    lines = _drop_repeated_lines(lines)
    lines = [ln for ln in lines if not _is_image_only_line(ln.strip())]
    lines = [ln for ln in lines if not _is_boilerplate_chrome_line(ln.strip())]
    lines = [ln for ln in lines if not _is_blob_line(ln.strip())]
    return lines


def _drop_repeated_lines(lines: list[str]) -> list[str]:
    """Rule 1 (#2759): keep a line's first occurrence; drop a later repeat of
    any line at least `_MIN_REPEATED_LINE_LEN` chars long."""
    seen: set[str] = set()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) >= _MIN_REPEATED_LINE_LEN:
            if stripped in seen:
                continue
            seen.add(stripped)
        kept.append(line)
    return kept


def _is_image_only_line(stripped: str) -> bool:
    """Rule 2 (#2759): the line carries nothing but a markdown image."""
    return bool(_IMAGE_ONLY_LINE.match(stripped))


def _is_boilerplate_chrome_line(stripped: str) -> bool:
    """Rule 3 (#2759): a SHORT line matching known UI chrome phrasing."""
    if len(stripped) > _MAX_BOILERPLATE_LINE_LEN:
        return False
    return any(pattern.search(stripped) for pattern in _BOILERPLATE_PHRASES)


def _is_blob_line(stripped: str) -> bool:
    """Rule 4 (#2759): a long line that is a JSON-like blob or carries a very
    long URL; a prose line this long is left as is, never truncated."""
    if len(stripped) <= _MAX_BLOB_LINE_LEN:
        return False
    return bool(_JSON_LIKE_LINE.match(stripped)) or bool(_LONG_URL_IN_LINE.search(stripped))
