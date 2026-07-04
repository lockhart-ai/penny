"""Penny-specific similarity operations that depend on LlmClient or penny's data model.

Pure math primitives (cosine, TCR, dedup, serialization) live in the shared
`similarity/` package — import those directly, not via this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from penny.llm.models import LlmError

if TYPE_CHECKING:
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)

# ── Safe embedding helper ─────────────────────────────────────────────────────


async def embed_text(
    client: LlmClient,
    text: str,
) -> list[float] | None:
    """Embed a single text string.  Returns None only on a transient embed failure.

    The embedding model is a required prerequisite, so the client is always
    present — there is no "no model configured" degraded path here.  A ``None``
    return means the embed call itself failed transiently; the caller keeps its
    non-embedded write and the startup backfill vectorizes it later.
    """
    try:
        vecs = await client.embed(text)
        return vecs[0]
    except LlmError:
        logger.warning("Failed to embed text: %.60s", text)
        return None
