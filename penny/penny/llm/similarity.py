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
    return means the embed call itself failed transiently across the client's
    retries; how the caller handles that differs.  Memory-entry writes
    (``collection_write`` / ``log_append``) REFUSE the write rather than store a
    vectorless, recall-invisible entry (#1412), while store-and-backfill paths (a
    collection's description anchor, logged messages) keep the row and let the
    startup backfill vectorize it later.
    """
    try:
        vecs = await client.embed(text)
        return vecs[0]
    except LlmError:
        logger.warning("Failed to embed text: %.60s", text)
        return None
