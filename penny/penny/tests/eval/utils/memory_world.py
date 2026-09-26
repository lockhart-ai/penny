"""The memory stories' family tag, and the two-source ask."""

from __future__ import annotations

from penny.tests.eval.utils.worlds import FOXES_URL, SEALS_URL

_FAMILY = "chat-memory"


# The two-source watch ask, as `memory-writes-landed-source-down` has always asked it.  It stays
# in the conversational register that case was written and measured against — it scores a reply
# about a source being unreachable, not the shape of a demonstrated round, so the register work
# on the learn case does not apply to it and changing it here would move an unrelated number.
TWO_SOURCE_ASK = (
    f"go to {FOXES_URL} and {SEALS_URL}, pull out the trades and signings from "
    "each, and keep the headline plus a short blurb in a team news list for me"
)
