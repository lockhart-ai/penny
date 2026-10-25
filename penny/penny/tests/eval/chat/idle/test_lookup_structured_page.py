"""A lookup whose ask names page STRUCTURE the page's own text doesn't literally
carry (#2816).

The contract is `docs/eval-case-design.md`.

This is the stress case for cutting the chat prompt's "no CSS selectors, XPaths,
or HTML parsing" sentence (`Prompt.CONVERSATION_TAIL_OPEN`'s closing line): the
guard it names — never ask the user for page structure, snippets, or selectors —
has no tool-level trigger (`BrowseTool` never exposes anything shaped like a
selector), so the only way to see whether gemma still needs telling is an ask
that TEMPTS the failure the sentence names. The page is a plain prose spec
sheet — a "Specs" block and a "Pricing" block, each line a fact, no markup of
any kind — but the user asks for a "row" in a "table," including one phrasing
that explicitly disclaims knowing the layout. A model that reads the page
anyway states the battery life and the base price regardless of whether a
literal table exists; a model that takes the "table"/"row" language at face
value and can't find one is the failure this case exists to catch — asking the
user back for the markup, the selector, or where on the page to look, instead
of just reading what's there.

**What is NOT claimed.** That she called `browse`, whether she used `extract`,
or how she words the reply — all ROUTE and spread, measured but never
asserted. The claim reads only whether the two facts the page states survive
into the reply, per `docs/principles.md` §4's "assert what survives, never
that the model refrained" — there is no claim here that the reply omits a
request for a selector, only that the facts came through regardless.

REPORT-ONLY (`min_pass_rate=None`): the ceilings these runs propose are the
code owner's to accept once the numbers have been read. The host is real
(cnet.com); the laptop, its review, and every spec and price are invented,
because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import ENTRIES_STORED, REPLY_SPREAD, TOOL_SEQUENCE
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "page-structure-ask"

# ── The world: one page, no markup, plain prose specs and pricing ───────────
#
# The page reads like a real review — a "Specs" block and a "Pricing" block,
# each line its own fact — but there is no table anywhere in the text: the ask
# will name "rows" and a "table" that the page never renders that way, so a
# model that insists on literal structure before answering has nothing to
# point at.

_PRODUCT_URL = "https://www.cnet.com/reviews/vantage-x200-review/"

_PRODUCT_PAGE = CannedPage(
    match="vantage-x200",
    text=(
        "Title: Vantage X200 review — specs and pricing, cnet.com\n"
        "The Vantage X200 is a 14-inch ultrabook built for frequent travelers, "
        "reviewed here after two weeks of daily use.\n"
        f"Source: {_PRODUCT_URL}\n"
        "\n"
        "Specs\n"
        "Weight: 1.1 kilograms, charger included.\n"
        "Battery life: rated at 14 hours of continuous video playback, about "
        "10 hours of mixed use.\n"
        "Display: 14-inch OLED panel, 2880x1800 resolution, 120Hz refresh.\n"
        "Processor: an 8-core chip with integrated graphics, no discrete GPU "
        "option.\n"
        "\n"
        "Pricing\n"
        "The base configuration, 16GB of memory and a 512GB drive, is $1,399.\n"
        "A 32GB/1TB configuration is $1,699.\n"
        "Both ship with a one-year warranty.\n"
        "\n"
        "Verdict\n"
        "A strong travel laptop if OLED and long battery life matter more than "
        "raw GPU power.\n"
    ),
)

# The two facts the ask is actually answered by — one word each, neither
# pluralizable or rephrasable in a way typography-folding wouldn't catch: the
# battery figure and the base price, each stated only once on the page.
_BATTERY_ANSWER = "14 hours"
_PRICE_ANSWER = "1,399"

_WORLD = World(
    name="a plain-prose spec page, asked about as a table",
    pages=(_PRODUCT_PAGE,),
    keeps=(),
    excludes=(),
    answers=(_BATTERY_ANSWER, _PRICE_ANSWER),
)

# ── The three wordings ───────────────────────────────────────────────────────
#
# All three name a "row"/"table" the page doesn't literally have; the second
# goes further and says outright that the layout is unknown — the phrasing
# closest to inviting "can you give me the selector" back.

_ASK = (
    f"can you check this laptop review for me — {_PRODUCT_URL} — what's in the "
    "battery life row of the specs table, and what's the price row say for "
    "the base config?"
)
_ASKS = (
    f"check the specs table on this listing — {_PRODUCT_URL} — i don't know "
    "the exact layout, but what's the battery life and the base price?",
    f"go look at {_PRODUCT_URL} — what's in the pricing table's row for the "
    "base config, and what does the battery life row say?",
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_structural_ask_still_gets_read_from_the_page(
    chat_eval: ChatEval, model: str
) -> None:
    """The ask names a table and rows the page's own text never renders that
    way. The facts it asks for are on the page in plain prose, and the reply
    states them regardless of the mismatch between the ask's words and the
    page's actual shape."""
    cohort = await chat_eval(
        case_id="lookup-page-structure-named-in-the-ask",
        behaviour=(
            "In the chat agent, when the ask names page structure (a table, a "
            "row) the page's own text doesn't literally render that way, "
            "Penny reads the page anyway and states the facts asked for."
        ),
        area=Area.READING_A_PAGE,
        model=model,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ASKS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # STORE — a read-only ask: nothing here sends her to the store.
    cohort.assert_what_the_store_held_survives()
    # COMPLETENESS — the facts the page states survive into the reply, however
    # the ask's own "table"/"row" words land.
    cohort.assert_the_reply_answers_the_ask()
    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
