"""The browse result picker — which search result to read, decided behind the
tool boundary (#2718, §2.7).

A search item's results list is never shown to the main loop: the picker ranks
it and the tool reads the ranked results, so the model always gets what the
pages say and never a list of titles to choose from.  The picker is a micro-
context-shaped decision: one question over a declared set, its output validated,
an honest escape when no usable answer comes back — never a guessed pick.

Two arms, decided at the construction site from config (the :class:`EntryDecider`
pattern):

- **Decision model** (``LLM_DECISION_MODEL`` set): Ollama's ``/v1/systemone``
  choice question — one question, one option per result plus ``none`` — ranked
  from the answer's per-option probabilities.  Measured ~0.4 s a pick on
  ``strands-decider:2b``.
- **Degraded** (no decision model): the shared text model answers the same
  question as one thinking-off draw.  Measured 33/33 valid picks on gemma.
  The preflight names this mode; it is never silent.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

from penny.config_params import RuntimeParams, token_budget_for
from penny.constants import PennyConstants
from penny.llm.call_link import CallLink
from penny.llm.decision_client import DecisionAnswer, DecisionClient, DecisionError
from penny.llm.models import LlmMessage

if TYPE_CHECKING:
    from penny.llm import LlmClient

logger = logging.getLogger(__name__)

# The pick question — stated over the STATE (what was searched, what it is
# looking for) and the principle, never over a fixture's wording or a verb the
# model should reason past.  Measured wording (33 logged searches): valid picks
# 33/33 on the degraded arm, as sensible as the chat model's own choices.
_PICK_INSTRUCTIONS = (
    "Which of these results is most likely to carry what is being looked for? "
    "Choose the one result whose page most likely carries it, or none when none "
    "of them is likely to."
)
# The option name and description for "none of these".
_NONE_OPTION = "none"
_NONE_CRITERION = "none of these results is likely to carry it"

# The fall-through question (#2718, the code owner's ruling): after each picked
# read, did the page turn out to have any of what was looked for?  A "no" (or an
# extraction that never came back usable) reads the next ranked result.
#
# The wording asks the decider to READ THE EXTRACT and report what it says,
# with criteria that name the two things an extract can say (found it / the
# page does not have it).  The first cut asked "does this extract carry any of
# what was looked for?" — on the measured failing state (looking for an adult
# ticket price, the extract answering "[1] — no current price for an adult
# ticket on the page.") the small decision model answered YES at 0.69: it keyed
# on the words the two texts share, not on what the extract says.  This wording
# answers no at 0.98 on that state and holds all four replayed controls (a
# carrying price extract 0.97, a carrying service extract 0.95, two none
# extracts 0.95/0.98).
_CARRIES_INSTRUCTIONS = (
    "Read what the extract says. Did the page turn out to have any of what was looked for?"
)
_CARRIES_CRITERIA = {
    "yes": "the extract reports finding it",
    "no": "the extract reports the page does not have it",
}

# The degraded arm's draw prompt — the same question in plain words, answered
# with just the number or the word none.  One reroll on an unparseable answer,
# then the honest escape.  Composed at draw time from the question constants.
_DEGRADED_PICK_PROMPT_TEMPLATE = (
    "You are picking which search result to read.\n\n"
    "Search: {query}\n"
    "Looking for: {looking_for}\n\n"
    "The results:\n"
    "{listed}\n\n"
    "Answer with just the number of the one result to read, or the word none."
)
_DEGRADED_PICK_ATTEMPTS = 2

# The degraded arm's accepted answer forms: a bare number, or the word none
# (case-insensitive, tolerating surrounding punctuation).  Anything else is an
# invalid answer — rerolled once, then the honest escape.
_ANSWER_NUMBER_RE = re.compile(r"^\D*?(\d+)\D*$")
_ANSWER_NONE_RE = re.compile(r"^\W*none\W*$", re.IGNORECASE)


class PickOutcome(StrEnum):
    """The enumerated outcome of one ranking — a closed set the tool renders one
    way each.  ``FAILED`` is the honest escape (no usable answer): the tool then
    renders the results list itself, so the model keeps the anchors."""

    RANKED = "ranked"
    NONE = "none"
    FAILED = "failed"


class PickResult(NamedTuple):
    """One ranking's outcome: the closed outcome plus the ranked result numbers
    (best first, 1-based list positions) on :attr:`PickOutcome.RANKED`."""

    outcome: PickOutcome
    ranked: list[int]


class ResultPicker:
    """Ranks a search's results behind the tool boundary.

    Holds the :class:`DecisionClient`, or ``None`` for the named degraded mode
    (the shared text model picks with thinking off).  Constructed once from
    config at the browse tool's construction; never ambient state."""

    def __init__(
        self,
        decision_client: DecisionClient | None,
        model_client: LlmClient | None,
    ) -> None:
        self._decision_client = decision_client
        self._model_client = model_client

    async def rank(
        self,
        *,
        query: str,
        looking_for: str,
        results: list[tuple[int, str, str]],
        call_link: CallLink,
    ) -> PickResult:
        """Rank one search's results.  ``results`` is ``(number, title, url)``
        in declared order; the returned numbers index that order.  An empty
        result list is :attr:`PickOutcome.NONE` outright — there is nothing to
        ask about."""
        if not results:
            return PickResult(PickOutcome.NONE, [])
        if self._decision_client is not None:
            return await self._rank_by_decision_model(
                query=query, looking_for=looking_for, results=results
            )
        return await self._rank_by_chat_draw(
            query=query, looking_for=looking_for, results=results, call_link=call_link
        )

    async def carries_any(
        self, *, looking_for: str, extract_text: str, call_link: CallLink
    ) -> bool:
        """Whether one picked page's extract carries any of what was looked for.

        A ``no`` answer is what sends the search on to the next ranked result.
        An unreadable answer counts as YES — the ruling reads the failure as
        benefit of the doubt, and the page's extract is reported as what it held
        either way.  The degraded mode never asks this question (the caller only
        asks when a decision model is configured); if asked there anyway, the
        answer is yes for the same reason."""
        client = self._decision_client
        if client is None:
            return True
        state = f"Looking for: {looking_for}\nExtract: {extract_text}"
        try:
            answer = await client.decide(
                state=state, instructions=_CARRIES_INSTRUCTIONS, criteria=_CARRIES_CRITERIA
            )
        except DecisionError as error:
            logger.warning("browse picker carry-check failed — treating as carried: %s", error)
            return True
        return answer.choice != "no"

    async def _rank_by_decision_model(
        self, *, query: str, looking_for: str, results: list[tuple[int, str, str]]
    ) -> PickResult:
        """The primary arm: one ``choice`` question over the numbered results
        plus ``none``, ranked from the answer's per-option probabilities."""
        client = self._decision_client
        if client is None:  # only reachable from rank()'s own dispatch
            return PickResult(PickOutcome.FAILED, [])
        criteria = {str(number): f"{title} — {url}" for number, title, url in results}
        criteria[_NONE_OPTION] = _NONE_CRITERION
        state = f"Search: {query}\nLooking for: {looking_for}"
        try:
            answer = await client.decide(
                state=state, instructions=_PICK_INSTRUCTIONS, criteria=criteria
            )
        except DecisionError as error:
            logger.warning("browse picker ranking failed: %s", error)
            return PickResult(PickOutcome.FAILED, [])
        if answer.choice == _NONE_OPTION:
            return PickResult(PickOutcome.NONE, [])
        known = {number for number, _, _ in results}
        try:
            chosen = int(answer.choice)
        except ValueError:
            logger.warning("browse picker answered outside its options: %r", answer.choice)
            return PickResult(PickOutcome.FAILED, [])
        if chosen not in known:
            logger.warning("browse picker answered outside its options: %r", answer.choice)
            return PickResult(PickOutcome.FAILED, [])
        return PickResult(PickOutcome.RANKED, _ranked_numbers(answer, chosen, known))

    async def _rank_by_chat_draw(
        self,
        *,
        query: str,
        looking_for: str,
        results: list[tuple[int, str, str]],
        call_link: CallLink,
    ) -> PickResult:
        """The degraded arm: the shared text model, thinking off, answers the
        same pick question as a plain draw.  An unparseable answer is rerolled
        once on the unchanged prompt; no usable answer is the honest escape."""
        if self._model_client is None:
            return PickResult(PickOutcome.FAILED, [])
        listed = "\n".join(f"{number}. {title} — {url}" for number, title, url in results)
        listed = f"{listed}\n{_NONE_OPTION} — {_NONE_CRITERION}"
        prompt = f"{_PICK_INSTRUCTIONS}\n\n" + _DEGRADED_PICK_PROMPT_TEMPLATE.format(
            query=query, looking_for=looking_for, listed=listed
        )
        messages = [LlmMessage(role="user", content=prompt)]
        runtime = self._model_client.runtime
        token_budget = token_budget_for(
            runtime if runtime is not None else RuntimeParams(),
            PennyConstants.BROWSE_PICK_AGENT_NAME,
        )
        for attempt in range(_DEGRADED_PICK_ATTEMPTS):
            response = await self._model_client.chat(
                messages=messages,
                agent_name=PennyConstants.BROWSE_PICK_AGENT_NAME,
                prompt_type=PennyConstants.BROWSE_PICK_PROMPT_TYPE,
                call_link=call_link.child(),
                token_budget=token_budget,
                thinking_enabled=False,
            )
            pick = _parse_pick_answer(response.content or "", {n for n, _, _ in results})
            if pick is not None:
                return pick
            logger.warning(
                "browse picker draw unusable — discarding and re-rolling %d/%d",
                attempt + 1,
                _DEGRADED_PICK_ATTEMPTS,
            )
        return PickResult(PickOutcome.FAILED, [])


def _ranked_numbers(answer: DecisionAnswer, chosen: int, known: set[int]) -> list[int]:
    """The ranked result numbers from a decision-model answer: the chosen pick
    first, then every other result by descending probability (a probability the
    endpoint did not report ranks last, in declared order)."""
    others = [n for n in known if n != chosen]
    probabilities = answer.probabilities or {}
    others.sort(key=lambda n: (-probabilities.get(str(n), 0.0), n))
    return [chosen, *others]


def _parse_pick_answer(text: str, known: set[int]) -> PickResult | None:
    """The degraded draw's answer parsed into a :class:`PickResult`, or ``None``
    when it names neither a known number nor ``none`` — the reroll trigger."""
    stripped = text.strip()
    if _ANSWER_NONE_RE.match(stripped):
        return PickResult(PickOutcome.NONE, [])
    match = _ANSWER_NUMBER_RE.match(stripped)
    if match is None:
        return None
    number = int(match.group(1))
    if number not in known:
        return None
    return PickResult(PickOutcome.RANKED, [number])
