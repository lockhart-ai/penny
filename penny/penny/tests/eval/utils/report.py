"""The transcript-integrated eval report grammar (#1725, iteration-6 / option-6b).

This is the PURE renderer for one sample's transcript: it turns a structured
``SampleTranscript`` (built from the persisted promptlog by ``conftest.py``) into the
per-step markdown tables the format spec (``docs/eval-report-format.md``) defines. No
model, no git, no DB — a hand-built ``SampleTranscript`` renders identically to one
extracted from a real run, which is what makes the whole-render tests possible.

The grammar (one fixed form per row type, used identically everywhere):

- **step header** — ``step N · 👤 | "message" | step-verdict`` (the markdown table header).
- **expected** — ``Cn [class]marker label`` in the body; the score cell is empty, or the
  verdict for a no-evidence-row contract (a whole-run/missing-action check).
- **💭** — an ALWAYS-collapsed ``<details>`` directly ABOVE the model action it produced,
  one per action; an empty thought renders as ``💭 (empty)``, never omitted.
- **actual** — one transcript event (🔧 call · 📥 result · 🤖 reply · 👤 nudge · 🧩 micro),
  its verdict on the anchor row (``glyph Cn — rationale · cause``), ``⚠ recovery event`` on
  a nudge row, else empty.
- **baseline** — the prior run's anchor event (diff mode), score cell = the prior verdict.
- **note** — free text, always last, score cell always empty.

Run-close contracts (whole-run properties with no single evidence row) render their
verdicts on their own ``expected`` rows in a trailing ``run-close`` table.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from penny.tests.eval.utils import cohort

# ── Actor glyphs (the transcript-event vocabulary) ───────────────────────────
ACTOR_USER = "👤"
ACTOR_CALL = "🔧"
ACTOR_RESULT = "📥"
ACTOR_REPLY = "🤖"
ACTOR_MICRO = "🧩"

# ── Micro-context role labels (#1759/#1773) — the input row names the scoped USER turn explicitly
# (the user mistook it for the system prompt); the output row keeps the bare drawn-value arrow.
# The label between glyph and arrow is the calling CONTEXT (``Event.context`` — the ledger identity
# of the sub-model: browse extraction, state classification, skill naming), so the three customers
# read as distinct actors and each row's label matches its system-prompt row's. ──
MICRO_CONTEXT_LABEL = "micro-context"  # the actor label when an event names no context
MICRO_IN_ARROW = "← user turn:"
MICRO_OUT_ARROW = "→"

# ── System-prompt row (#1759) — one always-collapsed row per distinct context's system prompt ──
SYSTEM_PROMPT_LABEL = "system prompt"

# ── Check class + scored/advisory markers ────────────────────────────────────
GATING_MARKER = "⚖"  # a scored check (counts toward the sample score)
ADVISORY_MARKER = "ℹ"  # flavour — renders, never scores
NA_MARK = "➖ n/a"  # a not-applicable check (its branch didn't run this sample)

# ── Verdict marks (the score-cell glyphs) ────────────────────────────────────
PASS_MARK = "✅"
FAIL_MARK = "❌"
REGRESSED_MARK = "✅→❌ **REGRESSED**"
FIXED_MARK = "❌→✅ **FIXED**"
RECOVERY_MARK = "⚠ recovery event"

# ── Row-label literals (column 1 of every table row) ─────────────────────────
ROW_EXPECTED = "expected"
ROW_THINKING = "💭"
ROW_ACTUAL = "actual"
ROW_BASELINE = "baseline"
ROW_NOTE = "note"

RUN_CLOSE_LABEL = "run-close"
RUN_CLOSE_TITLE = "whole-conversation contracts"
EMPTY_THINKING = "💭 (empty)"
NO_TURNS_PLACEHOLDER = (
    "_(no completed turns recorded — the sample produced no finished model call, "
    "e.g. a harness timeout)_"
)
TABLE_DIVIDER = "|---|---|---|"
CELL_TRUNCATE_LIMIT = 500  # an actual cell over this collapses into a single <details> (#1759)

# ── Sample-block grammar (the uniform-collapse skeleton, #1753) ───────────────
SAMPLE_ROW = "sample"  # every sample banner opens ``sample N — <banner>``


# ── Deterministic cell hygiene ───────────────────────────────────────────────
def escape_cell(text: str) -> str:
    """One table cell: escape ``|`` and render newlines as ``<br>`` so a multi-line body
    stays inside its cell (the cell-escaping rule)."""
    return text.replace("|", "\\|").replace("\n", "<br>")


def truncate_cell(text: str, limit: int = CELL_TRUNCATE_LIMIT) -> str:
    """The deterministic truncation rule (#1759): an over-long cell collapses into a SINGLE
    ``<details>`` — the summary is its first line + ``… (<n> chars)``, the FULL (escaped) text
    inside it. One copy, no visible head (consistent with everything-defaults-collapsed); the old
    head + nested-full form duplicated the head on expand. Escapes once, at the end."""
    if len(text) <= limit:
        return escape_cell(text)
    first_line = escape_cell(text.split("\n", 1)[0])
    full = escape_cell(text)
    return f"<details><summary>{first_line} … ({len(text)} chars)</summary>{full}</details>"


# ── The score-cell verdict (one check's outcome, rendered) ───────────────────
@dataclass
class Verdict:
    """One check's rendered outcome for a score cell. ``mark`` is the glyph (pass/fail/
    regressed/fixed/recovery/na); ``check_id`` the ``Cn``/``Gn`` anchor (omitted for a bare
    recovery/na cell); ``rationale``/``cause`` the observed-vs-expected note + failure cause;
    ``prior`` tags a baseline row's verdict as the prior run's."""

    mark: str
    check_id: str | None = None
    rationale: str | None = None
    cause: str | None = None
    prior: bool = False

    def render(self) -> str:
        parts = [self.mark]
        if self.check_id:
            parts.append(self.check_id)
        cell = " ".join(parts)
        if self.rationale:
            cell += f" — {self.rationale}"
        if self.cause:
            cell += f" · {self.cause}"
        if self.prior:
            cell += " *(prior run)*"
        return cell


def render_score(verdicts: list[Verdict]) -> str:
    """The score cell — every verdict on this row, joined by ``·`` (empty when none)."""
    return " · ".join(verdict.render() for verdict in verdicts)


# ── The rows of one step's table ─────────────────────────────────────────────
@dataclass
class Row:
    """One rendered table row: its column-1 label, the middle body cell, and the score cell.

    The body is stored VERBATIM (already the caller's chosen text); ``escape`` applies the
    cell hygiene at render time so a ``|`` or newline in a tool call can't break the table.
    A 💭 row carries its collapsed ``<details>`` as ``body`` and never escapes (it is markup)."""

    label: str
    body: str
    verdicts: list[Verdict] = field(default_factory=list)
    escape: bool = True

    def render(self) -> str:
        body = truncate_cell(self.body) if self.escape else self.body
        return f"| {self.label} | {body} | {render_score(self.verdicts)} |"


def thinking_row(thinking: str) -> Row:
    """A 💭 row — an always-collapsed ``<details>`` above the action it produced. Empty
    thinking renders as ``💭 (empty)`` (an empty thought before a degenerate act is signal)."""
    body = EMPTY_THINKING if not thinking.strip() else _thinking_details(thinking)
    return Row(ROW_THINKING, body, escape=False)


def micro_thinking_row(thinking: str, context: str | None = None) -> Row:
    """A 💭 row for a micro-context call — labelled ``thinking (<context>)`` so the sub-model's
    reasoning is attributed to the actor that produced it (#1773); an event naming no context
    keeps the generic ``thinking (micro-context)``."""
    if not thinking.strip():
        return Row(ROW_THINKING, EMPTY_THINKING, escape=False)
    body = _thinking_details(thinking, summary=f"thinking ({context or MICRO_CONTEXT_LABEL})")
    return Row(ROW_THINKING, body, escape=False)


def _thinking_details(thinking: str, summary: str = "thinking") -> str:
    """The collapsed ``<details>`` markup for a thinking trace (newlines collapsed to spaces so
    the whole trace stays inside one table cell)."""
    body = escape_cell(thinking.strip()).replace("<br>", " ")
    return f"<details><summary>{summary}</summary>{body}</details>"


# ── A step (a user turn and everything it produced) ──────────────────────────
@dataclass
class Step:
    """One conversational step: a user turn opens it, then its expected/💭/actual/note rows.
    ``verdict`` is the header's step-level roll-up (✅ all its checks passed · ❌ one failed ·
    ✅→❌ a flip · blank for a step with no scored checks)."""

    number: int
    user_message: str
    verdict: str
    rows: list[Row]

    def render(self) -> str:
        msg = escape_cell(self.user_message)
        header = f'| step {self.number} · {ACTOR_USER} | "{msg}" | {self.verdict} |'
        return "\n".join([header, TABLE_DIVIDER, *[row.render() for row in self.rows]])


@dataclass
class RunClose:
    """The trailing table of whole-conversation contracts — checks with no evidence row, each
    verdict on its own ``expected`` row. ``score`` is the case's ``k/n`` scored total."""

    score: str
    rows: list[Row]

    def render(self) -> str:
        header = f"| {RUN_CLOSE_LABEL} | {RUN_CLOSE_TITLE} | {self.score} |"
        return "\n".join([header, TABLE_DIVIDER, *[row.render() for row in self.rows]])


# ── The system-prompt row (#1759) — one collapsed block per distinct context ─
@dataclass
class SystemPrompt:
    """One distinct system prompt among a sample's promptlog calls (main agent + each micro-context
    flavour), rendered as an ALWAYS-collapsed ``<details>`` directly under the sample banner: the
    summary names the ``context`` (the agent name) + the prompt size, the verbatim prompt sits
    inside. Distinct prompts within a sample dedupe by text — a repeated main-loop prompt renders
    once. The user mistook a micro-context's user turn for its system prompt; this makes each
    context's real system prompt visible (its own row) without inflating any step table."""

    context: str
    text: str

    def render(self) -> str:
        summary = f"{SYSTEM_PROMPT_LABEL} — {self.context} ({len(self.text)} chars)"
        return f"<details><summary>{summary}</summary>\n\n{self.text}\n\n</details>"


# ── How a case reads at a glance ─────────────────────────────────────────────
#
# The default view of a run is ONE LINE PER CASE.  Everything else is behind a fold, so ~100
# cases is ~100 lines a person can page through, and any case opens to its full detail.
#
# The scale is a THRESHOLD scale, not a set of states: a rate of 0.61 should look like a warning
# at a glance rather than sharing a glyph with 0.05.  Colour is reserved for what is GATED —
# grey means measured with no threshold behind it, which is most of the report today and must
# never read as passing.  A reply-content claim at 1.00 is grey for exactly that reason: it has
# no floor and cannot carry one at this N, so a green tick would re-imply the gate the noise
# measurement says does not exist.

PASS_GLYPH = "🟢"
WARN_GLYPH = "🟡"
FAIL_GLYPH = "🔴"
UNGATED_GLYPH = "⚪"

PASS_ABOVE = 0.90
WARN_ABOVE = 0.50

GLYPH_KEY = (
    f"_{PASS_GLYPH} >90% · {WARN_GLYPH} 50–90% · {FAIL_GLYPH} <50% · "
    f"{UNGATED_GLYPH} ungated (measured, no threshold accepted)_"
)

# Worst-first, so the one line a reader scans carries the problem rather than the average.
_SEVERITY = {FAIL_GLYPH: 3, WARN_GLYPH: 2, UNGATED_GLYPH: 1, PASS_GLYPH: 0}


def worst_glyph(glyphs: Sequence[str]) -> str:
    """The most serious state among ``glyphs`` — what the case's one line carries."""
    return max(glyphs, key=lambda g: _SEVERITY[g], default=PASS_GLYPH)


def rate_glyph(rate: float) -> str:
    """Where a pass rate falls on the scale."""
    if rate > PASS_ABOVE:
        return PASS_GLYPH
    return WARN_GLYPH if rate >= WARN_ABOVE else FAIL_GLYPH


def assertion_glyph(row: cohort.AssertionRow) -> str:
    """A claim's colour, or grey where no floor can be accepted at this N."""
    if row.reads_model_prose:
        return UNGATED_GLYPH
    return rate_glyph(row.pass_rate)


def variance_glyph(feature: cohort.VarianceFeature) -> str:
    """A feature's colour.  No ceiling has been ACCEPTED anywhere yet — every one this report
    prints is proposed — so the honest answer is grey until one is."""
    return UNGATED_GLYPH


# ── The three sections a case reports ────────────────────────────────────────
#
# A case reports three things, and they are three different KINDS of claim that must never be
# mixed: whether Penny was CORRECT (A), whether she was STABLE (B), and whether the run can be
# believed at all (C).  C is read FIRST even though it renders last.
#
# Everything here RENDERS; nothing computes.  The numbers, the proposed floors and ceilings, the
# standings and the divergences are the cohort's, so a reader comparing the document against the
# data compares one arithmetic to one rendering rather than two.

# Declarative labels.  A heading NAMES the thing; the explanation belongs in the body, or
# nowhere.  What stood here — "What every sample was given — the ask, the world, the prompts",
# "Which samples to read", "Harness — samples too broken to count" — were sentences pretending
# to be headings, and a reader scanning a hundred cases reads labels, not prose.
SECTION_A = "Assertions"
SECTION_B = "Variance"
SECTION_C = "Excluded samples"

_SUMMARY_LINE = (
    "{glyph} **`{case_id}`** — assertions {held}/{claims}{lowest} · "
    "variance {var_glyph} max H {entropy:.3f} `{feature}` · {counts}"
)
_LOWEST = " (lowest {glyph} {rate:.2f} `{label}`)"
# The harness accounting lives HERE rather than in a section of its own: on a healthy run it is
# one line, and a section that renders as a stub every time is one people learn to skip.  The
# counts must still ADD UP — three unexplained samples on the surface that says whether a run can
# be believed is how 288 infrastructure failures came to be booked as behavioural.
_COUNTS = "{pooled} pooled + {control} control + {excluded} excluded = {driven} driven"
_NO_VARIANCE = "nothing pooled"

_ASSERTION_HEAD = "|  | assertion | held | rate | proposed floor |\n|---|---|---|---|---|"
_VARIANCE_HEAD = (
    "|  | feature | distinct | modal | entropy | proposed ceiling |\n|---|---|---|---|---|---|"
)
_PHRASING_HEAD = "| feature | phrasing | distinct | only under this wording |\n|---|---|---|---|"
_COST_HEAD = "| tokens | observed | proposed ceiling |\n|---|---|---|"

_NO_ASSERTIONS = "_(no assertions)_"
_NOTHING_POOLED = "_(nothing pooled — see the harness section)_"
_NO_PHRASING_OUTLIERS = "_No phrasing produced a value the others did not._"
_COST_LEAD = "**Cost, per sample.**"

_FLOOR_NOTE = (
    "_Floors are PROPOSED, never locked — accepting one is the code owner's act. A claim read "
    "out of the machine, the registry or the store can carry one; a claim read out of prose the "
    "model wrote is REPORTED and not floored at this N. That split is measured rather than a "
    "matter of taste: across two runs of identical code — same commit, same model, same upstream "
    "— a reply-content rate moved by 3 samples of 18 while every structural one moved by at most "
    "1, and ±3 of 18 is ±17 points, so a floor tight enough to catch a regression would flap and "
    "one loose enough not to would catch nothing._"
)
_CEILING_NOTE = (
    "_Ceilings are PROPOSED, not locked, and are one-sided — only a rise is a regression. "
    "Each is recorded as `(feature, model, N={n}, value)` and a comparison across either "
    "qualifier is REFUSED: normalised entropy is biased upward at small N (the same behaviour "
    "reads 0.527 at N=32 and 0.605 at N=15), and two models differ ~3x on the same feature "
    "(routine shape 0.53 against 0.18), so a shared ceiling would measure neither._"
)
_PHRASING_LEAD = (
    "_Per-phrasing rows below are DIAGNOSTIC and never locked — at 3 samples each there is no "
    "reliable per-phrasing entropy, so what is reported is the honest weaker signal: a wording "
    "that produced a value no other wording did. Phrasings are a coverage mechanism, and the "
    "pooled number above hides exactly what they are for._"
)
_COST_NOTE = (
    "_Per SAMPLE, never per run — a total is not comparable across cohort sizes. Input is OURS "
    "(prompt and context design), so a rise is what a prompt edit regresses; output is the "
    "MODEL's, so a rise on a fixed prompt is a model or config change. Both ceilings are "
    "one-sided, per model, and PROPOSED — and unlike the variance margin this band is a round "
    "number rather than a measured one._"
)

_DOMINANT = "Dominant failure class: **{reason}** ({count} of {total})."


def fold(summary: str, body: str) -> str:
    """One collapsible block.  The default view is every summary line and no body."""
    return f"<details><summary>{summary}</summary>\n\n{body}\n\n</details>"


@dataclass(frozen=True)
class CaseSections:
    """One case's three sections, rendered from what the cohort computed.

    ``model`` rides along because a proposed ceiling is meaningless without it — two models
    differ several-fold on the same feature, so a ceiling that did not name which one it was
    measured on could be read against the other and answer a question nobody asked."""

    case_id: str
    model: str = ""
    assertions: Sequence[cohort.AssertionRow] = ()
    variance: cohort.CohortVariance = field(default_factory=cohort.CohortVariance)
    cost: cohort.SampleCost | None = None

    def render(self) -> str:
        """The summary method: one line that carries the case's worst state, then the three
        sections behind folds."""
        blocks = [
            self.summary_line(),
            fold(f"{self._assertions_glyph()} {SECTION_A}", self._assertions()),
            fold(f"{self._variance_glyph()} {SECTION_B}", self._variance()),
        ]
        # Only when something was actually lost.  A clean run says so on the summary line and
        # spends no section on it; a degraded one gets the names and the dominant class.
        if self.variance.excluded:
            blocks.append(fold(f"{FAIL_GLYPH} {SECTION_C}", self._excluded()))
        return "\n\n".join(blocks)

    # ── The one line a reader sees by default ────────────────────────────
    def summary_line(self) -> str:
        """Both scores and the whole sample accounting, behind the worst glyph in the case."""
        held = sum(1 for row in self.assertions if row.at_full)
        top = self._top_feature()
        return _SUMMARY_LINE.format(
            glyph=self.glyph(),
            case_id=self.case_id,
            held=held,
            claims=len(self.assertions),
            lowest=self._lowest(),
            var_glyph=self._variance_glyph(),
            entropy=top.entropy if top else 0.0,
            feature=top.name if top else _NO_VARIANCE,
            counts=_COUNTS.format(
                pooled=self.variance.pooled,
                control=self.variance.control,
                excluded=len(self.variance.excluded),
                driven=self.variance.driven,
            ),
        )

    def glyph(self) -> str:
        """The case's worst state — what someone paging ~100 one-line entries reads."""
        lost = [FAIL_GLYPH] if self.variance.excluded else []
        return worst_glyph([self._assertions_glyph(), self._variance_glyph(), *lost])

    def _assertions_glyph(self) -> str:
        return worst_glyph([assertion_glyph(row) for row in self.assertions])

    def _variance_glyph(self) -> str:
        return worst_glyph([variance_glyph(f) for f in self.variance.features])

    def _top_feature(self) -> cohort.VarianceFeature | None:
        return max(self.variance.features, key=lambda f: f.entropy, default=None)

    def _lowest(self) -> str:
        """The claim that held least often — the one a reader should look at first."""
        scored = [row for row in self.assertions if row.total]
        worst = min(scored, key=lambda row: row.pass_rate, default=None)
        if worst is None or worst.at_full:
            return ""
        return _LOWEST.format(glyph=assertion_glyph(worst), rate=worst.pass_rate, label=worst.label)

    # ── A ────────────────────────────────────────────────────────────────
    def _assertions(self) -> str:
        if not self.assertions:
            return _NO_ASSERTIONS
        rows = "\n".join(_assertion_row(row) for row in self.assertions)
        return f"{_ASSERTION_HEAD}\n{rows}\n\n{_FLOOR_NOTE}"

    # ── B ────────────────────────────────────────────────────────────────
    def _variance(self) -> str:
        if not self.variance.features:
            return _NOTHING_POOLED
        rows = "\n".join(_variance_row(f, self.model) for f in self.variance.features)
        parts = [
            f"{_VARIANCE_HEAD}\n{rows}",
            _CEILING_NOTE.format(n=self.variance.pooled),
            self._phrasing_rows(),
        ]
        if self.variance.text is not None:
            parts.append(_text_spread_line(self.variance.text))
        if self.cost is not None:
            parts.append(_cost_block(self.cost, self.model))
        return "\n\n".join(parts)

    def _phrasing_rows(self) -> str:
        """Only the FLAGGED rows render: an unflagged phrasing agreeing with its neighbours is
        the ordinary case and says nothing a reader needs."""
        flagged = [
            (feature.name, row)
            for feature in self.variance.features
            for row in feature.phrasings
            if row.flagged
        ]
        if not flagged:
            return _NO_PHRASING_OUTLIERS
        rows = "\n".join(
            f"| `{name}` | {row.arm} | {row.distinct}/{row.n} | {_value_list(row.only_here)} |"
            for name, row in flagged
        )
        return f"{_PHRASING_LEAD}\n\n{_PHRASING_HEAD}\n{rows}"

    # ── C ────────────────────────────────────────────────────────────────
    def _excluded(self) -> str:
        """The samples too broken to count, by name, under the class that cost the most.

        Rendered ONLY when there are some — the accounting itself rides on the summary line,
        where it is one clause on a healthy run instead of a section that always says nothing."""
        named = "\n".join(
            f"- `{sample.name}` — {sample.reason}" for sample in self.variance.excluded
        )
        return f"{self._dominant()}\n\n{named}"

    def _dominant(self) -> str:
        """What cost this case its samples, named — so a reader meets the class before the list.

        A per-CASE view of the same event the run-health block counts per RUN; it is computed
        from this cohort's own exclusions rather than from that tally, which cannot say which
        case a fault landed in."""
        dominant = self.variance.dominant_exclusion
        if dominant is None:
            return ""
        reason, count = dominant
        return _DOMINANT.format(reason=reason, count=count, total=len(self.variance.excluded))


def _assertion_row(row: cohort.AssertionRow) -> str:
    floor = cohort.proposed_floor(row)
    proposal = f"`{floor.value:.2f}`" if floor.lockable else f"— {floor.note}"
    return (
        f"| {assertion_glyph(row)} | {row.label} | {row.passed}/{row.total} | "
        f"{row.pass_rate:.2f} | {proposal} |"
    )


def _variance_row(feature: cohort.VarianceFeature, model: str) -> str:
    ceiling = cohort.proposed_ceiling(feature, model)
    return (
        f"| {variance_glyph(feature)} | `{feature.name}` | {feature.distinct} | "
        f"{feature.modal}/{feature.n} ({feature.modal_share:.2f}) | {feature.entropy:.3f} | "
        f"`{ceiling.value:.2f}` @ {ceiling.model} N={ceiling.n} |"
    )


def _cost_block(cost: cohort.SampleCost, model: str) -> str:
    rows = "\n".join(
        f"| {label} | {observed:,.0f} | "
        f"`{observed * (1 + cohort.COST_CEILING_MARGIN):,.0f}` @ {model} |"
        for label, observed in (
            ("input tokens (ours — prompt and context)", cost.input_tokens),
            ("output tokens (the model's)", cost.output_tokens),
        )
    )
    tail = (
        f"Also per sample: {cost.calls:,.1f} calls · {cost.seconds:,.0f}s · "
        f"{cost.reasoning_tokens:,.0f} reasoning tokens ({cost.reasoning_share:.0%} of output)."
    )
    return f"{_COST_LEAD}\n\n{_COST_HEAD}\n{rows}\n\n{tail}\n\n{_COST_NOTE}"


def _text_spread_line(text: cohort.TextSpread) -> str:
    return (
        f"Reply text over {text.pairs} pairs — cosine mean {text.cosine_mean:.3f} "
        f"min {text.cosine_min:.3f} · containment mean {text.containment_mean:.3f}"
    )


def _value_list(values: Sequence[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "—"


# ── The case document: what every sample shares, stated ONCE ─────────────────
#
# Sharing is DECLARED, not discovered.  What stood here computed which LINES happened to be
# common across a case's prompts and rendered each sample as a diff against them — machinery
# that existed because samples could not be assumed to share anything.  Under the pooled cohort
# they do: one world, one seed set, K wordings of one ask.
#
# So prompts are grouped by EXACT TEXT and each distinct one renders once, verbatim, naming the
# samples that used it.  Measured on the reference port's own 18-sample run, four of the five
# contexts are byte-identical across every sample (state-classifier, skill-framer, skill-namer
# and browse-extract: 125,586 characters collapsing to 6,977), so declared sharing costs nothing
# where sharing is real.  The fifth is the finding: `chat` has 18 distinct texts because the
# self-state header feeds each sample its OWN minted collection and routine names back into its
# prompt — which is the cohort's `container name` feature, showing up in the prompt.  A line
# diff hid exactly that behind a marker, and it is the kind of thing a reader must see.
#
# The rule this keeps that the diff could not: every prompt renders VERBATIM.  A reader opening
# a sample's prompt reads what the model read, never a reconstruction assembled from two places.

PROMPT_VARIANTS_LABEL = "System prompts"
PHRASINGS_LABEL = "Phrasings ({count})"
WORLD_LABEL = "Seeded pages"
OUTLIERS_LABEL = "Outliers ({count})"
TEST_INPUTS_LABEL = "Test inputs"
REPRESENTATIVE_LABEL = "Representative sample"

_ALL_SAMPLES = "every sample"
_PROMPT_SUMMARY = "{label} — {chars:,} chars · {used}"
_PHRASING_HEAD_ROW = "| # | ask |\n|---|---|"
_OUTLIER_HEAD = "| feature | this sample | the representative |\n|---|---|---|"
_OUTLIER_LEAD = (
    "_An outlier is outlying on a SPECIFIC feature, so what renders here is the divergence and "
    "not the sample: the feature, this sample's value, and the representative's. The whole "
    "transcript is in the run artifact for a reader who wants it after seeing what changed._"
)
_NO_OUTLIERS = "_Every pooled sample carried the representative's shape._"
_SAMPLE_MAP_LEAD = "_The index into the section above — `modal` is the one to read._"


@dataclass(frozen=True)
class PromptVariant:
    """One DISTINCT system prompt, and the samples that were given it.

    Distinct by exact text: two samples share a prompt when they were handed the same bytes,
    which is a fact about the run rather than a judgement about how similar it is."""

    context: str
    text: str
    samples: list[str]
    total: int = 0

    def render(self) -> str:
        used = _ALL_SAMPLES if self.shared_by_all else ", ".join(self.samples)
        summary = _PROMPT_SUMMARY.format(label=self.context, chars=len(self.text), used=used)
        return fold(summary, self.text)

    @property
    def shared_by_all(self) -> bool:
        """Whether every sample in the cohort was handed this exact text."""
        return self.total > 0 and len(self.samples) == self.total


def prompt_variants(prompts: Sequence[tuple[str, SystemPrompt]], total: int) -> list[PromptVariant]:
    """Group ``(sample, prompt)`` pairs into one variant per distinct ``(context, text)``.

    Order is first-seen so a re-render reads identically, and a sample naming the same context
    twice counts once — a context is an actor, and a sample is handed one prompt per actor."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for sample, prompt in prompts:
        seen = grouped.setdefault((prompt.context, prompt.text), [])
        if sample not in seen:
            seen.append(sample)
    return [
        PromptVariant(context=context, text=text, samples=samples, total=total)
        for (context, text), samples in grouped.items()
    ]


def render_prompt_variants(variants: Sequence[PromptVariant]) -> str:
    """Every distinct prompt, once — the contexts every sample shared first."""
    if not variants:
        return ""
    ordered = sorted(variants, key=lambda v: (not v.shared_by_all, v.context))
    blocks = "\n\n".join(variant.render() for variant in ordered)
    return fold(PROMPT_VARIANTS_LABEL, blocks)


def render_phrasings(phrasings: Sequence[tuple[str, str]]) -> str:
    """The K wordings of the one ask, as a table.

    Tabular rather than a run of prose blocks: the wordings are a COVERAGE mechanism and the
    reason they sit together is to be read AGAINST each other, which a column does and stacked
    paragraphs do not."""
    if not phrasings:
        return ""
    rows = "\n".join(f"| {label} | {_cell(text)} |" for label, text in phrasings)
    return fold(PHRASINGS_LABEL.format(count=len(phrasings)), f"{_PHRASING_HEAD_ROW}\n{rows}")


def _cell(text: str) -> str:
    """One table cell: pipes escaped and newlines flattened so a long ask stays in its row."""
    return text.replace("|", "\\|").replace("\n", " ")


def render_seeded_world(world: str) -> str:
    """The world every sample was answered against, stated once."""
    return fold(WORLD_LABEL, world) if world.strip() else ""


def render_outliers(rows: Sequence[tuple[int, cohort.SampleStanding]]) -> str:
    """What each outlying sample did differently — the divergence, never the transcript.

    This is what makes a variant cohort readable at all.  Rendering each outlier's whole run to
    communicate one changed feature is three orders of magnitude of the wrong thing, and it is
    what made a single case's report 787,681 characters."""
    outliers = [(number, s) for number, s in rows if s.standing == cohort.Standing.OUTLIER]
    if not outliers:
        return fold(OUTLIERS_LABEL.format(count=0), _NO_OUTLIERS)
    blocks = [_OUTLIER_LEAD]
    for number, standing in outliers:
        table = "\n".join(
            f"| `{d.feature}` | {_code(d.value)} | {_code(d.modal)} |" for d in standing.divergences
        )
        body = f"{_OUTLIER_HEAD}\n{table}" if table else "_(no feature diverged)_"
        blocks.append(f"**{SAMPLE_ROW} {number}** ({standing.phrasing})\n\n{body}")
    return fold(OUTLIERS_LABEL.format(count=len(outliers)), "\n\n".join(blocks))


def _code(value: str) -> str:
    return f"`{value}`" if value else "—"


# The seam between what a reader meets BEFORE the representative sample and what belongs after
# it.  The sample is what they were sent to read, so it sits directly under the scores rather
# than below the setup; the setup is reference material and reads last.
CASE_TAIL_MARKER = "<!-- case tail -->"


def render_case_tail(
    *,
    prompts: Sequence[PromptVariant] = (),
    phrasings: Sequence[tuple[str, str]] = (),
    world: str = "",
    outliers: Sequence[tuple[int, cohort.SampleStanding]] = (),
) -> str:
    """What closes a case: the inputs it was given, then what the outlying samples did.

    Every part is optional, so a case declaring none of them closes with nothing at all."""
    inputs = "\n\n".join(
        part
        for part in (
            render_phrasings(phrasings),
            render_seeded_world(world),
            render_prompt_variants(prompts),
        )
        if part
    )
    parts = [
        fold(TEST_INPUTS_LABEL, inputs) if inputs else "",
        render_outliers(outliers) if outliers else "",
    ]
    return "\n\n".join(part for part in parts if part)


# ── The whole sample ─────────────────────────────────────────────────────────
@dataclass
class SampleTranscript:
    """One sample rendered end-to-end: the banner, its step tables, and the run-close table.

    ``banner`` is the full verdict tail after ``sample N — `` (verdict · k/n (score) · cause ·
    fragile · duration · calls).  A sample renders only its OWN sequence — the turns it was given,
    the calls it made, what came back, what it replied.  Its system prompts are not here: every
    sample in a cohort is handed the same ones, so they are stated once on the case document
    rather than restated eighteen times.  **Every** sample block folds whole under its banner
    summary — the uniform-collapse default (#1753); the visible skeleton is the banner rows,
    everything below one click deep.  ``placeholder`` (F2) replaces the body for a sample that
    produced no completed turn (a harness timeout), so the report never silently omits it."""

    number: int
    banner: str
    steps: list[Step]
    run_close: RunClose | None = None
    placeholder: str | None = None

    def render(self) -> str:
        return fold_sample(self.number, self.banner, self._body())

    def _body(self) -> str:
        if self.placeholder is not None:
            return self.placeholder
        blocks = [step.render() for step in self.steps]
        if self.run_close is not None:
            blocks.append(self.run_close.render())
        return "\n\n".join(blocks)


def render_sample(sample: SampleTranscript) -> str:
    """Render one sample's whole block (the module entry point) — always folded (#1753)."""
    return sample.render()


# ── The folded-block primitives + their inverse (the assembler's re-normalization seam, #1753) ──
def fold_sample(number: int, banner: str, body: str) -> str:
    """Collapse a sample's body under its banner summary — the uniform, ONLY rendering (#1753):
    every sample block is one click deep, its ``<summary>`` the banner row, its full body always a
    click away (default collapsed never means content removed, #1759)."""
    return f"<details><summary>{SAMPLE_ROW} {number} — {banner}</summary>\n\n{body}\n\n</details>"


# ── What the posted COMMENT carries, against what the artifact keeps ─────────
#
# These two were one document until the cohort grew.  At 5 samples of a 4-case run that was
# right: one rendering, on disk and in the comment, and nothing to disagree about.  At 18
# samples across ~100 cases it stops being a document anyone reads — measured on the reference
# port, ONE case is 787,681 chars, of which 68% of every sample is its thinking traces.
#
# So the split is now explicit and it runs in ONE direction:
#
#   the ARTIFACT (`<case_id>.md`) is the complete record — every sample, every trace, verbatim.
#   the COMMENT is an INDEX into it — the score, the map, and the samples worth opening.
#
# Nothing is invented for the comment and nothing is summarised that the artifact does not hold
# in full, so "collapsed never means removed" still binds where completeness is claimed.  What
# changed is which document claims it.  Both transforms below therefore run at ASSEMBLY, on
# markup this module itself emits, and neither can reach the artifact.

THINKING_SUMMARY_HEAD = "thinking — {chars:,} chars, in full in the artifact"
THINKING_TASTE = 160

# The exact form `_thinking_details` emits, matched back at assembly.  A structural rewrite of
# markup this module owns — the same idiom `split_case_transcript` and `parse_sample_block`
# already use, and not the line-level DISCOVERY that #1997 deleted: nothing here compares two
# samples to work out what they have in common.
_THINKING_BLOCK = re.compile(
    r"<details><summary>(?P<summary>thinking[^<]*)</summary>(?P<body>.*?)</details>"
)


def summarise_thinking(body: str) -> str:
    """Replace each thinking trace in a rendered sample body with its head and its length.

    The single biggest lever there is: a trace is what you want once you have DECIDED to read a
    sample, and noise in the document whose job is to tell you which sample that is.  The full
    text stays in the artifact, one path away, so this shortens the index and never the record."""

    def shorten(match: re.Match[str]) -> str:
        trace = match.group("body")
        # A trace no longer than the taste is left exactly as it is: rewriting it would add a
        # length and a pointer to text already shorter than both, making the document bigger to
        # say it had been shortened.  Derived, not tuned — the saving has to be real.
        if len(trace) <= THINKING_TASTE:
            return match.group(0)
        summary = match.group("summary")
        label = THINKING_SUMMARY_HEAD.format(chars=len(trace))
        context = summary[len("thinking") :]
        return f"<details><summary>{label}{context}</summary>{trace[:THINKING_TASTE]}…</details>"

    return _THINKING_BLOCK.sub(shorten, body)


OTHER_SAMPLES_LINE = (
    "_{count} other samples agreed with the representative — full transcripts in the run artifact._"
)

# The prompt fold this module emits, matched back at assembly.  Same idiom as the thinking
# rewrite: a structural transform on markup report.py owns, never a comparison between samples.
_PROMPT_FOLD = re.compile(
    r"<details><summary>(?P<label>[^<\n]*? \u2014 [\d,]+ chars \u00b7 (?P<used>[^<\n]*?))</summary>"
    r"\n\n(?P<body>.*?)\n\n</details>",
    re.S,
)
PROMPT_IN_ARTIFACT = "_This wording's prompt is in the run artifact._"


def elide_unused_prompts(text: str, keep: Sequence[str]) -> str:
    """Keep the prompts every sample shared and the ones the representative was given; point the
    rest at the artifact.

    The comment is an index, and the index needs the context the sample it carries was actually
    run with.  Measured on the reference port this is the difference between 141,342 characters
    of prompt and about 15,000: `chat` has one distinct text PER SAMPLE, because the self-state
    header feeds each its own minted names back, so carrying all of them means carrying the
    cohort eighteen times over to show a text that differs in three lines."""

    def prune(match: re.Match[str]) -> str:
        used = match.group("used")
        wanted = used == _ALL_SAMPLES or any(label in used.split(", ") for label in keep)
        if wanted:
            return match.group(0)
        return (
            f"<details><summary>{match.group('label')}</summary>"
            f"\n\n{PROMPT_IN_ARTIFACT}\n\n</details>"
        )

    return _PROMPT_FOLD.sub(prune, text)


def other_samples_line(count: int) -> str:
    """The samples the comment does not carry, as ONE line.

    Seventeen collapsed stubs each saying "not expanded here" is seventeen folds that say
    nothing, and at a hundred cases it is seventeen hundred. What a reader needs is the count and
    where the transcripts are; which of them diverged is the outlier section's job."""
    return OTHER_SAMPLES_LINE.format(count=count)


# The seam a sample block opens on — the folded form and the legacy bare heading. Public because
# it is also the ONLY place a run comment may be cut when it exceeds GitHub's comment cap (#1808):
# one definition of "a sample starts here", shared by the re-normalizer and the splitter.
SAMPLE_BLOCK_START = rf"(?:<details><summary>{SAMPLE_ROW} |#### {SAMPLE_ROW} )\d+ — "


# ── Internal seams: a sample too big to post as one fold (#1917) ─────────────
#
# The splitter may cut only BETWEEN sample folds, so a single fold larger than a comment
# is unpostable however the document is packed — and it happens: one degenerate reroll
# sample rendered 71,350 chars against the 65,536 cap.  Compression cannot reach it.  That
# fold was 25 DISTINCT thinking traces; there was no shared block to hoist and no
# duplicate to fold, which is what #1763's hoisting and any dedup are for.
#
# So an oversized sample renders as SEVERAL folds instead of one, each a complete
# ``<details>`` opening on the same ``SAMPLE_BLOCK_START`` seam — which gives the splitter
# legal cut points inside what used to be one atom, with no change to the splitter at all.
#
# THE SEAM IS THE STEP.  A step is already the body's unit of composition and already reads
# alone: a user turn and everything it produced, header row and all.  Cutting anywhere
# inside one would split a markdown table; cutting between two loses nothing, because
# nothing spans them.  (A cycle would be the other candidate, and it is not available here:
# this module renders transcripts and knows steps, never what a collector cycle is.)
#
# Both standing rules survive verbatim.  NOTHING IS TRUNCATED — every block appears exactly
# once, in order, and concatenating the parts' bodies reproduces the single fold's body
# byte for byte (asserted by ``test_report.py``).  And the guard still REFUSES what cannot
# be split losslessly: when one STEP alone exceeds the cap there is no seam inside it, and
# ``comment_split.unsplittable_reason`` says so rather than posting something cut.
_BODY_BLOCK_START = rf"(?:\| step |\| {RUN_CLOSE_LABEL} )"
_BODY_BOUNDARY = re.compile(rf"\n\n(?={_BODY_BLOCK_START})")

# What a part's banner gains so a reader knows the sample continues.  It rides INSIDE the
# banner, after the verdict tail, so ``SAMPLE_BLOCK_START`` matches it unchanged and the
# splitter needs to know nothing about parts.
SAMPLE_PART_SUFFIX = " · part {number} of {total}"


def body_blocks(body: str) -> list[str]:
    """A rendered sample body cut into its top-level blocks — the system-prompt folds, the
    step tables and the run-close table, in order.

    ``"\n\n".join(body_blocks(body)) == body`` for every body this module renders: the
    blocks are joined by exactly that separator, and a block never contains a blank line
    followed by another block's opening (a step table's rows are joined by single
    newlines, so no row can look like a step header)."""
    return _BODY_BOUNDARY.split(body) if body else []


def fold_sample_parts(number: int, banner: str, body: str, budget: int) -> str:
    """One sample rendered as folds that each fit ``budget`` — one fold when it already
    does (byte-identical to ``fold_sample``, which is the overwhelmingly common case), and
    otherwise its blocks packed greedily into as many folds as it takes.

    A single block bigger than ``budget`` still gets its own over-budget fold rather than
    being cut: the seam rule outranks the budget here exactly as it does in the splitter,
    and the refusal belongs to the guard that knows the hard cap."""
    whole = fold_sample(number, banner, body)
    blocks = body_blocks(body)
    if len(whole) <= budget or len(blocks) < 2:
        return whole
    groups = _packed_blocks(blocks, _block_budget(number, banner, budget))
    return BLOCK_SEPARATOR.join(
        fold_sample(
            number,
            banner + SAMPLE_PART_SUFFIX.format(number=index, total=len(groups)),
            BLOCK_SEPARATOR.join(group),
        )
        for index, group in enumerate(groups, start=1)
    )


# What joins a sample's top-level blocks, and what joins the folds an oversized sample is
# rendered as — one separator, because the second is the seam the splitter cuts on and the
# first is what ``body_blocks`` must invert.
BLOCK_SEPARATOR = "\n\n"


def _block_budget(number: int, banner: str, budget: int) -> int:
    """The budget left for a part's BLOCKS once its own fold markup is paid for — measured
    off an empty fold carrying the longest part banner this sample could get, so a part can
    never come out over budget because of its own wrapper."""
    widest = banner + SAMPLE_PART_SUFFIX.format(number=99, total=99)
    return budget - len(fold_sample(number, widest, ""))


def _packed_blocks(blocks: list[str], budget: int) -> list[list[str]]:
    """``blocks`` packed greedily into runs of at most ``budget`` characters, order kept and
    nothing dropped.  A block over budget on its own becomes a run of one."""
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for block in blocks:
        cost = len(block) + (len(BLOCK_SEPARATOR) if current else 0)
        if current and size + cost > budget:
            groups.append(current)
            current, size = [], 0
            cost = len(block)
        current.append(block)
        size += cost
    if current:
        groups.append(current)
    return groups


_SAMPLE_BOUNDARY = re.compile(rf"\n\n(?={SAMPLE_BLOCK_START})")
# Whether a split fragment IS a sample block — what tells a case-level preamble from the
# first sample when a transcript carries both.
_SAMPLE_BLOCK_OPENS = re.compile(SAMPLE_BLOCK_START)
_FOLDED_SAMPLE = re.compile(
    rf"\A<details><summary>{SAMPLE_ROW} (\d+) — (.*?)</summary>\n\n(.*)\n\n</details>\Z", re.DOTALL
)
_HEADING_SAMPLE = re.compile(rf"\A#### {SAMPLE_ROW} (\d+) — (.*?)(?:\n\n(.*))?\Z", re.DOTALL)


def split_sample_blocks(transcript: str) -> list[str]:
    """Split a case's rendered transcript into its per-sample blocks, in order — each either a
    folded ``<details>`` block or a bare ``#### `` heading (the assembler consumes both: a
    re-assembled prior run may carry the old unfolded failures)."""
    text = transcript.strip()
    return _SAMPLE_BOUNDARY.split(text) if text else []


def split_case_transcript(transcript: str) -> tuple[str, list[str]]:
    """Split a case's rendered transcript into ``(preamble, sample blocks)``.

    The preamble is everything above the FIRST sample fold — a case-level section written
    by the runner (the three-section report, #1995) rather than by any one sample. It is
    carried through verbatim rather than parsed: a case may state whatever it needs to
    above its samples, and nothing here has to know what that is. Empty when the transcript
    opens straight onto sample 1, which is what it did before case sections existed."""
    text = transcript.strip()
    if not text:
        return "", []
    blocks = _SAMPLE_BOUNDARY.split(text)
    if _SAMPLE_BLOCK_OPENS.match(blocks[0]):
        return "", blocks
    return blocks[0].strip(), blocks[1:]


def parse_sample_block(block: str) -> tuple[int, str, str]:
    """Recover ``(number, banner, body)`` from one rendered sample block — the folded form
    (``<details><summary>sample …``) or the bare heading (``#### sample …``). Raises on an
    unrecognized shape (fail loud rather than mangle a real report)."""
    stripped = block.strip()
    for pattern in (_FOLDED_SAMPLE, _HEADING_SAMPLE):
        match = pattern.match(stripped)
        if match:
            return int(match.group(1)), match.group(2), match.group(3) or ""
    raise ValueError(f"unrecognized sample block: {stripped[:60]!r}")


# ── The banner (per-sample stats line after ``sample N — ``) ─────────────────
def render_banner(
    *,
    passed: bool,
    score: float,
    passed_checks: int,
    total_checks: int,
    cause: str | None = None,
    fragile: bool = False,
    duration_s: int,
    calls: int,
    checks_evaluated: bool = True,
) -> str:
    """The per-sample banner tail (#1725 Final-additions #1): ``verdict · k/n (score) · cause ·
    fragile · duration · calls``. A timeout sample (``checks_evaluated=False``) omits ``k/n`` —
    its scorer never ran; a clean pass carries no cause; a fragile pass carries ``fragile``."""
    parts = ["✅ pass" if passed else "❌ fail"]
    if checks_evaluated:
        parts.append(f"{passed_checks}/{total_checks} ({score:.2f})")
    if fragile:
        parts.append("fragile")
    if cause:
        parts.append(cause)
    parts += [f"{duration_s}s", f"{calls} calls"]
    return " · ".join(parts)


# ── The event stream the extraction hands the builder ────────────────────────
class EventKind(StrEnum):
    """One transcript event. ``USER`` opens a step; the rest render as ``actual`` rows.

    ``CALL`` / ``REPLY`` / ``MICRO_OUT`` are model ACTIONS — each gets a 💭 row directly above
    it. ``NUDGE`` is a recovery injection (``⚠ recovery event``). ``MICRO_IN`` is the instruction
    + page content into the extraction sub-model; ``MICRO_OUT`` is its extracted value."""

    USER = "user"
    CALL = "call"
    RESULT = "result"
    REPLY = "reply"
    NUDGE = "nudge"
    MICRO_IN = "micro_in"
    MICRO_OUT = "micro_out"


_ACTOR_GLYPH = {
    EventKind.CALL: ACTOR_CALL,
    EventKind.RESULT: ACTOR_RESULT,
    EventKind.REPLY: ACTOR_REPLY,
    EventKind.NUDGE: ACTOR_USER,
    EventKind.MICRO_IN: ACTOR_MICRO,
    EventKind.MICRO_OUT: ACTOR_MICRO,
}
_ACTIONS = frozenset({EventKind.CALL, EventKind.REPLY, EventKind.MICRO_OUT})

# The micro-context direction arrow the renderer puts after the actor label (#1759), so the role
# vocabulary is single-sourced here and the ``MICRO_IN``/``MICRO_OUT`` event body carries only its
# content.
_MICRO_ARROW = {EventKind.MICRO_IN: MICRO_IN_ARROW, EventKind.MICRO_OUT: MICRO_OUT_ARROW}


@dataclass
class Event:
    """One extracted transcript event. ``body`` is its rendered content (verbatim; escaped at
    render — for a ``MICRO_IN``/``MICRO_OUT`` event it is the content ONLY, the ``<context> ←
    user turn:`` / ``<context> →`` label is the renderer's, #1759). ``thinking`` is the model
    reasoning that produced an ACTION (``None`` otherwise). ``context`` is the calling
    micro-context's ledger identity (#1773) — the actor label on a 🧩 row and on its 💭 row, so
    the three sub-models read apart and match their own system-prompt rows; ``None`` renders the
    generic ``micro-context``."""

    kind: EventKind
    body: str
    thinking: str | None = None
    context: str | None = None

    def glyph(self) -> str:
        return _ACTOR_GLYPH[self.kind]

    def actor_label(self) -> str:
        """The 🧩 actor's name — its calling context, or the generic label when it names none."""
        return self.context or MICRO_CONTEXT_LABEL

    def actual_body(self) -> str:
        """The ``actual`` row body: the glyph, the micro-context actor label + direction arrow
        (#1759/#1773) for a 🧩 event, then the content — so ``🧩 state-classifier ← user turn:
        <turn>`` reads both its actor and its role explicitly."""
        arrow = _MICRO_ARROW.get(self.kind)
        if arrow is None:
            return f"{self.glyph()} {self.body}"
        return f"{self.glyph()} {self.actor_label()} {arrow} {self.body}"


@dataclass
class CheckView:
    """A scored expectation, resolved against the transcript + baseline (the pure view the
    builder consumes). ``anchor_index`` is the event this check binds to (``None`` → run-close).
    ``regressed``/``fixed`` are the baseline flips; ``ignored`` is the n/a third state."""

    check_id: str
    label: str
    kind: str | None
    scored: bool
    ignored: bool
    ok: bool
    rationale: str | None = None
    cause: str | None = None
    anchor_index: int | None = None
    regressed: bool = False
    fixed: bool = False
    baseline_event: str | None = None  # diff mode: the prior run's anchor event
    baseline_ok: bool = True  # the prior run's verdict for that event

    def baseline_row(self) -> Row | None:
        """The diff-mode ``baseline`` row (the prior run's anchor event + its prior verdict), or
        ``None`` off-diff."""
        if self.baseline_event is None:
            return None
        mark = PASS_MARK if self.baseline_ok else FAIL_MARK
        return Row(ROW_BASELINE, self.baseline_event, [Verdict(mark, self.check_id, prior=True)])

    def expected_body(self) -> str:
        """The ``expected`` row body: ``Cn [class]marker label`` (marker omitted for n/a)."""
        marker = "" if self.ignored else (GATING_MARKER if self.scored else ADVISORY_MARKER)
        klass = f" [{self.kind}]" if self.kind else " "
        return f"{self.check_id}{klass}{marker} {self.label}".replace("  ", " ").strip()

    def verdict(self, *, on_anchor: bool) -> Verdict:
        """This check's rendered verdict. ``on_anchor`` (the actual row) carries the ``check_id``;
        an expected-row verdict for a passed no-evidence contract carries it too, an n/a shows
        ``➖ n/a`` with its reason and no id."""
        if self.ignored:
            return Verdict(NA_MARK, rationale=self.rationale)
        mark = self._mark()
        rationale = self.rationale if not self.ok else None
        cause = self.cause if not self.ok else None
        return Verdict(mark, check_id=self.check_id, rationale=rationale, cause=cause)

    def _mark(self) -> str:
        if self.regressed:
            return REGRESSED_MARK
        if self.fixed:
            return FIXED_MARK
        return PASS_MARK if self.ok else FAIL_MARK


def _step_verdict(checks: list[CheckView]) -> str:
    """The step header's roll-up glyph over its placed checks: ✅→❌ on any flip, ❌ on any
    failure, ✅ when at least one check passed, else blank (a step with no scored checks)."""
    scored = [check for check in checks if not check.ignored]
    if any(check.regressed for check in scored):
        return "✅→❌"
    if any(not check.ok for check in scored):
        return FAIL_MARK
    return PASS_MARK if scored else ""


def _event_rows(event: Event, verdicts: list[Verdict]) -> list[Row]:
    """The rows for one event: its 💭 (above an ACTION), then the ``actual`` row with its verdicts.
    A nudge's verdict is the fixed ``⚠ recovery event`` mark (the caller passes it)."""
    rows: list[Row] = []
    if event.kind in _ACTIONS and event.thinking is not None:
        if event.kind == EventKind.MICRO_OUT:
            rows.append(micro_thinking_row(event.thinking, event.context))
        else:
            rows.append(thinking_row(event.thinking))
    rows.append(Row(ROW_ACTUAL, event.actual_body(), verdicts))
    return rows


def build_sample(
    *,
    number: int,
    banner: str,
    events: list[Event],
    checks: list[CheckView],
    run_close_score: str,
    placeholder: str | None = None,
) -> SampleTranscript:
    """Assemble a sample from its extracted events + resolved checks (the pure builder).

    Steps segment on ``USER`` events. A check anchored to an event renders its ``expected`` row
    atop that event's step and its verdict on that event's ``actual`` row; a check with no anchor
    event falls to the run-close table. A nudge event renders ``⚠ recovery event``. Every sample
    folds whole at render (#1753) — the builder no longer decides fold-or-not."""
    if placeholder is not None:
        return SampleTranscript(number, banner, [], placeholder=placeholder)
    by_event: dict[int, list[CheckView]] = {}
    run_close_checks: list[CheckView] = []
    for check in checks:
        if check.anchor_index is None:
            run_close_checks.append(check)
        else:
            by_event.setdefault(check.anchor_index, []).append(check)
    steps = _build_steps(events, by_event)
    run_close = _build_run_close(run_close_checks, run_close_score) if run_close_checks else None
    return SampleTranscript(number, banner, steps, run_close=run_close)


def _build_steps(events: list[Event], by_event: dict[int, list[CheckView]]) -> list[Step]:
    """Segment the event stream into steps (a ``USER`` event opens each), attaching every event's
    verdicts to its row and every step's placed checks' ``expected`` rows atop it."""
    steps: list[Step] = []
    current: Step | None = None
    step_checks: list[CheckView] = []
    for index, event in enumerate(events):
        if event.kind == EventKind.USER:
            current = Step(len(steps) + 1, event.body, "", [])
            step_checks = []
            steps.append(current)
            continue
        if current is None:
            current = Step(1, "", "", [])
            step_checks = []
            steps.append(current)
        placed = by_event.get(index, [])
        step_checks.extend(placed)
        _insert_expected(current, placed)
        verdicts = _event_verdicts(event, placed)
        current.rows.extend(_event_rows(event, verdicts))
        current.verdict = _step_verdict(step_checks)
    return steps


def _event_verdicts(event: Event, placed: list[CheckView]) -> list[Verdict]:
    """The verdicts on one event's actual row — ``⚠ recovery event`` for a nudge, else each placed
    check's anchor verdict."""
    if event.kind == EventKind.NUDGE:
        return [Verdict(RECOVERY_MARK)]
    return [check.verdict(on_anchor=True) for check in placed]


def _insert_expected(step: Step, placed: list[CheckView]) -> None:
    """Add each placed check's ``expected`` row (and its diff-mode ``baseline`` row) at the TOP of
    its step, before the events — the step announces what it will be judged on."""
    top: list[Row] = []
    for check in placed:
        top.append(Row(ROW_EXPECTED, check.expected_body()))
        baseline = check.baseline_row()
        if baseline is not None:
            top.append(baseline)
    header = [row for row in step.rows if row.label in (ROW_EXPECTED, ROW_BASELINE)]
    rest = [row for row in step.rows if row.label not in (ROW_EXPECTED, ROW_BASELINE)]
    step.rows[:] = [*header, *top, *rest]


def _build_run_close(checks: list[CheckView], score: str) -> RunClose:
    """The run-close table — one ``expected`` row per whole-run contract (plus its diff-mode
    ``baseline`` row), its verdict in the score cell (these have no evidence row of their own)."""
    rows: list[Row] = []
    for check in checks:
        rows.append(Row(ROW_EXPECTED, check.expected_body(), [check.verdict(on_anchor=False)]))
        baseline = check.baseline_row()
        if baseline is not None:
            rows.append(baseline)
    return RunClose(score, rows)
