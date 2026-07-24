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
from dataclasses import dataclass, field
from enum import StrEnum

# ── Actor glyphs (the transcript-event vocabulary) ───────────────────────────
ACTOR_USER = "👤"
ACTOR_CALL = "🔧"
ACTOR_RESULT = "📥"
ACTOR_REPLY = "🤖"
ACTOR_MICRO = "🧩"

# ── Micro-context role labels (#1759) — the input row names the scoped USER turn explicitly (the
# user mistook it for the system prompt); the output row keeps the bare extracted-value arrow. ──
MICRO_IN_PREFIX = "micro-context ← user turn:"
MICRO_OUT_PREFIX = "micro-context →"

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


def micro_thinking_row(thinking: str) -> Row:
    """A 💭 row for a micro-context call — labelled ``thinking (micro-context)``."""
    if not thinking.strip():
        return Row(ROW_THINKING, EMPTY_THINKING, escape=False)
    body = _thinking_details(thinking, summary="thinking (micro-context)")
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


# ── The whole sample ─────────────────────────────────────────────────────────
@dataclass
class SampleTranscript:
    """One sample rendered end-to-end: the banner, its system-prompt rows, step tables, and the
    run-close table.

    ``banner`` is the full verdict tail after ``sample N — `` (verdict · k/n (score) · cause ·
    fragile · duration · calls). ``system_prompts`` (#1759) are the distinct per-context system
    prompts, rendered directly under the banner. **Every** sample block folds whole under its banner
    summary — the uniform-collapse default (#1753), superseding the old density-follows-failure
    split; the visible skeleton is the banner rows, everything below one click deep. ``placeholder``
    (F2) replaces the body for a sample that produced no completed turn (a harness timeout), so the
    report never silently omits it."""

    number: int
    banner: str
    steps: list[Step]
    run_close: RunClose | None = None
    placeholder: str | None = None
    system_prompts: list[SystemPrompt] = field(default_factory=list)

    def render(self) -> str:
        return fold_sample(self.number, self.banner, self._body())

    def _body(self) -> str:
        if self.placeholder is not None:
            return self.placeholder
        blocks = [prompt.render() for prompt in self.system_prompts]
        blocks += [step.render() for step in self.steps]
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


_BLOCK_START = rf"(?:<details><summary>{SAMPLE_ROW} |#### {SAMPLE_ROW} )\d+ — "
_SAMPLE_BOUNDARY = re.compile(rf"\n\n(?={_BLOCK_START})")
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

# The micro-context role prefix the renderer prepends between glyph and body (#1759), so the label
# is single-sourced here and the ``MICRO_IN``/``MICRO_OUT`` event body carries only its content.
_MICRO_PREFIX = {EventKind.MICRO_IN: MICRO_IN_PREFIX, EventKind.MICRO_OUT: MICRO_OUT_PREFIX}


@dataclass
class Event:
    """One extracted transcript event. ``body`` is its rendered content (verbatim; escaped at
    render — for a ``MICRO_IN``/``MICRO_OUT`` event it is the content ONLY, the ``micro-context ←
    user turn:`` / ``micro-context →`` label is the renderer's, #1759). ``thinking`` is the model
    reasoning that produced an ACTION (``None`` otherwise)."""

    kind: EventKind
    body: str
    thinking: str | None = None

    def glyph(self) -> str:
        return _ACTOR_GLYPH[self.kind]

    def actual_body(self) -> str:
        """The ``actual`` row body: the glyph, the micro-context role label (#1759) for a 🧩 event,
        then the content — so ``🧩 micro-context ← user turn: <turn>`` reads its role explicitly."""
        prefix = _MICRO_PREFIX.get(self.kind)
        return f"{self.glyph()} {prefix} {self.body}" if prefix else f"{self.glyph()} {self.body}"


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
        maker = micro_thinking_row if event.kind == EventKind.MICRO_OUT else thinking_row
        rows.append(maker(event.thinking))
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
    system_prompts: list[SystemPrompt] | None = None,
) -> SampleTranscript:
    """Assemble a sample from its extracted events + resolved checks (the pure builder).

    Steps segment on ``USER`` events. A check anchored to an event renders its ``expected`` row
    atop that event's step and its verdict on that event's ``actual`` row; a check with no anchor
    event falls to the run-close table. A nudge event renders ``⚠ recovery event``.
    ``system_prompts`` (#1759) render as collapsed rows directly under the banner. Every sample
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
    return SampleTranscript(
        number, banner, steps, run_close=run_close, system_prompts=system_prompts or []
    )


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
