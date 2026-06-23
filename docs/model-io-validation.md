# Model-I/O validation

Penny calls a model and gets back JSON: some text, maybe some tool calls. A pile
of informal checks decide whether to use that response, nudge the model, repair
it, reject a tool call, or stop. This document is the formal structure for those
checks — the counterpart to the static tool-argument validators (see the
"Tool argument validation" note in `penny/CLAUDE.md`), but for the *dynamic*
model I/O.

## One taxonomy

The keystone is `penny/validation/conditions.py`: **one catalog of every
condition we classify Penny's behaviour through**, each defined once. The goal is
a single coherent view shared by the user, Penny (the agent), and a maintainer —
not a check that means one thing in the loop and another in the run log.

Each `BehaviorCondition` declares *where it applies*, as data:

- `live` — caught while the run happens (a `ResponseValidator` disposition in the
  agentic loop, or the `send_message` arg-validation gate), so the model can
  recover.
- `run_flag` — surfaces after the fact as a `⚠` line on the run record that the
  `quality` collector reads and the addon badges, derived structurally from the
  persisted `promptlog`.
- `collector_only` — meaningful only for background/collector runs (vs. chat).

A condition can be both `live` and `run_flag`. That overlap is the point:

| Condition | live | run_flag | where it's enforced |
|---|---|---|---|
| `xml`, `empty`, `refusal`, `hallucinated_urls`, `tool_parse_error` | ✓ | | response-validator chain |
| `text_instead_of_tool` | ✓ | | response-validator chain (collector) |
| `no_work_done` | ✓ | ✓ | live: premature-`done()` reject · post-hoc: bail flag |
| `half_formed_send` | ✓ | ✓ | live: `send_message` arg gate · post-hoc: degenerate-send flag |
| `incomplete`, `tool_failures` | | ✓ | post-hoc classifier only |

`conditions.py` is a dependency-light leaf (only `constants` + pydantic, like
`text_validity`), so the loop, the run-health classifier, and the addon-serving
code all import it without a cycle. The `marker`/`detail` strings are the
canonical run-record text and are **frozen** — migration 0072's `quality` prompt
and the addon's TS type mirror them.

This catalog supersedes the old split between `ValidationReason` (live) and
`RunHealthFlag` (post-hoc); `RunOutcome` (the run's terminal state) is a separate
concern and is untouched.

## Live dispositions

`penny/validation/outcomes.py` defines what a validator returns — *what the loop
should do*, not just "reject":

- `Proceed` — response is fine (carries the post-repair value).
- `Retry(condition, nudge)` — re-call the model with a nudge, once per condition.
- `Repair(response)` — silently transform and continue the chain.
- `RejectToolCall(message)` — error tool-result for the call(s), continue loop.
- `NudgeContinue(message)` — append response + a user-turn nudge, continue loop.
- `Stop(response)` — end the loop now.

A `ResponseValidator` is `check(response, ctx: LoopContext) -> ValidationOutcome`
— pure: reads, returns a disposition, mutates nothing. `run_validators` runs an
agent's chain in order (repairs thread through; the first non-proceed
short-circuits). A new guard is a new validator added to an agent's chain, not a
new branch in the loop. Returning a typed disposition rather than raising is
deliberate (the loop reshapes into control flow immediately — a typed return
beats exception-as-control-flow).

## Where the existing checks land

Live (today in `agents/base.py`, moving into validators):

| Today | Condition | Disposition |
|---|---|---|
| `_check_response` XML branch | `xml` | `Retry` |
| `_check_response` empty branch | `empty` | `Retry` (final-step vs mid-loop nudge) |
| `_check_response` refusal branch | `refusal` | `Retry` |
| `_check_response` hallucinated-URL branch | `hallucinated_urls` | `Retry` |
| `LlmToolParseError` handling | `tool_parse_error` | `Retry` |
| strip tool calls when no tools available | — | `Repair` |
| `_clean_malformed_urls` / source-URL append | — | `Repair` |
| `handle_text_step` (collector) | `text_instead_of_tool` | `NudgeContinue` |
| `handle_premature_terminator` (collector) | `no_work_done` | `RejectToolCall` |
| `_abort_if_all_tools_failed` | — | `Stop` |

Post-hoc (today in `database/memory/objects.py`): `classify_run` /
`render_run_record` / `RunHealth` — these read the same catalog so the `⚠` lines
and badge keys come from one place.

`should_stop_loop` (a successful terminator tool) and the per-tool-call dedup
("you already made this exact call") are loop control that stays in the loop for
now; both are candidates to model as validators later if it reads cleanly.
