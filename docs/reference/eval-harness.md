# The eval harness and its reporting — reference

Covers the live-model eval harness's running, reporting and posting mechanics beyond what the map states as rules. Read this in full before changing how a run is scored, reported, or posted.

### Every case declares its area, and the list of cases is generated from them (#2214)

**Every case declares its area, and the list of cases is generated from them (#2214).** A
case passes its driver `area=` — a member of the closed `Area` set in `utils/catalogue.py`
(conversation machine · teaching a routine · memory · answering from the web · standing jobs ·
notifications · background collectors · reading a page · chat tools) — and, when it moves the
conversation machine along one edge, `edge=(from_state, to_state)`, validated against
`OUT_EDGES`. A classifier case names the state it draws from and the state its claim says the
draw lands in; a whole-turn case names the move its turn makes. The LAYER (classifier draw ·
whole turn · micro-context · collector cycle) is not declared: each driver states its own.
`docs/eval-catalogue.md` is the document generated from those declarations — a count per area,
each area's cases (id · behaviour sentence · first wording · edge), and one row per machine edge
naming the cases that cover it at the classifier-draw and whole-turn layers. **`make fix` writes
it and `make check` fails when it is stale**, so a PR that adds, renames or re-declares a case
commits the regenerated file. Collecting the cases calls no model and no endpoint: with
`EVAL_CATALOGUE` naming a file, each driver validates its case, appends it as a
`CatalogueEntry` and skips the test before a sample is stood up
(`python -m penny.tests.eval.utils.catalogue write|check <path>` runs the suite that way in a
child pytest, leaving `utils/` out — it holds no case). An eval test that reaches no driver
fails that run rather than going unlisted.

### An isolated draw is handed what production hands it — a REQUEST case declares its `parked_round` (#2084/#2099)

**An isolated draw is handed what production hands it — a REQUEST case declares its
`parked_round` (#2084/#2099).** `classifier_eval` builds each sample's snapshot through the
production `build_snapshot`, but it passed no `round_binding`, so a case drawn from
request saw a document with no `## The details this task is waiting on` section — the
routine, what the user already gave, what is still missing — which production always
renders there, since it reaches request only through the binder and records what that
settled on the move. A case now names the routine and the values its earlier turns
settled (`ParkedRound`); everything else is READ off the seeded registry row and built
through production's own `round_framing.round_shortfall`, so the case restates nothing the
routine already says. Refused loudly rather than absorbed: a `parked_round` on any other
state, a routine the case's `seed_skills` do not register, a value under a name the routine
does not declare, and a round with nothing left missing (which the binder would have FRAMED)
each name what they broke — the first before any sample runs, the rest at the draw, since
the registry row does not exist until the sample has seeded it. **The rule is a
biconditional and the guard holds both halves of it**: a `parked_round` on any state but
request is refused, and so is a REQUEST case that declares none, because a binding is
present exactly while a round is parked in request — so the next REQUEST case cannot be
written lean.

### Running the suite (`make eval-remote`)

**`make eval-remote` is how the suite runs** (resolved in the Makefile's `eval-remote`
recipe): an OpenAI-compatible provider at 8 xdist workers x 5
samples in flight, which the provider serves concurrently. That 40 is MEASURED, not
chosen: 80 in flight on a 10-core box was both slower and dirtier (samples began missing
their channel-readiness window), so raise `EVAL_REMOTE_WORKERS`/`EVAL_REMOTE_CONCURRENCY`
only with a measurement in hand. Every value stays individually overridable —
`EVAL_WORKERS=1 make eval-remote` is serial for debugging one case, `EVAL_SAMPLES=N`
changes the per-case sample count. The suite's model is gemma
(`google/gemma-4-26b-a4b-it`), pinned to Cloudflare; name a different roster entry with
`LLM_MODEL=<name> make eval-remote` (below). The embedding model is the one piece that
stays local (below).

### Which models the suite may measure (#1996/#2222)

**Which models the suite may measure is CONFIGURATION, and AT LEAST ONE is
required (#1996/#2222).** `EVAL_MODELS` in the primary checkout's `.env` is a JSON list of
`{"model": …, "provider": …}` entries (`penny/tests/eval/utils/roster.py`), and a run
**refuses to start** on an empty roster — the fail-fast shape `LLM_EMBEDDING_MODEL` already
has. The suite's PRs and checkpoints are measured on the roster's first entry; a further
entry stays configurable and runnable by name for a look, and such a run gates nothing. A
JSON list because that is how `PLUGINS` is configured, and because an env var name cannot
contain `/`. The recipe resolves this invocation's entry before anything is spent — the
first by default, or the one `LLM_MODEL` names, and a model **outside** the roster is
refused (it has no upstream to pin and none to record, which is the ad-hoc pass the
variable replaces).

### A configured provider is a hard PIN

**A configured provider is a hard PIN.** The resolved entry's provider becomes the run's
`LLM_PROVIDER`, which reaches the chat client as a `ProviderPin` (`penny/llm/models.py`,
the ONE place the routing field is built) and rides the request body's vendor passthrough as
`{"order": [<provider>], "allow_fallbacks": false}` — every call goes to exactly that
upstream, and nothing is ever routed somewhere nobody chose. There is no unpinned mode, and
an unset provider sends no routing field at all. Upstreams serving the same model are not
interchangeable: one gateway adapter was measured replacing every plain-text `role: tool`
content with `{}`, so a run routed there measured a model that never saw its own tool
results. **Size concurrency to what the pinned provider sustains** — a pin gets that one
provider's capacity and no other, so `EVAL_WORKERS` is chosen against it rather than against
the gateway's whole pool. A pinned provider the gateway filters out for a request (one whose
endpoint does not support tools, for instance) is refused by the up-front smoke call, never
silently re-routed. Name the provider as the gateway reports it back (OpenRouter answers
`Groq`, not `groq`): the answering provider still comes back on every response, the manifest
records it beside the pinned one (a differing pair renders as `not who answered` — the pin
did not hold as written), and the run health block tallies every provider that served the
run.

### The embedding model stays local

**The embedding model stays local**: every run points it at the local Ollama,
so moving the chat model never silently moves the vector space every memory case is scored
against (the dedup thresholds and `find` ranking sit on embeddinggemma's cosine geometry).
Its endpoint moves only if `LLM_EMBEDDING_API_URL` is set deliberately. That is also the
suite's throughput ceiling: ~7.6 embedding calls per sample all queue on one
local GPU, and median embed latency measured 91ms at 10 samples in flight, 161ms at 40 and
384ms at 80.

### A pinned provider is serialised (#2219)

**A pinned provider is serialised instead (#2219).** A pin gets that one provider's rate limit
and no other, so two runs pinned to the same provider at once draw HTTP 429s and lose samples.
`make eval-remote` therefore takes a lock named for the provider it pinned
(`scripts/eval-provider-lock.sh`: a directory in `EVAL_PROVIDER_LOCK_DIR`, default
`/tmp/penny-eval-provider-locks`, holding one file named for the holding shell's pid) just
before its samples start, and holds it until the recipe exits by any route (an EXIT trap). A
second run against the same provider waits — printing
`eval: provider <name> is busy — <lock> is held by pid <n>; waiting (<s>s of 3600s)` every 15
seconds — for at most `EVAL_PROVIDER_LOCK_WAIT` seconds (default 3600, room for the several
scoped runs a fleet queues), then stops with the lock, its holder and what to do. Runs pinned
to DIFFERENT providers never wait on each other. A lock whose holder is gone (a run killed
outright) is stale: the next run says so and clears it. No step can take a LIVE run's lock
away, however many runs are waiting and clearing at once: a pid file is only ever removed by
the run it names or by a run that read that pid as dead, and the directory only by `rmdir`,
which removes nothing that is not empty — so there is no second lock guarding the clearing,
and none to be left behind. An EMPTY lock directory (a run killed between making it and naming
itself) is cleared by a waiter that finds it empty on two looks in a row. A path that is not
such a lock — a file, a link, a directory holding anything but pid files — is refused and left
alone. Driven from real shells in `tests/eval/utils/test_provider_lock.py` (the override mounts
the script into the container for it), with the wait replaced so no test sleeps.

### One call proves the endpoint before a run commits to anything

**One call proves the endpoint before a run commits to anything**
(`penny.tests.eval.utils.endpoint_smoke`, run by the `eval-remote` recipe before it takes the
provider lock): every sample builds its own preflight, so an unserveable model is otherwise
discovered 755 times, concurrently, minutes deep — which is how a full-suite run was spent
against a model whose provider 404'd every call. The refusal carries the provider's own
message verbatim, because that message is the whole answer. **It RETRIES a transient fault
first (#1996)**: one un-retried `HTTP 502` blocked two whole runs while the same model
answered a direct call seconds later, so the check draws up to `SMOKE_ATTEMPTS` times on a
fault whose `LlmFault` is `transient` — and still refuses on the FIRST draw for one that is
not (a 404 is a verdict, not a moment), which keeps the fast, decisive refusal it exists
for. The probe carries the run's own provider pin, so what it proves is the routing the
samples will use, and the upstream that answered it is forwarded into the manifest.

### A run reports its own health (#1996)

**A run reports its own health (#1996)** (`penny.tests.eval.utils.run_health`, printed in the
terminal summary and written to the run dir as `health*.jsonl`, riding the same per-worker
record convention the per-case results use). A degraded run used to look
exactly like a healthy one — `6 passed, EXIT=0` over a cohort in which 34 of 48 samples
never produced their measured turn, all killed by 188 empty `choices` responses, with
nothing in the output naming a number. The block states **completed vs dead samples**, the
**fault classes with counts** (`no choices` · `429` · `5xx` · `timeout`, worst first), and
**every provider that served the run** with its own faults — a poisoned member of a routing
pool reads off the artifacts instead of costing a second run with a pin. The tally is a read
of VALUES: every chat attempt logs its `LlmFault` and serving provider as structured fields
(`penny/llm/models.py`), and a `logging.Handler` on the `penny` logger adds them up, so no
message text is ever matched. **A mostly-dead cohort FAILS**: a case must complete a strict
majority of its intended samples, refused whatever its surviving samples scored, because
dead samples are not missing at random — the faults that kill them
correlate with the work, so the survivors are a biased draw rather than a smaller one. The
run-level verdict moves the session's exit status, so a run computed from a fraction of its
cohort can never exit 0. **A sample can be lost, but never hang the run (#2168)**: a sample
whose boot fails (a preflight that refused) is voided the moment its Penny's run ends, named
by the fault class — unless the preflight failed on a TRANSIENT endpoint fault (#2230): every
failed check carries the `LlmFault` its call failed of (`CheckResult.fault`, riding
`PreflightError.report`), and when each one is `transient` (a timeout, a dropped or reset
connection, a 5xx, a rate limit) the sample is torn down and booted again from a fresh world
onto a database of its own, after a short doubling backoff, up to `SAMPLE_BOOT_ATTEMPTS`
boots. A verdict (a model the endpoint does not list, credentials refused, any other 4xx)
voids at once, and every re-boot is counted on the case's `CohortRecord.retried_boots`, which
the health block prints. Production's own preflight still aborts on the first failure. And
every drive of a sample runs under `SAMPLE_WALL_CLOCK_SECONDS` (30
minutes), past which it is stopped and voided by name (`the sample never finished — stopped at
its …s wall-clock bound`), so the case still closes and records its cohort. Stopping re-sends
the cancel until the task is actually down (`stop_task` in `tests/conftest.py`), because an
in-flight HTTP call can absorb a single cancel when a blocked event loop has also run past its
own deadline — which is how one sample once held a run silent for twenty minutes. Cases drive the
real chat/collector loops and make their claims over persisted DB state + sends; the claims are
counted and reported, never gated, and every case states `min_pass_rate=None`.

### A reply claim is false of a sample with no reply to read (#2252)

**A reply claim is false of a sample with no reply to read (#2252).** A CHAT turn can end on a
model-call abort rather than on a reply the model composed, and the delivered text is then
`PennyResponse.AGENT_MODEL_ERROR` verbatim — a framework apology stating no value of the
model's. `SampleObservation.aborted_on_model_error` (`tests/eval/utils/cohort.py`) is the chat
observer's own read of that: identity against the constant, never a copied string or a
substring match, because a chat turn's abort is otherwise unrecorded (`ChatAgent` stamps no
`run_outcome`, and `ControllerResponse.abort` is read only by the collector) — the delivered
reply is the one surviving signal, and a real reply that merely quotes the apology's words is a
reply, not an abort. `Cohort._answer` (`tests/eval/utils/assertions.py`) is the ONE place every
reply-kind claim (`Claim.kind == "reply"`) is answered, and it fails such a sample there with a
named rationale instead of scoring the apology as a reply that held; every state-kind claim, and
every other driver, reads exactly as it did before. The case document's `samples` row states the
count when it is non-zero, the sample's own banner carries `model error`, and `results.jsonl`
carries the flag per sample (`sample_aborted`, aligned with `sample_scores`).

### Every driver refuses a case that names nothing to drive

**Every driver refuses a case that names nothing to drive** — `chat_eval` with no `ask`, `collector_cycles_eval`
with no `arms`, a micro-context driver with no instruction, utterance or turns — one that states a floor
or leaves `min_pass_rate` unstated, and one that states no behaviour, declares no `area` or names an
`edge` the machine does not have, each with an error naming the case, before a sample runs. A
case is one ask in its wordings and the claims its body makes against the cohort that comes back;
no driver takes a per-sample scorer, and a structural pin in `make check` reads that off the
drivers' own AST.

### A sample's score is its cohort's claims, dealt back out

**A sample's score is its cohort's claims, dealt back out.** A case makes its claims over the
whole cohort, after every sample has run; at case close each claim's answers are redistributed to
the samples that gave them (`_cohort_checks` → `SampleResult.adopt`), one `Check` per claim per
sample. A sample scores claims-passed over claims-answered, a sample the pool refused scores 0.0
under its exclusion reason, and a `Check` carries the claim's `rationale` (the violating value —
``changed ['<key>']`` — rendered beside the outcome, so a ❌ is never bare). The `Check` /
`CheckOutcome` / `CheckView` shapes also carry an advisory flag (`scored=False`) and a
not-applicable state (`ignored`, rendered ➖); nothing in the harness sets either, since a claim is
strictly true or false of a sample and every claim counts.

### The console RESULT line; a green-via-recovery sample renders "fragile"

**The console RESULT line; a green-via-recovery sample renders "fragile."** Each case prints
`RESULT [<case>] mean … · all-pass K/N across N samples` — the mean of the per-sample scores
beside the count of samples every claim was true of — and nothing gates on either: the line is
printed for a person to read, after the dead-cohort refusal. In the per-sample
report a sample that passed but only after the loop refused/recovered a tool call (derived from
the promptlog via `sample_is_fragile`, no new model judgment) is marked
`✅ PASS · fragile`, so real-but-shaky reads distinctly from clean green. **The flag is DERIVED
from the settled score, the way the cause is (#2127)**: the runner observes the reroll while the
sample's database is live (`SampleResult.observe_rerolled`) and `_settle_fragile` reads the flag
off that fact and the score the case settled on. Stamped instead from the drive-time `passed` —
which a sample earns vacuously, having answered no claim yet — it degenerated into "the
run rerolled" and marked claim-FAILING samples fragile, on the banner and in the record.

### A re-rolled draw is read from the REPEATED CONTEXT, not from a marker (#1841)

**A re-rolled draw is read from the REPEATED CONTEXT, not from a marker (#1841).** `draw_rerolled`
(`tests/eval/conftest.py`) is the read, and the agent-loop tests assert a re-roll through it. A
discarded draw writes nothing into the conversation (#1839), so there is no marker to match. What
it still leaves is the **second draw itself**: `Agent._invoke_nondegenerate` re-calls the model on the
*unchanged* message list and `LlmClient.chat` persists every completed draw before returning, so
a re-rolled step is two `promptlog` rows carrying byte-identical `messages` — while an ordinary
step's context has grown by the turns the previous step appended and can never repeat.
A response validator's `Retry` re-draws the same way (#2160), so a validator retry reads as a
re-roll too. `MicroContext._draw_clean` re-draws the same way (and its outer shape-violation re-draw mints a
fresh run id each attempt, so the read keys on the repeated CONTEXT alone, never on the run), so
the ONE read — `draw_rerolled` → `_same_context_drawn_twice` — covers the main loop and every
micro-context. Keying on the repeat rather than on which conditions the loop rejects is what
keeps it honest for an `invalid_draw_conditions` entry nobody has added yet; the retired nudge
markers stay as a **legacy leg** so a promptlog written before #1840 still reads. **Declared
limit**: a draw the backend refused to parse (`LlmToolParseError`) raises before the client
persists anything, so that one re-roll leaves no row and no promptlog read can see it.

### Presenting eval runs on a PR (the report-as-comment process)

**Presenting eval runs on a PR (the report-as-comment process).** The harness emits a
verbatim per-case markdown report when `EVAL_REPORT_DIR` is set (`make eval-remote` passes it
through). The **PR body stays a stable description**; **every tweak + re-run pushes the
commit and ADDS A NEW COMMENT** with that run's report — never rewrite the body, never
overwrite a prior comment — so the iteration history (each review round's before/after)
survives. See `docs/agent-task-workflow.md` §4.

### Per-run artifacts + the required lever (#1692)

**Per-run artifacts + the required lever (#1692).** When `EVAL_REPORT_DIR` is set, the run
also writes three durable artifacts into that directory (beside the `<case_id>.md`
transcripts), emitted by `penny/tests/eval/utils/artifacts.py`: a `manifest.json` (the run's
input identity — commit SHA, dirty-diff filename, model + embedding model + `EVAL_SAMPLES`,
and the run's one-line **lever**), a `dirty.diff` (the working-tree diff saved verbatim,
only when dirty), and a `results.jsonl` (one mechanically-diffable record per case: run id,
case id, family tag, mean, all-pass rate, per-check outcomes, N, timings). The manifest also
renders as a markdown header (`commit · model · N · lever`) atop each `<case_id>.md` report.
**The lever is required**: `make eval-remote` forwards `EVAL_LEVER` (the run's hypothesis, e.g.
`EVAL_LEVER='moved instruction X from skill to prompt'`) alongside `EVAL_REPORT_DIR`, and a
report run with no lever **fails fast** with an actionable message before any sample runs —
the lever is what makes a score shift attributable to the change that caused it. The commit
SHA + dirty diff are computed **host-side** in the Makefile (`EVAL_COMMIT`/`EVAL_DIRTY_DIFF`)
because `.git` is not mounted in the eval container. The `family` tag defaults from the test
module name (`test_<x>` → `<x>`) or is set explicitly via a `family=` arg on a runner call.

### A sample runs on its own clock, which starts at one declared instant (#2207)

**A sample runs on its own clock, which starts at one declared instant (#2207).** An ask that
names a time relative to now is a different ask on the machine's clock at every run: "until 10pm
tonight" has already passed at 23:30, "until sunday night" is today on a Sunday, and a rule that
states no hour fires at whatever hour the run was made. So `_real_model_config` — the one seam
every runner builds its config through — sets `Config.clock` to a `PinnedClock`
(`tests/eval/utils/clock.py`) starting at **Wednesday 14 October 2026, 14:00 in the eval user's
zone**: mid-week, so a named day lies clearly ahead; early afternoon, so "tonight" is ahead on the
turn's own date and an hour-less rule does not pass for a morning one; clear of a daylight-saving
change and a month's end. A fixed date rather than a fixed hour of the run's date, because an hour
alone leaves the day of the week to the calendar. The clock **starts there and keeps time** — the
machine's clock moved by a constant offset — so rows written a moment apart keep their order; a
frozen clock would stamp a whole sample with one instant and collapse every recency read. Because
the runtime reads the time in one place (`penny/clock.py`, below), the whole sample is on it: the
`Current date and time` line, the instant `parse_expires_at` counts words from, the `created_at`
a rule is anchored at, the collector's readiness, every row the seeders and the turn write, and
the `state_transition` row a claim reads `turn_at` off — so `job_end.py` judges a stored end on
the clock the job was set up on. "Tonight" is the evening of the day that clock reads, for
production and the claim alike: neither rolls an hour already past to tomorrow. A seeder that
states a time as a distance counts it from `db.clock`, never the machine's. Pinned in
`make check` (`tests/test_clock.py`) end to end: a turn on a clock years away tells the model that
clock's time and leaves no row stamped off the machine's.

### Per-sample penny logs (#1909)

**Per-sample penny logs (#1909).** Beside each sample's DB (`<case_id>-<n>.db`) the harness
writes that sample's own logger output as `<case_id>-<n>.log` — the fourth durable per-sample
artifact, alongside the DB and the `<case_id>.md` transcript. `<n>` is `sample_number` — the
number the report calls that sample, so its name resolves to its own files (#2076: it did not,
and a reader following a report's name to the next sample's clean database read a correct
exclusion as a harness defect and filed it as one). One place derives it and every surface reads
it, **held by a structural pin** like the logging seam below: `make check` parses the harness's
own AST and refuses any second spelling of `sample_index + 1`, because a duplicate is invisible
while it agrees and a rebase is exactly where it stops agreeing. A model call that FAILS raises
before the client's persist step, so it writes no promptlog row: everything the run said about
the failure ("LLM chat failed", a timeout attempt, each discarded draw's reroll condition)
existed only as logger output, which pytest captures and then discards for every sample that
passes — the same failure class #1909's `RunAbort` covers from the DB side, and the only carrier
for a run that died on its FIRST call (no row exists to stamp). `sample_logging(db_path)`
attaches a DEBUG `FileHandler` to the `penny` logger for the span of ONE sample (so every line
is attributable, and the handler + raised level are both gone afterwards), and every runner
stands its sample up through the `eval_penny(config, server)` wrapper rather than
`run_penny_with_server` directly — one seam that covers the runners that exist and the ones
added later, **held by a structural pin** rather than by convention (`make check` parses the
harness's own AST and refuses any caller of `run_penny_with_server` but `eval_penny`; a runner
that bypasses the seam is byte-identical in every other respect and its only symptom is a log
file silently never written — measured on the multi-cycle collector runner, which was written
before the seam existed and survived a rebase still bypassing it). Mechanical and unfiltered: no
content filter, and the path is derived from the sample's own `db_path`, so off-report runs write
it into the throwaway tmp dir.

### Failure-cause partition (#1695)

**Failure-cause partition (#1695).** Every FAILED sample is tagged with its cause, derived
STRUCTURALLY (never a model judgment): **harness** (a `TimeoutError`), **pathology** (a known
model pathology fired), or **behavioral** (the model simply got it wrong — the real signal the
loop chases). Pathology is read post-hoc off the persisted promptlog by `run_exhibited_pathology`
(in `tests/eval/conftest.py`), which scans each row's RESPONSE with the SAME `text_validity`
detectors the agent-loop reroll guard runs live (`Agent._unusable_output_condition`): a punctuation
collapse (`DEGENERATE_OUTPUT`), a leaked Harmony envelope (`TOOL_CALL_LEAK`), a collapse-shaped
tool NAME, or a bare call-fragment reply. **The scan reads only the model's own OUTPUT** — each
row's persisted `response`, never the input `messages` — so text the model was handed (a page
quoting a collapse, a framework nudge) cannot tag a sample. Cause ordering:
pathology outranks a timeout (the poison is the root cause, the timeout its symptom), then harness,
then behavioral. `classify_cause` (pure, in `artifacts.py`) encodes the partition. **The cause is
DERIVED from carried facts, wherever the score settles (#2125)**: the runner observes the two
structural facts while the sample's database is live (`SampleResult.observe_faults` — did it time
out, did its persisted output carry poison) and one derivation reads the cause off them. A
sample scores nothing at drive time — its cohort's claims arrive at case close, long after that
database is gone — so `adopt` settles the cause again there against the score it just took on, and
a re-scored sample can never keep the cause its old score earned. `sample_causes` (in `artifacts.py`)
is the one list BOTH renderings read, so the record and the console tally cannot print different
numbers for one case. The `CaseArtifact` (`results.jsonl`) carries `sample_causes` (per-sample,
aligned with `sample_scores`; `null` = passed), `cause_counts`, and `pathology_excluded_mean` — the
honest read of model behaviour, the mean over every sample that is NOT a pathology failure, so a
degeneracy spike can't sink a case's score while its raw mean + pathology count stay visible. The
console RESULT line carries a second line with the same tally (`render_cause_summary`, single-sourced
so the PR-comment renderer renders one shape). Deterministic contracts (no live model): the pathology
scan + stamping + render in `tests/test_eval_harness.py`, the artifact fields + pure classifier in
`tests/eval/utils/test_artifacts.py`.

### Regression-aware reports + thinking at failed turns (#1693)

**Regression-aware reports + thinking at failed turns (#1693).** When `EVAL_BASELINE` names a
prior run's report directory (or its `results.jsonl`), the per-sample report diffs each
check against that prior run and marks a NOW-FAILING check **REGRESSED** (`❌ 🔻 REGRESSED`) when
it was FULLY GREEN there (`passed == total`) — a flip, distinct from a check that was already red.
The flip is per-check, per-`(case_id, label)`, reading the prior `CaseArtifact` records unchanged
(`penny/tests/eval/utils/baseline.py`); the regressed check names its baseline run inline in the checks
legend. `EVAL_BASELINE` is forwarded into the container like `EVAL_REPORT_DIR`/`EVAL_LEVER`; unset
(or a missing file — a first run) → no REGRESSED marks, no error. The report also lifts each LLM
call's **thinking** (already persisted on `promptlog.thinking`) into a collapsed `<details>` at
the failed/regressed **tool-call** turns only — passing turns are omitted so the comment never
bloats. On a first run with no baseline, thinking still renders at failed turns.

### Run-comment assembler (#1717)

**Run-comment assembler (#1717).** The per-run artifacts and per-case transcripts all exist after
a run, but no step composes them into THE ONE markdown comment that gets posted.
`penny/tests/eval/utils/assemble.py` is that step, and is where that document's shape is DEFINED —
there is no prose spec to drift from what is emitted. Given a completed run's report directory it
emits one document: a run-level **roll-up** only when the run spans more than one case (one counted
verdict over every check the run made, its spread reading, and the run's identity as a table —
commit, provider, embeddings, cost per sample, run id), in diff mode a `flips:` index, then one
section per case — the case document its `<case_id>.md` carries (its own manifest-header prefix
stripped) — and the local-artifacts footer. The comment carries **no RESULT line and no gate
line**: `mean` and `all-pass` are aggregates of per-sample scores, printed on the console and kept
in `results.jsonl`, and no case gates on either. Pure artifact consumption — no model, no git, no network — so it's exercised by plain
(non-eval) whole-render tests (`tests/eval/utils/test_assemble.py`), not a GPU run. Run it via
`python -m penny.tests.eval.utils.assemble <report_dir>` (prints the comment to stdout) or
`EVAL_REPORT_DIR=… make assemble` (the same dir `make eval-remote` wrote to). **One and only rendering
(#1753/#1759):** EVERY sample block folds whole under its banner — collapsed by default, its full
body always one click away ("default collapsed" never means the body is dropped) — identical in the
on-disk `.md` and the assembled comment. There is **no compact / banner-only form and no `--full` /
`EVAL_FULL` flag** (removed in #1759 — banner-only rendering dropped entirely). The assembler
re-normalizes each block via `report.split_sample_blocks`/`parse_sample_block`/`fold_sample`, so a
re-assembled pre-#1753 run (unfolded `#### ` failures) folds uniformly too.

### A case reports THREE SECTIONS, and states what its samples SHARE exactly once (#1997)

**A case reports THREE SECTIONS, and states what its samples SHARE exactly once (#1997).**
`report.py` was organised around `Verdict` — one pass/fail per check, rendered per sample as
`expected`/`actual` rows — a shape that cannot express variance at all and reads the harness
bucket as behavioural failure. Under #1994 a case reports **assertions** (pass counts, reported
and never gated), **variance** (entropy and textual spread against a ceiling) and **harness**
(samples too broken to count), so `report.CaseSections` renders those three as the document's
structure. It RENDERS and never computes: the numbers, the proposed ceilings and the standings are
all `cohort.py`'s, so a reader comparing document against data compares one arithmetic to one
rendering. Two claims the sections make carefully: **no assertion carries a floor** — a
deterministic check is expected to be strictly true of the run, so a threshold under one either
never fires or sits below the observed rate and blesses the defect as the contract, which the
table says in as many words where it ends. A reply-content claim is therefore not a special case:
it counts and colours on its own rate like any other, and where it was read from
(`cohort.Claim.kind`) decides only how its per-sample check renders and what it anchors to. And the
harness section names its **dominant failure class**, computed from the cohort's own exclusions
rather than from `run_health`'s run-level tally, which is per process and cannot say which case a
fault landed in.

### What every sample shares in the report

Around the sections, `report.render_case_document` states what every sample shares: **the ask in
its K wordings** (listed once — a sample names which it used), **the seeded world**
(`World.render`), **the system prompts**, and a **map saying which samples to open**. That map is
what makes #1994's "read ONE sample" step actionable: `cohort.standings` marks exactly ONE sample
MODAL (the first carrying the majority shape — naming eight equally puts the choice back on the
reader), every sample whose whole shape differs an OUTLIER, the rest TYPICAL, and a dead sample
DEAD. Standing reads the sample's WHOLE shape across every structural feature, so a sample
agreeing on tool sequence while minting its own routine shape is an outlier rather than being
folded away. Nothing is dropped — every sample still folds under its banner, one click deep.

### Durable local artifact home (#1734)

**Durable local artifact home (#1734).** "Stays local" means the **primary checkout's
`data/eval-artifacts/`**, not the running worktree's `data/`. The `./data` bind mount is relative
to the compose-file dir, so an eval run from a worktree would write artifacts under that worktree —
where they die when it's swept post-merge. So `make eval-remote` / `make assemble` resolve the primary
checkout host-side (from the shared git *common* dir — identical from every worktree) and
bind-mount its `data/eval-artifacts` at `/penny/eval-artifacts`. A report run (one declaring
`EVAL_LEVER`) with no explicit `EVAL_REPORT_DIR` defaults to a run-stamped
`/penny/eval-artifacts/run-<stamp>` (= `<primary>/data/eval-artifacts/run-<stamp>` on the host), so
artifacts **survive the worktree being removed** and `EVAL_BASELINE` can point a later run at a
prior run's durable path. An explicit `EVAL_REPORT_DIR` is always honored; a lever-less iteration
run stays ephemeral (no artifacts, no lever requirement). The Makefile derives the mount at parse
time, so `make -n eval-remote` shows the concrete `-v <primary>/data/eval-artifacts:/penny/eval-artifacts`
from any worktree.

### Posting a run: `make eval-report` one-shot + the unreviewed-run banner (#1757)

**Posting a run: `make eval-report` one-shot + the unreviewed-run banner (#1757).** Posting a run's
report is a **one-shot**, and the debt of *not* posting is made **structural** (structural-state-over-
model-judgment, applied to the eval loop). Every run's report belongs on its PR, and posting is a step
separate from running, so skipping it would leave no visible debt and nothing would interrupt the next
run (one session ran 4+ sequential report-runs without posting any). Two mechanisms:
`make eval-report PR=<n> [RUN=<run-dir-name>] [FORCE=1]` assembles the named run — default: **the run
THIS TREE measured** (see the next paragraph) in the durable home — captures the
containerized `python -m penny.tests.eval.utils.assemble` stdout **cleanly** (never `make assemble` piped
through stripping), posts it **verbatim** to PR `<n>` via `GH_TOKEN=$(make token) gh pr comment`, then
stamps a **`.posted` marker** (holding the comment URL) into the run dir. It **fails loudly** when `PR`
is unset, the run dir is missing, the token is empty, or the assemble output is empty; it is
**idempotent by the marker** — a run already carrying `.posted` re-posts only with `FORCE=1`, else prints
the existing URL and exits 0. And `make eval-remote`, **before anything is spent**, scans the durable
home for run dirs with a `manifest.json` but **no** `.posted` and prints a **loud, multi-line banner**
naming each (`⚠ N unreviewed eval run(s) … → post each: make eval-report PR=<n> …`) — **warn, never
block** (a multi-run sweep stays possible; the debt is just undeniable). **Marker semantics**: a
`manifest.json` = a completed run; a `.posted` marker = a posted (reviewed) run; a **lever-less
ephemeral run writes no `manifest.json`** so it never appears in the banner (by design — nothing to
review). The scan/resolve/banner logic is pure, plain-tested `penny/tests/eval/utils/checkpoint.py`
(`run_dirs` / `unreviewed_runs` / `runs_at_commit` / `render_banner` + a `resolve`/`banner` CLI), invoked
in-container by the recipes so it reads the mounted home; the Makefile checks the marker host-side. Both
recipes reuse the #1734 `EVAL_ARTIFACTS_HOST`/`_MOUNT` derivation, so `make -n eval-report` shows the
same `-v <primary>/data/eval-artifacts:/penny/eval-artifacts` mount from any worktree.

### A bare `RUN` resolves to THIS TREE'S run, or refuses (#2098)

**A bare `RUN` resolves to THIS TREE'S run, or refuses (#2098).** That durable home is **shared by every
worktree** — one mount, every fleet agent's runs in it — so "the most-recent completed run" was routinely
a *sibling's*: one porting agent posted another agent's six reports to its own PR and stamped the
sibling's run dir `.posted` under a comment URL on the wrong PR. The bare-`RUN` default is now keyed to
the **invoking tree's `HEAD`**: candidates are the completed, **unposted** run dirs whose manifest
`commit` equals it. **Exactly one** → post it; **zero** or **several** → the resolver refuses on stderr,
naming the home, the commit, the runs it found (and any run already posted at that commit — whose next
move is `RUN=<run-dir> FORCE=1`, not another eval). **There is no "most recent" fall back.** `HEAD` comes
from the Makefile's single `HEAD_COMMIT_CMD` (`git rev-parse HEAD 2>/dev/null || echo unknown`), expanded
into **both** recipe shells — `eval` records it as `EVAL_COMMIT` in the manifest, `eval-report` matches
against that — so the two derivations cannot drift. It is derived **host-side**, because a worktree's
`.git` is a file pointing at the primary checkout's gitdir that the container has no reason to mount; and
it is a deferred (`=`) variable rather than `$(shell …)`, so `make -n eval-report PR=<n>` prints the
resolution instead of running git. **A sentinel is not an identity**: a run dir whose `manifest.json`
cannot be read, records no `commit`, or records `UNKNOWN_COMMIT` (`"unknown"` — what *every* tree
writes when git cannot read its HEAD) has **no** measured commit (`None`), which never equals one; and
a caller passing that sentinel is refused before the candidates are counted. Otherwise two trees that
both failed to read their HEAD would share one identity and post each other's runs — #2098 again,
through the fix.

### Past the 64K comment cap, a run posts as MANY comments (#1808)

**Past the 64K comment cap, a run posts as MANY comments (#1808).** GitHub refuses a comment body over
65,536 characters, and `eval-report` had no split path — so an over-cap run could not be posted at all by
the supported route (`GraphQL: Body is too long … (addComment)`), and one 8-sample chat beat assembles to
~290K even after the shared prompts were stated once. The recipe now stages the assembled body into `<run-dir>/comment/body.md`
and cuts it with the pure, plain-tested `penny/tests/eval/utils/comment_split.py` (`sample_fold_segments` /
`partition_on_sample_folds` / `render_parts` / `split_run_comment` + the two guards + a CLI): **the only
legal seam is a sample-fold boundary** (`report.SAMPLE_BLOCK_START`, single-sourced with the assembler's
re-normalizer — a cut inside a `<details>` renders as broken markup), so **concatenating the parts
reproduces the document byte for byte** — nothing dropped, summarised, or re-wrapped
(collapsed-never-means-removed, applied to posting). Each part is headed `(report N of M — split for
GitHub's 64K comment cap; content verbatim)`; a run that FITS is ONE part with no header (posting is
byte-identical to before); the run header opens the document so it lands on part 1; `.posted` holds the
**first** part's URL so idempotency + the unreviewed-run banner stay honest; the budget is **~58K**, not
64K (the header, and GitHub counting differently from `wc -c`). One loud refusal guards what is posted:
a body **opening with build noise** (`docker compose`, `GIT_COMMIT=`, `#1 [internal]` — what a hand-piped
`make assemble` publishes; it happened, and eight comments had to be deleted).
