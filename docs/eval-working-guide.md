# The Eval Working Guide

How to plan a case or a change, run it, read what came back, and decide what to fix — the
working method, distilled from what has been learned running the suite.

**This doc does not restate the contract.** [`docs/principles.md`](principles.md) is Penny's
north star — the rules a mechanism or a test is designed by. [`docs/eval-case-design.md`](eval-case-design.md)
is the contract one case is written or ported to: what an assertion is, the three categories,
the porting checklist, cohort sizing. [`docs/eval-iteration-workflow.md`](eval-iteration-workflow.md)
is the loop a round of model-facing work runs in: beat → pairs → seeded world → run → report →
ruling. Where this guide and one of those overlap, it gives the point one line and a link, then
adds what working the suite has taught beyond it — a measured number, a cause, a fix, the
concrete command or file to look at. Read this guide after those three, as the field notes for
using them.

Who it's for: anyone planning a new case, changing a prompt or tool surface and needing to
measure it, running the suite, or reading a run to decide what to do next.

---

## 1 · What the suite is for

The suite is the list of behaviours Penny supports — `docs/eval-catalogue.md`, generated from the
cases themselves. One case is one behaviour under one entry condition, in five wordings, three
samples each: a cohort of fifteen ([`docs/eval-case-design.md` §4, §6](eval-case-design.md)).

- **Facts are asserted; behaviour is measured.** A claim is a fact about the end state. How the
  model got there — which tools, in what order, in what words — is measured as variance and
  never asserted. This is the whole of [`docs/principles.md` §4.1](principles.md).
- **Claims are counted and reported, never gated, and carry no floor.** A well-formed case
  scoring 4 of 15 is a finding about Penny, not about the case: the case lands, the gap is filed,
  and the fix comes after it, never folded into the port
  ([`docs/eval-case-design.md` §1, §10](eval-case-design.md)).
- **Reviewing a case judges the test, not the model:** is it a real use, is the setup fair, are
  facts asserted and behaviour left to variance, does it follow the principles
  ([`docs/eval-case-design.md` §11](eval-case-design.md)).
- **A gap is judged by four questions:** what was the user's goal; did Penny meet it; do the
  strict checks touch only facts of the input and the output; is everything else about the
  model's output treated as variance. Answering them in that order keeps a review from sliding
  into "the model should have said it differently," which is a variance complaint wearing a gap's
  clothes.
- **Only repeatable patterns are worked on.** The code owner: *"i don't wanna chase stray single
  failures, i'm only interested in repeatable patterns of major gaps."* One sample's one-off
  miss is not a finding; the same miss across phrasings or across a run's re-sample is.

## 2 · What a check is

A check is strictly true or false of one sample — no thresholds, no "mostly." There are exactly
three kinds, and nothing else is an assertion
([`docs/eval-case-design.md` §2](eval-case-design.md)):

| kind | reads off | what it asserts |
|---|---|---|
| **landed** | the conversation machine's walk | the state or outcome enum the turn ended in |
| **store** | the sample's own database, live | what the store holds afterwards — the row or entries as the turn left them, and that everything the ask did not name is preserved |
| **provenance** | the sample's inputs vs. its output | every specific value traces to what the round was given, read in both directions: nothing supplied is omitted, nothing is invented |

- **Assert what survives, never that the model refrained.** No "did not call X," no "wrote
  nothing." A turn that should do nothing is claimed as: landed in idle, the store as it was,
  nothing invented ([`docs/eval-case-design.md` §2 "Assert what survives"](eval-case-design.md)).
- **Never assert a route.** A tool name, an argument choice, a call count, or reply wording is
  *how*, not *what* — read in the tool-sequence and reply-spread variance features instead
  (§3 below).
- **Never force or inject a tool call.** A natural input leads the model to act, and the real
  gate answers. A mechanism — a gate, a parser, a bound — is proved separately by a deterministic
  mock test in `make check`, not by an eval case
  ([`docs/principles.md` §4.4](principles.md)).
- **A reply claim reads a fact, never vocabulary.** A token only one source carries, invented for
  the case, with no whitespace, since typography folding covers a declared set of spaces and not
  every space. A list of words a reply "should" contain is tuned to whichever model happened to
  be read first — the thing this design replaces.
- **A claim production already guarantees is not made.** If a validator already re-rolls a draw
  until it matches, a claim over the same thing runs 15/15 by construction and measures the
  validator, not the behaviour. Say so in the case rather than leaving a hole that reads as an
  oversight ([`docs/eval-case-design.md` §2 "Do not claim what production already
  validates"](eval-case-design.md)).
- **The world carries the condition, not a clause in the ask.** The negative direction of a
  behaviour is the same wordings over a different world. Where the two directions have different
  expected outcomes they are two cases, named as a pair, so a model that always answers one way
  cannot pass both.
- **The entry condition is seeded, not driven as an earlier turn.** One case is one model run.
- **Fixtures cohere with the state they claim and never lend their own phrasing to the prompt
  under test.** A seed proves itself before the turn runs (a premise probe), and `make check`
  holds each fixture's anchors against its own page.

**A budget is an assertion on behaviour, never a check.** A time, step, token or cost limit
that turns a slow or expensive sample into a failure is asserting *how* the model behaved, which
is the thing this design reserves for variance. Nothing in a case "goes red" on a budget: a
sample that ran long is read through the cost section (§3) and the run's own step cap, never
through a claim that fails it. The runtime's own step cap is a bound on the run, not a claim
about it — it stops a run from hanging, and says nothing about whether the run was correct.

**Where the ask calls for a judgement the world does not settle** — "the top three" of a list
the world never ranked — assert only what the world does settle: the count the ask stated, and
that each chosen item is a real member of the right set. Which ones were chosen is variance, not
a claim, because the world supplies no single correct selection to check against.

**A measurement must be able to see the failure.** Before concluding anything from a count, show
that the count could have come out the other way — the same point as
[`docs/principles.md` §4.6](principles.md), "a check must be able to fail." A surprising 0 of N
is as often a wrong check as a wrong model, and a 15 of 15 can be vacuous. Two harness misreads
found this way: a capitalised sentence-opening word read as a name, and "Yes" read as a stock
status. Both looked like passing checks until someone asked what a correct sample would have had
to do to fail them.

## 3 · Variance, and what it tells you

This is the part the contract docs don't carry, because it isn't a rule about what a case may
claim — it's what was learned pointing the same instrument at the suite, case after case.

What's measured per case: the distinct tool sequences its fifteen samples took and their
entropy, calls per sample, the spread of replies, and the shape of anything the model named or
minted. The case document marks one sample **modal** (the first to carry the majority shape) and
the differing ones **outliers**; those are the samples worth opening, not a scroll through all
fifteen.

**High tool-sequence variance means the model had several equally reasonable routes.** That is a
property of what it was offered or shown, not of the model, and it is reduced there —
instructions are the last resort, not the first. Measured across the tool surface, whole suite,
on the suite's model: distinct tool paths over the 89 shared cases fell 220 → 186, mean
tool-sequence entropy 0.246 → 0.185, calls per sample 1.27 → 1.21, and claims *rose*, 99.0% →
99.8% — so the reduction cost no case's claims.

Causes found so far, each with the fix that worked:

- **Overlapping tools** — two ways to do one read or write on the same store. In a run this shows
  up as a tool-sequence table with several equally-sized buckets, none dominant, on a case whose
  behaviour is simple. The fix: reason from the stores, not the tools — list each store, each
  kind of read and write on it, and give each exactly one tool, named `target_verb`. A tool is
  removed only when an exact equivalent remains, so no kind of read or write is lost.
- **Two words for one thing** in what the model reads — "collection" and "memory" for the same
  concept, "skill" and "routine" for the same mechanism. The spread shows as samples splitting
  cleanly along which word the prompt used nearest the decision. The fix is one word per thing
  everywhere the model reads; where the user's own word differs, say so once rather than
  switching vocabulary mid-document. Measured: removing the only sentence tying "routines" to
  skills took one wording to 4 of 6 passing, and restoring one bridging sentence brought it back
  to 30 of 30.
- **An optional argument** the model fills in when it isn't relevant to the ask. Adding an
  optional parameter to a tool is an input change to every case that offers that tool, not a
  no-op — measure the cases that touch it, not just the one the parameter was added for.
- **A defect below the model that looks like model variance.** A provider returned parallel tool
  calls as one call with concatenated arguments, and the client dropped or mangled them: 31 of
  3,511 calls, concentrated in two cases that showed five and two tool paths respectively.
  Splitting the calls correctly took both cases to one path. The lesson: read the raw logged
  responses before attributing a spread to the model — a transport bug reads exactly like
  indecision until someone opens the wire format.
- **A read that does not say its result is complete**, so the model reads again to be sure. Shows
  up as an extra, redundant call on some samples and not others, depending on whether the model
  happened to trust the first answer.
- **State that changed and was not shown** — something cleaned up after a round was called off —
  so each sample goes looking for it a different way. The model is reacting correctly to stale or
  missing state; the fix is showing the current state, not telling the model to stop looking.
- **One fact shown in two places**, so samples split on which one to read. The fix is to render
  the fact once, at the place the model is already looking, and stop rendering the second copy.
- **Stale seeded data** teaching a pattern the prompt no longer wants — a fixture is mentoring the
  model into a reply shape left over from a prior wording. Re-sample the fixture after the
  producing prompt changes ([`docs/eval-iteration-workflow.md` §3](eval-iteration-workflow.md)).

**Not every spread is worth removing.** Cases where the thing asked for does not exist explore,
and the step cap already bounds them — the code owner ruled that variance out of scope. Don't
spend a round chasing it.

**The one-call rule as an audit.** Any fact in Penny's stores should be one call from what she is
shown, with no guessed argument ([`docs/principles.md` §2.5](principles.md)). Samples that took
several calls to reach one fact point at an anchor that is not rendered — read the tool-sequence
outliers with this rule in mind before assuming the model was merely being thorough.

**Names are settled by measurement, not taste.** *"you'll know when you actually measure whether
the model can reason about the name."* A name that reads well to a person can still produce a
wide routine-name or tool-sequence spread; don't adjudicate naming in the abstract — run it.

### The two shapes a measured feature takes

- **A category with an entropy** — the tool sequence, the routine shape, a case's own named
  feature — is a value every sample either shares or doesn't: a distinct-value count, a modal
  share, a normalised entropy, and (where the spread is narrow enough) a proposed one-sided
  ceiling, per model, per N ([`docs/eval-case-design.md` §5](eval-case-design.md)).
- **A share or distribution** — a fraction read off each sample rather than a value samples
  agree on — reports as how many pooled samples had a reading, their median, their range, and
  the samples above a stated threshold. It carries no entropy and proposes no ceiling: a
  fraction is close to a different number on every sample whatever the behaviour, so entropy on
  it reads near 1.0 on a cohort in full agreement and says nothing.

**A case can declare its own measured feature, beside its claims, not just the shared ones in
`cohort.py`.** `MAILBOX_FIGURE_IN_REPLY` in `penny/penny/tests/eval/chat/idle/test_email_dispatch.py`
is a case-local `Feature`: whether the reply names a mailbox figure at all, measured only where
that reading is meaningful. `_STOCK_COUNT_VOLUNTEERED` in
`penny/penny/tests/eval/extractor/test_browse_extract_fields.py` is a case-local `Share`: of the
samples that answered the stock boolean, the fraction that also volunteered the count the page
supplies beside it — something the ask never requested, so it can't be a claim. **A measure the
harness cannot yet carry is a harness change**, designed by the coordinator, landing before the
case that needs it — a case never invents its own pooling arithmetic.

### Cost is variance

How much the model spent before it answered is behaviour, exactly like a tool sequence — so it
is measured, never a check, and never gates a run (the budget rule above). What's recorded today,
and where each number shows up:

- **Duration and call count are recorded per sample.** Every sample's own banner carries its
  wall-clock duration and its call count, read straight off the real promptlog's `duration_ms`
  per call — the same numbers production logs, not a harness stopwatch.
- **Tokens and reasoning share are a per-case average, in the case document's own Cost
  section** — beside Assertions and Variance, not inside either. Input tokens are split from
  output: input is the prompt and context Penny's own design controls, so a rise there is what
  a prompt edit regresses; output is the model's, so a rise on an unchanged prompt is a model or
  config change. Reasoning tokens (the provider's own count, else estimated from the
  thinking-vs-output character ratio, named as such) render as a share of output. Both carry a
  one-sided proposed ceiling, per model, with a round +10% margin rather than a measured one —
  cost isn't an entropy reading.
- **A re-rolled or discarded draw is read from the repeated context, not from a marker** — a
  discarded draw writes nothing into the conversation, so what gives it away is that the retry
  calls the model again on the *same* message list, leaving two promptlog rows with
  byte-identical `messages`. A sample that reached a **passing** result only after such a reroll
  is marked **fragile** on its banner (`✅ pass · fragile`) and recorded per sample in the run's
  `results.jsonl` (`sample_fragile`) — a loop-health signal, never a claim: a sample the claims
  already failed is never marked fragile.
- **What is not recorded**: no dollar or currency figure anywhere — cost is tokens, calls and
  seconds, never a price — and no per-sample token count, only the case-level average.

### Consequential vs. cosmetic, and reading the outliers

A divergence that implies a different end state (a different tool sequence, a different routine
shape, a different count of entries stored) is **consequential** — it's worth opening that one
sample. A divergence that is just a different word for the same outcome (the routine's invented
name, the reply's wording) is **cosmetic** — it's a system-level finding for the variance table,
never a fact about one sample. Getting this backwards is its own failure mode: on one run 8 of 9
outlier rows were the naming feature alone, reporting "60% of samples are outliers" where the
true statement was "1 of 15 diverged consequentially." Read the case's "Outliers" fold, not the
raw entropy number, to see which split applies.

A feature that reads **absent** on every sample (no tool call happened, no routine was named) is
not a feature in agreement — it's **blind**, and the report marks it that way. `0.000` entropy
means two different things depending on which one it is; don't read a blind feature as a clean
pass.

## 4 · Planning a case or a change

- **Start from the user's goal and write the behaviour as one sentence:** *"In the <locus>, when
  <X>, Penny <Y>."* This is `behaviour=` on the driver call
  ([`docs/eval-case-design.md` §4](eval-case-design.md)) — not a label for the report, the
  sentence a reader checks the case against.
- **Decide the layer** — a classifier draw, a whole turn, one micro-context, a collector cycle —
  and the area, and the machine edge when the case moves one. The layer picks the driver
  (`chat_eval`, `classifier_eval`, `collector_cycles_eval`, `extractor_eval`, `framer_eval`,
  `labeller_eval`, `binder_eval` — `penny/penny/tests/eval/conftest.py`); it isn't declared
  separately.
- **Where wordings come from depends on the layer.** For a chat case the five wordings are ways
  a user would actually say the ask. For a micro-context case the arm is **Penny's own text**,
  written upstream by the calling draw — the browse extractor's `extract` instruction, a
  collector's rendered program step — so its wordings are sampled from what that draw actually
  produced, never invented for the case
  ([`docs/eval-case-design.md` §6](eval-case-design.md)).
- **Write the five wordings so each carries the same facts**, and declare those facts beside them
  so a dropped one is caught in `make check` — a fixture that silently drops a fact on one
  wording is a case that measures five different behaviours and calls the result one.
- **The world has the shape and size of the real input, not a trimmed stand-in.** A sparse page
  or a five-link store misleads — it doesn't exercise what a real ninety-link front page or
  real-sized store does. A world is synthetic in content and faithful in structure: transcribe
  the shape of a recorded input (its sections, its counts, its odd members) with invented
  values. This is the genericising step of the log-to-test-to-fix loop (`penny/CLAUDE.md`, the
  live eval suite section): pull the real trigger from `promptlog`, confirm it reproduces, swap
  every real value for a synthetic one of the same shape, and confirm it still reproduces before
  committing the fixture.
- **Go through the three kinds of check (§2) and say what is claimed under each, or that it is
  empty.** An empty category is a report, not an omission — say which of the two it is.
- **A new case, or the existing one.** A case is one behaviour under one entry condition. If the
  ask or the world changes what a correct sample is, or changes the set of claims a case can
  make, it is a new case and the existing one is left exactly as it is. If a correct sample for
  one would be a correct sample for the other, it is the same case, phrased a sixth way —
  pooled in, never forked beside it ([`docs/eval-case-design.md` §6](eval-case-design.md), the
  phrasing-versus-scenario line).
- **For a change to what the model sees — a tool surface, a renamed concept, a prompt — write the
  cases first.** Before the tool surface was changed, six lookup cases were written and replayed
  at each commit, so every step had a measured before and after, rather than one number at the
  end with no way to attribute a move to a specific commit.
- **One change per run, named in the run's lever** (`EVAL_LEVER`) — this is what makes a score
  shift attributable to the change that caused it, not to whatever else moved alongside it.
- **What gets ruled on before anything is built.** The behaviour sentence, the world, the
  wordings and the claims are the code owner's to rule on, not the implementer's to infer — the
  agreed inputs and expected outcomes go into the ticket **verbatim**, and the case is built from
  that text, not from a plan paraphrasing it
  ([`docs/eval-iteration-workflow.md` §2, §8](eval-iteration-workflow.md)).

## 5 · Running

Runs go to OpenRouter with `make eval-remote`, pinned to one provider — upstreams serving one
model are not interchangeable, and one was measured blanking every tool result it returned.

```
make eval-remote
```

runs the suite's model (the roster's first `EVAL_MODELS` entry) at `make eval-remote`'s default 8
cases x 5 samples at once. To name the suite's model explicitly, scope a run to the cases a change
touches, and declare the run's lever:

```
LLM_MODEL=google/gemma-4-26b-a4b-it make eval-remote \
  EVAL_LEVER='<one-line hypothesis>' \
  EVAL_PYTEST_ARGS="<your case's test paths> -m eval -s"
```

`EVAL_WORKERS=1 make eval-remote` runs remote but serial, for debugging one case.
`LLM_MODEL=<a model EVAL_MODELS names> make eval-remote` runs a different roster entry by name — a
look that gates nothing; a PR's evidence is the suite's model.

- **Scope a run to the cases the change touches** — a whole-suite run needs the code owner's and
  the coordinator's go-ahead, never bundled into a round
  ([`docs/eval-iteration-workflow.md` §4, §8](eval-iteration-workflow.md);
  [`docs/agent-task-workflow.md` §8](agent-task-workflow.md)).
- **One run per provider at a time (the provider lock).** A pin gets only that provider's
  capacity, so two runs against the same pin draw HTTP 429s and lose samples; `make eval-remote`
  takes that provider's lock for the run's life and a second run against the same provider waits,
  printing whose run it is waiting on. Runs against different providers never wait on each other.
- **Do not run the gate or a build beside an eval on the same machine** — the two contend for the
  same resources and a run loses samples to it.
- **Measure where the defect was seen.** A behaviour observed on one model and provider is
  measured on that same model and provider, not assumed to carry over — which is why the
  suite's model is named explicitly on the command line rather than left to the roster's first
  entry: that entry can change, and a run against the wrong one answers a different question
  than the one asked.
- **Push, then measure: a number describes the tree it ran on.** The run's manifest records the
  commit, and it should equal the PR's head. A rebase that leaves the tree byte-identical to the
  measured head (`git rev-parse HEAD^{tree}` equal on both) keeps the run valid without a re-run —
  say so in a PR comment instead of re-running
  ([`CLAUDE.md` → Git Workflow](../CLAUDE.md#git-workflow)).
- **An account out of credit fails mid-run with HTTP 402.** Those log lines carry a key
  identifier and are never pasted anywhere public — the repo is public.
- **If the `make` wrapper is interrupted, the container may still finish.** Look at the run
  directory before re-running — a re-run against a run that actually completed wastes the
  provider's capacity on both.
- **Post with `make eval-report PR=<n>`** (optionally `RUN=<run-dir>` to name a specific run),
  run from the worktree that produced the run, after the PR exists. Each re-run adds a new PR
  comment — never rewrite or overwrite a prior one; the comment stream is the durable record
  ([`docs/agent-task-workflow.md` §8](agent-task-workflow.md)).

## 6 · Reading a run

**Health first.** Before any claim, check how much of the cohort actually completed, the fault
classes, and which provider answered — a degraded run is not evidence
([`docs/eval-case-design.md` §1, category C](eval-case-design.md)). A run that finished is not
automatically a healthy one to read further into if its manifest shows a different provider
answered than the one pinned.

### A case document has three sections, and an observed failure belongs in exactly one

| section | what it holds | how to recognise a failure that belongs here |
|---|---|---|
| **Assertions** | end-state facts only — where the machine landed, what the store holds, provenance | a claim's count is below what it was expected to run at (facts, never asserted as a floor — §2) |
| **Variance** | everything the model emitted on the way — tool sequence, routine shape, reply text, cost | an outlier sample, a feature's entropy rising past a proposed ceiling, a cost figure rising on an unchanged prompt |
| **Harness** | samples too broken to count at all — excluded before any claim was answered | a sample in the "Excluded samples" fold, a dead or mostly-dead cohort, a fault class in the health block |

Sort an observed failure into this table before doing anything else with it: a claim miss and a
tool-sequence outlier look alike on a quick read, but the first says Penny was wrong and the
second says Penny was merely inconsistent — and a dead sample says nothing about Penny at all,
because nothing was answered.

### Dead samples, failure causes, and the mostly-dead refusal

**A dead sample's cost is invisible.** A sample that never returns a usable result is excluded
before any claim is pooled, so whatever it was doing — however many calls, however long it ran —
is not measured as behaviour at all; it simply isn't there to read. If what's being studied is a
slow or runaway draw, establish first whether such samples complete or die: dying is a harness gap
to close (the sample needs to survive long enough to be measured) before any case built on it can
measure anything.

A sample ends up dead one of three ways: a **failed boot** that is a verdict rather than a
transient fault (an unresolvable model, refused credentials) voids it immediately; a failed boot
whose faults are all transient (a timeout, a dropped connection, a 5xx, a rate limit) retries with
a short, doubling backoff up to a fixed attempt budget before voiding it; and a sample that runs
past its **wall-clock bound** is stopped and voided regardless of why it was still running. Each
is recorded as an exclusion with a reason, never silently dropped.

Separately, every sample that *did* complete but whose claims failed is tagged with a
**failure-cause**, read structurally off its own promptlog rather than judged: **pathology** (a
known model pathology — a reroll-guard poison signal — fired, checked first, since the poison is
the root cause and anything downstream of it is a symptom), **harness** (a clean timeout, with no
pathology), or **behavioral** (everything else — the model simply got it wrong, the signal worth
chasing). A case's pathology-excluded score is the honest read of model behaviour with that noise
removed, reported beside the raw score.

**A mostly-dead cohort fails the run, whatever its survivors scored.** More than half of a case's
intended samples must have completed, because the faults that kill a sample correlate with the
work — the longest sample, the one that made the most calls, is the one most likely to draw the
bad provider or hit the rate limit — so a surviving minority is a biased draw, not a smaller
unbiased one, and its mean measures the bias rather than the case.

Then the claims, as passing counts, per case, beside a named reference run: claims, distinct tool
paths, calls per sample — never as a mean, never with a gloss
([`docs/eval-case-design.md` §1–2](eval-case-design.md)).

**Turns are the ground truth, not scores.** Open the sample's own database and read what the
model was sent, what it thought, what it called, and what came back. Confirm the mechanism under
test actually fired — count the log line it emits, check the ledger — rather than inferring it
from the claim count alone.

Where to look, concretely:

- **The per-sample database and log live in the run directory** (`EVAL_REPORT_DIR`), named by the
  sample's own number — the number every surface a reader meets it on uses, including the case
  report's sample banner: sample 11 is `<case_id>-11.db`, with its log beside it at
  `<case_id>-11.log`. Open the file the report names and no other — a report once called a sample
  `-11` while its evidence sat in `-10.db`, and a reader who followed the report's own name read a
  different sample's correct exclusion as a harness defect.
- **The `.db` is a SQLite file.** Its `promptlog` table holds, per model call, the messages sent,
  the model's thinking trace, and the response (including any tool calls and their results) — the
  same rows the case's `.md` report renders the sample's turn-by-turn fold from, so opening the
  database directly shows nothing the report doesn't, just at finer grain and in the model's own
  words rather than summarised.
- **The run directory also holds `manifest.json`** (commit, model, N, the run's lever) and
  `results.jsonl` (one record per case: claims, per-check outcomes, the representative sample's
  name, wording and reply) — read `manifest.json` first to confirm the run is the one you think it
  is before reading anything else in the directory.
- **The case document names the modal sample and lists the outliers.** Open the "Representative
  sample" fold for the modal one; open the "Outliers" fold for the samples that diverged
  consequentially, each shown as a small table of feature / this sample's value / the
  representative's value, pointing straight at which sample's `.db` to open next.
- **To count distinct tool sequences for a case**, read the "distinct" column on the "tool
  sequence" row of the case's variance table — it reports the count directly, beside the modal
  share, the entropy, and any proposed ceiling; there's no need to tally it by hand from the raw
  samples.

**Read the thinking.** The model reasons sensibly over what it is shown, so a wrong action is
usually a correct reading of a deficient presentation — the first question is what the state
failed to present, not whether the model should have known better
([`docs/principles.md` §3.1](principles.md)).

**Check the claim before blaming the model.** A failing claim can be the model doing the wrong
thing, or the claim asking a question that can't be answered true-or-false, or asserting a
phrasing — rule those out before treating a miss as a behavioural gap.

**When a model call fails, suspect exhausted re-rolls before infrastructure** — the suite re-rolls
an invalid draw from the same state, and a run that looks like it hit an outage more often hit its
own retry limit.

**Read the artifact, not a summary of it**, including a subagent's summary of it — a summary
describes what someone believes happened, and the posted report, the gate's exit line, and the
`git log` state say what did.

**Reporting a result:** the overall count; a per-case table; what worked and what did not; then
diagnoses last, each failure given as input, expected, actual, with the model's own words quoted
verbatim ([`docs/eval-iteration-workflow.md` §5](eval-iteration-workflow.md)). A number later
found wrong is corrected in the open, not quietly replaced.

**Sorting a whole-suite run** to choose what to work on: by the largest claim gap and by the
highest variance, then set aside the classes already ruled out (the exploration-bounded cases from
§3, the model-specific gaps from §7).

## 7 · Reasoning about a fix

The ladder, cheapest and most durable first: fix the data or the presentation; then the wording;
then code — never "make the model smarter"
([`docs/principles.md` §1, §3.1–3.2](principles.md)).

- **A rule keyed to a tool name, an argument position, or the failing sample's own phrasing is
  wrong by construction**, however well it fits the case that motivated it — it will not fire for
  a shape nobody enumerated ([`docs/principles.md` §2.6](principles.md)).
- **Wording is plain** — states the situation and what is permitted as well as what is not — and
  after a second patch to the same passage, the passage is rewritten whole, not given a third
  clause ([`docs/principles.md` §3.3](principles.md); the craft is in
  [`docs/prompt-writing-guide.md`](prompt-writing-guide.md)).
- **Each fix is measured on its own case against the reference run**, with a neighbouring case
  measured alongside it when the change could over-correct that neighbour. A batch of fixes ends
  with one whole-suite run, not one per fix.
- **The suite's model is the one being fixed for.** Gaps on another roster entry are noted, not
  chased — chasing them tunes the fix to a model nobody is shipping. A check whose pass depends on
  one model's phrasing is a defect in the check, not a fact about the other model.
- **An observation is not a measurement.** One recorded draw — a single sample read while
  diagnosing — is an *expectation* about what the cohort will show, not a fact about it yet. Say
  so in those words, and let the case's own cohort measure whether the expectation holds before
  treating it as settled.
