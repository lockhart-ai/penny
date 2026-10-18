# The Eval Working Guide

How to plan a case or a change, run it, read what came back, and decide what to fix — the
working method, distilled from what has been learned running the suite.

**This doc does not restate the contract.** [`docs/principles.md`](principles.md) is Penny's
north star — the rules a mechanism or a test is designed by. [`docs/eval-case-design.md`](eval-case-design.md)
is the contract one case is written or ported to: what an assertion is, the two categories,
the porting checklist, cohort sizing. [`docs/eval-iteration-workflow.md`](eval-iteration-workflow.md)
is the loop a round of model-facing work runs in: beat → pairs → seeded world → run → report →
ruling. Where this guide and one of those overlap, it gives the point one line and a link, then
adds what working the suite has taught beyond it — a measured number, a cause, a fix, the
concrete command or file to look at. Read this guide after those three, as the field notes for
using them.

Who it's for: anyone planning a new case, changing a prompt or tool surface and needing to
measure it, running the suite, or reading a run to decide what to do next.

---

## 1 · From a failure to a case to a fix

Penny's behaviour is managed through the suite. A failure seen in use becomes a case that
reproduces it; the fix is whatever makes that case's claims hold, its variance fall and its
token cost fall, without costing its neighbours. A fix with no case is unmeasured. A case whose
first run does not show the failure does not reproduce it.

This is **not** the same rule as porting an existing case (§2 below): a port lands the case as it
is and files any gap it shows for later ([`docs/eval-case-design.md` §10](eval-case-design.md)). A
failure-driven case carries its fix in the same issue and the same PR, proved by two runs — the one
that reproduces the failure, the one that shows it resolved.

The path, in order. Each step says who does it, what it produces, and where the detail lives:

1. **Observe** (the code owner). He says what went wrong, at whatever level he likes.
2. **Diagnose** (the coordinator, reading only — wide reading goes to a read-only kitten). Find the
   turn or cycle and read it off the runtime log and the prompt log. §5 "Distilling a failure"
   below.
3. **Classify the failure** (the coordinator) — a wrong decision on what the model was shown, a
   sensible decision on a wrong or missing input, or a deterministic defect below the model —
   because it decides what gets built. §5 below.
4. **Distil** (the coordinator): the user's goal, the correct end state as facts, the entry
   condition, the smallest input that still shows the failure. §5 below.
5. **Check what already covers it** (the coordinator), against `docs/eval-catalogue.md`. §5 below.
6. **Reproduce, then genericise** (the coordinator), replaying the recorded input before inventing
   values of the same shape. §5 below.
7. **Propose the case** (the coordinator) to the code owner, one `ask` card, using the form in §6
   "The case proposal" below — the behaviour sentence, the world, the wordings and the claims are
   his to rule on.
8. **Write the case and open the draft PR** (a kitten). The agreed proposal goes into the issue
   verbatim; the kitten builds the world and wordings and opens one draft PR for the issue
   ([`docs/agent-task-workflow.md`](agent-task-workflow.md); §8 "Building the world and the five
   wordings" below).
9. **Run it to show it reproduces** (a kitten), scoped, on the suite's model, at the case's commit
   — the "before" — posted to the PR (§9 "Running", §10 "Reading a run" below). It reproduces the
   failure when any of the three measures shows it: a claim fails, the spread is wide, or the
   token cost is high — whether the sample times out or finishes is not the point. Only when all
   three read healthy does the case not reproduce it: back to step 4.
10. **Push the fix to the same PR** (a kitten), one lever per round — presentation before wording
    before code (§11 "Reasoning about a fix" below).
11. **Run it again to show it resolves** (a kitten), at the new head, with the neighbours the
    change could disturb — the "after" — posted to the same PR as a new comment, compared against
    the "before" on all three measures. Further rounds repeat steps 10 and 11.
12. **Review and merge** (the coordinator, then the code owner). The PR adds an eval case, so the
    code owner reviews it; a batch of fixes still ends with one whole-suite run, asked of the code
    owner first.

**One issue, one PR.** The code owner: *"i only want a single pr … write the test, create the pr,
run it to show it reproduces the failure, then push the fix, run it again to show it resolves it,
then we have both runs on the pr."* The case and its fix live together; the two posted runs, each
at the commit it measured, are the evidence. Each run's `EVAL_LEVER` says which it is — for example
`reproduces: <the failure>` and `fix: <the lever>`.

## 2 · What the suite is for

The suite is the list of behaviours Penny supports — `docs/eval-catalogue.md`, generated from the
cases themselves. One case is one behaviour under one entry condition, in five wordings, three
samples each: a cohort of fifteen ([`docs/eval-case-design.md` §4, §6](eval-case-design.md)).

- **A case is read on three measures, never fewer.** The code owner:

  > "the measures for an eval test are
  > - checks passing
  > - variance
  > - token cost
  >
  > that lets us reproduce failure cases like the example here. we can do deterministic checks
  > that the facts that were present are returned. we can measure variance. we can measure cost.
  > we have all the tools to reproduce this kind of failure."

  Checks are the claims, strictly true or false (§3 below). Variance is everything the model
  emitted on the way — tool calls, routes, wording (§4 below). Token cost is what it spent
  getting there — input, output, reasoning tokens, and duration (§4's "Token cost, the third
  measure" below). The three answer different questions and are read together, never
  collapsed into one.
- **Facts are asserted; behaviour is measured.** A claim is a fact about the end state. How the
  model got there — which tools, in what order, in what words — is measured as variance and
  never asserted. This is the whole of [`docs/principles.md` §4.1](principles.md).
- **Claims are counted and reported, never gated, and carry no floor.** A well-formed case
  scoring 4 of 15 is a finding about Penny, not about the case — but what happens next depends on
  how the case got here, and the two must not be confused:
  - **Porting an existing case.** The case lands as it is; the gap is filed and the fix comes
    after it, never folded into the port
    ([`docs/eval-case-design.md` §1, §10](eval-case-design.md)).
  - **A case written to reproduce a failure found in use.** The case and its fix share one issue
    and one PR: the run that reproduces the failure and the run that resolves it are both posted
    to that PR (§1 above).
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

## 3 · What a check is

A check is strictly true or false of one sample — no thresholds, no "mostly." There are exactly
two kinds, and nothing else is an assertion
([`docs/eval-case-design.md` §2](eval-case-design.md)):

| kind | reads off | what it asserts |
|---|---|---|
| **store** | the sample's own database, live | what the store holds afterwards — the row or entries as the turn left them, that everything the ask did not name is preserved, and what reached the user |
| **provenance** | the sample's inputs vs. its output | every specific value traces to what the round was given, read in both directions: nothing supplied is omitted, nothing is invented |

- **Assert what survives, never that the model refrained.** No "did not call X," no "wrote
  nothing." A turn that should do nothing is claimed as: the store as it was,
  nothing invented ([`docs/eval-case-design.md` §2 "Assert what survives"](eval-case-design.md)).
- **Never assert a route.** A tool name, an argument choice, a call count, or reply wording is
  *how*, not *what* — read in the tool-sequence and reply-spread variance features instead
  (§4 below).
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
sample that ran long is read through the cost section (§4) and the run's own step cap, never
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

## 4 · Variance, and what it tells you

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

### Token cost, the third measure

How much the model spent before it answered stands beside checks and variance as its own
measure (§2 above) — not a kind of variance, and never a check that gates a run (the budget rule,
§3 above). A case can read as a clean pass on checks, with ordinary variance, and still cost far
more than its neighbours — that rise is what this measure is for. What's recorded today, and
where each number shows up:

- **Duration and call count are recorded per sample.** Every sample's own banner carries its
  wall-clock duration and its call count, read straight off the real promptlog's `duration_ms`
  per call — the same numbers production logs, not a harness stopwatch.
- **A model call that is cancelled before it returns writes no promptlog row today**
  (#2355), so a draw cut off by a timeout is missing from the token figures, and its cost
  shows only in the sample's duration and log until that is fixed.
- **Tokens and reasoning share are a per-case average, in the case document's own Cost
  section** — beside Assertions and Variance, not inside either. Input tokens are split from
  output: input is the prompt and context Penny's own design controls, so a rise there is what
  a prompt edit regresses; output is the model's, so a rise on an unchanged prompt is a model or
  config change. Reasoning tokens (the provider's own count, else estimated from the
  thinking-vs-output character ratio, named as such) render as a share of output. Both carry a
  one-sided proposed ceiling, per model, with a round +10% margin rather than a measured one —
  cost isn't an entropy reading.
- **The run as a whole states its cost too.** The assembled run summary carries a
  `cost / sample` row in its identity table and a token total for the run (input, output, and
  thinking where it is counted), and each case's row in the summary table carries its own input
  and output token figures. All of it is tokens, calls and seconds.
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

## 5 · Distilling a failure

This is the "how" behind steps 2–6 of §1's path: turning an observed failure into the smallest
thing a case can be built from.

**Find the turn or cycle.** Start with the runtime log (`data/penny/logs/penny.log`) for when it
happened, then the `promptlog` rows of that run (`penny/penny/database/models.py`'s `PromptLog`):
`agent_name` says which agent or micro-context made the call, `run_id` groups every call of one
agentic-loop invocation, and `messages`/`thinking`/`response` carry exactly what the model was
shown, thought, called, and got back. **There is no conversation machine any more**: a chat turn
runs the same single instruction every time, so there is no state transition to read for it.
**The live database is read only with the code owner's
go-ahead** — diagnosis otherwise reads the runtime log and, once it exists, a run's own per-sample
database, never the production DB without asking first.

**Classify the failure**, because the classification decides what gets built:

- **The model decided wrongly on what it was shown** → an eval case at the layer where that
  decision was made (chat turn, collector cycle, a micro-context).
- **The model decided sensibly on a wrong or missing input** → still an eval case — the end state
  was wrong — but the fix will be to presentation, not to the model's reasoning.
- **A deterministic defect below the model** (a parser, a gate, a bound) → a mock test in
  `make check`, not an eval case. Add an eval case too only if the same defect also showed up as a
  wrong end state or as variance.
- **The model gets lost.** It reasons at great length or loops, and answers late, wrongly, or not
  at all. The code owner: *"i expect a test that reproduces the world and the user ask and for the
  model to get lost in a similar death spiral. whether it times out or finishes is not the point,
  the point is that the token cost and variance will be high. that is the measurement. when the
  fix is applied the model should reply quickly and coherently."* This is still an eval case that
  reproduces the world and the ask: deterministic checks that the facts present in the world are
  returned, token cost and variance read as the measurement of the spiral itself, and the same
  case, re-run after the fix, showing a quick, coherent reply.

**Distil.** Write down, in this order: the user's goal in one sentence; the correct end state as
facts — where the machine lands, what the store holds, what the reply must and must not carry; the
entry condition — what the store, the machine and the world held when the turn began; and the
smallest input that still shows the failure.

**Check what already covers it.** Look the behaviour up in `docs/eval-catalogue.md` (`make
catalogue` generates it — gitignored, so it isn't sitting in a fresh checkout), by area and
by the machine edge it moves (§2's table groups cases by area; each case's row names its first
wording and, where it moves one, its edge). If an existing case covers the behaviour and passes,
say plainly what about the real input its world does not exercise — a size, a shape, a second
source, an entry already present. Then apply the rule: if the real input changes what a correct
sample is, or changes the set of claims a case can make, it is a **new** case and the existing one
is left exactly as it is; otherwise the existing case's world is wrong, and it is corrected in
place.

**Reproduce, then genericise.** Replay the recorded input on the suite's model to confirm the
failure is not a one-off — `penny/penny/tests/eval/utils/replay.py` reads a promptlog row **by
id** and replays its exact messages:

```
python -m penny.tests.eval.utils.replay --id <promptlog id> --samples 12
```

(run inside the penny container; a non-collector prompt, including every chat turn, is replayed
verbatim automatically). Once it reproduces, genericise: replace every real value with an invented
one of the same shape, and confirm it still reproduces before it goes into a fixture. If
genericising kills the repro, the trigger was in the specifics — narrow back to what it actually
was.

## 6 · The case proposal

What goes to the code owner, in one `ask` card, once a failure is distilled (§5 above) — the form
is short, and every line is filled or marked empty with the reason it's empty:

- **Seen:** what happened, in one or two lines, genericised.
- **Behaviour:** `"In the <locus>, when <X>, Penny <Y>."`
- **Layer and driver; area; edge** (if it moves the machine).
- **Covered today?** The nearest existing cases, and why they don't cover it.
- **World:** the sources and what each says; what the store and the machine hold at the start; the
  invented anchors; a line on how its shape and size match the real input.
- **Wordings:** the five, and the facts every one of them carries.
- **Landed / Store / Provenance:** the claims under each, or "empty" with the reason.
- **Measured:** the features, including any the case declares itself, and the token cost
  (duration, input/output/reasoning tokens) alongside them.
- **Expected on the first run, on the three measures:** which claims should fail, what should
  spread, and what the token cost should look like (high and why) — and whether any of that rests
  on a measurement or on one observation.
- **Model and provider.**
- **Rulings needed:** only real open questions, each with a recommendation.

**A filled example**, for an invented failure: asked to watch a listing on a second page, Penny
confirmed the job against the page named by an earlier, already-called-off round in the same
conversation, not the page the current ask named.

- **Seen:** the confirmation after a cold watch-ask named the wrong page — one from a round the
  user had already called off earlier in the same conversation.
- **Behaviour:** In the chat agent, when a cold ask names a new page to watch and an earlier round
  in the same conversation named a different page that was called off, Penny confirms the job
  against the page the current ask named.
- **Layer and driver; area; edge:** chat turn, `chat_eval`; Conversation machine; idle → apply.
- **Covered today?** `transition-idle-to-apply` drives a cold ask with no prior round in the
  conversation; it never exercises a conversation that already carries a called-off round naming
  a different page.
- **World:** two ordinary listing pages (the called-off one, the new one) and the empty
  collection the job writes to; the store holds nothing from the called-off round, since it was
  never applied; the called-off round is seeded as the conversation's own threaded history, not an
  invented row. Invented anchors: each page's URL, the job's watched-field token. Shape and size
  match the idle→apply cases already in the suite — two real-shaped listing pages and one prior
  closed round, not a trimmed single-turn stand-in.
- **Wordings:** five phrasings of the new ask ("watch this one instead", "actually, track this
  listing", …), each stating the new page's URL and the watch intent; none mentions the earlier
  page.
- **Landed / Store / Provenance:** Landed — apply. Store — one job created, bound to the new
  page's container; nothing left over from the called-off round. Provenance — the confirmation
  names the new page's token; the called-off page's token appears nowhere in the reply.
- **Measured:** tool sequence; token cost (duration, input/output/reasoning tokens) — ordinary
  for this case's shape, since nothing about naming the wrong page should cost more to reach.
- **Expected on the first run, on the three measures:** the provenance claim fails on most
  samples, carrying the called-off page's token instead of the new one; tool sequence stays
  ordinary; token cost stays ordinary too — this is a wrong-answer failure, not a death-spiral one,
  and this rests on the one observed sample, not yet a measured rate.
- **Model and provider:** the suite's model, gemma, pinned to Cloudflare.
- **Rulings needed:** whether "names the called-off page" belongs as a provenance claim (the
  ask's own page is a fact the reply either states correctly or doesn't) or stays a case-local
  measured feature — recommend the provenance claim, since the world settles which page is
  correct.

**What the coordinator settles itself, and what it asks.** High-level guidance only works if the
coordinator fills in everything the docs already decide:

- **Settled without asking:** five wordings by three samples; the suite's model; the layer (where
  the wrong decision was made); that claims are facts only and routes are measured; where the file
  goes; that values are invented.
- **Asked, in one card:** whether the behaviour sentence is the behaviour wanted; any claim that
  encodes a product judgement; new case or existing one, when the rule above doesn't settle it;
  anything about the world that changes what "correct" means.

## 7 · Planning a case or a change

- **Start from the user's goal and write the behaviour as one sentence:** *"In the <locus>, when
  <X>, Penny <Y>."* This is `behaviour=` on the driver call
  ([`docs/eval-case-design.md` §4](eval-case-design.md)) — not a label for the report, the
  sentence a reader checks the case against.
- **Decide the layer** — a whole turn, one micro-context, a collector cycle — and the area. The
  layer picks the driver (`chat_eval`, `collector_cycles_eval`, `end_to_end_eval`,
  `extractor_eval`, `notify_composer_eval` — `penny/penny/tests/eval/conftest.py`); it isn't
  declared separately.
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
- **Go through the two kinds of check (§3) and say what is claimed under each, or that it is
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

## 8 · Building the world and the five wordings

This is the "how" the plan above assumes: once the behaviour sentence, the layer and the
claims are settled, here is how to build the fixture — the world a case's turn answers
against, and the five wordings that pool into its cohort — for a chat case, a micro-context
case, and a collector case, each with one small worked example. The shape of the call that
takes them is [`docs/eval-case-design.md` §4](eval-case-design.md); cohort sizing and the
phrasing-versus-scenario line are [`docs/eval-case-design.md` §6](eval-case-design.md);
genericising a real page or wording into a synthetic one is [`docs/eval-iteration-workflow.md`
§3](eval-iteration-workflow.md). None of that is restated here.

### The world

A world is a `World` (`penny/penny/tests/eval/utils/worlds.py`): the pages a browse can
reach (`pages`, each a `CannedPage`), the collections already sitting in the store
(`stores`, each a `SynthCollection`), the mailbox (`mailbox`, `CannedEmail`), what a correct
sample must keep from each (`keeps`, one token tuple per source), what must never be kept
from any of them (`excludes`), and what the reply owes (`answers`). `World.sources` walks
pages, then stores, then mailbox as one list, so a page and a seeded collection render in
the same table — `World.render()` — and a claim never has two different grounds to check
against. One case declares one world; its samples share it and nothing else varies the
ground.

1. **Decide what the case's turn reads.** A chat or micro-context case usually needs only
   `pages`; a case about what Penny already knows needs `stores`; an email case adds
   `mailbox`. A collector case has its own page-and-entry-condition shape — below.
2. **Write the source(s), at the shape and size of the real input** — already in §7 above: a
   `CannedPage(match=..., text=...)` (matched by a lowercase substring of the request URL,
   `conftest.py`'s `install_browse`) reads like a real page — a `Title:` line, the URL in the
   visible body — transcribed from a recorded input's structure, not a five-link stand-in for
   a ninety-link page. A `SynthCollection` is `(name, description, entries)`, each entry
   written `"<key> — <content>"` (`SYNTH_KEY_SEPARATOR`) the way a real one is stored.
3. **Invent values that are checkable.** Every fact a claim reads is a token invented for
   this case, with no whitespace, appearing in exactly one place — so finding it in the store
   or the reply is evidence it came from there and nowhere else. `keeps` names a token **per
   source** (what identifies that source, not what the ask puts in scope); `excludes` names
   tokens that must appear on no source the case keeps; `answers` names what the reply itself
   must state. All three are declared on the `World`, and an empty tuple is a report — "this
   ask has nothing to keep" — not an omission.
4. **A source that fails is part of the world, not an exception to it.** `CannedPage(fails=
   True)` makes a matched read raise inside the browse tool; `ALL_BROWSES_FAIL` (`match=""`,
   `fails=True`) makes every source unreachable — the "browsed a lot, read nothing usable"
   world. Install either the way any page is installed, as a source on the `World`.
5. **Seed anything that must already be true through the store's own write path, never a
   hand-inserted row.** A case passes a `seed` — a `Seeder` (`Callable[[Database], None]`,
   declared in `tests/eval/conftest.py`) — built by the per-area world modules (`utils/worlds.py`,
   `dispatch_world.py`, `lookup_world.py`, `memory_world.py`) that write an entry through
   `memory.write(...)` or stand a job up through `db.memories.create_collection(...)`. A
   hand-built row gives the runtime something to compare against that no real turn could
   have produced.
6. **Make the seed prove itself before the turn runs**, with a loud assertion of the world it
   just built — `probe_seeded_world` in `chat/idle/test_chat_memory_stories.py` and
   `_assert_the_watch_world` in `collector/test_watch_cycles.py` are the pattern. A seed that
   has drifted from its own description fails here, never as a puzzling score after a paid
   run. Read any time-dependent fact off the sample's own `PinnedClock` (`utils/clock.py`)
   and `sample.timezone`, never off wall time.
7. **A behaviour's opposite direction is the same wordings over a different world.** Where
   the world the case needs for "fires" is not the world it needs for "does not fire" — e.g.
   a collection that already holds a matching entry versus one that holds nothing — that is
   two `World`s and two cases (§3's "the world carries the condition, not a clause in the
   ask"), never one case whose ask changes to flip the outcome.

### The five wordings

Once there is a world, write the ask (or instruction) in five wordings — the cohort's arms.
What varies is register, length, word order, and the user's own synonyms: terse against
chatty, "can you" against "could you", "put it on my list" against "stick it on my list".
What must never vary is the facts: every wording states every fact a claim below depends on,
because the claims are answered pooling all five, and a wording that drops a fact measures a
different, easier behaviour on that one arm and calls the result instability.

- **Declare the facts beside the wordings**, as a small case-local function run over `(ask,
  *also_phrased)` — see `assert_every_wording_carries_the_value` in
  `chat/apply/test_missing_value_arrives.py` or `_probe_unspoken` in
  `chat/idle/test_chat_memory_stories.py`. `make check` runs it and fails when a wording drops
  a fact — either because the fact is spelled the same in all five, or because it survives
  being reworded (declared as alternatives the function checks for).
- **A wording never borrows the phrasing of the prompt under test**, never names a tool, and
  never states the expected outcome — a wording that nudges the model toward the exact words
  the system prompt uses, or asks "how many are left" when the claim only needs "in stock",
  is lending the case's own answer to the thing it is measuring.
- **If a rewording changes what a correct sample is, it is a different case**, not a sixth
  wording — the phrasing-versus-scenario line,
  [`docs/eval-case-design.md` §6](eval-case-design.md).
- **Where the wordings come from depends on the layer:**
  - **Chat**: five ways a *user* would actually say the ask —
    `ask=` plus `also_phrased=` on `chat_eval`.
  - **A micro-context** (the browse extractor): the arm is
    **Penny's own text**, written upstream by the calling draw — the `extract` argument of a
    browse call. Five wordings are five ways that calling draw might have worded the same
    request, sampled from real upstream phrasing, never invented for the case —
    `instruction=` plus `also_instructed=` on `extractor_eval`.
  - **A collector**: there is no user turn, so the natural language is the `prompt` the job
    was configured with — free text the model wrote directly via `collection_configure`, run
    verbatim with no rendering seam — plus the prose of the page that answers it. `arms=` on
    `collector_cycles_eval` takes a `CycleArm` per wording, each carrying its own `seed`,
    `pages` and `world` — five wordings of one instruction, over five prose variants of one
    page, with the watched fact byte-identical across all five.
- **Three samples per wording, pooled: fifteen** (`samples_per_phrasing=3` on every driver).
  One wording failing where the other four pass is a finding about how that one wording
  reads, visible in the per-sample record — not grounds to drop it.

### Checking it before a run

`make catalogue` regenerates `docs/eval-catalogue.md` (gitignored, never committed) from every
declared case for a local look; `make check` runs the fixture-coherence probes (an anchor
really is on the page it claims, exactly once — e.g.
`test_each_extract_field_case_coheres_with_the_page_it_claims` in `test_eval_harness.py`), the
seed probes above, and the same declaration checks the catalogue's own collection runs — a
bad area, a bad edge, or a shared id — with no committed file to go stale. A case that passes
`make check` has a coherent world and five wordings before a single sample is run against a
live model.

### Three small worked examples

**A chat case** — `memory-look-up-and-save`
(`chat/idle/test_chat_memory_stories.py`). The world is one search-and-detail page pair plus
one empty collection the ask names; the subject is invented so the model cannot answer from
what it already knows:

```python
world=World(name="mistforge", pages=MULTIHOP_PAGES, keeps=(), excludes=(), stores=(_GAMES,)),
ask="can you look up Mistforge Tactics, read up on it, and save it to my games list?",
also_phrased=(
    "could you find out about Mistforge Tactics and put it on my games list?",
    "have a look at Mistforge Tactics, read up on it, and add it to my games list",
    "go read about Mistforge Tactics and save what you find to my games list",
    "look Mistforge Tactics up for me and stick it on my games list",
),
```

`keeps` and `excludes` are empty by report: the two pages share the subject's own name, so no
token picks out one over the other, and the ask rules nothing out. The constant fact every
wording carries is the subject's name and the one list to write it to; the claims
(`_wrote_into`, `_stored_the_subject`) read exactly those two things off the store afterward.

**A micro-context case** — `extract-fields-all-present`
(`extractor/test_browse_extract_fields.py`). One fixed listing page, read against an
instruction naming three fields, in five wordings of that instruction (none of which hints at
*how* to render the stock field, which would hand the model the anchor the claim reads):

```python
instruction="the item's name, its price and whether it is in stock",
also_instructed=(
    "what the item is called, what it costs, and whether it is available",
    "the product name, the price, and the stock status",
    "for the item on this page: its name, its price, and whether it can be bought",
    "pull out the name of the item, how much it costs, and whether it is in stock",
),
```

The world here is implicit — `extractor_eval` builds it from `url=`/`page=` as one
`CannedPage`, since a micro-context's whole ground is the one page it reads. The anchors
(`_ITEM_ANCHOR`, `_PRICE_ANCHOR`, `_STOCK_ANCHOR`) are the smallest unique spans the page
actually carries, held against the page itself by the `make check` coherence probe above.

**A collector case** — `watch-writes-the-first-reading`
(`collector/test_watch_cycles.py`). Five `Reading`s — one instruction wording plus one prose
variant of the same page each — built into five `CycleArm`s, with the entry condition (an
empty collection) and the facts (one url, one price) constant across all five:

```python
READINGS = (
    Reading("current-price", "the current price", _HEAD + …),
    Reading("asking-price", "the asking price on the page", _HEAD + …),
    …
)

def _arm(case: WatchCase, reading: Reading) -> CycleArm:
    return CycleArm(
        text=reading.extract,
        seed=_seeder(case, reading),
        pages=[_page(case, reading)],
        world=_world(case, reading),
    )
```

Each arm's own `_seeder` stands the job up through `db.memories.create_collection`, writing
the wording straight into `extraction_prompt` — no skill, no rendering seam — under the key
the instruction writes to, then asserts the whole world (`_assert_the_watch_world`) before
the cycle runs — including that this arm's wording really did reach the stored prompt. The
page's watched line (`_DATUM`) is byte-identical across all five prose variants, which is
what lets the case's claims name `499` rather than just "a plausible number."

## 9 · Running

Runs go to OpenRouter with `make eval-remote`, pinned to one provider — upstreams serving one
model are not interchangeable, and one was measured blanking every tool result it returned.

```
make eval-remote
```

runs the roster's first `EVAL_MODELS` entry at the default 8 cases x 5 samples at once. That entry
is whatever the primary checkout's `.env` lists first, which may not be the suite's model, so name
the suite's model explicitly, scope the run to the cases a change touches, and declare its lever:

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

## 10 · Reading a run

**Health first.** Before any claim, check how much of the cohort actually completed, the fault
classes, and which provider answered — a degraded run is not evidence
([`docs/eval-case-design.md` §1, category C](eval-case-design.md)). A run that finished is not
automatically a healthy one to read further into if its manifest shows a different provider
answered than the one pinned.

### A case document has three sections, and an observed failure belongs in exactly one

| section | what it holds | how to recognise a failure that belongs here |
|---|---|---|
| **Assertions** | end-state facts only — what the store holds or survived, provenance | a claim's count is below what it was expected to run at (facts, never asserted as a floor — §3) |
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
§4, the model-specific gaps from §11).

## 11 · Reasoning about a fix

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
- **Each fix is measured on its own case against the "before" run, on all three measures** —
  checks, variance, and token cost — not on whichever one motivated the fix; a presentation fix
  for a wrong answer can still move cost, and a death-spiral fix is judged on cost and variance
  falling as much as on the reply arriving at all. A neighbouring case is measured alongside it
  when the change could over-correct that neighbour. A batch of fixes ends with one whole-suite
  run, not one per fix.
- **The suite's model is the one being fixed for.** Gaps on another roster entry are noted, not
  chased — chasing them tunes the fix to a model nobody is shipping. A check whose pass depends on
  one model's phrasing is a defect in the check, not a fact about the other model.
- **An observation is not a measurement.** One recorded draw — a single sample read while
  diagnosing — is an *expectation* about what the cohort will show, not a fact about it yet. Say
  so in those words, and let the case's own cohort measure whether the expectation holds before
  treating it as settled.
