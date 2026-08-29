# Eval Case Design

How an eval case is built, and what it is allowed to claim. This is the contract for **writing
or porting one case** — read it before you touch a case file, not after.

The design was ratified in **#1994**, which carries the measurements behind every rule below.
This document is its usable form: what you decide, in what order, and what makes a check legal.
Where a rule needs its evidence, #1994 has it; it is cited rather than restated.

Read alongside:

- **[`docs/eval-iteration-workflow.md`](eval-iteration-workflow.md)** — the **loop** a round of
  eval work runs in (one beat → pairs → seeded world → run → report → ruling).
- **[`docs/agent-task-workflow.md`](agent-task-workflow.md) §4** — how a run is invoked, scoped,
  detached, and posted to its PR.
- **`penny/penny/tests/eval/utils/cohort.py`** — the arithmetic (`Claim`, `Feature`, entropy,
  proposed floors and ceilings). **`.../utils/assertions.py`** — the named claims a case makes.
  **`.../utils/worlds.py`** — what is true while the ask is answered.

---

## 1 · The premise

Run the same request many times and look at the spread.

- **Deterministically assert only the mechanical things about the machine** — where it landed and
  what it left behind.
- **Everything the model emits is variance** — reply text *and tool calls alike*. A tool call is
  model output exactly as a sentence is. Many routes reach one end state, so a route is measured,
  never asserted.

A case therefore makes **three different kinds of claim**, which must never be mixed — was Penny
**correct**, was she **stable**, and can the run be **believed at all**:

| | the claim | how it is judged |
|---|---|---|
| **A · assertions** | end state only — the four categories in §2 | a **pass-rate floor** (a fall below is a regression) |
| **B · variance** | model output — tool sequence, routine shape, names, reply text | a **one-sided ceiling** (only a *rise* is a regression) |
| **C · run health** | how much of the cohort actually ran, and what killed the rest | a **gate**: broken samples are excluded *before* pooling, and a mostly-dead cohort **fails the run** |

**C is read first even though it renders last**, and it is not a section a healthy case renders —
it is a gate plus an accounting clause (§1.2).

### 1.1 · What a case's report actually renders

A/B/C above is the *design's* decomposition. What you will read back after a run is this, in this
order — one line, then folds:

| | fold | carries |
|---|---|---|
| — | **the summary line** | both scores plus the whole sample accounting: `15 pooled + 3 control + 0 excluded = 18 driven` |
| A | **Assertions** | one table per spec category, and **a category nobody wrote a claim for renders as a visible gap** |
| B | **Variance** | per-feature entropy, proposed ceilings, the per-phrasing rows |
| B | **Cost** | input/output tokens per sample and their proposed ceilings (§5) |
| C | **Excluded samples** | **only when something was lost** — the dominant fault class, then the samples by name |
| — | **Representative sample** | the one sample you are asked to read: its prompts and its whole transcript, with a **Rejected draws** fold inside it (§5) |
| — | **Test inputs** | the ask in its K wordings, and the seeded world |
| — | **Outliers** | the samples that diverged **consequentially**, each with the feature and value that made it one |

### 1.2 · Run health — the gate under all of it

A degraded run used to look exactly like a healthy one. `6 passed, EXIT=0` was printed by a run in
which **34 of 48 samples never produced their measured turn**, every one killed by the same thing
188 times — a gateway answering HTTP 200 with an empty `choices` array until the client's retries
were gone. Another drew 325 rate limits and said nothing. Neither named a number, so the only way
to find out was to read per-sample logs by hand; and the only way to prove *which member of a
routing pool* had poisoned a run was to run the whole suite again with a pin. Historically this is
also how **429 of 5,452 harvested replies ended with an infrastructure error as the measured reply,
248 of them with reply checks scored *passed*** (#1994 §1).

So the run reports on itself, from values it counted rather than sentences it matched:

| | what it does | why it is not optional |
|---|---|---|
| **cohort accounting** | each case records how many samples it **intended** and how many produced their **measured turn** | a sample's `.db` exists from the moment the sample **starts**, so a file is not a result. One prototype run pooled 17 dead samples of 31 and it read as wild behavioural variance |
| **a fault tally by class *and* serving provider** | read from structured fields on every model-call attempt — never grepped from prose | a run can be silently poisoned by **one member of a routing pool**, and artifacts that record the model but not the provider cannot name it |
| **a verdict that fails the run** | the bar is per case: it must complete a **strict majority** of the samples it intended | a mostly-dead cohort is **not a smaller cohort — it is not a result** |

**Why a strict majority, and not "noisier with fewer samples".** Dead samples are not missing at
random: the faults that kill them **correlate with the work** — the long turn, the one that made the
most calls, the one that spent the most tokens is the one most likely to draw the bad provider or
hit the rate limit. So the survivors skew to the *short* samples, and their mean measures a
selection effect rather than the case. The bar sits where the surviving cohort can still be read as
the case. It is a judgement, not a derivation, which is why it is printed in the refusal rather than
buried in a constant — and why a run that squeaks past it still says so.

**The counts must add up.** Three unexplained samples on the surface that says whether a run can be
believed is how 288 infrastructure failures came to be booked as behavioural. That is why the
accounting rides on the summary line of every case — including the healthy ones, where it is one
clause — and the harness fold materialises only when there is something to name. A section that
renders as a stub every time is a section people learn to skip.

---

## 2 · Section A — the closed list

**Four categories. Nothing else is an assertion.**

| category | what it asserts | read off |
|---|---|---|
| `LANDED` | where the machine landed — the state transition the story is about | the conversation machine's walk |
| `STORE` | what the store holds — entries, keys, contents, record fields, registry rows | the sample's database, while it is still live |
| `PROVENANCE` | every specific value traces to what the model was **given** — user turns and tool results, **never Penny's own turns** | the sample's inputs vs its output |
| `DIRECTED_CHANGE` | perturb the world, re-run the same ask, the facts move with it | this cohort's world vs the control's |

**If a check fits none of these, it is not an assertion.** Delete it, or move it to variance.

`PROVENANCE` excludes Penny's own turns for a specific reason: a value she invents early in a turn
rides into the message history, and a later check that treats the history as source material lets
the fabrication launder itself.

`DIRECTED_CHANGE` is the claim that wording variation **cannot** make. If Penny were
pattern-completing from the shape of the request, every phrasing would name the same fact and every
phrasing would be right; only a different world can tell reading apart from completing.

### The non-negotiables

| rule | why | the evidence |
|---|---|---|
| **Never assert a route.** Assert a *property* of the routine ("it has a write step", "it names somewhere to act"), never its shape or its tool names. | Many routes reach one end state, and a skill is an arbitrary tool sequence — a name-keyed rule simply will not fire for a shape nobody enumerated. | #1993: three different tools all correctly reached the run record; the check had pinned one. |
| **A judgement call in the fixture is variance, not an assertion.** | Asserting a count asserts one reading of an ambiguous world. | Whether an appointment counts as a "signing" is genuinely ambiguous. |
| **An assertion about the store reads the WHOLE entry** — key *and* content. | A fact in the key and a blurb in the body is a perfectly good way to store it. | A prototype reported a 25/32 model failure that was entirely its own bug: it read content only. |
| **A sample `.db` exists from sample START, not completion.** Gate on completeness before pooling; file counts are not completions. | Otherwise dead samples are pooled as behaviour. | the machinery that closes this, and what it cost before it existed: §1.2 |
| **Never match a phrasing.** A reply check looking for a token you guessed in advance is the thing this design replaces. | Measurably both too strict and too loose in the same suite. | 31 replies that stated the recorded cadence correctly were failed; elsewhere an infra error string and a raw thinking leak both scored *passed* (#1994 §1). |

A claim **records, it does not raise**: `assert_*` states what the case claims and answers it for
every sample. Whether a rate is a failure is the recorded floor's job — which is what makes "run
it, read it, then lock it" possible at all.

---

## 3 · The porting checklist — run it in BOTH directions

This is the section that exists because of a real defect. The reference port was faithful to the
canonical case it came from and violated this design in three ways at once. **Run both columns,
every time, on every case.**

> ### Outward — for each check on the source case
>
> **Which of the four categories does it fall in?**
>
> **None → delete it. It does not port.** Being on the canonical case is not a reason to keep it:
> the canonical cases predate this design, so a check surviving there is evidence of nothing.
>
> ### Inward — for each of the four categories
>
> **Does the ported case make a claim in it?**
>
> **No → write one**, even though the source case had none to copy.

**The inward direction is the one nobody runs.** Outward feels like work — you are looking at a
check and deciding its fate. Inward looks like nothing is missing, because the thing that is
missing was never on the page. That is exactly what it cost: the reference port shipped **with no
store-side `DIRECTED_CHANGE` claim**, because the case it was ported from had none to copy. It
asserted that the *reply's* facts move with the world and never asserted that the *store's* do — so
a round that reads the perturbed world, says the right thing, and writes the wrong thing passes
every check in section A.

The two outward misses from the same port, as worked examples of what "fits no category" looks
like:

| check | why it is not an assertion | where it goes |
|---|---|---|
| `assert_each_page_was_read` — reads `sample.pages_read` | asserts that a **browse call happened**. That is a route. `LANDED`? No — a fetch is not a landing. `STORE`? No — nothing was stored. It is model output. | section B, inside tool sequence |
| `assert_the_reply_reports_what_was_stored` — asserts the reply text contains a token | a **phrasing match**, the thing the design exists to abolish. Its legitimate half — does the reply's content track the world? — is already `DIRECTED_CHANGE`'s positive direction. | delete: it is a duplicate wearing a regex |

Note what makes the first one seductive: it distinguishes "read the page and correctly found
nothing in scope" from "never looked", which is a real distinction worth having. It is still a
route. The end-state form of that question is a `STORE` claim about what was kept, plus the tool
sequence measured in section B — where a cohort that stopped browsing shows up as a variance rise
rather than as one sample's failed check.

### Two structural backstops — and what they do not cover

The checklist has help now, and it is worth knowing exactly how much:

- **The category is a required field from a closed enum** (`SpecCategory`, no default), so a check
  that fits none of the four **cannot be declared**. That makes the *outward* column a fact the
  code states rather than a review someone remembers to run.
- **The assertions table groups by category and renders an empty one as a visible gap**, so a
  missing `DIRECTED_CHANGE` claim shows as a hole in the report instead of as nothing at all.

Neither one writes the claim for you. A gap you can see is still a gap until the inward column is
run and something is written into it — and a claim can satisfy its enum while being the wrong claim.
The backstops make the omission *visible*; the checklist is what closes it.

### Copying the reference port

Sub-tickets say *copy the shape of `learn-demonstrated-round`*. Copy its **structure** — the
fixture, the world, the control drive, the `measure(...)` call, the parametrisation over models.
**Do not copy its claim list.** Every claim is re-derived against the checklist above, for your
case's own behaviour, or the next thirty cases inherit whichever checks happened to be on the
reference the afternoon you read it.

---

## 4 · The shape of a case

```python
@pytest.mark.parametrize("model", EVAL_MODELS)          # every case, both models — §7
async def test_<the behaviour, as a sentence>(chat_eval, model, <seed fixture>) -> None:
    cohort = await chat_eval(
        case_id="<behaviour>-<what it does>",
        model=model,
        seed=<the round this turn continues>,           # seeded, never hoped for
        world=<WORLD>,                                  # the pages, the keeps, the excludes
        ask=<THE ASK>,                                  # one request …
        also_phrased=<FOUR MORE WORDINGS>,              # … in five wordings of the SAME ask
        samples_per_phrasing=3,                         # 5 × 3 = 15 pooled
        min_pass_rate=None,                             # report-only until the owner locks it
        family=<FAMILY>,
    )
    # A · LANDED
    cohort.assert_machine_landed(ConversationState.<STATE>)
    # A · STORE
    cohort.assert_the_store_holds_an_entry()
    cohort.assert_nothing_excluded_was_stored()
    # A · PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    # A · DIRECTED_CHANGE — a SECOND VISIBLE DRIVE, beside the claim it serves.
    control = await chat_eval(..., world=<WORLD_CONTROL>, ask=<THE ASK>, samples_per_phrasing=3)
    cohort.assert_facts_moved_with_the_world(control)   # store side (gated) + reply side (reported)

    # B · what is measured, never asserted
    cohort.measure(TOOL_SEQUENCE, ROUTINE_SHAPE, CONTAINER_NAME, ENTRIES_STORED, TRANSITIONS,
                   REPLY_SPREAD)
```

The four category comments are load-bearing: they are where the **inward** pass is run. A case with
no `# A · DIRECTED_CHANGE` block is a case that skipped it.

`assert_facts_moved_with_the_world` declares **both halves** of `DIRECTED_CHANGE` — that what the
round **stored** moved with the world (`state: what it stored moved with the world`, always
decidable, so it is the one half that can carry a floor) and that the **reply's** facts moved, in
both directions (it names this world's facts, and none from the world it was not given; read out of
model prose, so reported and never floored — §8).

That second half is worth knowing the history of, because it is what this checklist is for. The
reference port shipped with the reply side only, and no store-side claim at all — not because
anyone judged it unnecessary, but because the case it was ported from had none to copy. **The
inward pass is what found it**, and it exists as a shared claim today because that pass ran. Expect
your case to have one of these: a category whose claim nobody could hand you, which only the
inward column will surface.

**A claim only one case makes stays inline in that case**, as a small local function. It graduates
into `assertions.py` at the **second** customer, not the first.

**The control is a visible drive.** The case makes it and passes it in; it is never made secretly
inside an assertion — a claim that quietly issues three more model calls is a nasty surprise. Its
samples come under the case's claims (each judged against its *own* world) but never enter the
variance pool.

---

## 5 · Section B — variance

Measured over the **pooled cohort**: normalised Shannon entropy over each feature's value
distribution, divided by `log(N)`. `0.0` is total agreement; `1.0` is every sample distinct. Reply
text is measured instead by `cosine_similarity` + `token_containment_ratio` over the pairs — two
views because they fail differently (an embedding says two replies are *about* the same thing,
which at fixed topic is nearly always true; containment says how much vocabulary they reuse).

Every feature carries a **consequence class**, declared on the feature and read at the `measure()`
site. It decides how the feature is *read*, not whether it is measured:

| class | means | features | rendering |
|---|---|---|---|
| **consequential** (the default) | a divergence implies a different end state | tool sequence, routine shape, entries stored, transitions | rendered individually in the outliers fold, with its evidence |
| **cosmetic** | a divergence is a different word for the same outcome | container name, reply text | measured, entropy reported, a ceiling proposed only where one could fire (§8), collapsed to one count line |

A feature not in that table is classified by the question, not by the list: **does a divergence here
imply a different end state?** Getting it wrong is not cosmetic in either direction — misfiling
container name as consequential once made 8 of 9 outlier rows container-name-only, reporting "60% of
samples are outliers" where the true statement was "1 of 15 diverged consequentially".

**A value that does not separate the models is a system-level finding** for the variance table,
never a per-sample outlier — repeating "this sample named the container differently" fifteen times
is how a section meant to name the samples worth looking at came to name every one of them.

**But name the right system.** Container naming was read for a long time as "unconstrained", on a
feature measuring **0.90 entropy in both models**. It is not unconstrained and never was: a
container's name is `derive_collection_name(routine name, bound values)`, a pure function, exposed
as `round_framing.container_name` precisely so a fixture cannot grow a second copy of the scheme.
Checked across all 18 samples of one run it holds exactly, and the bound-value half is **identical
in all 18**. What varied was the **routine name the framer invents** — eleven distinct names for one
routine across those eighteen samples. The container name is a pure function of it, so it carried
the framer's spread downstream and got the blame for it.

The lesson generalises past this feature, which is why it is here rather than in a ticket:

> **When a derived value is varying, name the thing it is derived FROM before calling anything
> unconstrained.**

Measuring the derived value attributes the spread to a component with no discretion, and points the
fix at code that cannot be fixed. The feature is being replaced by one that reads the routine name
directly, for exactly that reason.

**Report per-phrasing rows beside the pooled score.** Phrasings are a *coverage* mechanism, not a
variance one — model stochasticity carries essentially all the spread (~0.05 of it is phrasing; one
model produced 4.8 distinct routine shapes inside a *single* phrasing), which is what justifies
pooling at all. But the pooled number hides what phrasings are for: four phrasings measured
`H = 0.00, 0.52, 0.00, 0.00` — three perfectly deterministic, one that destabilised the case
completely — and pooled to `0.18`. The pooled number is the gate; the rows are the diagnostic
saying which wording moved it.

### Rejected draws are machinery, not output

A re-rolled draw is persisted whole in the promptlog, so a transcript built naively from it renders
every discarded attempt as a reply indistinguishable from the one the user received. **A text draw
renders as a reply only if it was delivered**; the rest go behind their own fold, labelled as
rejected draws. They are **never scored** — not by an assertion, not as a reply in the variance
spread. If your case's sample count and its rendered reply count disagree, this is the first thing
to check.

### Cost is a locked metric too

Every run already captures it; nothing used to compare it, so a prompt change that doubled the
context was invisible until someone noticed the bill.

| metric | whose it is | what a rise means |
|---|---|---|
| **input tokens** | **ours** — prompt and context design | we made the envelope bigger. This is the one a prompt edit regresses. |
| **output tokens** | the **model's** | on a fixed prompt, a model or config change |

**Per sample, never per run** — a total is not comparable across cohort sizes, the same trap the
entropy denominator is. Both carry a one-sided ceiling, per model.

---

## 6 · Cohort sizing, and the three things people confuse

**5 phrasings × 3 samples = 15 pooled, plus 3 control = 18 per case.**

| mechanism | is | pooled into variance? | serves |
|---|---|---|---|
| **phrasing** | *same world, different words* | **yes** — it is the cohort | coverage, and the pooled variance score |
| **control** | *same words, different world* | **never** | the `DIRECTED_CHANGE` assertion |
| **scenario** | a different ask reaching the same end state | no — it is a **different case** with its own 18 | its own behaviour |

Reading a control as an extra phrasing is the mistake this table exists to prevent: it folds a
deliberate difference into the spread and reports it as instability.

Why 15 and not fewer, measured by subsampling two real 32-sample cohorts:

- At N=15 the spread on entropy is ~±0.11 — *not precise*. **Separation is the job**: a variant
  cohort (modal 0.33) and a consistent one (modal 0.80) overlap **0.2%**. Destabilisation is
  caught; a 2-point wobble is not, which is the correct sensitivity for regression detection.
- **Below 15 it collapses** — at N=9 the spread is 0.33–0.43, wider than the gap between a good
  test and a bad one.

**Do not scale N with anything ambient.** A recorded ceiling is `(feature, model, N, value)`, so an
N that drifted with an environment variable would silently make every recorded threshold
incomparable.

---

## 7 · Every case runs on BOTH models

`@pytest.mark.parametrize("model", EVAL_MODELS)` is part of the case, not a nicety.

Porting or tuning against a single model bakes that model's quirks into the fixtures and the
assertions, and the suite then measures *how much like the incumbent* the next model is — the same
trap that rules out golden reference sets (§9).

**Thresholds are per-model and cannot be shared.** The two measured models differ ~3x on the same
features (routine-shape entropy 0.53 vs 0.18; tool-sequence 0.69 vs 0.19); one ceiling would be
either useless for the consistent model or permanently failing for the variant one. Where a feature
does **not** separate them — the routine name the framer invents, 0.90 on both — that is the
signature of a *system* defect rather than a model one, and it is only visible by running both.

---

## 8 · Thresholds — proposed, never silently locked

A floor or a ceiling is **proposed by a report** and **accepted by the code owner**. A ported case
lands `min_pass_rate=None` and reports its numbers.

| | recorded as | regression is |
|---|---|---|
| **floor** (section A) | `(claim, N, rate)` | the rate **fell below** it |
| **ceiling** (section B) | `(feature, model, N, value)` | the variance **rose above** it |

Both qualifiers on a ceiling are load-bearing, and a comparison across either is **refused** rather
than answered:

- **N** — normalised entropy is biased upward at small N: the same behaviour reads **0.527 at N=32
  and 0.605 at N=15**.
- **model** — see §7.

**Gated ≠ held.** A claim read out of **model prose** is *reported and not floored at this N* —
**even at full marks**. Across two runs of identical code, on the same commit and the same model, a
reply-content rate moved by **3 samples of 18** where every structural claim moved by at most 1. At
18 samples that is ±17 points: a floor tight enough to catch a real regression there would flap on
an ordinary re-run, and one loose enough not to flap would catch nothing. **A prose-read claim must
not be counted as a failure in the headline.**

A claim that does *not* hold on every sample is also not floored — for the opposite reason. The
misses are the naming work, and recording a floor underneath them would bless the defect as the
contract. The two reasons must never be blurred in the report.

**A saturated feature carries no ceiling.** The margin is a fixed +0.10, so a feature already near
the top of its range gets a ceiling with nothing above it. Measured: one read **0.761**, which the
margin turned into a proposed ceiling of **0.86** — on a statistic that tops out at 1.0. It has
almost nowhere left to rise, so that ceiling could never fire, and printing it implies a guard that
does not exist. **A threshold that cannot be crossed is worse than no threshold, because it reads as
protection.** So report the value, propose no ceiling, and **say why**. A feature up there is a
**diagnostic reading, not a gate** — and it is usually a defect to fix rather than a number to
record.

---

## 9 · Consistency is not correctness

A change could make Penny uniformly wrong and variance would **drop**. Three mechanisms, none
substitutable for another:

| mechanism | catches |
|---|---|
| **variance ceilings** | destabilisation |
| **assertions** | wrongness |
| **a human reading ONE sample at consistency** | wrong-but-stable — and the reading is sound *precisely because* the cohort agrees |

This is why the report names one sample **modal** and hands it to the reader rather than making
them choose.

### Alternatives measured and rejected — do not re-propose without new evidence

| rejected | why | measured |
|---|---|---|
| **embedding similarity to reference replies** | embeddings encode *topic*, and every discrimination here is at fixed topic | a topic-matched INVALID reply scored **0.696** against a valid one, while two valid replies scored **0.614** against each other |
| **golden / snapshot reference sets** | they encode the incumbent model's voice, so a model swap measures incumbent-likeness rather than validity | — |
| **a judge model** | relocates the problem: the judge then needs validating, and its verdict is an opinion rather than an assertion | — |
| **hand-written tool lexicons / verb lists** | the same fragility in a new costume | — |

---

## 10 · Porting does not fix Penny

**The port surfaces real defects. Record them, do not chase them.** A defect found while porting
gets its own ticket and is left alone.

The reason is not discipline for its own sake: fixing while porting measures the baseline against a
moving target, so when a number moves you cannot tell whether the *behaviour* changed or the
*measurement* did — and the bedrock ends up built out of both at once. Fixing them is what the
substrate is **for**, and it starts after the baseline is accepted.

Two other scope rules that come from the same place:

- **One representative case per behaviour**, with K phrasings of that same ask. Do **not** explode
  existing per-behaviour variants into K phrasings each. A dropped variant is recorded so it can
  return deliberately — each one probes a named temptation, so re-read its docstring rather than
  reinventing the reason it existed.
- **Completion is not "everything is ported."** It is **two consecutive full-suite runs, on both
  models, whose numbers agree.** One run gives thresholds that might be that run's noise.

---

## 11 · Where this lives in code

| file | holds |
|---|---|
| `penny/penny/tests/eval/utils/cohort.py` | the arithmetic — `SampleObservation`, `Claim`, `SpecCategory` (the closed four), `Feature` + `Consequence`, `normalised_entropy`, `pool`, `proposed_floor`, `proposed_ceiling`, `compare_to_ceiling`, the standings |
| `penny/penny/tests/eval/utils/assertions.py` | `Cohort` and the named claims a case makes against it |
| `penny/penny/tests/eval/utils/worlds.py` | `World` — the pages, the `keeps` token set per source, the `excludes` |
| `penny/penny/tests/eval/utils/run_health.py` | cohort accounting, the fault tally by class and provider, and the viability verdict — its module docstring is the fullest statement of the problem |
| `penny/penny/tests/eval/utils/report.py` | the case document — it renders and never computes |
| `penny/penny/tests/eval/conftest.py` | the `chat_eval` driver: `ask` / `also_phrased` / `world` / `seed` / `samples_per_phrasing` |

A `World`'s `keeps` is one token set **per source** — tokens appearing only on that page, so a
stored copy says which page it came from and an invented one matches neither. They identify the
**source**; they are **not** a list of what the ask puts in scope. A round told to collect trades
that reads a page whose only item is an appointment has correctly found nothing in scope, and
asserting that page's names would fail it.
