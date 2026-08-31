# Eval Case Design

How an eval case is built, and what it is allowed to claim. This is the contract for **writing
or porting one case** — read it before you touch a case file, not after.

**#1994** carries the measurements behind every rule below. This document is the usable form: what
you decide, in what order, and what makes a check legal. Where a rule needs its evidence, #1994 has
it; it is cited rather than restated.

Read alongside:

- **[`docs/eval-iteration-workflow.md`](eval-iteration-workflow.md)** — the **loop** a round of
  eval work runs in (one beat → pairs → seeded world → run → report → ruling).
- **[`docs/agent-task-workflow.md`](agent-task-workflow.md) §4** — how a run is invoked, scoped,
  detached, and posted to its PR.
- **`penny/penny/tests/eval/utils/cohort.py`** — the arithmetic (`Claim`, `Feature`, entropy,
  proposed ceilings). **`.../utils/assertions.py`** — the named claims a case makes.
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
| **A · assertions** | end state only — the three categories in §2 | **counted, never gated** — total checks passed of total checked, reported for a person to read |
| **B · variance** | model output — tool sequence, routine shape, names, reply text | a **one-sided ceiling** (only a *rise* is a regression) |
| **C · run health** | how much of the cohort actually ran, and what killed the rest | a **gate**: broken samples are excluded *before* pooling, and a mostly-dead cohort **fails the run** |

**C is read first, and you write nothing for it** — the harness gates completeness itself. Your one
stake in it is the non-negotiable in §2: a sample's `.db` exists from sample **start**, so a file is
not a result. Why a strict majority of intended samples is the bar is `run_health.py`'s module
docstring, which is its fullest statement.

**Nothing on the assertion side fails a run automatically.** Assertions are expected to run at 100%,
so a floor under them adds nothing: the count is reported and a person reads it. What still fails a
run on its own is **run health**, which refuses a dead cohort, and a **variance ceiling**, which
catches a rise.

## 2 · Section A — what an assertion is

**An assertion is a statement expected to be strictly true of the run.** At the end of a run each
one is **either true or false of that sample, and those are counted**. Everything else in this
section is a consequence of that sentence.

| consequence | |
|---|---|
| **There is no third outcome** | not "not applicable", not "unexercised", not "skipped". A third state makes a printed rate mean something other than what it says: `14/14` needs a gloss to read, `14/15` does not. |
| **A claim answers its own sentence** | where a precondition is absent the sentence is usually **false**, not unasked — *"the demonstrated write landed in the round's container"* is false when nothing was written, because no write landed there. Returning early on an absent precondition answers a different question from the one the label states. |
| **Claims do not partition blame** | they are independent statements about end state, so a sample that did nothing fails every claim about what it should have done. That is several unmet contracts, not one failure counted several times — and suppressing the repeats to tidy a report trades a check's truth for its presentation. |
| **Unanswerable is not an assertion** | a statement that cannot be answered true-or-false of a run is not one. Rewrite the sentence so it can be, or it belongs in section B. |

A sample the run-health gate excluded is **not** a third outcome: it left the population before any
claim was answered.

And one more, about *who* judges the count rather than what is counted: a claim **records, it does
not raise**. `assert_*` states the case's claim and answers it for every sample, and the run reports
the total — it never goes red on a miss and never stops a run. **Whether a number is a failure is a
person's call**, made against the reported count.

### The closed list

**Three categories. Nothing else is an assertion.**

| category | what it asserts | read off |
|---|---|---|
| `LANDED` | where the machine landed — the state transition the story is about | the conversation machine's walk |
| `STORE` | what the store holds — entries, keys, contents, record fields, registry rows | the sample's database, while it is still live |
| `PROVENANCE` | every specific value traces to what the model was **given** — user turns and tool results, **never Penny's own turns** | the sample's inputs vs its output |

**If a check fits none of these, it is not an assertion.** Delete it, or move it to variance.

`PROVENANCE` excludes Penny's own turns for a specific reason: a value she invents early in a turn
rides into the message history, and a later check that treats the history as source material lets
the fabrication launder itself.

**A sample is hermetic** — its own database, its own conversation, its own pages — and every claim
resolves against the world *that sample* was given. A model that ignores the page and emits a
plausible value fails the `STORE` claim on whichever world it was handed.

**A claim read out of model prose is noisier than a structural one — know this before you write
one.** Measured across two runs of identical code, on the same commit and the same model: a
reply-content rate moved by **3 samples** where every structural claim moved by **at most 1** (over
18 samples, so ±17 points; wider still at the 15 a case pools). The claim is legal and worth making
— `PROVENANCE`'s reply half is one — but read a few points of movement in its rate as the ordinary
noise of reading prose, not as a change in behaviour.

### The non-negotiables

| rule | why | the evidence |
|---|---|---|
| **Never assert a route.** Assert a *property* of the routine ("it has a write step", "it names somewhere to act"), never its shape or its tool names. | Many routes reach one end state, and a skill is an arbitrary tool sequence — a name-keyed rule simply will not fire for a shape nobody enumerated. | #1993: three different tools all correctly reached the run record; the check had pinned one. |
| **A judgement call in the fixture is variance, not an assertion.** | Asserting a count asserts one reading of an ambiguous world. | Whether an appointment counts as a "signing" is genuinely ambiguous. |
| **An assertion about the store reads the WHOLE entry** — key *and* content. | A fact in the key and a blurb in the body is a perfectly good way to store it. | A prototype reported a 25/32 model failure that was entirely its own bug: it read content only. |
| **A sample `.db` exists from sample START, not completion.** Gate on completeness before pooling; file counts are not completions. | Otherwise dead samples are pooled as behaviour. | 17 dead samples of 31 in one prototype run; `run_health.py` is the machinery that closes it |
| **Never match a phrasing.** A reply check looking for a token you guessed in advance is the thing this design replaces. | Measurably both too strict and too loose in the same suite. | 31 replies that stated the recorded cadence correctly were failed; elsewhere an infra error string and a raw thinking leak both scored *passed* (#1994 §1). |

---

## 3 · The porting checklist — run it in BOTH directions

A ported check is never inherited — it is re-derived. Fidelity to the case you are porting from is
not fidelity to this design, and the two come apart in both directions. **Run both columns, every
time, on every case.**

> ### Outward — for each check on the source case
>
> **Which of the three categories does it fall in?**
>
> **None → delete it. It does not port.** Being on the canonical case is not a reason to keep it:
> the canonical cases predate this design, so a check surviving there is evidence of nothing.
>
> ### Inward — for each of the three categories
>
> **Does the ported case make a claim in it?**
>
> **No → write one**, even though the source case had none to copy.

**The inward direction is the one nobody runs.** Outward feels like work — you are looking at a
check and deciding its fate. Inward looks like nothing is missing, because the thing that is missing
was never on the page.

Here is what it found on the reference port. The canonical case it was ported from carried **no
`PROVENANCE` claim of either kind**. Both of the ported case's — that every stored entry traces to
what the round was given, and that every specific value in the reply is sourced — exist *only*
because somebody ran the inward column and wrote them; there was nothing to copy. A port that had
run the outward column alone would have shipped a case that cannot tell a fact read off the page
from one the model invented, and every check it did carry would have passed.

Two checks that fit no category, as worked examples of how to recognise one:

| check | why it is not an assertion | where it goes |
|---|---|---|
| `assert_each_page_was_read` — reads `sample.pages_read` | asserts that a **browse call happened**. That is a route. `LANDED`? No — a fetch is not a landing. `STORE`? No — nothing was stored. It is model output. | section B, inside tool sequence |
| `assert_the_reply_reports_what_was_stored` — asserts the reply text contains a token | a **phrasing match** — a token somebody guessed in advance, which is the thing this design exists to abolish. | delete. Whether the reply describes what actually landed is a real question and is not answerable from prose — see §9 |

Note what makes the first one seductive: it distinguishes "read the page and correctly found
nothing in scope" from "never looked", which is a real distinction worth having. It is still a
route. The end-state form of that question is a `STORE` claim about what was kept, plus the tool
sequence measured in section B — where a cohort that stopped browsing shows up as a variance rise
rather than as one sample's failed check.

### Two structural backstops — and what they do not cover

The checklist has structural help, and it is worth knowing exactly how much:

- **The category is a required field from a closed enum** (`SpecCategory`, no default), so a check
  that fits none of the three **cannot be declared**. That makes the *outward* column a fact the
  code states rather than a review someone remembers to run.
- **The assertions table groups by category and renders an empty one as a visible gap**, so a case
  asserting nothing about, say, `PROVENANCE` shows as a hole in the report rather than as nothing
  at all.

Neither one writes the claim for you. A gap you can see is still a gap until the inward column is
run and something is written into it — and a claim can satisfy its enum while being the wrong claim.
The backstops make the omission *visible*; the checklist is what closes it.

### Copying the reference port

Sub-tickets say *copy the shape of `learn-demonstrated-round`*. Copy its **structure** — the
fixture, the world, the `measure(...)` call, the parametrisation over models.
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
        min_pass_rate=None,                             # assertions are never floored — §8
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

    # B · what is measured, never asserted
    cohort.measure(TOOL_SEQUENCE, ROUTINE_SHAPE, ROUTINE_NAME, ENTRIES_STORED, TRANSITIONS,
                   REPLY_SPREAD)
```

The three category comments are load-bearing: they are where the **inward** pass is run. A case
with no `# A · PROVENANCE` block is a case that skipped it. Expect your case to have one claim
nobody can hand you — a category the source case said nothing about, which only the inward column
will surface.

**One setup, one enactment, one set of assertions.** A case drives its ask **once**. Two enactments
in one test body is a structural antipattern: it turns the body into a script, and it hides a second
cohort inside a test named for one. Input variation is expressed as **parametrisation** — that is
what `also_phrased` is, and what the model axis is — never as a second drive.

**A claim only one case makes stays inline in that case**, as a small local function. It graduates
into `assertions.py` at the **second** customer, not the first.

---

## 5 · Section B — variance

Measured over the **pooled cohort**: normalised Shannon entropy over each feature's value
distribution, divided by `log(N)`. `0.0` is total agreement; `1.0` is every sample distinct. Reply
text is measured instead by `cosine_similarity` + `token_containment_ratio` over the pairs — two
views because they fail differently (an embedding says two replies are *about* the same thing,
which at fixed topic is nearly always true; containment says how much vocabulary they reuse).

Every feature carries a **consequence class**, declared on the feature and read at the `measure()`
site. It decides how the feature is *read*, not whether it is measured:

| class | means | features | what it makes a divergence |
|---|---|---|---|
| **consequential** (the default) | a divergence implies a different end state | tool sequence, routine shape, entries stored, transitions | a finding about **that sample** — worth looking at on its own |
| **cosmetic** | a divergence is a different word for the same outcome | routine name, reply text | a finding about the **system** — measured and reported, never a fact about one sample |

A feature not in that table is classified by the question, not by the list: **does a divergence here
imply a different end state?** Getting it wrong is not cosmetic in either direction. File a
maximally-spread feature as consequential and it makes almost every sample an outlier: measured, the
naming feature accounted for 8 of 9 outlier rows, reporting "60% of samples are outliers" where the
true statement was "1 of 15 diverged consequentially". **A value that does not separate the models
is a system-level finding for the variance table, never a per-sample outlier.**

### Measure the value, not something derived from it

> **When a value is varying, measure the thing it varies at — not something computed downstream
> from it.**

A container's name is `derive_collection_name(routine name, bound values)`: a pure function, exposed
as `round_framing.container_name` so a fixture cannot grow a second copy of the scheme. It has no
discretion at all. Measured across a run it holds exactly, and the bound-value half is
byte-identical in every sample — so a feature reading the *container* name is reading the **routine
name through a slug function**, under a label that hides what it is.

What varies is the framer's output, upstream: **ten distinct routine names for one routine** in
fifteen samples. So `ROUTINE_NAME` is the feature, read straight off the registry, and it is
**cosmetic** — `watch_price` and `monitor_listing_price` leave the same round, the same write and
the same container behind, so the spread belongs in the variance table as the framer's naming spread
and never as a fact about one sample.

Measuring something derived instead attributes the spread to code with no discretion, and points the
fix at something that cannot be fixed.

### A feature that read nothing is not a feature in agreement

A feature whose every sample reads its **absent** value — no tool call, no routine, a field the
draw never returned — scores `0.000`, which is the same number a cohort in perfect agreement
scores and the opposite finding. So a feature declares the reading that means it saw nothing
(`Feature.absent`), the pooler marks the case **blind**, and the report renders it red with no
proposed ceiling. A variance feature that cannot see an outlier is worse than an absent one:
the table prints a number either way, and only one of them is a measurement.

The same rule covers a half-measure: a reply spread whose cosine half could be computed on no
pair reports `0.000`, which in that table reads as *every pair maximally dissimilar* — the
strongest possible finding — rather than as the absence of one. It says so instead.

What this catches concretely: a non-chat fixture reusing a reader filtered to the chat agent's
own rows comes back empty on every sample. That is why each fixture writes **its own**
observation and **its own** completeness gate — chat's "no reply" condition applied to a
single-call context would void every sample and refuse the run.

**Report per-phrasing rows beside the pooled score.** Phrasings are a *coverage* mechanism, not a
variance one — model stochasticity carries essentially all the spread (~0.05 of it is phrasing; one
model produced 4.8 distinct routine shapes inside a *single* phrasing), which is what justifies
pooling at all. But the pooled number hides what phrasings are for: four phrasings measured
`H = 0.00, 0.52, 0.00, 0.00` — three perfectly deterministic, one that destabilised the case
completely — and pooled to `0.18`. The pooled number is the gate; the rows are the diagnostic
saying which wording moved it.

### Rejected draws are never scored

A re-rolled draw is persisted whole in the promptlog, so every discarded attempt is sitting there
looking like a reply. Only the **delivered** one is Penny's output. A rejected draw is working
machinery: **not scored by an assertion, and not a reply in the variance spread.**

---

## 6 · Cohort sizing, and the two things people confuse

**5 phrasings × 3 samples = 15 pooled per case.**

| mechanism | is | pooled into variance? | serves |
|---|---|---|---|
| **phrasing** | *same world, different words* | **yes** — it is the cohort | coverage, and the pooled variance score |
| **scenario** | a different ask reaching the same end state | no — it is a **different case** with its own 15 | its own behaviour |

The line that matters is phrasing versus scenario: five wordings of one ask pool into one number,
while a different ask is a different case — folding it in would average two behaviours into one
score and call the result instability.

**An arm is one input the behaviour is answered from, together with the world it is answered
against** — and the input is not always the user's words.

- **Chat**: five wordings of one ask, one world.
- **A microcontext**: five wordings of its instruction, one world. For the browse extractor that
  instruction is the `extract` argument of a browse call — **Penny's own words**, written
  upstream at the call site — so the cohort measures how stable the read is against how the
  calling draw happened to word it, which nothing else measures.
- **A collector**: no user turn exists, so its natural language is the `extract` instruction the
  job's rendered program carries, plus the prose of the page that answers it. That instruction is
  written by the `SkillSubstitution` on the `extract` path and reaches the model through the
  shipped instantiation seam — `retarget_writes` → `bind_parameters` → `render_skill` — so
  varying it varies a draw. **Never hand-author the program**: a program the instantiation seam
  cannot emit measures a render rather than a draw, which is §5's own trap one layer up. Five
  wordings of one instruction, against five prose variants of one page.

**A collector's facts are held CONSTANT across its arms** — one url, one set of bound values, one
pair of readings either side of the change — because the assertions hinge on them. The watched
datum line is byte-identical on every arm; what varies around it is the prose a real page carries
anyway: a seller blurb, a specification block, neighbouring items with their own prices,
housekeeping notes. Constant facts are what let a case name a value. *The store holds `$449` and
no longer holds `$499`* is an assertion; *the store holds something* is a shape, and a shape
cannot tell a watch that read the right price from one that produced a plausible number.

The arms nonetheless each carry **their own world**, which is why the world lives on the **arm**
rather than on the cohort: the pages differ, so a provenance claim has to trace against the page
that sample actually read. *One world with five wordings* is the special case of *five
`(input, world)` pairs where the world happens to be constant*. A claim is answered against the
world of the arm that produced the sample it is answering about.

Within one arm, the world is **fixed across its samples**. A case that varies it does so as a second
**input** axis, pooled exactly like phrasing — and **within** the 15, never as samples added beside
them.

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

## 8 · Thresholds — not yours to set

**Your case lands report-only** (`min_pass_rate=None`) and stays there.

- **Assertions carry no floor at all.** They are expected to run at 100%, so the run counts them and
  a person reads the count (§1).
- **Ceilings are the variance side's**, one-sided — only a rise is a regression. A run *proposes*
  one; **accepting it is the code owner's act**, never a case author's.

You do not write either. How a ceiling is recorded, why a comparison across model or N is refused,
and when a feature is too spread to carry one at all are settled by `cohort.py` and pinned by
`test_cohort.py`.

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

**A worked example, and an honest cost.** A known defect — the learn close reporting the *write
record* instead of the value it stored, seen on roughly half the samples (#2010) — is measured by
**nothing** in this design. It is not a `STORE` miss: the right value is in the store. It is not
`PROVENANCE`: the record it names is real. And reading it out of the reply text is the phrasing
match §2 forbids. It is exactly the third row — wrong-but-stable, catchable only by a human reading
one sample — and the finding lives on its ticket rather than in a check.

**Do not invent a category or a special case to keep it measured.** A category added to preserve one
measurement is how a closed list stops being closed, and the list being closed is what makes the
porting checklist decidable at all.

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
  models, whose numbers agree.** One run gives ceilings that might be that run's noise.

---

## 11 · Where this lives in code

| file | holds |
|---|---|
| `penny/penny/tests/eval/utils/cohort.py` | the arithmetic — `SampleObservation`, `Claim`, `SpecCategory` (the closed three), `Feature` + `Consequence`, `normalised_entropy`, `pool`, `proposed_ceiling`, `compare_to_ceiling`, the standings |
| `penny/penny/tests/eval/utils/assertions.py` | `Cohort` and the named claims a case makes against it |
| `penny/penny/tests/eval/utils/worlds.py` | `World` — the pages, the `keeps` token set per source, the `excludes`; carried per **arm** (`cohort.Arm`), not per cohort |
| `penny/penny/tests/eval/utils/run_health.py` | cohort accounting, the fault tally by class and provider, and the viability verdict — its module docstring is the fullest statement of the problem |
| `penny/penny/tests/eval/utils/report.py` | the case document — it renders and never computes |
| `penny/penny/tests/eval/conftest.py` | the drivers, and the `_arms` seam they share: `ask` / `also_phrased` / `world` / `seed` / `samples_per_phrasing` for chat, `instruction` / `also_instructed` for a browse extraction. Each fixture brings its **own** observation and its **own** completeness gate |

A `World`'s `keeps` is one token set **per source** — tokens appearing only on that page, so a
stored copy says which page it came from and an invented one matches neither. They identify the
**source**; they are **not** a list of what the ask puts in scope. A round told to collect trades
that reads a page whose only item is an appointment has correctly found nothing in scope, and
asserting that page's names would fail it.
