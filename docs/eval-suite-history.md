# Eval Suite History
A case-by-case, file-by-file record of how the live-model eval suite's cases and reporting mechanics came to be — moved out of `penny/CLAUDE.md` to keep that file to current working guidance. Read on demand, not as onboarding.

The coverage matrix is the two
agent shapes × answer-from-memory vs. browse-and-reason: `test_chat_reply.py`
(#1919, ported to the cohort structure by #2008 — the chat REPLY on an idle turn, as FIVE
cohorts of fifteen: two behaviours, one case per ENTRY CONDITION. Browse-and-answer is two
(the fact posted on the page she reaches, and the fact one link deep, which needs its own
sentence because the index's answer-bearing line is what carries the next address); nothing
to answer from is three (every source unreachable, a store with nothing in it, a store that
already holds what she is asked to record) — on the first two the claims are that the turn
ends back in idle and that every specific value stored or said traces to what she was given,
and whether the reply says the lookup failed is measured as reply spread. Answering out of
what the store already holds is `memory-cold-recall`'s, in `test_chat_memory_stories.py`. One
A/B PAIR shares an ask — the same five wordings against a readable page and against a world
where every source errors — so the negative direction is expressed as the WORLD rather than
as a clause. Every watched value is INVENTED and is ONE whitespace-free token (a posted
admission, a maker's surname), because `fold_typography` folds a
declared set of space characters rather than the whole category and a multi-word token can be
failed by a space nobody has met. Report-only throughout. What did NOT port: every
`tool_was_called` / `tool_not_called` / pages-served check (a ROUTE, and three of them keyed
to a tool NAME — measured as `TOOL_SEQUENCE` instead), five reply vocabularies somebody
guessed in advance, the price-SHAPED reply regex (subsumed by
`assert_every_value_in_the_reply_is_sourced`, which fails any number the reply supplies in a
world where nothing was read), the emoji voice advisory (prose, and section B has no shape for
a binary voice flag), and the `chat-reply-reflects-every-call` case whole — a message carrying
a save, a lookup and a recall is three behaviours in one turn. The duplicate case's entry
condition is now SEEDED rather than driven as a first turn, because a case is one entry
condition and ONE model run. It replaced `test_chat_response.py` + `test_narration_survival.py`,
whose `likes`-collection seed migration 0097 had left crashing at seed time and whose chitchat
case the canonical `transition-idle-to-idle` already covers on a stronger world),
`test_standing_collection.py` (a job that is already running, on the cohort structure
(#2008, tranche 3; #2215): SEVEN cases on one seeded world — `standing-notify-off` (the job's
own switch off), `standing-notify-on` (the same job seeded quiet, its switch back on),
`standing-archive` (retired as a tombstone that still holds what it gathered),
`standing-schedule-fix-prior` (re-timed, and every clock time the reply names is one the job
has had — any third hour is invented, since the ask supplies one and the seed the other, #1946),
`standing-end-changed` (given an end: the job stops when the ask said, read on the sample's
pinned clock through `utils/job_end.py`, and every time it fires until then is a time it was
already due), `standing-describe-routine` (read back: the reply names the page the routine
fetches, a token only the job's record carries, and every value it states traces to that record)
and `standing-list-running` (three jobs running: the reply carries one token per job that only
that job's record holds, with every job and everything each gathered as it was). Each claims
the job's own ROW as the turn left it — the
field the ask named moved, and PRESERVATION of everything else, read as the row's END STATE
through the ledger (`MechanismRecord.moved_this_run`: each field whose value now differs from
the prior the mutation ledger recorded for it, so a field a call merely restated has not moved)
— plus what the job gathered still being there and notifications still on for everything else.
No case claims what the turn did NOT do (`docs/principles.md` §4.3); which verb she used is a
route, measured in the tool sequence. Every case lands in `idle` — changing how a running job
behaves is idle by the machine's own boundary (#1927). The re-timing case reads the hour the
stored RULE fires at through production's own `next_occurrence`. The two directions of the
notification switch are two cases, each on a job seeded with the switch the other way.
Broadening a job's scope is NOT
here — that is a re-teach of the routine (code-owner ruling). Its seeded world is a taught
routine stood up through the production instantiation seam, which `test_speakable_log_reads.py`
reuses for the jobs its collector-runs case reads about),
`test_chat_memory_stories.py` (the memory verbs — save, recall, forget, update, fan-out, no-fire
— and `memory-delete-a-whole-list` (#2215): four plain lists, storage only, and an ask to get
rid of one by name; the claims are that the named list is ARCHIVED and still holds every entry
it held, and that each other list is still the live list it was with its entries. It is not
`memory-forget-then-list`, which names one NOTE and claims the list is still live — a correct
sample for either is wrong for the other),
`test_half_the_sources_landed.py` (#2149 — `memory-save-with-a-source-down`, the cohort-form
port of #1946's writes-landed case and the only live-model coverage that frame has: the user
names two pages and one list, ONE page answers and the other cannot be read, and what is
claimed is that the named list holds an entry this turn wrote from the page that answered,
that every entry this turn wrote to that list is from that page — the claim that reads an
entry invented in plain prose, which names nothing for store provenance to trace —
that nothing stored or said traces outside what the round was given — the dead page's own
names exist in nothing the model saw — and that the reply carries a token only the readable
page owns, the fact the save came from. No count of saved things is asserted: a count of saves
is a count of tool calls. Whether the reply admits the read failed is variance, never a claim —
that branch belongs to `test_chat_reply.py`),
`test_collector_honesty.py`, `test_retrieval.py`,
`test_browse_extract_fields.py` (#1942/#2059 — the browse EXTRACTION micro-context in
isolation, driven directly by the `extractor_eval` runner with no browse in the loop:
FOUR cases on the cohort structure over the THREE states a page can be in relative to its
instruction — it carries ALL of what was named, SOME of it, or NONE — with the middle
state reached on two page shapes: a listing that answers everything (the control), a
homepage read as a link index carrying titles and links and no summaries, a prose section
front carrying a byline under each headline and no timestamp, and a page carrying none of
what was asked for, which must still say so. The last two of those are a PAIR whose
members are separate cases because their EXPECTED OUTCOMES differ: a behaviour that always
answered `EXTRACTED:` passes the partly-present one and fails the none-present one. A
negative direction is stated inside its own case only where it shares the expected
outcome; where it has its own, it is a case of its own and the pair is named together.
Each case is five wordings of its `extract` instruction over ONE fixed page — the arm is
PENNY's own text, written upstream at the browse call site, so the cohort measures how
stable the read is against how the calling draw happened to word it — and hedged and
unhedged wordings pool into one cohort, since what the page carries is not a property of
how the ask was worded. LANDED is the outcome enum by equality, STORE is empty (one call
writes to no store), PROVENANCE is fact alignment in both directions: one claim per thing
the page supplies, plus `unsourced_specifics` over the whole structured answer. The
none-present case is the one place the omitted direction has no instrument — the page
supplies nothing an answer could carry — and it says so rather than making a vacuous
claim.
The page each case is given is what the content script returns for it, so the two halves
of #1942 meet in the fixtures; a coherence probe in `make check` holds every anchor against
its own page, exactly once),
`test_notifications.py` (the muting contract that retired `/mute` + `/unmute`, on the
cohort structure (#2008, tranche 3; #2215): `explicit-mute-request-mutes` (the user asks
for notifications to be muted, against the production cold start — the user is muted afterwards),
`explicit-unmute-request-unmutes` (the same ask in the other direction, on a world SEEDED muted —
the user is not muted afterwards) and
`notifications-no-fire` (a remark that names notifications and asks nothing — the turn stays in
idle with notifications still on, PRESERVATION of the one prior state that world holds). That
the header carries the switch (#1919) is a prepare-time PREMISE rather than a claim, and the
unmute case's premise also holds its seed: not being muted is true of a world nobody muted, so
its probe asserts the mute row and the header's muted line before the turn runs),
`test_choose_dispatch.py` (the fair pick, PORTED (#2008, tranche 3): ONE case,
`choose-dispatch-fires` — the option the reply reports is the one the run's own record chose,
read off the tool's shipped result template. A question about which option she prefers is not
a case: it has no fact to assert),
`test_speakable_log_reads.py` (looking back at her own logs, PORTED (#2008, tranche 3, folding
#2001): `speak-logread-penny-messages-recall` and `speak-logread-collector-runs`, each over a
world where the answer token exists in exactly ONE place — an out-of-window turn of hers, and a
failed cycle's stamped reason the header never carries — so the reply stating it cites that
place; #2001's world-integrity probe rides along. `speak-logread-browse-results` and
`speak-logread-user-messages-act` are not cases),
`test_image_dispatch.py` (the `generate_image` tool that retired `/draw`, PORTED (#2208): TWO
cases — `image-request-draws` (five wordings of one request for a picture, "draw me" and "make a
picture of" among them; the media store holds an image drawn this turn whose description names
both things the ask put in it) and `image-no-fire` (a remark about a painting asks for nothing;
the turn lands in idle and the reply invents nothing). The image backend is a canned stub in front
of the real tool, and the chat observer reads the media store as `SampleObservation.images`, a
drawn image being one with no source page), `test_email_dispatch.py`
(#2209/#2215 — the email tools that retired `/email` + `/zoho`, PORTED to the cohort structure:
THREE cases over one canned mailbox. `email-answers-from-the-message-asked-about` names a sender AND
a subject that only one message shares — asking by sender and asking by subject are one
behaviour, since a correct sample for either is correct for the other, and which field she
searches on is a route measured in the tool sequence — and claims the reply carries that
message's figure and none of the figures its neighbours carry (the sender's other mail, and
another sender's mail on the same subject). `email-absent-message` is a near miss: it names that
same sender, whose two messages each carry an amount, and a subject no message carries, in five
wordings that each say "email" or "inbox" — pinned in `make check`: a sender search returns the
two messages, the subject matches none. It claims idle, the newsletters collection unchanged, and
every stored and said value sourced. What the reply does with the amounts it was handed is
MEASURED, by a feature declared beside the case (`mailbox figure in the reply`: names a mailbox
figure / names none), and whether the reply says nothing was found is reply spread.
`email-remark-stays-idle` is a remark about the
volume of email: idle, the seeded newsletters collection unchanged, and every stored value
sourced — the reply is advice in her own words, measured as reply spread and never read for
provenance.
The mailbox is declared on the world (`World.mailbox`, `utils/mailbox.py`'s `CannedMailbox`,
whose deliberately lenient search rules are stated once there) and installed by the chat
driver behind production's own Fastmail tool builder, so only the backend is canned),
`test_skill_labelling.py` (#1828/#2058 — the run-end LEAF
labeller's canonical set: four cohort cases (`namer-tells-two-sources-apart` ·
`namer-names-a-search-spot-as-a-search` · `namer-names-every-spot-from-a-single-turn-teach` ·
`namer-names-every-spot-in-a-longer-routine`),
each one FIXTURE demonstration in five wordings over ONE ledger, so the spots and their current
names are identical on every arm; the `labeller_eval` runner, report-only, with each ledger's
input document pinned byte-for-byte by a drift probe in `make check`.  Their LANDED category is
empty and CLOSED UPSTREAM — coverage is `_labels_every_spot`'s and would run 15/15 by
construction — so every claim is about what the drawn lines SAY (a name that hardens to a
binding key, is not the argument's own name handed back, and comes with a line saying what
belongs there, plus a distinct-names claim where the ledger supplies two things one name could
collapse), all of them the *nothing omitted* direction: the *nothing invented* one has no legal
instrument for a draw whose open fields are an identifier and generic prose),
`test_skill_framing.py` (#1830/#2006/#2056 — the run-end skill FRAMER's decisions
covered in ISOLATION by **six cases**; the `framer_eval` runner, report-only, each fixture's
input document — the round's user turns, one per line — pinned by a drift probe in
`make check`. The page/url family classifies by NAME only, every other family
name-first-then-description, and name tokenization splits digit suffixes (`site1` →
`site` + `1`) so an ordinal pair written that way matches.  Each case is one ask in five
wordings pooled into a cohort of fifteen, claiming
the parameter SET by equality under LANDED (one drawn parameter per piece the ask requires,
and a COUNT claim for nothing else) and nothing else:
`framer-mints-only-the-piece-that-varies` (the ticker — the negative direction is the cadence
and the notification, pieces the ask already settled),
`framer-mints-both-pieces-when-they-are-different-kinds` (a catalog page and a book looked up
on it), the PAIR `framer-names-a-search-as-a-search` /
`framer-names-a-page-as-a-page-and-invents-no-search` (opposite expected answers — the
search-family parameter that is correct for the first is round 8's recorded invention for the
second, so neither is passable by a framer that always answers the same way),
`framer-frames-from-a-single-turn` (the purpose and the varying piece separated out of ONE
sentence rather than read off two turns), and
`framer-keeps-three-of-a-kind-as-three-parameters` (three URLS of the same kind, where run 1
measured every two-page sample folding them into one `sites — list of URLs`).  Their STORE
category is EMPTY — one call returns a typed result, moving no machine and writing to no
store — and their PROVENANCE category is EMPTY and says so: a framing's open fields are an
identifier and two lines of deliberately generic prose, and `unsourced_specifics` — the
suite's one instrument for the invented direction — reads a Title-Cased correct framing as
four inventions while passing a made-up exchange, so it does not transfer to them.  What they
deliberately do NOT claim is genericity: that the name says the KIND of task and never THIS
occasion fits none of the design's three assertion categories, being the converse of a
provenance claim, so it stays a reading of the modal sample.  Nor do they claim what
`_mints_a_usable_signature` already validates (a parameter minted at all, unique names, every
value a literal span), which would run 15/15 by construction.  Each ported case's five
wordings and the facts every one of them must carry are declared together as `PortedArms`,
held by a `make check` probe that renders each arm through the shipped
`build_framing_content` — `carries` for a fact spelled the same way on every arm, `any_of`
for one that survives being REWORDED (being told when it moves / changes / shifts), which is
otherwise the fact class that goes missing from one arm unnoticed), `test_skill_binding.py` (#1867/#2006/#2057 — the skill BINDER; the `binder_eval` runner,
report-only, each fixture's input document — the signature as it stands, then the user's
turns — pinned by a drift probe in `make check`. Membership, coverage and "is this even in
the user's words" are the PRODUCTION validator's, so an accepted draw never carries an
invented value; what the cases measure is whether it picked the RIGHT span and whether it
knew when to decline.  The slot's CANONICAL SET — SEVEN cohort cases, one ask in five
wordings each, claimed against `docs/eval-case-design.md` — covers the binder's whole decision space, {complete bind,
shortfall} × {single slot, multi slot} plus the completion draw: the page bound with
neither the cadence nor the end date riding in; a threshold condition the signature has
nowhere to put; two parameters of different kinds out of one message; TWO OF THE SAME KIND,
where nothing but the word the ask puts beside each says which is which, and a swap points
the job at the right pages in the wrong roles with every downstream check reading it as
correct; the shortfall on an ask that names an object and no page; the shortfall beside a
successful bind; and the COMPLETION DRAW — `bind_entry(…, settled=…)`'s warm shape, whose
document is the two turns a parked round has now heard with the settled parameter absent
from the offered signature, so the page it settled is in the words and no longer in the
question.  Every claim is an outcome enum under LANDED or a bound value under PROVENANCE
read in both directions; STORE is empty on all seven and says so, and three claims are
deliberately not made because production already validates them.  Each case declares the
terms its claims FORBID and derives the facts its arms must state from them, so a fact
nobody measures cannot be written down — the defect that shape closes is a claim label
promising to catch a padded condition while the terms it read forbade only the cadence),
`test_watch_cycles.py` (#2017/#2007 — the BACKGROUND half of the north star: whether the
COLLECTOR runs what chat built. Three cohort cases over one price watch, each ONE real collector
cycle whose behaviour is selected by its ENTRY CONDITION — what the collection already holds —
rather than by a script of cycles: the first reading (empty collection, the reading written and the
user told once), the unchanged reading (the write-gate STOP `KEY_EXISTS_UNCHANGED` ends the cycle,
nothing written, the send queue empty) and the moved reading (the standing key rewritten, the user
told once). The arms are five wordings of the job's `extract` instruction, rendered through the
shipped instantiation seam (retarget → `bind_parameters` → render) over one constant set of facts.
A watch over a different page, datum or routine is another program of the same behaviour, so it is
not a separate case. Driven by the `collector_cycles_eval` runner (the one collector driver there
is — each cycle's footprint kept apart as a `CycleObservation`), report-only, with the program and
the page variants pinned in `make check`). Browse is stubbed; a case supplies realistic pages on
its `World` — or, for a collector, on each `CycleArm` — which the driver installs as the
query-aware register (`install_browse` / `CannedPage` in `conftest.py`), so the claims a case makes
rest on a page the model had to read. A `CannedPage(fails=True)` makes a matched read
*error* (renders `## browse error:` without the real retry backoff), and the
shared `ALL_BROWSES_FAIL` catch-all makes every source unreachable — the way to
exercise read-failure honesty (a cycle that browsed a lot, read nothing, and must
not confabulate a write/success at `done()`). See `docs/archive/self-improvement-loop.md`.

**A driven turn is over when Penny has finished handling it, not when its reply was seen (#2195).**
The mock Signal server holds a reply the moment its send ARRIVES, which is before Penny has heard
back about it, and a message is handled on its own task that teardown does not wait for. A sample
that moved on from the first sight of its reply could close its channel under that send: the send
failed, and Penny logged `PennyResponse.DELIVERY_FAILURE` into the sample's database as a second
message the user received. So `_drive_turns` waits, after each reply, for
`BackgroundScheduler.foreground_active` to clear — the flag the channel drops as the last act of
handling a message, after the reply's send has come back and its row carries its external id.
What a sample's claims read was never the apology (the observation is taken before teardown, and
`reply` is the message the server captured); what it corrupted was the sample's `.db` artifact.
Pinned in `make check` by a server that holds the send unanswered.

**The nudge-loop / bare-`{}` boundary (#1732).** A #1731 spiral — a chat browse-extract run that
looped on parse-failure recovery nudges (re-browsing the same page each cycle) and died at the turn
timeout with a bare `{}` reply — was tagged **harness** (a `TimeoutError` ended it) but the *loop*,
not the clock, was the event. Two boundary questions, resolved to the narrowest honest rule:
- **The terminal `{}` IS pathology.** `is_call_fragment_reply` (in `text_validity.py`, the *shared*
  detector the live reroll guard and the scan both use) was widened to catch a reply that is nothing
  but an empty JSON object (`{}` / `{ }` / `{}\n`, `_EMPTY_OBJECT_REPLY_RE`) — the degenerate tail of
  a forced tool-call attempt leaked into the text channel. The old `{"`-prefix gate missed it (an
  empty object has no string key). Anchored to the whole stripped reply → zero-false-positive (a reply
  *about* `{}` carries prose; Penny never SENDS a bare `{}`), and the full call envelope
  `{"name":…,"arguments":…}` stays excluded (its teaching validators own it). Widening the *shared*
  detector — not just the scan — keeps #1695's "same detectors live and post-hoc" invariant: the live
  chat guard now discards+rerolls a `{}` reply too (a strict improvement — it was being delivered).
- **Nudge frames are NOT counted.** Repeated recovery nudges are *input* messages; counting them would
  read the model's input rather than its output and would need an arbitrary threshold. The nudge
  loop is a *symptom* whose *cause* is the model's own fragment OUTPUT — the terminal `{}` the widened
  detector now catches — so classifying on that output tags the spiral at the root. Cause ordering
  is unchanged: pathology still outranks the timeout, so the spiral reads pathology, not harness.

**The run summary's COVERAGE table (#2256).** `assemble.render_run_summary` renders a second table,
above its existing numbers table, in every model section: `| # | case | behaviour | input | Penny's
final reply | deterministic | variance |`. **behaviour** is the case's own stated sentence
(`CaseArtifact.behaviour`); **input** and **reply** come from the case's REPRESENTATIVE sample —
the one `cohort.standings` marks `Standing.MODAL`, the same sample the case document's own map
sends a reader to open — carried on the record as `CaseArtifact.representative`
(`artifacts.RepresentativeSample`: name · wording · reply), computed once by `conftest.py`'s
`_representative_sample` at case close and threaded into `eval_artifacts.record_case` alongside
`standing_counts`, since the assembler runs over the report dir alone with no cohort to re-derive
it from. **input** is the arm's own driven text (`cohort.arms[sample.arm].text`); **reply** is
`SampleObservation.representative_reply` — a chat sample's text `reply`, a micro-context sample's
structured `output_text` (it has no reply at all), or a collector sample's `reply` (what its cycle
sent) falling back to the honest `"(sent nothing)"` when it queued none. **deterministic** and
**variance** are the SAME cells the numbers table renders, read through the same
`_row_checks`/`_row_variance` functions, so the two tables cannot disagree. Both new artifact fields
are additive and default to `None`, so a `results.jsonl` record written before #2256 still decodes
and renders its three new cells as an honest `—`. Every cell is bounded (`INPUT_CELL_BUDGET` /
`REPLY_CELL_BUDGET`, ~200 / ~400 characters, cut at a word boundary with an `… (<n> chars
omitted)` marker — never silent) and escaped (`report.escape_cell`), so a pipe or a newline in a
reply can never break the table. Because the run summary carries no sample folds for
`comment_split.py`'s existing cut to find, a whole-suite table that outgrows the comment budget
splits on its OWN legal seam instead — a table ROW boundary
(`comment_split.partition_on_table_rows`/`split_run_summary`): a part that resumes a table repeats
its head + separator and states which rows it holds, so concatenating the parts loses no row.

**Report format v3 — transcript-integrated (#1725).** A sample's checks render *inside* its
transcript, in causal order (contract → thinking → action → verdict-on-action), rendered by the pure
`penny/tests/eval/utils/report.py` from a `SampleTranscript` the `conftest.py` extractor builds off the
persisted promptlog. Each sample is a series of **per-step tables** (a user turn opens each step):
one `expected` row per check (`Cn [class]⚖/ℹ label`), an ALWAYS-collapsed `💭 <details>` directly
above **every** model action (thinking for every call now, not only failed turns; `💭 (empty)` when
none), and an `actual` row per transcript event (`🔧`/`📥`/`🤖`/`👤 nudge`/`🧩 <context> ← user
turn:` in / `🧩 <context> →` out — **every** micro-context sub-model is an official actor with its
USER turn explicitly labelled, #1759/#1773) with the check verdict on its anchor row. Whole-run /
missing-action checks fall to a trailing **run-close** table. **A sample renders only its OWN
sequence (#1997)** — the turns it was given, the calls it made, what came back, what it replied.
Its per-context system prompts moved OFF the sample and onto the case document, which states each
distinct one once (see the case-document entry below); a micro-context's real system prompt is
still visible and still never mistaken for its user turn, just not restated under every sample.
An over-long `actual` cell collapses into a
**single** `<details>` (first line + `… (<n> chars)` summary, full text inside — one copy, no
visible head; #1759). Each sample opens with a **banner**
(`verdict · k/n (score) · fragile · cause · duration · calls`); **every** sample block folds whole
under its banner — the ONE and only rendering (uniform collapse, #1753/#1759: collapsed by default,
full body always a click away, identical in the `.md` and the comment; no compact/banner-only form,
no `--full`/`EVAL_FULL` — removed in #1759), and a harness-timeout sample gets an honest placeholder
(F2) inside its fold. The **run
header** carries, in diff mode, a **flips** index (`flips: <label> ✅→❌ (s…)`, one entry per
check fully green in the baseline that failed a sample here). The flips index resolves its baseline
from the run's **durable manifest reference** (`RunManifest.baseline`, recorded from `EVAL_BASELINE`
at eval time, #1752) — NOT from a re-read of the volatile env at assemble time — so `make assemble`
renders the same flips the per-row REGRESSED badges were baked from, even when it carries no
`EVAL_BASELINE` (an explicit `EVAL_BASELINE` at assemble time still overrides, for an ad-hoc re-diff).
Additive artifact fields carry it:
`CheckOutcome` gained `scored`/`cells[]`, `Check` gained `kind`, `CaseArtifact` gained
`sample_fragile[]`; `RunManifest` gained `baseline` (#1752). **No artifact is committed** — the PR comment is
the durable record; all raw artifacts (manifest/results.jsonl/`.md`/`.db`/dirty.diff) stay local and
`EVAL_BASELINE` diffs those local paths (#1725 policy). The format is pinned by whole-render tests
in `test_report.py` / `test_assemble.py` (+ extraction in `test_eval_harness.py`) rather than by a
document.

**Every micro-context renders as its own actor (#1773).** The v3 transcript admitted only
`browse-extract` rows, so the state classifier's draw (#1706) and the run-end skill labeller's
adjudication (#1770) were invisible — their system prompts rendered, the calls made against them did
not — and, being main-agent rows to the turn walk, their scoped slices surfaced as PHANTOM
`👤 user` steps ahead of the real one. Now the extractor (`conftest.py`) admits every micro-context
by its ledger identity and each 🧩 row is labelled with it (`🧩 state-classifier → STATE: apply`),
matching that context's system-prompt row. Where a batch splices is a declared **`MicroPlacement`** —
the causal relationship, not the agent: `DURING_CALL` (browse-extract → after the `extract=` browse
call that spawned it, the unchanged FIFO pairing) · `TURN_HEAD` (state-classifier, skill-framer and
skill-binder → right after the user turn they decide, all three drawing before the chat agent; the
classifier names the routine and the binder fills what that routine declares from the same words, so
the two render in ledger order) · `RUN_CLOSE` (skill-namer, and the notify composer closing a
collector cycle → after the run's last action). One generic walk drains the ledger-ordered batch
queue at those anchors, so a new customer is a row in `MICRO_CONTEXT_PLACEMENTS`, not new code; a
batch left unmatched renders at the end rather than vanishing (collapsed never means dropped, #1753).
**Completeness is pinned structurally (#2133)**: the binder shipped without joining the map and its
draw simply never rendered — nothing raised, no batch was dropped, and every apply/request transcript
showed the classifier's decision followed straight by the chat agent's first call, its reply
reporting a bound value the reader was never shown being bound. A test now reads every agent name
`micro_context.py` stamps on a draw, off that module's own AST, and fails when one is not in the map.
Deterministic — proven by whole-render tests over fixture promptlog rows, no GPU.

**The RECORD is the same scored object, not a second scoring of it (#2125).** `_PendingCase._grade`
is the one seam where a cohort's claims meet per-sample grading, so it settles everything the
`results.jsonl` record reports: a pooled sample `adopt`s the claims answered about it, and a sample
the pool refused is `exclude`d — score 0.0, the document's own exclusion reason where a failure's
labels go, and the `harness` cause an infrastructure loss earns. It used to be skipped, and an
excluded sample answers no claim, so `graded([])` scored it a vacuous **1.0**: a case whose document
said `14 pooled + 1 excluded` recorded fifteen passes and no causes — while its own
`standing_counts` said `dead: 1` — and a later run diffed against that record saw no regression.
The exclusion reason itself comes from `cohort.exclusion_reason` on both sides, so the document and
the record cannot name one lost sample two ways.

**A sample's BANNER states the verdict its case settled on, so the block is rendered LATE (#2127).**
The banner is the one line a reader sees before opening anything, and it was built while the
sample's database was live — where a sample has no verdict, so it read `✅ pass` above a
sample the claims failed or the pooler excluded. A sample's transcript is still ASSEMBLED at that
moment (its promptlog exists nowhere else), but it is held as a `_HeldSample` — the transcript, the
result, and the cost read off the live database — and RENDERED at the flush, which runs where the
case's scores settle: `_PendingCase.finish`, right after `_grade`. So `_sample_banner` stays the
one and only place a banner is made, and there is no second rendering to disagree with the first.

**Sharing is DECLARED, not discovered — the line-diffing machinery is GONE.**
`hoist_shared_prompt_blocks` / `shared_lines` / `unique_lines` / `shared_block_text` /
`_worth_hoisting` computed which LINES happened to be common across a case's prompts and rendered
each sample as a diff against a synthesized block, because samples could not be assumed to share
anything. Under the pooled cohort they do — one world, one seed set, K wordings of one ask — so
prompts are grouped by **exact text** (`report.prompt_variants`) and each distinct one renders
ONCE, verbatim, naming the samples that read it. Measured on the reference port's own 18-sample
run: four of five contexts (state-classifier, skill-framer, skill-namer, browse-extract) are
byte-identical across every sample, 125,586 characters collapsing to 6,977. The fifth is the
finding the diff was hiding — `chat` had **18 distinct texts**, differing in 3–4 lines of 83,
because the self-state header feeds each sample its own minted collection and routine names back
into its prompt. That is the cohort's `container name` variance showing up in the prompt, and the
old marker (`_[shared block — see above]_`) rendered it as an absence. The rule this keeps that a
diff could not: **every prompt renders verbatim**, so a reader opening one reads what the model
read rather than a reconstruction. A sample's own block therefore carries only its own sequence —
inputs, tool calls, tool results, reply — and `SampleTranscript` no longer has a `system_prompts`
field at all; `conftest` deposits each sample's prompts into the case accumulator while its
database is live, and the case document groups them at close.
`comment_split.py` is untouched — GitHub's cap has not moved, and a sample fold is still the only
legal seam.

**A fold bigger than one comment is REDUCED, not refused (#2135).** The seam rule means a single sample
fold over the hard cap has no legal cut inside it, and that used to end the whole post: one
`chat-answer-one-link-deep` sample thrashed for its whole step budget, rendered ONE step table of 78,258
characters inside a 94K fold, and its entire gpt-oss report went unposted while the gemma sibling's posted
normally. The measurement was correct (the thrashing itself is the filed Penny defect #2123) — it was the
posting route that could not carry it, and both remedies the refusal implied were wrong: a lower
`EVAL_SAMPLES` changes the N a ceiling is keyed to, and re-rolling for a smaller fold spends money to hide
a finding. So `report.reduce_sample_body` is the ONE reduction, deterministic, and what it gives up is
ORDERED so a fold barely over budget loses almost nothing: first every cell that folded a second copy of
its own text collapses to the summary `truncate_cell` already built for it (the first line and the length
— on the real fold that alone was 58,161 characters and no row went), then rows go **from the middle
outward** in two tiers, what came BACK and what was thought (results, micro-context draws, thinking,
baselines) before what was DONE and CLAIMED (tool calls with their arguments, expected rows, the reply).
A structural row — a step header, a divider, the run-close header — is never given up, and fenced text is
never read as a row, so the reduction can never edit the model's own quoted words. ONE marker stands at
the first gap saying what was elided, how much, and that the whole transcript is in the run artifact; the
case document's **health row** (`samples`) gains `· N reduced to fit the comment cap`, because an abridged
transcript is exactly the kind of fact a reader would otherwise assume the opposite of. It runs at
ASSEMBLE time over the case document already on disk — the same seam `summarise_thinking` and
`elide_unused_prompts` run at — so re-posting a finished run (`make eval-report RUN=… FORCE=1`) picks it
up with **no re-run**. With every fold bounded, a part over the cap is an INVARIANT VIOLATION rather than
a case: `comment_split.enforce_part_cap` raises `OversizedPartError` as the renderer bug it is instead of
refusing the caller.

**A PR shows ONE latest report, and a comparison against main (#2734).** Every rerun used to stack: a
PR that saw three or four reports carried three or four, and comparing one against `main` meant a kitten
writing one by hand. Mark-and-replace fixed the first half: every comment `eval-report` posts carries a
hidden HTML-comment marker — `<!-- penny-eval-report -->` on a per-case/summary report,
`<!-- penny-eval-compare -->` on a comparison — and before posting, the recipe lists the PR's existing
comments, keeps only the ones carrying that marker AND authored by the token's own login, and deletes
them (`penny/tests/eval/utils/pr_comments.py`'s `select_marked_comments`, the one pure decision, unit-
tested with no network). Resolving "the token's own login" needed its own small fix: the obvious
`gh api user` 403s for a GitHub App **installation** token ("Resource not accessible by integration") —
only `gh api graphql -f query='query { viewer { login } }'` answers it. `BASELINE=<run-dir>` builds and
posts (or replaces) the second half, ONE comparison comment against a named run — one row per case, the
branch's passing checks against the baseline's, a case moved 5 points or more marked `⚠`, a totals row
summed only over the cases each side actually drove. It refuses across providers (a comparison measures
neither one's behaviour otherwise) and warns inside the comment when the branch's merge base with the
baseline's commit is not that commit — checked HOST-side with plain `git merge-base`, the same reason
`HEAD_COMMIT_CMD` already runs there rather than in the eval container. `penny/tests/eval/utils/compare.py`
holds the builder (`CaseComparison`/`build_case_comparisons`/`totals_row`/`render_comparison`), reusing
`comment_split.split_run_summary` for its own over-cap split — a comparison table has the same row-
boundary seam the run summary does. `DRY_RUN=1` runs every read and skips every write, so the whole thing
previews against a real PR with nothing touched.
