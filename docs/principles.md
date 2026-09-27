# Penny's principles

These are the rules Penny is built, presented, tested and worked on by. Read them before
designing a feature, a prompt, a tool or an eval case.

**This file wins.** When a design doc, a prompt, a case or a comment disagrees with it, the
other text is stale: fix it. A principle changes only by the code owner's ruling, and the
change is made here first.

Rules marked **(often forgotten)** are the ones Claude sessions have most often had to be
reminded of. Read those twice.

Quotes are the code owner's own words.

---

## 1. What Penny is for

- **Talking to Penny should feel magical.** Net value out exceeds what you put in, because
  her architecture multiplies the value of what she is given instead of dividing it. Her
  state compounds.
- **Penny within Penny.** She is a long-lived, self-reflective system that reasons about her
  own state and behaviour. She is not a dormant request–response chat product.
- **Local-first is the precondition, not a budget compromise.** Three things follow from
  running locally, and a cloud can't offer them:
  - a long-lived process;
  - data personal enough to be worth compounding;
  - background cognition at near-zero cost.

  A small model is the price. The discipline in this file is how that price gets paid.
- **Single user, security first.** One person, many devices, one conversation. Permissions
  start at zero, and untrusted content never shares a context with tools and sensitive data.
- **The whole design in one sentence:** *"the model chooses the next tool stochastically
  given a deterministically derived internal state."* The model is the policy; the harness is
  the state-transition function. Every good fix is one of two things:
  - a state repair, making the right choice legible where the decision is made;
  - an action-space repair, giving the model fewer, better-typed choices.

  None of them is "make the model smarter".

## 2. How Penny is built

1. **Do as much deterministically as possible.** *"Do as much in code space as you can; leave
   the last bits of fuzzy reasoning to the model."* Where an action can be done in Python, it
   is done in Python. The line falls between mechanics, which go to Python, and anything the
   user reads, which the model writes.

2. **Give the model as few choices at a time as possible.**
   - The state machine decides the turn's state before the turn runs.
   - A collector's tool surface is only the calls its program makes.
   - A micro-context takes one decision.
   - The framework fills in what it already knows (an apply turn states only the job's terms).

   Each draw is a small, typed question.

3. **The model never writes historical records.** *"don't make the model do it at all."*
   - Ledgers, run records, mutation events, what a turn wrote and what a skill is are all
     derived by the framework at chokepoints.
   - The model acts; the system remembers.
   - When the model tells the user what happened, it reads a render of the record, never its
     own memory of it.

   Logs keep calls verbatim and never paraphrase them.

4. **Structural state over model judgment.** "Has this happened?" is a cursor, a flag or a row
   that is read. It is never re-decided by the model each time.

5. **Nothing is more than one guess-free call away.**
   - Every name, key and id the model needs is rendered verbatim where it reads. A guessed
     argument is a rendering bug.
   - Metadata renders where its data renders. A lookup tool for a join the row already
     carries is *"a rendering bug wearing a tool costume"*.
   - A design that changes what the model reads carries this reachability sketch on its
     ticket before it is built.

6. **A skill is an arbitrary tool sequence (often forgotten).**
   - Nothing may be keyed to a tool name, an argument position, or the wording of the example
     in front of you.
   - Ask: *what happens when a tool I have never heard of does this?*
   - Prefer a structural mark set once and read everywhere.
   - In model-facing text, name the state, not the verb.

7. **Enumerate decisions; let the model generate only content.**
   - Every decision the model makes is a pick from a declared set, with an honest escape
     outcome.
   - The output is validated against its declared shape.
   - An invalid draw is discarded and re-drawn from the same state, never argued with inside
     the conversation.
   - A hallucinated shape is rejected with a message that teaches the fix, never absorbed by
     loosening the parser.

8. **Degrade visibly, never silently.**
   - An unmet prerequisite, a failed capability or a degraded mode shows as an actionable
     signal.
   - Penny never claims an action succeeded unless the tool result says it did.
   - There are no silent fallbacks and no silent cuts: anything shortened says what it left
     out.

9. **A few composable primitives, not bespoke mechanisms.** Every recurring intent is a
   configuration of a small vocabulary the model can reason about. **Machinery needs a second
   customer (often forgotten):** fix the one instance, and build no guard, registry or tool
   for a class with one member.

10. **Change behaviour through data before code.** Prefer the highest rung that works:
    1. the user teaches Penny;
    2. a direct edit in the UI;
    3. a prompt change measured by the eval;
    4. code, or a migration, last.

    The goal is to manage Penny through Penny.

11. **Weight the ambient space toward Penny's own state.** The user's data can be re-reached
    from their next message; Penny's internal state can't be. So the self-state is rendered
    every turn, and the user's store appears as a map, not as its contents.

12. **Invent no limits.** No truncation, threshold or cap that nobody asked for. Prefer a
    structural rule (identity, set relation) over a tuned cutoff that drifts.

## 3. How the model is presented with state

1. **The model reacts rationally to what it is shown (often forgotten).** *"the model is
   acting rationally from the information that it is given; if it is not acting in the
   desired way, the data must self-evidently present the model with the correct choice;
   negative corrections are only sufficient for guarding against edge cases, not for
   promoting core workflows."*
   - When the model does the wrong thing, read its thinking first and ask what the state
     failed to show.
   - An instruction is the last rung, never the first.
   - After a state fix lands, delete the instructions it made unnecessary.

2. **A model is only as effective as the data it is presented.** Most "it can't" turns out to
   be "it was never shown it legibly". Fix what the model reads before touching its
   instructions, and fix the rendering before the prompt.

3. **Write prompts in plain words (often forgotten).**
   - Use short, direct sentences in common words, not a private dialect of precise-sounding
     phrases.
   - Give permission as well as prohibition ("it's fine to…").
   - Root instructions in what is, not in hypotheticals.
   - Keep literals the model must write short and common.
   - The second patch for the same behaviour means a wholesale rewrite, not a third clause.

   `docs/prompt-writing-guide.md` has the craft.

4. **Let real data guide the design.** Prototype against the real model and real logs; don't
   guess. Change one lever at a time and compare against a baseline.

5. **Tool results carry the next move.** A failure says what went wrong and how to fix it,
   naming the field or the anchor to copy. A diagnosis without a remedy is half a failure.

6. **Don't feed the model its own past output as context;** it fixates on it. Continuity
   comes from the substrate: facts, never imperatives, and structure, never narration. Order
   a document so the thing to write from comes first.

7. **State definitions are product semantics.** What a state means changes only by the code
   owner's ruling, and it is never tuned as an eval lever. A fixture's wording never appears
   in a prompt.

## 4. How behaviour is verified

The eval cases are where the code owner states what Penny should do. The contract is
`docs/eval-case-design.md`.

1. **Facts are asserted; behaviour is measured (often forgotten).** *"we deterministically
   validate facts; variance measures model behaviour."*
   - A claim is a fact about the world: what the store holds, what survived, where the
     machine landed, or a fact the reply cites (the interest it named, the price the page
     posts).
   - Which tools were called, whether the model retried, and how the reply is worded are
     variance features, never claims. Flailing shows up as spread.

2. **Every claim is strictly true or false, and fits one of three categories:** where the
   machine *landed*, what the *store* holds, and whether a stated value has *provenance* in
   what the round was given. A check that fits none of them is not an assertion.

3. **Assert what survives, never that the model refrained.** There are no claims like "wrote
   nothing" or "created no mechanism". Claim that what was already there is still there.

4. **The model makes every call itself.** No forced or injected tool calls: a natural input
   leads the model to act, and the real gate answers. No artificial prompts either; evals run
   the real code and the real prompts.

5. **A core set of unique cases.**
   - One case per core behaviour, stated as *"In the <locus>, when <X>, Penny <Y>."*
   - Five wordings × three samples, run on both roster models, with no pass floors.
   - A behaviour another case already captures is a duplicate, and duplicates are deleted.

6. **A well-formed case that exposes a Penny gap lands, and the gap is filed.** Harness defects
   are fixed immediately. Porting a case never fixes Penny mid-port.

7. **Prove the measurement could have seen the failure (often forgotten).** A pass from an
   instrument that could not have failed tells you nothing. Before trusting a number, show the
   check reads what it claims to read.

8. **The turns are the ground truth, not the score (often forgotten).** Read the sample: its
   turns, the model's thinking, the artifact itself. Never relay an agent's summary of it.
   Check the scorer before blaming the model.

9. **Consistency is not correctness.** Three instruments, none a substitute for another:
   - variance catches instability;
   - assertions catch wrongness;
   - a person reading one representative sample catches the wrong-but-stable.

   Rejected methods, not to be re-proposed without new evidence: embedding similarity, golden
   sets, model-as-judge, phrase lexicons.

10. **A number describes the tree that produced it.** Push before you measure, and check that
    the PR head is the commit the run recorded.

## 5. How work gets done

1. **The edges are the code owner's; the internals are the supervisor's.**
   - Eval cases belong to the code owner: their inputs, behaviour sentences and claims, and
     any new judgment on them.
   - Penny's runtime, the harness, CI and docs belong to the supervising session, which
     designs, reviews, approves and merges them, using his cases as the acceptance test.
   - Task agents never approve.
   - Still asked first: anything touching production, full-suite eval runs, and anything
     outside the repo.

2. **Apply his sentence, not a sharper rule derived from it (often forgotten).**
   - Implement exactly what was asked: no substitutions, no added scope, no carve-outs.
   - A disagreement is raised as a question.
   - A one-clause disclosure is not consent.

3. **Runs on his cases are joint checkpoints.** Scope each run to the case at hand, post the
   report, and stop for his read. A plan described in chat is not an approved chain of runs.

4. **Say what things are (often forgotten).**
   - Plain names; titles that name the work.
   - Numbers are passing counts.
   - Quote the evidence verbatim, because the code owner can't see tool output.

5. **State what is true.** No archaeology in code, docs or issues; history lives in the PR
   body. Dead code is removed, and a design issue holds one canonical design, rewritten in
   place.

6. **Every change comes with tests.** Integration over unit, folded into existing tests, with
   whole-render literals for anything the model reads. `make fix check` is the only gate.
   `docs/pr-review-guide.md` is the full rulebook for code.

7. **Isolation and the hard lines.**
   - Every editing session works in its own worktree.
   - The repo is public, so no private data ever appears in it.
   - The GitHub token is minted and checked before every call.
   - A dispatched agent reads only its own tree and never `.env`.

8. **GitHub is the durable state, not anyone's memory.** Reconcile against live queries. Before
   relaying a rule, check that its mechanism actually works.
