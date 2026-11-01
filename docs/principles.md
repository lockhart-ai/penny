# Penny's principles

These are Penny's core design principles: the rules her mechanics, the model's view of her
state, and her tests are designed by. Fall back on them whenever a question comes up about
how to architect a mechanism or a test. How work gets done — dispatch, review, PRs — lives in
the SOPs (`docs/agent-task-workflow.md`, `docs/agent-supervisor-runbook.md`), not here.

**This file wins.** When a design doc, a prompt, a case or a comment disagrees with it, the
other text is stale: fix it. A principle changes only by the code owner's ruling, and the
change is made here first.

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
- **Fast enough to chat with smoothly.** *"penny needs to be fast enough to chat with
  smoothly. spending 5m on thinking loops is a non starter, across the board."* Latency is
  judged on the hardware Penny runs on, where a local model reads and writes several times
  slower than an eval provider: every thousand thinking tokens is tens of seconds of the
  user waiting. A step that isn't reasoning, such as reading a page, runs without thinking,
  and a token budget only bounds the waste; it never makes minutes acceptable.
- **Gemma is the commander; the tools do the reasoning.** Gemma holds a light context and a
  small set of well-laid-out tools, and passes small pieces of information between them. The
  tools carry the work: deterministic logic first, then fast micro-contexts (a decision model,
  or the model with thinking off). The commander is the slowest step on the machine, so every
  token in its context earns its place, and a step it shouldn't have to coordinate is combined
  inside a tool.
- **Single user, security first.** One person, many devices, one conversation. Permissions
  start at zero, and untrusted content never shares a context with tools and sensitive data.
- **The whole design in one sentence:** *"the model chooses the next tool stochastically
  given a deterministically derived internal state."* The model is the policy; the harness is
  the state-transition function. Every good fix is one of two things:
  - a state repair, making the right choice legible where the decision is made;
  - an action-space repair, making a wrong call fail honestly at the tool that
    receives it, never taking the call away or deciding the model's next move for it.

  None of them is "make the model smarter".

## 2. How Penny is built

1. **Do as much deterministically as possible.** *"Do as much in code space as you can; leave
   the last bits of fuzzy reasoning to the model."* Where an action can be done in Python, it
   is done in Python. The line falls between mechanics, which go to Python, and anything the
   user reads, which the model writes.

2. **The model is free; the tools hold the line.** *"the pattern of how we
   guide the models behaviour is by moving more deterministic checks inside
   tool surfaces with honest replies back to the model … but it is free to
   call any tool in any order it likes."*
   - The model may call any tool, in any order. Nothing outside a tool decides
     which state a turn is in or what the model may do next.
   - Two things guide it: what its state shows it (§3), and the checks each
     tool makes before it acts.
   - A check guards an invariant of the world or the store: a job needs its
     instructions, a value already held is not written again, a schedule
     must parse. It never guards an order of calls. Ask: *would this check
     still be right if the model had reached this call some other way?*
   - A failed check replies honestly: what is wrong, why, and what would
     succeed (§3 item 5). It neither removes the tool nor absorbs the call.
   - The same holds at the edge between the model and the infrastructure.
     *"the system should clearly delineate between valid and invalid
     interactions between the deterministic infrastructure and the model and
     enforce and communicate those boundaries clearly, it's the model's job to
     adapt."* A malformed emission (an unknown tool name, arguments folded into
     a name, a mangled call) is rejected with a reply that says what is valid.
     It is never turned into a valid call by a special case: special cases for
     one model's mistakes grow into a web of rules specific to that model.
   - Designing a tool answers three questions, in this order:
     1. What must stay true in the store, whatever path the model took to
        this call? Updating a job keeps it the same job; it never quietly
        becomes a different one.
     2. How does the tool check it? Deterministically where it can; where it
        takes judgement, a micro-context behind the tool boundary picks from
        a declared set (item 8).
     3. What does the rejection teach? What was wrong, why, and the call that
        would succeed: *"there is no tool called `functions.done`; did you
        mean `done`?"* followed by how `done` works.
   - What only judgement can settle, such as whether something is worth
     telling the user, the model decides from the facts it is shown, not
     through a gate.
   - Not used: a classifier that picks the turn's state, per-state tool
     surfaces, or a micro-context that decides what the model does next.
     Each fails as soon as a user leaves the path it encodes.

3. **Penny follows industry-standard agent-harness practice.** *"penny should follow the
   industry standard practices for agent harness design"*
   - How turns, tool calls, tool results, thinking and harness feedback are represented,
     and how history is kept and trimmed, follows the conventions of mainstream model APIs
     and agent harnesses: Anthropic Messages, OpenAI, the model's own chat template, Claude
     Code and the Agent SDK. The model was trained on those conventions and reasons as if
     they hold.
   - A departure needs a measured reason, recorded where the mechanism lives: the
     convention it departs from and the measurement that justified it.
   - When Penny misbehaves, a departure from convention is the first suspect: a diagnosis
     names the convention and whether Penny follows it before anything else.

4. **The model never writes historical records.** *"don't make the model do it at all."*
   - Ledgers, run records, mutation events and what a turn wrote are all
     derived by the framework at chokepoints.
   - The model acts; the system remembers.
   - When the model tells the user what happened, it reads a render of the record, never its
     own memory of it.

   Logs keep calls verbatim and never paraphrase them.

5. **Structural state over model judgment.** "Has this happened?" is a cursor, a flag or a row
   that is read. It is never re-decided by the model each time.

6. **Nothing is more than one guess-free call away.**
   - Every name, key and id the model needs is rendered verbatim where it reads. A guessed
     argument is a rendering bug.
   - Metadata renders where its data renders. A lookup tool for a join the row already
     carries is *"a rendering bug wearing a tool costume"*.
   - A design that changes what the model reads carries this reachability sketch on its
     ticket before it is built.

7. **A job's tool sequence is arbitrary.**
   - Nothing may be keyed to a tool name, an argument position, or the wording of the example
     in front of you.
   - Ask: *what happens when a tool I have never heard of does this?*
   - Prefer a structural mark set once and read everywhere.
   - In model-facing text, name the state, not the verb.

8. **Micro-contexts sit behind tool boundaries; their decisions are enumerated.**
   - A fuzzy input that has to land in a closed set (what a page carries, whether a text
     names a time) is decided by a micro-context the tool calls, on the far side of the
     tool boundary. The tool validates its output and returns it as part of its honest
     reply. The micro-context serves the tool and never steers the model. Browse-extract is
     the pattern; it also keeps untrusted page content out of the main context.
   - When Python reads a micro-context's decision, it is a pick from a declared set, with an
     honest escape outcome.
   - That output is validated against its declared shape.
   - An invalid draw is discarded and re-drawn from the same state, never argued with inside
     the conversation.
   - A hallucinated shape is rejected with a message that teaches the fix, never absorbed by
     loosening the parser.
   - Output that goes back to the model as text, such as what a page carries or a message in
     Penny's voice, has no declared shape: it is plain text that says what it found and names
     plainly what it could not. A strict tagged shape on a fast draw can fail on form while
     the content is right.
   - A micro-context is also how a step gets fast. *"there are times when we want thinking on
     for reasoning about tool calls and decisions and emission formatting. there are times
     when we want thinking off for speed and focus and simplicity."* Narrowing a task to one
     fuzzy job lets it run with thinking off. The agent loop, which chooses tools and decides,
     keeps thinking. A micro-context that reads, extracts or phrases runs without it, and on
     local hardware that is the difference between seconds and minutes.

9. **Degrade visibly, never silently.**
   - An unmet prerequisite, a failed capability or a degraded mode shows as an actionable
     signal.
   - Penny never claims an action succeeded unless the tool result says it did.
   - There are no silent fallbacks and no silent cuts: anything shortened says what it left
     out.

10. **Penny reports honestly on what she actually did.** *"penny reports honestly on her
   actual actions."*
   - Every action ends in an account of itself: what was tried, how far it got, what it
     produced and what it did not. A failure is handled inside the tool: a tool that cannot
     finish stops gracefully and returns that account.
   - The model tells the user what happened, from that account. *"failures should be handled
     inside tools and the chat agent should be able to summarize their responses in her
     response back to the user."*
   - A fixed message stands in for her reply in one case only. *"the only time we should get
     a canned message back from penny is if the model runner actual broke and was
     unreachable."*
   - What she can say in the turn she can look up afterwards. The account is in the record,
     one call away, so asked later what went wrong she can answer.

11. **A few composable primitives, not bespoke mechanisms.** Every recurring intent is a
    configuration of a small vocabulary the model can reason about. **Machinery needs a second
    customer:** fix the one instance, and build no guard, registry or tool
    for a class with one member.

12. **Change behaviour through data before code.** Prefer the highest rung that works:
    1. the user teaches Penny;
    2. a direct edit in the UI;
    3. a prompt change measured by the eval;
    4. code, or a migration, last.

    The goal is to manage Penny through Penny.

13. **Weight the ambient space toward Penny's own state.** The user's data can be re-reached
    from their next message; Penny's internal state can't be. So the self-state is rendered
    every turn, and the user's store appears as a map, not as its contents.

14. **Structural rules over tuned thresholds.** Where a decision can rest on identity or a
    set relation, it does. A similarity cutoff drifts and has to be re-tuned, and a structural
    rule has nothing to tune.

## 3. How the model is presented with state

1. **The model reacts rationally to what it is shown.** *"the model is
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

3. **Write prompts in plain words.**
   - Use short, direct sentences in common words, not a private dialect of precise-sounding
     phrases.
   - Give permission as well as prohibition ("it's fine to…").
   - Root instructions in what is, not in hypotheticals.
   - Keep literals the model must write short and common.
   - The second patch for the same behaviour means a wholesale rewrite, not a third clause.

   `docs/prompt-writing-guide.md` has the craft.

4. **Let real data guide the design.** Prototype a shape against the real model and real
   logs, and let what it does decide the design. Don't design from guesses.

5. **Tool results carry the next move.** A failure says what went wrong and how to fix it,
   naming the field or the anchor to copy. A diagnosis without a remedy is half a failure.

6. **Don't feed the model its own past output as context;** it fixates on it. Continuity
   comes from the substrate: facts, never imperatives, and structure, never narration. Order
   a document so the thing to write from comes first.

7. **A state means one thing, everywhere.** A state's definition is product semantics: it
   renders identically wherever it appears and is never re-worded to make a case pass. A
   fixture's wording never appears in a prompt.

## 4. How behaviour is verified

The eval cases are where the code owner states what Penny should do. The contract is
`docs/eval-case-design.md`.

1. **Facts are asserted; behaviour is measured.** *"we deterministically
   validate facts; variance measures model behaviour."*
   - A claim is a fact about the world: what reached the user, what the store holds or
     survived, or a fact the reply cites (the interest it named, the price the page
     posts).
   - Which tools were called, how many times, whether the model retried, and how the reply
     is worded are variance features, never claims. Flailing shows up as spread.
   - Nothing that counts tool calls is a fact, including a count of saves the reply states. Ask
     Penny to read site 1 for fact A and site 2 for fact B, and let site 1 read while site 2
     fails. The claims are then: fact A is in the store and in the reply, traced to site 1,
     and nothing from site 2 appears anywhere.

2. **Every claim is strictly true or false, and fits one of two categories:** what the
   *store* holds — including what reached the user, which is a record the store holds — and
   whether a stated value has *provenance* in
   what the round was given. A check that fits none of them is not an assertion.

3. **Assert what survives, never that the model refrained.** There are no claims like "wrote
   nothing" or "created no mechanism". Claim that what was already there is still there.

4. **The model makes every call itself.** No forced or injected tool calls: a natural input
   leads the model to act, and the real gate answers. No artificial prompts either; evals run
   the real code and the real prompts.

5. **A core set of unique cases.**
   - One case per core behaviour, stated as *"In the <locus>, when <X>, Penny <Y>."*
   - Five wordings × three samples, run on the suite's model, with no pass floors.
   - A behaviour another case already captures is a duplicate, and duplicates are deleted.

6. **A check must be able to fail.** A pass from an instrument that could
   not have failed tells you nothing. Every claim and every measured feature has to read the
   thing it names, and a feature that read nothing is blind, not "in agreement".

7. **Consistency is not correctness.** Three instruments, none a substitute for another:
   - variance catches instability;
   - assertions catch wrongness;
   - a person reading one representative sample catches the wrong-but-stable.

   Rejected methods, not to be re-proposed without new evidence: embedding similarity, golden
   sets, model-as-judge, phrase lexicons.

8. **A test exposing a Penny gap is a good test.** A well-formed case that fails because
   Penny falls short stands as written; the gap is fixed in Penny, never by bending the case.

9. **Eval worlds use real proper nouns the model recognises.** *"we must use
   real proper nouns that the model recognizes we just need to swap them from the ones from
   my real logs"*
   - An invented place, person, brand or product reads as a simulation, and the model reasons
     from that: it doubts a search can find anything, so it offers instead of acting.
   - When a case is cloned from real use, each real name is swapped for another real one of
     the same kind, never an invented one.
   - Topics are still changed for privacy.

10. **Diagnose a case through its failing samples' thinking.** *"always use the
    thinking to diagnose eval cases. almost always the model is acting rationally, the solution
    is always the inputs, the prompt, the input data, the tool shape, etc. the thinking often
    will tell us where the gap is"*
    - Read the thinking before proposing a lever.
    - The thinking is where the rational-actor rule (§3.1) meets a case: it is the model's
      account of the state it saw, so it points at the gap in the inputs, not at the model.

11. **A bug is first recreated as a test that reproduces it; that test validates the fix; the
    test stays as the regression guard.** *"the way i like to work in general is: if we
    encounter a bug, we first recreate it as a test to reproduce the failure; then we use that
    reproducing test to validate a fix; then we keep the test to catch any future regressions.
    that should also be a principle"*
   - A case whose first run doesn't show the failure doesn't test the bug, and the bug is
     closed as not reproducible.
   - A reproducing case that passes once the fix lands stays in the suite and is never deleted
     for being green.
   - A case whose failure an earlier change already fixed still lands as the guard.

## 5. How a change is judged

Four questions, all of them, for every change to Penny:

1. **Does the model keep its freedom to choose its actions?**
2. **Does it remove a failure, or a whole class of them?**
3. **Is it faster?**
4. **Does the user get what they asked for?**

**Judge the outcome, not the route.** *"i asked gemma for something, did i get what i want
back, regardless of the internal sequencing. if i did get what i asked for back, then gemma
succeeded."* When the user gets what they asked for (a true answer from real sources, the
store as they meant it, fast enough to chat), the turn succeeded, whatever tools ran in
whatever order. The route is read to diagnose a bad outcome, never to grade a good one.
