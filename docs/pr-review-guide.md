# Penny PR Review Guide

A comprehensive checklist for reviewing pull requests against the project's established patterns, conventions, and hard-won lessons. Every rule here comes from CLAUDE.md files or feedback from production incidents.

---

## 1. Code Style

### Pydantic for All Structured Data
- [ ] All structured data (API payloads, config, internal messages) uses Pydantic models — no raw dicts
- [ ] Every `Tool.execute(**kwargs)` validates through a Pydantic args model (e.g., `SearchArgs(**kwargs)`) as its first line
- [ ] Tool results use structured Pydantic models where applicable
- [ ] **`str | SomeModel` (more generally `str | T`) return/field types are an antipattern.** A union of "a bare string" and "a structured value" is a half-finished refactor: callers must branch on the runtime type, and the string carries no place for the metadata the structured arm has (success, mutated, errors, source URLs). Pick one contract and make it uniform. If *any* return needs structure, give *every* return that structure — one model with sensible defaults (`ToolResult(message, success=True, mutated=False, ...)`) that the plain case constructs trivially — so the consumer has a single branch. The same applies to a method that returns `str` on success and a model on failure: model both. Seen repeatedly (tool returns, dispatch results); the fix is always "structure everything," never "union the two."

### Constants and Naming
- [ ] All string literals in logic are defined as constants or enums — no magic strings
- [ ] **All shared values — strings, numbers, emojis, thresholds, timeouts, magic numbers, batch sizes, retry counts, well-known identifiers — live in `penny/constants.py` (under `PennyConstants` or a related enum class). This applies regardless of where the constant is declared: NEVER as a module-level `_PRIVATE_CONSTANT = ...` at the top of `agents/foo.py` or `tools/bar.py`, AND NEVER as a `class FooThing: SOMETHING = "..."` class attribute on a domain class somewhere.** When a value is shared across more than one call site (or *will be*, e.g. a tool emits a value that another module renders), it belongs in `constants.py`. If the value is enumerable (a fixed small set of options), use a `StrEnum` so consumers reference symbolic names (`ProgressEmoji.THINKING`) rather than raw literals (`"\U0001f4ad"`). The only exceptions are file-local regex patterns (`_FOO_PATTERN = re.compile(...)`) that exist purely as a compiled-once optimization, and truly file-private values that no other module imports or duplicates.
- [ ] **Class-specific properties live on the class, not globally or in `constants.py`.** A class's own configuration — its identity, its loop caps, its mode names — belongs on the class as a class attribute (e.g. `HistoryAgent.PREFERENCE_EXTRACTOR_NAME = "preference-extractor"`). It is not a shared cross-cutting value, so it does not belong in `PennyConstants`, and it is not a module-private internal, so it does not belong as a module-level global.
- [ ] **No constant declared in BOTH the Python backend AND the TypeScript browser extension.** A page size, timeout, threshold, or any other shared value must have exactly one source of truth. Pick a side and have the other side either receive the value over the wire (backend → addon in payloads), request behavior in units the source side defines (e.g. addon asks for "N pages", backend owns the page size), or simply derive it from data the source sends. A "mirror constant" with a comment claiming to track the other side will drift — they always do. If you find yourself writing `// mirrors X on the server` or `# mirrors Y in protocol.ts`, that is the failure mode this rule exists to prevent.
- [ ] Variable names are fully spelled out — no abbreviations (`message` not `msg`, `config` not `cfg`, `format_args` not `fmt`). Standard short names (`i`, `n`, `db`) in tight loops or established domain terms are fine
- [ ] f-strings used everywhere — no string concatenation with `+`

### Method Size and Structure
- [ ] Every method is roughly 10-20 lines (hard max ~25). Long methods are decomposed into named steps via extraction — no new abstractions, just decomposition
- [ ] Every class has a summary method (after `__init__`) that composes calls to other methods, reading like a table of contents
- [ ] **No trivial pass-through methods on subclasses that re-state a parent's default.** If the parent has `def foo(self): return self.SOMETHING`, subclasses customize the value by overriding the class attribute (`SOMETHING = 16`), not by redeclaring the method. The pattern `def get_max_steps(self): return self.MAX_STEPS` on a subclass is a smell — the parent should define that body, and the child only sets the class attribute. Subclasses that genuinely need *different logic* (e.g., reading runtime config) override the method legitimately
- [ ] **No `_method = staticmethod(module_function)` aliases.** Don't bind a module-level function to a class as a "method" purely to expose it under `self.` — call sites should import and call the function directly. The class isn't adding behavior; the alias is just indirection
- [ ] **No keyed-indirection enums or attributes for single-value identifiers.** When an agent (or any class) has exactly one of something — one prompt type, one mode, one variant — don't wrap it in a `class FooType(StrEnum): CYCLE = "cycle"` with a `prompt_type = FooType.CYCLE` attribute on the class. The enum has one member, the attribute restates the agent's identity, and call sites pay the indirection cost for nothing. Use the agent's `name` (or other existing identity) directly, or hardcode the single string on the class. Multi-value enums where each variant maps to a distinct flow (e.g. `ChatPromptType` with `USER_MESSAGE`/`VISION_MESSAGE`/`VISION_CAPTION`) are still legitimate
- [ ] **No per-agent plumbing for values that are universal under the unified-tool-surface principle.** Every agent gets every tool with identical capabilities. So values that the same tool needs at execute time (recipient for `send_message`, primary user for cooldown filtering, etc.) belong inside the tool — resolved from `db` or shared state at call time — not threaded through agent constructors, `_run_cycle(user)` parameters, or `get_tools(user)` overrides. Penny is single-user, so per-agent recipient binding has nothing to choose between; let the tool look it up. Per-agent metadata that genuinely varies (the agent's own `name` for write attribution and cooldown filtering) is fine to plumb — but only the things that actually differ between agents

### Imports
- [ ] All imports are at the top of the file — no inline/inner imports inside functions or methods
- [ ] If a circular import exists, the fix is to move the shared type to a common location (e.g., `base.py`), not to defer the import
- [ ] `TYPE_CHECKING` guards are only used for type-only imports that would cause real circular dependency at runtime
- [ ] **An inner import is justified ONLY to break a *verified* runtime cycle.** Before accepting one, check that the cycle is real: if the imported symbol's module has no heavy/cyclic dependencies (a `types.py` of enums/errors, say) and is — or could be — imported at the top of the file without a cycle, the inner import is unjustified indirection; hoist it. "It's only used in one method" is not a reason to bury an import inside that method

### Dead Code
- [ ] No unused constants, variables, methods, or imports left behind after changes
- [ ] Follow the chain — if removing a method, also remove constants it was the only consumer of
- [ ] Follow the chain in BOTH directions — when you delete the only *producer* of a value, its type/`NamedTuple`/helper and every re-export of it are now dead too (deleting the method that built a `RunRecord` makes `RunRecord`, its import, and its `__all__` entry dead). A symbol kept "just in case" after its sole caller or sole producer is gone is dead code — remove it
- [ ] No `del param` statements at the top of a function/method body to "consume" an unused argument — this is dead-code dressing for a linter, not real code. If a parameter is genuinely unused but required by the override signature (e.g., parent class contract), document *why* in the docstring and leave the parameter alone. If the parameter isn't required, remove it from the signature

### Optional Values
- [ ] Optional fields use `None` (`str | None = None`), never empty string defaults (`""`)
- [ ] `""` serializes as empty string over the wire, which is NOT nullish in JS/TS — `"" ?? fallback` evaluates to `""`, breaking null-coalescing

---

## 2. Database

### Schema and Foreign Keys
- [ ] All relationships use proper FK references (`preference_id INTEGER REFERENCES preference(id)`) — never denormalize by storing copies of data from another table
- [ ] New tables follow the SQLModel pattern in `database/models.py`

### Ordering
- [ ] Datetime columns (`created_at`, `timestamp`, `learned_at`, etc.) used for recency ordering in all queries
- [ ] Auto-increment `id` columns are NEVER used to infer chronological order — IDs are for joins and lookups only

### Store Pattern
- [ ] Database access goes through domain-specific store classes (`db.messages`, `db.preferences`, `db.thoughts`, etc.)
- [ ] The `Database` class is a thin facade — no business logic, just creates and exposes stores
- [ ] Access pattern: `self.db.messages.log_message(...)`, NOT `self.db.log_message(...)`

### Migrations
- [ ] New migrations are numbered sequentially in `database/migrations/`
- [ ] Migration files have a `def up(conn)` function
- [ ] No duplicate migration number prefixes (enforced by `make migrate-validate`)
- [ ] Schema migrations use DDL (ALTER TABLE, CREATE INDEX); data migrations use DML (UPDATE, backfills)
- [ ] A single PR introduces at most one migration — multiple schema changes should be combined into one migration file

### Production Data
- [ ] **NEVER** modify production DB (runtime_config, preferences, thoughts, etc.) without explicit user approval
- [ ] Don't question or investigate whether runtime_config values are "taking effect" — if a value exists in the DB, the user put it there intentionally
- [ ] Never use RuntimeConfig to store application state (watermarks, cursors, progress trackers) — derive state from the relevant domain table's timestamps and foreign keys

### Data-Source Refactors (Replica → Facade, Table Moves, Row Deletes)
- [ ] When you change where data is read from, or delete rows in a migration, enumerate EVERY reader and migrate all of them: the model-facing tool, the addon/UI, recall, background cron, tests. A facade wired into one consumer leaves the others reading the now-empty source — and the suite stays green if those consumers have no test
- [ ] A read facade over a canonical table routes ALL of the base's read paths (`read_latest`/`read_since`/`read_recent`/`read_all`/`_embedded_rows`/counts) through the overridden source. Prefer overriding the low-level row primitives so every higher-level read inherits the new backing — a partially-overridden facade returns correct data on some methods and empty/stale on others
- [ ] A read view over a live append-only table filters to a completion sentinel (e.g. an outcome stamp) so it never surfaces in-flight rows — especially when the reader is also one of the writers (it must not act on its own partial run)
- [ ] The write path populates what the read path needs; a startup/periodic backfill is for historical rows only. If the write path already computes a value (an embedding computed for another purpose), persist it there — don't discard it and lean on the next restart's backfill, or the read path is stale for the whole interval
- [ ] A read-only facade over a canonical table has NO write path of its own: its `append`/`write` raises (e.g. `ReadOnlyMemoryError`), and tests/fixtures seed the *canonical* table directly (e.g. `log_message` for a messages facade, a `promptlog` run for a runs facade) — never the facade. A facade write that silently no-ops, or writes the wrong table, is a bug
- [ ] Dispatch that resolves a name to its facade/object depends on the registry/marker row existing — so the facade rows seeded by migrations in prod must also be seeded in any test that resolves them. `db.memory("user-messages")` returns `None` if the marker row is absent, and the call chain then `None`-dereferences or silently no-ops

### SQLModel Table Renames
- [ ] Renaming a `table=True` SQLModel class silently renames the physical table — and breaks every FK and every migration's `UPDATE`/`DELETE` target — unless `__tablename__` is pinned to the original name. Pin it whenever you rename the class but not the table

---

## 3. Query Efficiency and Performance

### Push Work Into SQL — Don't Materialize Then Paginate
- [ ] Filtering, pagination, cursors, and "latest N" go into the query (WHERE / ORDER BY / LIMIT / `since`), never materialize a whole table (or a whole partition of one) and slice/filter/break in Python
- [ ] `read_latest(k=1)` must issue `LIMIT 1` — not fetch every row and take `[0]`
- [ ] Red flag: `for row in store.read_all(): if <cond>: break` when only rows since a cursor are needed → use a `read_since(cursor)`-style bounded query. The early `break` saves iteration, not the fetch — the full result set is already in memory
- [ ] This does NOT conflict with "never invent limits" (Forbidden Patterns): a `LIMIT` that honors a caller's explicit `k` / page-size / cursor is correct; an *invented* cap the caller never asked for is the forbidden one

### Index Every Filter and Sort on a Growing Table
- [ ] Every WHERE column and ORDER BY column on an unbounded table (`messagelog`, `promptlog`, `memory_entry`) is index-backed — check `models.py` for `index=True` or a composite `__table_args__`
- [ ] A filter/sort on an unindexed column is a full scan that grows with history — flag it
- [ ] **A query that BOTH filters (`WHERE a = ?`) and sorts (`ORDER BY b`) needs a composite index that LEADS with the equality-filtered column, then the sort column (`(a, b)`).** An index on the sort column alone (`(b)`) lets the planner walk it already-ordered — but it must read and test `a` on every row, so a *sparse* match (`a = ?` hits few rows among many) walks the entire ordered index before `LIMIT` can fill. This is the trap that bit the per-collection runs panel: an index on `(timestamp)` made the plan *look* indexed while it scanned ~29k rows to find 3, a multi-second freeze
- [ ] **A partial index (`... WHERE outcome IS NOT NULL`) only helps a query whose filter it covers AND whose remaining filter/sort its key columns serve.** A partial index keyed on `(timestamp) WHERE outcome IS NOT NULL` does nothing for a query that also filters `target = ?` — make `target` the leading key (`(target, timestamp) WHERE outcome IS NOT NULL`). Keep the old index only if a different live query (e.g. the unscoped one) still needs the sort-only shape
- [ ] Leading-wildcard `LIKE '%x%'` cannot use an index — acceptable only for on-demand, bounded searches (addon entry search), never on a hot path
- [ ] Loading the full embedded corpus for vector/similarity search is expected (SQLite has no ANN index) — but a cooldown probe, a "latest one" lookup, or a count must be a bounded query, not a corpus load

### Verify the Plan, Not Just That an Index Exists
- [ ] For any new or changed hot-path query, confirm the plan with `EXPLAIN QUERY PLAN` against a **production-scale** copy of the DB — don't assume an index is used just because one exists on a column the query names
- [ ] `SEARCH ... USING INDEX (col=?)` is a seek (good). `SCAN ... USING INDEX ...` is a full walk of the index — the planner picked it for ordering or as a covering scan, but it still reads every entry; for a filtered query with `LIMIT`, that walk *is* the regression, not the fix
- [ ] Test the worst case: a *sparse* filter value (a target/collection/sender with few matching rows among a large total). A query that's fast when matches are dense degrades to a full scan when they're sparse and the `LIMIT` can never fill

### Don't Pay Whole-Corpus Cost on a Single-Item Path
- [ ] A method that aggregates across EVERY item (counts for all memories, a full-corpus load, a `GROUP BY` over an unbounded table) must not be called on a path that needs one item's value — e.g. a detail/click handler that opens ONE memory. Fetch just the one (a scoped `COUNT(...) WHERE name = ?`, a single-row lookup); reusing the all-items aggregate makes every invocation pay a cost that grows with total history
- [ ] Reusing a list-view helper on a detail view is the common disguise: it's correct, it's already written, and it's O(whole DB) per call. Flag when a "give me everything" method is called inside a "show me one" handler

### Aggregate and Detail Must Agree
- [ ] A count (inventory) and a listing (detail) derived from the same data use equivalent predicates, so the number shown matches what a read returns

---

## 4. Architecture and Design

### Python-Space Over Model-Space
- [ ] Deterministic actions (posting comments, creating labels, validating output) are handled in Python code, not delegated to the LLM
- [ ] Model-space is reserved for tasks that genuinely need reasoning (writing specs, analyzing code, generating responses)

### Pass Parameters, Don't Swap State
- [ ] No temporary swapping of instance state (e.g., `self.db`) to change behavior
- [ ] Dependencies are passed as parameters through the call chain

### No Ambient State for Cross-Cutting Concerns
- [ ] No `contextvars.ContextVar` for application logic (current user, current agent, current request, current author, etc.)
- [ ] No module-level globals that get mutated to "share" state between call sites
- [ ] No `get_current_X()` / `set_current_X()` function pairs that thread implicit state through the call chain
- [ ] If multiple call sites need the same value (author, user, request_id, tenant), pass it explicitly as a parameter — at construction time when stable for the instance's lifetime, or as a method argument when it varies per call
- [ ] Template method override + explicit dependency injection is the project pattern. Each subclass exposes its identity as a class attribute (e.g., `Agent.name`); callers read it directly and pass it through to whatever needs it
- [ ] Why: ambient state makes call graphs invisible, requires global setup/teardown in tests, leaks between concurrent async tasks, and breaks under refactor — anyone wiring a new caller silently inherits whatever the surrounding context happens to hold

### Template Method Over Conditionals
- [ ] Multiple modes/variants use building blocks composed by each variant — no flags or if/else chains
- [ ] Examples: agent system prompts (building blocks like `_identity_section()`, `_profile_section()`), notification modes (`NotificationMode` subclasses)

### Compositional Pattern (Two-Layer)
- [ ] When multiple variants share a pipeline but differ in configuration: (1) prompt composition — each mode picks building blocks, (2) pipeline composition — each mode declares properties the orchestrator reads
- [ ] New modes = new class, no touching the pipeline
- [ ] Preferred over if/else chains, flag-based toggling, or deep class hierarchies

### Polymorphic Dispatch Over Scattered Type Checks
- [ ] When behavior depends on a runtime name / type / shape, resolve it ONCE through a single dispatch to the right object (e.g. `db.memory(name)` returns a `Collection`, `Log`, or a facade), and have every caller call methods on that object. Don't repeat `if name in SOME_SET` / `if obj.type == X` at each call site, and don't scatter the same branch inside every method of a shared store/access layer
- [ ] The "facade" / variant behaviour belongs in **a class per variant** that overrides the operations — not as conditionals threaded through the callers and the access layer. A tool, recall path, or UI handler should never branch on the name or shape of the thing it was handed; the object encapsulates that
- [ ] Operations that don't apply to a variant refuse via a base-class no-op that raises a typed error — not by callers checking the type first (this is "No getattr Duck Typing" applied to dispatch: the base defines every op, each variant overrides the ones it supports)
- [ ] Put shared logic as high in the hierarchy as it goes (the common backing on the base; only the genuinely different variants override) — and don't manufacture empty pass-through subclasses for variants that add nothing over the base
- [ ] Smell: the same `if name == "X"` / `if isinstance(...)` / `if obj.type == ...` appears in more than one method or more than one caller → introduce the dispatch + polymorphic classes and delete every copy of the conditional

### Hoist Cross-Cutting Entrypoint Boilerplate Into a Base Template
- [ ] When N sibling classes repeat the same wrapper in their public entrypoint — the same `try/except → return str(exc)`, the same up-front validation, the same result framing — hoist it into a base class whose entrypoint (`execute`) wraps an abstract hook (`_run`) the subclasses implement. The cross-cutting concern is authored once; each subclass contains only its distinct body
- [ ] This is the template-method pattern applied to *boilerplate* (distinct from applying it to variant building blocks). Smell: the identical try/except or guard appears verbatim at the top or bottom of every subclass's `execute`
- [ ] Only the subclasses that actually share the concern inherit the base — don't force-fit siblings that don't (a tool that resolves+operates on a memory belongs on `MemoryTool`; one that does neither stays on plain `Tool`)

### No Client-Server Duplication
- [ ] Transformations exist in exactly one place — if the server does markdown-to-HTML, don't reimplement in the client
- [ ] Before writing a transformation, check if it already exists elsewhere

### No getattr Duck Typing
- [ ] Never use `getattr(obj, "method_name", None)` to check method availability
- [ ] Define methods on the base class as no-ops (with `return` to satisfy B027), override in subclasses that implement them

### No Shared State Races
- [ ] Async tasks receive dependencies as parameters — never reach into shared dicts/registries from async tasks expecting data to still be there
- [ ] Pass references directly when spawning tasks, don't have tasks look them up later

### Queues Over Locks
- [ ] For serializing async operations, prefer `asyncio.Queue` with a worker task over `asyncio.Lock`
- [ ] Pattern: callers enqueue `(data, result_future)` tuple and await the future; worker pops, processes, resolves

### Initialize at Startup
- [ ] Heavyweight setup (copying databases, creating resources) belongs at startup, not lazily inside handlers
- [ ] Static data captured at Docker build time via build args, not parsed at runtime

---

## 5. Error Handling

### Narrow Exceptions
- [ ] Catch the exact exception type expected (`asyncio.CancelledError`, `TimeoutError`, etc.)
- [ ] Never use `contextlib.suppress(Exception)` or `suppress(BaseException)` — they hide real bugs
- [ ] If multiple exceptions are possible, list them explicitly

### No Silent Fallbacks
- [ ] Never add `dict.get(key) or ""` or `or 0` or `or []` just to avoid dealing with a missing value — these mask bugs
- [ ] If a value might be absent, handle it with correct logic (e.g., `datetime.min` as a sortable sentinel) or let it raise
- [ ] Ask: does this default value make sense in downstream logic? If not, it's a bug mask

### No Silent Error Swallowing
- [ ] Never write catch blocks that silently swallow errors and fall through to a "fallback" implementation
- [ ] If a primary path fails, it must fail loudly (log the error, return an error state)
- [ ] Multiple strategies must be independently tested and selection must be explicit, not error-driven

### Errors Are Exceptions, Not Foreign-Typed Sentinel Returns
- [ ] A function that can fail raises a typed exception — it does NOT return `T | str` (the `str` being an error message) or any union that mixes an error sentinel of a *different type* into the success channel. The success value is unusable until every caller discriminates it, so the sentinel just pushes a type check onto each call site
- [ ] Smell: `x = resolve(...); if isinstance(x, str): return x` repeated at many call sites → `resolve` should raise, and the callers (or a shared base) catch it once
- [ ] A `None` return for "absent" is fine when absence is an ordinary, expected outcome handled inline; it is NOT fine as a stand-in for an error that carries a message — that's an exception

### Tool Failures Are Actionable to the Model
- [ ] Every tool failure surfaced to the model (a `ToolResult` with `success=False` — validation error, rejected/degenerate input, refused operation, missing key, wrapped external error) MUST tell the model two things: (1) *what went wrong* — the specific reason, naming the offending field or value — and (2) *how to correct it* — the concrete next action (provide a non-empty value, supply the full replacement text, call the right alternative tool). The tool result is the model's only recovery signal
- [ ] Reject a bare verdict with no remedy ("rejected", "invalid input", "error") and a silent no-op (returning success on an operation that changed nothing without saying why) — both leave the model nothing to act on, so it retries the same mistake or gives up. A diagnosis without a remedy is a half-failure
- [ ] Good: `check_extraction_prompt` quotes the actual length and the minimum and points at the required prompt shape; `update_entry`'s degenerate-content refusal names the reason AND suggests `collection_delete_entry` if removal was intended. Bad: `"Refused: content rejected."`

### Exceptions Self-Render and Share a Catchable Base
- [ ] Every exception in a family carries its data in `__init__` AND renders its own complete, surface-ready message via `str(self)`. Don't make one exception contain the whole message while a sibling carries only a bare name that each caller has to wrap in a format string — the handling should be uniformly `return str(exc)` for all of them
- [ ] Related exceptions that callers handle the same way share a base class, so one `except Base` (or a single tuple) covers them in one place — instead of a separate `except` + custom message per subclass scattered across callers
- [ ] Why: when the message lives in the exception it's authored once and stays consistent at every catch site; when it lives at the catch site, the same error renders different (and drifting) text depending on who caught it. Keep a subclass relationship that preserves existing `except` callers (e.g. a shape error that subclasses the broader type error those callers already catch)

### Verify Primary Path First
- [ ] Never write fallback/alternative code paths before verifying the primary path works with real output
- [ ] Pattern: (1) write primary path, (2) build and test against real input, (3) confirm correct output, (4) only then consider fallbacks

### Verify Imports
- [ ] When adding a new library dependency, verify the import resolves correctly (default vs named exports)
- [ ] Check the actual export shape — don't assume

### No Asserts in Production Code
- [ ] No `assert` statements in production (non-test) code — assertions get stripped under `python -O` and silently disable runtime checks
- [ ] Never use `assert x is not None` to satisfy a typechecker. Healthier patterns:
  - If the value being None is unreachable in practice but the type is `T | None`: narrow with `if x is None: continue` (skip), `raise ValueError(...)` (fail loudly with context), or refactor the upstream type so `None` isn't possible
  - If the value being None means a real bug: `raise` with a descriptive message, never `assert`
- [ ] `assert` is reserved for tests, where strip-on-optimize doesn't apply

---

## 6. Testing

### Test Invocation
- [ ] Tests run ONLY via `make fix check 2>&1 | tee /tmp/check-output.txt; echo "EXIT_CODE=$pipestatus[1]" >> /tmp/check-output.txt`
- [ ] Never use `make pytest`, `make check` alone, `docker compose run`, or any other variation
- [ ] Check EXIT_CODE first, then grep for FAILED or `error[` as needed

### All Changes Require Tests
- [ ] Every code change has corresponding test coverage — tests are part of the implementation, not a follow-up
- [ ] If all tests pass after behavior changes, that indicates a coverage gap — add tests that would fail if the change were reverted

### Integration Tests Preferred
- [ ] Test through public entry points (`agent.run()`, `has_work()`, full message flow) — not internal functions in isolation
- [ ] Unit tests only for pure utility functions with many edge cases (CODEOWNERS parsing, config loading)
- [ ] Mock at system boundaries (Ollama, Signal, GitHub CLI, Claude CLI) but let internal code execute end-to-end

### Test Organization
- [ ] Tests organized in this order: (1) comprehensive happy-path integration tests, (2) special success cases, (3) error/edge cases, (4) unit tests at the bottom
- [ ] Each primary variant/mode has a comprehensive test that exercises the entire code path

### Fold Into Existing Tests
- [ ] Prefer adding assertions to an existing test that covers the relevant code path over creating a new test function
- [ ] Only add a new test when no existing test covers the relevant code path

### Deterministic Tests
- [ ] All `random.random`, `random.choice`, and other random calls are monkeypatched in tests that assert on specific codepaths
- [ ] Even if a test "usually" takes the right path, a 1-in-3 chance of hitting the wrong branch causes flaky CI

### No Real Timers
- [ ] Use `wait_until(condition)` instead of `asyncio.sleep(N)` — poll for the expected side effect with a generous timeout
- [ ] For negative assertions (nothing should happen), verify immediately — don't sleep "to make sure"

### Cover All Codepaths
- [ ] Features work for ALL variants/modes — don't silently bail on a subset (e.g., seeded vs free thoughts, manual vs extracted preferences)
- [ ] If a variant genuinely needs different handling, call it out rather than silently skipping

### Exit Code Is Truth
- [ ] `make check` exit 0 = pass, exit 1 = fail. If it fails, something needs fixing
- [ ] Never attribute failures to "pre-existing issues" — main is always green (branch protection enforced)
- [ ] Never stash changes and test main to check if failures are pre-existing

---

## 7. Prompt Engineering

### System Prompt Structure
- [ ] Consistent `##` / `###` header hierarchy to delineate sections
- [ ] Standard structure: `## Identity` → `## Context` (with `###` subsections) → `## Instructions`
- [ ] Every agent overrides `_build_system_prompt(user)` composing from building blocks
- [ ] Tests assert on the exact full system prompt string to catch structural drift

### No Conflicting Instructions
- [ ] Read each instruction and verify it doesn't contradict another
- [ ] Thinking models are especially sensitive — contradictory signals cause extensive deliberation and empty output

### No Context Fixation
- [ ] Don't inject an agent's own previous outputs into its context unless there's a specific reason (like scoped dedup)
- [ ] Free-form previous outputs prime the model to revisit the same topics repeatedly

### Dry-Run Prompt Changes
- [ ] When modifying any LLM prompt, dry-run against real production prompt logs before deploying
- [ ] Test on 3+ diverse examples, not just the one that triggered the change
- [ ] Compare old output vs new output side-by-side

### Check Model Thinking
- [ ] When investigating prompt issues, check the `thinking` field in the `promptlog` table
- [ ] Search for keywords: "conflict", "but the instructions say", "contradicts", "wait,", "not sure if"

### Renamed/Removed Tools Leave No Dangling Model-Facing References
- [ ] When deleting or renaming a tool/capability, grep model-facing strings for the old name: tool descriptions, returned error/refusal messages, and stored `extraction_prompt`s (in migrations). A dangling reference points the model at a tool that no longer exists — it silently follows a dead pointer. A code-symbol grep is not enough; the string is the bug

---

## 8. Forbidden Patterns

These patterns have each caused production bugs or wasted days of debugging. Flag them immediately.

### CRITICAL: Never Invent Arbitrary Truncations or Limits

This is the single most recurring source of production bugs in this project. It has caused **days** of cumulative debugging across multiple incidents. The pattern is always the same: a "reasonable" limit or default is added that nobody asked for, it silently discards or corrupts data, and the resulting bug is far harder to trace than the original problem would have been.

**The rule is absolute: never invent a value the user didn't ask for.**

- [ ] No new `max_tokens` values — adding `max_tokens: 600` to a call caused a model to stop mid-thought before writing the actual response, making it look like a model bug when it was self-inflicted truncation
- [ ] No new character limits — no `content[:500]`, no `text[:1000]`, no `summary[:200]`
- [ ] No new `.slice()` caps in TypeScript — no `results.slice(0, 5)`, no `items.slice(0, 10)`
- [ ] No new array/list length caps — no `items[:N]` unless the user specified N
- [ ] No new `max_results`, `max_items`, `max_length`, `max_chars` parameters with invented defaults
- [ ] No lossy data transformations that discard information "to be safe"
- [ ] No "reasonable defaults" for limits — what seems reasonable is wrong; it silently breaks things downstream and the resulting bug is always harder to diagnose than the original problem
- [ ] Existing limits that are already in the codebase (like `MAX_CHARS` in `extract_text.ts`) are there for a tested reason — don't remove those. The rule is about not **inventing new ones**
- [ ] Build the simplest thing first with NO truncation, ship it, and only address breakage when it actually happens — not preemptively
- [ ] If the user explicitly asks for a limit with a specific value, implement exactly that value — don't round it or "improve" it

**How to spot this in review:** Search the diff for any newly introduced numeric literal that caps, truncates, slices, or limits data. If the PR description doesn't say "user requested this limit of N," reject it. Common disguises:
- `[:N]` slicing on strings, lists, or query results
- `max_tokens`, `max_length`, `max_results` parameters
- `TRUNCATION_LIMIT`, `MAX_ITEMS`, `RESULT_CAP` constants
- `.slice(0, N)` in TypeScript
- `content[:N] + "..."` ellipsis truncation
- `if len(x) > N: x = x[:N]` guard clauses
- `LIMIT N` in SQL queries that didn't have one before
- Default parameter values that cap output (e.g., `def get_items(limit: int = 10)`)

### Never Use Arbitrary Thresholds
- [ ] Don't hardcode numeric thresholds ("after 3 tool calls, do X") when the behavior should be based on structural conditions ("are tools still available?")
- [ ] If a threshold is truly needed, derive it from the relevant config parameter

### Never Loop-and-Bail for Independent Items
- [ ] When processing a list of independent items, NEVER use a pattern where one item's failure blocks all others
- [ ] Each independent item must be processed on its own merits — no "pick the best one, try it, bail if it fails"

### No Monkeypatching Library Internals
- [ ] Only use publicly exported library APIs
- [ ] If a capability isn't publicly exported, implement it independently — don't hook into internals

### No Cloud Assets
- [ ] All assets (CSS, fonts, icons, JS libraries) bundled locally — never loaded from CDNs or external URLs
- [ ] Install via npm and reference from node_modules or copy into the project

### No Abbreviated Variable Names
- [ ] `message` not `msg`, `config` not `cfg`, `format_args` not `fmt`, `context` not `ctx`

### No Default Empty Values
- [ ] `None` for optional values, never `""`

### No getattr Duck Typing
- [ ] Define methods on base class, don't use `getattr(obj, "method", None)`

### No Silent Fallbacks
- [ ] `or ""`, `or 0`, `or []` as fallbacks are bug masks

---

## 9. Async Patterns

- [ ] `asyncio.Queue` with worker for serialization, not `asyncio.Lock`
- [ ] Pass dependencies directly to async tasks — don't fish from shared dicts
- [ ] Catch `asyncio.CancelledError` specifically (it's `BaseException` in Python 3.9+, not `Exception`)
- [ ] Background tasks must be idempotent — cancelled work stays in queues and retried next cycle

---

## 10. Browser Extension (TypeScript)

- [ ] Related content rendered in the same DOM container — not as separate siblings in scrollable lists
- [ ] No client-side reimplementation of server-side transformations
- [ ] All assets bundled locally, no CDN references
- [ ] Strict extractors for known page types — no generic fallback when the fallback produces garbage
- [ ] Known page types use a `ready` flag and are excluded from the generic fallback chain
- [ ] Verify library imports resolve (default vs named exports) before building on them

---

## 11. Git and Workflow

- [ ] All changes go through PRs — never push directly to `main`
- [ ] Feature branches created from latest `main` (`git checkout main && git pull origin main` first)
- [ ] Rebase on latest `main` before building, testing, or committing
- [ ] Check PR is still open before every `git push` (`gh pr list --head <branch> --state open`)
- [ ] Push branch before `gh pr create` (requires branch to exist on remote)
- [ ] Use `make token` for all GitHub operations: `GH_TOKEN=$(make token) gh pr create ...`
- [ ] CLAUDE.md and README.md updated when making significant changes (new features, architecture, config, API, directory structure)

---

## 12. Similarity and Embedding Code

- [ ] All similarity logic lives in the `similarity/` package — agents don't implement their own
- [ ] `similarity/embeddings.py`: Pure math (cosine similarity, TCR, serialize/deserialize)
- [ ] `penny/ollama/similarity.py`: Composed operations using embeddings + OllamaClient

---

## 13. Response Style

- [ ] Sentence case for all Penny response strings ("Okay, I'll learn more about {topic}")
- [ ] Markdown tables converted to bullet points in Python (saves model tokens)

---

## Quick Reference: Red Flags

If you see any of these in a PR, flag immediately:

| Pattern | Why It's Wrong |
|---|---|
| `or ""` / `or 0` / `or []` as fallback | Masks bugs, breaks null-coalescing in JS/TS |
| `except Exception:` / `suppress(Exception)` | Too broad, hides real bugs |
| `getattr(obj, "method", None)` | Duck typing bypasses type system |
| `max_tokens: 600` or any invented limit | Self-inflicted truncation, days of debugging |
| `content[:500]` or any `[:N]` slice on data | Silent data loss, nobody asked for this |
| `.slice(0, N)` in TypeScript | Same as above, JS edition |
| `LIMIT 10` added to a query that had none | Silently drops results |
| `def foo(limit: int = 10)` new default cap | Invented constraint, will bite later |
| `if len(x) > N: x = x[:N]` guard clause | Preemptive truncation that masks real issues |
| `asyncio.sleep(N)` in tests | Fragile timing, use `wait_until()` |
| `from foo import bar` inside a function | Hidden dependency, reorganize modules instead |
| Raw dict passed through system | Must use Pydantic model |
| `ORDER BY id DESC` for recency | Must use datetime column |
| `self.db.do_thing(...)` bypassing store | Must go through `self.db.store.do_thing(...)` |
| Inline `"magic string"` in logic | Must be a constant or enum |
| `msg`, `cfg`, `fmt`, `ctx` variable names | Spell it out fully |
| `try: ... except: <fallback code>` | Primary must fail loudly, not silently fall through |
| `field: str = ""` on Pydantic model | Use `str \| None = None` |
| CDN link for CSS/JS/fonts | Bundle locally |
| `for x in store.read_all(): … break` | Loads whole table into memory then paginates in Python — push the filter into SQL |
| WHERE/ORDER BY on a column with no `index=True` | Full scan that grows with history |
| `WHERE a = ? ORDER BY b` with an index on `(b)` but not `(a, b)` | Sparse `a` walks the whole sort-ordered index before `LIMIT` fills — lead the index with the filter column |
| `SCAN … USING INDEX` for a filtered query with `LIMIT` | Index covers the sort but not the filter — still a full walk; confirm `SEARCH … (col=?)` with EXPLAIN QUERY PLAN at production scale |
| Whole-corpus aggregate (counts-for-all, full-corpus load) called from a single-item/detail handler | Per-call cost grows with total history — fetch just the one item's value |
| Renaming a `table=True` SQLModel class without `__tablename__` | Silently renames the physical table → breaks FKs and every migration's `UPDATE`/`DELETE` target |
| Facade/view that overrides only *some* read methods | Returns correct data on some calls, empty on others |
| Value computed on write path but only persisted by a backfill | Read path stale until next restart |
| Renamed/removed tool still named in a tool description or stored prompt | Model follows a dead pointer to a tool that no longer exists |
| `if name in SET:` / `if obj.type == X:` repeated across methods or callers | Scatter-branching on type/shape — dispatch once to a polymorphic object, delete the conditionals |
| `x = f(...); if isinstance(x, str): return x` | Error returned as a foreign-typed sentinel — raise a typed exception and catch it |
| Exception caught then wrapped in `f"... {name} ..."` while a sibling self-renders | Make the exception render its own message; handle uniformly via `str(exc)` |
| Same `try/except: return str(exc)` (or guard) at the top of every subclass `execute` | Hoist into a base template method that wraps an abstract `_run` |
| `from foo import Bar` inside a method when there's no real cycle | Inner import only for a *verified* runtime cycle — else hoist to top |
| Read-only facade with a write/`append` that no-ops instead of raising | Facade has no write path — raise; seed the canonical table in tests |
| Type/`NamedTuple`/re-export kept after its only producer was deleted | Dead code — follow the chain in both directions |
