# CLAUDE.md — Penny Project

## What Is Penny

Penny is a local-first AI agent that communicates via Signal, Discord, or a Firefox browser extension. Users send messages, Penny searches the web through the browser extension, reasons using a local LLM (Ollama by default, accessed via the OpenAI Python SDK against any OpenAI-compatible endpoint), and replies in a casual, relaxed style. It runs in Docker with host networking.

Penny is single-user — a personal assistant deployed locally for one person. Multiple devices (Signal phone, browser instances) connect as different devices of the same user, sharing a single conversation history.

Penny also has an autonomous development team (`penny-team/`) — Claude CLI agents that process GitHub Issues on a schedule, handling requirements, architecture, and implementation.

## Environment Notes

- **Logs**: Runtime logs are written to `data/penny/logs/penny.log`; agent logs are in `data/penny-team/logs/` (not docker compose logs)

## Git Workflow

Branch protection is enabled on `main`. All changes must go through pull requests.

- **Never push directly to `main`** — always create a feature branch
- Create a descriptive branch name (e.g., `add-codeowners-filtering`, `fix-scheduler-bug`)
- Commit changes to the branch, then push and create a PR
- **Use `make token` for GitHub operations** (host only): `GH_TOKEN=$(make token) gh pr create ...`
  - This generates a GitHub App installation token for authenticated `gh` CLI access
  - Agent containers already have `GH_TOKEN` set by the orchestrator — just use `gh` directly
- The user reviews and approves the PR (code-owner review); the **merge queue** does the merging — flag the PR with `gh pr merge <n> --auto --squash` ("merge when ready") so it enqueues itself once approved and green, runs the `merge_group` checks against latest `main`, and merges with no manual step

## Agent Supervision (task-agent fleets)

When work is fanned out to task agents (each owning one issue per `docs/agent-task-workflow.md`), a **supervisor** — the parent Claude session or the user — owns the fleet. **The operating procedure — bootstrapping a fresh session onto a fleet (from the meta ticket + live queries, never a prior session's summary), meta-ticket conventions, dispatch mechanics (worktree isolation, Opus for implementation subagents), wave planning, fleet-end — is [`docs/agent-supervisor-runbook.md`](docs/agent-supervisor-runbook.md); start there.** The SOP is the child's contract; these are the supervisor's standing duties:

- **Assignment**: one issue per agent, an explicit scope boundary, the SOP as its operating contract.
- **Heartbeat (the load-bearing duty)**: while any child is blocked on a serialized resource (the eval GPU queue) or a long external wait, check the fleet on a timer — every 30–60 minutes. Verify each waiting child's watched process still exists and its result artifact is progressing. A dead waiter never resurrects itself, and a resting agent wakes only when something wakes it: an unheartbeated fleet can sleep all night on top of finished results (this happened — four green-gated branches sat unpushed for ~7 hours).
- **Stall recovery**: wake a stalled child with "check your result log/artifacts FIRST — the result may already exist — then relaunch only what's missing", plus anything that changed while it slept (main moved, SOP amended).
- **Resource arbitration**: full-suite eval runs need explicit user approval; cross-session GPU contention is the supervisor's to surface to the user, not the children's to fight over.
- **Lifecycle**: relay merge/close events so children run their §9 cleanup; file the follow-up issues children report as out-of-scope findings.
- **Fleet-end sweep (the cleanup backstop)**: §9 makes cleanup each child's job, but a dormant child never hears about a merge unless someone relays it — so relayed events *plus* a terminal sweep, not either alone. Before ending a fleet session, inventory `git worktree list` against PR states and remove every tree whose PR is terminal (merged or closed), deleting its local+remote branch. **Locked-tree etiquette**: a locked worktree belongs to a *live* agent (the lock names its pid) — relay the merge event and let it run its own §9; never force-remove another session's locked tree. (Two fleet runs in a row ended with merged agents' trees needing a manual sweep; the backstop is now part of the job.)
- **Verify the mechanism, not the wording, when relaying a rule**: before telling children "do X via Y", functionally test that Y works the way the instruction assumes. A rule whose mechanism silently fails is worse than no rule — children believe they're compliant while violating it. (This happened: the "scope every eval" rule was relayed correctly, but `EVAL_PYTEST_ARGS` passed as an env var was silently discarded by make's `=` assignment, so agents ran full ~60-min suites for hours *thinking they were scoped*. One `make -n` dry-run would have caught it.)
- **Meta-ticket ownership (the status source of truth)**: when a fleet's work is tracked under a meta/tracking issue, the supervisor **owns that ticket end-to-end** — filing it is not the end of the job. Keep it current as children finish, and when the user asks "what's left?" / "are we done?", **reconcile the answer against the meta's actual sub-items via live `gh` queries** — never declare completion from memory. "Everything is done" while the meta still lists open sub-tickets is a false claim (this happened: multiple sessions filed meta tickets, declared done, and left them open over unfinished work). At fleet end, close the meta with a final status comment, or explicitly report what remains open and why.

## Documentation Maintenance

**IMPORTANT**: Always update CLAUDE.md and README.md after making significant changes to the codebase. This includes:
- New features or modules
- Architecture changes
- Configuration changes
- API changes
- Directory structure changes

Each sub-project has its own CLAUDE.md — update the relevant one(s).

## Directory Structure

```
penny/                          — Penny chat agent (Signal/Discord)
  penny/                        — Python package
  Dockerfile
  pyproject.toml
  CLAUDE.md                     — Penny-specific context
penny-team/                     — Autonomous dev team (Claude CLI agents)
  penny_team/                   — Python package
  scripts/
    entrypoint.sh               — Docker entrypoint
  Dockerfile
  pyproject.toml
  CLAUDE.md                     — Penny-team-specific context
github_api/                     — Shared GitHub API client (GraphQL + REST)
  api.py                        — GitHubAPI class (typed Pydantic return values)
  auth.py                       — GitHubAuth (App JWT token generation)
similarity/                     — Shared similarity primitives (penny + penny-team)
  embeddings.py                 — Pure math: cosine similarity, TCR, serialization
  dedup.py                      — Dedup strategies (TCR + embedding)
browser/                        — Firefox browser extension
  src/                          — TypeScript source
    protocol.ts                 — Typed WebSocket + runtime messaging protocol
    background/                 — WebSocket owner, tool dispatch, tab tracking
    sidebar/                    — Chat UI, page context toggle
    content/                    — Defuddle-based page extraction (esbuild bundled)
  sidebar/                      — Sidebar HTML + CSS
  icons/                        — Extension icons (rendered from SVG)
  manifest.json                 — WebExtensions manifest
  tsconfig.json                 — Strict TypeScript config
  build-content.mjs             — esbuild wrapper for content script
  package.json                  — Dependencies: defuddle, fontawesome, esbuild, web-ext
Makefile                        — Dev commands (make up, make check, make prod)
docker-compose.yml              — signal-api + penny + team services
docker-compose.override.yml     — Dev source volume overrides
scripts/
  watcher/                      — Auto-deploy service
  client-check.sh               — iOS client build + simulator test run (make client-check)
.github/
  workflows/
    check.yml                   — CI: runs make check on push/PR to main
    client-check.yml            — CI: runs make client-check on PRs touching penny-client/
  CODEOWNERS                    — Trusted maintainers (used by penny-team filtering)
docs/                           — Design documents and review guides
  pr-review-guide.md            — Canonical PR review checklist (used by /quality skill)
  agent-task-workflow.md        — Task-agent SOP: one ticket → worktree → gate → PR → shepherd → cleanup
  agent-supervisor-runbook.md   — Supervisor runbook: meta ticket, dispatch, waves, heartbeat, fleet-end
  browser-extension-architecture.md — Browser extension architecture & design
  channel-manager-plan.md       — Multi-channel implementation plan
  browser-tools-plan.md         — Browser tools implementation plan
  agent-memory-patterns.md      — Patterns for agent memory recall and dedup
  benchmarking-embedding-models.md — Embedding model benchmark results
  benchmarking-qwen35-vs-gpt-oss.md — qwen3.5 vs gpt-oss benchmark comparison
data/                           — Runtime data (gitignored)
  penny/                        — Penny runtime data
    penny.db                    — Production database
    backups/                    — DB backups (max 5)
    logs/                       — Penny runtime logs (penny.log)
  penny-team/                   — Agent team runtime
    logs/                       — Agent logs + prompts
    state/                      — Agent state files
  private/                      — Credentials (not in repo)
```

## Running

The project runs inside Docker Compose. A top-level Makefile wraps all commands:

```bash
make up               # Start all services (penny + team) with Docker Compose
make prod             # Deploy penny only (no team, no override)
make kill             # Tear down containers and remove local images
make build            # Build the penny Docker image
make team-build       # Build the penny-team Docker image
make token            # Generate GitHub App installation token for gh CLI
make check            # Format check, lint, typecheck, and run tests (penny + penny-team)
make pytest           # Run integration tests
make fmt              # Format with ruff (penny + penny-team)
make lint             # Lint with ruff (penny + penny-team)
make fix              # Format + autofix lint issues (penny + penny-team)
make typecheck        # Type check with ty (penny + penny-team)
make migrate-test     # Test database migrations against a copy of prod DB
make migrate-validate # Check for duplicate migration number prefixes
make signal-avatar    # Set Penny's Signal profile picture from penny.png
make client-check     # Build the iOS client + run PennyClientTests on a simulator (requires Xcode)
```

### Browser Extension Development

```bash
cd browser
npm install            # Install dependencies
npm run build          # Build TypeScript + bundle content script
npm run dev            # Build, watch, and launch Firefox with auto-reload
npm run ext            # Launch Firefox with web-ext (no build/watch)
```

`npm run dev` uses `web-ext` with `--firefox-profile=default-release --keep-profile-changes` to run in the user's real Firefox profile. The background script owns the WebSocket connection; the sidebar communicates via `browser.runtime` messaging.

On the host, dev tool commands run via `docker compose run --rm` in a temporary container (penny service for `penny/`, team service for `penny-team/`). Inside agent containers (where `LOCAL=1` is set), the same `make` targets run tools directly — no Docker-in-Docker needed.

`make prod` starts the penny service only (skips `docker-compose.override.yml` and the `team` profile). The watcher container handles auto-deploy when running the full stack via `make up`.

Prerequisites: signal-cli-rest-api on :8080 (for Signal), Ollama on :11434, browser extension for web search.

## CI

GitHub Actions runs `make check` (format, lint, typecheck, tests) on every push to `main` and on pull requests. The workflow builds the Docker images and runs all checks inside containers, same as local dev. Config is in `.github/workflows/check.yml`. Both penny and penny-team code are checked in CI.

Changes touching `penny-client/` additionally run `make client-check` on a macOS runner (`.github/workflows/client-check.yml`): it builds the iOS app and runs `PennyClientTests` on a freshly booted simulator via `scripts/client-check.sh` — the same script used locally. The fresh erase + boot per run is load-bearing (a reused simulator flakes with "failed preflight checks").

## Configuration (.env)

**Channel selection** (auto-detected if not set):
- `CHANNEL_TYPE`: "signal" or "discord"

**Signal** (required if using Signal):
- `SIGNAL_NUMBER`: Your registered Signal number
- `SIGNAL_API_URL`: signal-cli REST API endpoint (default: http://localhost:8080)

**Discord** (required if using Discord):
- `DISCORD_BOT_TOKEN`: Bot token from Discord Developer Portal
- `DISCORD_CHANNEL_ID`: Channel ID to listen to and send messages in

**Browser Extension** (optional):
- `BROWSER_ENABLED`: "true" to enable browser channel (default: false)
- `BROWSER_HOST`: WebSocket bind address (default: "localhost", use "0.0.0.0" in Docker)
- `BROWSER_PORT`: WebSocket port (default: 9090, must be exposed in docker-compose)

**LLM** (OpenAI-compatible endpoint — no Ollama-specific dependencies in the runtime):
- `LLM_API_URL`: API endpoint (default: http://host.docker.internal:11434)
- `LLM_MODEL`: Single text model for all penny agents — chat, thinking, history, notify, schedules (default: gpt-oss:20b)
- `LLM_API_KEY`: API key (default: "not-needed")
- `LLM_VISION_MODEL`: Vision model for image understanding (e.g., qwen3-vl). Optional; if unset, image messages get an acknowledgment response
- `LLM_VISION_API_URL` / `LLM_VISION_API_KEY`: Override endpoint for vision model
- `LLM_EMBEDDING_MODEL`: **Required.** Dedicated embedding model (e.g., embeddinggemma) — backs Penny's memory (preference dedup and similarity recall). Penny fails fast at startup if it is unset; there is no degraded, embedding-less mode
- `LLM_EMBEDDING_API_URL` / `LLM_EMBEDDING_API_KEY`: Override endpoint for embedding model
- `LLM_IMAGE_MODEL`: Image generation model (e.g., x/z-image-turbo). Optional; enables `/draw`. Uses Ollama's native REST API at `LLM_IMAGE_API_URL`
- `LLM_IMAGE_API_URL`: Ollama REST endpoint for image generation (default: http://host.docker.internal:11434)
- `OLLAMA_BACKGROUND_MODEL`: Used only by penny-team's Quality agent — if set, the Quality agent is registered. Not used by penny

**API Keys**:
- `CLAUDE_CODE_OAUTH_TOKEN`: OAuth token for Claude CLI Max plan (agent containers, via `claude setup-token`)
- `FASTMAIL_API_TOKEN`: API token for Fastmail JMAP email search (optional, enables `/email` command)
- `ZOHO_API_ID`: Zoho OAuth client ID (optional, enables `/zoho` command)
- `ZOHO_API_SECRET`: Zoho OAuth client secret (optional, enables `/zoho` command)
- `ZOHO_REFRESH_TOKEN`: Zoho OAuth refresh token (optional, enables `/zoho` command) — obtain via [OAuth flow](https://www.zoho.com/mail/help/api/using-oauth-2.html)
**GitHub App** (required for agent containers):
- `GITHUB_APP_ID`: GitHub App ID for authenticated API access
- `GITHUB_APP_PRIVATE_KEY_PATH`: Path to GitHub App private key file
- `GITHUB_APP_INSTALLATION_ID`: GitHub App installation ID for the repository

**Behavior**:
- `MESSAGE_MAX_STEPS`: Max agent loop steps per message (default: 8, runtime-configurable via `/config`)
- `IDLE_SECONDS`: Global idle threshold for all background tasks (default: 60, runtime-configurable via `/config`)
- `TOOL_TIMEOUT`: Tool execution timeout in seconds (default: 60)

**Logging**:
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `LOG_FILE`: Optional path to log file
- `LOG_MAX_BYTES`: Maximum log file size before rotation (default: 10485760 / 10 MB)
- `LOG_BACKUP_COUNT`: Number of rotated backup files to keep (default: 5)
- `DB_PATH`: SQLite database location (default: /penny/data/penny/penny.db)

## Testing Philosophy

- **Always use `make fix check`**: The only way to run tests is `make fix check 2>&1 | tee /tmp/check-output.txt; echo "EXIT_CODE=$pipestatus[1]" >> /tmp/check-output.txt`. Never use `make pytest`, `make check` alone, `docker compose run`, or any other ad-hoc invocation. Read `/tmp/check-output.txt` to inspect results afterward — check EXIT_CODE first, then grep for FAILED or `error\[` as needed.
- **penny-client (Swift) changes additionally require `make client-check`**: builds the iOS app and runs `PennyClientTests` on a simulator. Requires Xcode (`xcodebuild -version` to confirm); if Xcode is unavailable, push and rely on the `client-check` CI job — and say so in the PR instead of claiming the change verified.
- **Strongly prefer integration tests**: Test through public entry points (e.g., `agent.run()`, `has_work()`, full message flow) rather than testing internal functions in isolation
- **Fold assertions into existing tests**: Prefer adding assertions to an existing test that covers the relevant code path over creating a new test function
- **Unit tests only for pure utility functions**: CODEOWNERS parsing, config loading, and similar pure functions with many edge cases are acceptable as unit tests
- **Mock at system boundaries**: Mock external services (Ollama, Signal, GitHub CLI, Claude CLI) but let internal code execute end-to-end
- **Never rely on real timers**: Use `wait_until(condition)` instead of `asyncio.sleep(N)` — poll for the expected side effect (DB state, message count, etc.) with a generous timeout. Fixed sleeps are fragile on slow CI and waste time on fast machines

## Design Principles

- **Behavior changes via data/prompt before code**: Penny's behavior lives in editable data — `memory` rows (extraction prompts, `inclusion`/`recall` flags, intervals), `skills` entries, and runtime config — not just code. When changing *what Penny does* (a collector's job, when a skill fires, a notification policy), prefer the highest rung of this ladder that works: (1) a user message Penny acts on and persists as a skill; (2) a direct UI edit of the prompt/flags/entries in the addon; (3) a canonical prompt change validated through the live-model eval suite (`make eval` — see `penny/penny/tests/eval/` and `docs/self-improvement-loop.md`) before deploy; (4) a code change or data migration — only when the behavior isn't expressible as data (parsing, a new tool, scheduler mechanics). The goal is to manage Penny *through Penny* — observe → reason → mutate → persist → validate — falling back to code only where that's impossible or impractical. This does not contradict *Python-space over model-space*: that principle governs how a single deterministic action is implemented; this one governs where behavior is configured and how it gets corrected.
- **Python-space over model-space**: When an action can be handled deterministically in Python (e.g., posting a comment, creating a label, validating output), do it in the orchestrator rather than relying on the model to use the right tool. Model-space logic is non-deterministic and harder to test. Reserve model-space for tasks that genuinely need reasoning (writing specs, analyzing code, generating responses).
- **Structural state over model judgment**: When correctness hinges on "has this happened already?" — an item notified, a log entry consumed, a step done — represent that state *structurally* (a cursor high-water mark, a flag, an entry's location) so the answer is **read**, not re-decided by the model each cycle. A prose gate ("only notify if it's new") is non-deterministic and the model will rationalize around it; a forward-only cursor or a boolean can't be argued with. This is what fixed the duplicate-notification bug: the publish/subscribe layer (`published` flag + `read_published_latest`) replaced an in-prompt "is this new?" judgment with a per-consumer cursor. **Corollary — decouple cross-cutting concerns from data location**: drain a durable collection by *advancing a cursor*, not by *moving* its entries, so one collection is never forced to choose between being a library and being a queue. (See the pub/sub layer in `penny/CLAUDE.md`.)
- **Migrations are universal; deployment-specific data is a direct DB call**: A migration runs on *every* Penny deployment, so it may only touch things that exist identically everywhere — schema (DDL), and *seeded* data created by an earlier migration (referenced by its known key) or generic criteria (`WHERE author='skills'`, a content-shape `LIKE` match). Never target a row that only exists in *our* database — anything a user created at runtime (a chat-authored skill, a user's collection, a hand-set `runtime_config`) by its specific key/name: that key may be absent or mean something different elsewhere. To fix *our* deployment's runtime data, make a direct DB call against `data/penny/penny.db` (and **ask first** — see the production-DB rule), or let the relevant background loop reconcile it. Rule of thumb: if you can't guarantee the row is in a fresh post-migration DB, it doesn't belong in a migration.
- **Pass parameters, don't swap state**: Never temporarily swap instance state (e.g., `self.db`) to change behavior. Pass the dependency as a parameter through the call chain. Refactor interfaces to accept parameters rather than mutating shared state.
- **Capture static data at build time**: Data that doesn't change during a session (e.g., git commit info) should be captured at Docker build time via build args and environment variables, not parsed at runtime via subprocess calls.
- **Initialize at startup, not in handlers**: Heavyweight setup (copying databases, creating resources) belongs at startup (entrypoint scripts, Makefile, build steps), not lazily inside message or request handlers.
- **Template method over conditionals**: When a parent class has multiple modes or variants, define building blocks on the parent and let each variant compose them explicitly — no flags or if/else chains. Examples: agent system prompts (building blocks like `_identity_section()`, `_profile_section()`), notification modes (`NotificationMode` subclasses declare tools/prompt/context), preference commands (`ValenceConfig` NamedTuple).
- **Visible degradation over silent success**: Penny must never degrade silently. Every unmet prerequisite, failed capability, or wrong-config state should surface as a *visible, actionable signal* — to the user and/or in the run record — rather than Penny quietly doing less while appearing to work. The failure mode this guards against is the one the first external deployment exposed: almost nothing *errored*, everything *degraded* — wrong-channel routing, wrong-timezone reasoning, a disconnected browser, an unconfigured embedding model — several silent failures at once, none individually fatal, collectively making Penny feel broken-but-cheerful with no way for the user to see why. Concretely: unmet prerequisites are reported (a startup preflight), not swallowed; failed capabilities produce honest state, not optimistic success (this extends *actionable tool failures* to the model's user-facing claims — never assert an action succeeded or will recur unless the tool result confirms it); degraded modes are named ("recall is degraded — no embedding model configured"); raw tracebacks become actionable messages (what to enable, not a stack dump).

## Code Style

- **Pydantic for all structured data**: All structured data (API payloads, config, internal messages) must be brokered through Pydantic models — no raw dicts. This includes tool call arguments: every `Tool.execute(**kwargs)` must validate through a Pydantic args model (e.g., `SearchArgs(**kwargs)`) as its first line, and return structured Pydantic results where applicable
- **Constants for string literals**: All string literals must be defined as constants or enums — no magic strings in logic
- **Prefer f-strings**: Always use f-strings over string concatenation with `+`
- **Datetime columns for ordering, IDs for joining**: Always use datetime columns (`created_at`, `timestamp`, `learned_at`, etc.) for recency ordering in queries. Never use auto-increment IDs (`id`) to infer chronological order — IDs are for joins and lookups only
- **Always use foreign keys**: Never denormalize by storing copies of data that exists in another table. Use proper FK references (e.g., `preference_id REFERENCES preference(id)`) instead of duplicating column values
- **Short methods (10-20 lines)**: Every method should be roughly 10-20 lines (hard max ~25). Break long methods into named steps via extraction — don't add new abstractions, just decompose
- **Summary method at top**: Every class should have a summary method (after `__init__`) that composes calls to other methods, reading like a table of contents. This gives a bird's-eye view of the class's behavior from the top of its definition
- **Database stores pattern**: Database access is organized into domain-specific store classes (`db.messages`, `db.preferences`, `db.thoughts`, etc.). The `Database` class is a thin facade that creates and exposes stores. Access data via `self.db.messages.log_message(...)`, not `self.db.log_message(...)`

## PR Review Checklist

The canonical, exhaustive PR review checklist lives in [`docs/pr-review-guide.md`](docs/pr-review-guide.md). It's the source of truth for every rule the project enforces — code style, error handling, forbidden patterns, async patterns, testing discipline, prompt engineering. The `/quality` slash command reviews the current branch against it.

The Code Style and Design Principles sections above are the quick reference; the PR review guide is the full rulebook.
