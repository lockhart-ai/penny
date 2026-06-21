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
- The user will review and merge the PR

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
.github/
  workflows/
    check.yml                   — CI: runs make check on push/PR to main
  CODEOWNERS                    — Trusted maintainers (used by penny-team filtering)
docs/                           — Design documents and review guides
  pr-review-guide.md            — Canonical PR review checklist (used by /quality skill)
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
- `LLM_EMBEDDING_MODEL`: Dedicated embedding model (e.g., embeddinggemma). Optional; preferences stored without embeddings if unset
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
**GitHub App** (required for agent containers and `/bug` command):
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
- **Strongly prefer integration tests**: Test through public entry points (e.g., `agent.run()`, `has_work()`, full message flow) rather than testing internal functions in isolation
- **Fold assertions into existing tests**: Prefer adding assertions to an existing test that covers the relevant code path over creating a new test function
- **Unit tests only for pure utility functions**: CODEOWNERS parsing, config loading, and similar pure functions with many edge cases are acceptable as unit tests
- **Mock at system boundaries**: Mock external services (Ollama, Signal, GitHub CLI, Claude CLI) but let internal code execute end-to-end
- **Never rely on real timers**: Use `wait_until(condition)` instead of `asyncio.sleep(N)` — poll for the expected side effect (DB state, message count, etc.) with a generous timeout. Fixed sleeps are fragile on slow CI and waste time on fast machines

## Design Principles

- **Behavior changes via data/prompt before code**: Penny's behavior lives in editable data — `memory` rows (extraction prompts, `inclusion`/`recall` flags, intervals), `skills` entries, and runtime config — not just code. When changing *what Penny does* (a collector's job, when a skill fires, a notification policy), prefer the highest rung of this ladder that works: (1) a user message Penny acts on and persists as a skill; (2) a direct UI edit of the prompt/flags/entries in the addon; (3) a canonical prompt change validated through the live-model eval suite (`make eval` — see `penny/penny/tests/eval/` and `docs/self-improvement-loop.md`) before deploy; (4) a code change or data migration — only when the behavior isn't expressible as data (parsing, a new tool, scheduler mechanics). The goal is to manage Penny *through Penny* — observe → reason → mutate → persist → validate — falling back to code only where that's impossible or impractical. This does not contradict *Python-space over model-space*: that principle governs how a single deterministic action is implemented; this one governs where behavior is configured and how it gets corrected.
- **Python-space over model-space**: When an action can be handled deterministically in Python (e.g., posting a comment, creating a label, validating output), do it in the orchestrator rather than relying on the model to use the right tool. Model-space logic is non-deterministic and harder to test. Reserve model-space for tasks that genuinely need reasoning (writing specs, analyzing code, generating responses).
- **Structural state over model judgment**: When correctness hinges on "has this happened already?" — an item notified, a log entry consumed, a step done — represent that state *structurally* (a cursor high-water mark, a flag, an entry's location) so the answer is **read**, not re-decided by the model each cycle. A prose gate ("only notify if it's new") is non-deterministic and the model will rationalize around it; a forward-only cursor or a boolean can't be argued with. This is what fixed the duplicate-notification bug: the publish/subscribe layer (`published` flag + `read_published_latest`) replaced an in-prompt "is this new?" judgment with a per-consumer cursor. **Corollary — decouple cross-cutting concerns from data location**: drain a durable collection by *advancing a cursor*, not by *moving* its entries, so one collection is never forced to choose between being a library and being a queue. (See the pub/sub layer in `penny/CLAUDE.md`.)
- **Pass parameters, don't swap state**: Never temporarily swap instance state (e.g., `self.db`) to change behavior. Pass the dependency as a parameter through the call chain. Refactor interfaces to accept parameters rather than mutating shared state.
- **Capture static data at build time**: Data that doesn't change during a session (e.g., git commit info) should be captured at Docker build time via build args and environment variables, not parsed at runtime via subprocess calls.
- **Initialize at startup, not in handlers**: Heavyweight setup (copying databases, creating resources) belongs at startup (entrypoint scripts, Makefile, build steps), not lazily inside message or request handlers.
- **Template method over conditionals**: When a parent class has multiple modes or variants, define building blocks on the parent and let each variant compose them explicitly — no flags or if/else chains. Examples: agent system prompts (building blocks like `_identity_section()`, `_profile_section()`), notification modes (`NotificationMode` subclasses declare tools/prompt/context), preference commands (`ValenceConfig` NamedTuple).

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
