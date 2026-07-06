# CLAUDE.md — Penny Chat Agent

## Architecture Overview

```mermaid
flowchart TD
    User((User)) -->|message| Channel[Signal / Discord]

    subgraph Foreground["Foreground (ChatAgent)"]
        Channel -->|extract| CA[ChatAgent]
        CA -->|"prompt + tools"| FG_Ollama["LLM<br>(OpenAI SDK)"]
        FG_Ollama -->|tool call| Browse[BrowseTool]
        Browse -->|"read page"| Browser[Browser Extension]
        Browse -->|"web search"| Browser
        Browser -.->|results| FG_Ollama
        FG_Ollama -->|response| CA
    end

    CA -->|reply| Channel -->|send| User
    CA -->|log| DB[(SQLite)]

    subgraph Scheduler["Background Scheduler (when idle)"]
        direction TB

        SE[ScheduleExecutor] -->|"cron tasks"| FG_Ollama2["LLM<br>(OpenAI SDK)"]

        Coll[Collector dispatcher] -->|"per-collection prompt<br>+ scoped tools"| FG_Ollama3["LLM<br>(OpenAI SDK)"]
        Coll -.->|"reads memory rows<br>(extraction_prompt, interval,<br>last_collected_at)"| DB
        Coll -.->|"writes entries<br>scoped to one collection"| DB
        Coll -->|"send_message<br>(notify-shaped cycles)"| Channel
    end

    User -.->|"resets idle<br>cancels background"| Scheduler
```

- **Channels**: Signal (WebSocket + REST) or Discord (discord.py bot)
- **Ollama**: Local LLM inference (default model: gpt-oss:20b)
- **Vision**: Optional vision model (e.g., qwen3-vl) for processing image attachments from Signal
- **Image Generation**: Optional image model (e.g., x/z-image-turbo) for generating images via the `generate_image` chat tool (config-gated on `LLM_IMAGE_MODEL`)
- **Embedding Model**: Required dedicated embedding model (e.g., embeddinggemma) for preference deduplication and history embeddings — a hard prerequisite (startup fails fast if `LLM_EMBEDDING_MODEL` is unset), so memory never runs in a degraded, embedding-less mode
- **Browser Extension**: Web search and page reading — all web access goes through the connected browser
- **SQLite**: Logs all prompts and messages; stores preferences, thoughts, and conversation history

## Directory Structure

```
penny/
  penny.py            — Entry point. Penny class: creates agents, channel, scheduler
  config.py           — Config dataclass loaded from .env, channel auto-detection
  config_params.py    — ConfigParam + RuntimeParams: runtime-configurable settings with 3-tier lookup
  constants.py        — Enums (SearchTrigger, PreferenceValence), reaction emojis, browse constants
  prompts.py          — LLM prompt templates (chat conversation, vision, email-summarize).  Collector prompts live on memory rows (extraction_prompt) instead
  responses.py        — All user-facing response strings (PennyResponse class)
  startup.py          — Startup announcement message generation (git commit info)
  preflight.py        — Setup-health / preflight checks: one legible startup summary (Preflight → PreflightReport). Hard-fails (PreflightError, caught in main → exit 1) on an unreachable LLM endpoint or an unresolvable chat/embedding model; soft-warns on a missing vision/image model, a disconnected browser addon, or a mis-routed primary channel (routing-bug guard). Runs in Penny.run() after channel connectivity, before backfills
  datetime_utils.py   — Timezone derivation from location (geopy + timezonefinder)
  agents/
    base.py           — Agent base class: agentic loop, tool execution, Ollama integration
    models.py         — ChatMessage, ControllerResponse, MessageRole, ToolCallRecord, GeneratedQuery
    chat.py           — ChatAgent: conversation-mode agent (handles user messages with tools)
    recall.py         — build_recall_block: assembles ambient recall context from active memories
    collector.py      — Collector: single dispatcher agent driving every per-collection extractor
  scheduler/
    base.py           — BackgroundScheduler + Schedule ABC
    schedules.py      — PeriodicSchedule, AlwaysRunSchedule, DelayedSchedule implementations
    schedule_runner.py — ScheduleExecutor: runs user-created cron-based scheduled tasks
    send_queue_drainer.py — SendQueueDrainer: delivers queued send_message output on the send cooldown
  commands/
    __init__.py       — create_command_registry() factory
    base.py           — Command ABC, CommandRegistry
    models.py         — CommandContext, CommandResult, CommandError
    config.py         — /config: view and modify runtime settings
    index.py          — /commands: list available commands
    profile.py        — /profile: user info collection (name, location, DOB, timezone)
  tools/
    base.py           — Tool ABC (declares `args_model` + `run()`: validate args via the Pydantic model, then `execute`), ToolRegistry, ToolExecutor (drives `tool.run`)
    models.py         — ToolCall, ToolResult (uniform structured tool return: message/success/mutated/source_urls/narration), ToolDefinition, and per-tool arg models
    browse.py         — BrowseTool: web search and page reading via browser extension
    generate_image.py — GenerateImageTool: image generation via OllamaImageClient; stored to media, delivered side-channel (chat-only, gated on LLM_IMAGE_MODEL)
    content_cleaning.py — Post-processing for browse results (strips navigation, proxy images, boilerplate)
    search_emails.py  — SearchEmailsTool (JMAP + Zoho) — chat surface, config-gated on a mailbox
    read_emails.py    — ReadEmailsTool (JMAP + Zoho) — summarizes fetched bodies against the current message
    list_emails.py    — ListEmailsTool (folder listings; Zoho only)
    list_folders.py   — ListFoldersTool (available mailboxes; Zoho only)
    draft_email.py    — DraftEmailTool (compose + stage draft; Zoho only). All five retired the /email + /zoho commands (epic #1445) — built per turn by ChatAgent's email_tools_builder; NL-dispatch contract: tests/eval/test_email_dispatch.py
    notifications.py  — NotificationsMuteTool / NotificationsUnmuteTool: chat-surface tools over the MuteState row (`db.users`); retired /mute + /unmute
    schedule_tools.py — ScheduleCreateTool / ScheduleDeleteTool / ScheduleListTool: the chat agent's NL surface for recurring cron tasks (reuses the SCHEDULE_PARSE_PROMPT NL→cron parse; delete matches by embedding, never index). Registered on the chat surface in ChatAgent.get_tools
    memory_args.py    — Pydantic arg models for the memory tool surface
    memory_tools.py   — Tool subclasses: each funnels through `db.memory(name)` (the single dispatch) and calls a method on the returned `Memory` object, which refuses wrong-shape ops (collection ops on a log, log_read on a collection) via a base no-op (`WrongShapeError`) and read-only facades via `ReadOnlyMemoryError` — no tool branches on a name or shape. read_similar + memory_metadata are shape-agnostic. build_memory_tools(db, embedding_client, author) factory
  channels/
    __init__.py       — create_channel() factory, channel type constants
    base.py           — MessageChannel ABC, IncomingMessage, shared message handling
    signal/
      channel.py      — SignalChannel: httpx for REST, websockets for receive
      models.py       — Signal WebSocket envelope Pydantic models
    discord/
      channel.py      — DiscordChannel: discord.py bot integration
      models.py       — DiscordMessage, DiscordUser Pydantic models
  database/
    database.py       — Database facade: thin wrapper creating domain stores
    knowledge_store.py — KnowledgeStore: summarized web page content for factual recall
    message_store.py  — MessageStore: log_message, log_prompt, log_command, threads
    thought_store.py  — ThoughtStore: inner monologue persistence
    preference_store.py — PreferenceStore: add, query, dedup, embedding management
    send_queue_store.py — SendQueueStore: durable outbound queue (enqueue, next_pending, mark_sent)
    user_store.py     — UserStore: get_info, save_info, mute/unmute
    memory/           — the memory layer: `Memory` (base, memory_entry row access + shared similarity/cursor reads + shape-op no-ops) → `Collection` / `Log`, and the read-only facades `MessageLogMemory` (messagelog) / `RunLog` (promptlog); `MemoryStore` registry + the `memory(name)` dispatch factory; `types` (enums, errors, inputs); `_similarity` (pure dedup + retrieval math). `db.memory(name)` returns the right object; `db.memories` is the registry
    cursor_store.py   — CursorStore: per-agent read cursors into log-shaped memories
    media_store.py    — MediaStore: browsed images, matched to outgoing text by embedding at egress
    models.py         — SQLModel tables (see Data Model section)
    migrate.py        — Migration runner: file discovery, tracking table, validation
    migrations/       — Numbered migration files (0001–0025)
  llm/
    client.py         — LlmClient: OpenAI SDK wrapper (chat + embed + list_models via /v1/models) for any OpenAI-compatible backend (Ollama, omlx, etc.). list_models translates SDK errors into the LlmError hierarchy so the preflight can tell an unreachable endpoint from an unverifiable one
    image_client.py   — OllamaImageClient: Ollama-specific HTTP client for image generation and model listing
    models.py         — LlmMessage, LlmResponse, LlmToolCall, LlmError hierarchy (SDK-decoupled Pydantic types)
    embeddings.py     — Re-exports serialize/deserialize/cosine from shared similarity/ package
    similarity.py     — Penny-specific: embed_text, sentiment scores, novelty, preference vectors
  email/
    protocol.py       — EmailClient Protocol — shared interface for JMAP + Zoho email backends
  jmap/
    client.py         — JmapClient: Fastmail JMAP API client (httpx)
    models.py         — JmapSession, EmailAddress, EmailSummary, EmailDetail
  zoho/
    client.py         — ZohoClient: Zoho Mail API client (httpx + OAuth refresh)
    models.py         — Zoho Mail API Pydantic models
  validation/         — Model-I/O validation: the one behaviour taxonomy + the live disposition machinery
    conditions.py     — The behaviour taxonomy (keystone): ConditionKey + BehaviorCondition + CATALOG; one catalog of every condition we classify Penny's behaviour through (supersedes ValidationReason + RunHealthFlag). Dependency-light leaf
    outcomes.py       — ValidationOutcome disposition union (Proceed/Retry/Repair/RejectToolCall/NudgeContinue/Stop), ResponseValidator protocol, LoopContext, run_validators
    response_validators.py — Concrete validators (xml/empty/refusal/hallucinated-url/premature-done/text-instead-of-tool/done-json-bail/repairs), composed per-agent. NOT re-exported from __init__ (imports tools.memory_tools → database; would cycle)
  html_utils.py       — Shared HTML text extraction helpers
  text_validity.py    — Content-validity predicates (degenerate/blank/half-formed-send/description/extraction-prompt/**degeneration-collapse run**), dependency-light leaf shared by the memory write path, the tool arg-validators (memory_args + send_message), the collector readiness gate, the run-health classifier, and the agent-loop degeneracy guard (`is_degenerate_run`)
  tests/
    conftest.py       — Pytest fixtures for mocks and test config
    test_embeddings.py, test_similarity.py, test_periodic_schedule.py, test_scheduler.py
    mocks/
      signal_server.py  — Mock Signal WebSocket + REST server (aiohttp)
      llm_patches.py    — MockLlmClient: patches openai.AsyncOpenAI for chat + embed
    agents/           — Per-agent integration tests
      test_chat_agent.py, test_collector.py, test_agentic_loop.py,
      test_context.py
    channels/         — Channel integration tests
      test_signal_channel.py, test_signal_reactions.py, test_signal_vision.py,
      test_signal_formatting.py, test_startup_announcement.py
    commands/         — Per-command tests
      test_commands.py, test_config.py, test_debug.py,
      test_schedule.py, test_system.py, test_test_mode.py
    database/         — Migration validation tests
      test_migrations.py
    jmap/             — JMAP client tests
      test_client.py
    tools/            — Tool tests
      test_tool_timeout.py, test_tool_not_found.py, test_tool_reasoning.py,
      test_send_message.py, test_notifications.py, test_email_tools.py
Dockerfile            — Python 3.14-slim
pyproject.toml        — Dependencies and project metadata
```

## Agent Architecture

### Agent Base Class (`agents/base.py`)
The base `Agent` class implements the core agentic loop:
- Calls the LLM (via `LlmClient`) with available tools
- Executes tool calls via `ToolExecutor` with parameter validation
- Handles duplicate tool call prevention
- Appends source URLs to responses when model omits them

**System prompt building (template method pattern):**
Each agent overrides `_build_system_prompt(user)` to compose its prompt from reusable building blocks on the base class: `_identity_section()`, `_profile_section()`, `_instructions_section()`, `_context_block()`. No flags or conditionals — each agent explicitly declares what goes in its prompt. Tests assert on the exact full system prompt string to catch structural drift.

**Memory recall** is the single mechanism for surfacing memory contents in the system prompt, assembled in **two stages** (`_recall_section` in `agents/chat.py`):

1. **Stage 1 — collection routing** (`inclusion` flag: `always` / `relevant` / `never`): decides whether a memory participates at all. `always` is unconditional; `relevant` collections **compete** — only the top `RECALL_TOP_K` (default 1) by **current-message** cosine to the memory's content-reflective `description` anchor, clearing `MEMORY_INCLUSION_THRESHOLD` (default 0.40), are admitted; `never` is excluded. This is the prompt-shortening gate — off-topic collections drop out and only the single on-topic collection surfaces (not the long tail of adjacent ones). Scoring on the *current message alone* (not the whole history window) stops a collection from staying "sticky" across later, unrelated turns. `_included_memories`/`_top_relevant` in `agents/chat.py`. (An audit of real chat turns found ~43 recalled entries/message, mostly off-topic; competitive top-1 + current-message anchor is the pare-down.)
2. **Stage 2 — entry rendering** (`recall` flag: `all` / `relevant` / `recent`): for each included memory, picks which entries surface. `recent` is the newest-first slice; `all` is the full set; `relevant` is a hybrid ranking (embedding cosine fused with IDF-weighted lexical coverage via reciprocal-rank fusion, top-N, **no floor** — stage 1 already decided relevance). Lexical fusion surfaces instruction-shaped entries (skills, recipes) whose absolute cosine is low but whose vocabulary overlaps the query.

There is no bespoke per-section retrieval — knowledge, likes, dislikes, thoughts, skills, etc. all surface via this one path. The two flags are orthogonal: e.g. `inclusion=relevant, recall=all` shows every entry but only when the conversation is on-topic.

The chat turns array (alternating user/assistant messages passed via `history=`) is independent of the recall flag — it is reconstructed from the last N messages in `db.messages` regardless of which memories are active.

### Shared LLM Client Instances

All `LlmClient` instances are created centrally in `Penny.__init__()` and shared across agents and commands. `LlmClient` uses the OpenAI Python SDK and targets any OpenAI-compatible endpoint (Ollama's OpenAI-compat layer by default, or omlx/OpenAI cloud with a different `base_url`):

- `model_client`: Text model for all agents and commands
- `vision_model_client`: Optional vision model for image understanding
- `embedding_model_client`: Required embedding model for preference deduplication and similarity recall (always present — the model is a hard prerequisite)
- `image_client`: `OllamaImageClient` for the `generate_image` chat tool (image generation uses Ollama's native REST API, not OpenAI-compatible); wired into the ChatAgent, which registers `generate_image` only when it is present

### Specialized Agents

**ChatAgent** (`agents/chat.py`)
- Handles incoming user messages with the full tool surface (memory + browse)
- Chat-surface tools include `notifications_mute` / `notifications_unmute` (`tools/notifications.py`) — thin toggles over the `MuteState` row (`db.users`) that the model dispatches from natural language ("stop messaging me for a while" / "you can message me again"), replacing the retired `/mute` + `/unmute` commands. NL-dispatch contract: `tests/eval/test_notifications.py`
- Chat-surface tools also include `schedule_create` / `schedule_delete` / `schedule_list` (`tools/schedule_tools.py`), appended in `get_tools` — recurring cron tasks the model dispatches from natural language, replacing the retired `/schedule` + `/unschedule` commands. NL-dispatch contract: `tests/eval/test_schedule_dispatch.py`
- Email tools (`search_emails` / `read_emails`, plus `list_emails` / `list_folders` / `draft_email` on Zoho) are config-gated — present only when a mailbox is configured (Fastmail via `FASTMAIL_API_TOKEN`, Zoho via its OAuth triple; both behind the `EmailClient` protocol), retiring the `/email` + `/zoho` commands (epic #1445). Both configured → Fastmail wins (single user, one mailbox). The mailbox client is long-lived (built in `Penny._init_email`, closed in `shutdown`); `ChatAgent._email_tools_builder(user_query, today)` wraps it fresh each turn so `read_emails` summarises against the current question. A seeded skill (migration 0078) is the NL trigger. NL-dispatch contract: `tests/eval/test_email_dispatch.py`
- Likes/dislikes are dispatched from natural language onto the `likes` / `dislikes` memory collections via the always-present memory tools (`collection_write` / `collection_delete_entry` / `collection_read_latest`), replacing the retired `/like` + `/unlike` + `/dislike` + `/undislike` commands (epic #1445). Add ("I'm really into X"), remove-by-meaning ("forget about X" — matched by meaning, never exact text), and list ("what am I into?") are taught by seeded skills (migration 0079). These collections are what the ambient extractor fills and recall reads; the legacy `preference` table is untouched (its fate is #1301). NL-dispatch contract: `tests/eval/test_likes_dislikes.py`
- Prompt: identity + (profile + recall block + page hint) + instructions; recall block routes memories by `inclusion` (stage 1) then renders entries by `recall` (stage 2)
- Conversation history flows independently as alternating user/assistant turns passed via `history=`
- Vision captioning: when images are present and vision model is configured, captions the image first, then forwards a combined prompt to the text LLM
- Image generation: when an image model is configured, the `generate_image` tool is registered (mirroring the retired `/draw` command's conditionality). It generates the image via `OllamaImageClient`, stores it in the `media` table with an embedding of the description, and returns a text result naming what was drawn; the image is attached to the model's mirror-back reply at egress via `MediaStore.select_image` (the same side-channel browsed images use — nothing travels through the model)

**Collector** (`agents/collector.py`)
- One dispatcher agent for every kind of background extraction.  Each tick it picks the most-overdue ready collection from the `memory` table (where `extraction_prompt IS NOT NULL` and `now - last_collected_at >= collector_interval_seconds`), binds itself to that target via `self._current_target`, runs the agent loop with the target's extraction prompt as instructions and a tool surface scoped to writes against that single collection, then stamps `last_collected_at = now`.
- **Cursor gate (skip-when-no-new-input)** (`_input_pending`/`_live_cursors`, in `_is_ready` after the interval floor): a *log-driven* collection — one that reads a log via `log_read`, leaving a read cursor — is skipped **without entering the model** whenever every one of its live input logs is caught up (`head <= last_read_at`, probed with the same bounded `read_batch` the collector itself uses, uniform across the `messagelog`/`promptlog` facades and real logs). The cursors a collection already holds *are* its declared inputs (no separate spec); the gate ORs across them (any input behind → run). A cursor whose log the current `extraction_prompt` no longer names — left by a since-dropped `log_read` — is pruned (exact identifier substring match) so it can't lie about what the collection consumes. A collection with no live cursor (generative/browse-driven, or one that picks from another *collection*) returns `None` from `_input_pending` — not gate-eligible, runs on its plain interval. This is what lets a quiet log stop a collector cold yet have it resume the instant the log moves, without the catch-up lag a stretched throttle interval caused.
- **Pub/sub consumer gate** (`_is_consumer`/`_published_pending`): a *consumer* — a collection whose `extraction_prompt` calls `read_published_latest` (identified by that call, exactly as a log-reader is by `log_read("name")`) — has its inputs be every `published` collection rather than prompt-named logs. It wakes only when some published source has an entry past this consumer's own per-`(consumer, source)` cursor (a never-seen source starts `PUBLISHED_COLDSTART_LOOKBACK_SECONDS` = 1 week back, so a backlog isn't flooded), and it's throttle-exempt like a log-driven collection. Consumers are excluded from `_live_cursors` pruning (their cursors point at sources the prompt deliberately doesn't name).
- Replaces what used to be four bespoke agents: preference-extractor, knowledge-extractor, thinking, notify.  Each is now just a row in the `memory` table with its own `extraction_prompt` and `collector_interval_seconds`.
- **Pub/sub (producers + the notifier consumer)**: notification is decoupled from gathering. A *producer* collection just gathers — it sets `published=true` and never calls `send_message`. The `notifier` *consumer* drains the published stream and delivers each new entry once (`read_published_latest` + a forward-only cursor; the entry is never moved, so the source stays a durable library). A consumer owns its own cursors, so a future digest/email consumer drains the same surface independently. The chat agent maps a user's "tell me / keep me posted" request onto the `published` flag (the seeded `skills` teach this); it must NOT add a `send_message` step to a producer prompt. (The thoughts pipeline uses this model too: migration 0068 collapsed `unnotified-thoughts`/`notified-thoughts` into one published `thoughts` producer the notifier drains, and the now-unused `collection_move` tool was removed with it — `collection_merge` still uses the `move` *method* internally.)
- System collections currently driven by collectors:
  - `likes` / `dislikes` — extract user preferences from `user-messages` (300s)
  - `knowledge` — summarize web pages from `browse-results` (300s)
  - `thoughts` — inner-monologue producer (migration 0068): picks a random like, browses, drafts a thought, dedups against itself, writes (`published=true`, so the `notifier` delivers each new one; `inclusion=relevant`, so past thoughts surface in chat). Replaces the old `unnotified-thoughts` → `notified-thoughts` move-drain pair (5400s)
  - `skills` — reusable, topic-agnostic workflow patterns the chat agent follows (TRIGGER + STEPS entries surfaced via recall). **Grounded in the real collections that exist** (migration 0069): each cycle the collector calls `collection_catalog` (every user-built collection's intent + extraction_prompt, framework collectors hidden), distils the *kind* behind each, and reconciles against the existing skills — leaving a skill that already covers the kind, folding a *generalizable* recipe improvement into the matching skill's embedded extraction_prompt template (e.g. a collection that grew a "cross-check a reference source" step), but leaving collection-specific quirks (a tag prefix, a skipped media type) in that collection's own prompt, and never deleting. Replaces the old chat-reading loop that minted one-off skills from corrections that never recall-matched. Operate-the-system skills (archive/cadence/flip/scope/one-shot) have no source collection, so the loop never touches them; build-pattern skills (research-notify/silent) refine in place as collections drift. **A skill is always a positive recipe — a TRIGGER (the intent + example phrasings) and numbered tool-call STEPS — never a negative prohibition.** "Don't do X" warnings tied to a since-removed structure (e.g. "don't add a send_message step") are dead weight: the model has no memory of the old structure to need warning against, and the positive form (set `published: true`) already says everything. Migration 0069 rewrote the seeded skills into this one clean shape (21600s)
  - `notifier` — pub/sub consumer (migration 0067): each cycle `read_published_latest(n=1)` picks **one** `published` collection *at random* among those with an unseen entry and returns its oldest unseen entry, grounds it in `penny-messages`/`user-messages`, and `send_message`s it once; the cursor (not a move) guarantees once-only, so the source stays a durable library. **Random rotation, not a global oldest-first pool** (the selection lives entirely in `ReadPublishedLatestTool._select` — model-space is unchanged): a single collector run stamps many entries with near-identical timestamps, so global oldest-first drained a whole burst back-to-back (hours of one topic) and starved low-volume collections for days; picking a random eligible collection each cycle gives every collection with something new an equal shot, so they drain evenly. Mirrors the former `notified-thoughts` compose flow minus the move (600s ≈ the send cooldown). Its prompt is byte-identical to the eval's `notifier-delivers-published` contract.
  - `quality` — self-correcting collector (migration 0055, prompt refined through 0063, run-health flags added 0072, **suggest-not-apply from 0073**): reviews recent runs via `log_read("collector-runs")` — a **read facade over `promptlog`** that renders each run as a record (`[target] summary` header + a structural I/O counts line (`browses: A ok, B failed · reads · writes · sends`) + run-health `⚠` flags + the run's tool trace incl. `done()`). The record is produced by the **shared `render_run_record`/`classify_run`** (in `database/memory/objects.py`) — the *same* representation the addon's prompts tab reads (so what we use to judge a run and what Penny sees of it are one thing). `classify_run` flags five failure modes structurally from stored data: `bailed` (`⚠ NO WORK DONE` — reached `done()`/no tool call without any read/write/browse), `no_writes` (`⚠ NO WRITES` — a browse failed AND the run wrote nothing; the fruitless-run signal, derived from the rendered `## browse error:` headers since `BrowseTool` returns `success=True` on partial failure), `incomplete` (`⚠ INCOMPLETE` — hit the step ceiling), `tool_failures` (`⚠ TOOL FAILURES (n)` — from the persisted `promptlog.tool_failures` count), `degenerate_send` (`⚠ HALF-FORMED SEND` — a `send_message` with no real content, via `_similarity.degenerate_reason`/`is_unfinished_fragment`). Quality judges each run on two tiers: **tier 0** — did the collector follow its instructions at all? a `⚠ NO WORK DONE` bail is a regression; **tier 1** — for runs that executed, does the behaviour match the collection's `intent` (incl. a `⚠ HALF-FORMED SEND`, a `collector_run_history`-confirmed persistent `⚠ NO WRITES`, or a flag-less behaviour drift like a repeated/off-intent send)? `⚠ INCOMPLETE`/`⚠ TOOL FAILURES` and a `❌`/`💤` run that called real tools are capacity/transience — surfaced but IGNORED (never a fix). Before acting it reads the suspect collector's recent runs with **`collector_run_history(<collection>)`** to tell a one-off from a persistent pattern. **It SUGGESTS, never applies**: on confirmed drift it `send_message`s the user a three-part proposal (Observed / Proposed fix / a complete rewritten `extraction_prompt` as a numbered recipe) and calls **no `collection_update`** — the user approves and the chat agent makes the edit. Enforcement is prompt-only (the tool surface stays uniform across collectors); the eval (`tests/eval/test_quality_correction.py`, which asserts a suggestion was sent AND nothing was mutated) is the standing gate. (3600s base, auto-throttles toward the weekly cap on quiet cycles like any other collector)
- User-defined collections created via chat (`/collection_create` with an `extraction_prompt`) are picked up automatically on the next tick — no restart required.
- Tool surface: reads (unrestricted — including `collection_catalog`, which lists every user-built collection's description/intent/published/extraction_prompt, hiding logs + the framework collectors in `PennyConstants.SYSTEM_COLLECTIONS`; this is what the `skills` loop reads to reground on real collections) + entry mutations (`collection_write`, `update_entry`, `collection_delete_entry`) pinned to the bound target via the `_memory_scope()` hook + `log_append` + `send_message` (when channel wired) + browse + done — uniform across every collection, including `quality`. (An earlier `prompt_test` dry-run tool, given only to `quality`, was removed: gpt-oss couldn't reliably drive the dry-run → revise → apply loop — it would emit the revised prompt as text instead of a tool call and the cycle died without applying — so quality now rewrites directly and the next cycle re-checks. See `docs/self-improvement-loop.md`.)
- Cadence: `COLLECTOR_TICK_INTERVAL` (default 30s, idle-gated) drives the dispatcher; per-collection `collector_interval_seconds` controls each collection's pacing within that.
- **Auto-throttle** (`_apply_throttle`, runs after each non-cancelled cycle): the **fallback** for collections the cursor gate can't reach — generative/collection-driven ones with no live log cursor. Log-driven collections are **exempt** (a live cursor → early return): the gate skips their idle ticks before they run, so they never idle their way into a wider interval — and widening one would only re-introduce the catch-up lag the gate removes (new log entries waiting out a stretched floor). For the non-exempt fallback: after `COLLECTOR_THROTTLE_AFTER` (default 3) consecutive idle cycles a collection doubles its `collector_interval_seconds` (capped at `COLLECTOR_MAX_INTERVAL`, default 604800 = weekly) and resets its idle counter; a productive cycle snaps the interval back to `base_interval_seconds` (the user's intended cadence, stamped on create and re-set when the interval is edited) and clears the counter. (Migration 0064 resets all collectors' throttle state to base for the gate's clean slate.) "Produced work" (`_produced_work`) reads the per-call `ToolCallRecord.mutated` flag — set from each tool's structured `ToolResult` — so it counts a cycle as work only when a tool *actually changed durable state* (a row written, an entry moved/deleted, a message sent). A successful no-op (a duplicate-rejected write, an update/delete/move on a missing key, a muted/cooled-down send) carries `mutated=False` and reads as idle, unlike the old "a write tool didn't error" heuristic which counted duplicate-rejected writes as work and starved the throttle. Reads + `done()` = idle. Deterministic in Python — not the quality/model layer.

**ScheduleExecutor** (`scheduler/schedule_runner.py`)
- Background task: runs user-created cron-based scheduled tasks
- Checks every 60 seconds for due schedules (based on user timezone)
- Executes the schedule's prompt text via the agentic loop
- Sends results to the user via channel

## Scheduler System

The `scheduler/` module manages background tasks:

### BackgroundScheduler (`scheduler/base.py`)
- Runs tasks in priority order (schedule executor → collector dispatcher)
- **Skips agents with no work**: when an agent returns False, continues to the next eligible schedule in the same tick. Only breaks when an agent does real work.
- Tracks global idle threshold (default: 60s)
- Notifies schedules when messages arrive (resets timers)
- Passes `is_idle` boolean to schedules (whether system is past global idle threshold)
- **Cancels active background task** when a foreground message arrives (`notify_foreground_start()` calls `task.cancel()`), freeing Ollama immediately for the user's message. Cancelled tasks are idempotent — unprocessed items stay in their queues and are re-picked up on the next cycle
- Commands do NOT interrupt background tasks — they run cooperatively

### Schedule Types (`scheduler/schedules.py`)

**AlwaysRunSchedule**
- Runs regardless of idle state at a configurable interval
- Used for ScheduleExecutor (60s interval)
- Not affected by idle threshold — scheduled tasks run even during active conversations

**PeriodicSchedule**
- Runs periodically while system is idle at a configurable interval
- Used for the **SendQueueDrainer** (idle-gated, `SEND_QUEUE_DRAIN_INTERVAL` 60s — scheduled before the collector so queued messages deliver promptly) and the Collector dispatcher (idle-gated, COLLECTOR_TICK_INTERVAL default 30s); per-collection cadence lives on `memory.collector_interval_seconds`
- Tracks last run time and fires again after interval elapses
- Resets when a message arrives

Schedules run a `ScheduledTask` (`scheduler/base.py`) — a structural Protocol (`name` + `async execute() -> bool`). Background agents satisfy it, and so does the non-LLM `SendQueueDrainer`.

**DelayedSchedule**
- Runs after system becomes idle + random delay
- Available for future use (not currently used by any agent)

## Channel System

### MessageChannel ABC (`channels/base.py`)
- Defines interface: `listen()`, `wait_until_ready()`, `_send_raw()`, `send_typing()`, `extract_message()`
- Implements shared logic: `handle_message()`, `send_message()`, `send_response()`, `_log_and_send()`, `_typing_loop()`
- Holds references to chat agent, database, and scheduler
- **Startup readiness before proactive sends**: `Penny.run()` starts `channel.listen()` and the scheduler before startup notifications. Startup announcement/profile prompts run in a separate task that awaits `channel.wait_until_ready()` first. Channels whose send path depends on listener startup override readiness (`DiscordChannel` waits for its `_ready` event); channels that can send immediately inherit the no-op default. This prevents proactive startup sends from deadlocking before listeners such as the browser WebSocket server bind.
- **Outgoing chokepoint — every send is logged**: `_send_raw()` is the single abstract per-channel delivery primitive (the raw Signal REST / Discord / browser-WS send) and does NO logging. Both concrete base methods funnel through `_log_and_send()`, which logs an `OUTGOING` row to `messagelog` (so it surfaces in the `penny-messages` facade) immediately before calling `_send_raw()` — so nothing Penny sends can bypass the conversation record. `send_message(recipient, content, ...)` is the plain path (command results, error notices, onboarding/profile prompts, threading rejections, permission prompts, startup announcements) — it logs the text but computes no embedding and attaches no media (the embedding is filled by the startup backfill) and returns the platform external id. `send_response(recipient, content, parent_id, author, ...)` is the conversational path (chat replies, queued collector sends via the drainer, scheduled tasks) — it additionally embeds the text (stored on the row + reused for nearest-image matching) and returns the DB message id. `ChannelManager` overrides only `_send_raw()` to route to the resolved concrete channel, so the inherited base methods log exactly once (using the shared db) and the routed concrete channel never double-logs.
- **Progress tracker hook**: `_begin_progress(message)` is an optional override that returns a `ProgressTracker` (defined in `channels/base.py`). The tracker has two methods: `update(tools)` (called when a tool batch starts) and `clear()` (idempotent, called once on success and once again from the dispatch loop's `finally`). The default `_make_handle_kwargs` wires `progress.update` as `on_tool_start` for free, and the final response is always delivered via `send_response` so attachments and quote-replies work normally. Channels without a progress UI return `None`

### SignalChannel (`channels/signal/channel.py`)
- WebSocket connection for receiving messages
- REST API for sending messages, typing indicators, and reactions
- Handles quote-reply thread reconstruction
- **Startup connectivity validation**: `validate_connectivity()` retries DNS + a `GET /v1/about` probe up to `PennyConstants.SIGNAL_VALIDATE_MAX_ATTEMPTS` times with `SIGNAL_VALIDATE_RETRY_DELAY` between attempts (~60 s budget) so cold-boot startup can wait out signal-cli-rest-api's 30-60 s warmup. Each failed attempt is logged at WARNING; the final exhaustion is logged at ERROR and the `ConnectionError` is caught in `main()` and written to `penny.log` before exiting. `docker-compose.yml` also gates `penny` on a `curl /v1/about` healthcheck against `signal-api` via `depends_on: service_healthy`, so compose-managed startups never even hit the retry loop. Tests pass `max_attempts=1, retry_delay=0` to stay fast
- **In-flight progress as emoji reactions**: when a user message arrives, the channel reacts to it with 💭 (thinking) via `POST /v1/reactions`. As the agent's tool calls fire, `SignalProgressTracker.update()` swaps the reaction to a tool-specific emoji from `Tool.format_progress_emoji()` (BrowseTool returns 🔍 for searches, 📖 for URL reads). Signal limits each user to one reaction per message, so each new emoji cleanly replaces the previous — no clutter. When the agent finishes, `tracker.clear()` issues `DELETE /v1/reactions` to remove the reaction entirely, and the response is sent as a normal new message via `send_response` (with text + attachments + quote-reply, the same shape as before progress was added). The typing indicator runs alongside throughout. Why reactions instead of editing a "thinking..." text bubble: Signal mobile/desktop clients silently drop attachments added via message edit — even though the wire format technically allows them — so any final response with an image would lose its image. Reactions sidestep editing entirely

### DiscordChannel (`channels/discord/channel.py`)
- Uses discord.py for bot integration
- Listens to a single configured channel
- Handles 2000-character message limit by chunking
- Typing indicators auto-expire (no stop needed)
- **Privileged-intents startup guard**: the bot needs the **Message Content Intent** enabled in the Discord developer portal (Bot → Privileged Gateway Intents). Without it `client.start()` raises `discord.errors.PrivilegedIntentsRequired`. `listen()` catches exactly that, logs the actionable `DISCORD_PRIVILEGED_INTENTS_ERROR` one-liner (which intent + the portal link), and re-raises as `ConnectionError` so `main()`'s existing startup-connectivity handler surfaces it in `penny.log` and exits cleanly instead of dumping a raw traceback

### Channel Factory (`channels/__init__.py`)
- `create_channel()` creates appropriate channel based on config
- Auto-detects channel type from credentials if not explicit
- **Primary channel seeds a default device at startup** (Signal with `signal_number`, Discord with `discord_channel_id`; iOS registers its own on pairing). The browser addon stays `is_default=0`. This makes proactive routing structural — `ChannelManager._resolve_channel` prefers the default device's channel over a device-identifier match on the recipient, so proactive/autonomous sends land on the configured primary channel and can never be captured by a browser addon whose label the profile's sender was pinned to during onboarding (#1298). Conversational replies are unaffected — each concrete channel handles its own receive→reply loop and never goes through the manager.

## Command System

Penny supports slash commands sent as messages (e.g., `/config`, `/profile`). Commands are handled before the message reaches the agent loop.

### Architecture (`commands/`)
- **Command ABC** (`base.py`): Each command implements `name`, `description`, `aliases`, and `async execute(context) → CommandResult`
- **CommandRegistry** (`base.py`): Maps command names/aliases to handlers, dispatches messages starting with `/`
- **Factory** (`__init__.py`): `create_command_registry()` registers all built-in commands

### Built-in Commands (always registered)
- **/commands** (`index.py`): Lists all available commands with descriptions
- **/config** (`config.py`): View and modify runtime settings (e.g., `/config idle_seconds 600`). Reads/writes RuntimeConfig table in SQLite; changes take effect immediately
- **/profile** (`profile.py`): View or update user profile (name, location, DOB). Derives IANA timezone from location. Required before Penny will chat

Preferences are no longer slash commands — like/dislike add, remove-by-meaning, and list are dispatched from natural language onto the `likes` / `dislikes` memory collections (see the ChatAgent section). The retired `/like` + `/unlike` + `/dislike` + `/undislike` commands wrote the legacy `preference` table; the collections are what recall + the ambient extractor share.

There are no conditional (config-gated) commands — email retired onto the chat tool surface (see the ChatAgent email-tools bullet above); `/bug`, `/feature`, `/draw`, `/mute`, `/unmute`, `/schedule`, `/unschedule` all retired earlier in epic #1445.

### Runtime Configuration
- `/config` reads and writes to a `RuntimeConfig` table in SQLite
- `ConfigParam` definitions in `config_params.py` declare runtime-configurable settings with types and validation
- `RuntimeParams` class provides attribute access: `config.runtime.IDLE_SECONDS`
- Three-tier lookup chain: DB override → env override → ConfigParam.default
- Config values are read on each use (not cached), so changes take effect immediately
- Groups: Chat (max steps, search URL, context limits, retrieval thresholds, domain permission mode), Background (idle threshold, COLLECTOR_TICK_INTERVAL, COLLECTOR_THROTTLE_AFTER, COLLECTOR_MAX_INTERVAL, BACKGROUND_MAX_STEPS, dedup thresholds), Email (body max length, search/list limits, request timeout)

## Data Model

All tables defined in `database/models.py` as SQLModel classes:

- **PromptLog**: Every LLM call — `model`, `messages` (JSON), `response` (JSON), `thinking`, `duration_ms`, `agent_name`, `run_id`, `run_outcome`/`run_reason`/`run_target`, `tool_failures` (count of failed tool calls in the run, stamped on the last prompt alongside the outcome — the structural signal the run-health classifier reads)
- **MessageLog**: Every user/agent message — `direction`, `sender`, `content`, `parent_id` (thread chain), `external_id` (platform ID), `is_reaction`, `thought_id` FK (notification source)
- **UserInfo**: User profile — `name`, `location`, `timezone` (IANA), `date_of_birth`
- **CommandLog**: Command invocations — `command_name`, `command_args`, `response`, `error`
- **RuntimeConfig**: User-configurable settings — `key`, `value` (string, parsed on read)
- **Schedule**: User-created cron tasks — `cron_expression`, `prompt_text`, `user_timezone`
- **MuteState**: Per-user mute state — row exists = muted, delete = unmuted
- **Device**: Registered devices (Signal, Discord, browser addons) — used for multi-device routing and domain permission prompts
- **DomainPermission**: Per-domain allow/deny state for browser extension web access, synced across addons
- **Thought**: Inner monologue entries — `content` (full monologue), `title`, `image`, `valence`, `preference_id` FK (seed preference), `run_id`, `notified_at`
- **Preference**: Legacy user sentiment signals — `content`, `valence` (positive/negative), `source` (manual/extracted), `mention_count`, `embedding` (serialized float32 vector), `last_thought_at`. Extracted preferences must reach `PREFERENCE_MENTION_THRESHOLD` mentions before becoming thinking candidates. The `/like`-family commands that wrote `manual`-source rows are retired (like/dislike is now NL dispatch onto the `likes` / `dislikes` collections); this table still backs the thoughts pipeline's seed-preference pick and the onboarding profile check, so it stays until #1301 decides its fate
- **Knowledge**: Summarized web page content — `url` (unique), `title`, `summary` (prose paragraph), `embedding`, `source_prompt_id` FK (extraction watermark). One entry per URL, upserted on revisit
- **Memory**: Unified container for the task/memory framework — `name` (PK), `type` (`collection` or `log`), `description` (content-reflective; doubles as the stage-1 routing anchor), `description_embedding` (the anchor vector, backfilled at startup), `inclusion` (stage-1 routing: `always` / `relevant` / `never`), `recall` (stage-2 entry rendering: `all` / `relevant` / `recent`), `published` (pub/sub: when true, a consumer like the `notifier` drains this collection's new entries via `read_published_latest` — orthogonal to recall; opt-in, default false), `archived`. Collections are keyed sets with dedup on write; logs are append-only keyless streams
- **MemoryEntry**: One entry in a memory — `memory_name` FK, `key` (nullable for logs), `content`, `author`, `key_embedding`, `content_embedding`. Entries are immutable once written — `update` replaces content for a given key
- **AgentCursor**: Per-reader read progress through a log-shaped memory — `(agent_name, memory_name)` PK, `last_read_at` high-water mark. Advanced two-phase by the orchestrator (pending during a run, committed on success). For collectors the cursor owner is the **bound collection name**, not the constant `"collector"` identity — otherwise every collection reading the same log (e.g. the many that read `user-messages`) would collapse onto one shared cursor and starve each other
- **Media**: Images captured while browsing, delivered side-channel — `mime_type`, `data` (raw bytes), `source_url`, `title`, `embedding` (of title+URL). The browse tool stores every page image here; at channel egress `MediaStore.select_image` attaches the most relevant image, most-relevant-first: (1) **exact URL** — if the outgoing message links a page we captured, that page's own image (newest capture); (2) **same domain** — else the embedding-nearest image from a domain the message links; (3) **fallback** — else a uniform-random pick among the top-K embedding-nearest (jitter, so a centroid "magnet" image can't repeat on consecutive messages — applied *only* to this tier; URL/domain matches are deterministic). No floor, so a reply still carries an image whenever one matches. Zero model involvement — no `<media:ID>` tokens, no prompt changes
- **SendQueueItem** (`send_queue` table): Durable outbound message queue — `content`, `collection` (the collector that queued it), `created_at`, `sent_at` (nullable; `NULL` = still pending, single source of truth — no separate flag). `send_message` enqueues here instead of dropping a message when the autonomous-send cooldown hasn't elapsed; the `SendQueueDrainer` delivers the oldest pending row once the cooldown clears. Kept after delivery (stamped `sent_at`) as an audit trail

## Message Flow

1. Channel receives message → `extract_message()` → `IncomingMessage`
2. Channel calls `handle_message()`:
   - Checks for slash commands first (dispatches via `CommandRegistry`)
   - Notifies scheduler (resets idle timers, suspends background tasks)
   - Starts typing indicator loop
   - Calls `ChatAgent.handle()` which:
     - Finds parent message if quote-reply (via `external_id` lookup)
     - Walks thread history for context
     - Runs agentic loop with tools
   - Logs incoming message to DB
   - Sends response via `send_response()` (logs + sends); every other outgoing message (commands, status, prompts) goes through `send_message()`, which logs too — all sends funnel through `_log_and_send()` → `_send_raw()`
   - Stops typing indicator, resumes background tasks

## Thread/Context System

- Quote-replying continues a conversation thread
- `MessageLog.parent_id` creates a chain of messages
- `db.messages.get_thread_context()` walks the chain (up to 20 messages)

## Key Design Decisions

- **Browser-based search**: All web access (search, page reading) goes through the browser extension via BrowseTool. Text queries are converted to search URLs (configurable via `SEARCH_URL`). No third-party search APIs
- **URL fallback**: If the model's final response doesn't contain any URL, the agent appends the first source URL
- **Duplicate tool blocking**: Agent tracks called tools per message to prevent LLM tool-call loops
- **Tool-result framing (tagged first-person narration)**: every tool result is wrapped by `Tool.format_result(name, arguments, result)` (applied once in `Agent._collect_tool_results`) into a **tagged first-person** header — a narration line + a retained `(<tool> result)` machine tag — then the body: `You used \`<tool>\` and here's the result: (<tool> result)\n<body>` on success, `You tried to use \`<tool>\` but it didn't work: (<tool> result)\n<body>` on failure. The narration is resolved by `Tool._resolve_narration` in priority order: an explicit `ToolResult.narration` frame wins, else the registry-dispatched `Tool.to_result_narration(cls, arguments, result)` — the *result* twin of `to_action_str` (the pre-call status string), branching on `result.success` — else the generic default for an unregistered tool. **`BrowseTool` overrides `to_result_narration` (#1480)**: one first-person line for the whole call summarising search-vs-read from the `queries` arg and the outcome (`You searched for "…"` / `You opened <url>` / on a total failure `… but couldn't read anything`) — the per-page `## browse ...:` section headers stay in the body, so the batch is narrated once, not per section. The call `arguments` + the whole `ToolResult` are threaded from `_execute_single_tool` (which returns the `ToolResult`) → `_collect_tool_results`. **Framework-synthesised failures narrate their own frame (#1482).** The failures a tool can't report itself — tool-not-found (which has NO registered class to dispatch from), timeout, uncaught exception, and arg-validation — set `ToolResult.narration` to a specific first-person frame (`FRAMEWORK_NARRATION_NOT_FOUND` "…but there's no such tool.", `_TIMEOUT` "…but it timed out.", `_EXCEPTION` "…but it errored: `<e>`.", `_INVALID_ARGS` "…but the arguments were wrong:") so `format_result` leads with it instead of the generic line and doesn't double-frame the no-class case; the actionable **remedy** (the #1414 house template's diagnosis + how-to-fix tail — did-you-mean + available list, timeout guidance, "try a different approach", the per-field reasons) stays verbatim in `ToolResult.message`. Remaining per-tool narration overrides land in #1481 (memory/lifecycle). **The tag is load-bearing, not decorative.** The OpenAI `role: "tool"` + `tool_call_id` envelope is the standard "this is a tool result" signal, but gpt-oss:20b doesn't reliably honour it when the body reads like prose — it can mistake fetched data (e.g. a returned user message that itself reads like an instruction) for a fresh directive (#1332's #1 failure class). The terse `Result of your \`<tool>\` call:` header used to carry that disambiguation; a live-model probe showed pure-prose narration with no tag RAISED the call-as-text bail rate (5/6 vs. 3/6 tagged), so the `(<tool> result)` tag stays even as the header reads naturally (see `docs/tool-narration-plan.md`). Read tools additionally lead their body with a `N entries from \`<source>\` (ordering):` header via `_format_entries`. Framing happens after `record.failed` is computed (on the raw result) so failure detection is unaffected. The two tool-*shaped* injections that don't flow through `_collect_tool_results` carry the same framing at their own sites (constructing a `ToolResult` — page-context a success, dedup rejection a failure): the synthetic page-context browse pair (`ChatAgent._inject_page_context`, which also uses the standard `tool_call_id` envelope via `ChatAgent.PAGE_CONTEXT_TOOL_CALL_ID` and, being a direct URL read, now inherits BrowseTool's `You opened <url>` narration for free via #1480) and the duplicate-tool-call rejection (`Agent._dedup_tool_calls`); bespoke narration for those sites lands in #1485
- **Tool argument validation (Django-form pattern)**: every `Tool` declares an `args_model` — the Pydantic model that validates its call arguments — and a base `Tool.run(**kwargs)` constructs/validates it *before* `execute` is ever called (the `ToolExecutor` drives `run`, not `execute`). A `ValidationError` becomes an **actionable error tool response** (`success=False`, naming each bad field with the type+description hint from `parameters` plus any custom-validator message), so `execute` only ever sees valid args and concerns itself with the work + runtime/availability decisions. Validity criteria — required fields, types, and content rules (e.g. `send_message`'s half-formed-body check) — live as field/model validators **on the model**, never ad-hoc inside `execute`. Argless tools default to `NoArgs`. Non-existent tools still return a clear tool-not-found error from the executor. This is the single, uniform pre-execute gate; the premature-`done` guard is the one exception that can't be arg-validation (it depends on run state, so it lives in the loop — see *Collector recovery: nudge vs. error tool response*)
- **Model-I/O validation (one behaviour taxonomy)**: where tool-arg validation gates a tool call's *static* arguments, this gates the model's *dynamic* I/O — the response (text + tool calls). The keystone is `penny/validation/conditions.py`: **one catalog** (`BehaviorCondition` in `CATALOG`, keyed by `ConditionKey`) of every condition we classify Penny's behaviour through, so the user, Penny (her `quality` self-review), and a maintainer share a single coherent view rather than a check that means one thing in the loop and another in the run log. Each condition declares *where it applies* as data — `live` (caught during the run, so the model can recover), `run_flag` (surfaces post-hoc as a `⚠` line on the run record), `collector_only`. A condition can be both: `no_work_done` is refused live as a premature `done()` **and** flagged post-hoc as a bail; `half_formed_send` is gated live at the `send_message` arg-validator **and** flagged post-hoc. This catalog supersedes the old split between `ValidationReason` (live) and `RunHealthFlag` (post-hoc); `RunOutcome` (the run's terminal state) stays a separate concern. The live machinery is in `penny/validation/outcomes.py`: a `ResponseValidator` is `check(response, ctx: LoopContext) -> ValidationOutcome`, pure (reads, returns a disposition, mutates nothing); the disposition union is `Proceed` / `Retry(condition, nudge)` / `Repair(response)` / `RejectToolCall(message)` / `NudgeContinue(message)` / `Stop(response)` (a typed return, not an exception — the loop reshapes into control flow immediately). `run_validators` runs an agent's chain in order (repairs thread through; first non-proceed short-circuits). Validators are composed per-agent as class attributes (`Agent.response_validators`, `BackgroundAgent.response_validators` — the same response-shape guards with the collector-flavored empty-response nudge — and `BackgroundAgent.run_shape_validators`) that read like a table of contents — **a new guard is a new validator in the list, not a new branch in the loop.** This replaced base.py's inline `_check_response` if-ladder + the `handle_text_step`/`handle_premature_terminator` guards. The post-hoc `classify_run`/`render_run_record` (`database/memory/objects.py`) source their `⚠` marker/detail strings from the same catalog, so what we enforce live and what we flag after a run are one definition — the strings are **frozen** (migration 0072's quality prompt + the addon TS type mirror them). **Import-cycle gotcha**: `validation/__init__.py` re-exports only the database-free leaves (`conditions`, `outcomes`); it must NOT import `response_validators` (which pulls `tools.memory_tools` → `penny.database`), because the database layer imports the `conditions` leaf and importing any `penny.validation` submodule runs `__init__` first — eagerly importing `response_validators` there closes a cycle through `penny.database`. Import validators directly from `penny.validation.response_validators`. Design: `docs/model-io-validation.md`
- **Actionable tool failures**: Every tool failure (validation error, rejected/degenerate input, refused operation, missing key, external error) MUST return a `ToolResult` whose message tells the model two things: (1) *what went wrong* — the specific reason, naming the offending field/value — and (2) *how to correct it* — the concrete next action (provide a non-empty value, supply the full replacement text, call the right alternative tool, etc.). The tool result is the model's only feedback channel: a bare "rejected" or a silent no-op gives it nothing to recover from, so it retries the same mistake or gives up. A diagnosis without a remedy is a half-failure. Examples: `check_extraction_prompt` quotes the length and the minimum and points at the prompt shape; `update_entry`'s degenerate-content refusal names the reason *and* suggests `collection_delete_entry` if removal was intended. This is a hard rule, enforced in review — see the error-handling section of `docs/pr-review-guide.md`
- **Queued sends, not dropped**: `send_message` no longer owns *when* a message goes out — it enqueues into `db.send_queue` (after the refusal/truncation/mute content gates) and returns the literal `"Message sent."` (`mutated=True`). The deterministic `SendQueueDrainer` (an idle-gated `PeriodicSchedule`, no LLM) delivers the oldest pending row once the flat-interval cooldown clears: `now - last_penny_message ≥ SEND_COOLDOWN_SECONDS` (bypassed when the user has spoken since Penny's last message — the send is then conversational). So a cooldown *delays* a message instead of losing it. The `"Message sent."` literal is the tool's success signal — enqueue **is** the successful handoff, so a consumer prompt can key on it. Timing lives in Python (the drainer), not model-space. **Message validity vs. delivery decisions are split by where each runs**: a half-formed body (blank / punctuation-only, bare URL, bail-out phrase, an unfinished ellipsis+?/! **tail** like `"Hi there! ......???"`, or an ellipsis-truncated tail) is rejected by a validator on `SendMessageArgs` — so it fails *before* `execute` (the `Tool.run` gate) with `success=False` + the specific, **actionable** reason (the defect + the next move — never a generic "send the COMPLETE message", which misdirected when the send was already substantive) and the model resends a complete message — while refusal / no-recipient / mute are no-op delivery declines that stay in `execute` (they need runtime state). The half-formed validator reuses the shared `half_formed_send_reason` (in **`penny/text_validity.py`** — a dependency-light leaf, so `tools/models.py` can import it without closing an import cycle through `penny.database`), the same rule the run-health classifier flags `⚠ HALF-FORMED SEND` on, so what Penny refuses to send and what she flags as a regression are one definition (the validator is the pre-send refusal; the flag is the after-the-fact backstop). **The send gate judges the message AS A WHOLE, not on a substring hit** (#1386): a substantive, deliberate message that merely EMBEDS a degenerate fragment mid-text (a `quality` suggestion quoting the bad send it observed) is a complete message and is delivered — catching an in-flight collapse in the model's *own* output stays the **agent-loop reroll guard's** job (`is_degenerate_run` on raw output). The strict SUBSTRING poison check stays on `degenerate_reason` (the corpus write gate — a collapse run anywhere corrupts a stored entry). `degenerate_reason`/`is_blank`/`is_unfinished_fragment` and the corpus content filter live in `text_validity` too; the `database.memory` package re-exports them so its public import surface is unchanged (`objects.py` imports them directly from `text_validity`). Known gap (#1397): the reroll guard checks `is_degenerate_run` on every serialized tool-call arg incl. `send_message`, so a suggestion quoting a genuine `"......???"` collapse is still discarded upstream of the (now permissive) send gate — pinned in `tests/agents/test_agentic_loop.py`
- **Collector recovery: nudge vs. error tool response**: a collector acts only through tool calls, and the loop corrects three distinct slips differently. (1) *Plain text, no tool call* (the model narrates instead of acting) → `TextInsteadOfToolValidator` appends the stray text + a `COLLECTOR_TOOL_CALL_NUDGE` as a **user turn** and continues. (2) *A coherent but wrong tool call* → an **error tool response** (`success=False` in that call's tool-result field), so the model sees the failure narration (`You tried to use \`<tool>\` but it didn't work: (<tool> result)\nError…`) and retries — never a user-turn nudge. (3) *Empty content, no tool call* → the collector chain's `EmptyResponseValidator` retries with `COLLECTOR_CONTINUE_NUDGE` — a **tool-call demand** naming `done()`, not the chat `CONTINUE_NUDGE` ("Please provide your response."), which would invite an unparseable prose reply (the per-agent nudge is the one entry that differs between `Agent.response_validators` and `BackgroundAgent.response_validators`). The **premature-`done` guard** (`PrematureDoneValidator`) is case (2): a first-move `done()` — before any read/write/browse — is the `⚠ NO WORK DONE` bail (the model declaring "no new matches" without checking), so it's refused with a `COLLECTOR_PREMATURE_DONE_REJECTION` tool-result and the loop continues (a failed `done` doesn't stop it; see `should_stop_loop`); the model must make a real tool call first. Premature only when this step's calls are all `done` AND no non-`done` call has run yet (the same "no real work" test as run-health `bailed`); a batched `[log_read, done]` or a `done` after any read is honoured. Bounded by `max_steps` — on the final step there's no room to retry, so the `done` closes the cycle (like the text-bail fallback). The send-tool's half-formed/truncation gates are case (2) on `send_message`. (4) *A `done()` emitted as bare JSON text* — the model composes schema-valid `done` arguments (`{"success": …, "summary": …}`, optionally `reasoning`) but fails to route them through the tool-call channel (gpt-oss's native fallback on Harmony backends; the dominant call-shaped text bail in production) → `DoneJsonBailValidator`, a shape-specific case (1): a **teaching `NudgeContinue`** (`COLLECTOR_DONE_JSON_NUDGE`) that names exactly what the model did (wrote `done`'s arguments as text — nothing was recorded) and the exact next move (make the real `done(success=…, summary=…)` tool call), so the model itself re-emits the call. **Never a `Repair`**: fabricating a tool call the model didn't make would coerce a malformed emission into a healthy one — repair is reserved for well-formed calls that transport/parsing mangled (e.g. the Harmony token strip); anything the *model* got wrong gets a teaching response. Also matches the full `{"name": "done", "arguments": {…}}` envelope. Unambiguous by construction (the `{success, summary}` key set is unique to `done`; any extra key falls through to the generic nudge), collector-only (in `BackgroundAgent.run_shape_validators`, ordered before `TextInsteadOfToolValidator` so the specific teaching outranks the generic nudge), surfaced via a WARNING naming the `DONE_JSON_BAIL` condition, and bounded by `max_steps` like the generic text bail.
- **Degeneracy guard (discard-and-reroll)**: gpt-oss:20b occasionally collapses mid-generation into a run of `.`/`…`/`?` ("...??…?..") when its context grows large — measured with a sharp onset around ~4K prompt tokens (a short abortive output, ~59 tokens), most often landing inside a tool-call argument.  The collapse is *poison*: it corrupts the entry/message it lands in, and once fed back into the conversation the next step is ~4× more likely to collapse too.  `Agent._invoke_nondegenerate` (in `agents/base.py`) checks every raw model output — text content, serialised tool-call arguments (the validation chain never sees tool-call responses; they short-circuit), **and** tool-call names — via `is_degenerate_run` / `is_degenerate_tool_name`, and on a hit **discards** the output (never appending it — appending is the contagion path) and re-rolls the *unchanged* context up to `DEGENERATE_REROLL_ATTEMPTS` (3); if it still can't get a clean draw it returns `None`, throwing out the run rather than acting on poison.  The name check applies only to a name that is **unregistered AND collapse-shaped** (`Functions?????`, `funcs.done?` — before the guard, those flowed to a tool-not-found error that kept the poison in context); an unregistered but plausible identifier (a near-miss like `collection_metadata`, or a Harmony-token-wrapped valid name — a repair case for the future Harmony strip, not poison) keeps the executor's tool-not-found path with its "Did you mean X?" hint.  The detector is also folded into `degenerate_reason`, so `collection_write`/`update_entry`/`send_message` refuse poison content — nothing degenerate reaches `memory_entry`/`messagelog` even if the loop guard is bypassed.  The regex is tuned against the prompt-log corpus for zero false positives on legitimate punctuation ("Wait... what?!", code `...`).  Historical `promptlog` rows keep their poison (audit trail; never re-fed to the model).  Named `DEGENERATE_OUTPUT` in the condition catalog
- **Two agent shapes**: ChatAgent (turn-driven, user-facing, lifecycle tools only) and Collector (single dispatcher across all collections, scoped entry-mutation tools).  Plus ScheduleExecutor for user-defined cron tasks
- **Priority scheduling**: Schedule executor → Collector dispatcher (Collector returns False when no collection is ready, so the scheduler skips it)
- **Always-run schedules**: User-created schedules run regardless of idle state; the Collector waits for idle
- **Global idle threshold**: Single configurable idle time (default: 60s) controls when idle-dependent tasks become eligible
- **Background cancellation**: Foreground message processing cancels active background tasks (`task.cancel()`) to free the LLM immediately; cancelled work is idempotent and retried next cycle
- **Commands don't interrupt background**: Slash commands run cooperatively without cancelling the active background task
- **Vision captioning**: When images are present and `LLM_VISION_MODEL` is configured, the vision model captions the image first with a vision-specific system prompt, then a combined prompt is forwarded to the text LLM. Search tools are disabled for image messages
- **Image side-channel**: Browsed images never travel through the model. The browse tool decodes each page's image (base64 data URI from the extension), stores the bytes in the `media` table with an embedding of the page title+URL, and the agent loop carries no attachments. At egress (`send_response`), the outgoing text is embedded once (reused for the `penny-messages` log) and `MediaStore.select_image` attaches the most relevant image: the cited page's own capture when the message links a source (exact URL → same domain, both deterministic), else a jittered pick among the top-K embedding-nearest (jitter only on this embedding fallback, so a magnet image can't repeat). No floor, so a reply carries an image whenever one matches (a tangential or funny mismatch beats no image). The `generate_image` chat tool rides this same path — it stores the drawn image in the `media` table with an embedding of the description, and the mirror-back reply (which describes the same subject) matches it at egress. This replaced a model-carried `<media:ID>`/inline-URL token scheme that couldn't reliably thread image references through multi-page replies
- **Channel abstraction**: Signal and Discord share the same interface; easy to add more platforms
- **Async throughout**: asyncio, httpx.AsyncClient, openai.AsyncOpenAI, discord.py
- **Host networking**: Docker container uses --network host for simplicity (all services on localhost)
- **Pydantic everywhere**: All external data validated with Pydantic models
- **Table-to-bullets**: Markdown tables converted to bullet points in Python (saves model tokens vs. prompting "no tables")
- **Normal casing**: All user-facing strings (status messages, error messages, acknowledgments) use standard sentence casing — not all lowercase
- **Memory framework (Stages 1–5, 9, 10)**: A unified data primitive — *memory* — with two shapes (collection and log) and one access class `MemoryStore`. Collections dedup on write via a three-signal disjunction (key TCR, key cosine, content cosine — each with strict and relaxed thresholds in `PennyConstants`). Any strict hit, or any two relaxed hits, rejects the write. Logs append without dedup. Stage 2a added 21 model-facing memory tools (`memory_tools.py`). Stage 3 added `build_recall_block` (`recall.py`) — assembles ambient recall context for the chat agent's system prompt by dispatching each active memory by recall mode (`recent`/`relevant`/`all`); paired logs (`user-messages` + `penny-messages`) merge chronologically into a single Conversation section. **Polymorphic `Memory` objects + system-log facades**: the memory layer is a class per shape/backing (`penny/database/memory/`). `db.memory(name)` is the single dispatch — it returns a `Collection` or `Log` (both `memory_entry`-backed, the native store on the base) or a read-only facade, and every tool/recall/addon caller operates on that object polymorphically (wrong-shape ops refuse via base no-ops; nothing branches on a name or shape). `user-messages`/`penny-messages` are `MessageLogMemory` facades over `messagelog` (the object overrides the row primitives to read by direction, synthesizing `MemoryEntry`; a message has two authors — the user/incoming or Penny/outgoing; `append` refuses) and `collector-runs` is a `RunLog` facade over `promptlog` (renders each completed run as a record; `append` refuses) — no duplicated `memory_entry` rows, and the facade marker rows are seeded by migration so dispatch finds them. The cursor *read* logic lives on the `Log` base, uniform across backings; the reader's pending/commit cursor lifecycle stays in `LogReadTool`. `browse-results` is the one remaining real memory log (the browse tool writes it; it has no canonical table behind it). `messagelog.embedding` (for `read_similar` over messages) is filled by the startup backfill, which vectorizes any embedding-bearing table — nothing is copied between tables. Author is passed explicitly as a constructor argument or method parameter — write-capable tools take `author: str` at construction (`build_memory_tools(db, embedding_client, author)`), `BrowseTool(..., author=...)` is built per-agent with `author=self.name`, and `channel.send_response(..., author=...)` requires callers to pass it. No ambient/contextvar state. Embeddings are computed at write time (not lazily) so similarity reads work the moment a memory is reconfigured. `db.memories` replaces the per-domain stores that agents will be ported onto in subsequent stages. See `docs/task-framework-plan.md` (design) and `docs/memory-implementation-plan.md` (staged rollout)

## Dependencies

- `websockets`, `httpx`, `python-dotenv`, `pydantic`, `sqlmodel`, `openai`, `discord.py`, `psutil`, `dateparser`, `timezonefinder`, `geopy`, `pytz`, `croniter`, `PyJWT`
- Dev: `ruff` (lint/format), `ty` (type check), `pytest`, `pytest-asyncio`, `aiohttp` (mock Signal server)
- Python 3.14+

## Database Migrations

File-based migration system in `database/migrations/` (currently 0001–0025):
- Each migration is a numbered Python file (e.g., `0001_initial_schema.py`) with a `def up(conn)` function
- Two types: **schema** (DDL — ALTER TABLE, CREATE INDEX) and **data** (DML — UPDATE, backfills), both use `up()`
- Runner in `database/migrate.py` discovers files, tracks applied migrations in `_migrations` table
- Runs on startup before `create_tables()` in `penny.py`
- `make migrate-test`: copies production DB, applies migrations to copy, reports success/failure
- `make migrate-validate`: checks for duplicate migration number prefixes (also runs in `make check`)
- Rebase-only policy: if two PRs create the same migration number, the second must rebase and renumber
- Run standalone: `python -m penny.database.migrate [--test] [--validate] [db_path]`

Notable migrations:
- 0001: Initial schema (all core tables)
- 0002: `thought.notified_at` column
- 0003: Preference deduplication
- 0004: Drop `entity` and `fact` tables (old knowledge system removed)
- 0005: `preference.last_thought_at` column
- 0006: `messagelog.thought_id` FK (links messages to notification thoughts)
- 0007: `thought.preference_id` FK (links thoughts to seed preferences)
- 0008: `preference.source` + `preference.mention_count` (mention threshold gating)
- 0009: Drop `searchlog.extracted` column
- 0010: Reset reaction `processed` state
- 0011: Drop `preference.source_period_start/end` columns
- 0012: Fix `is_reaction` flag on historical reaction rows
- 0013: Reset conversation history watermarks
- 0014: Add embedding columns (preference, knowledge, etc.)
- 0015: `thought.title` column
- 0016: `device` table (multi-device routing)
- 0017: `thought.image_url` column
- 0018: `thought.valence` column
- 0019: `domain_permission` table (browser extension allowlist)
- 0020: Rename `thought.image_url` → `thought.image`
- 0021: `promptlog.agent_name` + `promptlog.run_id` columns
- 0022: `promptlog.outcome` + `thought.run_id` columns
- 0023: Add `knowledge` table, drop `conversationhistory` (replaced by knowledge + related messages)
- 0024: Drop legacy `searchlog` table (never written to since browser-based search)
- 0025: Add `memory`, `memory_entry`, `agent_cursor`, `media` tables (task/memory framework Stage 1)
- 0026: Seed system log memories — `user-messages`, `penny-messages`, `browse-results` (Stage 9)
- 0027: Backfill memory framework from existing tables — `messagelog` → user/penny logs, `preference` → likes/dislikes, `thought` → notified/unnotified-thoughts, `knowledge` → knowledge collection (Stage 10)
- 0028: Disable ambient recall for `penny-messages` — duplicates the conversation turns array
- 0029: Re-enable ambient recall for `penny-messages` — chat-turn duplication is now handled by the self-match exclusion (#1006) and short-anchor noise by the low-info filter, so historical Penny replies should surface again
- 0030–0042: extraction-prompt fixes and incremental collector/collection tweaks (see individual files)
- 0043: Seed the `skills` collection — workflow patterns (TRIGGER + STEPS) the chat agent follows via recall, plus a collector that extracts/refines/removes skills from chat over time
- 0044: Split the single `recall` flag into two-stage recall — add `inclusion` (`always`/`relevant`/`never`, stage-1 routing) and `description_embedding` columns, derive inclusion from the old recall value (off→never, recent/all→always, relevant→relevant), collapse `recall=off`→`recent`, and force `skills`/`user-messages`/`penny-messages`/`user-profile`/`likes`/`dislikes`/`knowledge` to `inclusion=always`
- 0045: Rewrite the seeded skills that taught the old single-flag model (`recall: "off"` for silent — now an invalid enum) to the inclusion/recall split; nulls their content embeddings so the startup backfill re-vectorizes
- 0046: Add `title` and `embedding` columns to the `media` table (image side-channel: stores title+URL embedding for nearest-image egress matching)
- 0047: Add composite `(run_id, timestamp)` index on `promptlog` (serves the addon's run-pagination GROUP BY + run-outcome lookups); drop the redundant single-column `run_id` index from 0021
- 0048: Add composite `(agent_name, run_id, timestamp)` index on `promptlog` (serves the addon's per-agent prompt-log filter — without it the filtered GROUP BY full-scans and freezes the asyncio loop)
- 0049: Partition collector read-cursors per collection — seed `(collection, log)` cursors from the old shared `(collector, log)` value, then drop the dead `collector`/`knowledge-extractor`/`preference-extractor` rows (companion to keying the cursor on the bound collection in `get_tools`)
- 0050: Add `memory.intent` — the user's stated goal for a collection, set once at create (immutable by the agent's `collection_update` tool; editable only via the user/UI path)
- 0051: Add `promptlog_fts` FTS5 full-text index (over `response`+`thinking`) + sync triggers for the addon's prompt search — a leading-wildcard LIKE can't use a B-tree index
- 0052: Rebuild `promptlog_fts` to drop the `messages` column for instances that applied the original 3-column 0051 (input scaffolding is shared across runs and made search match boilerplate)
- 0053: Add `memory.base_interval_seconds` (snap-back cadence, backfilled from `collector_interval_seconds`) + `memory.consecutive_idle_runs` for collector auto-throttle
- 0054: Replace `promptlog.run_success` (bool) with `run_outcome` (tri-state `RunOutcome`: failed | no_work | worked | cancelled) — backfilled best-effort (success→worked, failure→failed); the work/no-work split isn't recoverable for old rows
- 0055: Seed the `quality` self-correcting collector (inclusion=never, 1h base interval) + its extraction_prompt — graduates the prototype so every instance gets it
- 0056: Switch the quality collector to cursor-based log reads (so the auto-throttle can't widen its window past unread entries)
- 0057: Unify `log_read_next`/`log_read_recent` into one caller-dispatched `log_read` across all seeded extraction_prompts; drop the notify collector's `penny-messages` read (structural dedup via `collection_move`); quality reviews the whole batch
- 0058: Rework the quality prompt around run inspection — read the `collector-runs` index, `log_get` the suspicious runs for their full trace, judge behaviour-vs-intent; drop the `penny-messages` read (cursor drift); skip `❌` run failures as capacity, not drift
- 0059: System-log facades (one migration for the refactor) — rename read tools in stored prompts (`read_latest(`→`collection_read_latest(`, `collection_metadata(`→`memory_metadata(`); rewrite the `quality` prompt to review runs via plain `log_read("collector-runs")` (a `promptlog` facade; no `log_get`/`penny-messages`) and `notify` to pick with `collection_read_random`; drop the dead `memory_entry` rows for `collector-runs`/`user-messages`/`penny-messages` (now facades over `promptlog`/`messagelog`; marker rows stay); add the `ix_promptlog_completed_runs` partial index for bounded run-index reads
- 0061: Add the `send_queue` table — durable outbound message queue (`content`, `collection`, `created_at`, nullable `sent_at`) with a partial index on the pending tail (`WHERE sent_at IS NULL`). Backs `send_message`'s enqueue + the `SendQueueDrainer`
- 0062: Add the `ix_promptlog_target_runs` partial index on `promptlog (run_target, timestamp) WHERE run_outcome IS NOT NULL` — the addon's per-collection collector-runs panel filters by `run_target`, which the `timestamp`-only 0059 index couldn't seek; a sparse collection scanned the whole completed-run history (multi-second freeze on memory click). The 0059 index stays for the unscoped `collector-runs` log
- 0065: Add the `published` pub/sub flag to `memory` (opt-in, default 0) — a collection's new entries are a consumable stream a consumer drains via `read_published_latest`; orthogonal to `inclusion`/`recall`
- 0066: Rewrite the seeded research/notify skills for the `published` flag — notify = set `published=true` (no `send_message` step in the producer body), flip = toggle the flag, scope = broaden the prompt; collapses the old notify/silent skill duplication
- 0067: Seed the `notifier` consumer collection — drains the published stream (`read_published_latest` → ground → send), once-only by cursor; the prompt is the eval's validated `notifier-delivers-published` contract
- 0068: Unify the thoughts pipeline onto pub/sub — collapse `unnotified-thoughts`/`notified-thoughts` into one published `thoughts` producer the notifier drains; move existing thoughts in, seed the notifier cursor to head (no backlog re-send), archive the old pair. The `collection_move` tool is removed with it
- 0069: Reground the `skills` collector on the real collections — rewrite its `extraction_prompt` to read `collection_catalog` and reconcile skills against the collections that exist (distil the kind, fold generalizable recipe improvements into the matching skill, leave collection-specific quirks, never delete) instead of reading chat. Also cleans the skill set: scrub the chat-derived `author='skills'` one-offs (generic) + the orphan `Scheduled digest` seed, and rewrite the surviving seeded skills into the one clean shape (positive TRIGGER + numbered tool-call STEPS — dropping legacy `send_message` negatives and the redundant `[key]` self-title prefix). Touches only universal data (seeded skills by key + generic criteria), never deployment-specific chat-created entries — those the reconcile loop fixes at runtime. Adds the `collection_catalog` read tool + `PennyConstants.SYSTEM_COLLECTIONS` (built-in collectors the catalog hides)
- 0071: Add `promptlog.tool_failures` (nullable INTEGER) — the per-run count of failed tool calls (`ToolCallRecord.failed`), stamped on the run's last prompt by `set_run_outcome`.  Persists the otherwise-lost failure signal so the run-health classifier reads it structurally instead of parsing tool-result text
- 0072: Teach the `quality` collector the new run-health flags, conservatively — the shared run record now also carries `⚠ INCOMPLETE` / `⚠ TOOL FAILURES` / `⚠ HALF-FORMED SEND`, so the quality prompt is rewritten to act on `⚠ HALF-FORMED SEND` (tier 1: the collector sent the user junk; the fix composes the complete message before the one send) while explicitly IGNORING `⚠ INCOMPLETE` and `⚠ TOOL FAILURES` as capacity/transience (never a rewrite — avoids over-correcting healthy collectors).  Validated by eval contracts in `tests/eval/test_quality_correction.py` (a half-formed-send repair + two over-correction guards)
- 0073: Switch the `quality` collector from applying `collection_update` to **suggesting** a fix via `send_message` for user approval (it had made destructive edits to a healthy collection); detection stays structural, the edit moves to the chat agent after the user OKs it
- 0074: Delete `memory_entry` rows corrupted by a gpt-oss **degeneration collapse** — a run of `.`/`…`/`?` ("...??…?..") the model emits mid-generation on a large context, which before the loop guard could land in a stored entry.  A one-time *generic content-shape* cleanup (frozen copy of `is_degenerate_run`, applied in Python over each row); fresh installs match nothing.  Going forward the agent-loop reroll guard + the corpus write gate keep new poison out
- 0076: Seed the "Mute or unmute notifications" skill — a TRIGGER + numbered STEPS recipe teaching the chat agent to dispatch a pause/resume request onto the `notifications_mute` / `notifications_unmute` tools (the retired `/mute` + `/unmute` commands).  Operate-the-system skill (`author='system'`, no source collection), so the skills reconcile loop leaves it alone.  Lifts the NL-dispatch reliability the `tests/eval/test_notifications.py` contract gates
- 0077: Seed the schedule-dispatch skills — TRIGGER + numbered STEPS teaching the chat agent to route "every morning send me X" → `schedule_create`, "you can stop the morning summaries" → `schedule_delete` (by meaning), and "what's scheduled?" → `schedule_list`. Retires the `/schedule` + `/unschedule` commands onto the chat tool surface (epic #1445). Operate-the-system skills (no source collection), `author='system'`, idempotent
- 0078: Seed the "Look up email" skill — TRIGGER + numbered STEPS teaching the chat agent to route email questions ("did I get an email from X?", "check my email for Y") onto `search_emails` → `read_emails` → answer (with `list_emails`/`list_folders`/`draft_email` when available), and to stay quiet on a casual grumble about email volume. Retires the `/email` + `/zoho` commands onto the chat tool surface (epic #1445). Operate-the-system skill, `author='system'`, idempotent. NL-dispatch contract: `tests/eval/test_email_dispatch.py`
- 0079: Seed the likes/dislikes-dispatch skills — TRIGGER + numbered STEPS teaching the chat agent to route "I'm really into X" → `collection_write("likes"|"dislikes")`, "forget about X" → `collection_delete_entry` (matched by meaning, never index/exact text), and "what am I into?" → `collection_read_latest`. Retires the `/like` + `/unlike` + `/dislike` + `/undislike` commands onto the `likes` / `dislikes` memory collections (epic #1445, issue #1451). The legacy `preference` table is untouched (its fate is #1301). Operate-the-system skills (no source collection), `author='system'`, idempotent. NL-dispatch contract: `tests/eval/test_likes_dislikes.py`

## Extending

- **New tool**: Subclass `Tool` in tools/, implement `name`, `description`, `parameters`, `async execute()`, add to agent's tool list in penny.py
- **New channel**: Implement `MessageChannel` ABC, create models, add to `create_channel()` factory
- **New agent type**: Subclass `Agent`, implement `execute()` for background tasks or custom `handle()` for message processing
- **New command**: Subclass `Command` in commands/, implement `name`, `description`, `execute()`, register in `create_command_registry()`
- **New schedule type**: Subclass `Schedule`, implement `should_run()`, `reset()`, `mark_complete()`
- **New LLM backend**: Any OpenAI-compatible endpoint works via `LlmClient` — just set `base_url` / `api_key`. Non-OpenAI-compatible backends can implement the `LlmClient` interface directly (`async chat()`, `async embed()`)

## Test Infrastructure

Strongly prefer end-to-end integration tests over unit tests. Test through public entry points with mocks for external services. Prefer folding new assertions into existing tests over adding new test functions — only add a new test when no existing test covers the relevant code path.

**Mocks** (in `tests/mocks/`):
- `MockSignalServer`: WebSocket + REST server using aiohttp, captures outgoing messages and typing events
- `MockLlmClient` (`llm_patches.py`): Monkeypatches `openai.AsyncOpenAI` so `LlmClient` returns canned `LlmResponse` objects; configurable via `set_default_flow()` or `set_response_handler()`; tracks `requests` and `embed_requests` for assertions

**Fixtures** (in `tests/conftest.py`):
- `TEST_SENDER`: Standard test phone number constant
- `signal_server`: Starts mock Signal server on random port
- `mock_llm`: Patches the OpenAI SDK with configurable responses
- `make_config`: Factory for creating test configs with custom overrides
- `running_penny`: Async context manager for running Penny with cleanup (uses WebSocket detection, not sleep)
- `setup_llm_flow`: Factory to configure mock LLM for message + background task flow
- `wait_until(condition, timeout, interval)`: Polls a condition every 50ms until true or timeout (10s default)

**Test Timing** — never use `asyncio.sleep(N)` in tests:
- Use `wait_until(lambda: <condition>)` to poll for expected side effects (DB state, message count, etc.)
- `scheduler_tick_interval` is set to 0.05s in test config (vs 1.0s production) so scheduler-dependent tests complete quickly
- `running_penny` detects WebSocket connection via `signal_server._websockets` instead of sleeping
- For negative assertions (nothing should happen), verify immediately — don't sleep to "make sure"

**Test Flow**:
1. Start mock Signal server (random port)
2. Monkeypatch the OpenAI SDK (via `mock_llm`)
3. Create Penny with test config pointing to Signal mock
4. Push message through mock Signal WebSocket
5. `wait_until` the expected side effect (outgoing message, DB change, etc.)
6. Assert on captured messages, LLM requests, DB state

**Performance**: Test suite runs in ~30s (`scheduler_tick_interval` set to 0.05s in tests)

### Live-model eval suite (`tests/eval/`)

A separate suite of **contract tests against a real Ollama model** — the canonical
coverage of Penny's core use cases and the yardstick for swapping models. It's
gated behind the `eval` marker (excluded from `make check`/`make pytest`; run via
`make eval`, default 5 samples/case, `EVAL_SAMPLES=N` to override). Cases drive the
real chat/collector loops and score persisted DB state + sends at a `pass_rate`
threshold (`min_pass_rate=None` = report-only). The coverage matrix is the two
agent shapes × answer-from-memory vs. browse-and-reason: `test_chat_response.py`,
`test_collection_lifecycle.py`, `test_extractors.py`, `test_skills_extractor.py`,
`test_quality_correction.py`, `test_collector_honesty.py`, `test_retrieval.py`,
`test_peripheral.py`, `test_notifications.py` (NL-dispatch of the mute/unmute
tools that retired `/mute` + `/unmute`), `test_command_tools.py` (NL-dispatch
contracts for the command-retirement tools), `test_email_dispatch.py`
(NL-dispatch of the email tools that retired `/email` + `/zoho`),
`test_likes_dislikes.py` (NL-dispatch of like/dislike add/remove-by-meaning/list
onto the `likes` / `dislikes` collections, retiring `/like` + `/unlike` +
`/dislike` + `/undislike`). Browse is stubbed; a case injects realistic pages via the
`browse=` kwarg (query-aware `install_browse` / `CannedPage` in `conftest.py`) to
score multi-step tool reasoning. A `CannedPage(fails=True)` makes a matched read
*error* (renders `## browse error:` without the real retry backoff), and the
shared `ALL_BROWSES_FAIL` catch-all makes every source unreachable — the way to
exercise read-failure honesty (a cycle that browsed a lot, read nothing, and must
not confabulate a write/success at `done()`). See `docs/self-improvement-loop.md`.

#### Every model-facing change ships a durable eval contract — validated per change, not batched

Any change that alters how the model behaves — a prompt/`extraction_prompt` edit,
a tool description, a loop/nudge/retry/validation mechanism, a tool-surface
change, **or a change to what the model READS** (how a run record, tool result,
or recall block is rendered) — **must land with a `tests/eval/` case that encodes
the behaviour it establishes or fixes.** The eval suite is both the regression net
(a future prompt tweak can't silently undo it) and the *written contract* for what
we expect the model to do. A model-facing change without an eval contract is
incomplete — the next change will regress it and nothing will catch it.

**Validate each change as you build it, not in one batch at the end.** A multi-part
change (e.g. honest-`done()` guidance, then a structural counts line, then a flag the
self-review reads) is several independent asks of the model — each needs its own
`make eval` gate *as it lands*, because a single eval at the end can't tell you *which*
lever the model did or didn't understand. Shipping the code first and "evaluating later"
has shipped levers here that the model ignored (a phase-1 `done()` guidance scored 0/3
until its wording was sharpened — measurable only because it was eval'd on its own, with
a no-change baseline to beat). Run a **baseline first** when you can (the change vs. its
absence) so the eval shows the lever is load-bearing, not just that the case passes.

**Changes to what the model reads get a NON-REGRESSION eval against the EXISTING cases.**
A rendering change (a new line/flag in the run record the `quality` collector reads) is
an input shift to every consumer of that surface, even though it edits no prompt. It
ships green only after the existing cases that read that surface (`test_quality_correction.py`)
are re-run on the branch and shown to still hold — the new signal must not flip a
leave-alone case into a spurious rewrite, nor mute a corrective one. The shared
`render_run_record` is read by both the collector prompts and the addon, so one rendering
edit moves both; the existing suite is the proof it didn't regress them.

**Authoring a case is not running it.** Every prompt change must be *dry-run through
the harness against the live model* (`make eval` / a focused case) and the result read
**before you commit the prompt** — a case you wrote but didn't run tells you nothing
about whether the model actually complies, and "coverage" without execution has shipped
broken prompts here before. When it fails, read the model's thinking (auto-dumped on
failed samples), not just the scorer line — that's where the reason lives.

**The eval runs the SHIPPED prompts, and a failure is often stale data — not model
incapacity.** A fresh eval DB is built exactly like prod (`create_tables()` then
`migrate()`), so the seeded skills / `extraction_prompt`s the model follows are the ones
migrations ship — the eval tests them as deployed, not a hand-built copy. Two consequences:

- *Make a migration the single source of truth for a seeded prompt.* Have the eval drive
  the **migration-seeded** version (seed only the case's inputs, not a second copy of the
  prompt) so what's validated is byte-for-byte what ships — no fixture copy to drift.
- *Suspect stale seeded data before model incapacity.* Real example from this codebase: the
  chat agent set a new `published` flag **correctly** but *also* bolted on a `send_message`
  step and created duplicate collections — not because it couldn't reason about the flag,
  but because the seeded **skills** still taught the old pattern. We nearly renamed a schema
  field chasing a "comprehension" problem that was actually stale data. The thinking trace
  tells you which; fix at the highest rung (data/skill > prompt > code — see the root
  Design Principles). Corollary: adding a seeded collection in a migration changes the chat
  agent's Memory Inventory, so the verbatim system-prompt test will fail until you update
  its expected string — that's the test working, not a regression.

The workflow:

1. **While iterating**, drive the fix with `replay.py` / focused low-N runs (below) — fast, throwaway, may read the local prod DB.
2. **Before opening the PR**, lift the validated behaviour into a committable, privacy-safe `tests/eval/` case: genericize any real data into synthetic topics (per the log→test→fix loop), and assert on persisted DB state / sends / run outcome, never on wording.
3. **If the failure is stochastic** and can't be reproduced by seeding alone, *force the trigger deterministically and let the real model drive the rest* — e.g. `nudge_eval`'s `_InjectTextBail` forces one plain-text bail, then the live model must recover through the production nudge — so the contract is exercised on every run, not ~25% of them.
4. **Pair it with a deterministic mock test in `tests/`** when there's a mechanism to pin (loop control, branching, bounds): `make check` owns the fast mechanism proof, `make eval` owns the live model-behaviour contract.

#### The log → test → fix loop (durable process for correcting model behaviour)

This is the canonical way to identify and fix *any* undesired model behaviour — never
guess at a fix from the code, drive it from a real failing example:

1. **Find candidates in the DB.** The `promptlog` records every model call (messages,
   tools, response, outcome). Query it for real instances of the failure (e.g. collector
   runs whose only tool call is `done()`). There are almost always plenty.
2. **Pull the full verbatim input.** Extract the *entire* JSON the model saw — system
   prompt, every chat/tool turn, the tool definitions — the exact one, not a paraphrase.
   `penny/tests/eval/replay.py` does this: it reads a `promptlog` row **by id** and
   replays it against the live model (so the harness itself carries no prompt content —
   privacy-safe to commit while the real prompt stays in the local, gitignored DB).
3. **Run it to confirm the failure reproduces** verbatim. If it doesn't, you haven't
   captured the real trigger yet — keep looking.
4. **Genericize for privacy.** Swap out any PII / real-topic mentions for synthetic
   equivalents (the repo is public — see the privacy rule). This is what turns a
   verbatim replay into a committable `fixtures.py` case.
5. **Run it again to confirm the genericized version still reproduces** the failure. If
   genericizing killed the repro, the trigger was in the specifics — narrow it back.
6. **Now tweak the prompt to correct it**, re-running the (now report-only or gated)
   eval case until it passes — and watch the *other* cases to catch over-correction.

Caveat: the loop only applies when the failure is a *model decision on a visible input*.
If the model is making the right call on what it's shown but the input itself is wrong
(e.g. a bailout run rendered header-only, so quality can't see it), that's a
data/rendering bug — fix it in Python first (so the signal is visible), *then* the loop
applies. Distinguish "model ignored the signal" from "model was never shown the signal"
before reaching for a prompt change.

Second caveat — **check the scorer before you blame the model.** A surprising `0/N`
(especially on a report-only case) is as often a mis-specified scorer as a real
failure: read the model's actual tool calls (the auto-dumped thinking) and confirm it
did the wrong thing before tuning the prompt. This session, two `0/3`s were both scorer
bugs — one counted a *distinct* seeded skill as a "duplicate", another penalised a bare
`example.com` that appeared only in an example phrasing. The model was correct; the
contract was wrong. Verify the scorer encodes the intended contract first.

#### Iterating fast: focused low-N first, full suite last

Live-model cases are slow — a quality cycle is ~120-180s **per sample**, so a careless
`4 cases × N=5` loop is ~60 min. Don't iterate that way. While converging a fix:

- Run ONLY the case(s) you're actively changing, at low N — `EVAL_SAMPLES=3 pytest
  "…::test_repairs_done_only_bailout" -m eval -s`. That's ~9 min, not ~60.
- Add a guard case (e.g. `test_healthy` for over-correction) only when a change could
  plausibly affect it, not every loop.
- To inspect *why* a single run failed, run one cycle against the seeded fixture and
  dump its promptlog rows (input messages, thinking, tool calls) — one cycle, ~3 min,
  shows exactly what the model saw and did at each step.
- Run the FULL suite at the default N **once, at the end**, to confirm nothing
  regressed — not as the iteration loop.

#### Always read the model's thinking on a failure — that's where the "why" lives

When a prompt change doesn't move the pass rate, the reason is almost never visible
from the scorer's one-line failure — it's in the model's **thinking trace**. (Real
example: the publish-flag cases failed not because the model misunderstood `published`
— it set it correctly — but because a *stale seeded skill* still told it to add a
`send_message` step. Only the thinking + tool trace made that obvious.) So **reviewing
the thinking is a required step of every prompt-iteration loop, not a last resort.**

The harness does this for you: `_dump_thinking` (in `tests/eval/conftest.py`) prints
each LLM call's thinking + tool calls for **every failed sample automatically** —
pytest surfaces captured stdout for failed tests, so the traces land in the failure
report with no flag needed. Set `EVAL_DUMP_THINKING=1` to also dump passing samples
when you want the full picture. The eval DB is an ephemeral `--rm` container, so this
read-before-discard is the only place the reasoning survives — don't skip it.
