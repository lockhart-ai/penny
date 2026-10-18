> **Archived** — removed from the live tree in `73c51e1c` (2026-03-06, #653 "Remove entity/fact knowledge system"). Replaced by the preference/memory system (`docs/reference/database.md`).

---

# Knowledge System v2: Design

> **Status**: Fully implemented. All phases (1–5) plus /memory, /interests, /like entity creation, and entity cleaner are complete. See #325 for the tracking issue.

## Goal

Penny learns what the user likes and is interested in, finds information about those things, proactively grows that knowledge over time, and uses it to have better conversations and surface relevant new information.

---

## Core Principles

### 1. Entity creation is user-gated

Only user-triggered actions can introduce new entities into the knowledge store. User messages, `/learn`, `/schedule`, and other user commands trigger searches — those searches can produce new entities. Penny's autonomous background processing can discover facts about known entities, but never create new ones.

This prevents the knowledge store from filling up with irrelevant entities extracted from tangential search result content (conference sponsors, companies mentioned in passing, scraped web copy).

### 2. Fact extraction is universal

Any search result or message can produce facts about known entities, regardless of who triggered the search. When the extraction pipeline processes content, it always looks for facts about entities that already exist.

### 3. Penny's autonomous enrichment is fact-only

When Penny has exhausted all processable messages and searches, she can trigger new searches — but only to discover facts about known entities. No new entities can be discovered by Penny autonomously. This keeps the learn agent focused on deepening knowledge the user actually cares about.

---

## Core Concept

Two parallel datasets grow together:

**World Knowledge** — entities and facts that exist objectively. "The KEF LS50 Meta is a bookshelf speaker. It costs $1,600. It uses a MAT driver." Sourced from user-triggered searches and user messages. Every fact tracks where it came from and when.

**User Interest Graph** — what the user cares about and how much. "User likes audio equipment (high confidence). User is interested in KEF LS50 Meta (very high — searched 3 times, used /learn, reacted positively). User dislikes sports (medium — said so once)." Built from accumulated engagements across all interactions.

These two datasets combine to drive everything: what Penny says in conversation, what it researches next, and what it proactively shares.

---

## Three Processes

### Process 1: User-Triggered Discovery

User actions trigger searches. Those searches can create new entities and new facts.

**Triggers:**
- User sends a message → Penny searches → SearchLog created with `trigger=user_message`
- User sends `/learn <topic>` → Penny generates a sequence of searches → SearchLogs created with `trigger=learn_command`
- User sends `/schedule` task → scheduled execution triggers searches → SearchLogs created with `trigger=user_message`

**Extraction pipeline processes these SearchLogs:**
- Can create **new entities** from the content
- Can create **new facts** for new or known entities
- Creates engagements tracking user interest

### Process 2: Extraction (Background)

The extraction pipeline processes unprocessed SearchLogs and MessageLogs periodically during idle. Its behavior depends on the search trigger:

| SearchLog trigger | New entities? | New facts? | Engagements? | Notifications? |
|---|---|---|---|---|
| `user_message` | Yes | Yes | Yes | Yes |
| `learn_command` | Yes | Yes | Yes | Yes |
| `penny_enrichment` | **No** | Yes | No | Yes |

For messages, the pipeline always allows new entity creation (messages are user-triggered by definition).

### Process 3: Penny Enrichment (Background)

When Penny is idle and has no unprocessed content, she picks the highest-priority known entity and searches for new facts. The search is tagged `trigger=penny_enrichment`, so the extraction pipeline will only extract facts for known entities.

**Priority scoring:** `interest_score × (1 / fact_count)`

**Adaptive behavior:**

| Entity State | Mode | Search Strategy | Messaging |
|---|---|---|---|
| Few facts (0-5) | Enrichment | Broad queries to fill gaps | Message on any substantial findings |
| Moderate facts (5-15) | Enrichment | Targeted queries for what's NOT known | Message on meaningful new info |
| Many facts (15+) | Briefing | "What's new since [date]" queries | Message only if genuinely novel |
| Negative interest | Skip | — | — |

---

## User Entry Points

### 1. Message (primary interaction)

User sends a message. Penny responds:

- Before responding, retrieve relevant entities via embedding similarity against the message
- Inject known facts into the response prompt so Penny can reference what it already knows
- **Knowledge sufficiency check**: if existing facts are enough to answer the question, respond from memory without searching. If not, search as usual but with better context.
- After responding, the extraction pipeline processes the search results and the user's message
- Since this is user-triggered, new entities can be created from the search results

### 2. /learn [topic] (active research)

A first-class learning system with full provenance tracking.

**`/learn <topic>`:**
1. Creates a `LearnPrompt` record (stored, tracked, queryable)
2. Responds immediately with acknowledgment
3. Generates 3-5 varied search queries from the user's prompt via LLM
4. Executes each search in sequence, creating SearchLogs linked back to the LearnPrompt
5. The extraction pipeline processes those SearchLogs later, discovering entities and facts
6. Since these are user-triggered (`trigger=learn_command`), new entities can be created

**`/learn` (no args):** Shows LearnPrompt status with provenance chain:

```
Queued learning

1) 'find me stuff about speakers' ✓
   - wharfedale denton 85 (17 facts)
   - kef ls50 meta (8 facts)
2) 'ai conferences in europe' ...
   - ml prague 2026 (3 facts)
```

The chain: `LearnPrompt → SearchLog (via learn_prompt_id) → Fact (via source_search_log_id) → Entity (via entity_id)`

### 3. /like, /unlike, /dislike, /undislike (preference management)

Explicit preference manipulation:
- `/like espresso machines` — adds a like preference, high-strength positive engagement, and **creates an entity** for the topic if none exists
- `/dislike sports` — adds a dislike preference, high-strength negative engagement
- `/unlike espresso machines` — removes the like
- `/undislike sports` — removes the dislike

`/like` is a user-triggered entity creation path — the entity gets a `LIKE_COMMAND` engagement (strength 0.8), making it an immediate candidate for enrichment by the learn agent.

### 4. /memory (knowledge browsing)

View and manage the knowledge store:
- `/memory` — list all entities with fact counts
- `/memory [number]` — show entity details and facts
- `/memory [number] delete` — remove an entity

### 5. /interests (interest graph visibility)

View what Penny thinks you care about:
- `/interests` — show ranked entities by computed interest score
- Displays: entity name, interest score, fact count, last activity

### 6. Passive engagements (no user action required)

- **Emoji reactions** to Penny's messages — positive reaction = positive engagement, negative = negative. **Negative reaction on a proactive message = strong "stop" engagement** that sharply drops research priority.
- **Conversation patterns** — repeated questions, follow-ups accumulate as engagements
- **Search patterns** — what the user asks Penny to search for reveals what they care about

---

## Data Model

### Entity

```
entity
  id          INTEGER PRIMARY KEY
  user        TEXT (indexed)
  name        TEXT              -- lowercased canonical name
  created_at  TIMESTAMP
  updated_at  TIMESTAMP
  embedding   BLOB (nullable)  -- serialized float32 vector
```

### Fact

Individual facts with full provenance tracking:

```
fact
  id                    INTEGER PRIMARY KEY
  entity_id             INTEGER FK → entity (indexed)
  content               TEXT
  source_url            TEXT (nullable)
  source_search_log_id  INTEGER FK → searchlog (indexed)
  source_message_id     INTEGER FK → messagelog (indexed)
  learned_at            TIMESTAMP
  embedding             BLOB (nullable)
```

### SearchLog

Search results with trigger tracking:

```
searchlog
  id               INTEGER PRIMARY KEY
  timestamp        TIMESTAMP (indexed)
  query            TEXT (indexed)
  response         TEXT
  duration_ms      INTEGER (nullable)
  extracted        BOOLEAN DEFAULT FALSE
  trigger          TEXT DEFAULT 'user_message' (indexed)  -- user_message | learn_command | penny_enrichment
  learn_prompt_id  INTEGER FK → learnprompt (nullable, indexed)
```

### LearnPrompt (new)

First-class learning prompt with lifecycle tracking:

```
learnprompt
  id                  INTEGER PRIMARY KEY
  user                TEXT (indexed)
  prompt_text         TEXT              -- original user text
  status              TEXT DEFAULT 'active' (indexed)  -- active | completed
  searches_remaining  INTEGER DEFAULT 0
  created_at          TIMESTAMP
  updated_at          TIMESTAMP
```

### Engagement

User interest events:

```
engagement
  id                INTEGER PRIMARY KEY
  user              TEXT (indexed)
  entity_id         INTEGER FK → entity (nullable, indexed)
  preference_id     INTEGER FK → preference (nullable, indexed)
  engagement_type   TEXT (indexed)  -- see Engagements Reference
  valence           TEXT           -- positive | negative | neutral
  strength          FLOAT          -- 0.0-1.0
  source_message_id INTEGER FK → messagelog (nullable, indexed)
  created_at        TIMESTAMP (indexed)
```

**Heat** — persistent interest score on each entity, maintained by the `HeatEngine`. Heat accumulates from engagement touches (+3.0 per positive engagement) and decays over wall-clock time (7-day half-life). Drives research priority, notification selection, and context injection ranking.

### Preference

```
preference
  id          INTEGER PRIMARY KEY
  user        TEXT (indexed)
  topic       TEXT
  type        TEXT              -- like | dislike
  created_at  TIMESTAMP
  embedding   BLOB (nullable)
```

### Preference-entity relationships (computed, not stored)

No explicit join table. Preferences and entities both have embeddings — relationships computed via cosine similarity on the fly.

### Embedding generation

All embeddings generated by Ollama's embedding model (local, no API cost). Stored as binary blobs in SQLite (10,240 bytes per embedding = 2,560 float32s). Similarity search via cosine distance computed in Python.

---

## Background Loops

### Loop 1: Extraction Pipeline

**Trigger**: Periodic during idle. Processes unextracted SearchLogs and unprocessed MessageLogs.

**Three phases:**

1. **Search log processing** — For each unextracted SearchLog:
   - Check `trigger` to determine mode (`allow_new_entities` = true for user-triggered, false for penny-triggered)
   - Two-pass entity/fact extraction via LLM
   - Create engagements (for user-triggered searches only)
   - Send fact discovery notification to user (for user-triggered searches only)
   - Mark as extracted

2. **Message processing** — For each unprocessed message:
   - Extract entities and facts (always allows new entities — messages are user-triggered)
   - Extract preferences from emoji reactions
   - Create follow-up question engagements
   - Mark as processed

3. **Embedding backfill** — Generate embeddings for entities/facts/preferences that don't have them

**Two-mode extraction:**
- **Full mode** (user-triggered): Identifies known AND new entities, creates both
- **Known-only mode** (penny-triggered): Only matches against known entities, never creates new ones. Uses a specialized prompt that instructs the LLM to only look for known entities.

**Entity name validation** (applied to new entity candidates in full mode):

When the LLM identifies new entity candidates, they pass through a two-layer validation before creation. This prevents garbage entities (web scraping artifacts, verbose descriptions, tangential mentions) from polluting the knowledge store.

1. **Structural filter** (deterministic, cheap) — reject candidates that are clearly not entity names:
   - Name exceeds 8 words (paragraphs-as-names, verbose descriptions)
   - Contains LLM output artifacts: `{topic}`, `{desccription}`, `confidence score:`, `-brief:`
   - Starts with a digit + period (numbered list items from structured output)
   - Contains URLs, markdown formatting (`**`), or newlines

2. **Semantic filter** (embedding-based) — reject candidates that are topically unrelated to the triggering content:
   - Embed the candidate name and the search query (or user message) that triggered extraction
   - Reject if cosine similarity falls below threshold (tunable, ~0.35)
   - Catches tangential entities that happen to appear in search results but aren't what the user asked about

The structural filter does the heavy lifting — in testing against production data, it correctly rejected 31% of entities (verbose descriptions, metadata artifacts, web boilerplate) with zero false positives. The semantic filter provides a secondary defense against topically unrelated entities that pass structural checks, though it has a precision/recall trade-off at the margins (some legitimate proper nouns with low query similarity may be rejected).

### Loop 2: Penny Enrichment (Learn Agent)

**Trigger**: Periodic during idle. Picks the highest-priority known entity.

**What it does:**
1. Score all entities: `interest × (1/fact_count)`
2. Pick the top candidate
3. Search for it (enrichment or briefing mode)
4. Tag the SearchLog as `trigger=penny_enrichment`

The extraction pipeline picks up the SearchLog on its next pass (known-only mode — no new entities, just facts). Notification to the user comes from the extraction pipeline, same as all other searches.

### Loop 3: Entity Cleaner

**Trigger**: Once per 24 hours.

- Identifies duplicate entities via LLM
- Merges facts and engagements to canonical entity
- Deduplicates facts
- Regenerates embeddings

---

## Fact Discovery Notifications

When the extraction pipeline discovers new facts, it sends one proactive message per entity:

- **Known entity, new facts**: "I just learned some new stuff about {entity}..."
- **New entity**: "I just discovered {entity} and learned {count} facts about it"

Each message covers one entity and its new facts. If extraction finds facts about 3 entities, that's 3 messages. Notifications include an image when available.

### Proactive messaging cadence

Penny uses exponential backoff for proactive messages, reset by user engagement:

- **Start eager**: when the user is actively chatting, Penny messages discoveries immediately as they're found
- **Back off on silence**: each proactive message sent without a user reply doubles the delay before the next one (e.g. 0 → 1min → 2min → 4min → 8min → ...)
- **Reset on engagement**: any user action (message, reaction, command) resets the backoff to zero — Penny becomes proactive again immediately

This matches the user's energy: when they're engaged, Penny is chatty and shares everything. When they go quiet, Penny gradually quiets down too. No fixed idle threshold — the backoff is purely driven by whether the user is responding.

---

## Engagements Reference

| Engagement | Source | Strength | Example |
|---|---|---|---|
| /learn command | User types /learn X | Very high (1.0) | `/learn kef ls50 meta` |
| /like command | User types /like X | High (0.8) | `/like audio equipment` |
| /dislike command | User types /dislike X | High negative (0.8) | `/dislike sports` |
| Explicit statement | "I love X" in message | High (0.7) | "I'm really into espresso" |
| Search initiated | User searches for X | Medium-high (0.6) | "search for LS50 Meta reviews" |
| Follow-up question | User asks more about X | Medium (0.5) | "what about the LS50's bass response?" |
| Negative reaction on proactive msg | 👎 on unsolicited message | High negative (0.8) | "stop telling me about this" |
| Positive reaction on proactive msg | 👍 on unsolicited message | Medium (0.5) | "more like this" |
| Positive reaction on normal msg | 👍 on user-initiated message | Medium-low (0.3) | casual approval |
| Mentioned in passing | X appears but isn't focus | Low (0.2) | "I was listening on my LS50s when..." |

---

## Implementation Phases

### Phase 1 — Search Trigger Tracking + LearnPrompt Model

Add `trigger` and `learn_prompt_id` columns to `SearchLog`. Create `LearnPrompt` table. Extend `log_search()` and `SearchTool` to pass trigger through. Add LearnPrompt CRUD to database. All additive — no behavior changes.

**Files**: `models.py`, `migrations/0019_*.py`, `database.py`, `constants.py`, `builtin.py`

### Phase 2 — Two-Mode Extraction Pipeline + Entity Validation

Add `allow_new_entities` parameter to extraction. Check `search_log.trigger` to determine mode. Add known-only identification prompt. Penny-triggered searches produce facts only. Add structural and semantic validation for new entity candidates — reject names that are too long, contain LLM output artifacts, or are semantically distant from the triggering query.

**Files**: `extraction.py`, `prompts.py`, `constants.py`, `tests/agents/test_extraction.py`
**Blocked by**: Phase 1

### Phase 3 — Learn Agent Trigger Tagging

Set `trigger=penny_enrichment` on SearchTool before learn agent searches. Ensures extraction pipeline processes these in known-only mode.

**Files**: `learn.py`, `tests/agents/test_learn.py`
**Blocked by**: Phase 1

### Phase 4 — /learn Command Rewrite

Create LearnPrompt on `/learn <topic>`. Generate search sequence via LLM. Execute searches with `trigger=learn_command` and `learn_prompt_id`. New `/learn` status view showing provenance chain.

**Files**: `learn.py`, `prompts.py`, `responses.py`, `tests/commands/test_learn.py`
**Blocked by**: Phase 1, Phase 2

### Phase 5 — Fact Discovery Notifications

Track new facts per user per entity during extraction. Batch notifications per cycle. Skip for penny_enrichment. Include images.

**Files**: `extraction.py`, `responses.py`, `tests/agents/test_extraction.py`
**Blocked by**: Phase 2

### Phase 6 — Documentation

Update CLAUDE.md files and this design doc with implementation details.

**Blocked by**: All previous phases

---

## Dependency Graph

```
Phase 1 — Search Trigger Tracking + LearnPrompt Model
  (no dependencies, purely additive)
         │
         ├──────────────────┐
         ↓                  ↓
Phase 2 — Two-Mode       Phase 3 — Learn Agent
  Extraction Pipeline       Trigger Tagging
         │
         ├──────────────────┐
         ↓                  ↓
Phase 4 — /learn          Phase 5 — Fact Discovery
  Command Rewrite           Notifications
         │                  │
         └──────────────────┘
                  ↓
Phase 6 — Documentation
```

### Parallelism

- **Phases 2 and 3** can proceed in parallel once Phase 1 is done
- **Phases 4 and 5** can proceed in parallel once Phase 2 is done
- **Phase 6** is last

---

## System Diagram

Bird's eye view of how information moves through the system:

```mermaid
flowchart TD
    USER((User))

    USER -->|sends| MSG[User Message]
    USER -->|runs| LEARN[/learn]
    USER -->|reacts| RXN[Reaction]

    MSG -->|triggers| REPLY[Penny Reply]
    MSG -->|may trigger| MSRCH[Message Search]
    LEARN -->|triggers| LSRCH[Learn Search]

    MSG -->|extracted by| EXT[Extraction Pipeline]
    MSRCH -->|extracted by| EXT
    LSRCH -->|extracted by| EXT
    ESRCH -->|extracted by| EXT

    EXT -->|produces| ENT[Entity]
    EXT -->|produces| FACT[Fact]
    EXT -->|produces| ENG[Engagement]

    ENT --- FACT

    RXN -->|produces| ENG
    REPLY -->|can receive| RXN
    NOTIF -->|can receive| RXN

    ENG -->|computes| SCORE[Interest Score]

    SCORE -->|prioritizes| ESRCH[Enrichment Search]
    SCORE -->|selects| NOTIF[Notification]

    NOTIF -->|surfaces| ENT
    NOTIF -->|sends to| USER

    REPLY -->|sends to| USER
```

---

## Design Decisions

1. **User-gated entity creation** — The single most impactful change. Prevents garbage entities, irrelevant research targets, and engagement inflation. All three overnight issues (#317, #319, #320) are solved at the architectural level.
2. **Trigger tracking on SearchLog** — Simple string column, extensible. The extraction pipeline switches behavior based on a single field check. No complex state management.
3. **LearnPrompt as first-class object** — Enables the provenance chain (prompt → searches → facts → entities) and the `/learn` status view. Replaces the old pattern of "create entity + engagement and hope the research loop finds it."
4. **Enrichment loop is search-only** — The learn agent just scores entities and triggers searches. The extraction pipeline handles all fact extraction and user notification, keeping each process focused on one responsibility.
5. **Per-entity notifications** — One proactive message per (entity, new facts) pair. Gives the user clear, focused updates rather than a wall of discoveries.
6. **Known-only prompt variant** — Rather than filtering LLM output after the fact, give it a different prompt that only asks for known entity matches. Cheaper and more accurate.
7. **Default trigger is user_message** — All existing SearchLogs and the normal message flow get entity-creating behavior. Only the learn agent explicitly opts out.
