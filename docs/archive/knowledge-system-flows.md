> **Archived** — removed from the live tree in `73c51e1c` (2026-03-06, #653 "Remove entity/fact knowledge system"). Replaced by the preference/memory system (`docs/reference/database.md`).

---

# Knowledge System v2 — Sequence Diagrams

## Search Triggers

### User Message Search

```mermaid
sequenceDiagram
    actor User
    participant Penny
    participant DB as Database

    User->>Penny: "what's a good bookshelf speaker?"
    Penny->>DB: search + log (trigger=user_message)
    Penny->>User: response
```

### /learn Search Sequence

```mermaid
sequenceDiagram
    actor User
    participant Penny
    participant DB as Database

    User->>Penny: "/learn ai conferences in europe"
    Penny->>DB: create LearnPrompt(status=active)
    Penny->>User: "Okay, I'll learn more about ai conferences in europe"
    Penny->>DB: generate 3-5 queries, execute each (trigger=learn_command, learn_prompt_id=1)
    Penny->>DB: complete LearnPrompt
```

### Enrichment Search

```mermaid
sequenceDiagram
    participant LL as Learn Agent
    participant DB as Database

    LL->>DB: score entities (interest x 1/fact_count x staleness)
    LL->>DB: search top candidate + log (trigger=penny_enrichment)
```

## Extraction Pipeline

### Full Mode (user_message, learn_command)

New entities allowed. Validates candidates before creation.

```mermaid
sequenceDiagram
    actor User
    participant Extract as Extraction Pipeline
    participant DB as Database
    participant LLM as Ollama

    Extract->>DB: get unprocessed SearchLogs (user-triggered)
    Extract->>LLM: identify entities in search results
    Note over Extract: Validate new candidates (structural + semantic filters)
    Extract->>DB: create validated entities
    Extract->>DB: extract + store facts for all present entities
    Extract->>DB: mark SearchLogs as processed
    Extract->>User: one proactive message per (entity, new facts)
```

### Known-Only Mode (penny_enrichment)

No new entities. Facts only for known entities.

```mermaid
sequenceDiagram
    actor User
    participant Extract as Extraction Pipeline
    participant DB as Database

    Extract->>DB: get unprocessed SearchLogs (penny-triggered)
    Extract->>DB: extract + store facts for known entities only
    Extract->>DB: mark SearchLogs as processed
    Extract->>User: one proactive message per (entity, new facts)
```

## Engagement Signals

### SEARCH_INITIATED

Recorded by extraction when it finds entities in a user-triggered search.

```mermaid
sequenceDiagram
    participant Extract as Extraction Pipeline
    participant DB as Database

    Note over Extract: Processing user-triggered SearchLog
    Extract->>DB: found entity "kef ls50 meta" in search results
    Extract->>DB: add_engagement(SEARCH_INITIATED, strength=0.6, entity=kef_ls50)
```

### FOLLOW_UP_QUESTION

User asks about an entity Penny already knows.

```mermaid
sequenceDiagram
    actor User
    participant Penny
    participant DB as Database

    User->>Penny: "tell me more about the kef ls50 meta"
    Penny->>DB: entity match found via embedding similarity
    Penny->>DB: add_engagement(FOLLOW_UP_QUESTION, strength=0.5, entity=kef_ls50)
```

### LIKE_COMMAND / DISLIKE_COMMAND

Explicit preference. /like also creates the entity if it doesn't exist.

```mermaid
sequenceDiagram
    actor User
    participant Penny
    participant DB as Database

    User->>Penny: "/like mechanical keyboards"
    Penny->>DB: create Preference + Entity
    Penny->>DB: add_engagement(LIKE_COMMAND, strength=0.8, entity=mechanical_keyboards)
    Penny->>DB: find similar entities via embedding, boost each

    User->>Penny: "/dislike sports"
    Penny->>DB: create Preference
    Penny->>DB: add_engagement(DISLIKE_COMMAND, strength=-0.8, entity=sports)
```

### EMOJI_REACTION

Reactions on messages. Positive reinforces; negative suppresses.

```mermaid
sequenceDiagram
    actor User
    participant DB as Database

    User->>DB: 👍 on message about "obsidian"
    Note over DB: add_engagement(EMOJI_REACTION, strength=0.5, entity=obsidian)

    User->>DB: 👎 on proactive message about "sourdough"
    Note over DB: add_engagement(EMOJI_REACTION, strength=-0.8, entity=sourdough)
```

## Maintenance

### Entity Cleaner

Daily pass. Deduplicates entities and facts, merges engagement history.

```mermaid
sequenceDiagram
    participant Cleaner as Entity Cleaner
    participant DB as Database
    participant LLM as Ollama

    Cleaner->>DB: get all entities
    Cleaner->>LLM: identify duplicates (e.g. "kef ls50" vs "kef ls50 meta")
    LLM-->>Cleaner: duplicate groups + canonical names
    Cleaner->>DB: merge facts + engagements to canonical entity
    Cleaner->>DB: deduplicate facts
    Cleaner->>DB: regenerate embeddings
```

## Views

### /learn Status (Provenance Chain)

```mermaid
sequenceDiagram
    actor User
    participant Penny
    participant DB as Database

    User->>Penny: "/learn"
    Penny->>DB: LearnPrompt → SearchLogs → Facts → Entities

    Penny->>User: 1) 'speakers' ✓ — wharfedale (17), kef (8)<br/>2) 'ai conferences' ... — neurips (3)
```
