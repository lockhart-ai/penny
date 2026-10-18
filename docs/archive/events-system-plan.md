> **Archived** — removed from the live tree in `73c51e1c` (2026-03-06, #653 "Remove entity/fact knowledge system"). Replaced by the preference/memory system (`docs/reference/database.md`).

---

# Knowledge System V4: Events (Time-Aware Knowledge)

## Overview

V4 introduces **Events** — things that happen at a point in time, potentially involving multiple entities. This enables Penny to track ongoing stories, breaking news, and developments about topics the user cares about.

The three new primitives:
- **Event**: A time-stamped occurrence linking multiple entities (M2M)
- **FollowPrompt**: An ongoing monitoring subscription (like LearnPrompt but never-ending)
- **TheNewsAPI.com integration**: Structured news feed for event discovery

## News API: TheNewsAPI.com

**Endpoint**: `GET https://api.thenewsapi.com/v1/news/all`
- Query by keyword (`search`), date range (`published_after`), language
- Returns structured articles: `title`, `description`, `url`, `published_at`, `source`

**Client**: `NewsTool` in `tools/news.py` — uses `httpx` async HTTP client.

## Data Model

### Event table
- `headline`, `summary`, `occurred_at`, `source_url`, `external_id` (URL for dedup)
- `notified_at` — when user was told
- `embedding` — headline embedding for similarity dedup
- M2M link to entities via `EventEntity` junction table

### FollowPrompt table
- `prompt_text` — user's natural language topic
- `query_terms` — LLM-generated search terms (JSON list)
- `status` — active | cancelled
- `last_polled_at` — round-robin fairness for polling

## Architecture

### EventAgent
Runs on PeriodicSchedule (between ExtractionPipeline and LearnAgent). Each tick:
1. Get next FollowPrompt to poll (oldest `last_polled_at`)
2. Query TheNewsAPI.com `/all` with prompt's `query_terms`
3. Three-layer dedup: URL match → normalized headline → embedding similarity (0.90, 7-day window)
4. Create Event records for new articles
5. Link entities via LLM extraction (full mode — creates new entities)
6. Update `last_polled_at`

Only instantiated if `NEWS_API_KEY` is configured.

### NotificationAgent (extended)
Third stream added between learn completions and fact discoveries:
1. Learn completion announcements (bypass backoff)
2. **Event notifications** (respect backoff, timeliness bonus)
3. Fact discovery notifications (respect backoff)

Event scoring: `sum(linked_entity_interest) + timeliness_bonus` where timeliness = `2^(-hours_since / half_life)` (default half-life: 24h).

### Commands
- `/follow <topic>`: Create FollowPrompt, LLM generates `query_terms`
- `/follow`: List active subscriptions
- `/unfollow <N>`: Cancel subscription
- `/events`: Recent events (7-day window), sorted by `occurred_at` DESC
- `/events <N>`: Full detail with linked entities

### Config Params
- `EVENT_POLL_INTERVAL` (3600s) — minimum seconds between event agent polls
- `EVENT_DEDUP_SIMILARITY_THRESHOLD` (0.90) — embedding cosine similarity for headline dedup
- `EVENT_DEDUP_WINDOW_DAYS` (7) — days to look back for dedup comparison
- `EVENT_TIMELINESS_HALF_LIFE_HOURS` (24.0) — half-life for event timeliness decay in notifications

## Implementation PRs

1. PR #484: Event + EventEntity data models, migration, EventStore
2. PR #485: FollowPrompt data model, migration, FollowPromptStore
3. PR #486: TheNewsAPI.com client (NewsTool)
4. PR #487: /follow and /unfollow commands
5. PR #488: /events command
6. PR #489: EventAgent
7. PR #490: Event notification stream
8. PR #491: Documentation updates
