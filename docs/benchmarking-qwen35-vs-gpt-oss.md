# Benchmarking qwen3.5:35b vs gpt-oss:20b for Agentic Workloads (Ollama, Apple Silicon)

**Date**: February 26, 2026
**Hardware**: Apple Silicon (Mac), Ollama 0.17.1
**Models**: `qwen3.5:35b` (just released), `gpt-oss:20b`

## Context

We run [Penny](https://github.com/lockhart-ai/penny), a local-first AI agent that uses Ollama for all LLM inference. Penny's workload is heavily agentic — entity extraction, fact extraction, sentiment analysis, schedule parsing, and conversational chat — with most tasks requiring structured JSON output via Ollama's `format` parameter (JSON schema mode).

We previously tested `qwen3:32b` and found it unusable due to excessive thinking tokens (15k+) and unreliable structured JSON output. When `qwen3.5:35b` dropped, we ran a systematic comparison against our production model (`gpt-oss:20b`) using 11 real prompts pulled from our production prompt log.

## What the prompts do

Penny builds a knowledge graph from conversations and web searches. When the user asks a question, Penny searches the web via Perplexity, then runs the results through a multi-step extraction pipeline — all powered by local LLM calls. The key tasks:

**Entity identification (known + new)**: Given search results or a user message and a list of ~hundreds of already-known entities, the model must: (a) identify which known entities appear in the text by cross-referencing the list, and (b) identify genuinely new entities not already in the list, with short taglines. This requires carefully reading the known list and matching — not just extracting every noun. The prompt includes rules like "use short canonical names (1-5 words)", "skip vague concepts, dates, locations, institutions", etc.

**Known-only entity matching**: A simpler variant — given search results and a known entity list, return ONLY entities from the known list that appear. No new entity creation. The key constraint: "do NOT identify new entities — only match against the known list."

**Fact extraction**: Given search results about a specific entity and a list of already-known facts, extract new specific, verifiable facts. Rules include "do NOT paraphrase facts already listed", "do NOT store negative facts", "keep each fact concise (one sentence)", and "if no genuinely new facts are found, return an empty list."

**Sentiment extraction**: Analyze a user message for opinions about named entities. Only return entities where the user expresses clear positive or negative sentiment — skip neutral mentions.

**Schedule parsing**: Convert natural language like "daily 9:30am check the news" into structured components: a timing description, the prompt text, and a cron expression. The prompt specifies the user's timezone.

**Other tasks**: Search query generation for the `/learn` command, git commit message → casual one-liner transformation, and conversational chat (with and without tool definitions for web search).

All structured tasks use Ollama's `format` parameter with Pydantic-derived JSON schemas to enforce output structure.

## Setup

We extracted 11 representative prompts from our SQLite prompt log, covering these tasks:

| # | Task | JSON? | Prompt Size |
|---|------|-------|-------------|
| 1 | Fact extraction (from search results) | Yes (schema) | 3.1k chars |
| 2 | Known entity matching (from search results) | Yes (schema) | 15k chars |
| 3 | Entity identification — known + new (from search results) | Yes (schema) | 15k chars |
| 4 | Entity identification — known + new (from user message) | Yes (schema) | 26k chars |
| 5 | Chat with tool definitions (search tool) | No | 4.4k chars |
| 6 | Chat without tools | No | 1.2k chars |
| 7 | Sentiment extraction | Yes (schema) | 743 chars |
| 8 | Schedule parsing (natural language → cron) | Yes (`"json"`) | 927 chars |
| 9 | Search query generation | Yes (schema) | 194 chars |
| 10 | Git commit → casual announcement | No | 1.4k chars |
| 11 | Fact extraction (from user message) | Yes (schema) | 813 chars |

The prompts with large sizes (15k-26k chars) are big because they include the full known entity list (hundreds of entities) that the model must cross-reference against.

Each prompt was sent to both models sequentially (no GPU contention) via the Ollama `/api/chat` endpoint.

## Issue 1: Thinking mode is catastrophic

With thinking enabled (the default for qwen3.5), the model is essentially unusable for our workload:

- **Benchmark 1 (fact extraction)**: qwen3.5 spent **536 seconds** generating **40,071 characters** of thinking. It performed a thorough, correct analysis — identifying all the right facts, cross-checking against known facts, filtering negatives, going through 6+ revision cycles. Then it output: `{"facts": ["A man wearing a red shirt and blue pants."]}` — a hallucinated response completely disconnected from its own reasoning.

- The thinking-to-output bridge appears broken when combined with structured output constraints. The model reasons correctly in the thinking trace but the constrained decoding for the JSON response produces garbage.

- Thinking token volume is worse than qwen3: **25k-40k chars** of thinking vs qwen3's ~15k, with wall times of 5-9 minutes per prompt.

All subsequent benchmarks were run with `think: false`.

## Issue 2: Ollama's `format` parameter is ignored

With thinking disabled, we discovered that qwen3.5 **completely ignores Ollama's `format` parameter** — both JSON schema mode and simple `"json"` mode. The model returns well-structured plain text (bullet points) instead of JSON, even when the format constraint is set.

This works fine with `gpt-oss:20b`, which respects the `format` parameter as a hard constraint.

**Workaround**: Adding explicit JSON instructions to the prompt text itself (e.g., `Respond with JSON only: {"facts": ["fact1", ...]}`) does produce valid JSON. This is how we ran the final benchmarks.

Note: This may be an Ollama integration issue rather than a model issue — the qwen3.5 Ollama release uses the new `RENDERER qwen3.5` / `PARSER qwen3.5` directives rather than a traditional chat template, and format enforcement may not be fully implemented yet.

## Results: Speed (think=false, explicit JSON instructions)

| Benchmark | gpt-oss:20b | qwen3.5:35b | Winner |
|---|---|---|---|
| Fact extraction (search) | 35.5s | 28.6s | qwen 1.2x |
| Known entity matching | 17.7s | 13.9s | qwen 1.3x |
| Entity ID (search, new+known) | 51.2s | 40.9s | qwen 1.3x |
| Entity ID (message, new+known) | 30.7s | 28.4s | qwen 1.1x |
| Chat with tools | 9.9s | 20.8s | **gpt-oss 2.1x** |
| Chat without tools | 7.7s | 8.6s | ~tie |
| Sentiment extraction | 6.0s | 1.7s | qwen 3.6x |
| Schedule parsing | 8.3s | 3.7s | qwen 2.3x |
| Query generation | 2.3s | 1.5s | qwen 1.5x |
| Git announcement | 6.4s | 2.2s | qwen 2.9x |
| Message fact extraction | 7.2s | 2.1s | qwen 3.4x |

**qwen3.5 is faster on 9/11 benchmarks.** The speed advantage comes from not generating thinking tokens — gpt-oss generates 300-8400 chars of thinking per prompt, which takes time even though it improves accuracy. Raw throughput is actually lower: qwen at ~23 tok/s vs gpt-oss at ~53 tok/s.

gpt-oss is notably faster on the chat-with-tools benchmark (2x), suggesting overhead from tool definition processing in qwen3.5.

### Detailed benchmark output

```
Benchmark                         gpt-oss:20b    qwen3.5:35b    Speedup   JSON gpt  JSON qwen
------------------------------ -------------- -------------- ---------- ---------- ----------
fact_extraction                       35537ms        28584ms      1.24x         ok         ok
known_entity_identification           17654ms        13901ms      1.27x         ok         ok
entity_identification_search          51234ms        40911ms      1.25x         ok         ok
entity_identification_message         30706ms        28403ms      1.08x         ok         ok
chat_with_tools                        9945ms        20764ms      0.48x          -          -
chat_no_tools                          7723ms         8599ms      0.90x          -          -
sentiment_extraction                   6031ms         1687ms      3.57x         ok         ok
schedule_parse                         8259ms         3650ms      2.26x         ok         ok
learn_query_generation                 2321ms         1534ms      1.51x         ok         ok
git_commit_announcement                6391ms         2177ms      2.94x          -          -
message_fact_extraction                7243ms         2126ms      3.41x         ok         ok
```

### Thinking tokens and output tokens

gpt-oss uses moderate thinking on every prompt (300-8400 chars). qwen3.5 was run with `think: false`.

```
Benchmark                        gpt-oss think      qwen think  gpt out tok qwen out tok
------------------------------ --------------- --------------- ------------ ------------
fact_extraction                         4301ch             0ch          268          460
known_entity_identification             2076ch             0ch           20           36
entity_identification_search            8419ch             0ch          180          671
entity_identification_message           3682ch             0ch           60          183
chat_with_tools                          332ch             0ch          413          379
chat_no_tools                           1043ch             0ch          354          160
sentiment_extraction                    1315ch             0ch            2            7
schedule_parse                          1387ch             0ch           44           49
learn_query_generation                   316ch             0ch           12           12
git_commit_announcement                 1139ch             0ch          294           12
message_fact_extraction                 1519ch             0ch           18           17
```

Note: qwen3.5 generally produces more output tokens for the same task (460 vs 268 for fact extraction, 671 vs 180 for entity ID). Combined with the lower raw throughput (23 tok/s vs 53 tok/s), this means qwen's wall-time advantage comes entirely from skipping thinking — not from being a faster model.

## Results: Correctness

This is where things fall apart. Out of 11 benchmarks:

### Correct on both (6/11)
- **Fact extraction** (#1): Both produced valid, relevant facts. qwen was slightly more verbose.
- **Sentiment extraction** (#7): Both correctly returned empty (no strong sentiment in the input).
- **Query generation** (#9): Both generated good search queries.
- **Git announcement** (#10): Both produced good casual one-liners.
- **Message fact extraction** (#11): Both extracted the same core fact.
- **Chat without tools** (#6): Both produced reasonable conversational responses.

### Incorrect from qwen3.5 (5/11)

- **Known entity matching** (#2): The prompt provides a list of known entities and says "ONLY return entities from the known list. Do NOT identify new entities." The search results discuss the Swedish prog band Ragnarök and mention labels/venues in passing. gpt-oss correctly returned only `["ragnarök (swedish progressive rock band)"]` — the one entity that was actually in the known list. qwen returned `["ragnarök (swedish progressive rock band)", "big wreck", "silence records", "byteatern"]` — three of those were mentioned in the text but were **not in the provided known list**. It pattern-matched text mentions instead of cross-referencing the list.

- **Entity ID from message** (#4): The user message mentions several audio speakers. The known entity list included "kef r3 meta", "Sonus Faber Lumina II", "JBL HDI 1600", and "Monitor Audio Silver 50". gpt-oss correctly matched 4 of them as known and identified 1 new entity. qwen returned `"known": []` — an empty list — and put everything as `"new"`, completely failing to match entities that were exact matches in the known list.

- **Entity ID from search** (#3): A news article search result. qwen put 14 items in the "known" list including "strait of hormuz", "galápagos", and "cyberpunk: edgerunners" — none of which were in the actual known entity list. It also created new entities for "russia", "ukraine", "kyiv", "nigeria" — all locations/countries that the prompt rules explicitly say to skip. gpt-oss correctly identified 1 known entity and 8 new entities, all following the naming and filtering rules.

- **Schedule parsing** (#8): The prompt says `User timezone: America/Toronto` and the user input is "daily 9:30am." gpt-oss correctly returned `"cron_expression": "30 9 * * *"` (9:30 AM in the user's timezone). qwen returned `"30 14 * * *"` — it converted to UTC, ignoring the timezone instruction in the prompt.

- **Chat with tools** (#5): Neither model actually invoked the search tool (both answered from training data), but qwen was also 2x slower.

### Pattern

The failures cluster around tasks requiring **careful cross-referencing** — reading a provided list, matching against it, and respecting constraints. These are exactly the tasks where gpt-oss's thinking tokens (2-8k chars of reasoning through the list) pay off. Without thinking, qwen3.5 resorts to pattern matching and gets the details wrong.

## Conclusion

For our agentic workload, **qwen3.5:35b is not a viable replacement for gpt-oss:20b**:

1. **Thinking mode is broken** with structured output — correct reasoning but hallucinated JSON responses, at 10x the latency.
2. **Ollama's format parameter is ignored** — requires prompt-level JSON instructions as a workaround.
3. **Without thinking, correctness degrades significantly** on tasks requiring list cross-referencing and constraint following (5/11 benchmarks wrong).
4. **Speed advantage is real but narrow** — faster wall time due to no thinking overhead, but lower raw throughput (23 vs 53 tok/s).

The fundamental tension: qwen3.5 needs thinking tokens for accuracy on complex prompts, but its thinking mode is too slow and produces disconnected output. gpt-oss:20b's moderate thinking (300-8k chars) hits the sweet spot — enough reasoning for correctness without blowing up latency.

We'll check back when Ollama's qwen3.5 integration matures (format enforcement, thinking stability), but for now gpt-oss:20b remains our production model.

## Addendum: Ollama 0.17.3 retest (February 26, 2026)

Ollama 0.17.3 shipped a fix for "tool calls in the Qwen 3 and Qwen 3.5 model families not being parsed correctly if emitted during thinking." We retested the known entity matching benchmark (#2) — one of the tasks qwen3.5:35b previously failed — with thinking re-enabled. We also tested `qwen3.5:27b`, the dense (non-MoE) variant.

### Results (known entity matching, thinking enabled)

| Model | Time | Thinking | Output tokens | Correct? |
|-------|------|----------|---------------|----------|
| gpt-oss:20b | 11.8s | 1,108 chars | 18 | Yes |
| qwen3.5:35b | 103.7s | 6,666 chars | 41 | Yes |
| qwen3.5:27b | 578.6s | 12,089 chars | 21 | Yes |

All three models now return the correct answer: `["ragnarök (swedish progressive rock band)"]`. Previously, qwen3.5:35b with thinking disabled returned 4 entities including 3 hallucinated ones not in the known list.

### Key observations

- **Thinking fixes correctness for qwen3.5**, but the speed penalty is prohibitive. The 35b MoE model is 9x slower than gpt-oss; the 27b dense model is 49x slower.
- **qwen3.5:27b is not viable on Apple Silicon**: 188s warmup, 578s for a single prompt, 12k chars of thinking to reach the same answer gpt-oss gets in 1.1k chars.
- **Thinking efficiency matters more than model size**: gpt-oss needs 6x less thinking to reach the correct answer. The qwen models reason correctly but verbosely — lots of "let me re-check" cycles that don't improve the output.
- **Benchmark scores vs real-world performance**: qwen3.5 benchmarks well on MMLU/HumanEval/GPQA, but these test raw capability in free-form output. They don't test constrained generation (JSON schema + list cross-referencing simultaneously), which is our primary workload. Training objective and inference architecture matter more than benchmark rankings for local deployment.

### Decision

No change — gpt-oss:20b remains our production model. The Ollama 0.17.3 fix makes qwen3.5 *correct* with thinking enabled, but not *practical* for latency-sensitive local inference.
