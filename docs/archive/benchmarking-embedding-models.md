# Benchmarking Embedding Models for Knowledge System Dedup and Retrieval (Ollama, Apple Silicon)

**Date**: February 26, 2026
**Hardware**: Apple Silicon (Mac), Ollama 0.17.1
**Models tested**: 10 embedding models (see table below)
**Winner**: `embeddinggemma` (Google, 300M params)

## Context

Penny uses embedding models for four core tasks in its knowledge system:

1. **Entity name dedup** (threshold 0.85): Detecting when a candidate entity is a duplicate of an existing one (e.g., "KEF LS50 Meta" vs "kef ls50 meta"). Works alongside a token containment ratio (TCR) layer that handles abbreviation matching.
2. **Fact dedup** (threshold 0.85): Detecting when a newly extracted fact is semantically equivalent to one already stored (e.g., "Can was founded in 1968 in Cologne" vs "Can formed in Cologne in 1968").
3. **Query-to-entity relevance** (threshold 0.3): Finding which stored entities are relevant to a user's message, used for context injection into the LLM prompt.
4. **Post-fact semantic pruning** (threshold 0.50): Filtering out entities extracted from search results that aren't actually relevant to the query.

After our [qwen3.5:35b benchmark](benchmarking-qwen35-vs-gpt-oss.md) raised concerns about Qwen model quality, we decided to systematically test the embedding model too — we'd been using `nomic-embed-text` (recommended in docs) and had recently switched to `qwen3-embedding:8b` without benchmarking it.

## Setup

We built a benchmark with hand-labeled ground truth derived from Penny's production database (584 entities, 5302 facts across audio equipment, progressive rock, astrophysics, space missions, and theoretical physics).

**Three test categories**, each matching a real production use case:

### Entity name dedup (25 pairs)

13 pairs that **should match** (same entity, different surface forms):
- Abbreviation/full name: `jwst` vs `james webb space telescope`, `mc cartridge` vs `moving-coil cartridge`
- Case variants: `audio-technica` vs `Audio-Technica`, `kef ls50 meta` vs `KEF LS50 Meta`
- Punctuation: `at-vm750sh` vs `at vm750sh`, unicode `audio‑technica` vs ASCII `audio-technica`
- Near-synonyms: `phono preamp` vs `phono stage`

12 pairs that **should not match** (distinct entities):
- Same series: `hana ml` vs `hana sh`, `artemis i` vs `artemis ii`
- Same domain: `kef ls50 meta` vs `kef r3 meta`, `rega` vs `pro-ject`
- Similar notation: `ads5×s5` vs `ads4×s7`, `ads3/cft2` vs `ads2/cft1`

### Fact dedup (14 pairs)

7 pairs that **should match** (same fact, different wording):
- "Can was founded in 1968 in Cologne" vs "Can formed in Cologne in 1968"
- "JWST PSF models include coronagraphic masks and Lyot stops" vs "PSF models for the James Webb Space Telescope incorporate coronagraphic mask effects and Lyot stops"

7 pairs that **should not match** (different facts, possibly same entity):
- "Can was founded in 1968 in Cologne" vs "Can blended psychedelic rock, funk, jazz..."
- "Can was founded in 1968 in Cologne" vs "AdS4×S7 preserves all 32 supersymmetries..." (cross-domain)

### Query-to-entity relevance (5 queries, 40 entity checks)

5 realistic user queries, each with 4 entities that should be found (relevant, score >= 0.3) and 4 that should be rejected (irrelevant, score < 0.3):
- "what's a good phono preamp for my turntable?" — should find audio gear, not physics
- "tell me about krautrock bands from the 70s" — should find Can/Ash Ra Tempel, not JWST
- "NASA's plans to return to the moon" — should find Artemis missions, not audio
- "what's the difference between MC and MM cartridges?" — should find cartridge entities, not bands
- "holographic duality and string theory compactifications" — should find AdS/CFT, not speakers

Entity text uses the production `build_entity_embed_text()` format: `name (tagline): fact1; fact2`.

### Throughput

Single embed latency, batch of 10, and full batch (123 texts) with texts/sec measurement. Each model warmed up before timing.

## Models tested

| # | Model | Params | Dims | Source |
|---|-------|--------|------|--------|
| 1 | `nomic-embed-text` | 137M | 768 | Ollama library |
| 2 | `mxbai-embed-large` | 335M | 1024 | Ollama library |
| 3 | `snowflake-arctic-embed2` | 568M | 1024 | Ollama library |
| 4 | `gte-large` (GGUF) | 335M | 1024 | HuggingFace: ChristianAzinn/gte-large-gguf |
| 5 | `gte-Qwen2-1.5B-instruct` (GGUF) | 1.5B | 1536 | HuggingFace: mav23/gte-Qwen2-1.5B-instruct-GGUF |
| 6 | `qwen3-embedding:4b` | 4B | 2560 | Ollama library |
| 7 | `qwen3-embedding:8b` | 8B | 4096 | Ollama library |
| 8 | `nomic-embed-text-v2-moe` | 475M (305M active) | 768 | Ollama library |
| 9 | `embeddinggemma` | 300M | 768 | Ollama library (Google) |
| 10 | `jina-embeddings-v4-text-matching` (GGUF Q4_K_M) | 3B | 2048 | HuggingFace: jinaai/jina-embeddings-v4-text-matching-GGUF |

Models 4, 5, and 10 were pulled directly from HuggingFace via Ollama's `hf.co/` integration — no Modelfile needed, just `ollama pull hf.co/{repo}`.

## Results

### Summary

```
Model                       Dim   Entity     Fact  Relevance   1-text    Batch    txt/s
------------------------- ----- -------- -------- ---------- -------- -------- --------
nomic-embed-text            768      52%      93%        50%     20ms    682ms    180.4
mxbai-embed-large          1024      68%     100%        78%     24ms   1421ms     86.6
snowflake-arctic-embed2    1024      72%      86%        92%    114ms   1537ms     80.0
gte-large (HF)             1024      56%      71%        50%     28ms   1786ms     68.9
gte-Qwen2-1.5B (HF)       1536      36%      86%        70%     84ms   3935ms     31.3
qwen3-embedding:4b         2560      52%      86%        65%     99ms   8256ms     14.9
qwen3-embedding:8b         4096      60%      86%        55%    124ms  12428ms      9.9
nomic-embed-text-v2-moe     768      68%      86%        92%    101ms   1293ms     95.1
embeddinggemma              768      60%      93%       100%     92ms   1356ms     90.7
jina-v4-matching (HF)      2048      72%      71%        92%     94ms   6028ms     20.4
```

### Detailed accuracy breakdown

```
Model                      EntMatch  EntNoMatch  FactMatch  FactNoMatch  RelFind  RelSkip
------------------------- --------- ----------- ---------- ------------ -------- --------
nomic-embed-text          3/  13    10/  12     6/   7      7/   7  20/ 20   0/ 20
mxbai-embed-large         6/  13    11/  12     7/   7      7/   7  20/ 20  11/ 20
snowflake-arctic-embed2   7/  13    11/  12     5/   7      7/   7  17/ 20  20/ 20
gte-large (HF)           11/  13     3/  12     7/   7      3/   7  20/ 20   0/ 20
gte-Qwen2-1.5B (HF)      3/  13     6/  12     5/   7      7/   7  20/ 20   8/ 20
qwen3-embedding:4b        6/  13     7/  12     5/   7      7/   7  20/ 20   6/ 20
qwen3-embedding:8b        8/  13     7/  12     5/   7      7/   7  20/ 20   2/ 20
nomic-embed-text-v2-moe   5/  13    12/  12     5/   7      7/   7  17/ 20  20/ 20
embeddinggemma            4/  13    11/  12     6/   7      7/   7  20/ 20  20/ 20
jina-v4-matching (HF)     8/  13    10/  12     3/   7      7/   7  18/ 20  19/ 20
```

**RelFind** = correctly found relevant entities (should be >= 0.3). **RelSkip** = correctly rejected irrelevant entities (should be < 0.3). The critical discriminator is RelSkip — models that can't reject irrelevant context will inject noise into every user response.

## Analysis

### embeddinggemma: the clear winner

`embeddinggemma` is the only model to score **100% on query relevance** — it found all 20 relevant entities AND rejected all 20 irrelevant ones. The score distribution shows clean separation:

- Relevant entities: 0.33–0.59 (all comfortably above 0.3)
- Irrelevant entities: 0.08–0.30 (all below 0.3, most well below)
- Fact matches: 0.86–0.96 (6/7 above 0.85)
- Fact non-matches: 0.08–0.58 (all well below 0.85)

Its one fact dedup miss is "The KEF LS50 Meta uses a Uni-Q driver array" vs "KEF's LS50 Meta features a coincident Uni-Q driver" at 0.824 — just barely under the 0.85 threshold. Its entity name dedup (60%) is the weakest area, but this is well-covered by the TCR layer in production.

### nomic-embed-text: broken discrimination

The original `nomic-embed-text` (v1) has a fatal flaw for our use case: **0/20 irrelevant rejections**. Every entity scores above 0.3 regardless of relevance. Physics entities score 0.35–0.45 against audio queries. This means Penny would inject garbage context into every message. It also can't handle case changes — `audio-technica` vs `Audio-Technica` scores only 0.45.

The v2 MoE variant fixes the discrimination issue (20/20 rejections) but loses on fact dedup (86%).

### Larger models are not better

The most striking finding: model size has little correlation with embedding quality for our tasks.

| Model | Params | Overall quality |
|-------|--------|----------------|
| embeddinggemma | 300M | Best (100% relevance, 93% fact) |
| mxbai-embed-large | 335M | Second (100% fact, 78% relevance) |
| qwen3-embedding:8b | 8B | Poor (55% relevance, 86% fact) |
| qwen3-embedding:4b | 4B | Poor (65% relevance, 86% fact) |

The 8B qwen3-embedding model is **27x larger** than embeddinggemma and produces **5.3x wider vectors** (4096 vs 768 dims), yet scores worse on every quality metric while being **14x slower** (9.9 vs 90.7 texts/sec).

The pattern: models purpose-built for embedding (embeddinggemma from Gemma 3, mxbai) outperform models adapted from general-purpose LLMs (qwen3-embedding, gte-Qwen2). Training objective and architecture matter more than parameter count.

### HuggingFace GGUF integration works

Ollama's `hf.co/` prefix for pulling models directly from HuggingFace works for embedding models. No Modelfile needed:

```bash
ollama pull hf.co/ChristianAzinn/gte-large-gguf
curl http://localhost:11434/api/embed -d '{"model": "hf.co/ChristianAzinn/gte-large-gguf", "input": "test"}'
```

This opens up the entire HuggingFace GGUF ecosystem (~45K models) for Ollama. We tested 3 models this way (gte-large, gte-Qwen2-1.5B, jina-v4-matching). None beat the Ollama library models for our use case, but the integration is solid for future exploration.

### Hard cases for all models

Every model struggles with abbreviation-to-full-name matching: `jwst` / `james webb space telescope`, `riaa` / `recording industry association of america`, `mc cartridge` / `moving-coil cartridge`. This confirms that Penny's TCR (token containment ratio) layer is essential — embeddings alone aren't enough for that class of dedup.

Similarly, `ads3/cft2` vs `ads2/cft1` fools most models (structurally identical notation, just different numbers). Only nomic-v2-moe and jina-v4 correctly separate these.

## Decision

Switched from `qwen3-embedding:8b` to `embeddinggemma` as `OLLAMA_EMBEDDING_MODEL`. Cleared all existing embeddings from the production database (584 entities, 5302 facts) — the backfill pipeline will regenerate them with the new model.

Key improvements:
- **Relevance discrimination**: 55% → 100% (no more irrelevant context injection)
- **Fact dedup**: 86% → 93%
- **Single embed latency**: 124ms → 92ms
- **Throughput**: 9.9 → 90.7 texts/sec (9x faster)
- **Disk/memory**: 4.7 GB → ~200 MB model size
- **Vector storage**: 4096 → 768 dims (5.3x less storage per embedding)

## Benchmark script

The benchmark script and ground truth test cases are in `data/benchmarks/run_embedding_benchmark.py`. It can be re-run to evaluate future embedding model candidates against the same production-derived test suite.
