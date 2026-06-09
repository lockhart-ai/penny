# Prompt-validation harness

Long-lived validation for Penny's model-facing prompts: the chat agent's
collection/skill **authoring**, and the recall **retrieval** design. These
are **not** pytest — they need a live LLM (gpt-oss) and embedding model
(embeddinggemma) on Ollama and take minutes, so they run by hand, not in
`make check`.

The harness reads the real prompt text out of the penny source via AST and
talks to Ollama through the OpenAI-compatible endpoint. It does **not**
import the `penny` package, so it runs on the host without the full dep set.

## Privacy

Everything here is synthetic. The repo is public — **no real user data** in
fixtures, prompts, or test cases. The fixtures use deliberately generic
topics (board games, espresso gear, houseplants, sci-fi novels). For ad-hoc
runs against real production traffic, a runner may add a `--live-db` flag
that pulls from `penny.db` directly; that data never lands in the repo.

## Running

One entry point, from the repo root. Each model-driven case runs
`--samples` times (default 5) to surface variance.

```bash
# Everything (all suites, 5 samples each)
PYTHONPATH=. uv run --python 3.12 --with openai -m scripts.prompt_validation

# List suites + cases
PYTHONPATH=. uv run --python 3.12 --with openai -m scripts.prompt_validation --list

# One suite
PYTHONPATH=. uv run --python 3.12 --with openai -m scripts.prompt_validation \
    --suite collection_lifecycle

# One case, more samples (e.g. chasing variance on a flaky case)
PYTHONPATH=. uv run --python 3.12 --with openai -m scripts.prompt_validation \
    --suite collection_lifecycle --case update-focus-swap --samples 10
```

Env overrides: `LLM_MODEL` (default `gpt-oss:20b`), `LLM_EMBEDDING_MODEL`
(default `embeddinggemma`), `LLM_API_URL` (default `http://localhost:11434`).

## Suites

- **`collection_lifecycle`** — chat agent authoring/operating collections
  against the REAL source prompts: create (notify / silent / digest /
  cadence-in-request), update (add scope / focus-swap / silent↔notify flip /
  cadence), archive (done / cancelled), query (read / from recall), and
  abstain cases (ambiguous one-shot, implicit prep → must NOT create).
- **`skills_extractor`** — the background collector that grows/tunes skills:
  teach / correct-subtractive / correct-scope / deprecate / lift / quiet-cycle.
- **`novel_patterns`** — generalization to intents with no seeded skill
  (URL watcher, recurring reminder, chat-pattern extraction, tool-gap email);
  lenient "sensible behavior" check, prints what the model did.
- **`retrieval`** — two-stage recall: stage-1 collection routing (inclusion
  flag + description-anchor gate, history-aware) + stage-2 hybrid entity
  scoring (embedding cosine + IDF-lexical, RRF). Embedding-deterministic
  (one pass). This is the prototype for the in-flight recall redesign.

## Files

- `__main__.py` — CLI entry point + suite registry.
- `_harness.py` — shared infra: AST prompt loading, Ollama client (chat +
  embed), the multi-turn `converse()` loop, per-call timing/token metrics,
  sampling + reporting.
- `fixtures.py` — synthetic messages and collections (scrubbed).
- `collection_lifecycle.py`, `skills_extractor.py`, `novel_patterns.py`,
  `retrieval.py` — the suites.

## What "good" looks like

The run prints a per-case pass-rate table and a `PERF` line (calls, avg
wall-clock, in/out tokens, generation tok/s), and exits non-zero if any
case failed. Model-driven cases are expected to be high but not always
100% — gpt-oss has run-to-run variance, which is exactly why each case is
sampled. When iterating a prompt, candidate text can live inline in a suite
until its pass-rate is solid, then graduate into source.
