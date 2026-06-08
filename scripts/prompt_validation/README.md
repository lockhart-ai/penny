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

From the repo root:

```bash
# Authoring: does gpt-oss write content-reflective descriptions + the right
# inclusion flag + correct silent/notify handling in collection_create?
PYTHONPATH=. uv run --python 3.12 --with openai \
    python scripts/prompt_validation/lifecycle.py

# Retrieval: two-stage routing (collection inclusion gate + hybrid top-N
# entity scoring) on synthetic message fixtures.
PYTHONPATH=. uv run --python 3.12 --with openai \
    python scripts/prompt_validation/retrieval.py
```

Env overrides: `LLM_MODEL` (default `gpt-oss:20b`), `LLM_EMBEDDING_MODEL`
(default `embeddinggemma`), `LLM_API_URL` (default `http://localhost:11434`),
`N_SAMPLES` (default 5).

## Files

- `_harness.py` — shared: AST prompt loading, Ollama client (chat + embed),
  per-call timing/token metrics, sampling + reporting.
- `fixtures.py` — synthetic messages and collections (scrubbed).
- `lifecycle.py` — collection-authoring dry-run (candidate tool/skill text
  inline; graduates to source once the pass-rate is solid).
- `retrieval.py` — two-stage recall routing + hybrid (cosine + IDF-lexical,
  RRF) entity scoring prototype.

## What "good" looks like

Each runner prints a per-case pass-rate table and a `PERF` line (calls, avg
wall-clock, in/out tokens, generation tok/s). Authoring should be ~20/20;
retrieval reports skill-in-top-N and stage-1 routing accuracy against the
fixtures.
