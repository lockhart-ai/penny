# Model precision — reference

Covers which gemma build prod runs and why, the measured effect of weight precision on
Penny's behaviour, and how to repeat the measurement. Read this before changing prod's
model build, before trusting the eval suite to predict a prod failure, or before diagnosing
a prod breakdown the suite doesn't reproduce.

## 1. Which build prod runs, and why

Prod runs gemma (`google/gemma-4-26b-a4b-it`) on oMLX, at **8-bit** — MLX 8-bit, model id
`gemma-4-26b-a4b-it-8bit` — not a 4-bit build.

The eval suite runs the same model at bf16 on OpenRouter, pinned to CoreWeave. The suite's
run only predicts prod when prod's build behaves like the suite's provider. 8-bit does; 4-bit
does not (below).

**Standing rule**: prod runs a build whose behaviour the eval suite's provider matches. When
prod misbehaves and the suite doesn't reproduce it, check prod's precision before anything
else.

## 2. The finding

Most of the failures where Penny meant to act and then fell apart come from 4-bit weights —
not from which harness or runner served them. It doesn't matter whether the 4-bit build is
QAT or regular, or whether it runs on oMLX or Ollama.

Each cell is a passing count: how many of the samples did **not** repeat the failure.

| Failure class | QAT 4-bit (oMLX) | Regular 4-bit MLX (oMLX) | nvfp4 (Ollama) | 8-bit MLX (oMLX) | bf16 (CoreWeave) |
|---|---|---|---|---|---|
| Re-fetches pages it already has, or doubts the link numbers | 13/30 | 27/30 | 24/30 | 28/30 | 30/30 |
| Announces a lookup, or says it checked, then ends the turn with no call | 83/206 | 110/210 | 116/205 | 160/208 | 164/210 |
| Ends a collector step with neither a call nor text | 3/30 | 6/30 | 5/27 | 23/30 | 22/30 |
| Ends a turn with no call where one was due (short turn) | 24/30 | 19/30 | 24/28 | 30/30 | 30/30 |
| Writes a collection entry with no source link | 13/88 | 6/90 | 17/90 | 26/90 | 38/90 |
| Repeats a near-identical browse query | 18/27 | 12/22 | 7/19 | 7/25 | 10/26 |
| Writes collection rules into the job prompt (an old tool presentation, since fixed) | 0/30 | 1/30 | 1/30 | 0/30 | 0/30 |

Notes:

- Regular 4-bit fails like QAT 4-bit on most classes. The link-number doubt row is the
  exception: there, the failure is mostly QAT's.
- 8-bit tracks bf16.
- The repeated-query class runs backwards from the others: the 4-bit builds stall out of the
  turn before they've browsed enough to repeat a query.
- Two regular 4-bit samples ran to the 32,768-token cap without finishing; across 450 8-bit
  samples, none did.
- Failures that remain at every precision (some "says it checked" stalls, linkless entries,
  repeated queries) are Penny's to fix — any runner, at any precision, shows them.
- N excludes samples that errored at the transport.

## 3. Cost of 8-bit

Decode tokens/second, median of warm samples, one request at a time, only the measured model
loaded, nothing swapping or sleeping (see the gotchas below):

| Prompt | QAT 4-bit | Regular 4-bit | 8-bit |
|---|---|---|---|
| Short chat turn (~11k tokens in) | 48.6 | 50.4 | 41.4 |
| Long chat turn (~53k tokens in) | 27.1 | 27.7 | 24.4 |
| Collector step (~13k tokens in) | 48.2 | 49.4 | 41.1 |

8-bit decodes about 15% slower — it reads twice the weight bytes per token. Cold prefill of
the 53k-token prompt runs about 235 s on both QAT and 8-bit, and prefix caching behaves the
same at both. Weights take 26 GB against 15 GB, on a 64 GB M1 Max.

## 4. Method (how to repeat it)

Replay the exact stored prompt of a failing prod turn, N times, against each runner/build
being compared, and apply one deterministic classifier per failure class identically to every
runner's output. 30 samples per row.

**Replay tool**: `penny/penny/tests/eval/utils/replay.py`, run against a row's promptlog id
(`--id`). Its live flags (`--help`, or read the module):

- `--mode plain` replays the row's messages/tools verbatim, re-rendered through today's code
  against whatever `LLM_API_URL`/`LLM_MODEL` is set to — auto-selected for a non-collector
  (chat) prompt.
- `--mode verbatim-request`, for a row whose `request` column was stored (#2875), sends that
  row's own exact wire request unchanged — no re-render at all — overriding only `--model`
  and `--provider` to retarget a different runner than the one that originally served it.
  This is the mode that proves the bytes match rather than approximating them.
- `--samples N` sets the replay count (default 12; use 30 for this measurement).
- `--db` points at the SQLite DB holding the `promptlog` row.

Comparing N runners means running the tool once per runner, pointed at that runner's
endpoint (`LLM_API_URL`/`LLM_MODEL`, and `LLM_API_KEY`/`LLM_PROVIDER` for a remote one —
read the same way `make eval-remote` reads them). Five more flags (#2884):

- `--concurrency N` draws N samples in parallel against the runner instead of one at a
  time (default 1 — serial); size it to what the endpoint actually sustains.
- `--jsonl PATH` writes one full JSON record per sample (its tool calls with arguments,
  the FULL thinking, the FULL reply, this sample's own timing, any error) as each sample
  finishes, so a classifier can read every finished sample without re-running them even if
  a long run is killed partway through.
- `--metrics-db PATH` (optional — a fresh tempfile by default) is the throwaway, writable
  DB each sample's own prefill/decode timing is logged to and read back from, so comparing
  runners' speed needs no separate benchmark run — it rides the same replay.
- `--mode ollama-native` sends the SAME rendered raw-template prompt through Ollama's
  native `/api/generate` (`raw: true`) instead of the OpenAI-compatible `/v1/completions`
  every other raw-template column here uses, for a deployment where the latter chat-wraps
  the prompt instead of serving it faithfully — a raw-faithful replay of an Ollama row
  needs no script outside this tool.
- `--dump-thinking` prints each sample's FULL thinking to stdout, not just the short tail
  every mode prints by default — the `--jsonl` record always carries the full thinking
  regardless of this flag.

A runner is only comparable if it is **raw-template faithful** — it serves gemma's own
vendored chat template byte-for-byte, rather than wrapping the prompt in a template of its
own or stripping gemma's control tokens. Two checks:

- a bare `<bos>` prompt (no other content) reads back with `prompt_tokens == 1` on the raw
  completions endpoint. This is the same check Penny's own startup preflight runs
  (`LlmClient.check_raw_template_faithfulness` in `penny/penny/llm/client.py`) before trusting
  the raw transport against a given endpoint.
- gemma's own control markers (`<|tool_call>`, `<|channel>`) survive intact in the model's
  output rather than being parsed away or never appearing.

Ollama's OpenAI-compatible `/v1/completions` wraps the prompt in a chat template of its own,
so it fails the first check and is not faithful. Ollama's native `/api/generate`, with
`raw: true`, sends the prompt exactly as given and is faithful — this is the endpoint the
nvfp4 row above was measured through, not `replay.py`'s OpenAI-compatible transport.

On OpenRouter, CoreWeave serves gemma raw-template faithfully. Several other gemma providers
on OpenRouter wrap the prompt in their own chat template, or strip gemma's control markers,
and are not faithful.

## 5. Measurement gotchas

- **Host sleep.** The Mac idle-sleeps when nobody is using it, waking only briefly for
  maintenance. Generation stalls mid-run when this happens, and wall time and tok/s readings
  from that window are meaningless. Hold it awake with `caffeinate -i -s` for the run, and
  check `pmset -g log` for `Sleep`/`DarkWake` entries inside the measurement window before
  trusting a number. The same sleep stalls prod Penny on that host — it is not only a
  measurement artifact.
- **Swap.** macOS's reported swap "used" total stays high long after memory pressure has
  actually cleared, so it alone doesn't say whether a run is swapping live. Gate on the
  `vm_stat` Swapins/Swapouts counters staying flat over 60 s, with one model loaded, before
  timing anything.
- **One model loaded.** Unload every other model through oMLX's admin API before timing a
  build — a second model resident, even idle, changes memory pressure and decode speed.
