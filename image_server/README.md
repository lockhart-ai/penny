# image_server

A small local HTTP server that generates images with [mflux](https://github.com/filipstrand/mflux)'s
FLUX.2 [klein] model. It runs on the host, **outside Docker** — it needs Metal, which a Linux
container doesn't have — and is the backend Penny's `generate_image` chat tool talks to (#2792).

## Why a separate process

Penny runs in Docker; mflux needs Apple's Metal/MLX, which isn't available inside a Linux
container. So this server runs directly on the host, under its own interpreter, and Penny
reaches it over HTTP at `host.docker.internal:8765` (the default `LLM_IMAGE_API_URL`).

## Setup

1. Install mflux as a uv tool (if you haven't already):
   ```
   uv tool install mflux
   ```
   Pillow must stay below version 12 in that tool's environment — a newer Pillow breaks
   mflux's import.
2. The server itself needs **no extra dependencies** beyond mflux (it's stdlib-only Python) —
   run it with the same interpreter:
   ```
   ~/.local/share/uv/tools/mflux/bin/python image_server/server.py
   ```
   or `make image-server-run` (reads `MFLUX_PYTHON`, overridable if your tool lives elsewhere).
3. On startup, the server downloads and caches the model weights (~8 GB, first run only) and
   loads the model into memory (~15s) **before it starts accepting connections** — so a
   reachable `/health` always means the model is ready, never "still loading". It stays
   resident for the server's whole lifetime (the machine has memory to spare, so there's no
   idle-unload to reason about); every `/generate` after that just denoises (~12s at 512px,
   ~35s at 1024px on an M1 Max, 4 steps).

## Running it as a launchd agent

`make image-server-plist` renders `com.penny.imageserver.plist.template` with this checkout's
absolute paths and prints the result — it does not install anything. To actually run it as a
background agent:

```
make image-server-plist > /tmp/com.penny.imageserver.plist
cp /tmp/com.penny.imageserver.plist ~/Library/LaunchAgents/com.penny.imageserver.plist
launchctl load ~/Library/LaunchAgents/com.penny.imageserver.plist
```

Logs go to `data/penny/logs/image_server.log` (the server's own log) and
`data/penny/logs/image_server.{out,err}.log` (launchd's stdout/stderr capture).

## Configuration

Environment variables read by `server.py` (set them in the launchd plist's
`EnvironmentVariables`, or export them before a manual run):

| Variable | Default | Meaning |
|---|---|---|
| `IMAGE_SERVER_MODEL` | `flux2-klein-4b` | The model id the server serves; a request naming a different id gets a 400 |
| `IMAGE_SERVER_PORT` | `8765` | Port to bind, on `127.0.0.1` only |
| `IMAGE_SERVER_QUANTIZE` | `4` | mflux quantization bits |

Penny-side configuration (root `CLAUDE.md`'s `.env` table): `LLM_IMAGE_MODEL` (must match
`IMAGE_SERVER_MODEL`), `LLM_IMAGE_API_URL` (default `http://host.docker.internal:8765`),
`IMAGE_WIDTH`/`IMAGE_HEIGHT`/`IMAGE_STEPS` (default `768`/`768`/`4`).

## API

- `GET /health` → `{"status": "ok", "served_model": "flux2-klein-4b", "model_loaded": true}` — always `true`: the model loads before the server binds, so a reachable `/health` is always ready
- `POST /generate` with `{"model": "flux2-klein-4b", "prompt": "...", "width": 768, "height": 768, "steps": 4}`
  (`seed` optional) → `{"image": "<base64 PNG>"}`, or a non-2xx with `{"error": "<reason>"}` —
  `400` for a bad/unsupported request, `503` if a generation is already in flight (one at a
  time; the client retries this, bounded), `500` for a generation failure.

## Tests

`image_server/test_server.py` drives the request/response logic against a fake generator (no
mflux, no Metal needed) — `server.py` imports mflux lazily, only inside `MfluxGenerator.load`,
so it stays importable and testable without it. These tests
run as part of `make check`'s pytest step (see `PYTEST_ARGS` in the root `Makefile` and the
`./image_server:/image_server:ro` mount in `docker-compose.override.yml`) — the same gate as
everything else, even though mflux itself is never installed there.
