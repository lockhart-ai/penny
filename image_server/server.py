"""A small local HTTP server that generates images with mflux's FLUX.2 [klein].

Runs OUTSIDE Docker, under the mflux uv tool's own interpreter
(``~/.local/share/uv/tools/mflux/bin/python``) — it needs Metal, which a Linux
container doesn't have. Penny's chat agent (in Docker) talks to it over HTTP via
``penny.llm.image_client.MfluxImageClient`` at ``host.docker.internal``.

Deliberately stdlib-only (``http.server``, ``json``, ``threading`` — no Pydantic,
no FastAPI/uvicorn): the mflux uv tool environment is minimal and this script
should not need extra packages installed into it just to run as a launchd agent.
mflux itself is imported LAZILY, inside ``MfluxGenerator.load`` — never at module
scope — so this module (and its request/response logic) is importable and
unit-testable from Penny's own Docker test environment, which has no mflux and no
Metal, via a fake ``ImageGenerator``.

One generation at a time: a ``threading.Lock`` serialises every call into the
generator, and a second request arriving while one is in flight gets a plain
``503`` naming that rather than queuing or blocking — the client (`MfluxImageClient`)
is the one that waits, with a bounded, configured retry.

The model loads ONCE, synchronously, in ``main()`` before the server starts
accepting connections — the machine has memory to spare, so klein just stays
resident for the server's whole lifetime; there's no idle-unload to reason about.
Because loading happens before the socket binds, a reachable ``/health`` always
means the model is ready (the request simply couldn't arrive any earlier).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol

logger = logging.getLogger("image_server")

DEFAULT_MODEL = "flux2-klein-4b"
DEFAULT_PORT = 8765
DEFAULT_QUANTIZE = 4

# MLX keeps freed buffers around for reuse instead of returning them to the OS. Left
# uncapped, that cache grows with every generation's working set and never shrinks —
# measured at 25.3 GB resident after 4½ days (model weights are a few GB at
# quantize=4). Capping it bounds that growth; clearing it after each request (see
# ``MfluxGenerator.release``) returns what's cached beyond the model itself.
MLX_CACHE_LIMIT_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB


class ImageGenerationError(RuntimeError):
    """A generation attempt failed — its message is the reason the response carries."""


class ImageGenerator(Protocol):
    """What the server needs from a model backend — real (mflux) or faked (tests)."""

    def load(self) -> None:
        """Load the model. Called once, synchronously, before the server starts."""

    @property
    def is_loaded(self) -> bool:
        """Whether ``load()`` has completed."""

    def generate(self, *, prompt: str, width: int, height: int, steps: int, seed: int) -> bytes:
        """Generate one image and return it as PNG bytes. Raises ImageGenerationError."""

    def release(self) -> None:
        """Release resources held by the last generation. Called after every generate()
        attempt, success or failure."""


class MfluxGenerator:
    """The real backend: mflux's FLUX.2 [klein], loaded once and kept resident."""

    def __init__(self, model_name: str = DEFAULT_MODEL, quantize: int = DEFAULT_QUANTIZE) -> None:
        self._model_name = model_name
        self._quantize = quantize
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        # Imported here, not at module scope (see the module docstring): this is the
        # only place mflux needs to exist.
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux2.variants import Flux2Klein

        logger.info(
            "Loading mflux model %r (quantize=%d), MLX cache capped at %d bytes...",
            self._model_name,
            self._quantize,
            MLX_CACHE_LIMIT_BYTES,
        )
        self._model = Flux2Klein(
            model_config=ModelConfig.from_name(model_name=self._model_name),
            quantize=self._quantize,
        )
        # Imported here too (see the module docstring) — set once, right after load,
        # before the first generate() can grow the cache.
        import mlx.core as mx

        mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
        logger.info("Model %r loaded", self._model_name)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(self, *, prompt: str, width: int, height: int, steps: int, seed: int) -> bytes:
        assert self._model is not None, "generate() called before load()"
        try:
            generated = self._model.generate_image(
                seed=seed,
                prompt=prompt,
                width=width,
                height=height,
                guidance=1.0,
                num_inference_steps=steps,
                scheduler="flow_match_euler_discrete",
            )
        except Exception as error:  # mflux raises various exceptions; name them honestly
            raise ImageGenerationError(str(error)) from error
        buffer = io.BytesIO()
        generated.image.save(buffer, format="PNG")
        return buffer.getvalue()

    def release(self) -> None:
        # Imported here, not at module scope (see the module docstring).
        import mlx.core as mx

        mx.clear_cache()


class ImageServer(ThreadingHTTPServer):
    """Holds the (already-loaded) generator, the served model id, and the one-at-a-time lock."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        *,
        generator: ImageGenerator,
        served_model: str,
    ) -> None:
        super().__init__(address, handler_cls)
        self.generator = generator
        self.served_model = served_model
        self.lock = threading.Lock()


class ImageRequestHandler(BaseHTTPRequestHandler):
    """``GET /health``, ``POST /generate`` — everything else 404s."""

    server: ImageServer  # narrows the stdlib's untyped `self.server`

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        logger.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._respond(404, {"error": f"unknown path {self.path!r}"})
            return
        self._respond(
            200,
            {
                "status": "ok",
                "served_model": self.server.served_model,
                "model_loaded": self.server.generator.is_loaded,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._respond(404, {"error": f"unknown path {self.path!r}"})
            return
        try:
            body = self._read_json_body()
            request = _parse_generate_request(body, served_model=self.server.served_model)
        except ValueError as error:
            self._respond(400, {"error": str(error)})
            return
        self._handle_generate(request)

    def _handle_generate(self, request: _GenerateRequest) -> None:
        if not self.server.lock.acquire(blocking=False):
            self._respond(503, {"error": "busy, a generation is already in progress"})
            return
        try:
            self._generate_and_respond(request)
        finally:
            # Every attempt, success or failure, releases the backend's GPU cache
            # before the lock does — see MfluxGenerator.release.
            self.server.generator.release()
            self.server.lock.release()

    def _generate_and_respond(self, request: _GenerateRequest) -> None:
        try:
            png_bytes = self.server.generator.generate(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                seed=request.seed,
            )
        except ImageGenerationError as error:
            self._respond(500, {"error": str(error)})
            return
        self._respond(200, {"image": base64.b64encode(png_bytes).decode("ascii")})

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("empty request body")
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed JSON body: {error}") from error
        if not isinstance(parsed, dict):
            raise ValueError("request body must be a JSON object")
        return parsed

    def _respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _GenerateRequest:
    """A validated ``/generate`` request — built only by ``_parse_generate_request``."""

    def __init__(self, *, prompt: str, width: int, height: int, steps: int, seed: int) -> None:
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.seed = seed


def _parse_generate_request(body: dict, *, served_model: str) -> _GenerateRequest:
    """Validate a ``/generate`` body, raising ``ValueError`` (→ 400) on anything wrong."""
    model = body.get("model")
    if model != served_model:
        raise ValueError(f"unsupported model {model!r} — this server serves {served_model!r}")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' is required and must be a non-empty string")
    try:
        return _GenerateRequest(
            prompt=prompt,
            width=int(body.get("width", 768)),
            height=int(body.get("height", 768)),
            steps=int(body.get("steps", 4)),
            seed=int(body.get("seed", int.from_bytes(os.urandom(4), "big"))),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"'width'/'height'/'steps'/'seed' must be integers: {error}") from error


def _configure_logging() -> None:
    """Log to data/penny/logs/image_server.log, a sibling of the rest of Penny's logs."""
    log_dir = Path(__file__).resolve().parent.parent / "data" / "penny" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_dir / "image_server.log"), logging.StreamHandler()],
    )


def main() -> None:
    _configure_logging()
    model_name = os.environ.get("IMAGE_SERVER_MODEL", DEFAULT_MODEL)
    port = int(os.environ.get("IMAGE_SERVER_PORT", str(DEFAULT_PORT)))
    quantize = int(os.environ.get("IMAGE_SERVER_QUANTIZE", str(DEFAULT_QUANTIZE)))

    generator = MfluxGenerator(model_name=model_name, quantize=quantize)
    generator.load()  # synchronous, BEFORE binding — a reachable /health is always ready
    server = ImageServer(
        ("127.0.0.1", port), ImageRequestHandler, generator=generator, served_model=model_name
    )
    logger.info(
        "image_server listening on 127.0.0.1:%d, serving %r (resident, no idle-unload)",
        port,
        model_name,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down (keyboard interrupt)")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
