"""Request/response tests for ``image_server.server`` with mflux faked out.

Runs under Penny's own test environment (`make check`'s pytest run — see
``PYTEST_ARGS`` in the root Makefile) even though mflux is never installed there:
``server.py`` imports mflux lazily inside ``MfluxGenerator.load``, so these tests
drive the HTTP/threading/locking logic against a fake ``ImageGenerator`` and
never touch that import. Stdlib-only (``urllib``), to mirror the server's own
no-extra-deps stance.

No unbounded waits anywhere here: every server thread started by a test is a
daemon and is ``shutdown()`` + ``server_close()`` + ``join(timeout=...)``'d in
teardown. (A real hang was found and fixed here (#2792): ``shutdown()`` on a
``socketserver`` whose ``serve_forever()`` was never actually running blocks
forever, because the internal "I'm shut down" event is only ever set from inside
``serve_forever()``'s own loop — so every test that constructs a real
``ImageServer`` must also run it.)
"""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from image_server.server import ImageGenerationError, ImageRequestHandler, ImageServer

_SERVED_MODEL = "flux2-klein-4b"
_ONE_PX_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01"
    b"\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeGenerator:
    """An ``ImageGenerator`` the tests control directly — no mflux, no Metal."""

    def __init__(self) -> None:
        self.loaded = False
        self.calls: list[dict] = []
        self.release_calls = 0
        self.fail_with: str | None = None

    def load(self) -> None:
        self.loaded = True

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def generate(self, *, prompt: str, width: int, height: int, steps: int, seed: int) -> bytes:
        assert self.loaded, "generate() called before load()"
        self.calls.append(
            {"prompt": prompt, "width": width, "height": height, "steps": steps, "seed": seed}
        )
        if self.fail_with is not None:
            raise ImageGenerationError(self.fail_with)
        return _ONE_PX_PNG

    def release(self) -> None:
        self.release_calls += 1


class _Response:
    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self.payload = payload


def _request(base_url: str, path: str, *, body: dict | None = None) -> _Response:
    url = f"{base_url}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return _Response(resp.status, json.loads(resp.read()))
    except urllib.error.HTTPError as error:
        return _Response(error.code, json.loads(error.read()))


@pytest.fixture
def generator() -> FakeGenerator:
    """Loaded up front, mirroring ``main()``: the model is ready before any server exists."""
    gen = FakeGenerator()
    gen.load()
    return gen


@pytest.fixture
def running_server(generator: FakeGenerator) -> Iterator[tuple[ImageServer, str]]:
    server = ImageServer(
        ("127.0.0.1", 0), ImageRequestHandler, generator=generator, served_model=_SERVED_MODEL
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "server thread did not stop within 5s"


def test_health_reports_ready(running_server: tuple[ImageServer, str]) -> None:
    """Loading happens before the socket ever binds, so a reachable /health is always ready —
    there is no "not loaded yet" state a request could observe."""
    _, base_url = running_server
    response = _request(base_url, "/health")
    assert response.status == 200
    assert response.payload == {
        "status": "ok",
        "served_model": _SERVED_MODEL,
        "model_loaded": True,
    }


def test_generate_success_returns_base64_png(
    running_server: tuple[ImageServer, str], generator: FakeGenerator
) -> None:
    _, base_url = running_server
    response = _request(
        base_url, "/generate", body={"model": _SERVED_MODEL, "prompt": "a teal dragon"}
    )
    assert response.status == 200
    assert base64.b64decode(response.payload["image"]) == _ONE_PX_PNG
    assert len(generator.calls) == 1
    call = generator.calls[0]
    assert call["prompt"] == "a teal dragon"
    assert (call["width"], call["height"], call["steps"]) == (768, 768, 4)
    assert isinstance(call["seed"], int)
    assert generator.release_calls == 1, "release() must run after a successful generation"


def test_generate_failure_releases_cache(
    running_server: tuple[ImageServer, str], generator: FakeGenerator
) -> None:
    generator.fail_with = "boom"
    _, base_url = running_server
    response = _request(
        base_url, "/generate", body={"model": _SERVED_MODEL, "prompt": "a cat"}
    )
    assert response.status == 500
    assert "boom" in response.payload["error"]
    assert generator.release_calls == 1, "release() must run after a failed generation too"


def test_generate_rejects_an_unsupported_model(running_server: tuple[ImageServer, str]) -> None:
    _, base_url = running_server
    response = _request(
        base_url, "/generate", body={"model": "some-other-model", "prompt": "a cat"}
    )
    assert response.status == 400
    assert _SERVED_MODEL in response.payload["error"]
    assert "some-other-model" in response.payload["error"]


def test_generate_rejects_a_missing_prompt(running_server: tuple[ImageServer, str]) -> None:
    _, base_url = running_server
    response = _request(base_url, "/generate", body={"model": _SERVED_MODEL})
    assert response.status == 400
    assert "prompt" in response.payload["error"]


def test_generate_while_busy_returns_503(running_server: tuple[ImageServer, str]) -> None:
    server, base_url = running_server
    server.lock.acquire()  # simulate a generation already in flight
    try:
        response = _request(
            base_url, "/generate", body={"model": _SERVED_MODEL, "prompt": "a cat"}
        )
    finally:
        server.lock.release()
    assert response.status == 503
    assert "busy" in response.payload["error"]
