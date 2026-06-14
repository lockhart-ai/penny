"""Fixtures for the live-model eval suite.

Construction reuses the integration-test isolation core (``running_penny``)
with a config whose model points at the real Ollama endpoint — no second
construction path, no stubs.  Each case samples N runs (the model is
stochastic) and asserts a pass-rate threshold against PERSISTED DB state, which
is the real contract.  See docs/self-improvement-loop.md.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import ceil

import pytest

from penny.config import Config
from penny.constants import ChannelType
from penny.database import Database
from penny.database.memory_store import EntryInput, Inclusion, RecallMode
from penny.database.models import Memory
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, run_penny_with_server
from penny.tests.eval.fixtures import SynthCollection
from penny.tests.mocks.signal_server import MockSignalServer

# Samples per case.  Override with EVAL_SAMPLES=2 for a quick smoke run.
SAMPLES = int(os.environ.get("EVAL_SAMPLES", "5"))

# Embedding backfill batch size for seeded memory.
_EMBED_BATCH = 100

# A scorer reads persisted DB state (+ the pre-run collection names) and returns
# a list of failure strings — empty means the sample passed.
Scorer = Callable[[Database, set[str]], list[str]]
Seeder = Callable[[Database], None]


@dataclass
class SampleResult:
    passed: bool
    fails: list[str]


def _real_model_config(
    make_config: Callable[..., Config], *, signal_api_url: str, db_path: str
) -> Config:
    """A test Config pointed at the real Ollama text + embedding models.

    Reads endpoint/model from the environment so the same suite runs on the
    host (localhost) and inside the penny container (host.docker.internal),
    falling back to local defaults.  ``signal_api_url`` binds to the sample's
    own mock server so samples never share a channel.
    """
    return make_config(
        signal_api_url=signal_api_url,
        llm_model=os.environ.get("LLM_MODEL", "gpt-oss:20b"),
        llm_api_url=os.environ.get("LLM_API_URL", "http://localhost:11434"),
        llm_embedding_model=os.environ.get("LLM_EMBEDDING_MODEL", "embeddinggemma"),
        db_path=db_path,
    )


def seed_user(db: Database) -> None:
    """Create the test user + register their Signal device.

    Each sample uses a fresh DB, so the ``test_user_info`` fixture (bound to one
    path) doesn't apply — seed the user explicitly after Penny builds the DB.
    """
    db.users.save_info(
        sender=TEST_SENDER,
        name="Test User",
        location="Seattle, WA",
        timezone="America/Los_Angeles",
        date_of_birth="1990-01-01",
    )
    db.devices.register(ChannelType.SIGNAL, TEST_SENDER, "Test Signal", is_default=True)


def seed_collection(
    db: Database,
    synth: SynthCollection,
    *,
    extraction_prompt: str | None = None,
    intent: str | None = None,
    interval: int | None = None,
) -> None:
    """Create a synthetic collection + its entries (key = text before ' — ')."""
    db.memories.create_collection(
        synth.name,
        synth.description,
        Inclusion(synth.inclusion),
        RecallMode.RELEVANT,
        extraction_prompt=extraction_prompt,
        collector_interval_seconds=interval,
        intent=intent,
    )
    db.memories.write(
        synth.name,
        [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in synth.entries],
        author="user",
    )


def collection_names(db: Database) -> set[str]:
    """Every memory name currently in the DB — the pre-run snapshot for scorers."""
    return {memory.name for memory in db.memories.list_all()}


def new_collections(db: Database, before: set[str]) -> list[Memory]:
    """Collections that didn't exist before the run — what the model created."""
    return [memory for memory in db.memories.list_all() if memory.name not in before]


async def _embed_seeds(penny: Penny) -> None:
    """Vectorize seeded memory so stage-1/2 recall behaves like prod.

    Penny's startup backfill ran on the empty DB before we seeded; re-run it so
    seeded descriptions/entries get embeddings the recall path can match.
    """
    if penny.embedding_model_client is None:
        return
    await penny._backfill_memory_embeddings(_EMBED_BATCH)
    await penny._backfill_description_embeddings(_EMBED_BATCH)


def _assert_threshold(case_id: str, results: list[SampleResult], min_pass_rate: float) -> None:
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    need = ceil(min_pass_rate * total)
    if passed < need:
        failures = "\n".join(
            f"  [{i + 1}] {'; '.join(result.fails)}"
            for i, result in enumerate(results)
            if not result.passed
        )
        pytest.fail(
            f"{case_id}: {passed}/{total} passed (need >={need}, rate {min_pass_rate}):\n{failures}"
        )


# A chat-eval runner: (case_id, message, scorer, optional seeder) -> asserts threshold.
ChatEval = Callable[..., Awaitable[None]]


@pytest.fixture
def chat_eval(make_config: Callable[..., Config], tmp_path) -> ChatEval:
    """Drive the real chat flow N times for one user message and score each run.

    Each sample is fully hermetic — its own mock Signal server, DB, and
    real-model Penny: seed user (+ any case seed), embed the seeds, push the
    message, wait for the reply, then score persisted state.  A per-sample
    server is essential: a shared one leaks a prior sample's shut-down channel,
    which then errors on the next sample's broadcast.  A timeout counts as a
    failed sample, not a crash.
    """

    async def _run(
        *,
        case_id: str,
        message: str,
        score: Scorer,
        seed: Seeder | None = None,
        samples: int = SAMPLES,
        min_pass_rate: float = 0.75,
        timeout: float = 120.0,
    ) -> None:
        results: list[SampleResult] = []
        for sample_index in range(samples):
            server = MockSignalServer()
            await server.start()
            try:
                config = _real_model_config(
                    make_config,
                    signal_api_url=f"http://localhost:{server.port}",
                    db_path=str(tmp_path / f"{case_id}-{sample_index}.db"),
                )
                async with run_penny_with_server(config, server) as penny:
                    seed_user(penny.db)
                    if seed is not None:
                        seed(penny.db)
                    await _embed_seeds(penny)
                    before = collection_names(penny.db)
                    try:
                        await server.push_message(sender=TEST_SENDER, content=message)
                        await server.wait_for_message(timeout=timeout)
                    except TimeoutError:
                        results.append(SampleResult(False, ["no reply within timeout"]))
                        continue
                    fails = score(penny.db, before)
                    results.append(SampleResult(not fails, fails))
            finally:
                await server.stop()
        _assert_threshold(case_id, results, min_pass_rate)

    return _run
