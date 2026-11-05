"""The A/B replay resolves its target from the roster, and one call's arms agree (#3001).

The failure #3001 closes is an invalid comparison: arms measured on different models or
upstreams, or a re-render that does not match the bytes prod sent. So every test here
pins the thing that makes an A/B valid — one resolved target for every arm in a call,
and a refusal when the re-render is unfaithful.
"""

from __future__ import annotations

import json
import sqlite3
import zlib
from types import SimpleNamespace

import pytest

from penny.llm import raw_template
from penny.llm.models import PromptTransport
from penny.tests.eval.utils.replay import (
    ReplayVariant,
    _variants_from_spec,
    replay_variants,
)

_EVAL_MODELS = json.dumps(
    [
        {"model": "vendor/model-a", "provider": "SomeCloud"},
        {"model": "vendor/model-b", "provider": "OtherCloud"},
    ]
)

_MESSAGES = [
    {"role": "system", "content": "You are Penny."},
    {"role": "user", "content": "Run your cycle."},
]

# The pin, byte for byte, that every arm in one call must carry: the resolved
# roster entry's own hard pin, fallbacks off.
_PIN = {"order": ["SomeCloud"], "allow_fallbacks": False}


def _install_roster(monkeypatch, roster: str = _EVAL_MODELS) -> None:
    monkeypatch.setenv("EVAL_MODELS", roster)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def _write_row(tmp_path, transport: str, body: dict, messages=None) -> str:
    """A throwaway promptlog DB holding ONE row whose ``request`` column carries the
    zlib-compressed SentRequest, exactly what ``_load_request`` reads back."""
    db_path = str(tmp_path / "promptlog.db")
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE promptlog (id INTEGER PRIMARY KEY, messages TEXT, tools TEXT, "
            "agent_name TEXT, run_target TEXT, request BLOB)"
        )
        sent = {
            "transport": transport,
            "endpoint": "https://fake/v1",
            "model": "old-model",
            "body": body,
            "header_names": [],
        }
        connection.execute(
            "INSERT INTO promptlog (id, messages, tools, agent_name, run_target, request) "
            "VALUES (1, ?, NULL, 'chat', 'test', ?)",
            (
                json.dumps(messages if messages is not None else _MESSAGES),
                zlib.compress(json.dumps(sent).encode()),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return db_path


class _CapturingClient:
    """The SDK boundary stood in for: every ``create`` is recorded verbatim so a test
    can assert what each arm actually sent, and answers a plain reply with no calls."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.completions = SimpleNamespace(create=self._create)

    async def _create(self, **body: dict) -> SimpleNamespace:
        self.sent.append(body)
        # One response shape serves both transports: the chat path reads
        # ``choices[0].message``, the raw path reads ``choices[0].text``.
        choice = SimpleNamespace(
            message=SimpleNamespace(content="done", tool_calls=None, model_extra={}),
            text="done",
        )
        return SimpleNamespace(choices=[choice])


@pytest.fixture
def captures(monkeypatch) -> list[dict]:
    client = _CapturingClient()
    monkeypatch.setattr(
        "penny.tests.eval.utils.replay._client_for", lambda api_url, api_key: client
    )
    return client.sent


async def _run_two_arms(db_path: str, captures: list[dict]) -> None:
    await replay_variants(
        1,
        [ReplayVariant(name="baseline"), ReplayVariant(name="swapped", user_message="hello! ")],
        2,
        db_path=db_path,
        api_url="http://fake/v1",
        api_key="key",
    )
    assert len(captures) == 4  # 2 arms x 2 samples, every one through the same client


class TestEveryArmCarriesTheSameResolvedTarget:
    async def test_model_and_pin_identical_across_all_arms(
        self, tmp_path, monkeypatch, captures
    ) -> None:
        _install_roster(monkeypatch)
        body = {"model": "old-model", "messages": _MESSAGES}
        await _run_two_arms(_write_row(tmp_path, PromptTransport.CHAT.value, body), captures)
        assert {call["model"] for call in captures} == {"vendor/model-a"}
        assert {
            json.dumps(call["extra_body"]["provider"], sort_keys=True) for call in captures
        } == {json.dumps(_PIN, sort_keys=True)}

    async def test_unset_transforms_ride_as_recorded_and_a_swap_lands(
        self, tmp_path, monkeypatch, captures
    ) -> None:
        """The baseline arm's bytes are the stored ones; the swapped arm differs only
        where its transform says."""
        _install_roster(monkeypatch)
        body = {"model": "old-model", "messages": _MESSAGES}
        await _run_two_arms(_write_row(tmp_path, PromptTransport.CHAT.value, body), captures)
        baseline, swapped = captures[0], captures[2]
        assert baseline["messages"] == _MESSAGES
        assert swapped["messages"][-1]["content"] == "hello! "
        assert swapped["messages"][:-1] == _MESSAGES[:-1]

    async def test_a_thinking_toggle_swaps_the_same_keys_llmclient_composes(
        self, tmp_path, monkeypatch, captures
    ) -> None:
        _install_roster(monkeypatch)
        body = {"model": "old-model", "messages": _MESSAGES, "reasoning": {"enabled": True}}
        db_path = _write_row(tmp_path, PromptTransport.CHAT.value, body)
        await replay_variants(
            1,
            [ReplayVariant(name="off", thinking=False)],
            1,
            db_path=db_path,
            api_url="http://fake/v1",
            api_key="key",
        )
        sent = captures[0]
        assert "reasoning" not in sent  # the off form omits the key entirely, not False
        assert sent["reasoning_effort"] == "none"


class TestTheByteIdentityCheck:
    async def test_a_transformed_raw_row_re_renders_from_the_matching_state(
        self, tmp_path, monkeypatch, captures
    ) -> None:
        """A raw_template row whose stored prompt byte-matches today's render is
        retargeted and re-rendered from the transformed messages."""
        _install_roster(monkeypatch)
        prompt = raw_template.render_prompt(_MESSAGES, None, thinking_enabled=True)
        body = {"model": "old-model", "prompt": prompt}
        db_path = _write_row(tmp_path, PromptTransport.RAW_TEMPLATE.value, body)
        await replay_variants(
            1,
            [ReplayVariant(name="reworded", system_prompt="You are someone else.")],
            1,
            db_path=db_path,
            api_url="http://fake/v1",
            api_key="key",
        )
        sent = captures[0]
        expected = raw_template.render_prompt(
            [{"role": "system", "content": "You are someone else."}, _MESSAGES[1]],
            None,
            thinking_enabled=True,
        )
        assert sent["prompt"] == expected
        assert sent["model"] == "vendor/model-a"

    async def test_an_unfaithful_re_render_is_refused(
        self, tmp_path, monkeypatch, captures
    ) -> None:
        """Neither thinking state's re-render matches the stored prompt — the stored
        bytes came from something else (a continuation, an older template) — so an A/B
        on this row would compare against different bytes. Refused, nothing sent."""
        _install_roster(monkeypatch)
        body = {"model": "old-model", "prompt": "bytes today's template cannot produce"}
        db_path = _write_row(tmp_path, PromptTransport.RAW_TEMPLATE.value, body)
        with pytest.raises(ValueError, match="byte-match"):
            await replay_variants(
                1,
                [ReplayVariant(name="baseline")],
                1,
                db_path=db_path,
                api_url="http://fake/v1",
                api_key="key",
            )
        assert captures == []


class TestTheRefusalsBeforeAnythingIsSent:
    async def test_an_off_roster_model_refuses_the_whole_call(
        self, tmp_path, monkeypatch, captures
    ) -> None:
        _install_roster(monkeypatch)
        monkeypatch.setenv("LLM_MODEL", "vendor/model-c")
        db_path = _write_row(tmp_path, PromptTransport.CHAT.value, {"messages": _MESSAGES})
        with pytest.raises(Exception, match="vendor/model-c"):
            await replay_variants(
                1,
                [ReplayVariant(name="baseline")],
                1,
                db_path=db_path,
                api_url="http://fake/v1",
                api_key="key",
            )
        assert captures == []

    async def test_duplicate_and_unnamed_arms_are_refused(
        self, tmp_path, monkeypatch, captures
    ) -> None:
        _install_roster(monkeypatch)
        db_path = _write_row(tmp_path, PromptTransport.CHAT.value, {"messages": _MESSAGES})
        with pytest.raises(ValueError, match="duplicate"):
            await replay_variants(
                1,
                [ReplayVariant(name="a"), ReplayVariant(name="a")],
                1,
                db_path=db_path,
                api_url="http://fake/v1",
                api_key="key",
            )
        with pytest.raises(ValueError, match="at least one"):
            await replay_variants(
                1, [], 1, db_path=db_path, api_url="http://fake/v1", api_key="key"
            )
        assert captures == []


class TestTheSpecIsValidatedAsAWhole:
    def test_an_unknown_key_is_refused_rather_than_absorbed(self) -> None:
        with pytest.raises(SystemExit, match="system_pormpt"):
            _variants_from_spec(json.dumps([{"name": "a", "system_pormpt": "typo"}]))

    def test_a_valid_spec_parses_into_typed_arms(self) -> None:
        variants = _variants_from_spec(
            json.dumps(
                [
                    {"name": "baseline"},
                    {"name": "more", "system_prompt_append": "One more rule."},
                    {"name": "off", "thinking": False},
                ]
            )
        )
        assert [variant.name for variant in variants] == ["baseline", "more", "off"]
