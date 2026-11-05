"""The eval config runs on the roster's model and pin, refusing everything else (#3001).

``_real_model_config`` is the one seam every live-model sample builds its config
through, so the roster resolution lives there: an eval can no more run on a model or
upstream `EVAL_MODELS` does not define than a replay can.
"""

from __future__ import annotations

import json

import pytest

from penny.tests.eval.conftest import _real_model_config
from penny.tests.eval.utils.roster import RosterError

_EVAL_MODELS = json.dumps([{"model": "vendor/model-a", "provider": "SomeCloud"}])


def _capture_config(**kwargs):
    """The ``make_config`` seam stood in for: what the resolution decided is what the
    config carries, and nothing more is needed to see it."""
    return kwargs


def _resolve_config(monkeypatch, **env):
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    return _real_model_config(
        _capture_config, signal_api_url="http://localhost:1", db_path="sample.db"
    )


class TestTheEvalConfigRunsOnTheRosterOnly:
    def test_a_rosterless_eval_refuses(self, monkeypatch) -> None:
        """The local-Ollama-without-a-roster path is gone: an eval names its roster."""
        with pytest.raises(RosterError):
            _resolve_config(monkeypatch, EVAL_MODELS=None)

    def test_the_rosters_first_entry_and_its_pin_are_what_the_config_runs(
        self, monkeypatch
    ) -> None:
        config = _resolve_config(
            monkeypatch, EVAL_MODELS=_EVAL_MODELS, LLM_MODEL=None, LLM_PROVIDER=None
        )
        assert config["llm_model"] == "vendor/model-a"
        assert config["llm_provider"] == "SomeCloud"

    def test_an_exported_provider_mismatching_the_entry_refuses(self, monkeypatch) -> None:
        """The same up-front refusal the replay helpers give a stale hand-set pin."""
        with pytest.raises(RosterError, match="StaleDocs"):
            _resolve_config(
                monkeypatch,
                EVAL_MODELS=_EVAL_MODELS,
                LLM_MODEL=None,
                LLM_PROVIDER="StaleDocs",
            )
