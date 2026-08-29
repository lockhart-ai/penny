"""The configured model roster: required, validated, and refused loudly (#1996).

The failure this closes is not a crash — it is a run that measures less than it claims to
and says nothing. So every refusal here is checked for the thing that makes it useful: the
variable's name, and what to write in it.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess

import pytest

from penny.tests.eval.utils.roster import (
    EVAL_MODELS_ENV,
    MINIMUM_MODELS,
    EvalModel,
    RosterError,
    main,
    parse_roster,
    render_resolution,
    resolve,
)

_ROSTER = json.dumps(
    [
        {"model": "vendor/model-a", "provider": "SomeCloud"},
        {"model": "vendor/model-b", "provider": "OtherCloud"},
    ]
)


class TestTheRosterIsRequired:
    """Fail at startup, the way an unset embedding model does — never mid-run."""

    def test_an_unset_roster_names_the_variable_and_the_shape(self) -> None:
        with pytest.raises(RosterError) as refusal:
            parse_roster(None)
        message = str(refusal.value)
        assert EVAL_MODELS_ENV in message
        assert '{"model"' in message  # the shape to write, not just the complaint

    def test_a_blank_roster_reads_as_unset(self) -> None:
        with pytest.raises(RosterError):
            parse_roster("   ")

    def test_one_model_is_not_enough_and_the_refusal_says_why(self) -> None:
        """A suite tuned against one model measures how much like it the next one is."""
        with pytest.raises(RosterError) as refusal:
            parse_roster(json.dumps([{"model": "vendor/model-a", "provider": "SomeCloud"}]))
        message = str(refusal.value)
        assert f"at least {MINIMUM_MODELS} models" in message
        assert "vendor/model-a via SomeCloud" in message  # what IS configured, quoted back

    def test_a_malformed_roster_is_refused_rather_than_half_read(self) -> None:
        with pytest.raises(RosterError) as refusal:
            parse_roster('["vendor/model-a", "vendor/model-b"]')
        assert EVAL_MODELS_ENV in str(refusal.value)

    def test_an_unknown_key_is_refused_rather_than_absorbed(self) -> None:
        """A typo'd field would otherwise mean "no provider", silently."""
        with pytest.raises(RosterError):
            parse_roster(json.dumps([{"model": "a", "providers": "X"}, {"model": "b"}]))

    def test_a_well_formed_roster_parses_into_typed_entries(self) -> None:
        roster = parse_roster(_ROSTER)
        assert roster == [
            EvalModel(model="vendor/model-a", provider="SomeCloud"),
            EvalModel(model="vendor/model-b", provider="OtherCloud"),
        ]

    def test_an_entry_may_name_no_provider(self) -> None:
        """A direct endpoint has none to name, and that is a stated fact, not a gap."""
        roster = parse_roster(json.dumps([{"model": "a"}, {"model": "b"}]))
        assert roster[0].provider is None
        assert roster[0].render() == "a (no provider named)"


class TestWhichEntryThisInvocationRuns:
    def test_the_first_entry_is_the_default(self) -> None:
        assert resolve(parse_roster(_ROSTER), None).model == "vendor/model-a"

    def test_a_named_model_is_picked_out_of_the_roster_with_its_provider(self) -> None:
        entry = resolve(parse_roster(_ROSTER), "vendor/model-b")
        assert entry.provider == "OtherCloud"

    def test_a_model_outside_the_roster_is_refused_and_the_roster_is_listed(self) -> None:
        """An unconfigured model has no upstream to prefer and none to record.

        That is the ad-hoc pass this variable replaces, so it is refused — and the refusal
        renders the roster verbatim, so adding the model is a copy rather than a lookup.
        """
        with pytest.raises(RosterError) as refusal:
            resolve(parse_roster(_ROSTER), "vendor/model-c")
        message = str(refusal.value)
        assert "vendor/model-c" in message
        assert "vendor/model-a via SomeCloud" in message
        assert "vendor/model-b via OtherCloud" in message


class TestWhatTheRecipeReads:
    def test_the_resolution_renders_the_two_lines_the_makefile_parses(self) -> None:
        assert render_resolution(EvalModel(model="vendor/model-a", provider="SomeCloud")) == [
            "eval: model = vendor/model-a",
            "eval: preferred provider = SomeCloud",
        ]

    def test_a_providerless_entry_renders_no_provider_line(self) -> None:
        assert render_resolution(EvalModel(model="local-model")) == ["eval: model = local-model"]

    def test_the_cli_exit_code_is_what_stops_the_run(self, monkeypatch, capsys) -> None:
        monkeypatch.delenv(EVAL_MODELS_ENV, raising=False)
        assert main([]) == 1
        assert EVAL_MODELS_ENV in capsys.readouterr().err

        monkeypatch.setenv(EVAL_MODELS_ENV, _ROSTER)
        assert main(["vendor/model-b"]) == 0
        out = capsys.readouterr().out
        assert "eval: model = vendor/model-b" in out
        assert "eval: preferred provider = OtherCloud" in out

        # A stray argument is a usage error, distinct from an unconfigured roster.
        assert main(["a", "b"]) == 2


# ── What the Makefile actually hands this parser (#1997) ─────────────────────
#
# The roster is configuration, and configuration is read from the primary checkout's `.env` by
# the Makefile's own `from_env` helper.  That helper ended in `tr -d '"'` — written to unquote a
# scalar like `LLM_API_KEY="abc"`, and it stripped EVERY quote, so a JSON value arrived as
# `[{model:openai/gpt-oss-20b,...}]` and every `.env`-configured run refused to start.
#
# The bug was invisible from both ends.  Validating the FILE says it parses; validating the
# PARSER says it accepts JSON; neither exercises the transformation between them.  And an agent
# passing `EVAL_MODELS` through the environment takes the other branch of `$${EVAL_MODELS:-...}`
# and sails straight past it — so the only path that breaks is the one a human uses.
#
# The guard therefore runs the REAL helper, lifted out of the Makefile text rather than
# reimplemented here: a reimplementation is a second copy of the logic, and a second copy is
# what drifts silently back into the bug it was written to catch.


def _from_env_helper(env_file: pathlib.Path) -> str:
    """The `from_env` shell function, verbatim from the Makefile that defines it."""
    for parent in pathlib.Path(__file__).resolve().parents:
        makefile = parent / "Makefile"
        if not makefile.is_file():
            continue
        text = makefile.read_text()
        match = re.search(r"^\s*(from_env\(\) \{.*?\};)", text, re.M)
        if match is None:
            continue
        # Make's own escaping is the only thing undone: `$$` is how a recipe spells a shell `$`,
        # and the env path is a make variable.  The pipeline itself is carried over untouched.
        return match.group(1).replace("$$", "$").replace("$(EVAL_PRIMARY_ENV)", str(env_file))
    raise AssertionError("no Makefile defining from_env() above this test")


def _resolve_through_the_makefile(tmp_path: pathlib.Path, name: str, line: str) -> str:
    """What `from_env <name>` yields for a `.env` holding ``line`` — run in a real shell."""
    env_file = tmp_path / ".env"
    env_file.write_text(line + "\n")
    helper = _from_env_helper(env_file)
    done = subprocess.run(
        ["sh", "-c", f"{helper} from_env {name}"], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


class TestTheMakefileHandsTheRosterWhatIsConfigured:
    """A roster that parses in `.env` and not after the Makefile has read it is not configured."""

    def test_a_json_roster_survives_being_read_out_of_the_env_file(self, tmp_path) -> None:
        """The regression: `tr -d '"'` made this exact value unparseable, so the config path
        #1996 added was the one path that could not work."""
        resolved = _resolve_through_the_makefile(
            tmp_path, EVAL_MODELS_ENV, f"{EVAL_MODELS_ENV}={_ROSTER}"
        )
        assert resolved == _ROSTER, "the Makefile mangled the roster on its way to the parser"
        assert len(parse_roster(resolved)) == MINIMUM_MODELS, "and the parser still accepts it"

    def test_a_quoted_scalar_still_arrives_unquoted(self, tmp_path) -> None:
        """The paired over-correction guard: the helper exists to unquote a credential, and a
        fix that stopped doing that would trade this bug for an unusable API key."""
        assert _resolve_through_the_makefile(tmp_path, "LLM_API_KEY", 'LLM_API_KEY="abc"') == "abc"
        assert _resolve_through_the_makefile(tmp_path, "LLM_API_KEY", "LLM_API_KEY='abc'") == "abc"
        assert _resolve_through_the_makefile(tmp_path, "LLM_API_KEY", "LLM_API_KEY=abc") == "abc"
