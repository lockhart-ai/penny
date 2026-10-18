> **Archived** — removed from the live tree in `5c38d08c` (2026-02-08, #84 "Restructure penny-team to mirror penny layout, split CLAUDE.md"), dropped as stale once that PR split its content into `penny-team/CLAUDE.md`. Replaced by Glade-task kittens (`docs/agent-task-workflow.md`), after `penny-team/CLAUDE.md` itself was later removed in #1598.

---

# Penny Agent Orchestrator

Python-based orchestrator that manages autonomous Claude CLI agents. Each agent runs on a schedule, processing work via GitHub Issues.

## Quick Start

```bash
uv run --python 3.12 penny-team/orchestrator.py              # Run continuously
uv run --python 3.12 penny-team/orchestrator.py --once       # Run all due agents once and exit
uv run --python 3.12 penny-team/orchestrator.py --list       # Show registered agents
```

## How It Works

The orchestrator loops every 30 seconds, checking which agents are due to run. When an agent is due, it:

1. Reads the agent's `CLAUDE.md` prompt
2. Calls `claude -p <prompt> --dangerously-skip-permissions --verbose --output-format stream-json`
3. Streams JSON events in real-time, logging tool calls and text output as they happen
4. Captures final output to `data/logs/<agent-name>.log`
5. Records success/failure and duration

Output streams to the terminal in real-time so you can watch agents work. Ctrl+C stops the orchestrator cleanly.

## Agents

Each agent is a directory with a single `CLAUDE.md` file (the prompt) and an entry in `orchestrator.py`:

```
penny-team/
  orchestrator.py          # Main loop
  base.py                  # Agent base class
  product-manager/
    CLAUDE.md              # PM agent prompt (requirements gathering)
  architect/
    CLAUDE.md              # Architect agent prompt (detailed specs)
  worker/
    CLAUDE.md              # Worker agent prompt (implementation)
```

### Product Manager

Gathers requirements on a 5-minute cycle (600s timeout, requires `requirements` label):
- Posts requirements on issues moved to `requirements`
- Responds to user questions and feedback about requirements
- User moves issue to `specification` when satisfied

### Architect

Writes detailed specifications on a 5-minute cycle (600s timeout, requires `specification` label):
- Reads approved requirements and writes detailed specs
- Responds to user feedback on specs
- User moves issue to `in-progress` when satisfied

### Worker

Implements features on a 5-minute cycle (1800s timeout, requires `in-progress`/`in-review` labels):
- Reads the spec, creates a feature branch (`issue-<N>-<slug>`), writes code and tests
- Runs `make check` to validate (format, lint, typecheck, tests) — retries up to 3 times
- Creates a PR with `Closes #N` linking back to the issue
- Moves issue to `in-review` when PR is ready
- Detects failing CI checks on `in-review` PRs and fixes them automatically
- Addresses PR review comments on `in-review` issues

## Adding a New Agent

1. Create a directory: `penny-team/my-agent/`
2. Write a `CLAUDE.md` prompt defining the agent's behavior
3. Register it in `orchestrator.py`:

```python
def get_agents() -> list[Agent]:
    return [
        Agent(
            name=AGENT_PM,
            interval_seconds=PM_INTERVAL,
            required_labels=[LABEL_REQUIREMENTS],
        ),
        Agent(
            name="my-agent",
            interval_seconds=300,
            required_labels=["some-label"],
        ),
    ]
```

## Agent Configuration

The `Agent` class accepts:

| Parameter | Default | Description |
|---|---|---|
| `name` | required | Agent identifier; prompt loaded from `penny-team/<name>/CLAUDE.md` |
| `interval_seconds` | 3600 | How often the agent runs |
| `working_dir` | project root | Working directory for Claude CLI |
| `timeout_seconds` | 600 | Max runtime before killing the process |
| `model` | None | Override Claude model (e.g. "opus") |
| `allowed_tools` | None | Restrict which tools Claude can use |
| `required_labels` | None | GitHub issue labels to check before running (any match = has work) |

## Streaming Output

The orchestrator uses `--verbose --output-format stream-json` to get real-time output from Claude CLI. Events are parsed and logged as they arrive:

- **`assistant`** events — agent text and tool calls (e.g. `[worker] [tool] Bash`)
- **`result`** events — final output captured for the per-agent log file

This means you can watch agents think, call tools, and produce output live in the terminal rather than waiting for the full run to complete.

## Logs

- `data/logs/orchestrator.log` — orchestrator events (start, stop, agent runs)
- `data/logs/<agent-name>.log` — raw Claude output per agent, appended each cycle

## GitHub Issue Labels

Each label maps to exactly one agent. Issues move through labels as a state machine:

`backlog` → `requirements` → `specification` → `in-progress` → `in-review` → closed

- **`backlog`** — Idea captured, not yet selected (no agent)
- **`requirements`** — PM gathers requirements
- **`specification`** — Architect writes detailed spec
- **`in-progress`** — Worker implements the spec
- **`in-review`** — Worker addresses PR feedback
