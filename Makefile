# Check tool configuration (single source of truth for tool parameters)
RUFF_TARGETS = penny/
# Exclude the live-model eval suite from the default test run — it's slow and
# needs a running Ollama, so it never runs in make check / CI (see make eval).
PYTEST_ARGS = penny/tests/ -v -m "not eval"
# -s streams the PERF lines (wall time + tok/s, printed per case) live.
EVAL_PYTEST_ARGS ?= penny/tests/eval/ -v -m eval -s
# FIFO ticket directory for serializing make eval on the single-tenant GPU.
EVAL_QUEUE_DIR ?= /tmp/penny-eval-queue

# --- Durable eval artifacts (#1734) ------------------------------------------
# Eval artifacts (per-sample DBs, results.jsonl, manifests, transcripts) must
# survive the worktree that ran them being swept post-merge. The `./data` bind
# mount is relative to the compose-file dir (the CWD — a worktree when an agent
# runs `make eval`), so anything written under it dies with that tree. Resolve
# the PRIMARY checkout host-side from the shared git *common* dir — identical
# from the primary checkout and every worktree, since all worktrees share one
# `.git` — and mount its `data/eval-artifacts` at a stable container path. So
# eval artifacts always land in the primary tree no matter which worktree ran
# the eval. (stderr swallowed + `.` fallback so a non-repo/container parse of
# this Makefile can't error; the mount is only wired onto eval/assemble.)
EVAL_PRIMARY_CHECKOUT := $(shell dirname "$$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" 2>/dev/null)
EVAL_ARTIFACTS_HOST := $(EVAL_PRIMARY_CHECKOUT)/data/eval-artifacts
EVAL_ARTIFACTS_MOUNT := /penny/eval-artifacts

.PHONY: up prod prod-ios kill clean-project-images docker-prune build browser-build client-check fmt lint fix typecheck check pytest eval assemble token migrate-test migrate-validate

# --- Docker Compose ---

# Enable the `signal` compose profile (the signal-api container + penny's
# startup gate on it) only when SIGNAL_NUMBER is set. Non-Signal deployments
# (Discord, iOS) leave it empty — as .env.example now ships — so `up`/`prod`
# start penny alone and never wait on a signal-api they don't use.
SIGNAL_PROFILE := $(shell awk -F= '/^[[:space:]]*SIGNAL_NUMBER[[:space:]]*=/{v=$$2; gsub(/["'\'' ]/,"",v); if (v!="") print "--profile signal"}' .env 2>/dev/null)

up: browser-build
	docker compose $(SIGNAL_PROFILE) down --remove-orphans
	GIT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) \
	GIT_COMMIT_MESSAGE=$$(git log -1 --pretty=%B 2>/dev/null | tr '\n' ' ' | sed 's/ *$$//' || echo unknown) \
	SNAPSHOT=1 \
	docker compose $(SIGNAL_PROFILE) up --build

prod: browser-build
	docker compose -f docker-compose.yml $(SIGNAL_PROFILE) down --remove-orphans
	GIT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) \
	GIT_COMMIT_MESSAGE=$$(git log -1 --pretty=%B 2>/dev/null | tr '\n' ' ' | sed 's/ *$$//' || echo unknown) \
	SNAPSHOT=1 \
	docker compose -f docker-compose.yml $(SIGNAL_PROFILE) up --build

prod-ios: browser-build
	GIT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) \
	GIT_COMMIT_MESSAGE=$$(git log -1 --pretty=%B 2>/dev/null | tr '\n' ' ' | sed 's/ *$$//' || echo unknown) \
	SNAPSHOT=1 \
	docker compose -f docker-compose.yml run --rm --service-ports --no-deps --build -e CHANNEL_TYPE=ios penny

# Tear down this compose project's containers and remove its locally-built
# images (alias of clean-project-images, kept for its familiar name).
kill: clean-project-images

# Remove THIS compose project's containers, locally-built images and anonymous
# volumes without touching other projects or the shared build cache. The penny
# service is pinned to the fixed `penny` tag, so `make check` from any worktree
# overwrites one shared image rather than leaving a project-scoped one behind;
# `--rmi local` drops that tag (rebuilt on the next `up`, and skipped here while a
# container holds it, e.g. production). Run this at §9 teardown. Safe to run with
# no containers up — `down` is a no-op and `--rmi local` still drops the images.
# `--volumes` only clears this project's anonymous volumes; penny's persistent
# data lives in bind mounts (./data), which it never touches.
clean-project-images:
	docker compose $(SIGNAL_PROFILE) down --rmi local --volumes --remove-orphans

# Best-effort global reclaim for when Docker has eaten the disk: drop stopped
# containers, dangling (untagged) images, the build cache, and unused volumes.
# Unlike clean-project-images this is NOT project-scoped, but it never removes a
# tagged image or a volume still in use — safe to run anytime. Each step is
# `|| true` so a busy resource can't fail the target. (A full disk once traced to
# 56 leftover agent images + ~23GB of stale build cache — clean-project-images
# clears the former per project, this clears the latter machine-wide.)
docker-prune:
	docker container prune -f || true
	docker image prune -f || true
	docker builder prune -f || true
	docker volume prune -f || true

build:
	GIT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) \
	GIT_COMMIT_MESSAGE=$$(git log -1 --pretty=%B 2>/dev/null | tr '\n' ' ' | sed 's/ *$$//' || echo unknown) \
	docker compose build penny

browser-build:
	cd browser && npm install && npm run build

# Build the iOS client and run PennyClientTests on a simulator (requires Xcode).
# CI runs the same script on any PR touching penny-client/ (client-check.yml).
client-check:
	./scripts/client-check.sh

# Print a GitHub App installation token for use with gh CLI
# Usage: GH_TOKEN=$(make token) gh pr create ...
# auth.py is pure stdlib, so it runs in the penny service (github_api is mounted
# there at /shared/github_api); the GitHub App creds come from the mounted .env.
token:
	@docker compose run --rm --no-deps --entrypoint "" penny uv run python /shared/github_api/auth.py 2>/dev/null

# --- Code quality (auto-detects host vs container via LOCAL env var) ---

ifdef LOCAL
# Inside a container — run tools directly
RUN = cd penny &&
EVAL_RUN = cd penny &&
else
# On host — run tools inside Docker containers
# --no-deps: dev tools don't need signal-api healthy (would block on first run)
RUN = docker compose run --rm --no-deps penny
# eval/assemble additionally bind-mount the primary checkout's durable artifact
# dir (#1734) at $(EVAL_ARTIFACTS_MOUNT) — the only two targets that read/write
# it — so their output survives the running worktree being swept.
EVAL_RUN = docker compose run --rm --no-deps -v "$(EVAL_ARTIFACTS_HOST):$(EVAL_ARTIFACTS_MOUNT)" penny
endif

fix: $(if $(LOCAL),,build)
	$(RUN) ruff format $(RUFF_TARGETS)
	$(RUN) ruff check --fix $(RUFF_TARGETS)

typecheck: $(if $(LOCAL),,build)
	$(RUN) ty check --exit-zero-on-warning $(RUFF_TARGETS)

check: $(if $(LOCAL),,build)
	$(RUN) ruff format --check $(RUFF_TARGETS)
	$(RUN) ruff check $(RUFF_TARGETS)
	$(RUN) ty check --exit-zero-on-warning $(RUFF_TARGETS)
	$(RUN) python -m penny.database.migrate --validate
	$(RUN) pytest $(PYTEST_ARGS)
	cd browser && npm install --silent && npx tsc --noEmit

pytest: $(if $(LOCAL),,build)
	$(RUN) pytest $(PYTEST_ARGS)

# Live-model contract suite — drives the REAL agents against a running Ollama
# (gpt-oss + embeddinggemma) on synthetic seeds. Slow and stochastic, so it's
# kept out of make check; run it by hand to validate prompt/behaviour changes.
# Forwards the model endpoint into the container (defaulting to the docker host,
# where Ollama runs); override LLM_MODEL / LLM_EMBEDDING_MODEL / EVAL_SAMPLES on
# the host to taste, e.g. `EVAL_SAMPLES=2 make eval`.
# GPU queue: strictly first-come-first-served via ticket files. Each invocation
# takes a ticket in EVAL_QUEUE_DIR and runs only when its ticket is the oldest
# LIVE one (tickets whose holder PID is gone are reaped, so a killed waiter can
# never wedge the line) and no eval container already holds the GPU. The ticket
# is held until the eval finishes — later arrivals cannot jump the queue. While
# waiting, prints queue position and the current GPU holder for observability.
# Durable reports (#1734): a report run (one that declares its EVAL_LEVER) with
# no explicit EVAL_REPORT_DIR defaults to a run-stamped dir under the primary
# checkout's mounted data/eval-artifacts, so artifacts survive the worktree that
# ran the eval. An explicit EVAL_REPORT_DIR is always honored; a lever-less
# iteration run stays ephemeral (no artifacts, no lever requirement) as before.
eval: $(if $(LOCAL),,build)
	@mkdir -p "$(EVAL_QUEUE_DIR)" "$(EVAL_ARTIFACTS_HOST)"; \
	ticket="$$(date +%s)-$$(printf '%08d' $$$$)"; \
	echo $$$$ > "$(EVAL_QUEUE_DIR)/$$ticket"; \
	trap 'rm -f "$(EVAL_QUEUE_DIR)/$$ticket"' EXIT INT TERM; \
	while :; do \
		head=""; ahead=0; \
		for t in $$(ls "$(EVAL_QUEUE_DIR)" 2>/dev/null | sort); do \
			pid=$$(cat "$(EVAL_QUEUE_DIR)/$$t" 2>/dev/null || true); \
			if [ -z "$$pid" ] || ! kill -0 "$$pid" 2>/dev/null; then rm -f "$(EVAL_QUEUE_DIR)/$$t"; continue; fi; \
			if [ -z "$$head" ]; then head="$$t"; fi; \
			if [ "$$t" = "$$ticket" ]; then break; fi; \
			ahead=$$((ahead + 1)); \
		done; \
		busy=$$(docker ps --no-trunc --format '{{.Names}} {{.Command}}' 2>/dev/null | grep -E 'tests/eval|-m eval' | awk '{print $$1}' | head -1); \
		if [ "$$head" = "$$ticket" ] && [ -z "$$busy" ]; then break; fi; \
		echo "eval queued: $$ahead ahead of us$${busy:+; GPU held by $$busy} (ticket $$ticket)"; \
		sleep $$((15 + $$$$ % 10)); \
	done; \
	report_dir="$${EVAL_REPORT_DIR}"; \
	if [ -z "$$report_dir" ] && [ -n "$${EVAL_LEVER}" ]; then \
		report_dir="$(EVAL_ARTIFACTS_MOUNT)/run-$$(date -u +%Y%m%dT%H%M%SZ)"; \
		echo "eval: reports → $$report_dir  (durable host dir: $(EVAL_ARTIFACTS_HOST))"; \
	fi; \
	$(EVAL_RUN) env \
		LLM_API_URL="$${LLM_API_URL:-http://host.docker.internal:11434}" \
		LLM_MODEL="$${LLM_MODEL:-gpt-oss:20b}" \
		LLM_EMBEDDING_MODEL="$${LLM_EMBEDDING_MODEL:-embeddinggemma}" \
		EVAL_SAMPLES="$${EVAL_SAMPLES:-5}" \
		EVAL_REPORT_DIR="$$report_dir" \
		EVAL_BASELINE="$${EVAL_BASELINE}" \
		EVAL_DUMP_THINKING="$${EVAL_DUMP_THINKING}" \
		EVAL_LEVER="$${EVAL_LEVER}" \
		EVAL_COMMIT="$$(git rev-parse HEAD 2>/dev/null || echo unknown)" \
		EVAL_DIRTY_DIFF="$$(git diff HEAD 2>/dev/null)" \
		pytest $(EVAL_PYTEST_ARGS)

# Assemble a completed eval run's artifacts (manifest.json + results.jsonl + the
# per-case <case_id>.md transcripts) into THE postable PR comment (#1717) and
# print it to stdout. Pure artifact consumption — no model, no GPU, no queue — so
# it runs straight through without the eval queue. Mounts the same durable
# artifact dir `make eval` writes to (#1734), so it reads a run that outlived its
# worktree. EVAL_REPORT_DIR names the specific run dir to assemble (a run-stamped
# subdir under $(EVAL_ARTIFACTS_MOUNT)); it's read from the recipe's shell env
# (`$${…}`), not a make `=` var, so `EVAL_REPORT_DIR=… make assemble` takes
# effect. Defaults to the durable mount root. EVAL_BASELINE is forwarded the same
# way — an explicit override re-diffs against a different baseline; unset, the run's
# own manifest-recorded baseline drives the flips index (#1752).
# EVERY sample folds whole under its banner — collapsed by default, its full body
# always a click away, identical in the on-disk `.md` and this comment (#1753/#1759).
# There is no compact/banner-only form and no `--full` flag.
assemble: $(if $(LOCAL),,build)
	@mkdir -p "$(EVAL_ARTIFACTS_HOST)"
	$(EVAL_RUN) env EVAL_BASELINE="$${EVAL_BASELINE}" python -m penny.tests.eval.assemble "$${EVAL_REPORT_DIR:-$(EVAL_ARTIFACTS_MOUNT)}"

migrate-test: $(if $(LOCAL),,build)
	$(RUN) python -m penny.database.migrate --test

migrate-validate: $(if $(LOCAL),,build)
	$(RUN) python -m penny.database.migrate --validate

signal-avatar:
	@python3 -c " \
	import base64, json, os, urllib.request; \
	number = os.environ.get('SIGNAL_NUMBER', ''); \
	api = os.environ.get('SIGNAL_API_URL', 'http://localhost:8080'); \
	f = open('penny.png', 'rb'); avatar = base64.b64encode(f.read()).decode(); f.close(); \
	data = json.dumps({'name': 'Penny', 'avatar': avatar}).encode(); \
	req = urllib.request.Request(api + '/v1/profiles/' + number, data=data, headers={'Content-Type': 'application/json'}, method='PUT'); \
	urllib.request.urlopen(req, timeout=10); \
	print('Signal avatar set for ' + number) \
	"
