# Check tool configuration (single source of truth for tool parameters)
RUFF_TARGETS = penny/
# Exclude the live-model eval suite from the default test run — it's slow and
# needs a running Ollama, so it never runs in make check / CI (see make eval).
PYTEST_ARGS = penny/tests/ -v -m "not eval"
# -s streams the PERF lines (wall time + tok/s, printed per case) live.
EVAL_PYTEST_ARGS ?= penny/tests/eval/ -v -m eval -s

# --- Eval profiles -----------------------------------------------------------
# HOW a run is driven, named rather than reassembled from six variables each time.
# The two are not interchangeable-with-different-numbers: they are different machines.
#
#   local  (default) — the GPU on this box. One sample at a time, because the GPU
#                      serialises them anyway; asking for more only adds contention.
#                      One model, no provider, no roster: it serves what it has.
#   remote           — an OpenAI-compatible provider. Samples AND cases in parallel,
#                      because the provider serves them concurrently and the only
#                      local work left is the embedding model. Its models + their
#                      preferred upstreams come from EVAL_MODELS in .env (below).
#
# `make eval` is local; `make eval-remote` is remote. Every value a profile picks is
# still individually overridable, so a remote SERIAL run (debugging a single case) or a
# scoped local run needs no new target — just the variable.
EVAL_PROFILE ?= local
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

# --- Remote model endpoints --------------------------------------------------
# `make eval` forwards model config as explicit env vars rather than reading the
# mounted `/penny/.env`, because it runs from a WORKTREE and only the primary
# checkout has a real `.env`.  A remote OpenAI-compatible provider (OpenRouter)
# adds a credential to that list, and a credential belongs in `.env`, not in
# shell history — so each one is read from the host shell first and from the
# PRIMARY checkout's `.env` second, which makes `LLM_API_KEY=...` in that file
# work from any worktree.  Values are forwarded into the container's env and
# never echoed.
EVAL_PRIMARY_ENV := $(EVAL_PRIMARY_CHECKOUT)/.env
# Where the GPU lives, from inside the eval container.  Both the chat and the
# embedding endpoint default here, so pointing ONLY the chat model at a remote
# provider leaves memory's embeddings on the local Ollama that serves them —
# no remote chat provider serves `embeddinggemma`.
LLM_LOCAL_ENDPOINT := http://host.docker.internal:11434
# What "the model is on THIS machine's GPU" looks like in an endpoint URL. One definition,
# read both by the on_gpu decision and by the queue's busy probe, so the two can never
# disagree about which runs contend for the GPU — they did, and a remote run held the
# queue against a local one while touching no GPU at all.
LOCAL_ENDPOINT_HOSTS := host.docker.internal|localhost|127.0.0.1
# What each profile defaults to. The remote pair is the MEASURED product: 8 workers x 5
# samples = 40 in flight was both faster and cleaner than 80 on a 10-core box, where the
# extra contention cost more than the parallelism bought and samples began missing their
# channel-readiness window. Raise them together only with a measurement in hand.
EVAL_LOCAL_MODEL := gpt-oss:20b
EVAL_REMOTE_ENDPOINT := https://openrouter.ai/api
EVAL_REMOTE_WORKERS := 8
EVAL_REMOTE_CONCURRENCY := 5
# The .env key holding the remote profile's credential. A profile already names the
# endpoint and the model it points at; the credential is the third thing that travels with
# them, so naming it here keeps the RESOLUTION generic rather than teaching the plumbing
# about one provider — LLM_API_KEY still wins from the shell, then from .env, and this is
# only the last fallback, for the provider this profile actually defaults to. Point the
# profile somewhere else and you set LLM_API_KEY, exactly as before.
EVAL_REMOTE_API_KEY_VAR := OPENROUTER_API_KEY
# The remote profile's MODELS are not a make variable at all: they live in EVAL_MODELS in
# the primary checkout's .env, resolved by `penny.tests.eval.roster` before a run spends
# anything. There is deliberately no remote model default left to fall back to — a default
# here is exactly the ad-hoc single-model pass that made a run's model unrecoverable
# afterwards and its provider inexpressible.
# The marker `make eval-report` stamps into a posted run dir (#1757). Must match
# `penny.tests.eval.checkpoint.POSTED_MARKER` — the recipe checks it host-side.
POSTED_MARKER := .posted
# The line `penny.tests.eval.endpoint_smoke` prints the answering provider on. Must match
# `endpoint_smoke.PROVIDER_LINE_PREFIX` — the recipe reads the provider off it and forwards
# it as EVAL_PROVIDER, so the manifest records WHERE the model was served from (#1996).
SMOKE_PROVIDER_LINE := eval: chat provider =
# The two lines `penny.tests.eval.roster` prints its resolution on. Must match
# `roster.MODEL_LINE_PREFIX` / `roster.PREFERRED_PROVIDER_LINE_PREFIX` — the recipe reads
# the run's model and its PREFERRED upstream off them.
ROSTER_MODEL_LINE := eval: model =
ROSTER_PROVIDER_LINE := eval: preferred provider =
# Where a LEVER-LESS run drops its per-process health records. A report run uses its own
# run dir; an ephemeral iteration run has none, and this is inside the --rm container, so
# the records die with the run exactly as the rest of its artifacts do. The run still gets
# a whole-run health block, which is the point — a degraded run must be legible whether or
# not anyone meant to keep it.
EVAL_EPHEMERAL_HEALTH_DIR := /tmp/penny-eval-health
# Where `make eval-report` stages what it posts (#1808): the assembled body, and the
# comment parts the splitter cuts it into when it exceeds GitHub's 64K comment cap.
# Inside the run dir (so it rides the durable artifact home and is diagnosable after
# the fact) but in a SUBDIR, so it is never mistaken for a run dir itself.
COMMENT_SUBDIR := comment
COMMENT_BODY := body.md
# Alias for the make binary used to mint the token inside the eval-report recipe. Referenced
# through this alias, NOT the literal `$(MAKE)`, on purpose: GNU make executes any recipe line
# containing the literal `$(MAKE)`/`${MAKE}` even under `-n` (recursive-make tracing), and
# eval-report's single logical recipe line also posts a PR comment + writes a marker — side
# effects `make -n eval-report` must NOT trigger. The alias keeps the dry-run a true dry-run.
SUBMAKE := $(MAKE)

.PHONY: up prod prod-ios kill clean-project-images docker-prune build browser-build client-check client-services-check fmt lint fix typecheck check pytest eval eval-remote eval-report assemble token migrate-test migrate-validate

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

# Run service-layer tests without building or launching PennyDev/PennyTestflight.
client-services-check:
	./scripts/client-services-check.sh

# Print a GitHub App installation token for use with gh CLI
# Usage: GH_TOKEN=$(make token) gh pr create ...
# auth.py is pure stdlib, so it runs in the penny service (github_api is mounted
# there at /shared/github_api); the GitHub App creds come from the mounted .env.
#
# FAILS LOUDLY, and the guard lives HERE rather than in every caller.  Compose
# chatter on stderr is held back while the run succeeds (callers capture stdout,
# so only the token is ever printed) and released verbatim the moment it does
# not — an empty token with no reason is what this target used to hand back, and
# a caller cannot diagnose a missing credential from an unready mount from a
# revoked key without it.  Empty output on a zero exit is treated as failure too:
# the whole point of the target is that success means a usable token.
token:
	@err="$$(mktemp)"; \
	tok="$$(docker compose run --rm --no-deps --entrypoint "" penny \
		uv run python /shared/github_api/auth.py 2>"$$err")"; \
	status=$$?; \
	if [ $$status -ne 0 ] || [ -z "$$tok" ]; then \
		echo "make token: could not mint a GitHub App installation token." >&2; \
		[ -s "$$err" ] && sed 's/^/  /' "$$err" >&2; \
		echo "  (checked: docker running? .env mounted with GITHUB_APP_* set?)" >&2; \
		rm -f "$$err"; \
		exit 1; \
	fi; \
	rm -f "$$err"; \
	printf '%s\n' "$$tok"

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
	cd browser && npm install --silent && npm run typecheck && npm test

pytest: $(if $(LOCAL),,build)
	$(RUN) pytest $(PYTEST_ARGS)

# Live-model contract suite — drives the REAL agents against a real model
# (a chat model + embeddinggemma) on synthetic seeds. Slow and stochastic, so it's kept
# out of make check; run it by hand to validate prompt/behaviour changes.
#
#   make eval                                  the local GPU, one sample at a time
#   make eval-remote                           a provider, 8 cases x 5 samples at once
#   EVAL_SAMPLES=2 make eval                   fewer samples per case
#   EVAL_PYTEST_ARGS="<node ids> -m eval -s" make eval-remote      scoped to some cases
#   EVAL_WORKERS=1 make eval-remote            remote but SERIAL, for debugging one case
#   LLM_MODEL=<a model EVAL_MODELS names> make eval-remote         the roster's other model
#
# WHICH models a remote run may measure is CONFIGURATION, not a make variable: EVAL_MODELS
# in the primary checkout's .env is a JSON list of {"model": ..., "provider": ...} entries,
# at least two of them, and a remote run refuses to start without it (see
# penny.tests.eval.roster for why two, and why the requirement is remote-only). The recipe
# resolves this invocation's entry — the first by default, or the one LLM_MODEL names — and
# forwards its provider as a PREFERENCE with fallbacks ON, never a hard pin: pinning hard
# put 325 rate limits on one endpoint at a concurrency the same run handled with zero
# unpinned. Which upstream actually ANSWERED is recorded instead, per call.
#
# The embedding model is NOT part of the profile: it stays on the local Ollama in both,
# so moving the chat model never silently moves the vector space every memory case is
# scored against. Override LLM_EMBEDDING_API_URL deliberately if you mean to.
# Before anything is spent, ONE call proves the endpoint serves the model (see
# penny.tests.eval.endpoint_smoke). Every sample builds its own preflight, so without this
# an unserveable model is discovered 755 times, concurrently, minutes in — which is how a
# full-suite run was spent against a model whose provider 404'd every call. The refusal
# carries the provider's own message, because that message is the whole answer.
# Run identity is per-INVOCATION, not per-second: the stamp carries the shell's pid, because
# two runs starting in the same second would otherwise share a report directory AND a run id
# and quietly write one corrupted run that looks like a valid one — the manifest write-once,
# both xdist workers named gw0 appending to one results file, per-sample DBs on colliding
# paths. Agents dispatched together start within milliseconds of each other by construction,
# so this is the ordinary case for concurrent remote evals rather than a rare race. The
# stamp still leads the name, so run dirs keep sorting chronologically.
# Test-level parallelism: pass `-n N` through EVAL_PYTEST_ARGS (pytest-xdist) to run N
# CASES at once, on top of the EVAL_CONCURRENCY samples each case runs at. Every worker
# is a separate PROCESS resolving the run from its own environment, so EVAL_RUN_ID is
# fixed here once and forwarded — otherwise each would stamp its own clock into the run
# id and one directory would hold N runs. Each worker appends to its own results file;
# assemble reads them all.
# Remote endpoints: point LLM_API_URL at any OpenAI-compatible provider and the
# suite runs against it — e.g. `LLM_API_URL=https://openrouter.ai/api
# LLM_MODEL=openai/gpt-oss-20b make eval`, with LLM_API_KEY in the primary
# checkout's .env. The chat endpoint moving does NOT move the embedding one (it
# defaults to the local Ollama independently), and a remote chat model skips the
# GPU queue below, since it contends for no GPU. An empty key on a remote
# endpoint fails the run up front rather than 401-ing every sample.
# GPU queue: LOCAL RUNS ONLY, and it exists for exactly one reason — this machine has one
# GPU. A remote run takes no ticket, waits for nothing, and is invisible to the probe, so
# any number of them proceed at once; that is what lets several agents drive remote evals
# concurrently without colliding with each other or with someone working locally.
# For a local run it is strictly first-come-first-served via ticket files: each takes a
# ticket in EVAL_QUEUE_DIR and runs only when its ticket is the oldest LIVE one (tickets
# whose holder PID is gone are reaped, so a killed waiter can never wedge the line) and no
# LOCAL eval container already holds the GPU. The ticket is held until the eval finishes —
# later arrivals cannot jump the queue. While waiting, prints queue position and the
# current GPU holder for observability.
# The busy probe reads the endpoint out of the container's own command and matches only
# LOCAL_ENDPOINT_HOSTS: it used to match any eval container, so a remote run held the line
# against a local one while touching no GPU at all.
# Durable reports (#1734): a report run (one that declares its EVAL_LEVER) with
# no explicit EVAL_REPORT_DIR defaults to a run-stamped dir under the primary
# checkout's mounted data/eval-artifacts, so artifacts survive the worktree that
# ran the eval. An explicit EVAL_REPORT_DIR is always honored; a lever-less
# iteration run stays ephemeral (no artifacts, no lever requirement) as before.
# The remote entry point: `make eval-remote`, plus anything else you would pass to
# `make eval`. Sets the profile and hands off — a target-specific variable reaches the
# prerequisite, so there is no recursive make here and `make -n eval-remote` stays a true
# dry-run.
eval-remote: EVAL_PROFILE = remote
eval-remote: eval

eval: $(if $(LOCAL),,build)
	@mkdir -p "$(EVAL_ARTIFACTS_HOST)"; \
	banner="$$($(EVAL_RUN) python -m penny.tests.eval.checkpoint banner "$(EVAL_ARTIFACTS_MOUNT)" 2>/dev/null || true)"; \
	if [ -n "$$banner" ]; then printf '%s\n' "$$banner"; fi; \
	from_env() { sed -n "s/^$$1=//p" "$(EVAL_PRIMARY_ENV)" 2>/dev/null | tail -1 | tr -d '"'; }; \
	case "$(EVAL_PROFILE)" in \
		local)  url_default="$(LLM_LOCAL_ENDPOINT)"; model_default="$(EVAL_LOCAL_MODEL)"; \
			workers_default=1; concurrency_default=1; key_var="" ;; \
		remote) url_default="$(EVAL_REMOTE_ENDPOINT)"; model_default=""; \
			workers_default=$(EVAL_REMOTE_WORKERS); concurrency_default=$(EVAL_REMOTE_CONCURRENCY); \
			key_var="$(EVAL_REMOTE_API_KEY_VAR)" ;; \
		*) echo "eval: unknown EVAL_PROFILE '$(EVAL_PROFILE)' — expected 'local' or 'remote'" >&2; exit 1 ;; \
	esac; \
	workers="$${EVAL_WORKERS:-$$workers_default}"; \
	concurrency="$${EVAL_CONCURRENCY:-$$concurrency_default}"; \
	model="$${LLM_MODEL:-$$model_default}"; \
	preferred=""; \
	if [ "$(EVAL_PROFILE)" = remote ]; then \
		roster_log="$$(mktemp)"; \
		if ( $(EVAL_RUN) env EVAL_MODELS="$${EVAL_MODELS:-$$(from_env EVAL_MODELS)}" \
			python -m penny.tests.eval.roster $$model ) > "$$roster_log" 2>&1; then rostered=1; else rostered=0; fi; \
		cat "$$roster_log"; \
		model="$$(sed -n 's/^$(ROSTER_MODEL_LINE) //p' "$$roster_log" | tail -1)"; \
		preferred="$$(sed -n 's/^$(ROSTER_PROVIDER_LINE) //p' "$$roster_log" | tail -1)"; \
		rm -f "$$roster_log"; \
		if [ "$$rostered" = 0 ]; then exit 1; fi; \
	fi; \
	llm_url="$${LLM_API_URL:-$$url_default}"; \
	llm_key="$${LLM_API_KEY:-$$(from_env LLM_API_KEY)}"; \
	if [ -z "$$llm_key" ] && [ -n "$$key_var" ]; then llm_key="$$(from_env "$$key_var")"; fi; \
	embed_url="$${LLM_EMBEDDING_API_URL:-$(LLM_LOCAL_ENDPOINT)}"; \
	embed_key="$${LLM_EMBEDDING_API_KEY:-$$(from_env LLM_EMBEDDING_API_KEY)}"; \
	if echo "$$llm_url" | grep -qE '$(LOCAL_ENDPOINT_HOSTS)'; then on_gpu=1; else on_gpu=0; fi; \
	if [ "$$on_gpu" = 0 ]; then \
		echo "eval: chat model is REMOTE ($$llm_url) — skipping the local GPU queue; embeddings stay at $$embed_url"; \
		if [ -z "$$llm_key" ]; then \
			echo "eval: no credential — set LLM_API_KEY (or $(EVAL_REMOTE_API_KEY_VAR)) in $(EVAL_PRIMARY_ENV), or LLM_API_KEY in the shell." >&2; \
			exit 1; \
		fi; \
	fi; \
	echo "eval: profile $(EVAL_PROFILE) — $$model at $$llm_url$${preferred:+ preferring $$preferred} · $$workers worker(s) x $$concurrency sample(s) in flight"; \
	smoke_log="$$(mktemp)"; \
	if ( $(EVAL_RUN) env LLM_API_URL="$$llm_url" LLM_API_KEY="$$llm_key" \
		LLM_MODEL="$$model" LLM_PROVIDER="$$preferred" \
		LLM_EMBEDDING_API_URL="$$embed_url" LLM_EMBEDDING_API_KEY="$$embed_key" \
		LLM_EMBEDDING_MODEL="$${LLM_EMBEDDING_MODEL:-embeddinggemma}" \
		python -m penny.tests.eval.endpoint_smoke ) > "$$smoke_log" 2>&1; then smoked=1; else smoked=0; fi; \
	cat "$$smoke_log"; \
	provider="$$(sed -n 's/^$(SMOKE_PROVIDER_LINE) //p' "$$smoke_log" | tail -1)"; \
	rm -f "$$smoke_log"; \
	if [ "$$smoked" = 0 ]; then \
		echo "eval: refusing to start — the endpoint above will not serve this model." >&2; \
		exit 1; \
	fi; \
	if [ "$$on_gpu" = 1 ]; then \
		mkdir -p "$(EVAL_QUEUE_DIR)"; \
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
			busy=$$(docker ps --no-trunc --format '{{.Names}} {{.Command}}' 2>/dev/null | grep -E 'tests/eval|-m eval' | grep -E "LLM_API_URL=[^ ]*($(LOCAL_ENDPOINT_HOSTS))" | awk '{print $$1}' | head -1); \
			if [ "$$head" = "$$ticket" ] && [ -z "$$busy" ]; then break; fi; \
			echo "eval queued: $$ahead ahead of us$${busy:+; GPU held by $$busy} (ticket $$ticket)"; \
			sleep $$((15 + $$$$ % 10)); \
		done; \
	fi; \
	stamp="$$(date -u +%Y%m%dT%H%M%SZ)"; \
	commit="$$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	run_key="$$stamp-$$(printf '%05d' $$$$)"; \
	run_id="$${EVAL_RUN_ID:-run-$$run_key-$$(printf %.8s "$$commit")}"; \
	report_dir="$${EVAL_REPORT_DIR}"; \
	if [ -z "$$report_dir" ] && [ -n "$${EVAL_LEVER}" ]; then \
		report_dir="$(EVAL_ARTIFACTS_MOUNT)/run-$$run_key"; \
		echo "eval: reports → $$report_dir  (durable host dir: $(EVAL_ARTIFACTS_HOST))"; \
	fi; \
	health_dir="$${report_dir:-$(EVAL_EPHEMERAL_HEALTH_DIR)/$$run_key}"; \
	$(EVAL_RUN) env \
		LLM_API_URL="$$llm_url" \
		LLM_API_KEY="$$llm_key" \
		LLM_MODEL="$$model" \
		LLM_EMBEDDING_API_URL="$$embed_url" \
		LLM_EMBEDDING_API_KEY="$$embed_key" \
		LLM_EMBEDDING_MODEL="$${LLM_EMBEDDING_MODEL:-embeddinggemma}" \
		LLM_TIMEOUT="$${LLM_TIMEOUT}" \
		EVAL_SAMPLES="$${EVAL_SAMPLES:-5}" \
		EVAL_CONCURRENCY="$$concurrency" \
		EVAL_REPORT_DIR="$$report_dir" \
		EVAL_BASELINE="$${EVAL_BASELINE}" \
		EVAL_DUMP_THINKING="$${EVAL_DUMP_THINKING}" \
		EVAL_LEVER="$${EVAL_LEVER}" \
		EVAL_RUN_ID="$$run_id" \
		EVAL_HEALTH_DIR="$$health_dir" \
		LLM_PROVIDER="$$preferred" \
		EVAL_PREFERRED_PROVIDER="$$preferred" \
		EVAL_PROVIDER="$$provider" \
		EVAL_COMMIT="$$commit" \
		EVAL_DIRTY_DIFF="$$(git diff HEAD 2>/dev/null)" \
		pytest $(EVAL_PYTEST_ARGS) $$( [ "$$workers" -gt 1 ] && echo "-n $$workers" )

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

# Post a completed eval run's assembled report to its iteration PR as a comment — the ONE-SHOT that
# makes the joint-checkpoint rule (run → report → STOP for joint review) STRUCTURAL (#1757). It
# assembles the named run (RUN=run-<stamp>) or, by default, the MOST-RECENT completed run under the
# durable artifact home (the same #1734 EVAL_ARTIFACTS_HOST/_MOUNT derivation eval/assemble use),
# posts the assembled markdown VERBATIM to PR=<n>, then stamps a `.posted` marker holding the comment
# URL into the run dir. The assemble output is captured CLEANLY by invoking the containerized module
# DIRECTLY (never `make assemble` piped through stripping); the token is minted inside the recipe.
# Idempotent by the marker: a run already posted re-posts ONLY with FORCE=1 — otherwise it prints the
# existing comment URL and exits 0. Fails loudly (never a silent no-op) when PR is unset, the run dir
# is missing, the token is empty, or the assembled output is empty. `make -n eval-report PR=<n>`
# shows the same durable-home resolution (`-v <primary>/data/eval-artifacts:/penny/eval-artifacts`).
# OVER THE 64K CAP (#1808): GitHub refuses a comment body over 65,536 chars and an 8-sample chat beat
# assembles to ~290K, so the body is staged into <run>/$(COMMENT_SUBDIR)/ and cut there by
# `penny.tests.eval.comment_split` — on SAMPLE-FOLD boundaries only, each part headed `report N of M`,
# posted in order, and `.posted` stamped with the FIRST part's URL so idempotency + the unreviewed-run
# banner stay honest. The splitter also REFUSES a body opening with build noise (`docker compose`,
# `GIT_COMMIT=`, `#1 [internal]`) — the pollution a hand-piped `make assemble` publishes.
eval-report: $(if $(LOCAL),,build)
	@if [ -z "$(PR)" ]; then \
		echo "eval-report: PR is required — usage: make eval-report PR=<n> [RUN=<run-dir-name>] [FORCE=1]" >&2; \
		exit 1; \
	fi; \
	run="$(if $(filter command line,$(origin RUN)),$(RUN),)"; \
	if [ -z "$$run" ]; then \
		run="$$($(EVAL_RUN) python -m penny.tests.eval.checkpoint latest "$(EVAL_ARTIFACTS_MOUNT)" 2>/dev/null)"; \
		if [ -z "$$run" ]; then \
			echo "eval-report: no completed run dirs under $(EVAL_ARTIFACTS_HOST) — run make eval first" >&2; \
			exit 1; \
		fi; \
	fi; \
	host_dir="$(EVAL_ARTIFACTS_HOST)/$$run"; \
	if [ ! -d "$$host_dir" ]; then \
		echo "eval-report: run dir not found: $$host_dir" >&2; \
		exit 1; \
	fi; \
	if [ -f "$$host_dir/$(POSTED_MARKER)" ] && [ -z "$(FORCE)" ]; then \
		echo "eval-report: $$run already posted → $$(cat "$$host_dir/$(POSTED_MARKER)")"; \
		echo "eval-report: re-post with FORCE=1"; \
		exit 0; \
	fi; \
	host_parts="$$host_dir/$(COMMENT_SUBDIR)"; \
	mount_parts="$(EVAL_ARTIFACTS_MOUNT)/$$run/$(COMMENT_SUBDIR)"; \
	mkdir -p "$$host_parts"; \
	if ! $(EVAL_RUN) env EVAL_BASELINE="$${EVAL_BASELINE}" python -m penny.tests.eval.assemble "$(EVAL_ARTIFACTS_MOUNT)/$$run" > "$$host_parts/$(COMMENT_BODY)"; then \
		echo "eval-report: assemble failed for $$run" >&2; \
		exit 1; \
	fi; \
	if [ ! -s "$$host_parts/$(COMMENT_BODY)" ]; then \
		echo "eval-report: assemble produced no output for $$run — is it a completed run?" >&2; \
		exit 1; \
	fi; \
	if ! names="$$($(EVAL_RUN) python -m penny.tests.eval.comment_split "$$mount_parts/$(COMMENT_BODY)" "$$mount_parts")"; then \
		echo "eval-report: could not prepare $$run for posting" >&2; \
		exit 1; \
	fi; \
	tok="$$($(SUBMAKE) token)"; \
	if [ -z "$$tok" ]; then \
		echo "eval-report: make token returned empty — run from the primary checkout with a real .env" >&2; \
		exit 1; \
	fi; \
	first=""; \
	for name in $$names; do \
		url="$$(GH_TOKEN=$$tok gh pr comment "$(PR)" --body-file "$$host_parts/$$name")"; \
		if [ -z "$$url" ]; then \
			echo "eval-report: gh pr comment returned no URL for $$name$${first:+ (already posted: $$first)}" >&2; \
			exit 1; \
		fi; \
		echo "eval-report: posted $$name → $$url"; \
		if [ -z "$$first" ]; then first="$$url"; fi; \
	done; \
	printf '%s\n' "$$first" > "$$host_dir/$(POSTED_MARKER)"; \
	echo "eval-report: posted $$run → PR #$(PR) as $$(printf '%s\n' $$names | wc -l | tr -d ' ') comment(s); marker → $$first"

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
