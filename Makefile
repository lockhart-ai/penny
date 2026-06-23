# Check tool configuration (single source of truth for tool parameters)
RUFF_TARGETS = penny/
# Exclude the live-model eval suite from the default test run — it's slow and
# needs a running Ollama, so it never runs in make check / CI (see make eval).
PYTEST_ARGS = penny/tests/ -v -m "not eval"
# -s streams the PERF lines (wall time + tok/s, printed per case) live.
EVAL_PYTEST_ARGS = penny/tests/eval/ -v -m eval -s
TEAM_RUFF_TARGETS = penny_team/
TEAM_PYTEST_ARGS = tests/ -v

.PHONY: up prod kill build team-build browser-build fmt lint fix typecheck check pytest eval token migrate-test migrate-validate

# --- Docker Compose ---

up: browser-build
	GIT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) \
	GIT_COMMIT_MESSAGE=$$(git log -1 --pretty=%B 2>/dev/null | tr '\n' ' ' | sed 's/ *$$//' || echo unknown) \
	SNAPSHOT=1 \
	docker compose --profile team up --build

prod: browser-build
	GIT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) \
	GIT_COMMIT_MESSAGE=$$(git log -1 --pretty=%B 2>/dev/null | tr '\n' ' ' | sed 's/ *$$//' || echo unknown) \
	SNAPSHOT=1 \
	docker compose -f docker-compose.yml up --build penny signal-api

kill:
	docker compose --profile team down --rmi local --remove-orphans

build:
	GIT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) \
	GIT_COMMIT_MESSAGE=$$(git log -1 --pretty=%B 2>/dev/null | tr '\n' ' ' | sed 's/ *$$//' || echo unknown) \
	docker compose build penny

team-build:
	docker compose build team

browser-build:
	cd browser && npm install && npm run build

# Print a GitHub App installation token for use with gh CLI
# Usage: GH_TOKEN=$(make token) gh pr create ...
token:
	@docker compose --profile team run --rm --no-deps --entrypoint "" pm uv run python /shared/github_api/auth.py 2>/dev/null

# --- Code quality (auto-detects host vs container via LOCAL env var) ---

ifdef LOCAL
# Inside a container — run tools directly
RUN = cd penny &&
TEAM_RUN = cd penny-team &&
else
# On host — run tools inside Docker containers
# --no-deps: dev tools don't need signal-api healthy (would block on first run)
RUN = docker compose run --rm --no-deps penny
TEAM_RUN = docker compose run --rm --no-deps team
endif

fix: $(if $(LOCAL),,build team-build)
	$(RUN) ruff format $(RUFF_TARGETS)
	$(RUN) ruff check --fix $(RUFF_TARGETS)
	$(TEAM_RUN) ruff format $(TEAM_RUFF_TARGETS)
	$(TEAM_RUN) ruff check --fix $(TEAM_RUFF_TARGETS)

typecheck: $(if $(LOCAL),,build team-build)
	$(RUN) ty check --exit-zero-on-warning $(RUFF_TARGETS)
	$(TEAM_RUN) ty check --exit-zero-on-warning $(TEAM_RUFF_TARGETS)

check: $(if $(LOCAL),,build team-build)
	$(RUN) ruff format --check $(RUFF_TARGETS)
	$(RUN) ruff check $(RUFF_TARGETS)
	$(RUN) ty check --exit-zero-on-warning $(RUFF_TARGETS)
	$(RUN) python -m penny.database.migrate --validate
	$(RUN) pytest $(PYTEST_ARGS)
	$(TEAM_RUN) ruff format --check $(TEAM_RUFF_TARGETS)
	$(TEAM_RUN) ruff check $(TEAM_RUFF_TARGETS)
	$(TEAM_RUN) ty check --exit-zero-on-warning $(TEAM_RUFF_TARGETS)
	$(TEAM_RUN) pytest $(TEAM_PYTEST_ARGS)
	cd browser && npm install --silent && npx tsc --noEmit

pytest: $(if $(LOCAL),,build team-build)
	$(RUN) pytest $(PYTEST_ARGS)
	$(TEAM_RUN) pytest $(TEAM_PYTEST_ARGS)

# Live-model contract suite — drives the REAL agents against a running Ollama
# (gpt-oss + embeddinggemma) on synthetic seeds. Slow and stochastic, so it's
# kept out of make check; run it by hand to validate prompt/behaviour changes.
# Forwards the model endpoint into the container (defaulting to the docker host,
# where Ollama runs); override LLM_MODEL / LLM_EMBEDDING_MODEL / EVAL_SAMPLES on
# the host to taste, e.g. `EVAL_SAMPLES=2 make eval`.
eval: $(if $(LOCAL),,build)
	$(RUN) env \
		LLM_API_URL="$${LLM_API_URL:-http://host.docker.internal:11434}" \
		LLM_MODEL="$${LLM_MODEL:-gpt-oss:20b}" \
		LLM_EMBEDDING_MODEL="$${LLM_EMBEDDING_MODEL:-embeddinggemma}" \
		EVAL_SAMPLES="$${EVAL_SAMPLES:-5}" \
		pytest $(EVAL_PYTEST_ARGS)

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
