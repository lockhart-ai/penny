# Use Python 3.14 slim image
FROM python:3.14-slim

# git — scripts/store_git_info.sh (#2891) shells out to it on the HOST before a
# real build, but its own test (test_store_git_info.py) runs INSIDE this image
# under `make check`, building a temp repo to exercise it against.
# sqlite3 — the deploy loop's idle check (scripts/penny-idle.sh, #3026) reads the
# newest promptlog/messagelog timestamps through the sqlite3 CLI, and its test
# (#3026) drives that real file from inside the container too.
RUN apt-get update && apt-get install -y --no-install-recommends git sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Disable __pycache__
ENV PYTHONDONTWRITEBYTECODE=1

# Add /shared to PYTHONPATH for shared github_api module
ENV PYTHONPATH=/shared

# Set working directory
WORKDIR /penny

# Copy dependency files and install dependencies (cached layer)
COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml --group dev

# Accept build args for git info (after deps so changing them doesn't bust the cache)
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}
# The same commit as a LABEL (#2973): the deploy loop's startup check reads it
# back with `docker image inspect` to learn which commit the image tagged
# `penny` was actually built from — the replacement for the old host-side
# `.changelog_cursor`, which a built-but-never-started image advanced for it.
LABEL git_commit=${GIT_COMMIT}

# Copy source
COPY penny/ ./penny/

# Install the package itself
RUN uv pip install --system --no-deps .

# Copy and setup entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run the agent
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "penny.penny"]
