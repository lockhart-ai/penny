# Use Python 3.14 slim image
FROM python:3.14-slim

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
