# Build:  docker build -t ml-acopf .
# Run:    docker run --rm ml-acopf list-networks
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

RUN uv run python -c "from ml_acopf.cli import ensure_julia_ready; ensure_julia_ready()"

ENTRYPOINT ["uv", "run", "--no-sync", "ml_acopf"]
CMD ["--help"]