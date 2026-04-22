FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    TUBEMIND_DATA_DIR=/app/.local

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY static ./static
COPY tubemind ./tubemind

RUN uv sync --frozen --no-dev

CMD [".venv/bin/python", "-m", "tubemind"]
