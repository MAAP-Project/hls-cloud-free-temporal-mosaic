FROM ghcr.io/astral-sh/uv:0.9.17-python3.13-bookworm-slim@sha256:4f0bb0bed02fc05d6361b4ed2c37cddd40ce6478abb222a2f30036e0888f6db5

WORKDIR /app/hls-cloud-free-temporal-mosaic

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY main.py run.sh ./
RUN chmod +x run.sh
