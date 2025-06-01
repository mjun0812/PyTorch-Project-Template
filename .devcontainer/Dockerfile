# syntax=docker/dockerfile:1
ARG PYTHON="3.11"
FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/python:${PYTHON}
ARG PYTHON="3.11"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=ja_JP.UTF-8 \
    TZ=Asia/Tokyo \
    PATH="/.venv/bin:$PATH" \
    UV_PYTHON_INSTALL_DIR="/opt/python" \
    UV_PYTHON=${PYTHON} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    jq \
    ca-certificates \
    build-essential

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
