# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ARG PYTHON="3.11"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=ja_JP.UTF-8 \
    TZ=Asia/Tokyo \
    PATH="/.venv/bin:$PATH" \
    UV_PYTHON_INSTALL_DIR="/opt/python" \
    UV_PYTHON=${PYTHON} \
    # 標準出力・標準エラーのストリームのバッファリングを行わない
    PYTHONUNBUFFERED=1 \
    # .pycファイルを生成しない
    PYTHONDONTWRITEBYTECODE=1

# apt packageのミラー先を日本に変更して高速化
# --no-install-recommends: 依存を最小限にしてinstall
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    zsh \
    locales \
    sudo \
    gosu \
    vim \
    ninja-build \
    curl \
    ca-certificates \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    build-essential \
    && locale-gen ja_JP.UTF-8

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen \
    && rm -rf /root/.cache/uv
