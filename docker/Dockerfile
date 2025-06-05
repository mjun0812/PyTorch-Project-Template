# syntax=docker/dockerfile:1

ARG BUILDER_IMAGE="nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04"
ARG RUNNER_IMAGE="ubuntu:22.04"

FROM $BUILDER_IMAGE AS builder

ARG PWD

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR="/opt/python" \
    UV_PROJECT_ENVIRONMENT="/opt/venv" \
    # インストール後にソースファイルをバイトコードにコンパイルする
    UV_COMPILE_BYTECODE=1 \
    # wheelからパッケージをコピーしてインストールする
    UV_LINK_MODE=copy \
    # 標準出力・標準エラーのストリームのバッファリングを行わない
    PYTHONUNBUFFERED=1 \
    # .pycファイルを生成しない
    PYTHONDONTWRITEBYTECODE=1

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR ${PWD}
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    ulimit -n 8192 \
    && uv sync --locked --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

FROM $RUNNER_IMAGE AS production

COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=ja_JP.UTF-8 \
    TZ=Asia/Tokyo \
    PATH="/opt/venv/bin:$PATH" \
    UV_PYTHON_INSTALL_DIR="/opt/python" \
    UV_PROJECT_ENVIRONMENT="/opt/venv" \
    # 標準出力・標準エラーのストリームのバッファリングを行わない
    PYTHONUNBUFFERED=1 \
    # .pycファイルを生成しない
    PYTHONDONTWRITEBYTECODE=1

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
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
    tzdata

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && locale-gen ja_JP.UTF-8 \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && echo "Asia/Tokyo" > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata
COPY ./docker/zshrc /etc/zsh/zshrc
COPY --chmod=755 ./docker/entrypoint.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
