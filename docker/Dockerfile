# syntax=docker/dockerfile:1

ARG BUILDER_IMAGE="nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04"
ARG BASE_IMAGE="nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04"

FROM $BUILDER_IMAGE AS builder

ARG PYTHON
ENV UV_PYTHON_INSTALL_DIR="/opt/python" \
    UV_PYTHON=${PYTHON} \
    DEBIAN_FRONTEND=noninteractive

# apt packageのミラー先を日本に変更して高速化
# --no-install-recommends: 依存を最小限にしてinstall
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen


FROM $BASE_IMAGE AS production

COPY --from=builder /bin/uv /bin/uv
COPY --from=builder /opt/python /opt/python
COPY --from=builder /.venv /.venv

ARG PYTHON
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
    tzdata

RUN locale-gen ja_JP.UTF-8 \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && echo "Asia/Tokyo" > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata
COPY ./docker/zshrc /etc/zsh/zshrc
COPY --chmod=755 ./docker/entrypoint.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
