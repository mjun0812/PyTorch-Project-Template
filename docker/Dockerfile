# syntax=docker/dockerfile:1

ARG BUILDER_IMAGE="nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04"
ARG BASE_IMAGE="nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04"

FROM $BUILDER_IMAGE AS builder
LABEL maintainer="mjun"

ARG PYTHON
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
# rye setting
ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"

# apt packageのミラー先を日本に変更して高速化
# --no-install-recommends: 依存を最小限にしてinstall
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    curl \
    libgl1-mesa-dev

RUN curl -sSf https://rye.astral.sh/get | RYE_NO_AUTO_INSTALL=1 RYE_TOOLCHAIN_VERSION=$PYTHON RYE_INSTALL_OPTION="--yes" bash && \
    rye config --set-bool behavior.global-python=true && \
    rye config --set-bool behavior.use-uv=true

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=requirements.lock,target=requirements.lock \
    --mount=type=bind,source=requirements-dev.lock,target=requirements-dev.lock \
    rye pin $PYTHON && \
    rye sync --no-lock

FROM $BASE_IMAGE AS production
COPY --from=builder /opt/rye /opt/rye
COPY --from=builder /.venv /.venv

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=ja_JP.UTF-8
ENV TZ=Asia/Tokyo
ENV PATH="/.venv/bin:$PATH"
ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"
# 標準出力・標準エラーのストリームのバッファリングを行わない
ENV PYTHONUNBUFFERED=1
# .pycファイルを生成しない
ENV PYTHONDONTWRITEBYTECODE=1

RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list
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
    gosu \
    ninja-build \
    curl \
    libgl1-mesa-dev

RUN locale-gen ja_JP.UTF-8 \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

COPY ./docker/zshrc /etc/zsh/zshrc
COPY --chmod=755 ./docker/entrypoint.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
