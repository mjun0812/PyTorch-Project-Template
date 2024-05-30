ARG BASE_IMAGE
# Fromの前後でARGのスコープが変わるので注意
# 上のARGはFROMより前に書くこと
FROM $BASE_IMAGE
LABEL maintainer="mjun"

# gitをインストールするときに必要 tzdataをinstallされてしまうため
ENV DEBIAN_FRONTEND=noninteractive
# ロケールを日本語UTF-8に設定
ENV LANG ja_JP.UTF-8
ENV TZ Asia/Tokyo
# 標準出力・標準エラーのストリームのバッファリングを行わない
ENV PYTHONUNBUFFERED=1
# rye setting
ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"

# apt packageのミラー先を日本に変更して高速化
# --no-install-recommends: 依存を最小限にしてinstall
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    zsh \
    locales \
    sudo \
    gosu \
    vim \
    emacs \
    gosu \
    ninja-build \
    curl \
    libgl1-mesa-dev \
    # ロケールを日本語UTF-8に設定
    # これによりDocker環境下で日本語入力が可能となる
    && locale-gen ja_JP.UTF-8 \
    # clean apt-get cache
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/* \
    # exec sudo commannd with no password
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

COPY ./docker/zshrc /etc/zsh/zshrc

RUN curl -sSf https://rye.astral.sh/get | RYE_NO_AUTO_INSTALL=1 RYE_INSTALL_OPTION="--yes" bash && \
    rye config --set-bool behavior.global-python=true && \
    rye config --set-bool behavior.use-uv=true

# Required for rye
COPY ./.python-version ./pyproject.toml ./requirements* ./README.md ./
RUN rye pin "$(cat .python-version)" && \
    rye sync

ENV PATH="/.venv/bin:$PATH"
COPY --chmod=755 ./docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
