ARG CUDA_VERSION
ARG PYTHON

FROM ghcr.io/mjun0812/cuda${CUDA_VERSION}-python${PYTHON}-runtime-server:latest

# 環境変数 Versionなど
LABEL maintainer="mjun"

ARG POETRY_VERSION="1.7.1"
ENV POETRY_HOME=/opt/poetry
ENV PATH=$POETRY_HOME/bin:$PATH

# pip install from requirements.txt
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python - \
    && poetry config virtualenvs.create false
COPY ./poetry.* ./pyproject.toml* ./
RUN poetry install --no-interaction --no-ansi --no-root
