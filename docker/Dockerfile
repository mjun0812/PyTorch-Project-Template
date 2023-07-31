ARG CUDA_VERSION
ARG PYTHON

FROM ghcr.io/mjun0812/cuda${CUDA_VERSION}-python${PYTHON}-server:latest

# 環境変数 Versionなど
LABEL maintainer="mjun"

ARG POETRY_VERSION="1.5.1"
ENV POETRY_HOME=/opt/poetry
ENV PATH=$POETRY_HOME/bin:$PATH

# pip install from requirements.txt
COPY ./poetry.* ./pyproject.toml* ./
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python - \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi
