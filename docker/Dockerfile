ARG CUDA_VERSION
ARG PYTHON

FROM ghcr.io/mjun0812/cuda${CUDA_VERSION}-python${PYTHON}-server:latest

# 環境変数 Versionなど
LABEL maintainer="mjun"

# pip install from requirements.txt
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
