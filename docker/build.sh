#!/bin/bash

cd $(dirname $0)
cd ../

CUDA_VERSION="12.8.1"

DESCRIPTION=$(cat <<< "PyTorch Project Docker
Option:
    -c, --cuda:   CUDA Version. default to $CUDA_VERSION"
)
for OPT in "$@"; do
    case $OPT in
        '-h' | '--help')
            echo "$DESCRIPTION"
            exit 0;
        ;;
        '-c' | '--cuda')
            CUDA_VERSION="$2"
            shift 2
        ;;
    esac
done

IMAGE_NAME=$(basename $(pwd))
IMAGE_NAME=$(echo $IMAGE_NAME | tr '[:upper:]' '[:lower:]')
BUILDER_IMAGE="nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04"
RUNNER_IMAGE="ubuntu:22.04"

mkdir -p result

docker build \
    --build-arg BUILDER_IMAGE=${BUILDER_IMAGE} \
    --build-arg RUNNER_IMAGE=${RUNNER_IMAGE} \
    --build-arg PWD=$(pwd) \
    -t "${USER}/${IMAGE_NAME}-server:latest" \
    -f docker/Dockerfile .
