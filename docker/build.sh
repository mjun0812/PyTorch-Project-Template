#!/bin/bash

cd $(dirname $0)
cd ../

PYTHON="3.11"
CUDA_VERSION="12.4.0"

DESCRIPTION=$(cat <<< "CUDA + Python Docker
同階層にpoetry, requirements.txtを置くと自動でパッケージがインストールされます．

Option:
    -p, --python: python version. default to $PYTHON 
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
BUILDER_IMAGE="nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04"
BASE_IMAGE="nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04"

mkdir -p result

docker build \
    --build-arg BUILDER_IMAGE=${BUILDER_IMAGE} \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --build-arg PYTHON=${PYTHON} \
    -t "${USER}/${IMAGE_NAME}-server:latest" \
    -f docker/Dockerfile .
