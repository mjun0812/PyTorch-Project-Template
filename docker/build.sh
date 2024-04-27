#!/bin/bash

cd $(dirname $0)
cd ../
IMAGE_NAME=$(basename $(pwd))
# to lowercase
IMAGE_NAME=$(echo $IMAGE_NAME | tr '[:upper:]' '[:lower:]')

CUDA_VERSION="12.1.1"

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

docker pull nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04

docker build \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    -t "${USER}/${IMAGE_NAME}-server:latest" \
    -f docker/Dockerfile .
