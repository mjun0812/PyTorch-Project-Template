#!/bin/bash

cd $(dirname $0)
cd ../
IMAGE_NAME=$(basename $(pwd))
# to lowercase
IMAGE_NAME=$(echo $IMAGE_NAME | tr '[:upper:]' '[:lower:]')

PYTHON="3.11"
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
        '-p' | '--python')
            PYTHON="$2"
            shift 2
        ;;
        '-c' | '--cuda')
            CUDA_VERSION="$2"
            shift 2
        ;;
    esac
done

docker pull ghcr.io/mjun0812/cuda${CUDA_VERSION}-python${PYTHON}-runtime-server

docker build \
    --build-arg PYTHON=${PYTHON} \
    --build-arg CUDA_VERSION=${CUDA_VERSION} \
    -t "${USER}/${IMAGE_NAME}-server:latest" \
    -f docker/Dockerfile .

