#!/bin/bash

cd $(dirname $0)
cd ../
IMAGE_NAME=$(basename $(pwd))
pwd

docker run \
    -it \
    -d \
    --gpus all \
    --rm \
    --shm-size=128g \
    --volume "$(dirname $(pwd))/dataset:/home/${USER}/dataset" \
    --volume "$(pwd)/result:/home/${USER}/workspace/result" \
    --name "${IMAGE_NAME}" \
    "${USER}/${IMAGE_NAME}-server:latest" \
    $@
