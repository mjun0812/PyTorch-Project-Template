#!/bin/bash

cd $(dirname $0)
cd ../
IMAGE_NAME=$(basename $(pwd))
IMAGE_NAME=$(echo $IMAGE_NAME | tr '[:upper:]' '[:lower:]')

docker run \
    -it \
    --gpus all \
    --rm \
    --shm-size=128g \
    --hostname $(hostname) \
    --env DISPLAY=$DISPLAY \
    --entrypoint "./docker/entrypoint.sh" \
    --volume $HOME/.Xauthority:$HOME/.Xauthority:rw \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --volume $HOME/.cache:$HOME/.cache \
    --volume "$(dirname $(pwd))/dataset:$(dirname $(pwd))/dataset" \
    --volume "$(pwd):$(pwd)" \
    --volume "$(pwd)/result:$(pwd)/result" \
    --workdir $(pwd) \
    --name "${IMAGE_NAME}-$(date '+%s')" \
    "${USER}/${IMAGE_NAME}-server:latest" \
    $@
