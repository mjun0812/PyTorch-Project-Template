#!/bin/bash

cd $(dirname $0)
cd ../
IMAGE_NAME=$(basename $(pwd))
pwd

docker run \
    -it \
    --gpus all \
    --rm \
    --shm-size=128g \
    --env DISPLAY=$DISPLAY \
    --volume $HOME/.Xauthority:$HOME/.Xauthority:rw \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --volume $HOME/.cache:$HOME/.cache \
    --volume "$(dirname $(pwd))/dataset:/home/${USER}/dataset" \
    --volume "$(pwd)/result:/home/${USER}/workspace/result" \
    --name "${IMAGE_NAME}-$(date '+%s')" \
    "${USER}/${IMAGE_NAME}-server:latest" \
    $@

