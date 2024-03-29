#!/bin/bash

cd $(dirname $0)
cd ../
IMAGE_NAME=$(basename $(pwd))
IMAGE_NAME=$(echo $IMAGE_NAME | tr '[:upper:]' '[:lower:]')
USER_ID=`id -u`
GROUP_ID=`id -g`
GROUP_NAME=`id -gn`
USER_NAME=$USER

USE_QUEUE="-i"

for OPT in "$@"; do
    case $OPT in
        '-q' | '--queue')
            USE_QUEUE=""
            shift 1;
        ;;
    esac
done

GPU_OPTION=""
if type nvcc > /dev/null 2>&1; then
    # Use GPU
    GPU_OPTION="--gpus all"
fi

docker run \
    -t \
    $USE_QUEUE \
    $GPU_OPTION \
    --rm \
    --shm-size=128g \
    --hostname $(hostname) \
    --ipc=host \
    --net=host \
    --ulimit memlock=-1 \
    --env DISPLAY=$DISPLAY \
    --env USER_NAME=$USER_NAME \
    --env USER_ID=$USER_ID \
    --env GROUP_NAME=$GROUP_NAME \
    --env GROUP_ID=$GROUP_ID \
    --volume $HOME/.Xauthority:$HOME/.Xauthority:rw \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --volume $HOME/.cache:$HOME/.cache \
    --volume "$(pwd):$(pwd)" \
    --volume "$(pwd)/dataset:$(pwd)/dataset" \
    --volume "$(pwd)/result:$(pwd)/result" \
    --volume "$(pwd)/model_zoo:$(pwd)/model_zoo" \
    --workdir $(pwd) \
    --name "${IMAGE_NAME}-$(date '+%s')" \
    "${USER}/${IMAGE_NAME}-server:latest" \
    ${@:-zsh}
