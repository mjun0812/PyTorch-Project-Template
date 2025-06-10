#!/bin/bash

cd $(dirname $0)
cd ../

IMAGE_NAME=$(basename $(pwd))
IMAGE_NAME=$(echo $IMAGE_NAME | tr '[:upper:]' '[:lower:]')
USER_ID=`id -u`
GROUP_ID=`id -g`
GROUP_NAME=`id -gn`
USER_NAME=$USER
PWD=$(pwd)

# Use GPU if available
GPU_OPTION=""
if docker system info | grep -qE '^\s*Runtimes: .*nvidia.*'; then
    GPU_OPTION="--gpus all"
fi

# Check if TTY is available (not in CI environment)
TTY_OPTION=""
if [ -t 0 ] && [ -t 1 ]; then
    TTY_OPTION="-it"
fi

# Mount symlinks in dataset directory
SYMLINK_MOUNTS=""
for symlink in $(find "${PWD}/dataset" -type l); do
    target=$(dirname "$symlink")/$(readlink "$symlink")
    [ -e "$target" ] && SYMLINK_MOUNTS+=" -v $target:$target"
done

docker run \
    $TTY_OPTION \
    $GPU_OPTION \
    --rm \
    --shm-size=128g \
    --hostname $(hostname) \
    --net host \
    --ipc=host \
    --ulimit memlock=-1 \
    --env DISPLAY=$DISPLAY \
    --env USER_NAME=$USER_NAME \
    --env USER_ID=$USER_ID \
    --env GROUP_NAME=$GROUP_NAME \
    --env GROUP_ID=$GROUP_ID \
    -v $HOME/.Xauthority:$HOME/.Xauthority:rw \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.cache:$HOME/.cache \
    -v ${PWD}:${PWD} \
    -v ${PWD}/dataset:${PWD}/dataset \
    -v ${PWD}/result:${PWD}result \
    $SYMLINK_MOUNTS \
    --workdir ${PWD} \
    --name "${IMAGE_NAME}-$(date '+%s')" \
    "${USER}/${IMAGE_NAME}-server:latest" \
    ${@:-zsh}
