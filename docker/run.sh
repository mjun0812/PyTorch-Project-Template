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

SYMLINK_MOUNTS=""
# datasetディレクトリ以下のシンボリックリンクを探し、リンク先をマウントする
for symlink in $(find "$(pwd)/dataset" -type l); do
    target=$(dirname "$symlink")/$(readlink "$symlink")
    [ -e "$target" ] && SYMLINK_MOUNTS+=" -v $target:$target"
done

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
    -v $HOME/.Xauthority:$HOME/.Xauthority:rw \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.cache:$HOME/.cache \
    -v "$(pwd):$(pwd)" \
    -v "$(pwd)/dataset:$(pwd)/dataset" \
    -v "$(pwd)/result:$(pwd)/result" \
    -v "$(pwd)/model_zoo:$(pwd)/model_zoo" \
    $SYMLINK_MOUNTS \
    --workdir $(pwd) \
    --name "${IMAGE_NAME}-$(date '+%s')" \
    "${USER}/${IMAGE_NAME}-server:latest" \
    ${@:-zsh}
