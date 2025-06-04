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

USE_QUEUE="-i"
USE_JUPYTER=""
USE_MLFLOW_UI=""

for OPT in "$@"; do
    case $OPT in
        '-q' | '--queue')
            USE_QUEUE=""
            shift 1;
        ;;
        '--jupyter')
            USE_JUPYTER="-p 38888:38888"
            shift 1;
        ;;
        '--mlflow-ui')
            USE_MLFLOW_UI="-p 38880:38880"
            shift 1;
        ;;
    esac
done

GPU_OPTION=""
if type nvcc > /dev/null 2>&1; then
    # Use GPU
    GPU_OPTION="--gpus all"
fi

# datasetディレクトリ以下のシンボリックリンクを探し、リンク先をマウントする
SYMLINK_MOUNTS=""
for symlink in $(find "${PWD}/dataset" -type l); do
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
    --net host \
    --ipc=host \
    --ulimit memlock=-1 \
    $USE_JUPYTER \
    $USE_MLFLOW_UI \
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
