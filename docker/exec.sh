#!/bin/bash

DESCRIPTION=$(cat <<< "CUDA + Python Docker
同階層にpoetry, requirements.txtを置くと自動でパッケージがインストールされます

Option:
    -v, --volume: Mount Volume. default is $PYTHON
    -c, --cuda:   CUDA Version. default is $CUDA_VERSION
    -u, --ubuntu: Ubuntu Version. default is $UBUNTU
    --cudnn:      CUDNN Version. default is $CUDNN
    --prefix:     Set Container Name Prefix. Ex.hoge-cuda112-server-[prefix]
    -d: Detach(background) run."
)

CUDA_VERSION="11.3.1"
UBUNTU="20.04"
CUDNN="8"
CONTAINER_NAME_PREFIX=""

VOLUME="${HOME}/workspace"

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
        '-u' | '--ubuntu')
            UBUNTU="$2"
            shift 2
        ;;
        '--cudnn')
            CUDNN="$2"
            shift 2
        ;;
        '-v' | '--volume')
            VOLUME="$2"
            shift 2
        ;;
        '-d' | '--detach')
            DETACH='-d'
            shift 2
        ;;
        '--prefix')
            CONTAINER_NAME_PREFIX="-$2"
            shift 2
        ;;
    esac
done

USER_ID=`id -u`
GROUP_ID=`id -g`
GROUP_NAME=`id -gn`
USER_NAME=$USER

# 同じコンテナ名が存在しているか確認
CONTAINER_NAME="${USER_NAME}-cuda${CUDA_VERSION//./}-server${CONTAINER_NAME_PREFIX}"
if [ -z `docker ps -aq -f name=${CONTAINER_NAME}` ]; then
    docker run \
        -it \
        --gpus all \
        --net host \
        --rm \
        $DETACH \
        --shm-size=128g \
        --env DISPLAY=$DISPLAY \
        --volume $HOME/.Xauthority:$HOME/.Xauthority:rw \
        --volume $HOME/.cache:$HOME/.cache \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --volume $VOLUME:$HOME/workspace \
        --name "${CONTAINER_NAME}" \
        "${USER_NAME}/cuda${CUDA_VERSION//./}-server:latest"
else
    cat <<< "Your container(name: ${CONTAINER_NAME}) is already existed.

1. attach existing container:
    > docker attach ${CONTAINER_NAME}
2. remove existing container:
    > docker container rm ${CONTAINER_NAME}
3. create new container add container name prefix:
    > ./exec.sh ... --prefix hoge"
fi
