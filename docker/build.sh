#!/bin/bash

cd $(dirname $0)
cd ../
IMAGE_NAME=$(basename $(pwd))
# to lowercase
IMAGE_NAME=$(echo $IMAGE_NAME | tr '[:upper:]' '[:lower:]')

PYTHON="3.10.7"
CUDA_VERSION="11.6.2"
UBUNTU="20.04"
CUDNN="8"
SUDO_PASSWORD=$USER

DESCRIPTION=$(cat <<< "CUDA + Python Docker
同階層にpoetry, requirements.txtを置くと自動でパッケージがインストールされます．

Option:
    -p, --python: python version. default to $PYTHON 
    -c, --cuda:   CUDA Version. default to $CUDA_VERSION
    -u, --ubuntu: Ubuntu Version. default to $UBUNTU
    --cudnn:      CUDNN Version. default to $CUDNN
    --sudo-pass:  SUDO Password in container. default to '$SUDO_PASSWORD' (username)"
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
        '-u' | '--ubuntu')
            UBUNTU="$2"
            shift 2
        ;;
        '--cudnn')
            CUDNN="$2"
            shift 2
        ;;
        '--sudo-pass')
            SUDO_PASSWORD="$2"
            shift 2
        ;;
    esac
done

USER_ID=`id -u`
GROUP_ID=`id -g`
GROUP_NAME=`id -gn`
USER_NAME=$USER

# get-pipは，3.7以前と後でスクリプトの場所が変わる
GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"
if [[ "$PYTHON" == *3.6* ]]; then
    GET_PIP_URL="https://bootstrap.pypa.io/pip/3.6/get-pip.py"
elif [[ "$PYTHON" == *3.5* ]]; then
    GET_PIP_URL="https://bootstrap.pypa.io/pip/3.5/get-pip.py"
fi

docker build \
    --build-arg PYTHON=${PYTHON} \
    --build-arg USER_NAME=${USER_NAME} \
    --build-arg USER_ID=${USER_ID} \
    --build-arg GROUP_ID=${GROUP_ID} \
    --build-arg GROUP_NAME=${GROUP_NAME} \
    --build-arg CUDA_VERSION=${CUDA_VERSION} \
    --build-arg SUDO_PASSWORD=${SUDO_PASSWORD} \
    --build-arg CUDNN=${CUDNN} \
    --build-arg UBUNTU=${UBUNTU} \
    --build-arg GET_PIP_URL=${GET_PIP_URL} \
    -t "${USER_NAME}/${IMAGE_NAME}-server:latest" \
    -f docker/Dockerfile .

