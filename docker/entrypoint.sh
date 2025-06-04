#!/bin/bash

USER_ID=${USER_ID:-1000}
GROUP_ID=${GROUP_ID:-1000}
USER_NAME=${USER_NAME:-user}
GROUP_NAME=${GROUP_NAME:-group}

userdel -r ubuntu > /dev/null 2>&1 # For Ubuntu 24.04 image
groupadd -g ${GROUP_ID} ${GROUP_NAME} > /dev/null 2>&1
useradd -u ${USER_ID} -g ${GROUP_NAME} -G sudo -o -m ${USER_NAME} -s /usr/bin/zsh > /dev/null 2>&1
export HOME=/home/${USER_NAME}
touch $HOME/.zshrc
chown $USER_NAME:$GROUP_NAME $HOME

if [[ $# -eq 0 ]]; then
    exec /usr/sbin/gosu ${USER_NAME} zsh
else
    /usr/sbin/gosu ${USER_NAME} "$@"
fi
