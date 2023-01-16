#!/bin/bash

if [[ $# -eq 0 ]]; then
  exec "/usr/bin/zsh"
else
  exec "$@"
fi

