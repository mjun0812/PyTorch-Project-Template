#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
TARGET_DIR=$SCRIPT_DIR/../result
cd $TARGET_DIR
mlflow ui --host 0.0.0.0 --port 38880
