#!/bin/zsh

NUM_NODES=$1
NUM_GPU_PER_NODE=$2
RANK_0_HOST=$3
RANK_0_PORT=$4
shift 4
JOB_ID=$RANDOM
OMP_NUM_THREADS=1 torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${NUM_GPU_PER_NODE} \
    --rdzv_id=${JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${RANK_0_HOST}:${RANK_0_PORT} \
    "$@" gpu.multi=true
