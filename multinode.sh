#!/bin/zsh

# Usage: ./multinode.sh <NUM_NODES> <NUM_GPU_PER_NODE> <NODE_RANK> <RANK_0_HOST> <OTHER_ARGS>

NUM_NODES=$1
NUM_GPU_PER_NODE=$2
NODE_RANK=$3
RANK_0_HOST=$4
shift 4
JOB_ID=$RANDOM

export OMP_NUM_THREADS=1

torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${NUM_GPU_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --rdzv_id=${JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${RANK_0_HOST} \
    "$@" gpu.multi=true
