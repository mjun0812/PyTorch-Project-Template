#!/bin/zsh

# Usage: ./multinode.sh <NUM_NODES> <NUM_GPU_PER_NODE> <JOB_ID> <NODE_RANK> <RANK_0_HOST> <OTHER_ARGS>

NUM_NODES=$1
NUM_GPU_PER_NODE=$2
JOB_ID=$3
NODE_RANK=$4
RANK_0_HOST=$5
shift 5

export OMP_NUM_THREADS=1

torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${NUM_GPU_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --rdzv_id=${JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${RANK_0_HOST} \
    "$@" gpu.multi=true
