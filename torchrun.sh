#!/bin/zsh
NUM_GPU=$1
shift
OMP_NUM_THREADS=1 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=${NUM_GPU} "$@" gpu.multi=true
