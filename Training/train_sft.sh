#!/bin/bash

set -x
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export WANDB_MODE=offline
# export CUDA_VISIBLE_DEVICES=4,5,6,7
NNODES=${NNODES:=1}
# NPROC_PER_NODE=${NPROC_PER_NODE:=$(nvidia-smi --list-gpus | wc -l)}
NPROC_PER_NODE=2
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}

if [[ "$WORLD_SIZE" == "1" ]]; then
    additional_args="$additional_args --standalone"
else
    additional_args="--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi
echo $additional_args
torchrun \
    --nnodes=$NNODES \
    --nproc-per-node=$NPROC_PER_NODE \
    --node-rank=$NODE_RANK \
    $additional_args \
    tasks/omni/train_cheers.py \
    configs/multimodal/cheers/und_gen_train/sft.yaml \
    "$@" \
    2>&1 | tee log.txt