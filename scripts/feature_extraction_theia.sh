#!/bin/bash
DEBUG_MODE=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG_MODE=1; shift ;;
        *) echo "Unknown parameter: $1"; shift ;;
    esac
done

if [ $DEBUG_MODE -eq 1 ]; then
    echo "Running in debug mode"
    export CUDA_VISIBLE_DEVICES=0
    NPROC_PER_NODE=1
    DEBUG_ARG="--debug"
else
    echo "Running in normal mode"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7,8
    NPROC_PER_NODE=8
    DEBUG_ARG=""
fi

export NCCL_P2P_DISABLE=1

python preprocessing_theia/feature_extraction.py $DEBUG_ARG \
  --dataset=HD1M \
  --dataset-root=/mnt/nfs/HAG/wangxuanhan/datasets/preprocessed_theia_data \
  --output-path=/mnt/nfs/HAG/wangxuanhan/datasets/preprocessed_theia_data \
  --model=pretrained_models/sapiens/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_epoch_98_512x384.pth \
  --split=train \
  --num-gpus=8 \
  --batch-size=100