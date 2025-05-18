#!/bin/bash
DEBUG_MODE=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG_MODE=1; shift ;;
        *) echo "Unknown parameter: $1"; shift ;;
    esac
done

output_dir="work_dirs/lup1m_path-b_to_vit_tiny_cross_from_cls_patch"
# output_dir="work_dirs/debug"
if [ $DEBUG_MODE -eq 1 ]; then
    echo "Running in debug mode"
    export CUDA_VISIBLE_DEVICES=4
    NPROC_PER_NODE=1
    DEBUG_ARG="--debug"
    output_dir="work_dirs/debug"
else
    echo "Running in normal mode"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    NPROC_PER_NODE=8
    DEBUG_ARG=""
fi

# output_dir="work_dirs/debug"
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi
script_path=$(realpath "$0")
cp "$script_path" "$output_dir/"
cp "main_align_pretrain_v2.py" "$output_dir/"
echo "start training"
# export NCCL_P2P_DISABLE=1
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port=29602 main_align_pretrain_v2.py $DEBUG_ARG \
  --batch_size=256 --accum_iter=1 \
  --model=saipv1_kd_vit_tiny_patch16_cross_atten \
  --data_path=data/LUP1M \
  --norm_pix_loss \
  --mask_ratio=0.75 \
  --epochs=200 \
  --blr=2.5e-4 \
  --weight_decay=0.05 \
  --warmup_epochs=10 \
  --height=256 \
  --width=128 \
  --crop_height=128 \
  --crop_width=64 \
  --global_crops_scale 0.8 1. \
  --local_crops_scale 0.05 0.8 \
  --output_dir=$output_dir \
  --log_dir=$output_dir \
  --teacher_model=expert_vit_base \
  --teacher_pretrained='pretrained_models/humanbench_vit_base.pth' \
  --local_crops_number=6 \
  --start_epoch=0 \
  