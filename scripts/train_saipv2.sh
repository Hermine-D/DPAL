#!/bin/bash
DEBUG_MODE=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG_MODE=1; shift ;;
        *) echo "Unknown parameter: $1"; shift ;;
    esac
done

output_dir="work_dirs/hd1m_PAHTl_vit_tiny_clc"
if [ $DEBUG_MODE -eq 1 ]; then
    echo "Running in debug mode"
    export CUDA_VISIBLE_DEVICES=0
    NPROC_PER_NODE=1
    DEBUG_ARG="--debug"
    output_dir="work_dirs/debug"
else
    echo "Running in normal mode"
    export CUDA_VISIBLE_DEVICES=9
    NPROC_PER_NODE=1
    DEBUG_ARG=""
fi

# output_dir="work_dirs/debug"
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi
script_path=$(realpath "$0")
cp "$script_path" "$output_dir/"
cp "main_align_pretrain_saipv2.py" "$output_dir/"
echo "start training"
export NCCL_P2P_DISABLE=1
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port=29601 main_align_pretrain_saipv2.py $DEBUG_ARG \
  --batch_size=256 --accum_iter=1 \
  --model=saip_kd_vit_tiny_patch16_pathlarge \
  --student_pretrained="" \
  --data_path=data/HD1M \
  --norm_pix_loss \
  --mask_ratio=0.75 \
  --epochs=100 \
  --blr=2.5e-4 \
  --weight_decay=0.05 \
  --warmup_epochs=10 \
  --height=224 \
  --width=224 \
  --crop_height=128 \
  --crop_width=96 \
  --global_crops_scale 0.8 1. \
  --local_crops_scale 0.05 0.8 \
  --output_dir=$output_dir \
  --log_dir=$output_dir \
  --teacher_model=expert_path_large \
  --teacher_pretrained='pretrained_models/PATH/PATHl.pth' \
  --local_crops_number=2 \
  --start_epoch=89 \
  
## path ####
  ###  --model=saip_kd_vit_tiny_patch16_pathlarge \
  ##  --teacher_model=expert_path_large \
  #  --teacher_pretrained='pretrained_models/PATH/PATHl.pth' \


## dinov2 ##
  ### --model=saip_kd_vit_tiny_patch16_dinov2 \
  ## --teacher_model=expert_dinov2_large \
  ## --teacher_pretrained='pretrained_models/dinov2/dinov2_vitl14_pretrain.pth' \
'''
  