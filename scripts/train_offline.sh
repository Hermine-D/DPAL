#!/bin/bash
DEBUG_MODE=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG_MODE=1; shift ;;
        *) echo "Unknown parameter: $1"; shift ;;
    esac
done

output_dir="work_dirs/hd1m_sapiens0.3b_vit_tiny_offline_512x384"
if [ $DEBUG_MODE -eq 1 ]; then
    echo "Running in debug mode"
    export CUDA_VISIBLE_DEVICES=4
    NPROC_PER_NODE=1
    DEBUG_ARG="--debug"
    output_dir="work_dirs/debug"
else
    echo "Running in normal mode"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7,8
    NPROC_PER_NODE=8
    DEBUG_ARG=""
fi

# output_dir="work_dirs/debug"
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi
script_path=$(realpath "$0")
cp "$script_path" "$output_dir/"

export NCCL_P2P_DISABLE=1
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port=29501 main_align_sapiens_pretrain_offline.py $DEBUG_ARG \
  --batch_size=64 --accum_iter=1 \
  --model=saip_kd_vit_tiny_patch16_sapiens_0_3b \
  --data_path=/mnt/nfs/HAG/wangxuanhan/datasets/preprocessed_theia_data \
  --feature_models=sapiens_0.3b_coco_best_coco_AP_epoch_98_512x384 \
  --norm_pix_loss \
  --mask_ratio=0.75 \
  --epochs=100 \
  --blr=2.5e-4 \
  --weight_decay=0.05 \
  --warmup_epochs=10 \
  --height=512 \
  --width=384 \
  --crop_height=128 \
  --crop_width=96 \
  --global_crops_scale 1. 1. \
  --local_crops_scale 0.05 0.8 \
  --output_dir=$output_dir \
  --log_dir=$output_dir \
  --teacher_model=expert_sapiens_0_3b \
  --teacher_pretrained=pretrained_models/sapiens/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_epoch_98_512x384.pth \
  --local_crops_number=2 \
  --start_epoch=0 \
  --num_workers=16 \
  --shuffle \
  --shuffle_buffer_size=1024 \
  # --feature_norm \

# python -m torch.distributed.launch --nproc_per_node 8 main_saip_plus_pretrain.py \
#   --batch_size 128 --accum_iter 2 \
#   --model csl_kd_vit_tiny_patch16 --data_path data/lupsub-1m \
#   --norm_pix_loss \
#   --mask_ratio 0.75 \
#   --epochs 100 --blr 2.5e-4 --weight_decay 0.05 --warmup_epochs 10 \
#   --height 256 --width 128 --crop_height 128 --crop_width 64 \
#   --global_crops_scale 0.8 1. \
#   --local_crops_scale 0.05 0.8 \
#   --output_dir work_dirs/luplm_csl_expert_mae_vit_b_100e_vit_tiny \
#   --log_dir work_dirs/luplm_csl_expert_mae_vit_b_100e_vit_tiny \
#   --teacher_model expert_vit_base --teacher_pretrained pretrained_models/mae_pretrain_vit_base.pth --local_crops_number 8 --start_epoch 13