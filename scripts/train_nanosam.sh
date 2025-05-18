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
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port=29502 nanosam.py $DEBUG_ARG \
    --images=/mnt/hdd1/wangxuanhan/datasets/LUP-subset/images \
    --output_dir=work_dirs/hd1m_repa_align_expert_sapiens_100e_vit_tiny_train_offline_nano \
    --model_name=resnet18 \
    --teacher_image_encoder_engine==models/sapiens_0.3b/sapiens_image_encoder_bs16.engine \
    --batch_size=128 \
    --num_workers=8
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