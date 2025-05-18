NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7,8 \
python -m torch.distributed.launch --nproc_per_node 8 --master_port=29501  main_align_sapiens_pretrain_large2small.py \
  --batch_size=128 --accum_iter=2 \
  --model=saip_kd_vit_tiny_patch16_sapiens_0_3b_large2small \
  --data_path=data/HD1M \
  --norm_pix_loss \
  --mask_ratio=0.75 \
  --epochs=100 \
  --blr=2.5e-4 \
  --weight_decay=0.05 \
  --warmup_epochs=10 \
  --height=256 \
  --width=192 \
  --crop_height=128 \
  --crop_width=96 \
  --global_crops_scale 0.8 1. \
  --local_crops_scale 0.05 0.8 \
  --output_dir=work_dirs/hd1m_repa_align_expert_sapiens_100e_vit_tiny_posemb224_multiscale_train_large2small \
  --log_dir=work_dirs/hd1m_repa_align_expert_sapiens_100e_vit_tiny_posemb224_multiscale_train_large2small \
  --teacher_model=expert_sapiens_0_3b \
  --teacher_pretrained=pretrained_models/sapiens/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth \
  --local_crops_number=2 \
  --start_epoch=0 \

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