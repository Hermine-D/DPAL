NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7,8 \
python -m torch.distributed.launch --nproc_per_node=8  preprocessing/feature_extraction.py \
  --model_path='pretrained_models/sapiens/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth' \
  --image_folder='data/HD1M' \
  --output_folder='/mnt/hdd4/wangxuanhan/datasets/preprocessed_HD1M' \
  --batch_size=8