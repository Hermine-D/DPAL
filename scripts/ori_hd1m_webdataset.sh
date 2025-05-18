NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 \
python preprocessing_theia/organize_hd1m_webdataset.py \
  --dataset=HD1M \
  --output-path=/mnt/nfs/HAG/wangxuanhan/datasets/preprocessed_theia_data/ \
  --imagenet-raw-path=data/HD1M \
  --split=train \
