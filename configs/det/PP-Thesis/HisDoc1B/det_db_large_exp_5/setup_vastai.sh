#!/bin/bash

cd /workspace/

git clone https://github.com/chulanpro5/PaddleOCR-chulanpro.git
# Navigate to the workspace directory
cd /workspace/PaddleOCR-chulanpro/

pip install -r requirements.txt

# Hugging Face dataset
huggingface-cli download PPThes/datasets \
  --repo-type dataset \
  --include "paddle_det_v2_100k_zip/*" \
  --local-dir ./train_data/

huggingface-cli download PPThes/datasets \
  --repo-type dataset \
  --include "paddle_det_v2_eval_500_zip/*" \
  --local-dir ./train_data/


# # Move data to fit with configuration
# mv  ./train_data/rec/paddle_v5_ss1/train ./train_data/rec
# mv  ./train_data/rec/paddle_v_ss1/val ./train_data/rec

wandb login 3490b1643699f6bb61887e484c8cfd1aa4cac0e1

# unzip

cd ./train_data/paddle_det_v2_100k_zip/

7z x paddle_det_v2_100k.zip

cd /workspace/PaddleOCR-chulanpro/

cd ./train_data/paddle_det_v2_eval_500_zip/

7z x paddle_det_v2_eval_500.zip

cd /workspace/PaddleOCR-chulanpro/





#python -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7,8,9,10,11' tools/train.py --config configs/rec/PP-Thesis/HisDoc1B/base/rec_svtrv2_base_ch_exp_5.yml -o Global.eval_batch_step="[15408,1926]"
#python -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7' tools/train.py --config configs/rec/PP-Thesis/HisDoc1B/base/rec_svtrv2_base_ch_exp_5.yml -o Global.eval_batch_step="[15408,1926]"	
#python3 -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7,8,9,10,11' --log_dir ./logs tools/train.py -c configs/rec/PP-Thesis/HisDoc1B/base/rec_focalnet_base_ch_exp_5v2.yml -o Train.loader.batch_size_per_card=128 Eval.loader.batch_size_per_card=128

python3 -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7,8,9' --log_dir ./output/det_db_large_exp_5/logs tools/train.py --config configs/det/PP-Thesis/HisDoc1B/det_db_large_exp_5/det_db_large_exp_5.yml

huggingface-cli upload PPThes/checkpoints \
  ./output/det_db_large_exp_5/ \
  ./det_db_large_exp_5/ \
  --repo-type=dataset \
  --token=<token>