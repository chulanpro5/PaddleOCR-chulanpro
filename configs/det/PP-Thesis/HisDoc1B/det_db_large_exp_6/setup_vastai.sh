#!/bin/bash

cd /workspace/

git clone https://github.com/chulanpro5/PaddleOCR-chulanpro.git
# Navigate to the workspace directory
cd /workspace/PaddleOCR-chulanpro/

pip install -r requirements.txt

# Hugging Face dataset
huggingface-cli download chulanpro5/datasets \
  --repo-type dataset \
  --include "paddle_det_v2_200k_zip/*" \
  --local-dir ./train_data/ \
  --token=<token>

huggingface-cli download chulanpro5/datasets \
  --repo-type dataset \
  --include "paddle_det_v2_eval_500_zip/*" \
  --local-dir ./train_data/ \
  --token=<token>


# # Move data to fit with configuration
# mv  ./train_data/rec/paddle_v5_ss1/train ./train_data/rec
# mv  ./train_data/rec/paddle_v_ss1/val ./train_data/rec

wandb login <wab_token>

# unzip

cd ./train_data/paddle_det_v2_200k_zip/

7z x paddle_det_v2_200k.zip -mmt=on

cd /workspace/PaddleOCR-chulanpro/

cd ./train_data/paddle_det_v2_eval_500_zip/

7z x paddle_det_v2_eval_500.zip

cd /workspace/PaddleOCR-chulanpro/





#python -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7,8,9,10,11' tools/train.py --config configs/rec/PP-Thesis/HisDoc1B/base/rec_svtrv2_base_ch_exp_6.yml -o Global.eval_batch_step="[15408,1926]"
#python -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7' tools/train.py --config configs/rec/PP-Thesis/HisDoc1B/base/rec_svtrv2_base_ch_exp_6.yml -o Global.eval_batch_step="[15408,1926]"	
#python3 -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7,8,9,10,11' --log_dir ./logs tools/train.py -c configs/rec/PP-Thesis/HisDoc1B/base/rec_focalnet_base_ch_exp_6v2.yml -o Train.loader.batch_size_per_card=128 Eval.loader.batch_size_per_card=128

python3 -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7' --log_dir ./output/det_db_large_exp_6/logs tools/train.py --config configs/det/PP-Thesis/HisDoc1B/det_db_large_exp_6/det_db_large_exp_6.yml

# python3 -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7,8,9,10,11,12,13' --log_dir ./output/det_db_large_exp_6/logs tools/train.py --config configs/det/PP-Thesis/HisDoc1B/det_db_large_exp_6/det_db_large_exp_6_continue.yml

huggingface-cli upload chulanpro5/checkpoints \
  ./output/det_db_large_exp_6/ \
  ./det_db_large_exp_6/ \
  --repo-type=dataset \
  --token=<token>