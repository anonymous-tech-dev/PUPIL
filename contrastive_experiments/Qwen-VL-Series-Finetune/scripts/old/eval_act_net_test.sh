#!/bin/bash
set -e
export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

VIDEOS="/workspace/Pupil/contrastive_experiments/dataset/videos"
ANNS="/workspace/Pupil/contrastive_experiments/dataset/annotations"

python tools/eval_activitynet.py \
  --model_path output/activitynet_qwen3vl8b_contrastive_1gpu \
  --data_path ${ANNS}/test_llava.json \
  --image_folder ${VIDEOS} \
  --max_new_tokens 64 \
  --batch_size 1
``