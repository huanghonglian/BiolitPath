#!/bin/bash

MODEL_NAME_OR_PATH=SapBERT-from-PubMedBERT-fulltext
OUTPUT_DIR=./tmp/SapBERT-multi

python train_multi.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --use_cuda \
    --topk 20 \
    --epoch 5 \
    --train_batch_size 16\
    --initial_sparse_weight 0\
    --learning_rate 1e-5 \
    --max_length 25 \
    --dense_ratio 0.5