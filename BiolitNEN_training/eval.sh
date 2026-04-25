#!/bin/bash

MODEL_NAME_OR_PATH=./tmp/SapBERT-multi
OUTPUT_DIR=./output/
DATA_DIR=./datasets/ncbi-disease/

python eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dictionary_path ${DATA_DIR}/test_dictionary.txt \
    --data_dir ${DATA_DIR}/processed_test \
    --output_dir ${OUTPUT_DIR} \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions \
    --score_mode hybrid
