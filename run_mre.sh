#!/usr/bin/env bash
source ~/.bashrc

DATASET_NAME="mnre"
BERT_NAME="bert-base-uncased"
VIT_NAME="openai/clip-vit-base-patch32"
EXP_NAME="test"

CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --model_name="bert-base-inter-re" \
        --experiment_name=${EXP_NAME} \
        --vit_name=$VIT_NAME \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=20 \
        --batch_size=32 \
        --lr=1e-5 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --max_seq=80 \
        --prompt_len=4 \
        --aux_size=128 \
        --rcnn_size=64 \
        --do_train \
        --save_path="ckpt" \
        --write_path="logs" \
        --device='cuda' \
        --num_workers=10 \
