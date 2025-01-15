#!/bin/bash

echo "ecg_instruct_45k_mapped"

python train_encoder.py \
--data=ecg_instruct_45k_mapped \
--model=clip \
--gpus=4,5 \
--dis \
--dev

echo "----------------------------------------"

python train_encoder.py \
--data=ecg_instruct_45k_mapped \
--model=clip \
--device=cuda:4 \
--dev

# echo "----------------------------------------"

# echo "pretrain_mimic_mapped"

# python train_encoder.py \
# --data=pretrain_mimic_mapped \
# --gpus=4,5 \
# --dis

# echo "----------------------------------------"

# python train_encoder.py \
# --data=pretrain_mimic_mapped \
# --device=cuda:4

# echo "----------------------------------------"

# echo "ecg-qa_mimic-iv-ecg_mapped"

# python train_encoder.py \
# --data=ecg-qa_mimic-iv-ecg_mapped \
# --gpus=4,5 \
# --dis

# echo "----------------------------------------"

# python train_encoder.py \
# --data=ecg-qa_mimic-iv-ecg_mapped \
# --device=cuda:4

# echo "----------------------------------------"

# echo "ecg-qa_ptbxl_mapped"

# python train_encoder.py \
# --data=ecg-qa_ptbxl_mapped \
# --gpus=4,5 \
# --dis

# echo "----------------------------------------"

# python train_encoder.py \
# --data=ecg-qa_ptbxl_mapped \
# --device=cuda:4

