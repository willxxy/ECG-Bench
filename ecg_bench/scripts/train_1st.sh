#!/bin/bash

echo "pretrain_mimic_mapped"

python main.py \
--data=pretrain_mimic_mapped \
--model=mtae \
--device=cuda:7 \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--train=first \
--dev

# python main_first.py \
# --data=ecg_instruct_45k_mapped \
# --model=clip \
# --gpus=4,5 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --dis \
# --train=first \
# --dev

# python main_first.py \
# --data=ecg_instruct_45k_mapped \
# --model=vit \
# --gpus=4,5 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --dis \
# --train=first \
# --dev

# # echo "----------------------------------------"

# python main_first.py \
# --data=ecg_instruct_45k_mapped \
# --model=clip \
# --device=cuda:4 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --train=first \
# --dev

# python main_first.py \
# --data=ecg_instruct_45k_mapped \
# --model=vit \
# --device=cuda:4 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --train=first \
# --dev

# echo "----------------------------------------"

# echo "pretrain_mimic_mapped"

# python train_encoder.py \
# --data=pretrain_mimic_mapped \
# --model=clip \
# --gpus=4,5 \
# --dis \
# --dev

# echo "----------------------------------------"

# python train_encoder.py \
# --data=pretrain_mimic_mapped \
# --model=clip \
# --device=cuda:0 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --dev

# echo "----------------------------------------"

# echo "ecg-qa_mimic-iv-ecg_mapped"

# python train_encoder.py \
# --data=ecg-qa_mimic-iv-ecg_mapped \
# --gpus=4,5 \
# --dis

# echo "----------------------------------------"

# python train_encoder.py \
# --data=ecg-qa_mimic-iv-ecg_mapped \
# --model=clip \
# --device=cuda:0 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --dev

# echo "----------------------------------------"

# echo "ecg-qa_ptbxl_mapped"

# python train_encoder.py \
# --data=ecg-qa_ptbxl_mapped \
# --gpus=4,5 \
# --dis

# echo "----------------------------------------"

# python train_encoder.py \
# --data=ecg-qa_ptbxl_mapped \
# --device=cuda:0 \
# --model=clip \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --dev

