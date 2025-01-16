#!/bin/bash

echo "ecg_instruct_45k_mapped"

python train_encoder.py \
--data=ecg_instruct_45k_mapped \
--model=clip \
--gpus=3,5 \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--dis \
--dev

python train_encoder.py \
--data=ecg_instruct_45k_mapped \
--model=vit \
--gpus=3,5 \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--dis \
--dev

# echo "----------------------------------------"

python train_encoder.py \
--data=ecg_instruct_45k_mapped \
--model=clip \
--device=cuda:3 \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--dev

python train_encoder.py \
--data=ecg_instruct_45k_mapped \
--model=vit \
--device=cuda:3 \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--dev

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

