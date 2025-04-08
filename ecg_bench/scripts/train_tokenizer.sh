#!/bin/bash

# python train_tokenizer.py \
# --num_merges=3500 \
# --sampled_files=data/sampled_450000_200.txt \
# --num_processes=12 \
# --percentiles=data/mimic_percentiles_2500_250_300000.npy \
# --train


# python train_tokenizer.py \
# --num_merges=3500 \
# --sampled_files=data/sampled_450000_200.txt \
# --num_processes=12 \
# --percentiles=data/mimic_percentiles_2500_250_300000.npy \
# --ecg_tokenizer=data/tokenizer_3500_450000.pkl



# python train_tokenizer.py \
# --num_merges=3500 \
# --sampled_files=data/sampled_450000_200.txt \
# --num_processes=12 \
# --percentiles=data/mimic_percentiles_2500_250_300000.npy \
# --train \
# --dev


python train_tokenizer.py \
--num_merges=5000 \
--sampled_files=data/sampled_300000_random.txt \
--num_processes=6 \
--percentiles=data/mimic_percentiles_2500_250_300000.npy \
--train \
--instance_normalize \
--dev


python train_tokenizer.py \
--num_merges=5000 \
--sampled_files=data/sampled_300000_random.txt \
--num_processes=6 \
--percentiles=data/mimic_percentiles_2500_250_300000.npy \
--ecg_tokenizer=data/tokenizer_5000_300000_True_True.pkl \
--instance_normalize \
--dev
