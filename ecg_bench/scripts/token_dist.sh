#!/bin/bash

python token_distribution.py \
--ecg_tokenizer=./data/tokenizer_5000_300000_instance_super.pkl \
--list_of_paths=./data/mimic/preprocessed_1250_250/*.npy \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--num_processes=6 \
--instance_normalize