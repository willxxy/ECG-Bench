#!/bin/bash

python token_distribution.py \
--ecg_tokenizer=./data/tokenizer_5000_450000.pkl \
--list_of_paths=./data/mimic/preprocessed_2500_250/*.npy \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--num_processes=12