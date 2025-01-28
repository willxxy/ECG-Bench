#!/bin/bash

python track_encoding.py \
--ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
--list_of_paths=./data/ptb/preprocessed_500_250/*.npy \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy