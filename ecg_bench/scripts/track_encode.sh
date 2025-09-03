#!/bin/bash

python track_encoding.py \
--ecg_tokenizer=./data/tokenizer_5000_300000_instance_super.pkl \
--list_of_paths=./data/mimic/preprocessed_1250_250/*.npy