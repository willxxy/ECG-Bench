#!/bin/bash

# python train_tokenizer.py \
# --num_merges=5000 \
# --sampled_files=data/sampled_300000_random.txt \
# --num_cores=6 \
# --train_tokenizer \
# --instance_normalize \
# --dev


python train_tokenizer.py \
--num_merges=5000 \
--sampled_files=data/sampled_300000_random.txt \
--num_cores=6 \
--ecg_tokenizer=data/tokenizer_5000_300000_instance.pkl \
--instance_normalize \
--dev
