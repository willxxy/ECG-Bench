#!/bin/bash


python main_end2end.py \
--data=pretrain_mimic_mapped \
--model=llama-3.2-1b \
--gpus=3,0 \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
--dis \
--peft \
--train=end2end \
--dev

echo "----------------------------------------"


python main_end2end.py \
--data=pretrain_mimic_mapped \
--model=llama-3.2-1b \
--device=cuda:3 \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
--peft \
--train=end2end \
--dev