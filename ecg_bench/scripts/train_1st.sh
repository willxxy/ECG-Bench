#!/bin/bash

# Define array of models
# models=("stmem" "clip" "siglip" "merl" "mlae" "vit" "mtae")
# models=("mlae" "vit" "mtae")

# # Loop through each model
# for model in "${models[@]}"; do
#     python main.py \
#     --data=ecg-qa_mimic-iv-ecg_mapped \
#     --model=$model \
#     --dis \
#     --gpus=2,3,7 \
#     --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
#     --train=first \
#     --batch_size=64 \
#     --epochs=100 \
#     --log
# done


python main.py \
--data=ecg-qa_mimic-iv-ecg_mapped \
--model=clip \
--device=cuda:7 \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--train=first \
--batch_size=64 \
--epochs=100 \
--dev