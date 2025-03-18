#!/bin/bash

# Define array of models
models=("stmem" "clip" "siglip" "merl" "mlae" "vit" "mtae")

# Loop through each model
for model in "${models[@]}"; do
    python main.py \
    --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
    --model=$model \
    --dis \
    --gpus=2,3 \
    --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
    --train=first \
    --batch_size=64 \
    --seg_len=1250 \
    --epochs=50 \
    --log
done

### SINGLE GPU 
# for model in "${models[@]}"; do
#     python main.py \
#     --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
#     --model=$model \
#     --device=cuda:7 \
#     --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
#     --train=first \
#     --batch_size=64 \
#     --epochs=100 \
#     --seg_len=1250 \
#     --log
# done