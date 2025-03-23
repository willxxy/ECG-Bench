#!/bin/bash

# Define array of models
models=("stmem" "merl" "mlae" "mtae")
# did stmem

# Loop through each model
# for model in "${models[@]}"; do
#     python main.py \
#     --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
#     --model=$model \
#     --dis \
#     --gpus=2,3,7 \
#     --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
#     --train=first \
#     --batch_size=64 \
#     --seg_len=1250 \
#     --epochs=50
# done

### SINGLE GPU 
for model in "${models[@]}"; do
    python main.py \
    --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
    --model=$model \
    --device=cuda:3 \
    --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
    --train=first \
    --batch_size=64 \
    --epochs=50 \
    --seg_len=1250 \
    --log
done