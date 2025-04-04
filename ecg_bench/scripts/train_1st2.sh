#!/bin/bash

models=("vit" "clip" "siglip")

### MULTI GPU
for model in "${models[@]}"; do
    python main.py \
    --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
    --model=$model \
    --device=cuda:4 \
    --train=first \
    --batch_size=64 \
    --seg_len=1250 \
    --epochs=50 \
    --instance_normalize \
    --attn_implementation=flash_attention_2 \
    --log
done

# ### SINGLE GPU 
# for model in "${models[@]}"; do
#     python main.py \
#     --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
#     --model=$model \
#     --device=cuda:3 \
#     --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
#     --train=first \
#     --batch_size=64 \
#     --epochs=50 \
#     --seg_len=1250 \
#     --instance_normalize \
#     --log
# done