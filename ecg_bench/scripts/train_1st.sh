#!/bin/bash

models=("siglip" "clip")
# "stmem" "merl" "mlae" "mtae" " done

# VIT DONE
# CLIP SO SLOW

# ### MULTI GPU
for model in "${models[@]}"; do
    python main.py \
    --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
    --model=$model \
    --device=cuda:2 \
    --train=first \
    --batch_size=256 \
    --seg_len=1250 \
    --epochs=50 \
    --instance_normalize \
    --attn_implementation=flash_attention_2 \
    --log
done


models=("vit" "clip" "siglip" )

### MULTI GPU
for model in "${models[@]}"; do
    python main.py \
    --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
    --model=$model \
    --device=cuda:2 \
    --train=first \
    --batch_size=256 \
    --seg_len=1250 \
    --epochs=10 \
    --instance_normalize \
    --attn_implementation=flash_attention_2 \
    --image \
    --log
done