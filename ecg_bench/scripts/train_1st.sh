#!/bin/bash

# models=("stmem" "merl" "mlae" "mtae" "siglip" "clip" "vit")
models=("merl")
data=("ecg-qa-mimic-iv-ecg-250-1250")
# data=("ecg_instruct_45k_mapped_1250") 

### MULTI GPU
for model in "${models[@]}"; do
    python main.py \
    --data=$data \
    --model=$model \
    --device=cuda:0 \
    --train=first \
    --batch_size=64 \
    --seg_len=1250 \
    --epochs=50 \
    --instance_normalize \
    --attn_implementation=flash_attention_2 \
    --log
done



# models=("vit" "clip" "siglip")

# for model in "${models[@]}"; do
#     python main.py \
#     --data=test-ecg \
#     --model=$model \
#     --device=cuda:6 \
#     --train=first \
#     --batch_size=64 \
#     --seg_len=1250 \
#     --epochs=50 \
#     --instance_normalize \
#     --attn_implementation=flash_attention_2 \
#     --image \
#     --dev
# done