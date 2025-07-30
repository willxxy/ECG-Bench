#!/bin/bash

models=("stmem" "merl" "mlae" "mtae" "siglip" "clip" "vit")

# ### MULTI GPU
# for model in "${models[@]}"; do
#     python main.py \
#     --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
#     --model=$model \
#     --device=cuda:4 \
#     --train=first \
#     --batch_size=256 \
#     --seg_len=1250 \
#     --lr=8e-5 \
#     --weight_decay=1e-4 \
#     --epochs=50 \
#     --instance_normalize \
#     --attn_implementation=flash_attention_2 \
#     --log
# done


models=("vit" "clip" "siglip" )

for model in "${models[@]}"; do
    python main.py \
    --data=ecg-qa_ptbxl_mapped_1250 \
    --model=$model \
    --device=cuda:6 \
    --train=first \
    --batch_size=1 \
    --seg_len=1250 \
    --epochs=50 \
    --instance_normalize \
    --attn_implementation=flash_attention_2 \
    --image \
    --dev
done