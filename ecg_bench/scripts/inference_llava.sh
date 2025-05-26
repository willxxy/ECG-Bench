#!/bin/bash

checkpoint=(
"clip_llama-3.2-1b-instruct_8_1_512_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None_False" 
"siglip_gemma-2-2b-it_2_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None_False" 
"vit_gemma-2-2b-it_2_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_True_None_None_None_False" 
)

for c in "${checkpoint[@]}"; do
    # Extract the model prefix (clip_, merl_, etc.) from the checkpoint name
    model_prefix=$(echo "$c" | cut -d'_' -f1)
    
    python main.py \
    --data=ecg_instruct_45k_mapped_1250 \
    --model=${model_prefix}_gemma-2-2b-it \
    --device=cuda:4 \
    --seg_len=1250 \
    --peft \
    --inference=second \
    --checkpoint=./runs/ecg_instruct_45k_mapped_1250/0/$c \
    --system_prompt=./data/system_prompt_e2e.txt \
    --batch_size=1 \
    --epochs=1 \
    --instance_normalize \
    --image
done