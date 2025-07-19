#!/bin/bash

checkpoint=(
# "clip_qwen2.5-1.5b-instruct_4_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None_False" 
# "siglip_qwen2.5-1.5b-instruct_4_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None_False" 
"vit_llama-3.2-1b-instruct_8_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None_user_input_False" 
)
data=("ecg_instruct_45k_mapped_1250")

for d in "${data[@]}"; do
    for c in "${checkpoint[@]}"; do
        # Extract the model prefix (clip_, merl_, etc.) from the checkpoint name
        model_prefix=$(echo "$c" | cut -d'_' -f1)
        
        python main.py \
        --data=ecg-qa_ptbxl_mapped_1250 \
        --model=vit_llama-3.2-1b-instruct \
        --device=cuda:7 \
        --seg_len=1250 \
        --peft \
        --inference=second \
        --system_prompt=./data/system_prompt_e2e.txt \
        --batch_size=1 \
        --pad_to_max=1024 \
        --instance_normalize \
        --attn_implementation=flash_attention_2 \
        --checkpoint=./runs/ecg-qa_ptbxl_mapped_1250/0/$c
    done
done