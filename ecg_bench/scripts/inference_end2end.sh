#!/bin/bash

checkpoint=(
# "clip_qwen2.5-1.5b-instruct_4_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None_False" 
# "siglip_qwen2.5-1.5b-instruct_4_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None_False" 
"llama-3.2-1b-instruct_adam_2_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None_1_False" 
)
data=("ecg-qa-ptbxl-250-1250")
folds=(1 2 3 4 5)

for d in "${data[@]}"; do
    for c in "${checkpoint[@]}"; do
        # Extract the model prefix (clip_, merl_, etc.) from the checkpoint name
        model_prefix=$(echo "$c" | cut -d'_' -f1)
        for fold in "${folds[@]}"; do
            python main.py \
            --data=ecg-qa-ptbxl-250-1250 \
            --model=llama-3.2-1b-instruct \
            --device=cuda:7 \
            --seg_len=1250 \
            --peft \
            --ecg_tokenizer=./data/tokenizer_5000_300000_True_None.pkl \
            --inference=end2end \
            --system_prompt=./data/system_prompt_e2e.txt \
            --batch_size=1 \
            --pad_to_max=1024 \
            --attn_implementation=flash_attention_2 \
            --checkpoint=./runs/ecg-qa-ptbxl-250-1250/0/$c \
            --fold="$fold"
        done
    done
done