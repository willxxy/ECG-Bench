#!/bin/bash

data=("ecg_instruct_45k_mapped_1250")
models=("vit_llama-3.2-1b-instruct" "siglip_llama-3.2-1b-instruct" "clip_llama-3.2-1b-instruct")

for d in "${data[@]}"; do
    for model in "${models[@]}"; do
        python main.py \
        --data=$d \
        --model=$model \
        --device=cuda:4 \
        --seg_len=1250 \
        --peft \
        --ecg_tokenizer=./data/tokenizer_5000_300000_instance.pkl \
        --train=second \
        --system_prompt=./data/system_prompt_e2e.txt \
        --batch_size=8 \
        --pad_to_max=512 \
        --epochs=1 \
        --instance_normalize \
        --attn_implementation=flash_attention_2 \
        --log
    done
done
