#!/bin/bash

data=("ecg_instruct_45k_mapped_1250")
models=("merl_qwen2.5-1.5b-instruct")

for d in "${data[@]}"; do
    for model in "${models[@]}"; do
        python main.py \
        --data=$d \
        --model=$model \
        --device=cuda:5 \
        --seg_len=1250 \
        --peft \
        --ecg_tokenizer=./data/tokenizer_5000_300000_instance.pkl \
        --train=second \
        --system_prompt=./data/system_prompt_e2e.txt \
        --batch_size=4 \
        --pad_to_max=1024 \
        --epochs=1 \
        --encoder_checkpoint=./runs/ecg-qa_mimic-iv-ecg_mapped_1250/0/merl_256_50_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None \
        --instance_normalize \
        --attn_implementation=flash_attention_2 \
        --log
    done
done
