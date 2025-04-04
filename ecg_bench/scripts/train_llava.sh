#!/bin/bash

data=("ecg-qa_ptbxl_mapped_1250" "pretrain_mimic_mapped_1250" "ecg_instruct_45k_mapped_1250" "ecg_instruct_pulse_mapped_1250" "ecg-qa_mimic-iv-ecg_mapped_1250")


for d in "${data[@]}"; do
    python main.py \
    --data=$d \
    --model=vit_llama-3.2-1b-instruct \
    --device=cuda:4 \
    --seg_len=1250 \
    --peft \
    --train=second \
    --system_prompt=./data/system_prompt_e2e.txt \
    --batch_size=2 \
    --pad_to_max=1024 \
    --epochs=1 \
    --instance_normalize \
    --attn_implementation=flash_attention_2 \
    --image \
    --log
done
