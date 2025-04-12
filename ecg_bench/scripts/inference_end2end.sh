#!/bin/bash

data=("ecg-qa_ptbxl_mapped_1250" "pretrain_mimic_mapped_1250" "ecg_instruct_45k_mapped_1250" "ecg_instruct_pulse_mapped_1250" "ecg-qa_mimic-iv-ecg_mapped_1250")


for d in "${data[@]}"; do
    if [ "$d" = "ecg_instruct_pulse_mapped_1250" ]; then
        data_arg="ecg_bench_pulse_mapped_1250"
    else
        data_arg="$d"
    fi
    
    python main.py \
    --data=$data_arg \
    --model=llama-3.2-1b-instruct \
    --device=cuda:3 \
    --ecg_tokenizer=./data/tokenizer_5000_300000_instance.pkl \
    --seg_len=1250 \
    --peft \
    --inference=end2end \
    --checkpoint=./runs/$d/0/llama-3.2-1b-instruct_8_1_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None \
    --system_prompt=./data/system_prompt_e2e.txt \
    --batch_size=1 \
    --pad_to_max=1024 \
    --epochs=1 \
    --instance_normalize \
    --attn_implementation=flash_attention_2
done