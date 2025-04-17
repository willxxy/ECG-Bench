#!/bin/bash

data=("ecg-qa_ptbxl_mapped_1250" "pretrain_mimic_mapped_1250" "ecg_instruct_45k_mapped_1250" "ecg_instruct_pulse_mapped_1250" "ecg-qa_mimic-iv-ecg_mapped_1250")
checkpoints=(
    "llama-3.2-1b-instruct_8_1_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None"
    "vit_llama-3.2-1b-instruct_8_1_0.0001_0.9_0.99_1e-08_500_0.01_True_True_None_None_None"
    "siglip_llama-3.2-1b-instruct_8_1_0.0001_0.9_0.99_1e-08_500_0.01_True_True_None_None_None"
    "clip_llama-3.2-1b-instruct_8_1_0.0001_0.9_0.99_1e-08_500_0.01_True_True_None_None_None"
    "stmem_llama-3.2-1b-instruct_8_1_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None"
    "merl_llama-3.2-1b-instruct_8_1_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None"
)

for d in "${data[@]}"; do
    if [ "$d" = "ecg_instruct_pulse_mapped_1250" ]; then
        data_arg="ecg_bench_pulse_mapped_1250"
    else
        data_arg="$d"
    fi
    
    for ckpt in "${checkpoints[@]}"; do
        python organize_results.py \
        --checkpoint=./runs/$d/0/$ckpt
    done
done