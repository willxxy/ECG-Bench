#!/bin/bash

# data=("ecg-qa_ptbxl_mapped_1250" "pretrain_mimic_mapped_1250" "ecg_instruct_45k_mapped_1250" "ecg_instruct_pulse_mapped_1250" "ecg-qa_mimic-iv-ecg_mapped_1250")
data=("ecg_instruct_45k_mapped_1250")
checkpoints=(
    "qwen2.5-3b_2_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None_False"
)

for d in "${data[@]}"; do
    if [ "$d" = "ecg_instruct_pulse_mapped_1250" ]; then
        data_arg="ecg_bench_pulse_mapped_1250"
    else
        data_arg="$d"
    fi
    
    for ckpt in "${checkpoints[@]}"; do
        python organize_results.py \
        --checkpoint=./runs/$d/0/$ckpt/train_no_inf_no
    done
done
