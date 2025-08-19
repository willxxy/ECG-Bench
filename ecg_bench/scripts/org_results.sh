#!/bin/bash

datasets=("ecg-qa_ptbxl-250-1250" "ecg-qa-mimic-iv-ecg-250-1250" "ecg-instruct-45k-250-1250" "ecg-instruct-pulse-250-1250" "pretrain-mimic-250-1250") # add more datasets here
checkpoints=(
    "llama-3.2-3b-instruct_2_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_True_combined_report_5_False"
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