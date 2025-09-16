#!/bin/bash

# data=("ecg-qa_ptbxl_mapped_1250" "pretrain_mimic_mapped_1250" "ecg_instruct_45k_mapped_1250" "ecg_instruct_pulse_mapped_1250" "ecg-qa_mimic-iv-ecg_mapped_1250")
data=("ecg_instruct_45k_mapped_1250")
# data=("ecg-qa_mimic-iv-ecg_mapped_1250")
# data=("ecg-qa_ptbxl_mapped_1250")
# data=("pretrain_mimic_mapped_1250")
# retrieval_base="feature"
# retrieved_information="combined"
# rag_k=1
# rag_prompt_mode="system_prompt"
# normalized_rag_features=True

checkpoints='llama-3.2-1b-instruct_adam_2_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_True_1_None_None_feature_report_1_system_prompt_None_False'
# checkpoints='qwen2.5-3b_2_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None_False'

for d in "${data[@]}"; do
    if [ "$d" = "ecg_instruct_pulse_mapped_1250" ]; then
        data_arg="ecg_bench_pulse_mapped_1250"
    else
        data_arg="$d"
    fi
    
    for ckpt in "${checkpoints[@]}"; do
        python organize_results.py \
        --checkpoint=./runs/$d/0/$ckpt/
    done
done