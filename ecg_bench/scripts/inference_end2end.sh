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


#!/usr/bin/env bash

# ------------------- CONFIGURABLE LISTS -------------------
llms=("llama-3.2-1b-instruct" "qwen2.5-1.5b-instruct" "gemma-2-2b-it")
# llms=("gemma-2-2b-it")
datasets=("ecg-qa_ptbxl_mapped_1250" "ecg-qa_mimic-iv-ecg_mapped_1250" "ecg_instruct_45k_mapped_1250" "")          # add more datasets here
# ----------------------------------------------------------

for data in "${datasets[@]}"; do
  for llm in "${llms[@]}"; do
    python main.py \
    --data="$data" \
    --model="${llm}" \
    --device=cuda:4 \
    --ecg_tokenizer=./data/tokenizer_5000_300000_instance.pkl \
    --train=end2end \
    --batch_size=2 \
    --seg_len=1250 \
    --epochs=1 \
    --peft \
    --instance_normalize \
    --pad_to_max=1024 \
    --attn_implementation=flash_attention_2 \
    --system_prompt=./data/system_prompt_e2e.txt \
    --log
  done
done



# models=("vit" "clip" "siglip" )

# for model in "${models[@]}"; do
#     python main.py \
#     --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
#     --model=$model \
#     --device=cuda:6 \
#     --train=first \
#     --batch_size=8 \
#     --seg_len=1250 \
#     --epochs=2 \
#     --instance_normalize \
#     --attn_implementation=flash_attention_2 \
#     --image \
#     --log
# done