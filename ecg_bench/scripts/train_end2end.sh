#!/usr/bin/env bash

# ------------------- CONFIGURABLE LISTS -------------------
llms=("llama-3.2-1b-instruct" "qwen2.5-1.5b-instruct" "gemma-2-2b-it")
# llms=("gemma-2-2b-it")
datasets=("ecg-qa_ptbxl_mapped_1250" "ecg-qa_mimic-iv-ecg_mapped_1250" "ecg_instruct_45k_mapped_1250" "ecg_instruct_pulse_mapped_1250" "pretrain_mimic_mapped_1250")          # add more datasets here
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