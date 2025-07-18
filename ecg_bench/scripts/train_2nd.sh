#!/bin/bash

# models=("stmem" "merl" "mlae" "mtae" "siglip" "clip" "vit")
models=("clip")
# llms=("gemma-2-2b-it" "llama-3.2-1b-instruct" "qwen2.5-1.5b-instruct")
llms=("llama-3.2-1b-instruct")

# ### MULTI GPU
for llm in "${llms[@]}"; do
    for model in "${models[@]}"; do
        python main.py \
        --data=ecg-qa_ptbxl_mapped_1250 \
        --model=vit_llama-3.2-1b-instruct \
        --device=cuda:0 \
        --train=second \
        --batch_size=8 \
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