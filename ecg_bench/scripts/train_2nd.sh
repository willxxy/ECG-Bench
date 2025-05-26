#!/bin/bash

models=("stmem" "merl" "mlae" "mtae" "siglip" "clip" "vit")
llms=("gemma-2-2b-it" "llama-3.2-1b-instruct" "qwen2.5-1.5b-instruct")

# ### MULTI GPU
for llm in "${llms[@]}"; do
    for model in "${models[@]}"; do
        python main.py \
        --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
        --model=$model_$llm \
        --device=cuda:6 \
        --train=second \
        --batch_size=256 \
        --seg_len=1250 \
        --epochs=50 \
        --instance_normalize \
        --attn_implementation=flash_attention_2 \
        --encoder_checkpoint=./runs/ecg-qa_mimic-iv-ecg_mapped_1250/0/$model_8_1_0.0001_0.9_0.99_1e-08_500_0.01_True_True_None_None_None \
        --dev
    done
done


models=("vit" "clip" "siglip" )

for model in "${models[@]}"; do
    python main.py \
    --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
    --model=$model \
    --device=cuda:6 \
    --train=first \
    --batch_size=256 \
    --seg_len=1250 \
    --epochs=50 \
    --instance_normalize \
    --attn_implementation=flash_attention_2 \
    --image \
    --log
done