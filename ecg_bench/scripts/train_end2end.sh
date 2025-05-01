#!/bin/bash

data=("ecg_instruct_45k_mapped_2500")


for d in "${data[@]}"; do
    python main.py \
    --data=$d \
    --model=llama-3.2-1b-instruct \
    --device=cuda:5 \
    --ecg_tokenizer=./data/tokenizer_5000_300000_instance.pkl \
    --seg_len=2500 \
    --peft \
    --train=end2end \
    --system_prompt=./data/system_prompt_e2e.txt \
    --batch_size=8 \
    --pad_to_max=1024 \
    --epochs=1 \
    --instance_normalize \
    --attn_implementation=flash_attention_2 \
    --log
done
