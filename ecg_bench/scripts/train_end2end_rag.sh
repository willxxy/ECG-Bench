#!/bin/bash

# data=("ecg-qa_ptbxl_mapped_1250")


# for d in "${data[@]}"; do
#     python main.py \
#     --data=$d \
#     --model=llama-3.2-1b-instruct \
#     --device=cuda:7 \
#     --ecg_tokenizer=./data/tokenizer_5000_300000_instance.pkl \
#     --seg_len=1250 \
#     --peft \
#     --train=end2end \
#     --system_prompt=./data/system_prompt_e2e.txt \
#     --batch_size=2 \
#     --pad_to_max=2048 \
#     --epochs=1 \
#     --instance_normalize \
#     --attn_implementation=flash_attention_2 \
#     --rag \
#     --load_rag_db=./data/mimic/rag_metadata.json \
#     --load_rag_db_idx=./data/mimic/combined.index \
#     --log
# done


data=("ecg-qa_ptbxl_mapped_1250")


for d in "${data[@]}"; do
    python main.py \
    --data=$d \
    --model=llama-3.2-1b-instruct \
    --device=cuda:7 \
    --ecg_tokenizer=./data/tokenizer_5000_300000_instance.pkl \
    --seg_len=1250 \
    --peft \
    --train=end2end \
    --system_prompt=./data/system_prompt_e2e.txt \
    --batch_size=2 \
    --pad_to_max=4096 \
    --epochs=1 \
    --instance_normalize \
    --attn_implementation=flash_attention_2 \
    --rag \
    --load_rag_db=./data/mimic/rag_metadata.json \
    --load_rag_db_idx=./data/mimic/combined.index \
    --log
done