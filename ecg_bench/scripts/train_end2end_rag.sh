#!/bin/bash

# data=("ecg-qa_ptbxl_mapped_1250")

retrieval_base="signal"

# for d in "${data[@]}"; do
python main.py \
    --data=ecg_instruct_45k_mapped_1250 \
    --model=llama-3.2-1b-instruct \
    --device=cuda:1 \
    --ecg_tokenizer=./data/tokenizer_5000_300000_instance.pkl \
    --seg_len=1250 \
    --peft \
    --train=end2end \
    --system_prompt=./data/system_prompt_e2e.txt \
    --batch_size=2 \
    --pad_to_max=1024 \
    --epochs=1 \
    --instance_normalize \
    --attn_implementation=flash_attention_2 \
    --rag \
    --rag_k=1 \
    --rag_prompt_mode=system_prompt \
    --retrieval_base=$retrieval_base \
    --retrieved_information=combined \
    --load_rag_db=./data/mimic/rag_metadata.json \
    --load_rag_db_idx=./data/mimic/${retrieval_base}.index \
    --log
# done