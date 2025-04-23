#!/bin/bash

data=("ecg_instruct_pulse_mapped_1250" "ecg-qa_mimic-iv-ecg_mapped_1250")
# "ecg-qa_ptbxl_mapped_1250" "pretrain_mimic_mapped_1250"  done

# "ecg_instruct_45k_mapped_1250" done


for d in "${data[@]}"; do
    python main.py \
    --data=$d \
    --model=mlae_llama-3.2-1b-instruct \
    --device=cuda:4 \
    --seg_len=1250 \
    --peft \
    --ecg_tokenizer=./data/tokenizer_5000_300000_instance.pkl \
    --train=second \
    --system_prompt=./data/system_prompt_e2e.txt \
    --batch_size=8 \
    --pad_to_max=1024 \
    --epochs=1 \
    --instance_normalize \
    --attn_implementation=flash_attention_2 \
    --encoder_checkpoint=./runs/ecg-qa_mimic-iv-ecg_mapped_1250/0/mlae_256_50_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None \
    --log
done
