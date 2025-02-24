#!/bin/bash

python main.py \
--data=ecg_instruct_pulse_mapped_2500 \
--model=llama-3.2-1b \
--dis \
--gpus=2,3,7 \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
--seg_len=2500 \
--peft \
--train=end2end \
--system_prompt=./data/system_prompt_e2e.txt \
--batch_size=6 \
--ports=12357 \
--pad_to_max=1024 \
--log

# python main.py \
# --data=ecg_instruct_pulse_mapped_2500 \
# --model=llama-3.2-1b \
# --dis \
# --gpus=2,3,7 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
# --seg_len=2500 \
# --peft \
# --train=end2end \
# --system_prompt=./data/system_prompt_e2e.txt \
# --batch_size=2 \
# --ports=12357


# python main.py \
# --data=ecg_instruct_45k_mapped \
# --model=llama-3.2-1b \
# --device=cuda:5 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
# --peft \
# --train=end2end \
# --system_prompt=./data/system_prompt_e2e.txt \
# --dev


# python main.py \
# --data=pretrain_mimic_mapped \
# --model=llama-3.2-1b-instruct \
# --device=cuda:7 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
# --peft \
# --train=end2end \
# --dev

# python main.py \
# --data=ecg_instruct_45k_mapped \
# --model=llama-3.2-1b \
# --gpus=6,7 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
# --dis \
# --peft \
# --train=end2end \
# --system_prompt=./data/system_prompt_e2e.txt \
# --dev

# echo "----------------------------------------"


# python main.py \
# --data=pretrain_mimic_mapped \
# --model=gemma-2b \
# --device=cuda:7 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
# --peft \
# --train=end2end \
# --dev


# python main.py \
# --data=pretrain_mimic_mapped \
# --model=opt-1.3b \
# --gpus=4,6 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
# --dis \
# --peft \
# --train=end2end \
# --dev

# echo "----------------------------------------"


# python main.py \
# --data=pretrain_mimic_mapped \
# --model=opt-1.3b \
# --device=cuda:6 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
# --peft \
# --train=end2end \
# --dev


# python main.py \
# --data=pretrain_mimic_mapped \
# --model=gpt2-xl \
# --gpus=4,6 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
# --dis \
# --peft \
# --train=end2end \
# --dev

# echo "----------------------------------------"


# python main.py \
# --data=pretrain_mimic_mapped \
# --model=gpt2-xl \
# --device=cuda:6 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
# --peft \
# --train=end2end \
# --dev