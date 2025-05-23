# python main.py \
# --data=ecg_instruct_45k_mapped \
# --model=mlae_llama-3.2-1b \
# --device=cuda:7 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --peft \
# --train=second \
# --encoder_checkpoint=mlae_1_2_0.0001_0.9_0.99_1e-08_500_0.01 \
# --encoder_data=pretrain_mimic_mapped \
# --system_prompt=./data/system_prompt_e2e.txt \
# --attn_implementation=flash_attention_2 \
# --dev

# python main.py \
# --data=ecg_instruct_45k_mapped \
# --model=mlae_llama-3.2-1b \
# --dis \
# --gpus=6,7 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --peft \
# --train=second \
# --encoder_checkpoint=mlae_1_2_0.0001_0.9_0.99_1e-08_500_0.01 \
# --encoder_data=pretrain_mimic_mapped \
# --system_prompt=./data/system_prompt_e2e.txt \
# --attn_implementation=flash_attention_2 \
# --dev

# python main.py \
# --data=pretrain_mimic_mapped \
# --model=siglip_llama-3.2-1b \
# --device=cuda:6 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --peft \
# --train=second \
# --encoder_checkpoint=siglip_1_2_0.0001_0.9_0.99_1e-08_500_0.01 \
# --encoder_data=pretrain_mimic_mapped \
# --dev


python main.py \
--data=ecg_instruct_45k_mapped_500 \
--model=llama-3.2-1b \
--batch_size=1 \
--device=cuda:7 \
--ecg_tokenizer=./data/tokenizer_3500_450000.pkl \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--peft \
--train=end2end \
--attn_implementation=flash_attention_2 \
--system_prompt=./data/system_prompt_e2e.txt \
--dev