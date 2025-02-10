# python main.py \
# --data=ecg_instruct_45k_mapped \
# --model=llama-3.2-1b \
# --device=cuda:7 \
# --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
# --ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
# --peft \
# --inference=end2end \
# --checkpoint=llama-3.2-1b_1_2_0.0001_0.9_0.99_1e-08_500_0.01 \
# --system_prompt=./data/system_prompt_e2e.txt \
# --dev



python main.py \
--data=ecg_instruct_45k_mapped \
--model=siglip_llama-3.2-1b \
--device=cuda:7 \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--peft \
--inference=second \
--checkpoint=siglip_llama-3.2-1b_1_2_0.0001_0.9_0.99_1e-08_500_0.01 \
--system_prompt=./data/system_prompt_e2e.txt \
--encoder_checkpoint=siglip_1_2_0.0001_0.9_0.99_1e-08_500_0.01 \
--encoder_data=pretrain_mimic_mapped \
--dev