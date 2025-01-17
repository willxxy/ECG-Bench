python main_end2end.py \
--data=pretrain_mimic_mapped \
--model=llama-3.2-1b \
--device=cuda:3 \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
--peft \
--inference=end2end \
--checkpoint=end2end_llama-3.2-1b_2_2_0.0001_0.9_0.99_1e-08_500_0.01 \
--dev