python main.py \
--data=pretrain_mimic_mapped \
--model=merl_llama-3.2-1b \
--gpus=4,6 \
--dis \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--peft \
--train=second \
--encoder_checkpoint=merl_1_2_0.0001_0.9_0.99_1e-08_500_0.01 \
--encoder_data=pretrain_mimic_mapped \
--dev

python main.py \
--data=pretrain_mimic_mapped \
--model=merl_llama-3.2-1b \
--device=cuda:6 \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--peft \
--train=second \
--encoder_checkpoint=merl_1_2_0.0001_0.9_0.99_1e-08_500_0.01 \
--encoder_data=pretrain_mimic_mapped \
--dev


python main.py \
--data=pretrain_mimic_mapped \
--model=merl_llama-3.2-1b \
--device=cuda:6 \
--percentiles=./data/mimic_percentiles_2500_250_300000.npy \
--peft \
--inference=second \
--encoder_checkpoint=merl_1_2_0.0001_0.9_0.99_1e-08_500_0.01 \
--encoder_data=pretrain_mimic_mapped \
--checkpoint=merl_llama-3.2-1b_1_2_0.0001_0.9_0.99_1e-08_500_0.01 \
--dev