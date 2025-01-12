python train_tokenizer.py \
--num_merges=3500 \
--sampled_files=data/sampled_300000_200.txt \
--num_processes=6 \
--percentiles=data/mimic_percentiles_2500_250_300000.npy \
--train


python train_tokenizer.py \
--num_merges=3500 \
--sampled_files=data/sampled_300000_200.txt \
--num_processes=6 \
--percentiles=data/mimic_percentiles_2500_250_300000.npy \
--tokenizer=data/tokenizer_3500_300000.pkl