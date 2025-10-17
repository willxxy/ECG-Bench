# Sample ECG files
python -m ecg_bench.train_ecg_byte \
    --path_to_ecg_npy=./ecg_bench/data/mimic/preprocessed_2500_250 \
    --num_samples=800000

# # Train ECG Byte Tokenizer from sampled files
python -m ecg_bench.train_ecg_byte \
    --sampled_file=./ecg_bench/configs/ecg_tokenizers/sampled_800000_random.txt \
    --num_merges=6000 \
    --num_cores=12 \
    --num_samples=800000

# # Load ECG Byte Tokenizer
python -m ecg_bench.train_ecg_byte \
    --ecg_tokenizer=./ecg_bench/configs/ecg_tokenizers/tokenizer_5000_300000.pkl

# # Verify ECG Byte Tokenizer
python -m ecg_bench.train_ecg_byte \
    --ecg_tokenizer=./ecg_bench/configs/ecg_tokenizers/tokenizer_5000_300000.pkl \
    --sampled_file=./ecg_bench/configs/ecg_tokenizers/sampled_300000.txt