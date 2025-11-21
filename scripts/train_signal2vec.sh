CUDA_VISIBLE_DEVICES=0 python -m ecg_bench.train_signal2vec \
--sampled_file=./ecg_bench/configs/ecg_tokenizers/sampled_500000_random.txt \
--ecg_tokenizer=./ecg_bench/configs/ecg_tokenizers/ecg_byte_tokenizer_6000_800000.pkl
