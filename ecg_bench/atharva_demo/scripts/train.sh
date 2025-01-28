python main.py \
--model=meta-llama/Llama-3.2-1B \
--tokenizer_check=tokenizer_3500 \
--batch_size=1 \
--pad_to_max=1020 \
--peft \
--num_merges=3500 \
--epochs=1 \
--percentiles=./data/mimic_dataset_stats.npy \
--dataset=mimic_500 \
--device=cuda:4