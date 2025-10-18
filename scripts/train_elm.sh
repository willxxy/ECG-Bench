# single GPU (works as you had it)
datasets=(
  # ecg-qa-mimic-iv-ecg-250-1250
  # ecg-qa-ptbxl-250-1250
  # pretrain-mimic-250-1250
  ecg-instruct-45k-250-1250
  # ecg-bench-pulse-250-1250
  # ecg-instruct-pulse-250-1250
)
for data in "${datasets[@]}"
do
  python -m ecg_bench.train_elm \
    --ecg_token \
    --llm=llama-3.2-1b-instruct \
    --data="$data" \
    --ecg_tokenizer=./ecg_bench/configs/ecg_tokenizers/ecg_byte_tokenizer_5000_500000.pkl \
    --device=cuda:1 \
    --peft \
    --batch_size=2 \
    --attention_type=flash_attention_2 \
    --system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
    --wandb
    echo "Finished training on $data"
    echo "-----------------------------------"
done

# python -m ecg_bench.train_elm \
# --ecg_signal \
# --batch_size=2 \
# --llm=llama-3.2-1b-instruct \
# --encoder=siglip-base-patch16-224 \
# --data=ecg-instruct-45k-250-1250 \
# --device=cuda:1 \
# --peft \
# --attention_type=flash_attention_2 \
# --system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
# --dev