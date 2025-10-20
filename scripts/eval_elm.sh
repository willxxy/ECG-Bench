python -m ecg_bench.evaluate_elm \
--ecg_signal \
--llm=llama-3.2-1b-instruct \
--encoder=projection \
--data=ecg-instruct-45k-250-1250 \
--device=cuda:1 \
--peft \
--attention_type=flash_attention_2 \
--system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
--elm_ckpt=./ecg_bench/runs/training/elm/2/checkpoints/epoch_best.pt

# python -m ecg_bench.evaluate_elm \
# --ecg_signal \
# --llm=llama-3.2-1b-instruct \
# --encoder=merl \
# --data=ecg-instruct-45k-250-2500 \
# --device=cuda:1 \
# --peft \
# --attention_type=flash_attention_2 \
# --system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
# --dev