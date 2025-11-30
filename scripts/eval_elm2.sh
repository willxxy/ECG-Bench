# python -m ecg_bench.evaluate_elm \
# --ecg_signal \
# --llm=llama-3.2-1b-instruct \
# --encoder=projection \
# --data=ecg-instruct-45k-250-1250 \
# --device=cuda:2 \
# --peft \
# --attention_type=flash_attention_2 \
# --system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
# --elm_ckpt=./ecg_bench/runs/training/elm/0/checkpoints/epoch_best.pt \
# --no_signal

# python -m ecg_bench.evaluate_elm \
# --ecg_signal \
# --llm=llama-3.2-3b-instruct \
# --encoder=projection \
# --data=ecg-instruct-45k-250-1250 \
# --device=cuda:5 \
# --peft \
# --attention_type=flash_attention_2 \
# --system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
# --elm_ckpt=./ecg_bench/runs/training/elm/10/checkpoints/epoch_best.pt

python -m ecg_bench.evaluate_elm \
--ecg_signal \
--llm=qwen2.5-7b-instruct \
--encoder=projection \
--data=ecg-instruct-45k-250-1250 \
--device=cuda:5 \
--peft \
--attention_type=flash_attention_2 \
--system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
--elm_ckpt=./ecg_bench/runs/training/elm/12/checkpoints/step_epoch_0_step_31530.pt

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