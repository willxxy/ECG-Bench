python -m ecg_bench.evaluate_elm \
--ecg_token \
--llm=llama-3.2-1b-instruct \
--data=ecg-instruct-45k-250-2500 \
--ecg_tokenizer=./ecg_bench/configs/ecg_tokenizers/ecg_byte_tokenizer_5000_500000.pkl \
--device=cuda:6 \
--peft \
--attention_type=flash_attention_2 \
--system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
--dev

python -m ecg_bench.evaluate_elm \
--ecg_signal \
--llm=llama-3.2-1b-instruct \
--encoder=merl \
--data=ecg-instruct-45k-250-2500 \
--device=cuda:1 \
--peft \
--attention_type=flash_attention_2 \
--system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
--dev