python demo.py \
--model=llama-3.2-1b-instruct \
--device=cuda:3 \
--ecg_tokenizer=./data/tokenizer_5000_300000_instance.pkl \
--peft \
--inference=end2end \
--checkpoint=./runs/ecg_instruct_45k_mapped_500_250/0/llama-3.2-1b_1_2_0.0001_0.9_0.99_1e-08_500_0.01 \
--system_prompt=./data/system_prompt_e2e.txt \
--attn_implementation=flash_attention_2 \
--seg_len=1250
