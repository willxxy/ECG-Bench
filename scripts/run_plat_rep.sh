echo "ecg instruction 45k 250 1250"
echo "Running Platonic Representation Hypothesis Analysis (Separate)"
python -m ecg_bench.run_plat_rep_hyp \
--ecg_signal \
--llm=llama-3.2-1b-instruct \
--encoder=projection \
--data=ecg-instruct-45k-250-1250 \
--device=cuda:1 \
--peft \
--attention_type=flash_attention_2 \
--system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
--output_hidden_states \
--no_signal \
--elm_ckpt=./ecg_bench/runs/training/elm/0/checkpoints/epoch_best.pt \
--plat_rep_type=separate

echo "Running Platonic Representation Hypothesis Analysis (Separate) Zero-shot"
python -m ecg_bench.run_plat_rep_hyp \
--ecg_signal \
--llm=llama-3.2-1b-instruct \
--encoder=projection \
--data=ecg-instruct-45k-250-1250 \
--device=cuda:1 \
--peft \
--attention_type=flash_attention_2 \
--system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
--output_hidden_states \
--no_signal \
--plat_rep_type=separate


echo "Running Platonic Representation Hypothesis Analysis (Combined)"
python -m ecg_bench.run_plat_rep_hyp \
--ecg_signal \
--llm=llama-3.2-1b-instruct \
--encoder=projection \
--data=ecg-instruct-45k-250-1250 \
--device=cuda:1 \
--peft \
--attention_type=flash_attention_2 \
--system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
--output_hidden_states \
--elm_ckpt=./ecg_bench/runs/training/elm/0/checkpoints/epoch_best.pt \
--plat_rep_type=combined


echo "Running Platonic Representation Hypothesis Analysis (Combined) Zero-shot"
python -m ecg_bench.run_plat_rep_hyp \
--ecg_signal \
--llm=llama-3.2-1b-instruct \
--encoder=projection \
--data=ecg-instruct-45k-250-1250 \
--device=cuda:1 \
--peft \
--attention_type=flash_attention_2 \
--system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
--output_hidden_states \
--plat_rep_type=combined

echo "----------------------------------------"
echo "ecg ptb qa xl"
echo "Running Platonic Representation Hypothesis Analysis (Separate)"
python -m ecg_bench.run_plat_rep_hyp \
--ecg_signal \
--llm=llama-3.2-1b-instruct \
--encoder=projection \
--data=ecg-qa-ptbxl-250-1250 \
--device=cuda:1 \
--peft \
--attention_type=flash_attention_2 \
--system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
--output_hidden_states \
--no_signal \
--elm_ckpt=./ecg_bench/runs/training/elm/1/checkpoints/epoch_best.pt \
--plat_rep_type=separate

echo "Running Platonic Representation Hypothesis Analysis (Separate) Zero-shot"
python -m ecg_bench.run_plat_rep_hyp \
--ecg_signal \
--llm=llama-3.2-1b-instruct \
--encoder=projection \
--data=ecg-qa-ptbxl-250-1250 \
--device=cuda:1 \
--peft \
--attention_type=flash_attention_2 \
--system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
--output_hidden_states \
--no_signal \
--plat_rep_type=separate


echo "Running Platonic Representation Hypothesis Analysis (Combined)"
python -m ecg_bench.run_plat_rep_hyp \
--ecg_signal \
--llm=llama-3.2-1b-instruct \
--encoder=projection \
--data=ecg-qa-ptbxl-250-1250 \
--device=cuda:1 \
--peft \
--attention_type=flash_attention_2 \
--system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
--output_hidden_states \
--elm_ckpt=./ecg_bench/runs/training/elm/1/checkpoints/epoch_best.pt \
--plat_rep_type=combined


echo "Running Platonic Representation Hypothesis Analysis (Combined) Zero-shot"
python -m ecg_bench.run_plat_rep_hyp \
--ecg_signal \
--llm=llama-3.2-1b-instruct \
--encoder=projection \
--data=ecg-qa-ptbxl-250-1250 \
--device=cuda:1 \
--peft \
--attention_type=flash_attention_2 \
--system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
--output_hidden_states \
--plat_rep_type=combined
