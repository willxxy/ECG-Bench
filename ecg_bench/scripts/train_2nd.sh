#!/usr/bin/env bash
# ------------------- CONFIGURABLE LISTS -------------------
encoders=("stmem" "merl" "mlae" "mtae" "siglip" "clip" "vit")
encoders_checkpoints=("stmem_256_50_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None" "merl_256_50_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None" "mlae_256_50_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None" "mtae_256_50_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None")
llms=("gemma-2-2b-it" "llama-3.2-1b-instruct" "qwen2.5-1.5b-instruct")
datasets=("ecg-qa_ptbxl-250-1250" "ecg-qa-mimic-iv-ecg-250-1250" "ecg-instruct-45k-250-1250" "ecg-instruct-pulse-250-1250" "pretrain-mimic-250-1250") # add more datasets here
# ----------------------------------------------------------

for data in "${datasets[@]}"; do
    for llm in "${llms[@]}"; do
        for i in "${!encoders[@]}"; do
            encoder="${encoders[$i]}"
            
            # Get the corresponding checkpoint for this encoder
            if [ $i -lt ${#encoders_checkpoints[@]} ]; then
                encoder_checkpoint="${encoders_checkpoints[$i]}"
                checkpoint_path="./runs/ecg-qa_mimic-iv-ecg_mapped_1250/0/$encoder_checkpoint"
            else
                # For encoders without specified checkpoints (siglip, clip, vit)
                # you might want to use a default path or skip the checkpoint parameter
                checkpoint_path=""
            fi
            
            echo "Running with encoder: $encoder, checkpoint: $encoder_checkpoint"
            
            python main.py \
                --data="$data" \
                --model="${encoder}_${llm}" \
                --device=cuda:7 \
                --train=second \
                --batch_size=2 \
                --seg_len=1250 \
                --epochs=1 \
                --peft \
                --instance_normalize \
                --pad_to_max=1024 \
                --attn_implementation=flash_attention_2 \
                --system_prompt=./data/system_prompt_e2e.txt \
                $([ -n "$checkpoint_path" ] && echo "--encoder_checkpoint=$checkpoint_path") \
                --dev
        done
    done
done


models=("vit" "clip" "siglip" )

for model in "${models[@]}"; do
    python main.py \
    --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
    --model=$model \
    --device=cuda:6 \
    --train=first \
    --batch_size=8 \
    --seg_len=1250 \
    --epochs=2 \
    --instance_normalize \
    --attn_implementation=flash_attention_2 \
    --image \
    --log
done