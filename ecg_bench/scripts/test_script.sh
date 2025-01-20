#!/bin/bash

# Global settings
GPUS="4,5"
SINGLE_GPU="cuda:4"
MODEL_NAME="all"  # Options: all, llama-3.2-1b, clip, vit

# Function to run end2end tests
run_end2end() {
    echo "Training End2End for $MODEL_NAME"
    
    if [ "$MODEL_NAME" = "llama-3.2-1b" ]; then
        # Distributed training
        python main_end2end.py \
        --data=pretrain_mimic_mapped \
        --model=$MODEL_NAME \
        --gpus=$GPUS \
        --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
        --ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
        --dis \
        --peft \
        --train=end2end \
        --dev

        echo "----------------------------------------"

        # Single GPU training
        python main_end2end.py \
        --data=pretrain_mimic_mapped \
        --model=$MODEL_NAME \
        --device=$SINGLE_GPU \
        --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
        --ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
        --peft \
        --train=end2end \
        --dev

        # Inference
        echo "----------------------------------------"
        echo "Inferencing End2End"
        
        python main_end2end.py \
        --data=pretrain_mimic_mapped \
        --model=$MODEL_NAME \
        --device=$SINGLE_GPU \
        --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
        --ecg_tokenizer=./data/tokenizer_3500_300000.pkl \
        --peft \
        --inference=end2end \
        --checkpoint=llama-3.2-1b_2_2_0.0001_0.9_0.99_1e-08_500_0.01 \
        --dev
    fi
}

# Function to run first stage tests
run_first() {
    echo "Training First for $MODEL_NAME"
    
    if [ "$MODEL_NAME" = "clip" ] || [ "$MODEL_NAME" = "vit" ] || [ "$MODEL_NAME" = "merl" ]; then
        # Distributed training
        python main_first.py \
        --data=pretrain_mimic_mapped \
        --model=$MODEL_NAME \
        --gpus=$GPUS \
        --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
        --dis \
        --train=first \
        --dev

        echo "----------------------------------------"

        # Single GPU training
        python main_first.py \
        --data=pretrain_mimic_mapped \
        --model=$MODEL_NAME \
        --device=$SINGLE_GPU \
        --percentiles=./data/mimic_percentiles_2500_250_300000.npy \
        --train=first \
        --dev
    fi
}

# Run appropriate tests based on model name
case $MODEL_NAME in
    "all")
        MODEL_NAME="llama-3.2-1b"
        run_end2end
        echo "=========================================="
        
        for model in "clip" "vit" "merl"; do
            MODEL_NAME=$model
            run_first
            echo "=========================================="
        done
        ;;
    "llama-3.2-1b")
        run_end2end
        ;;
    "clip"|"vit"|"merl")
        run_first
        ;;
    *)
        echo "Invalid MODEL_NAME. Please choose from: all, llama-3.2-1b, clip, merl,or vit"
        exit 1
        ;;
esac