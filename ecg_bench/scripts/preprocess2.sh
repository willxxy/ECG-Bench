#!/bin/bash

### BASE DATA
# Define an array of base_data values to process
BASE_DATA_VALUES=("code15" "cpsc" "csn" "ptb" "mimic")

# Loop through each base_data value
for base_data in "${BASE_DATA_VALUES[@]}"; do
    echo "Processing base_data: $base_data"
    python preprocess_ecg_ref.py --base_data=$base_data --seg_len=1250 --dev
    
    # Special case for mimic: also process with seg_len=2500
    if [ "$base_data" = "mimic" ]; then
        echo "Processing mimic with seg_len=2500"
        python preprocess_ecg_ref.py --base_data=$base_data --seg_len=2500 --dev
    fi
done


MAP_DATA_VALUES=("ecg_bench_pulse" "ecg_instruct_pulse" "pretrain_mimic" "ecg_instruct_45k" "ecg-qa_ptbxl" "ecg-qa_mimic-iv-ecg")

for map_data in "${MAP_DATA_VALUES[@]}"; do
    echo "Processing map_data: $map_data"
    python preprocess_ecg_ref.py --map_data=$map_data --seg_len=1250
done
