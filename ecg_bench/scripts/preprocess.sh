#!/usr/bin/env bash
BASE_DATA_VALUES=("ptb" "mimic" "code15" "cpsc" "csn")
SEG_LENS=(1250 2500 500)

for base_data in "${BASE_DATA_VALUES[@]}"; do
  for seg_len in "${SEG_LENS[@]}"; do
    if [ "$base_data" = "mimic" ]; then
      echo "Sampling $base_data with seg_len=$seg_len"
      python preprocess_ecg.py \
        --base_data="$base_data" \
        --seg_len="$seg_len" \
        --preprocess_files \
        --sample_files --random_sampling
    else
      echo "Preprocessing $base_data with seg_len=$seg_len"
      python preprocess_ecg.py \
        --base_data="$base_data" \
        --seg_len="$seg_len" \
        --preprocess_files
    fi
  done
done

### MAPPING DATA
MAP_DATA_VALUES=("ecg_bench_pulse" "ecg_instruct_pulse" "pretrain_mimic" "ecg_instruct_45k" "ecg-qa_ptbxl" "ecg-qa_mimic-iv-ecg")

for map_data in "${MAP_DATA_VALUES[@]}"; do
    echo "Processing map_data: $map_data"
    python preprocess_ecg.py --map_data=$map_data --seg_len=1250
    python preprocess_ecg.py --map_data=$map_data --seg_len=500
    python preprocess_ecg.py --map_data=$map_data --seg_len=2500
done



# ### MIXING DATA Example
# MIX_DATA_VALUES=("ecg_instruct_45k_mapped_1250,ecg_bench_pulse_mapped_1250")
# for mix_data in "${MIX_DATA_VALUES[@]}"; do
#     echo "Processing mix_data: $mix_data"
#     python preprocess_ecg.py --mix_data=$mix_data --dev
# done
