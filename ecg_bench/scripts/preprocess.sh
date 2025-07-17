# #!/bin/bash

# ### BASE DATA
# BASE_DATA_VALUES=("ptb" "mimic" "code15" "cpsc" "csn")

# for base_data in "${BASE_DATA_VALUES[@]}"; do
#     echo "Processing base_data: $base_data"
#     python preprocess_ecg.py --base_data=$base_data --seg_len=1250 --toy --preprocess_files --dev
#     # python preprocess_ecg.py --base_data=$base_data --seg_len=1250 --toy --load_rag_db=./data/mimic/rag_metadata.json --load_rag_db_idx=./data/mimic/combined.index # FOR RAG
    
#     # Special case for mimic: also process with seg_len=2500
#     if [ "$base_data" = "mimic" ]; then
#         echo "Processing mimic with seg_len=2500"
#         python preprocess_ecg.py --base_data=$base_data --seg_len=2500 --toy --sample_files --random_sampling --dev # or --stratified_sampling
#     fi
# done

# ### MAPPING DATA
# MAP_DATA_VALUES=("ecg_bench_pulse" "ecg_instruct_pulse" "pretrain_mimic" "ecg_instruct_45k" "ecg-qa_ptbxl" "ecg-qa_mimic-iv-ecg")
MAP_DATA_VALUES=("ecg_instruct_45k")

for map_data in "${MAP_DATA_VALUES[@]}"; do
    echo "Processing map_data: $map_data"
    python preprocess_ecg.py --map_data=$map_data --seg_len=1250
done



# ### MIXING DATA Example
# MIX_DATA_VALUES=("ecg_instruct_45k_mapped_1250,ecg_bench_pulse_mapped_1250")
# for mix_data in "${MIX_DATA_VALUES[@]}"; do
#     echo "Processing mix_data: $mix_data"
#     python preprocess_ecg.py --mix_data=$mix_data --dev
# done

# base_data="mimic"
# python preprocess_ecg.py --base_data=$base_data --seg_len=1250 --create_rag_db