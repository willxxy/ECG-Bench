# BASE_DATA_VALUES=("ptb" "mimic" "code15" "cpsc" "csn")
# SEG_LENS=(1250 2500 500)

# for base_data in "${BASE_DATA_VALUES[@]}"; do
#   for seg_len in "${SEG_LENS[@]}"; do
#     if [ "$base_data" = "mimic" ]; then
#       echo "Sampling $base_data with seg_len=$seg_len"
#       python preprocess_ecg.py \
#         --base_data="$base_data" \
#         --seg_len="$seg_len" \
#         --preprocess_files \
#         --sample_files --random_sampling
#     else
#       echo "Preprocessing $base_data with seg_len=$seg_len"
#       python preprocess_ecg.py \
#         --base_data="$base_data" \
#         --seg_len="$seg_len" \
#         --preprocess_files
#     fi
#   done
# done

python ecg_bench/preprocess.py \
  --map_data="ecg-qa_ptbxl" \
  --segment_len=1250 \
  --target_sf=250 