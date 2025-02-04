#!/bin/bash

### BASE DATA
# CUDA_VISIBLE_DEVICES=1 python preprocess_ecg.py --data=ptb --seg_len=500
# python preprocess_ecg.py --data=mimic --toy --seg_len=500
# python preprocess_ecg.py --data=code15 --toy --seg_len=500
python preprocess_ecg.py --data=code15 --map_data=ecg_instruct_pulse --seg_len=500

# python preprocess_ecg.py --data=mimic --seg_len=2500 --toy
# CUDA_VISIBLE_DEVICES=1 python preprocess_ecg.py --data=ptb --seg_len=2500


# ### MAPPING DATA
# python preprocess_ecg.py --data=mimic --map_data=ecg_instruct_45k --seg_len=500
# python preprocess_ecg.py --data=mimic --map_data=pretrain_mimic --seg_len=500

# python preprocess_ecg.py --data=mimic --map_data=ecg-qa_mimic-iv-ecg --seg_len=500
# python preprocess_ecg.py --data=ptb --map_data=ecg-qa_ptbxl --seg_len=500
