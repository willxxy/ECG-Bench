# CUDA_VISIBLE_DEVICES=1 python preprocess_ecg.py --data=ptb --seg_len=500
# python preprocess_ecg.py --data=mimic --toy --seg_len=500

# python preprocess_ecg.py --data=mimic --seg_len=2500 --toy
# CUDA_VISIBLE_DEVICES=1 python preprocess_ecg.py --data=ptb --seg_len=2500

python preprocess_ecg.py --data=mimic --map_data=ecg_instruct_45k --seg_len=500
python preprocess_ecg.py --data=mimic --map_data=pretrain_mimic --seg_len=500