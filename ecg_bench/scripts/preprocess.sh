# python preprocess_ecg.py --data=mimic
# CUDA_VISIBLE_DEVICES=1 python preprocess_ecg.py --data=ptb

python preprocess_ecg.py --data=mimic --seg_len=2500 --dev
# CUDA_VISIBLE_DEVICES=1 python preprocess_ecg.py --data=ptb --seg_len=2500