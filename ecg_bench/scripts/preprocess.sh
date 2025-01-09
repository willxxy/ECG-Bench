CUDA_VISIBLE_DEVICES=1 python preprocess_ecg.py --data=ptb
python preprocess_ecg.py --data=mimic --toy

python preprocess_ecg.py --data=mimic --seg_len=2500 --toy
# CUDA_VISIBLE_DEVICES=1 python preprocess_ecg.py --data=ptb --seg_len=2500