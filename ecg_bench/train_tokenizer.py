import numpy as np
import time
import argparse

from ecg_bench.utils.dir_file_utils import FileManager
from ecg_bench.utils.tokenizer_utils import ECGByteTokenizer

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_merges', type = int, default = 3500, help='Please choose the vocabulary size')
    parser.add_argument('--sampled_files', type=str, default = None, help='Please specify the path to the .txt file of sampled ecgs')
    parser.add_argument('--num_processes', type=int, default=2, help='Number of processes for multiprocessing')
    parser.add_argument('--percentiles', type=str, default = None, help = 'Please specify the path to the calculated percentiles')
    parser.add_argument('--train', action = 'store_true', default = None, help = 'Please specify whether to train the tokenizer')
    parser.add_argument('--tokenizer', type = str, default = None, help = 'If you want to just load the tokenizer, please specify the path to the .pkl file.')
    parser.add_argument('--dev', action = 'store_true', default = False, help = 'Use this flag to run the script in development mode')
    return parser.parse_args()

def main(args: argparse.Namespace):
    tokenizer = ECGByteTokenizer(args, FileManager)
    
    if args.train:
        tokenizer.train_tokenizer()
    if args.tokenizer != None:
        tokenizer.verify_tokenizer()

if __name__ == '__main__':
    main(get_args())
