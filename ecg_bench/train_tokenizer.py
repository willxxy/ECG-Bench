import numpy as np
import time
import argparse

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_merges', type = int, default = 3500, help='Please choose the vocabulary size')
    parser.add_argument('--sampled_files', type=str, default = None, help='Please specify the path to the .txt file of sampled ecgs')
    parser.add_argument('--num_processes', type=int, default=2, help='Number of processes for multiprocessing')
    parser.add_argument('--percentiles', type=str, default = None, help = 'Please specify the path to the calculated percentiles')
    parser.add_argument('--train', action = 'store_true', default = None, help = 'Please specify whether to train the tokenizer')
    parser.add_argument('--loaded', type = str, default = None, help = 'If you want to just load the tokenizer, please specify the path to the .pkl file.')
    return parser.parse_args()

def main(args: argparse.Namespace):
    
    pass

if __name__ == '__main__':
    main(get_args())
