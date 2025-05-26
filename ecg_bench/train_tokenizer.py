import numpy as np
import time
import argparse

from ecg_bench.utils.dir_file_utils import FileManager
from ecg_bench.utils.ecg_tokenizer_utils import ECGByteTokenizer
from ecg_bench.config import get_args

def main(args: argparse.Namespace):
    ecg_tokenizer = ECGByteTokenizer(args, FileManager)
    
    if args.train_tokenizer:
        ecg_tokenizer.train_tokenizer()
    if args.ecg_tokenizer != None:
        ecg_tokenizer.verify_tokenizer()

if __name__ == '__main__':
    main(get_args())
