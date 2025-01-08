import argparse
import os

from ecg_bench.utils.dir_file_utils import FileManager  # Updated import path
from ecg_bench.utils.preprocess_utils import PreprocessECG

def get_args():
    parser = argparse.ArgumentParser(description = None)
    parser.add_argument('--data', type = str, default = None, help = 'Please choose the dataset')
    parser.add_argument('--seg_len', type = int, default = 500, help = 'Please choose the segment length')
    parser.add_argument('--target_sf', type = int, default = 250, help = 'Please choose the target sampling frequency')
    return parser.parse_args()
    
def main(args: argparse.Namespace):
    PreprocessECG(args, FileManager).preprocess_instance()
    
    np_file = FileManager.open_npy(f"./data/{args.data}/preprocessed_{args.seg_len}_{args.target_sf}/{os.listdir(f'./data/{args.data}/preprocessed_{args.seg_len}_{args.target_sf}')[0]}")
    print(np_file)
    print(np_file['ecg'].shape, np_file['ecg'].dtype)
    print(np_file['report'])
    print('---------------------------------')
    
    

if __name__ == '__main__':
    main(get_args())