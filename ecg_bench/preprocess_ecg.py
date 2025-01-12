import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

from ecg_bench.utils.dir_file_utils import FileManager  # Updated import path
from ecg_bench.utils.preprocess_utils import PreprocessECG

def get_args():
    parser = argparse.ArgumentParser(description = None)
    parser.add_argument('--data', type = str, default = None, help = 'Please choose the dataset')
    parser.add_argument('--seg_len', type = int, default = 500, help = 'Please choose the segment length')
    parser.add_argument('--target_sf', type = int, default = 250, help = 'Please choose the target sampling frequency')
    parser.add_argument('--num_cores', type = int, default = 4, help = 'Please choose the number of cores, usually 4-6 is enough')
    parser.add_argument('--num_percentiles', type = int, default = 300000, help = 'Please choose the number of samples for calculating percentiles')
    parser.add_argument('--num_tok_samples', type = int, default = 300000, help = 'Please choose the number of samples for training the tokenizer')
    parser.add_argument('--max_clusters', type = int, default = 200, help = 'Please choose the maximum number of clusters to consider during sampling for training tokenizer')
    parser.add_argument('--dev', action = 'store_true', default = False, help = 'Use this flag to run the script in development mode')
    parser.add_argument('--toy', action = 'store_true', default = False, help = 'Use this flag to create a toy dataset')
    return parser.parse_args()
    
def main(args: argparse.Namespace):
    if args.data == 'mimic' or args.data == 'ptb':
        preprocessor = PreprocessECG(args, FileManager)
        preprocessor.preprocess_batch()
        
        if args.data == 'mimic' and args.seg_len == 2500:
            # Perform sampling and calculate percentiles for mimic unsegmented data
            preprocessor.get_percentiles()
            preprocessor.stratified_sampling()

if __name__ == '__main__':
    main(get_args())