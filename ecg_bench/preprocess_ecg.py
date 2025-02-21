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
    parser.add_argument('--data', type = str, default = None, help = 'Please choose the base dataset')
    parser.add_argument('--map_data', type = str, default = None, help = 'Please choose the external dataset to map to base dataset')
    parser.add_argument('--seg_len', type = int, default = 500, help = 'Please choose the segment length')
    parser.add_argument('--target_sf', type = int, default = 250, help = 'Please choose the target sampling frequency')
    parser.add_argument('--num_cores', type = int, default = 12, help = 'Please choose the number of cores, usually 4-6 is enough')
    parser.add_argument('--num_percentiles', type = int, default = 300000, help = 'Please choose the number of samples for calculating percentiles')
    parser.add_argument('--num_tok_samples', type = int, default = 450000, help = 'Please choose the number of samples for training the tokenizer')
    parser.add_argument('--max_clusters', type = int, default = 200, help = 'Please choose the maximum number of clusters to consider during sampling for training tokenizer')
    parser.add_argument('--dev', action = 'store_true', default = False, help = 'Use this flag to run the script in development mode')
    parser.add_argument('--toy', action = 'store_true', default = False, help = 'Use this flag to create a toy dataset')
    return parser.parse_args()
    
def main(args: argparse.Namespace):
    fm = FileManager()
    preprocessor = PreprocessECG(args, fm)
    
    if (args.data == 'mimic' or args.data == 'ptb' or args.data == 'code15') and (args.map_data == None):
        preprocessor.preprocess_batch()
        if args.data == 'mimic' and args.seg_len == 2500:
            preprocessor.get_percentiles()
            preprocessor.stratified_sampling()
    
    if fm.ensure_directory_exists(folder = f'./data/{args.data}/preprocessed_{args.seg_len}_{args.target_sf}'):
        if args.map_data != None:
            preprocessor.map_external_datasets()
    

if __name__ == '__main__':
    main(get_args())