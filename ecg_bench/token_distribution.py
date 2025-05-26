import argparse
import glob

from ecg_bench.utils.ecg_tokenizer_utils import ECGByteTokenizer
from ecg_bench.utils.dir_file_utils import FileManager
from ecg_bench.utils.viz_utils import VizUtil
from ecg_bench.config import get_args

def get_args():
    parser = argparse.ArgumentParser(description = None)
    parser.add_argument('--ecg_tokenizer', type = str, default = None, help = 'Please choose ECG tokenizer')
    parser.add_argument('--list_of_paths', type = str, default = None, help='Please specify the path to the list of paths')
    parser.add_argument('--percentiles', type = str, default = None, help='Please specify the path to the percentiles')
    parser.add_argument('--num_processes', type = int, default = 6, help='Please specify the number of processes')
    parser.add_argument('--dev', action = 'store_true', default = None, help = 'Use this flag to run the script in development mode')
    parser.add_argument('--instance_normalize', action = 'store_true', default = None, help = 'Use this flag to run the script in instance normalization mode')
    return parser.parse_args()

def main(args):
    fm, viz = FileManager(), VizUtil()
    ecg_tokenizer_utils = ECGByteTokenizer(args, fm)
    list_of_paths = glob.glob(args.list_of_paths)
    print('Length of list of paths:', len(list_of_paths))
    token_counts, token_lengths = ecg_tokenizer_utils.analyze_token_distribution(list_of_paths)
    print('Plotting token distribution...')
    viz.plot_distributions(token_counts, token_lengths, args.ecg_tokenizer.split('/')[-1].split('_')[1])
    print('Finished Analyzing Token Distribution!')
if __name__ == '__main__':
    main(get_args())