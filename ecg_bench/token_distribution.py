import argparse
import glob

from ecg_bench.utils.ecg_tokenizer_utils import ECGByteTokenizer
from ecg_bench.utils.dir_file_utils import FileManager
from ecg_bench.utils.viz_utils import VizUtil
from ecg_bench.config import get_args

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