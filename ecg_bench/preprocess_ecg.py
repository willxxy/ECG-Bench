import argparse
from ecg_bench.utils.dir_file_utils import FileManager  # Updated import path
from ecg_bench.utils.preprocess_utils import PreprocessECG

def get_args():
    parser = argparse.ArgumentParser(description = None)
    parser.add_argument('--data', type = str, default = None, help = 'Please choose the dataset')
    parser.add_argument('--seg_len', type = int, default = None, help = 'Please choose the segment length')
    return parser.parse_args()
    
def main(args: argparse.Namespace):
    PreprocessECG(args, FileManager).preprocess()

if __name__ == '__main__':
    main(get_args())