import argparse

def get_args():
    parser = argparse.ArgumentParser(description = None)
    parser.add_argument('--data', type = str, default = None, help = 'Please choose the dataset')
    parser.add_argument('--seg_len', type = int, default = None, help = 'Please choose the segment length')
    return parser.parse_args()
    
def main(args: argparse.Namespace):
    pass

if __name__ == '__main__':
    main(get_args())