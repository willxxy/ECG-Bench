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
    
        
    ## NOW ADD SAMPLING AND PERCENTILES
    # np_file = FileManager.open_npy(f"./data/{args.data}/preprocessed_{args.seg_len}_{args.target_sf}/{os.listdir(f'./data/{args.data}/preprocessed_{args.seg_len}_{args.target_sf}')[0]}")
    # print(np_file)
    # print(np_file['ecg'].shape, np_file['ecg'].dtype)
    # print(np_file['report'])
    # print('---------------------------------')
    
    # percentiles = FileManager.open_npy(f"./data/{args.data}_percentiles_{args.seg_len}_{args.target_sf}_{args.num_percentiles}.npy")
    # print(percentiles['p1'])
    # print(percentiles['p99'])
    
    # def normalize_ecg(ecg_data, percentiles):
    #     p1 = percentiles['p1']
    #     p99 = percentiles['p99']
    #     return (ecg_data - p1) / (p99 - p1)

    # # Assuming np_file['ecg'] is the 2500, 12 array
    # normalized_ecg = normalize_ecg(np_file['ecg'], percentiles)
    # print(normalized_ecg)
    
    # import matplotlib.pyplot as plt

    # def plot_ecg_lead(ecg_data, output_file):
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(ecg_data)
    #     plt.title(f'ECG')
    #     plt.xlabel('Sample')
    #     plt.ylabel('Amplitude')
    #     plt.grid(True)
    #     plt.savefig(output_file)
    #     plt.close()

    # # Plot and save the first lead (index 0)
    # plot_ecg_lead(np_file['ecg'][0], output_file=f"./data/unnomr_lead_1_plot.png")
    # plot_ecg_lead(normalized_ecg[0], output_file=f"./data/norm_lead_1_plot.png")
    
    # original_ecg, _ = FileManager.open_ecg('./data/mimic/files/p1012/p10129438/s42988869/42988869')
    # print(original_ecg.shape)
    # plot_ecg_lead(original_ecg[:, 0], output_file=f"./data/orig_lead_1_plot.png")
    
    

if __name__ == '__main__':
    main(get_args())