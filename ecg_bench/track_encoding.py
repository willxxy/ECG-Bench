import argparse
import glob

from ecg_bench.utils.ecg_tokenizer_utils import ECGByteTokenizer
from ecg_bench.utils.dir_file_utils import FileManager
from ecg_bench.utils.viz_utils import VizUtil

def get_args():
    parser = argparse.ArgumentParser(description = None)
    parser.add_argument('--ecg_tokenizer', type = str, default = None, help = 'Please choose ECG tokenizer')
    parser.add_argument('--percentiles', type = str, default = None, help='Please specify the path to the percentiles')
    parser.add_argument('--list_of_paths', type = str, default = None, help='Please specify the path to the list of paths')
    parser.add_argument('--num_plots', type = int, default = 10, help='Please specify the number of plots to generate')
    parser.add_argument('--dev', action = 'store_true', default = None, help = 'Use this flag to run the script in development mode')
    return parser.parse_args()

def main(args):
    
    fm, viz = FileManager(), VizUtil()
    ecg_tokenizer_utils = ECGByteTokenizer(args, fm)
    
    list_of_paths = glob.glob(args.list_of_paths)
    
    global_id_to_color = {}
    count = 0
    for path in list_of_paths[:args.num_plots]:
        sample_signal = fm.open_npy(path)['ecg']
        for lead in range(sample_signal.shape[0]):
            single_lead = sample_signal[lead]
            norm_single_lead, _ = ecg_tokenizer_utils.normalize(single_lead)
            single_lead_str = ecg_tokenizer_utils._to_symbol_string(single_lead)
            encoded_ids, segment_map = ecg_tokenizer_utils.track_encoding(single_lead_str)
            
            new_ids = set(encoded_ids) - set(global_id_to_color.keys())
            if new_ids:
                new_colors = viz.generate_distinct_colors(len(new_ids))
                global_id_to_color.update(zip(sorted(new_ids), new_colors))
                
            viz.visualize_bpe_encoding(norm_single_lead, encoded_ids, 
                                       segment_map, ecg_tokenizer_utils.lead_order[lead], 
                                       global_id_to_color, count)
        count += 1
            

if __name__ == '__main__':
    main(get_args())