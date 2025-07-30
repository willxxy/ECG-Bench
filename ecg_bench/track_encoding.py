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
    
    global_id_to_color = {}
    count = 0
    for path in list_of_paths[:args.num_plots]:
        sample_signal = fm.open_npy(path)['ecg']
        for lead in range(sample_signal.shape[0]):
            single_lead = sample_signal[lead]
            if args.instance_normalize:
                norm_single_lead, _, _ = ecg_tokenizer_utils.instance_normalize(single_lead)
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
    print('Finished Tracking Encoding!')

if __name__ == '__main__':
    main(get_args())