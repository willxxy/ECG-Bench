import argparse
import os
import pandas as pd
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

from ecg_bench.utils.dir_file_utils import FileManager
from ecg_bench.utils.preprocess_utils import PrepareDF, PreprocessBaseECG, PreprocessMapECG, SampleBaseECG
from ecg_bench.utils.rag_utils import RAGECGDatabase

def get_args():
    parser = argparse.ArgumentParser(description = "ECG preprocessing pipeline")
    parser.add_argument('--base_data', type = str, default = None, help = 'Base dataset to preprocess')
    parser.add_argument('--map_data', type = str, default = None, help = 'External dataset to map to base dataset')
    parser.add_argument('--seg_len', type = int, default = 500, help = 'Segment length')
    parser.add_argument('--target_sf', type = int, default = 250, help = 'Target sampling frequency')
    parser.add_argument('--num_cores', type = int, default = 12, help = 'Number of cores for parallel processing')
    parser.add_argument('--num_percentiles', type = int, default = 300000, help = 'Number of samples for calculating percentiles')
    parser.add_argument('--num_tok_samples', type = int, default = 300000, help = 'Number of samples for training the tokenizer')
    parser.add_argument('--random_sampling', action = 'store_true', default = False, help = 'Use random sampling')
    parser.add_argument('--stratified_sampling', action = 'store_true', default = False, help = 'Use stratified sampling')
    parser.add_argument('--sample_percentiles', action = 'store_true', default = False, help = 'Sample percentiles')
    parser.add_argument('--sample_files', action = 'store_true', default = False, help = 'Sample files')
    parser.add_argument('--preprocess_files', action = 'store_true', default = False, help = 'Preprocess files')
    parser.add_argument('--max_clusters', type = int, default = 200, help = 'Maximum number of clusters for tokenizer training')
    parser.add_argument('--dev', action = 'store_true', default = False, help = 'Run in development mode')
    parser.add_argument('--toy', action = 'store_true', default = False, help = 'Create a toy dataset')
    parser.add_argument('--create_rag_db', action = 'store_true', default = False, help = 'Create a RAG database')
    parser.add_argument('--load_rag_db', type = str, default = None, help = 'Load a RAG database')
    parser.add_argument('--load_rag_db_idx', type = str, default = None, help = 'Load a RAG database index')
    return parser.parse_args()
    
def main(args: argparse.Namespace):
    fm = FileManager()
    if args.map_data == None:
        if args.preprocess_files:
            df_preparer = PrepareDF(args, fm)
            if not fm.ensure_directory_exists(file=f'./data/{args.base_data}/{args.base_data}.csv'):
                df_preparer.prepare_df()
            df = df_preparer.get_df()
            preprocess_base_ecg = PreprocessBaseECG(args, fm, df)
            preprocess_base_ecg.preprocess_batch()
        
        if args.sample_percentiles or args.sample_files:
            sample_base_ecg = SampleBaseECG(args, fm)
            if args.sample_percentiles:
                if not fm.ensure_directory_exists(file = f'./data/{args.base_data}_percentiles_{args.seg_len}_{args.target_sf}_{args.num_percentiles}.npy'):
                    sample_base_ecg.get_percentiles()
            if args.sample_files:
                if args.random_sampling:
                    if not fm.ensure_directory_exists(file = f'./data/sampled_{args.num_tok_samples}_random.txt'):
                        sample_base_ecg.random_sampling()
                elif args.stratified_sampling:
                    if not fm.ensure_directory_exists(file = f'./data/sampled_{args.num_tok_samples}_{args.max_clusters}.txt'):
                        sample_base_ecg.stratified_sampling()
        elif args.create_rag_db:
            rag_db = RAGECGDatabase(args, fm)
            rag_db.test_search()
    else:
        preprocess_map_ecg = PreprocessMapECG(args, fm)
        preprocess_map_ecg.map_data()
    
if __name__ == '__main__':
    main(get_args())