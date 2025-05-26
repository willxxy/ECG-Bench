import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

from ecg_bench.utils.dir_file_utils import FileManager
from ecg_bench.utils.preprocess_utils import PrepareDF, PreprocessBaseECG, PreprocessMapECG, SampleBaseECG, PreprocessMixECG
from ecg_bench.utils.rag_utils import RAGECGDatabase
from ecg_bench.config import get_args
    
def main(args: argparse.Namespace):
    fm = FileManager()
    
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
                    
    if args.create_rag_db != None or args.load_rag_db != None or args.load_rag_db_idx != None:
        rag_db = RAGECGDatabase(args, fm)
        rag_db.test_search()
        
    if args.map_data != None:
        preprocess_map_ecg = PreprocessMapECG(args, fm)
        preprocess_map_ecg.map_data()
        
    if args.mix_data != None:
        preprocess_mix_ecg = PreprocessMixECG(args, fm)
        preprocess_mix_ecg.mix_data()
    
if __name__ == '__main__':
    main(get_args())