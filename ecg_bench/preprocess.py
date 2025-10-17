from ecg_bench.preprocessors.base_ecg import BaseECG
from ecg_bench.preprocessors.mix_ecg import MixECG
from ecg_bench.preprocessors.map_ecg import MapECG
from ecg_bench.preprocessors.prepare_df import PrepareDF
from ecg_bench.configs.config import get_args
from ecg_bench.utils.file_manager import FileManager
from ecg_bench.configs.constants import CONFIG_DIR

if __name__ == "__main__":
    fm = FileManager()
    args = get_args("preprocess")
    config_save_path = f"{CONFIG_DIR}/preprocessing"
    fm.save_config(config_save_path, args)
    if args.preprocess:
        df_preparer = PrepareDF(args, fm)
        if not fm.ensure_directory_exists(f"./data/{args.base_data}/{args.base_data}.csv"):
            df_preparer.prepare_df()
        df = df_preparer.get_df()
        base_ecg = BaseECG(args, fm, df)
        base_ecg.preprocess_batch()
    if args.map_data:
        mapper = MapECG(args, fm)
        mapper.map_data()
    if args.mix_data:
        mixer = MixECG(args, fm)
        mixer.mix_data()
