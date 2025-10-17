from ecg_bench.ecg_tokenizers.build_ecg_tokenizers import BuildECGByte
from ecg_bench.ecg_tokenizers.ecg_sampler import SampleECG
from ecg_bench.configs.config import get_args
from ecg_bench.utils.file_manager import FileManager
from ecg_bench.configs.constants import CONFIG_DIR

if __name__ == "__main__":
    mode = "ecg_tokenizer"
    fm = FileManager()
    args = get_args(mode)
    config_save_path = f"{CONFIG_DIR}/ecg_tokenizers"
    fm.ensure_directory_exists(folder=config_save_path)
    fm.save_config(config_save_path, args)

    if args.path_to_ecg_npy:
        SampleECG(args).random_sampling()
    if args.ecg_tokenizer or args.sampled_file:
        BuildECGByte(args, mode)
