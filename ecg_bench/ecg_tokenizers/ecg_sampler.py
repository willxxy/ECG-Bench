import argparse
import glob
import os
import random


class SampleECG:
    """Main class for sampling base datas for percentiles and sapmles for tokenizer training"""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def random_sampling(self):
        print("Collecting ECG files for random sampling...")
        file_paths = glob.glob(os.path.join(self.args.path_to_ecg_npy, "*.npy"))

        print(f"Randomly sampling {self.args.num_samples} files from {len(file_paths)} total files...")
        sampled_files = random.sample(
            file_paths,
            min(self.args.num_samples, len(file_paths)),
        )

        save_path = f"./ecg_bench/configs/ecg_tokenizers/sampled_{self.args.num_samples}_random.txt"
        print(f"Sampled {len(sampled_files)} files.")
        with open(save_path, "w") as f:
            f.writelines(f"{file}\n" for file in sampled_files)
