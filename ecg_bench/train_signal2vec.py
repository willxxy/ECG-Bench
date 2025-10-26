# main.py
from ecg_bench.ecg_tokenizers.build_signal2vec import BuildSignal2Vec
from ecg_bench.configs.config import get_args
from ecg_bench.utils.file_manager import FileManager
from ecg_bench.configs.constants import CONFIG_DIR
from ecg_bench.ecg_tokenizers.build_ecg_tokenizers import BuildECGByte

if __name__ == "__main__":
    mode = "signal2vec"
    fm = FileManager()
    args = get_args(mode)
    args.save_path = f"{CONFIG_DIR}/signal2vec"

    fm.ensure_directory_exists(folder=args.save_path)
    fm.save_config(args.save_path, args)
    ecg_byte_builder = BuildECGByte(args, mode)
    s2v = BuildSignal2Vec(fm, ecg_byte_builder, args)
    # s2v.train()

    emb = BuildSignal2Vec.load_embeddings(f"{args.save_path}/embeddings.pt")
    example_id = 1727
    print("Vector for token 1727:", emb[example_id][:8])
