from datasets import load_dataset
from tqdm import tqdm
from collections import Counter, defaultdict

from ecg_bench.configs.constants import HF_CACHE_DIR
from ecg_bench.configs.config import get_args
from ecg_bench.dataloaders.build_dataloader import BuildDataLoader

if __name__ == "__main__":
    mode = "train"
    args = get_args("train")
    dataloader_builder = BuildDataLoader(mode, args)

    data = load_dataset(f"willxxy/{args.data}", split=f"fold{args.fold}_train", cache_dir=HF_CACHE_DIR).with_transform(
        dataloader_builder.decode_batch
    )

    counter = Counter()
    samples = defaultdict(list)

    max_iters = 20000

    for idx, item in enumerate(tqdm(data, desc="Counting categories + collecting samples")):
        if idx >= max_iters:
            break

        cat, question, answer = item["text"]

        if isinstance(answer, list):
            answer = " ".join(answer)

        counter[cat] += 1

        if len(samples[cat]) < 5:
            samples[cat].append((question, answer))

    print("\n=== CATEGORY COUNTS ===")
    for c, n in counter.items():
        print(f"{c}: {n}")

    print("\n=== SAMPLE QUESTIONS + ANSWERS (up to 5 each) ===")
    for c, qa_list in samples.items():
        print(f"\n--- {c} ---")
        for q, a in qa_list:
            print(f"Q: {q}")
            print(f"A: {a}")
            print()
