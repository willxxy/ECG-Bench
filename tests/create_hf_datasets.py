import argparse
import json
import pathlib
import re
from collections import defaultdict
import os

import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from huggingface_hub import login


class Splitter:
    _PATIENT_EXTRACTORS = {
        "mimic": lambda p: max(re.findall(r"p\d+", p), key=len, default=None),
        "ptb": lambda p: (m := re.search(r"_([0-9]+)_hr", p)) and m.group(1),
    }

    @staticmethod
    def _dataset_from_path(path: str) -> str:
        parts = pathlib.Path(path).parts
        try:
            return parts[parts.index("data") + 1].lower()
        except (ValueError, IndexError):
            return ""

    def __init__(self, seed: int):
        self.seed = seed

    def _patient_id(self, ecg_path: str):
        ds = self._dataset_from_path(ecg_path)
        extractor = self._PATIENT_EXTRACTORS.get(ds)
        return extractor(ecg_path) if extractor else None

    def split_dataset(self, data, train_ratio: float = 0.7):
        rng = np.random.default_rng(self.seed)
        n_total = len(data)
        n_train_target = int(round(n_total * train_ratio))

        pid2idx = defaultdict(list)
        loose = []

        for idx, item in enumerate(data):
            pid = self._patient_id(item["ecg_path"])
            if pid:
                pid2idx[pid].append(idx)
            else:
                loose.append([idx])

        groups = list(pid2idx.values()) + loose
        rng.shuffle(groups)

        group_sizes = [len(g) for g in groups]
        sorted_indices = sorted(range(len(groups)), key=lambda i: -group_sizes[i])

        train_groups, test_groups = [], []
        current_train_size = 0

        for idx in sorted_indices:
            if current_train_size + group_sizes[idx] <= n_train_target:
                train_groups.append(idx)
                current_train_size += group_sizes[idx]
            else:
                test_groups.append(idx)

        train_idx = [i for idx in train_groups for i in groups[idx]]
        test_idx = [i for idx in test_groups for i in groups[idx]]

        print(f"  Target: {n_train_target}, Actual: {len(train_idx)} (diff: {abs(len(train_idx) - n_train_target)})")

        return [data[i] for i in train_idx], [data[i] for i in test_idx]


def encode_row(item: dict) -> dict:
    if "text" in item:
        item = dict(item)
        item["text"] = json.dumps(item["text"], ensure_ascii=False, separators=(",", ":"))
    return item


def decode_batch(batch: dict) -> dict:
    if "text" in batch:
        out = []
        for t in batch["text"]:
            try:
                out.append(json.loads(t))
            except Exception:
                out.append(t)
        batch["text"] = out
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo_id", type=str, default="willxxy/test-ecg")
    ap.add_argument("--load", action="store_true")
    args = ap.parse_args()

    if args.load:
        ds = load_dataset(args.repo_id, split="fold1_train").with_transform(decode_batch)
        print(f"fold1_train: {len(ds)} samples")
        print(f"  keys: {list(ds[0].keys())}")
        print(f"  text: {ds[0]['text']}")
        return

    with open(args.input) as f:
        data = json.load(f)

    print(f"Dataset Length: {len(data)}")

    folds_dict = {}
    for k in range(args.folds):
        splitter = Splitter(seed=args.seed + k)
        train, test = splitter.split_dataset(data, train_ratio=args.train_ratio)

        train_pids = {splitter._patient_id(d["ecg_path"]) for d in train if splitter._patient_id(d["ecg_path"])}
        test_pids = {splitter._patient_id(d["ecg_path"]) for d in test if splitter._patient_id(d["ecg_path"])}
        assert not (train_pids & test_pids), f"Fold {k + 1}: patient overlap"

        folds_dict[f"fold{k + 1}"] = {"train": train, "test": test}
        print(f"Fold {k + 1}: train={len(train)}, test={len(test)}")

    with open(args.output, "w") as f:
        json.dump(folds_dict, f, indent=2)

    print(f"\nSaved {args.folds} folds to {args.output}")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("export HF_TOKEN=hf_xxx and retry")

    login(token=token, new_session=False)

    features = Features({
        "ecg_path": Value("string"),
        "text": Value("string"),
        "name": Value("string"),
    })

    splits = {}
    for fold_name, parts in folds_dict.items():
        splits[f"{fold_name}_train"] = Dataset.from_list([encode_row(d) for d in parts["train"]], features=features)
        splits[f"{fold_name}_test"] = Dataset.from_list([encode_row(d) for d in parts["test"]], features=features)

    DatasetDict(splits).push_to_hub(args.repo_id, token=token)


if __name__ == "__main__":
    main()
