import argparse
import json
import pathlib
import re
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from huggingface_hub import login


# ----------  splitting helper (your original logic, un-modified) ----------
class Splitter:
    _PATIENT_EXTRACTORS = {
        "mimic": lambda p: max(re.findall(r"p\\d+", p), key=len, default=None),
        "ptb":   lambda p: (m := re.search(r"_([0-9]+)_hr", p)) and m.group(1),
        # add more here if you get new datasets
    }

    @staticmethod
    def _dataset_from_path(path: str) -> str:
        parts = pathlib.Path(path).parts
        try:
            return parts[parts.index("data") + 1].lower()
        except (ValueError, IndexError):
            return ""

    def __init__(self, seed: int):
        self.args = SimpleNamespace(seed=seed)   # only needs .seed

    # ---------------------------------------------------------------------
    def _patient_id(self, ecg_path: str):
        ds = self._dataset_from_path(ecg_path)
        extractor = self._PATIENT_EXTRACTORS.get(ds)
        return extractor(ecg_path) if extractor else None

    # ---------------------------------------------------------------------
    def split_dataset(self, data, train_ratio: float = 0.7):
        rng = np.random.default_rng(self.args.seed)
        n_total = len(data)
        n_train_target = int(round(n_total * train_ratio))

        groups = []
        loose  = []
        pid2idx = defaultdict(list)

        for idx, item in enumerate(data):
            pid = self._patient_id(item["ecg_path"])
            if pid:
                pid2idx[pid].append(idx)
            else:
                loose.append([idx])

        groups.extend(pid2idx.values())
        groups.extend(loose)

        rng.shuffle(groups)

        group_sizes = [len(g) for g in groups]

        train_groups_idx = []
        test_groups_idx = []
        current_train_size = 0

        sorted_indices = sorted(range(len(groups)), key=lambda i: -group_sizes[i])

        for idx in sorted_indices:
            if current_train_size + group_sizes[idx] <= n_train_target:
                train_groups_idx.append(idx)
                current_train_size += group_sizes[idx]
            else:
                test_groups_idx.append(idx)

        best_diff = abs(current_train_size - n_train_target)
        improved = True
        max_iterations = 0
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i, train_idx in enumerate(train_groups_idx):
                for j, test_idx in enumerate(test_groups_idx):
                    new_train_size = current_train_size - group_sizes[train_idx] + group_sizes[test_idx]
                    new_diff = abs(new_train_size - n_train_target)

                    if new_diff < best_diff:
                        train_groups_idx[i] = test_idx
                        test_groups_idx[j] = train_idx
                        current_train_size = new_train_size
                        best_diff = new_diff
                        improved = True
                        break
                if improved:
                    break

        train_idx = []
        test_idx = []

        for idx in train_groups_idx:
            train_idx.extend(groups[idx])

        for idx in test_groups_idx:
            test_idx.extend(groups[idx])

        print(f"  Target train size: {n_train_target}, Actual: {len(train_idx)} (diff: {abs(len(train_idx) - n_train_target)})")

        return [data[i] for i in train_idx], [data[i] for i in test_idx]


def encode_row(item: dict) -> dict:
    if "text" in item:
        item = dict(item)  # shallow copy
        item["text"] = json.dumps(item["text"], ensure_ascii=False, separators=(",", ":"))
    return item

def decode_batch(batch: dict) -> dict:
    # Restore original Python object at read time
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
    ap.add_argument("--input", help="flat JSON created by your mapping step")
    ap.add_argument("--output", help="where to write the folds JSON")
    ap.add_argument("--folds",  type=int, default=5)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--repo_id", type=str, default="willxxy/test-ecg")
    ap.add_argument("--load", action="store_true", default = None, help="load back the dataset after pushing to HF")
    args = ap.parse_args()

    if args.load:
        # --- 3) load back and lazily decode so your dataloader sees originals -----
        ds_all = load_dataset(args.repo_id).with_transform(decode_batch)
        # Access: ds_all["fold3_train"], ds_all["fold5_test"], ...

        fold1_train = load_dataset(args.repo_id, split="fold1_train").with_transform(decode_batch)
        print("fold1_train example text (restored):", fold1_train[0]["text"])
        print("fold1_train keys:", fold1_train[0].keys())
        print("len fold1_train:", len(fold1_train))

        fold1_test = load_dataset(args.repo_id, split="fold1_test").with_transform(decode_batch)
        print("fold1_test example text (restored):", fold1_test[0]["text"])
        print("fold1_test keys:", fold1_test[0].keys())
        print("len fold1_test:", len(fold1_test))

        print("type", type(fold1_train))
        fold1_train_small = fold1_train.select(range(10))
        fold1_test_small  = fold1_test.select(range(10))

        print(len(fold1_train_small))   # → 10
        print(fold1_train_small[0])     # row 0 still decodes through your transform
        print(len(fold1_test_small))   # → 10
        print(fold1_test_small[0])     # row 0 still decodes through your transform
    else:
        with open(args.input) as f:
            data = json.load(f)

        print("Dataset Length:", len(data))

        folds_dict, summary = {}, []
        for k in range(args.folds):
            splitter = Splitter(seed=args.seed + k)
            train, test = splitter.split_dataset(data, train_ratio=args.train_ratio)

            train_pids = {
                splitter._patient_id(d["ecg_path"])
                for d in train
                if splitter._patient_id(d["ecg_path"]) is not None
            }
            test_pids = {
                splitter._patient_id(d["ecg_path"])
                for d in test
                if splitter._patient_id(d["ecg_path"]) is not None
            }
            overlap = train_pids & test_pids
            assert not overlap, (
                f"Fold {k+1}: patient-ID overlap between train & test → {sorted(overlap)}"
            )

            folds_dict[f"fold{k+1}"] = {"train": train, "test": test}

            print(f"Fold {k+1}: train={len(train):>5}, test={len(test):>5}")
            if train: print("  ▸ sample train ecg_path:", train[0]["ecg_path"])
            if test:  print("  ▸ sample test  ecg_path:", test[0]["ecg_path"])
            summary.append((len(train), len(test)))

        with open(args.output, "w") as f:
            json.dump(folds_dict, f, indent=2)

        print("\nSummary (train / test sizes per fold)")
        for i, (tr, te) in enumerate(summary, 1):
            print(f"  Fold {i}: {tr} / {te}")
        print(f"\n✓ Saved {args.folds} folds to {args.output}")


        # --- 0) authenticate -------------------------------------------------------
        import os
        token = os.getenv("HF_TOKEN")           # set once in your shell
        if not token:
            raise RuntimeError("export HF_TOKEN=hf_xxx and retry")

        login(token=token, new_session=False)   # no prompt, no network check

        # --- 1) build HF DatasetDict with JSON-encoded `text` ---------------------
        # Reuse folds_dict already in memory (no need to re-read from disk)
        splits = {}

        features = Features({
            "ecg_path": Value("string"),
            "text"    : Value("string"),  # JSON string on disk
            "name"    : Value("string"),
        })

        for fold_name, parts in folds_dict.items():      # fold1 … fold5
            train_enc = [encode_row(d) for d in parts["train"]]
            test_enc  = [encode_row(d) for d in parts["test"]]
            splits[f"{fold_name}_train"] = Dataset.from_list(train_enc, features=features)
            splits[f"{fold_name}_test"]  = Dataset.from_list(test_enc,  features=features)

        dataset = DatasetDict(splits)

        # # --- 2) push ---------------------------------------------------------------
        dataset.push_to_hub(args.repo_id, token=token)


if __name__ == "__main__":
    main()
