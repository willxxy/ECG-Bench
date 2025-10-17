import glob
import numpy as np
import json
import os
import random
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import pandas as pd
import wfdb
import yaml
import argparse

from ecg_bench.utils.gpu_setup import is_main, barrier, broadcast_value


class FileManager:
    """A class for managing file operations and directory handling."""

    @staticmethod
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

    @staticmethod
    def save_config(save_path: Union[str, Path], args: argparse.Namespace):
        args_dict = {k: v for k, v in vars(args).items() if not k.startswith("_")}
        with open(f"{save_path}/config.yaml", "w") as f:
            yaml.dump(args_dict, f, default_flow_style=False)

    @staticmethod
    def open_json(path: Union[str, Path]) -> dict:
        """Load and parse a JSON file."""
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def save_json(data: dict, path: Union[str, Path]):
        """Save a dictionary to a JSON file."""
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def open_npy(path: Union[str, Path]) -> np.ndarray:
        """Load a NumPy array from a .npy file."""
        return np.load(path, allow_pickle=True).item()

    @staticmethod
    def open_ecg(path: Union[str, Path]):
        signal, fields = wfdb.rdsamp(path)
        return signal, fields["fs"]

    @staticmethod
    def next_run_id(base: Union[str, Path]) -> str:
        base = Path(base)
        base.mkdir(parents=True, exist_ok=True)
        nums = [int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
        return str(max(nums) + 1 if nums else 0)

    @staticmethod
    def ensure_directory_exists(
        folder: Optional[Union[str, Path]] = None,
        file: Optional[Union[str, Path]] = None,
    ) -> bool:
        """If `folder` is provided, ensure it exists and return True.
        If `file` is provided, ensure its parent dir exists and return whether the file exists.
        Exactly one of `folder` or `file` must be provided.
        """
        if (folder is None) == (file is None):
            raise ValueError("Provide exactly one of 'folder' or 'file'.")

        if folder is not None:
            d = Path(folder)
            d.mkdir(parents=True, exist_ok=True)
            return True

        p = Path(file)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.exists()

    @staticmethod
    def align_signal_text_files(signal_dir: Union[str, Path], text_dir: Union[str, Path]) -> Tuple[List[str], List[str]]:
        """Align signal and text files based on their indices.
        Returns tuple of aligned signal and text file paths.
        """

        def get_index(filename: str) -> Optional[Tuple[int, int]]:
            if match := re.search(r"(\d+)_(\d+)", os.path.basename(filename)):
                return tuple(map(int, match.groups()))
            return None

        # Get files and their indices
        signal_files = {get_index(f): f for f in glob.glob(os.path.join(signal_dir, "*.npy")) if get_index(f)}
        text_files = {get_index(f): f for f in glob.glob(os.path.join(text_dir, "*.json")) if get_index(f)}

        # Find common indices and create aligned lists
        common = sorted(set(signal_files) & set(text_files))
        return ([signal_files[i] for i in common], [text_files[i] for i in common])

    @staticmethod
    def sample_N_percent(items: List[Any], N: float = 0.1) -> List[Any]:
        """Sample N percent of items from a list."""
        if not 0 <= N <= 1:
            raise ValueError("N must be between 0 and 1")
        size = max(1, int(len(items) * N))
        return random.sample(items, size)

    @classmethod
    def sample_N_percent_from_lists(
        cls, list1: List[Any], list2: Optional[List[Any]] = None, N: float = 0.05
    ) -> Union[List[Any], Tuple[List[Any], List[Any]]]:
        """Sample N percent of items from one or two lists."""
        if list2 and len(list1) != len(list2):
            raise ValueError("Lists must have same length")
        indices = cls.sample_N_percent(range(len(list1)), N)
        result1 = [list1[i] for i in indices]
        return (result1, [list2[i] for i in indices]) if list2 else result1

    @staticmethod
    def clean_dataframe(df: "pd.DataFrame") -> Tuple["pd.DataFrame", bool, int]:
        """Check for NaN values in DataFrame and remove rows containing NaN."""
        has_nan = df.isna().any().any()

        if has_nan:
            rows_before = len(df)

            cleaned_df = df.dropna()

            dropped_rows = rows_before - len(cleaned_df)

            print(f"Found and removed {dropped_rows} rows containing NaN values")
            print(f"Remaining rows: {len(cleaned_df)}")

            return cleaned_df
        print("No NaN values found in DataFrame")
        return df


def setup_experiment_folders(base_run_dir: Union[str, Path], args: argparse.Namespace) -> tuple[Path, Path]:
    """
    Rank 0 picks run_id and creates both dirs, broadcasts run_id, then barrier.
    Everyone returns the same (config_dir, run_dir) as Paths.
    """
    base_run_dir = Path(base_run_dir)
    fm = FileManager()
    fm.ensure_directory_exists(folder=base_run_dir)

    if is_main():
        run_id = fm.next_run_id(base_run_dir)
        run_dir = base_run_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        fm.save_config(run_dir, args)
    else:
        run_id, run_dir = None, None

    run_id = broadcast_value(run_id, src=0)
    if not is_main():
        run_dir = base_run_dir / run_id

    barrier()
    return run_dir
