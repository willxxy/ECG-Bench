import os
import json
import numpy as np
import pickle
import glob
import re
import random
import wfdb
from typing import Tuple, List, Dict, Optional, Union, Any
from pathlib import Path

class FileManager:
    """A class for managing file operations and directory handling."""
    
    @staticmethod
    def open_json(path: Union[str, Path]) -> dict:
        """Load and parse a JSON file."""
        with open(path) as f:
            return json.load(f)
    
    @staticmethod
    def open_npy(path: Union[str, Path]) -> np.ndarray:
        """Load a NumPy array from a .npy file."""
        return np.load(path, allow_pickle=True).item()
    
    @staticmethod
    def open_ecg(path: Union[str, Path]):
        signal, fields = wfdb.rdsamp(path)
        return signal, fields['fs']
    
    @staticmethod
    def load_vocab_and_merges(path: Union[str, Path]) -> Tuple[dict, dict]:
        """Load vocabulary and merges from a pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def ensure_directory_exists(path: Union[str, Path]) -> bool:
        """
        Check if path exists and create directory if needed.
        Returns True if path exists (file or dir), False otherwise.
        """
        path = Path(path)
        if path.exists():
            if path.is_file():
                print(f"File exists: {path}")
                return True
            print(f"Directory exists: {path}")
            return True
        
        if path.suffix:  # Has file extension
            print(f"File does not exist: {path}")
            return False
            
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
        return True
    
    @staticmethod
    def align_signal_text_files(signal_dir: Union[str, Path], text_dir: Union[str, Path]) -> Tuple[List[str], List[str]]:
        """
        Align signal and text files based on their indices.
        Returns tuple of aligned signal and text file paths.
        """
        def get_index(filename: str) -> Optional[Tuple[int, int]]:
            if match := re.search(r'(\d+)_(\d+)', os.path.basename(filename)):
                return tuple(map(int, match.groups()))
            return None
            
        # Get files and their indices
        signal_files = {get_index(f): f for f in glob.glob(os.path.join(signal_dir, '*.npy')) if get_index(f)}
        text_files = {get_index(f): f for f in glob.glob(os.path.join(text_dir, '*.json')) if get_index(f)}
        
        # Find common indices and create aligned lists
        common = sorted(set(signal_files) & set(text_files))
        return ([signal_files[i] for i in common],
                [text_files[i] for i in common])
    
    @staticmethod
    def sample_N_percent(items: List[Any], N: float = 0.1) -> List[Any]:
        """Sample N percent of items from a list."""
        if not 0 <= N <= 1:
            raise ValueError("N must be between 0 and 1")
        size = max(1, int(len(items) * N))
        return random.sample(items, size)
    
    @classmethod
    def sample_N_percent_from_lists(cls, list1: List[Any], list2: Optional[List[Any]] = None, N: float = 0.05) -> Union[List[Any], Tuple[List[Any], List[Any]]]:
        """Sample N percent of items from one or two lists."""
        if list2 and len(list1) != len(list2):
            raise ValueError("Lists must have same length")
        indices = cls.sample_N_percent(range(len(list1)), N)
        result1 = [list1[i] for i in indices]
        return (result1, [list2[i] for i in indices]) if list2 else result1
    
    @staticmethod
    def clean_dataframe(df: "pandas.DataFrame") -> Tuple["pandas.DataFrame", bool, int]:
        """
        Check for NaN values in DataFrame and remove rows containing NaN.
        """
        has_nan = df.isna().any().any()
        
        if has_nan:
            rows_before = len(df)
            
            cleaned_df = df.dropna()
            
            dropped_rows = rows_before - len(cleaned_df)
            
            print(f"Found and removed {dropped_rows} rows containing NaN values")
            print(f"Remaining rows: {len(cleaned_df)}")
            
            return cleaned_df
        else:
            print("No NaN values found in DataFrame")
            return df