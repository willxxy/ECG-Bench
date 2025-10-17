import argparse
import pickle
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import time
import random

import ecg_byte
from ecg_bench.utils.file_manager import FileManager


class BuildECGTokenizers:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.normalize_epsilon = 1e-6
        self.fm = FileManager()

    @staticmethod
    def open_tokenizer(path: Union[str, Path]) -> Tuple[dict, dict]:
        """Open a pickled tokenizer file and return the vocabulary and merges."""
        with open(path, "rb") as f:
            vocab, merges = pickle.load(f)
        return vocab, merges

    @staticmethod
    def save_tokenizer(vocab: dict, merges: dict, path: Union[str, Path]):
        """Save a tokenizer vocabulary and merges to a pickled file."""
        with open(path, "wb") as f:
            pickle.dump((vocab, merges), f)

    def normalize(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        min_vals = np.min(ecg_signal)
        max_vals = np.max(ecg_signal)
        normalized = (ecg_signal - min_vals) / (max_vals - min_vals + self.normalize_epsilon)
        clipped_normalized = np.clip(normalized, 0, 1)
        return clipped_normalized, (min_vals, max_vals)

    def denormalize(self, ecg_signal: np.ndarray, min_max_vals: Tuple[float, float]) -> np.ndarray:
        min_vals, max_vals = min_max_vals
        return ecg_signal * (max_vals - min_vals) + min_vals


class BuildECGByte(BuildECGTokenizers):
    def __init__(self, args: argparse.Namespace, mode: str):
        super().__init__(args)
        assert args.ecg_tokenizer or args.sampled_file, "ecg_tokenizer and/or sampled_file must be provided"
        self.symbols = list("abcdefghijklmnopqrstuvwxyz")
        self.len_symbols = len(self.symbols)

        if mode in ["train", "eval", "inference", "post_train"]:
            self.build_ecg_byte()
        elif mode == "ecg_tokenizer":
            if self.args.sampled_file and not self.args.ecg_tokenizer:
                self.train_ecg_byte()
            if self.args.ecg_tokenizer and self.args.sampled_file:
                self.build_ecg_byte()
                self.verify_ecg_byte()

    def train_ecg_byte(self):
        symbolic_sequence = self.mp_process_ecg_to_symbol()
        print(f"Total ECGs processed: {len(list(symbolic_sequence))}")
        print(list(symbolic_sequence)[:100])
        start_time = time.time()
        ids, vocab, merges = ecg_byte.byte_pair_encoding(symbolic_sequence, self.args.num_merges, self.args.num_cores)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Byte pair encoding executed in {execution_time:.2f} seconds")
        print("Shared vocabulary across all ECGs:")
        print(f"Original length: {len(symbolic_sequence)}")
        print(f"Encoded length: {len(ids)}")
        print(f"Compression ratio: {len(symbolic_sequence) / len(ids):.2f}X")
        print(f"Vocabulary size: {len(vocab)}")
        output_path = f"./ecg_bench/configs/ecg_tokenizers/ecg_byte_tokenizer_{self.args.num_merges}_{self.args.num_samples}.pkl"
        self.save_tokenizer(vocab, merges, output_path)

    def verify_ecg_byte(self):
        with open(self.args.sampled_file) as file:
            lines = file.readlines()
        ecg_path = random.choice(lines).strip()

        ecg_signal = self.fm.open_npy(ecg_path)["ecg"]
        ecg_signal_shape = ecg_signal.shape
        print(f"ECG signal shape: {ecg_signal_shape}")

        symbolic_sequence, min_max_vals = self.ecg_to_symbol(ecg_signal)
        print(f"Processed ECG signal to text (first 100 characters): {symbolic_sequence[:100]}...")
        print(f"Total tokens: {len(symbolic_sequence)}")

        ecg_tokens = self.encode(symbolic_sequence)
        print(f"Encoded ECG (first 20 tokens): {ecg_tokens[:20]}...")
        print(f"Total tokens: {len(ecg_tokens)}")

        decoded_symbolic_sequence = self.decode(ecg_tokens)
        print(f"Decoded text (first 100 characters): {decoded_symbolic_sequence[:100]}...")
        print(decoded_symbolic_sequence == symbolic_sequence)

        decoded_signal = self.symbol_to_ecg(
            np.array(list(decoded_symbolic_sequence)).reshape(ecg_signal.shape),
            min_max_vals,
        )
        max_diff = np.max(np.abs(ecg_signal - decoded_signal))
        print(f"Maximum difference between original and decoded: {max_diff}")

    def mp_process_ecg_to_symbol(self):
        with open(self.args.sampled_file) as f:
            file_paths = [ln.strip() for ln in f if ln.strip()]

        if getattr(self.args, "dev", False):
            self.args.num_samples = min(25, len(file_paths))
            file_paths = file_paths[: self.args.num_samples]

        seen = set()
        file_paths = [p for p in file_paths if not (p in seen or seen.add(p))]

        if not file_paths:
            raise ValueError("No file paths found to process.")

        ncores = max(1, int(self.args.num_cores))
        chunksize = max(1, len(file_paths) // (ncores * 4) or 1)

        with mp.Pool(processes=ncores) as pool:
            it = pool.imap_unordered(self.process_ecg_to_symbol, file_paths, chunksize=chunksize)
            ecg_strings = list(tqdm(it, total=len(file_paths), desc="Training ECG Byte Tokenizer"))

        ecg_strings = [s for s in ecg_strings if s]

        if not ecg_strings:
            raise ValueError("No valid ECG files were processed. Check your inputs and warnings above.")

        return "".join(ecg_strings)

    def process_ecg_to_symbol(self, ecg_path):
        try:
            data = self.fm.open_npy(ecg_path)
            if "ecg" not in data:
                print(f"Warning: 'ecg' key missing in {ecg_path}. Skipping.")
                return None
            ecg_array = data["ecg"]
            if not isinstance(ecg_array, np.ndarray):
                print(f"Warning: ECG in {ecg_path} is not a numpy array. Skipping.")
                return None
            if np.any(np.isnan(ecg_array)) or np.any(np.isinf(ecg_array)):
                print(f"Warning: NaN/Inf values in {ecg_path}. Skipping.")
                return None
            return self.ecg_to_symbol(ecg_array)[0]
        except Exception as e:
            print(f"Error processing {ecg_path}: {e!s}")
            return None

    def build_ecg_byte(self):
        self.vocab, self.merges = self.open_tokenizer(self.args.ecg_tokenizer)

    def encode(self, symbols: str):
        return ecg_byte.encode_symbol(symbols, self.merges)

    def decode(self, ecg_tokens):
        return "".join(self.vocab[token_id] for token_id in ecg_tokens)

    def ecg_to_symbol(self, ecg_signal: np.ndarray) -> str:
        normalized_ecg_signal, min_max_vals = self.normalize(ecg_signal)
        quantized_signal = self.quantize(normalized_ecg_signal)
        symbols = self.quantized_to_symbol(quantized_signal)
        return "".join(symbols.flatten()), min_max_vals

    def quantize(self, clipped_normalized: np.ndarray) -> np.ndarray:
        return np.minimum(
            np.floor(clipped_normalized * self.len_symbols),
            self.len_symbols - 1,
        ).astype(np.uint8)

    def quantized_to_symbol(self, quantized_signal: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda x: self.symbols[x])(quantized_signal)

    def symbol_to_ecg(self, symbols: str, min_max_vals: Tuple[float, float]) -> np.ndarray:
        quantized_signal = self.symbol_to_quantized(symbols)
        dequantized_signal = self.dequantize(quantized_signal)
        ecg_signal = self.denormalize(dequantized_signal, min_max_vals)
        return ecg_signal

    def symbol_to_quantized(self, symbols: str):
        return np.vectorize(lambda x: self.symbols.index(x))(symbols)

    def dequantize(self, quantized_signal: np.ndarray) -> np.ndarray:
        return quantized_signal / (len(self.symbols) - 1)
