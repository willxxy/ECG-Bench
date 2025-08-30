import multiprocessing as mp
import os
import time
from collections import Counter

import bpe
import numpy as np
from tqdm import tqdm

from ecg_bench.utils.viz_utils import VizUtil


class ECGByteTokenizer:
    """Main class for training and verifying the tokenizer. In this class, we do the following:
        1. Train the tokenizer using the Byte Pair Encoding algorithm.
        2. Verify the tokenizer by encoding and decoding a sample ECG signal.
    
    Currently, this is a naive BPE algorithm. In the future, we will add more sophisticated merging methods.
    """

    def __init__(self, args, fm):
        self.args = args
        self.fm = fm
        self.n = 3000 if self.args.dev else None
        self.lead_order = ["I", "II", "III", "aVL", "aVR", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        if self.args.ecg_tokenizer != None:
            self.vocab, self.merges = self.fm.open_tokenizer(self.args.ecg_tokenizer)
        self.symbols = list("abcdefghijklmnopqrstuvwxyz")
        self.len_symbols = len(self.symbols)
        self.ep1 = 1e-6
        self.ep2 = 0.5
        self.viz = VizUtil()

    def train_tokenizer(self):
        all_string_signals = self.discretize_ecgs(self.args.sampled_files, self.n)

        print(f"Total ECGs processed: {len(list(all_string_signals))}")
        print(list(all_string_signals)[:100])

        start_time = time.time()
        ids, vocab, merges = bpe.byte_pair_encoding(all_string_signals, self.args.num_merges, self.args.num_cores)
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Byte pair encoding executed in {execution_time:.2f} seconds")
        print("Shared vocabulary across all ECGs:")
        print(f"Original length: {len(all_string_signals)}")
        print(f"Encoded length: {len(ids)}")
        print(f"Compression ratio: {len(all_string_signals) / len(ids):.2f}X")
        print(f"Vocabulary size: {len(vocab)}")
        num_sample_files = self.args.sampled_files.split("/")[-1].split("_")[1]
        self.fm.save_tokenizer(vocab, merges, f"./data/tokenizer_{self.args.num_merges}_{num_sample_files}_{self.args.instance_normalize}_{self.args.dev}.pkl")
        print("Vocabulary and merges saved")

    def verify_tokenizer(self):
        path_to_ecg = f"./data/ptb/preprocessed_1250_250/{os.listdir('./data/ptb/preprocessed_1250_250')[0]}"
        original_ecg = self.fm.open_npy(path_to_ecg)["ecg"]
        symbol_signal = self.process_ecg(path_to_ecg)
        print(f"Processed ECG signal to text (first 100 characters): {symbol_signal[:100]}...")
        print(f"Total tokens: {len(symbol_signal)}")

        encoded_ecg = self.encode_symbol(symbol_signal, self.merges)
        print(f"Encoded ECG (first 20 tokens): {encoded_ecg[:20]}...")
        print(f"Total tokens: {len(encoded_ecg)}")
        print(f"Compression ratio: {len(symbol_signal) / len(encoded_ecg):.2f}X")

        decoded_text = self.decode_token(encoded_ecg, self.vocab)
        print(f"Decoded text (first 100 characters): {decoded_text[:100]}...")
        print(decoded_text == symbol_signal)

        # Store signal range during normalization for instance-based denormalization
        if self.args.instance_normalize:
            _, _, signal_range = self.instance_normalize(original_ecg)
            decoded_signal = self.instance_denormalize(np.array(list(decoded_text)).reshape(original_ecg.shape), signal_range)

        max_diff = np.max(np.abs(original_ecg - decoded_signal))
        print(f"Maximum difference between original and decoded: {max_diff}")


        self.viz.plot_2d_ecg(decoded_signal, "decoded_ecg_2d", save_path = "./pngs/", sample_rate = 250)
        self.viz.plot_1d_ecg(decoded_signal, "decoded_ecg_1d", save_path = "./pngs/", sample_rate = 250)

        self.viz.plot_2d_ecg(original_ecg, "original_ecg_2d", save_path = "./pngs/", sample_rate = 250)
        self.viz.plot_1d_ecg(original_ecg, "original_ecg_1d", save_path = "./pngs/", sample_rate = 250)

    def encode_symbol(self, text, merges):
        return bpe.encode_symbol(text, merges)

    def decode_token(self, encoded_ids, vocab):
        return "".join(vocab[token_id] for token_id in encoded_ids)

    def instance_normalize(self, signal):
        min_vals = np.min(signal)
        max_vals = np.max(signal)
        normalized = (signal - min_vals) / (max_vals - min_vals + self.ep1)
        clipped_normalized = np.clip(normalized, 0, 1)
        scaled_signal = np.minimum(
            np.floor(clipped_normalized * self.len_symbols),
            self.len_symbols - 1,
        ).astype(np.uint8)
        symbol_signal = np.vectorize(lambda x: self.symbols[x])(scaled_signal)
        return clipped_normalized, symbol_signal, (min_vals, max_vals)

    def instance_denormalize(self, symbol_signal, signal_range):
        min_vals, max_vals = signal_range
        scaled_signal = np.vectorize(lambda x: self.symbols.index(x))(symbol_signal)
        clipped_normalized = scaled_signal / (len(self.symbols) - 1)
        return clipped_normalized * (max_vals - min_vals) + min_vals

    def _to_symbol_string(self, ecg_array):
        if self.args.instance_normalize:
            _, symbol_signal, signal_range = self.instance_normalize(ecg_array)
            return "".join(symbol_signal.flatten())

    def process_ecg(self, ecg_path):
        try:
            data = self.fm.open_npy(ecg_path)
            if "ecg" not in data:
                print(f"Warning: ECG data not found in {ecg_path}. File may be corrupted or in unexpected format.")
                return ""
            ecg_array = data["ecg"]
            if np.any(np.isnan(ecg_array)) or np.any(np.isinf(ecg_array)):
                print(f"Warning: NaN or Inf values detected in {ecg_path}. Skipping this ECG.")
                return ""
            return self._to_symbol_string(ecg_array)
        except Exception as e:
            print(f"Error processing ECG file {ecg_path}: {e!s}")
            return ""

    def discretize_ecgs(self, file_path, n=None):
        def file_path_generator():
            with open(file_path) as file:
                for i, line in enumerate(file):
                    if n is not None and i >= n:
                        break
                    yield line.strip()

        file_paths = list(file_path_generator())

        with mp.Pool(processes=self.args.num_cores) as pool:
            ecg_strings = list(
                tqdm(
                    pool.imap(self.process_ecg, file_paths),
                    total=len(file_paths),
                    desc="Discretizing ECGs"))

        # Filter out empty strings that could result from errors
        ecg_strings = [s for s in ecg_strings if s]
        if not ecg_strings:
            raise ValueError("No valid ECG files were processed. Check your input files and error messages.")

        return "".join(ecg_strings)

    def analyze_single_ecg(self, path):
        try:
            data = self.fm.open_npy(path)
            if "ecg" not in data:
                print(f"Warning: ECG data not found in {path}. File may be corrupted or in unexpected format.")
                return Counter(), 0

            signal = data["ecg"]
            if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                print(f"Warning: NaN or Inf values detected in {path}. Skipping this ECG.")
                return Counter(), 0

            single_lead_str = self._to_symbol_string(signal)
            all_encoded_ids = list(self.encode_symbol(single_lead_str, self.merges))
            return Counter(all_encoded_ids), len(all_encoded_ids)
        except Exception as e:
            print(f"Error analyzing ECG file {path}: {e!s}")
            return Counter(), 0

    def analyze_token_distribution(self, test_data):
        with mp.Pool(self.args.num_cores) as pool:
            results = list(
                tqdm(
                    pool.imap(self.analyze_single_ecg, (path for path in test_data)),
                    total=len(test_data),
                    desc=f"Analyzing token distribution with {self.args.num_cores} processes"))

        token_counts = Counter()
        token_lengths = []
        valid_results_count = 0

        for count, length in results:
            if length > 0:
                token_counts.update(count)
                token_lengths.append(length)
                valid_results_count += 1

        if valid_results_count == 0:
            print("Warning: No valid ECG files were analyzed. Check your input files and error messages.")
            return Counter(), []

        print(f"Successfully analyzed {valid_results_count} out of {len(test_data)} ECG files.")
        return token_counts, token_lengths

    def track_encoding(self, single_lead_str):
        ids = list(single_lead_str.encode("utf-8"))
        segment_map = [(i, i+1) for i in range(len(ids))]
        for batch in tqdm(self.merges, desc = "Tracking Encoding"):
            pair, new_id = batch
            new_ids = []
            new_segment_map = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                    new_ids.append(new_id)
                    new_segment_map.append((segment_map[i][0], segment_map[i+1][1]))
                    i += 2
                else:
                    new_ids.append(ids[i])
                    new_segment_map.append(segment_map[i])
                    i += 1
            ids = new_ids
            segment_map = new_segment_map
        return ids, segment_map
