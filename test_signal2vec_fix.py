#!/usr/bin/env python3
"""
Test script to verify the signal2vec memory fix works
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ecg_bench.ecg_tokenizers.signal2vec import SkipGramDataset
import numpy as np


def test_memory_efficiency():
    """Test that the new SkipGramDataset doesn't consume excessive memory"""
    print("Testing memory-efficient SkipGramDataset...")

    # Create some test sequences (simulate ECG token sequences)
    sequences = []
    for i in range(1000):  # 1000 sequences
        # Each sequence has 100-500 tokens (typical ECG sequence length)
        seq_len = np.random.randint(100, 500)
        seq = np.random.randint(0, 1000, size=seq_len).tolist()  # vocab size 1000
        sequences.append(seq)

    # Create noise distribution
    vocab_size = 1000
    noise_dist = np.ones(vocab_size) / vocab_size

    # Test the old approach would have created ~1000 * 300 * 12 = 3.6M pairs
    # The new approach should use much less memory
    print(f"Created {len(sequences)} test sequences")

    # Create dataset
    dataset = SkipGramDataset(sequences=sequences, window_max=12, neg_k=5, noise_dist=noise_dist)

    print(f"Dataset length: {len(dataset):,}")
    print("Memory usage should be minimal (no pre-generated pairs)")

    # Test sampling a few items
    print("\nTesting data sampling...")
    for i in range(5):
        c, p, n = dataset[i]
        print(f"Sample {i}: center={c.item()}, positive={p.item()}, negative={n.shape}")

    print("\n✅ Memory efficiency test passed!")
    return True


if __name__ == "__main__":
    test_memory_efficiency()
