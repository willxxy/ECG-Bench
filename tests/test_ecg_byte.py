import argparse
import pytest
from ecg_bench.ecg_tokenizers.build_ecg_tokenizers import BuildECGByte

pytestmark = [pytest.mark.ecg_byte]


def test_ecg_byte():
    arg_sets = [
        dict(
            ecg_tokenizer="./ecg_bench/configs/ecg_tokenizers/ecg_byte_tokenizer_6000_800000.pkl",
            dev=True,
            sampled_file=None,
        ),
        dict(
            sampled_file="./ecg_bench/configs/ecg_tokenizers/sampled_800000_random.txt",
            dev=True,
            ecg_tokenizer=None,
            num_cores=6,
            num_merges=1000,
        ),
        dict(
            ecg_tokenizer="./ecg_bench/configs/ecg_tokenizers/ecg_byte_tokenizer_6000_800000.pkl",
            sampled_file="./ecg_bench/configs/ecg_tokenizers/sampled_800000_random.txt",
            dev=True,
        ),
    ]

    for args in arg_sets:
        BuildECGByte(argparse.Namespace(**args), "ecg_tokenizer")
