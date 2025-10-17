#!/usr/bin/env bash
set -euo pipefail

echo "==> Running ECG Byte tests"
pytest -q tests/test_ecg_byte.py -k "test_ecg_byte" -s

echo "==> Running Hugging Face transformers tests"
pytest -q tests/test_transformers.py -k "test_random_model_loading or test_random_dataset_loading" -s

echo "==> Running single-GPU + FlashAttention tests"
pytest -q tests/test_single_gpu.py -k "test_cuda_available or test_basic_tensor_ops" -s

echo
echo "==> To run DDP test (example with 2 GPUs):"
echo "torchrun --standalone --nproc_per_node=2 -m pytest -q -s tests/test_multi_gpu.py::test_ddp_allreduce"
