import pytest
import torch


pytestmark = [pytest.mark.gpu]


def test_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU-only tests.")
    print("[single_gpu] CUDA is available")
    assert torch.cuda.device_count() >= 1, "No CUDA device detected."


def test_basic_tensor_ops():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU-only tests.")
    dev = torch.device("cuda:0")
    print("[single_gpu] Allocating tensors on cuda:0")
    a = torch.randn(256, 256, device=dev, dtype=torch.float16)
    b = torch.randn(256, 256, device=dev, dtype=torch.float16)
    print("[single_gpu] Performing matmul")
    c = a @ b
    print("[single_gpu] Matmul done; validating outputs")
    assert c.is_cuda and c.shape == (256, 256)
    # sanity check numerical range (won’t be exact; just not NaN/inf)
    assert torch.isfinite(c).all()
