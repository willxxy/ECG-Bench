import pytest
import torch

pytestmark = [pytest.mark.flashattn, pytest.mark.gpu]


def has_flashattn():
    try:
        from flash_attn import flash_attn_func  # noqa: F401

        return True
    except Exception:
        return False


def test_flash_attn_minimal():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; FlashAttention is GPU-only.")
    if not has_flashattn():
        pytest.skip("flash-attn not installed.")

    from flash_attn import flash_attn_func

    dev = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("[flash_attn] Creating q,k,v tensors on cuda:0")
    b, s, h, d = 2, 32, 8, 64  # tiny and quick
    q = torch.randn(b, s, h, d, device=dev, dtype=dtype)
    k = torch.randn(b, s, h, d, device=dev, dtype=dtype)
    v = torch.randn(b, s, h, d, device=dev, dtype=dtype)

    print("[flash_attn] Running flash_attn_func")
    out = flash_attn_func(q, k, v, causal=False)
    print("[flash_attn] Done; validating outputs")
    assert out.shape == (b, s, h, d)
    assert out.is_cuda and out.dtype == dtype
    assert torch.isfinite(out).all()
