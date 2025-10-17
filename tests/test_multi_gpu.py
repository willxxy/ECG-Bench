import os
import pytest
import torch
import torch.distributed as dist
from datetime import timedelta

pytestmark = [pytest.mark.dist, pytest.mark.gpu]


def _init_ddp():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; DDP NCCL test requires GPU.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        pytest.skip("Run with torchrun (provides RANK/WORLD_SIZE/LOCAL_RANK).")

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.cuda.device_count():
        pytest.skip(f"LOCAL_RANK {local_rank} >= cuda device count {torch.cuda.device_count()}")

    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=60))

    rank = dist.get_rank()
    world = dist.get_world_size()
    print(f"[ddp] Initialized process group rank={rank} world_size={world} local_rank={local_rank}")
    return rank, world, local_rank


def _cleanup_ddp():
    if dist.is_initialized():
        try:
            print("[ddp] barrier before destroy")
            dist.barrier()
        finally:
            print("[ddp] destroy process group")
            dist.destroy_process_group()


def test_ddp_allreduce():
    rank = world = local_rank = None
    try:
        rank, world, local_rank = _init_ddp()
        dev = torch.device(f"cuda:{local_rank}")

        print(f"[ddp] rank={rank}: creating tensor")
        x = torch.full((4,), float(rank + 1), device=dev)

        print(f"[ddp] rank={rank}: all_reduce start")
        dist.all_reduce(x, op=dist.ReduceOp.SUM)

        expected = float(world * (world + 1) // 2)  # sum 1..world
        print(f"[ddp] rank={rank}: validate expected={expected}")
        assert torch.allclose(x, torch.full_like(x, expected)), f"rank {rank}: {x.tolist()} != {expected}"
        torch.cuda.synchronize(dev)
    finally:
        _cleanup_ddp()
