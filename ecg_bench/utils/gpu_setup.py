# ecg_bench/utils/gpu_setup.py
import torch, argparse, os, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(get_local_rank())


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main() -> bool:
    return get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_value(val, src: int = 0):
    """Broadcast a small Python object (e.g., str/int) without GPU assumptions."""
    if not (dist.is_available() and dist.is_initialized()):
        return val
    obj = [val]
    dist.broadcast_object_list(obj, src=src)
    return obj[0]


def train_dev_break(enabled: bool, batch: dict, loss_value: float) -> bool:
    if not enabled:
        return False
    should_break = False
    if is_main():
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {type(value)}")
        print("loss", loss_value)
        should_break = True
    return broadcast_value(should_break, src=0)


class GPUSetup:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def setup_gpu(self, model: torch.nn.Module, find_unused_parameters) -> torch.nn.Module:
        device = self.get_device()
        model = model.to(device)
        if getattr(self.args, "distributed", False):
            model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=find_unused_parameters)
        if is_main():
            print(f"find_unused_parameters: {find_unused_parameters}")
        return model

    def get_device(self) -> torch.device:
        return self.get_multi_device() if getattr(self.args, "distributed", False) else self.get_single_device()

    def get_single_device(self) -> torch.device:
        dev = getattr(self.args, "device", None)
        return torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))

    def get_multi_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{get_local_rank()}")
        return torch.device("cpu")

    def print_model_device(self, model: torch.nn.Module, name: str) -> None:
        if is_main():
            print(f"{name} device:", next(model.parameters()).device)
