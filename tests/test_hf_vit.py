import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import ViTForMaskedImageModeling, AutoModel, AutoImageProcessor
from ecg_bench.configs.constants import HF_CACHE_DIR


def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def test_model(model_class):
    model = model_class.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=HF_CACHE_DIR)
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=HF_CACHE_DIR)

    local_rank = setup_ddp()
    model = DDP(model.to(f"cuda:{local_rank}"), device_ids=[local_rank])

    images = torch.rand(2, 3, 224, 224)
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(f"cuda:{local_rank}")

    bool_masked_pos = torch.zeros(2, 196, dtype=torch.bool, device=f"cuda:{local_rank}")
    for i in range(2):
        bool_masked_pos[i, torch.randperm(196)[:29]] = True
    outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
    loss = outputs.loss

    loss.backward()
    print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    test_model(ViTForMaskedImageModeling)
    test_model(AutoModel)
