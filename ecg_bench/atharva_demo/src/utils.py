import torch.nn as nn



def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_attention_like_mask(pad_id, numbers):
    return [0 if num == pad_id else 1 for num in numbers]