import torch
from torch import nn


class HuggingFaceEncoder(nn.Module):
    def __init__(
        self,
        vision_encoder,
        mapping: dict,
        extra_configs: dict,
        embed_out=None,
    ):
        super(HuggingFaceEncoder, self).__init__()
        self.vision_encoder = vision_encoder
        self.mapping = mapping
        if embed_out is not None:
            self.embed_out = embed_out
        self.extra_configs = extra_configs

    def forward(self, batch):
        kwargs = {
            model_key: batch[value_key].to(self.vision_encoder.device) for model_key, value_key in self.mapping.items() if value_key in batch
        } | self.extra_configs
        out = self.vision_encoder(**kwargs)
        return out

    @torch.no_grad()
    def get_encoder_embeddings(self, batch):
        kwargs = {
            model_key: batch[value_key].to(self.vision_encoder.device) for model_key, value_key in self.mapping.items() if value_key in batch
        } | self.extra_configs
        out = self.vision_encoder(**kwargs)
        return self.embed_out(out)
