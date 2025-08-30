import torch
from torch import nn


class SIGLIP(nn.Module):
    def __init__(self, siglip):
        super(SIGLIP, self).__init__()
        self.siglip = siglip

    def forward(self, batch):
        out = self.siglip(input_ids = batch["siglip_input_ids"].to(self.siglip.device),
                        attention_mask = batch["siglip_att_mask"].to(self.siglip.device),
                        pixel_values = batch["siglip_pixel"].to(self.siglip.device),
                        return_loss = True)
        return out

    @torch.no_grad()
    def get_embeddings(self, batch):
        self.siglip.eval()
        out = self.siglip(input_ids = batch["siglip_input_ids"].to(self.siglip.device),
                        attention_mask = batch["siglip_att_mask"].to(self.siglip.device),
                        pixel_values = batch["siglip_pixel"].to(self.siglip.device),
                        return_loss = False)
        return out.image_embeds
