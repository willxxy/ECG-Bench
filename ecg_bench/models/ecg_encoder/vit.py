import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, vit):
        super(ViT, self).__init__()
        self.vit = vit
            
    def forward(self, batch):
        out = self.vit( pixel_values = batch['vit_pixel'].to(self.device),
                        bool_masked_pos = batch['mask'].to(self.device))
        return out