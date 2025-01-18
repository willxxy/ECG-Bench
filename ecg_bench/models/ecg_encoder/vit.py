import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, vit):
        super(ViT, self).__init__()
        self.vit = vit
            
    def forward(self, batch):
        out = self.vit( pixel_values = batch['vit_pixel'].to(self.vit.device),
                        bool_masked_pos = batch['vit_mask'].to(self.vit.device),)
        return out