import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, vit):
        super(ViT, self).__init__()
        self.vit = vit
            
    def forward(self, batch):
        out = self.vit(pixel_values = batch['vit_pixel'].to(self.vit.device),
                        bool_masked_pos = batch['vit_mask'].to(self.vit.device))
        return out
    
    @torch.no_grad()
    def get_embeddings(self, batch):
        self.vit.eval()
        out = self.vit(pixel_values = batch['vit_pixel'].to(self.vit.device),
                        bool_masked_pos = batch['vit_mask'].to(self.vit.device),
                        output_hidden_states=True)
        all_hidden_states = torch.stack(out.hidden_states)
        averaged_layers = torch.mean(all_hidden_states, dim=0)
        averaged_heads = torch.mean(averaged_layers, dim=1)
        return averaged_heads
        
        