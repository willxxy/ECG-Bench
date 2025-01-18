import torch.nn as nn

class CLIP(nn.Module):
    def __init__(self, clip):
        super(CLIP, self).__init__()
        self.clip = clip
        
    def forward(self, batch):
        out = self.clip(input_ids = batch['clip_input_ids'].to(self.clip.device),
                        attention_mask = batch['clip_att_mask'].to(self.clip.device),
                        pixel_values = batch['clip_pixel'].to(self.clip.device),
                        return_loss = True)
        return out