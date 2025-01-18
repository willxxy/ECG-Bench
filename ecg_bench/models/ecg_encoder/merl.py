import torch
import torch.nn as nn
from torch.nn.functional import normalize
from collections import namedtuple

### We thank the authors of https://github.com/cheliu-computation/MERL-ICML2024/tree/main for the main code.
### We modify the architecture to fit our setting in the MERL class

CombinedOutput = namedtuple('CombinedOutput', ['loss', 'out'])

class MERL(nn.Module):
    def __init__(self, resnet, text_embeder, device = None, args = None):
        super(MERL, self).__init__()
        self.resnet = resnet
        self.text_embeder = text_embeder
        self.args = args
        if device == None:
            self.device = self.resnet.device
        else:
            self.device = device
        
        
    def forward(self, batch):
        out = self.resnet(batch)
        return out
        
    
    def calc_loss(self, ecg_emb1, ecg_emb2, proj_ecg_emb, proj_text_emb):
        if self.args.dis:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            
            with torch.no_grad():
                gathered_proj_ecg = [torch.zeros_like(proj_ecg_emb) for _ in range(world_size)]
                gathered_proj_text = [torch.zeros_like(proj_text_emb) for _ in range(world_size)]
                gathered_ecg1 = [torch.zeros_like(ecg_emb1) for _ in range(world_size)]
                gathered_ecg2 = [torch.zeros_like(ecg_emb2) for _ in range(world_size)]
                
                torch.distributed.all_gather(gathered_proj_ecg, proj_ecg_emb)
                torch.distributed.all_gather(gathered_proj_text, proj_text_emb)
                torch.distributed.all_gather(gathered_ecg1, ecg_emb1)
                torch.distributed.all_gather(gathered_ecg2, ecg_emb2)
            
            gathered_proj_ecg[rank] = proj_ecg_emb
            gathered_proj_text[rank] = proj_text_emb
            gathered_ecg1[rank] = ecg_emb1
            gathered_ecg2[rank] = ecg_emb2
            
            all_proj_ecg = torch.cat(gathered_proj_ecg, dim=0)
            all_proj_text = torch.cat(gathered_proj_text, dim=0)
            all_ecg1 = torch.cat(gathered_ecg1, dim=0)
            all_ecg2 = torch.cat(gathered_ecg2, dim=0)
        
            cma_loss, acc1, acc5 = clip_loss(all_proj_ecg, all_proj_text, device=self.device)
            uma_loss, _, _ = clip_loss(all_ecg1, all_ecg2, device=self.device)
        else:
            cma_loss, acc1, acc5 = clip_loss(proj_ecg_emb, proj_text_emb, device=self.device)
            uma_loss, _, _ = clip_loss(ecg_emb1, ecg_emb2, device=self.device)
        return cma_loss + uma_loss
