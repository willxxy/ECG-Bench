import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from collections import namedtuple
CombinedOutput = namedtuple('CombinedOutput', ['loss', 'out'])

from ecg_bench.utils.model_utils import get_resnet
from ecg_bench.utils.model_utils import AttentionPool2d

class MERL(nn.Module):
    def __init__(self, merl):
        super(MERL, self).__init__()
        self.merl = merl
        if self.merl.args.train == 'second' or self.merl.args.inference == 'second':
            self.avgpool = nn.AdaptiveAvgPool1d((1))
    
    def forward(self, batch):
        if self.merl.args.train == 'first':
            out = self.merl(signal = batch['signal'].to(self.merl.device),
                            input_ids = batch['merl_input_ids'].to(self.merl.device),
                            attention_mask = batch['merl_att_mask'].to(self.merl.device))
        elif self.merl.args.train == 'second':
            out = self.merl(signal = batch['signal'].to(self.merl.device))
        return out
    
    @torch.no_grad()
    def get_embeddings(self, batch):
        self.merl.eval()
        out = self.merl(signal = batch['signal'].to(self.merl.device)).out
        out = self.avgpool(out)
        out = out.squeeze(2)
        return out
    
class MERLPretrain(nn.Module):
    def __init__(self, resnet_type, lm, args, device):
        super(MERLPretrain, self).__init__()
        self.args = args
        self.device = device
        self.resnet = get_resnet(resnet_type).to(self.device)
        self.lm = lm.to(self.device)
        
        if self.args.train == 'first':
            self.proj_out = 256
            self.proj_hidden = 256
            self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1)
            if self.args.seg_len == 1250:
                spacial_dim = 79
            else:
                spacial_dim = 32
            self.att_pool_head = AttentionPool2d(spacial_dim=spacial_dim,
                                                        embed_dim=self.proj_out, 
                                                        num_heads=4, 
                                                        output_dim=self.proj_out)
            self.avgpool = nn.AdaptiveAvgPool1d((1))
            self.dropout1 = nn.Dropout(p=0.1)
            self.dropout2 = nn.Dropout(p=0.1)
            self.linear1 = nn.Linear(self.proj_out, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_out, self.proj_out, bias=False)
            
            self.proj_t = nn.Sequential(
                nn.Linear(768, self.proj_hidden),
                nn.GELU(),
                nn.Linear(self.proj_hidden, self.proj_out),
            )
            
    def forward(self, signal, input_ids = None, attention_mask = None):
        out = self.resnet(signal)
        if self.args.train == 'first':
            ecg_emb = self.downconv(out)
            proj_ecg_emb, _ = self.att_pool_head(ecg_emb)
            proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)

            ecg_emb = self.avgpool(ecg_emb).view(ecg_emb.shape[0], -1)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))        
            proj_ecg_emb = normalize(proj_ecg_emb, dim=-1)
            
            text_emb = self.get_text_emb(input_ids, attention_mask)
            proj_text_emb = self.proj_t(text_emb.contiguous())
            proj_text_emb = normalize(proj_text_emb, dim=-1)
            
            combined_loss = self.calc_loss(ecg_emb1, ecg_emb2, proj_ecg_emb, proj_text_emb)
        elif self.args.train == 'second' or self.args.inference == 'second':
            combined_loss = 0
        
        return CombinedOutput(
            loss=combined_loss,
            out = out
        )
    
    @torch.no_grad()
    def get_text_emb(self, input_ids, attention_mask):
        text_emb = self.lm(input_ids=input_ids,
                           attention_mask=attention_mask).pooler_output
        return text_emb

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
        
            cma_loss = self.merl_loss(all_proj_ecg, all_proj_text)
            uma_loss = self.merl_loss(all_ecg1, all_ecg2)
        else:
            cma_loss = self.merl_loss(proj_ecg_emb, proj_text_emb)
            uma_loss = self.merl_loss(ecg_emb1, ecg_emb2)
        return cma_loss + uma_loss


    def merl_loss(self, x, y, temperature=0.07):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        sim = torch.einsum('i d, j d -> i j', x, y) * 1 / temperature

        labels = torch.arange(x.shape[0]).to(self.device)

        loss_t = F.cross_entropy(sim, labels) 
        loss_i = F.cross_entropy(sim.T, labels) 

        return (loss_t + loss_i)