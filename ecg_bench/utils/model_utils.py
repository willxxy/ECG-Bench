import torch
import torch.nn as nn
from torch.nn.functional import normalize
from collections import namedtuple
import torch.nn.functional as F

CombinedOutput = namedtuple('CombinedOutput', ['loss', 'out'])

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, spacial_dim + 1, embed_dim) / embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)        
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(0, 2, 1) # convert X shape (B, C, L) to (B, L, C)

        self.cls_tokens = self.cls_token + self.positional_embedding[:, :1, :]
        self.cls_tokens = self.cls_tokens.expand(x.shape[0], -1, -1) 
        x = torch.cat((self.cls_tokens, x), dim=1)
        x = x + self.positional_embedding[:, :, :].to(x.dtype)  # (L+1)NC
        x, att_map = self.mhsa(x[:, :1, :], x, x, average_attn_weights=True)
        x = self.c_proj(x)
        return x.squeeze(0), att_map[:, :, 1:]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 1x1 Convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # 3x3 Convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
    
def get_resnet(args):
    if args.model == 'resnet':
        return ResNet(BasicBlock, [2, 2, 2, 2])
    elif args.model == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3])
    elif args.model == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3])
    elif args.model == 'resnet101':
        return ResNet(Bottleneck, [3, 4, 23, 3])
    elif args.model == 'resnet152':
        return ResNet(Bottleneck, [3, 8, 36, 3])
    
    
class MERLPretrain(nn.Module):
    def __init__(self, training_utils, lm):
        super(MERLPretrain, self).__init__()
        self.training_utils = training_utils
        self.resnet = get_resnet(self.training_utils.args)
        self.lm = lm
        
        self.proj_out = 256
        self.proj_hidden = 256
        self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1)
        self.att_pool_head = AttentionPool2d(spacial_dim=32,
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
        
    def forward(self, batch):
        out = self.resnet(batch['signal'].to(self.resnet.device))
        ecg_emb = self.downconv(out)
        proj_ecg_emb, _ = self.att_pool_head(ecg_emb)
        proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)

        ecg_emb = self.avgpool(ecg_emb).view(ecg_emb.shape[0], -1)
        ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
        ecg_emb2 = self.dropout2(self.linear2(ecg_emb))        
        proj_ecg_emb = normalize(proj_ecg_emb, dim=-1)
        
        text_emb = self.get_text_emb(batch)
        proj_text_emb = self.proj_t(text_emb.contiguous())
        proj_text_emb = normalize(proj_text_emb, dim=-1)
        
        combined_loss = self.calc_loss(ecg_emb1, ecg_emb2, proj_ecg_emb, proj_text_emb)
        
        return CombinedOutput(
            loss=combined_loss,
            out = out
        )
    
    @torch.no_grad()
    def get_text_emb(self, batch):
        text_emb = self.lm(input_ids=batch['input_ids'].to(self.resnet.device),
                           attention_mask=batch['att_mask'].to(self.resnet.device)).pooler_output
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
        
            cma_loss, _, _ = self.merl_loss(all_proj_ecg, all_proj_text)
            uma_loss, _, _ = self.merl_loss(all_ecg1, all_ecg2)
        else:
            cma_loss, _, _ = self.merl_loss(proj_ecg_emb, proj_text_emb)
            uma_loss, _, _ = self.merl_loss(ecg_emb1, ecg_emb2)
        return cma_loss + uma_loss
    
    def precision_at_k(self, output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def merl_loss(self, x, y, temperature=0.07):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        sim = torch.einsum('i d, j d -> i j', x, y) * 1 / temperature

        labels = torch.arange(x.shape[0]).to(self.resnet.device)

        loss_t = F.cross_entropy(sim, labels) 
        loss_i = F.cross_entropy(sim.T, labels) 

        i2t_acc1, i2t_acc5 = self.precision_at_k(
            sim, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            sim.T, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        return (loss_t + loss_i), acc1, acc5
    
class MERLFinetune(nn.Module):
    def __init__(self, training_utils):
        super(MERLFinetune, self).__init__()
        self.training_utils = training_utils
        self.resnet = get_resnet(self.training_utils.args)
        self.combined_loss = 0
        
    def forward(self, batch):
        out = self.resnet(batch['signal'].to(self.resnet.device))
        
        return CombinedOutput(
            loss=self.combined_loss,
            out = out
        )