# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>
# Modified work Copyright 2025 ECG-Bench authors. <https://github.com/willxxy/ECG-Bench>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this repository https://github.com/bakqui/ST-MEM.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

### ENCODER / MLAE_VIT ###
from typing import Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from collections import namedtuple
CombinedOutput = namedtuple('CombinedOutput', ['loss', 'out'])
from ecg_bench.models.encoder.st_mem import ViT, TransformerBlock
from ecg_bench.models.encoder.mtae import MTAE


__all__ = ['MLAE_ViT', 'mlae_vit_small', 'mlae_vit_base']


class MLAE_ViT(ViT):
    def __init__(self,
                 seq_len: int,
                 patch_size: int,
                 num_leads: int,
                 num_classes: Optional[int] = None,
                 width: int = 768,
                 depth: int = 12,
                 mlp_dim: int = 3072,
                 heads: int = 12,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super(ViT, self).__init__()
        assert num_leads % patch_size == 0, 'The number of leads must be divisible by the patch size.'
        self._repr_dict = {'seq_len': seq_len,
                           'patch_size': patch_size,
                           'num_leads': num_leads,
                           'num_classes': num_classes if num_classes is not None else 'None',
                           'width': width,
                           'depth': depth,
                           'mlp_dim': mlp_dim,
                           'heads': heads,
                           'dim_head': dim_head,
                           'qkv_bias': qkv_bias,
                           'drop_out_rate': drop_out_rate,
                           'attn_drop_out_rate': attn_drop_out_rate,
                           'drop_path_rate': drop_path_rate}
        self.width = width
        self.depth = depth

        # embedding layers
        num_patches = num_leads // patch_size
        patch_dim = seq_len * patch_size
        self.to_patch_embedding = nn.Sequential(Rearrange('b (n p) t -> b n (p t)', p=patch_size),
                                                nn.LayerNorm(patch_dim),
                                                nn.Linear(patch_dim, width),
                                                nn.LayerNorm(width))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, width))
        self.cls_embedding = nn.Parameter(torch.randn(width))

        # transformer layers
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(input_dim=width,
                                     output_dim=width,
                                     hidden_dim=mlp_dim,
                                     heads=heads,
                                     dim_head=dim_head,
                                     qkv_bias=qkv_bias,
                                     drop_out_rate=drop_out_rate,
                                     attn_drop_out_rate=attn_drop_out_rate,
                                     drop_path_rate=drop_path_rate_list[i])
            self.add_module(f'block{i}', block)
        self.dropout = nn.Dropout(drop_out_rate)
        self.norm = nn.LayerNorm(width)

        # classifier head
        self.head = nn.Identity() if num_classes is None else nn.Linear(width, num_classes)


def mlae_vit_small(num_leads, num_classes=None, seq_len=2250, patch_size=1, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      width=384,
                      depth=12,
                      heads=6,
                      mlp_dim=1536,
                      **kwargs)
    return MLAE_ViT(**model_args)


def mlae_vit_base(num_leads, num_classes=None, seq_len=2250, patch_size=1, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      width=768,
                      depth=12,
                      heads=12,
                      mlp_dim=3072,
                      **kwargs)
    return MLAE_ViT(**model_args)
###

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

__all__ = ['MLAE', 'mlae_vit_small_dec256d4b', 'mlae_vit_base_dec256d4b']


class MLAE(MTAE):
    def __init__(self,
                 seq_len: int = 2250,
                 patch_size: int = 1,
                 num_leads: int = 12,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 decoder_embed_dim: int = 256,
                 decoder_depth: int = 4,
                 decoder_num_heads: int = 4,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 norm_layer: nn.Module = nn.LayerNorm,
                 norm_pix_loss: bool = False,
                 device: str = 'cuda'):
        super(MTAE, self).__init__()
        self.device = device
        self._repr_dict = {'seq_len': seq_len,
                           'patch_size': patch_size,
                           'num_leads': num_leads,
                           'embed_dim': embed_dim,
                           'depth': depth,
                           'num_heads': num_heads,
                           'decoder_embed_dim': decoder_embed_dim,
                           'decoder_depth': decoder_depth,
                           'decoder_num_heads': decoder_num_heads,
                           'mlp_ratio': mlp_ratio,
                           'qkv_bias': qkv_bias,
                           'norm_layer': str(norm_layer),
                           'norm_pix_loss': norm_pix_loss}
        self.patch_size = patch_size
        self.num_patches = num_leads // patch_size
        # --------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = MLAE_ViT(seq_len=seq_len,
                                patch_size=patch_size,
                                num_leads=num_leads,
                                width=embed_dim,
                                depth=depth,
                                mlp_dim=mlp_ratio * embed_dim,
                                heads=num_heads,
                                qkv_bias=qkv_bias)
        self.to_patch_embedding = self.encoder.to_patch_embedding
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # MAE decoder specifics
        self.to_decoder_embedding = nn.Linear(embed_dim, decoder_embed_dim)

        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([TransformerBlock(input_dim=decoder_embed_dim,
                                                              output_dim=decoder_embed_dim,
                                                              hidden_dim=decoder_embed_dim * mlp_ratio,
                                                              heads=decoder_num_heads,
                                                              dim_head=64,
                                                              qkv_bias=qkv_bias)
                                             for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, seq_len)
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def patchify(self, series):
        """
        series: (batch_size, num_leads, seq_len)
        x: (batch_size, n, patch_size * seq_len)
        """
        p = self.patch_size
        assert series.shape[2] % p == 0
        x = rearrange(series, 'b (n p) t -> b n (p t)', p=p)
        return x

    def unpatchify(self, x):
        """
        x: (batch_size, n, patch_size * seq_len)
        series: (batch_size, num_leads, seq_len)
        """
        series = rearrange(x, 'b n (p t) -> b (n p) t')
        return series


def mlae_vit_small_dec256d4b(device,**kwargs):
    model = MLAE(embed_dim=384,
                 depth=12,
                 num_heads=6,
                 decoder_embed_dim=256,
                 decoder_depth=4,
                 decoder_num_heads=4,
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 device=device,
                 **kwargs)
    return model


def mlae_vit_base_dec256d4b(device,**kwargs):
    model = MLAE(embed_dim=768,
                 depth=12,
                 num_heads=12,
                 decoder_embed_dim=256,
                 decoder_depth=4,
                 decoder_num_heads=4,
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 device=device,
                 **kwargs)
    return model

class MLAE_Ours(nn.Module):
    def __init__(self, mlae):
        super(MLAE_Ours, self).__init__()
        self.mlae = mlae
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, batch):
        return self.mlae(batch['signal'].to(self.mlae.device))
    
    @torch.no_grad()
    def get_embeddings(self, batch):
        self.mlae.eval()
        out = self.mlae(batch['signal'].to(self.mlae.device))
        out = out.out.permute(0, 2, 1)
        out = self.avgpool(out)
        out = out.squeeze(-1)
        return out