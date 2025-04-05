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
### VIT ###

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from collections import namedtuple
CombinedOutput = namedtuple('CombinedOutput', ['loss', 'out'])

__all__ = ['ViT', 'vit_small', 'vit_base']


class DropPath(nn.Module):
    '''
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    '''
    def __init__(self, drop_prob: float, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PreNorm(nn.Module):
    def __init__(self,
                 dim: int,
                 fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    MLP Module with GELU activation fn + dropout.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 drop_out_rate=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(drop_out_rate),
                                 nn.Linear(hidden_dim, output_dim),
                                 nn.Dropout(drop_out_rate))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 heads: int = 8,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == input_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop_out_rate)
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=qkv_bias)

        if project_out:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, output_dim),
                                        nn.Dropout(drop_out_rate))
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 heads: int = 8,
                 dim_head: int = 32,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        attn = Attention(input_dim=input_dim,
                         output_dim=output_dim,
                         heads=heads,
                         dim_head=dim_head,
                         qkv_bias=qkv_bias,
                         drop_out_rate=drop_out_rate,
                         attn_drop_out_rate=attn_drop_out_rate)
        self.attn = PreNorm(dim=input_dim,
                            fn=attn)
        self.droppath1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        ff = FeedForward(input_dim=output_dim,
                         output_dim=output_dim,
                         hidden_dim=hidden_dim,
                         drop_out_rate=drop_out_rate)
        self.ff = PreNorm(dim=output_dim,
                          fn=ff)
        self.droppath2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.droppath1(self.attn(x)) + x
        x = self.droppath2(self.ff(x)) + x
        return x


class ViT(nn.Module):
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
        super().__init__()
        assert seq_len % patch_size == 0, 'The sequence length must be divisible by the patch size.'
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
        num_patches = seq_len // patch_size
        patch_dim = num_leads * patch_size
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (n p) -> b n (p c)', p=patch_size),
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

    def reset_head(self, num_classes: Optional[int] = None):
        del self.head
        self.head = nn.Identity() if num_classes is None else nn.Linear(self.width, num_classes)

    def forward_encoding(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape
        cls_embeddings = repeat(self.cls_embedding, 'd -> b d', b=b)
        x, ps = pack([cls_embeddings, x], 'b * d')
        x = x + self.pos_embedding[:, :n + 1]

        x = self.dropout(x)
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)

        cls_embeddings, _ = unpack(x, ps, 'b * d')

        return self.norm(cls_embeddings)

    def forward(self, series):
        x = self.forward_encoding(series)
        return self.head(x)

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str


def vit_small(num_leads, num_classes=None, seq_len=2250, patch_size=75, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      width=384,
                      depth=12,
                      heads=6,
                      mlp_dim=1536,
                      **kwargs)
    return ViT(**model_args)


def vit_base(num_leads, num_classes=None, seq_len=2250, patch_size=75, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      width=768,
                      depth=12,
                      heads=12,
                      mlp_dim=3072,
                      **kwargs)
    return ViT(**model_args)


### ENCODER / STE_MEM_VIT ###
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


__all__ = ['ST_MEM_ViT', 'st_mem_vit_small', 'st_mem_vit_base']


class ST_MEM_ViT(nn.Module):
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
        super().__init__()
        assert seq_len % patch_size == 0, 'The sequence length must be divisible by the patch size.'
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
        num_patches = seq_len // patch_size
        patch_dim = patch_size
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (n p) -> b c n p', p=patch_size),
                                                nn.LayerNorm(patch_dim),
                                                nn.Linear(patch_dim, width),
                                                nn.LayerNorm(width))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, width))
        self.sep_embedding = nn.Parameter(torch.randn(width))
        self.lead_embeddings = nn.ParameterList(nn.Parameter(torch.randn(width))
                                                for _ in range(num_leads))

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

    def reset_head(self, num_classes: Optional[int] = None):
        del self.head
        self.head = nn.Identity() if num_classes is None else nn.Linear(self.width, num_classes)

    def forward_encoding(self, series):
        num_leads = series.shape[1]
        if num_leads > len(self.lead_embeddings):
            raise ValueError(f'Number of leads ({num_leads}) exceeds the number of lead embeddings')

        x = self.to_patch_embedding(series)
        b, _, n, _ = x.shape
        x = x + self.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # lead indicating modules
        sep_embedding = self.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
        lead_embeddings = torch.stack([lead_embedding for lead_embedding in self.lead_embeddings]).unsqueeze(0)
        lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n + 2, -1)
        x = x + lead_embeddings
        x = rearrange(x, 'b c n p -> b (c n) p')

        x = self.dropout(x)
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)

        # remove SEP embeddings
        x = rearrange(x, 'b (c n) p -> b c n p', c=num_leads)
        x = x[:, :, 1:-1, :]

        x = torch.mean(x, dim=(1, 2))
        return self.norm(x)

    def forward(self, series):
        x = self.forward_encoding(series)
        return self.head(x)

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str


def st_mem_vit_small(num_leads, num_classes=None, seq_len=2250, patch_size=75, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      width=384,
                      depth=12,
                      heads=6,
                      mlp_dim=1536,
                      **kwargs)
    return ST_MEM_ViT(**model_args)


def st_mem_vit_base(num_leads, num_classes=None, seq_len=2250, patch_size=75, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      width=768,
                      depth=12,
                      heads=12,
                      mlp_dim=3072,
                      **kwargs)
    return ST_MEM_ViT(**model_args)


#####

#### ST_MEM ###
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional

__all__ = ['ST_MEM', 'st_mem_vit_small_dec256d4b', 'st_mem_vit_base_dec256d4b']


def get_1d_sincos_pos_embed(embed_dim: int,
                            grid_size: int,
                            temperature: float = 10000,
                            sep_embed: bool = False):
    """Positional embedding for 1D patches.
    """
    assert (embed_dim % 2) == 0, \
        'feature dimension must be multiple of 2 for sincos emb.'
    grid = torch.arange(grid_size, dtype=torch.float32)

    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= (embed_dim / 2.)
    omega = 1. / (temperature ** omega)

    grid = grid.flatten()[:, None] * omega[None, :]
    pos_embed = torch.cat((grid.sin(), grid.cos()), dim=1)
    if sep_embed:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed, torch.zeros(1, embed_dim)],
                              dim=0)
    return pos_embed


class ST_MEM(nn.Module):
    def __init__(self,
                 seq_len: int = 2250,
                 patch_size: int = 75,
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
        super().__init__()
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
        self.num_patches = seq_len // patch_size
        self.num_leads = num_leads
        # --------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = ST_MEM_ViT(seq_len=seq_len,
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
            torch.zeros(1, self.num_patches + 2, decoder_embed_dim),
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
        self.decoder_head = nn.Linear(decoder_embed_dim, patch_size)
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.encoder.pos_embedding.shape[-1],
                                            self.num_patches,
                                            sep_embed=True)
        self.encoder.pos_embedding.data.copy_(pos_embed.float().unsqueeze(0))
        self.encoder.pos_embedding.requires_grad = False

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    self.num_patches,
                                                    sep_embed=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.encoder.sep_embedding, std=.02)
        torch.nn.init.normal_(self.mask_embedding, std=.02)
        for i in range(self.num_leads):
            torch.nn.init.normal_(self.encoder.lead_embeddings[i], std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, series):
        """
        series: (batch_size, num_leads, seq_len)
        x: (batch_size, num_leads, n, patch_size)
        """
        p = self.patch_size
        assert series.shape[2] % p == 0
        x = rearrange(series, 'b c (n p) -> b c n p', p=p)
        return x

    def unpatchify(self, x):
        """
        x: (batch_size, num_leads, n, patch_size)
        series: (batch_size, num_leads, seq_len)
        """
        series = rearrange(x, 'b c n p -> b c (n p)')
        return series

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: (batch_size, num_leads, n, embed_dim)
        """
        b, num_leads, n, d = x.shape
        len_keep = int(n * (1 - mask_ratio))

        noise = torch.rand(b, num_leads, n, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=2)

        # keep the first subset
        ids_keep = ids_shuffle[:, :, :len_keep]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, d))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([b, num_leads, n], device=x.device)
        mask[:, :, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        x: (batch_size, num_leads, seq_len)
        """
        # embed patches
        x = self.to_patch_embedding(x)
        b, _, n, _ = x.shape

        # add positional embeddings
        x = x + self.encoder.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # masking: length -> length * mask_ratio
        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = torch.zeros([b, self.num_leads, n], device=x.device)
            ids_restore = torch.arange(n, device=x.device).unsqueeze(0).repeat(b, self.num_leads, 1)

        # apply lead indicating modules
        # 1) SEP embedding
        sep_embedding = self.encoder.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, self.num_leads, -1, -1) + self.encoder.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, self.num_leads, -1, -1) + self.encoder.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
        # 2) lead embeddings
        n_masked_with_sep = x.shape[2]
        lead_embeddings = torch.stack([self.encoder.lead_embeddings[i] for i in range(self.num_leads)]).unsqueeze(0)
        lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n_masked_with_sep, -1)
        x = x + lead_embeddings

        x = rearrange(x, 'b c n p -> b (c n) p')
        for i in range(self.encoder.depth):
            x = getattr(self.encoder, f'block{i}')(x)
        x = self.encoder.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.to_decoder_embedding(x)

        # append mask embeddings to sequence
        x = rearrange(x, 'b (c n) p -> b c n p', c=self.num_leads)
        b, _, n_masked_with_sep, d = x.shape
        n = ids_restore.shape[2]
        mask_embeddings = self.mask_embedding.unsqueeze(1)
        mask_embeddings = mask_embeddings.repeat(b, self.num_leads, n + 2 - n_masked_with_sep, 1)

        # Unshuffle without SEP embedding
        x_wo_sep = torch.cat([x[:, :, 1:-1, :], mask_embeddings], dim=2)
        x_wo_sep = torch.gather(x_wo_sep, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, d))

        # positional embedding and SEP embedding
        x_wo_sep = x_wo_sep + self.decoder_pos_embed[:, 1:n + 1, :].unsqueeze(1)
        left_sep = x[:, :, :1, :] + self.decoder_pos_embed[:, :1, :].unsqueeze(1)
        right_sep = x[:, :, -1:, :] + self.decoder_pos_embed[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x_wo_sep, right_sep], dim=2)

        # lead-wise decoding
        x_decoded = []
        x_latents = []
        for i in range(self.num_leads):
            x_lead = x[:, i, :, :]
            for block in self.decoder_blocks:
                x_lead = block(x_lead)
            x_latents.append(x_lead)
            x_lead = self.decoder_norm(x_lead)
            x_lead = self.decoder_head(x_lead)
            x_decoded.append(x_lead[:, 1:-1, :])
        x = torch.stack(x_decoded, dim=1)
        x_latents = torch.stack(x_latents, dim=1)
        return x, x_latents

    def forward_loss(self, series, pred, mask):
        """
        series: (batch_size, num_leads, seq_len)
        pred: (batch_size, num_leads, n, patch_size)
        mask: (batch_size, num_leads, n), 0 is keep, 1 is remove,
        """
        target = self.patchify(series)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (batch_size, num_leads, n), mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self,
                series,
                mask_ratio=0.75):
        recon_loss = 0
        pred = None
        mask = None
        latent, mask, ids_restore = self.forward_encoder(series.to(self.to_decoder_embedding.weight.dtype), mask_ratio)
        pred, x_latents = self.forward_decoder(latent, ids_restore)
        recon_loss = self.forward_loss(series, pred, mask)
        return CombinedOutput(
            loss=recon_loss,
            out = x_latents
        )

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str


def st_mem_vit_small_dec256d4b(device,**kwargs):
    model = ST_MEM(embed_dim=384,
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


def st_mem_vit_base_dec256d4b(device,**kwargs):
    model = ST_MEM(embed_dim=768,
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

class ST_MEM_Ours(nn.Module):
    def __init__(self, st_mem):
        super(ST_MEM_Ours, self).__init__()
        self.st_mem = st_mem
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
    def forward(self, batch):
        return self.st_mem(batch['signal'].to(self.st_mem.device))
    
    @torch.no_grad()
    def get_embeddings(self, batch):
        self.st_mem.eval()
        out = self.st_mem(batch['signal'].to(self.st_mem.device))
        out = out.out.permute(0, 3, 1, 2)
        out = self.avgpool(out)
        out = out.squeeze(-1).squeeze(-1)
        return out
