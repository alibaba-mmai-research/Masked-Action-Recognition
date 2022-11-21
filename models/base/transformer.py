#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Transformers. """

import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from einops import rearrange, repeat
from models.base.backbone import BACKBONE_REGISTRY
from models.base.base_blocks import (
    STEM_REGISTRY, BRANCH_REGISTRY, HEAD_REGISTRY, DropPath, BaseHead
)

from models.utils.init_helper import lecun_normal_, trunc_normal_, _init_transformer_weights, _init_transformer_weights_xavier


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    import numpy as np
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


def get_3dsinusoid_encoding_table(h, w, t, d_hid):
    temperature=10000.
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_t = torch.arange(t, dtype=torch.float32)
    grid_w, grid_h, grid_t = torch.meshgrid(grid_w, grid_h, grid_t)
    assert d_hid % 6 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = d_hid // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_t = torch.einsum('m,d->md', [grid_t.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h), torch.sin(out_t), torch.cos(out_t)], dim=1)[None, :, :]


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, ff_dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    Self-attention module. 
    Currently supports both full self-attention on all the input tokens,
    or only-spatial/only-temporal self-attention. 

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer.
    and 
    Gedas Bertasius, Heng Wang, Lorenzo Torresani.
    Is Space-Time Attention All You Need for Video Understanding?

    Modified from 
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(
        self,
        dim,
        num_heads=12,
        attn_dropout=0.,
        ff_dropout=0.,
        einops_from=None,
        einops_to=None,
        **einops_dims,
    ):
        super().__init__()
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = dim_head ** -0.5

        self.to_qkv         = nn.Linear(dim, dim * 3)
        self.attn_dropout   = nn.Dropout(attn_dropout)
        self.proj           = nn.Linear(dim, dim)
        self.ff_dropout     = nn.Dropout(ff_dropout)

        if einops_from is not None and einops_to is not None:
            self.partial = True
            self.einops_from = einops_from
            self.einops_to = einops_to
            self.einops_dims = einops_dims
        else:
            self.partial = False

    def forward(self, x):
        if self.partial:
            return self.forward_partial(
                x,
                self.einops_from,
                self.einops_to,
                **self.einops_dims,
            )
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # import os
        # for i in range(12):
        #     if not os.path.exists(f"./debug/transformer_visualization/layer_{i}.pyth"):
        #         break
        # torch.save(attn,f"./debug/transformer_visualization/layer_{i}.pyth")
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

    def forward_partial(self, x, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q *= self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_attn = (cls_q @ k.transpose(1,2)).softmax(-1)
        cls_attn = self.attn_dropout(cls_attn)
        cls_out = cls_attn @ v

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r = r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim = 1)
        v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        attn = (q_ @ k_.transpose(1, 2)).softmax(-1)
        attn = self.attn_dropout(attn)
        x = attn @ v_

        # merge back time or space
        x = rearrange(x, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        x = torch.cat((cls_out, x), dim = 1)

        # merge back the heads
        x = rearrange(x, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

@STEM_REGISTRY.register()
class PatchEmbedStem(nn.Module):
    """ 
    Video to Patch Embedding.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        channels        = cfg.DATA.NUM_INPUT_CHANNELS       if cfg is not None else 3   # default 3
        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 16
        patch_size      = cfg.VIDEO.BACKBONE.PATCH_SIZE     if cfg is not None else 16  # default 16
        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        if isinstance(dim, list):
            dim = dim[0]

        num_patches_per_image = (image_size // patch_size) ** 2
        num_patches = num_patches_per_image * num_frames

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches

        self.conv1 = nn.Conv3d(
            in_channels     =channels, 
            out_channels    =dim, 
            kernel_size     =[1, patch_size, patch_size], 
            stride          =[1, patch_size, patch_size], 
        )

    def forward(self, x):
        b, c, t, h, w, p = *x.shape, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        x = self.conv1(x)
        # b, c, t, h, w -> b, c, p (p: num patches)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # b, c, p -> b, p, c
        x = x.permute(0, 2, 1)
        return x

@STEM_REGISTRY.register()
class TubeletEmbeddingStem(nn.Module):
    """ 
    Video to Tubelet Embedding.
    """
    def __init__(self, cfg, reshape_output=True):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        channels        = cfg.DATA.NUM_INPUT_CHANNELS       if cfg is not None else 3   # default 3
        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 16
        patch_size      = cfg.VIDEO.BACKBONE.PATCH_SIZE     if cfg is not None else 16  # default 16
        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        tubelet_size    = cfg.VIDEO.BACKBONE.TUBELET_SIZE   if cfg is not None else 2

        num_patches_per_image = (image_size // patch_size) ** 2
        num_patches = num_patches_per_image * num_frames

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.reshape_output = reshape_output
        self.conv1 = nn.Conv3d(
            in_channels     =channels, 
            out_channels    =dim, 
            kernel_size     =[tubelet_size, patch_size, patch_size], 
            stride          =[tubelet_size, patch_size, patch_size], 
        )

    def forward(self, x):
        b, c, t, h, w, p = *x.shape, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        x = self.conv1(x)
        if self.reshape_output:
            # b, c, t, h, w -> b, c, p (p: num patches)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            # b, c, p -> b, p, c
            x = x.permute(0, 2, 1)
        return x


@STEM_REGISTRY.register()
class MvitEmbeddingStem(nn.Module):
    """ 
    Video to Tubelet Embedding.
    """
    def __init__(self, cfg, reshape_output=True):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()
        image_size      = cfg.DATA.TRAIN_CROP_SIZE
        channels        = cfg.DATA.NUM_INPUT_CHANNELS
        num_frames      = cfg.DATA.NUM_INPUT_FRAMES
        dim             = cfg.VIDEO.BACKBONE.STEM.EMBED_DIM
        kernel_size     = cfg.VIDEO.BACKBONE.STEM.KERNEL_SIZE
        stride          = cfg.VIDEO.BACKBONE.STEM.STRIDE

        self.image_size = image_size
        self.num_frames = num_frames
        self.reshape_output = reshape_output
        self.conv1 = nn.Conv3d(
            in_channels     =channels, 
            out_channels    =dim, 
            kernel_size     =kernel_size, 
            stride          =stride, 
            padding         =[kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2],
        )

    def forward(self, x):
        x = self.conv1(x)
        if self.reshape_output:
            # b, c, t, h, w -> b, c, p (p: num patches)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            # b, c, p -> b, p, c
            x = x.permute(0, 2, 1)
        return x


@BRANCH_REGISTRY.register()
class BaseTransformerLayer(nn.Module):
    def __init__(self, cfg, dim_override=None, num_heads_override=None, attn_dropout_override=None,
                 ff_dropout_override=None, mlp_mult_override=None, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        dim             = dim_override       if dim_override is not None else cfg.VIDEO.BACKBONE.NUM_FEATURES
        num_heads       = num_heads_override if num_heads_override is not None else cfg.VIDEO.BACKBONE.NUM_HEADS
        attn_dropout    = attn_dropout_override if attn_dropout_override is not None else cfg.VIDEO.BACKBONE.ATTN_DROPOUT
        ff_dropout      = ff_dropout_override if ff_dropout_override is not None else cfg.VIDEO.BACKBONE.FF_DROPOUT
        mlp_mult        = mlp_mult_override if mlp_mult_override is not None else cfg.VIDEO.BACKBONE.MLP_MULT
        drop_path       = drop_path_rate

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

@BRANCH_REGISTRY.register()
class TimesformerLayer(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        patch_size      = cfg.VIDEO.BACKBONE.PATCH_SIZE     if cfg is not None else 16  # default 16
        drop_path       = drop_path_rate
        
        num_patches = (image_size // patch_size) ** 2

        self.norm_temporal = nn.LayerNorm(dim, eps=1e-6)
        self.attn_temporal = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout,
            einops_from='b (f n) d', einops_to='(b n) f d', n = num_patches
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout,
            einops_from='b (f n) d', einops_to='(b f) n d', f = num_frames
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn_temporal(self.norm_temporal(x)))
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

@BACKBONE_REGISTRY.register()
class Transformer(nn.Module):
    def __init__(
        self,
        cfg,
        mode="video"
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()

        mode = mode.upper()
        backbone_cfg = getattr(cfg, mode).BACKBONE

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        num_features    = backbone_cfg.NUM_FEATURES         if cfg is not None else 768 # default 768
        patch_size      = backbone_cfg.PATCH_SIZE           if cfg is not None else 16  # default 16
        depth           = backbone_cfg.DEPTH                if cfg is not None else 12  # default 12
        drop_path       = backbone_cfg.DROP_PATH            if cfg is not None else 16  # default 16
        use_cls_token   = backbone_cfg.USE_CLS_TOKEN        if hasattr(backbone_cfg, 'USE_CLS_TOKEN') else True
        if hasattr(backbone_cfg, "TUBELET_SIZE"):
            tubelet_size = backbone_cfg.TUBELET_SIZE         if cfg is not None else 2
        else:
            tubelet_size = 1

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.use_cls_token = use_cls_token
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size

        # constructs the tokenization module.
        self.stem = STEM_REGISTRY.get(backbone_cfg.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)

        if use_cls_token:
            self.pos_embd = nn.Parameter(torch.zeros(1, self.num_patches + 1, num_features))
            self.cls_token = nn.Parameter(torch.randn(1, 1, num_features))
        else:
            # self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches_per_frame, num_features))
            self.pos_embd           = get_sinusoid_encoding_table(self.num_patches_per_frame, num_features)

        # construct transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        if use_cls_token:
            trunc_normal_(self.pos_embd, std=.02)
            trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, x, mask=None):
        if isinstance(x, dict):
            x = x["video"]

        x = self.stem(x)

        if self.use_cls_token:
            cls_token = self.cls_token.repeat((x.shape[0],1,1))
            x =  torch.cat((cls_token, x), dim = 1)

        if type(self.pos_embd) is nn.Parameter:
            x += self.pos_embd.expand_as(x)
        else:
            x += self.pos_embd.clone().detach().to(x.device).expand_as(x)

        if mask is not None:
            return self.forward_pretrain(x, mask)
        else:
            return self.forward_finetune(x)
    
    def forward_finetune(self, x):
        x = self.layers(x)
        x = self.norm(x)
        if self.use_cls_token:
            return x[:, 0]
        else:
            return x.mean(dim=1)

    def forward_pretrain(self, x, mask):
        bt, n, c = x.size()

        mask = mask.flatten(0, 1)
        x = x[mask, :].reshape(bt, -1, c)
        x = self.layers(x)
        x = self.norm(x)
        x = rearrange(x, "(b t) n c -> b t n c", t=self.num_patches//self.num_patches_per_frame)
        return x

    def get_num_layers(self):
        return len(self.layers), 0


@BACKBONE_REGISTRY.register()
class FactorizedTransformer(nn.Module):
    """
    The factorized transformer. 

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer. 
    """
    def __init__(
        self,
        cfg,
        mode="video"
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()

        mode = mode.upper()
        backbone_cfg = getattr(cfg, mode).BACKBONE

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        num_features    = backbone_cfg.NUM_FEATURES         if cfg is not None else 768 # default 768
        patch_size      = backbone_cfg.PATCH_SIZE           if cfg is not None else 16  # default 16
        depth           = backbone_cfg.DEPTH                if cfg is not None else 12  # default 12
        depth_temp      = backbone_cfg.DEPTH_TEMP           if cfg is not None else 4   # default 4
        drop_path       = backbone_cfg.DROP_PATH            if cfg is not None else 16  # default 16
        use_cls_token   = backbone_cfg.USE_CLS_TOKEN        if hasattr(backbone_cfg, 'USE_CLS_TOKEN') else True
        pos_embd_learnable = backbone_cfg.POS_EMBD_LEARNABLE if hasattr(backbone_cfg, 'POS_EMBD_LEARNABLE') else True
        init_approach = backbone_cfg.INIT_APPROACH if hasattr(backbone_cfg, 'INIT_APPROACH') else 'trunc_normal_'
        if hasattr(backbone_cfg, "TUBELET_SIZE"):
            tubelet_size = backbone_cfg.TUBELET_SIZE         if cfg is not None else 2
        else:
            tubelet_size = 1
        if hasattr(cfg.PRETRAIN, "MAE"):
            self.tem_agg_type = cfg.PRETRAIN.MAE.TEM_AGG_TYPE
        else:
            self.tem_agg_type = 'sum'

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.use_cls_token = use_cls_token
        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size

        # constructs the tokenization module.
        self.stem = STEM_REGISTRY.get(backbone_cfg.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)

        # both spatial and temporal embeddings/cls_token needs to be constructed
        # for the factorized transformer video model 
        if use_cls_token:
            assert pos_embd_learnable
            self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches_per_frame + 1, num_features))
            self.temp_embd          = nn.Parameter(torch.zeros(1, num_frames // tubelet_size + 1, num_features))
            self.cls_token          = nn.Parameter(torch.randn(1, 1, num_features))
            self.cls_token_out = nn.Parameter(torch.randn(1, 1, num_features))
        else:
            if pos_embd_learnable:
                self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches_per_frame, num_features))
                self.temp_embd          = nn.Parameter(torch.zeros(1, num_frames // tubelet_size, num_features))
            else:
                self.pos_embd           = get_sinusoid_encoding_table(self.num_patches_per_frame, num_features)
                self.temp_embd          = get_sinusoid_encoding_table(num_frames // tubelet_size, num_features)

        # construct spatial transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth+depth_temp)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # construct temporal transformer layers
        self.layers_temporal = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i+depth])
            for i in range(depth_temp)])

        self.norm_out = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        if type(self.pos_embd) is nn.Parameter:
            trunc_normal_(self.pos_embd, std=.02)
        if type(self.temp_embd) is nn.Parameter:
            trunc_normal_(self.temp_embd, std=.02)
        if use_cls_token:
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_token_out, std=.02)
        if init_approach == 'trunc_normal_':
            self.apply(_init_transformer_weights)
        elif init_approach == 'xavier':
            self.apply(_init_transformer_weights_xavier)

    def forward(self, x, mask=None):
        if isinstance(x, dict):
            x = x["video"]

        h, w = x.shape[-2:]

        actual_num_patches_per_frame = (h // self.patch_size) * (w // self.patch_size)
        x = self.stem(x)
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            assert not self.training 
            x = rearrange(x, "b (t n) c -> (b t) n c", n=actual_num_patches_per_frame)
        else:
            x = rearrange(x, "b (t n) c -> (b t) n c", n=self.num_patches_per_frame)
        if self.use_cls_token:
            cls_token = self.cls_token.repeat((x.shape[0],1,1))
            x = torch.cat((cls_token, x), dim = 1)

        # to make the input video size changable
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            actual_num_pathces_per_side = int(math.sqrt(actual_num_patches_per_frame))
            if not hasattr(self, "new_pos_embd") or self.new_pos_embd.shape[1] != (actual_num_pathces_per_side**2+1):
                cls_pos_embd = self.pos_embd[:,0,:].unsqueeze(1)
                pos_embd = self.pos_embd[:, 1:, :]
                num_patches_per_side = int(math.sqrt(self.num_patches_per_frame))
                pos_embd = pos_embd.reshape(1, num_patches_per_side, num_patches_per_side, -1).permute(0,3,1,2)
                pos_embd = torch.nn.functional.interpolate(
                    pos_embd, size=(actual_num_pathces_per_side,actual_num_pathces_per_side), mode="bilinear"
                ).permute(0,2,3,1).reshape(1, actual_num_pathces_per_side**2, -1)
                self.new_pos_embd = torch.cat((cls_pos_embd, pos_embd), dim=1)
            x += self.new_pos_embd
        else:
            if type(self.pos_embd) is nn.Parameter:
                x += self.pos_embd
            else:
                x += self.pos_embd.clone().detach().to(x.device).expand_as(x)
        if mask is not None:
            return self.forward_pretrain(x, mask)
        else:
            return self.forward_finetune(x)

    def forward_finetune(self, x):
        x = self.layers(x)
        if self.use_cls_token:
            x = self.norm(x)[:, 0]
        else:
            x = self.norm(x).mean(dim=1)

        x = rearrange(x, "(b t) c -> b t c", t=self.num_patches//self.num_patches_per_frame)

        if self.use_cls_token:
            cls_token_out = self.cls_token_out.repeat((x.shape[0], 1, 1))
            x = torch.cat((cls_token_out, x), dim=1)
        else:
            x_s = x.clone()

        if type(self.temp_embd) is nn.Parameter:
            temp_embd = self.temp_embd.expand_as(x)
        else:
            temp_embd = self.temp_embd.clone().detach().to(x.device).expand_as(x)
        x += temp_embd

        x = self.layers_temporal(x)
        x = self.norm_out(x)

        if self.use_cls_token:
            return x[:, 0]
        else:
            if self.tem_agg_type == "sum":
                return (x + x_s).mean(dim=1)
            elif self.tem_agg_type == "avg":
                return (x/2.0 + x_s/2.0).mean(dim=1)
            elif self.tem_agg_type == "concat":
                return torch.cat([x, x_s], dim=-1).mean(dim=1)

    def forward_pretrain(self, x, mask):
        b, t, p = mask.size()
        bt, n, c = x.size()
        mask = mask.flatten(0, 1)
        x = x[mask, :].reshape(bt, -1, c)
        x = self.layers(x)
        x_t = self.norm(x).mean(dim=1)

        x_t = rearrange(x_t, "(b t) c -> b t c", t=self.num_patches//self.num_patches_per_frame)
        if type(self.temp_embd) is nn.Parameter:
            temp_embd = self.temp_embd.expand_as(x_t)
        else:
            temp_embd = self.temp_embd.clone().detach().to(x_t.device).expand_as(x_t)
        x_t += temp_embd
        # temp_embed = temp_embd[:, :, None, :]

        x_t = self.layers_temporal(x_t)
        x_t = self.norm_out(x_t)
        x = rearrange(x, "(b t) n c -> b t n c", t=self.num_patches//self.num_patches_per_frame)
        if self.tem_agg_type == 'sum':
            x = x + x_t[:, :, None, :]
            # pos_embed_mask = pos_embed_mask + temp_embed
            # pos_embed_vis = pos_embed_vis + temp_embed
        elif self.tem_agg_type == 'avg':
            x = (x + x_t[:, :, None, :]) / 2.0
            # pos_embed_mask = (pos_embed_mask + temp_embed) / 2.0
            # pos_embed_vis = (pos_embed_vis + temp_embed) / 2.0
        elif self.tem_agg_type == 'concat':
            x = torch.cat([x, x_t[:, :, None, :].expand_as(x)], dim=-1)
            # pos_embed_mask = torch.cat([pos_embed_mask, temp_embed[:, :, None, :].expand_as(pos_embed_mask)], dim=-1)
            # pos_embed_vis = torch.cat([pos_embed_vis, temp_embed[:, :, None, :].expand_as(pos_embed_vis)], dim=-1)
        return x

    def get_num_layers(self):
        return len(self.layers), len(self.layers_temporal)


@BACKBONE_REGISTRY.register()
class JointTransformer(nn.Module):
    """
    The factorized transformer. 

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer. 
    """
    def __init__(
        self,
        cfg,
        mode="video"
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()

        mode = mode.upper()
        backbone_cfg = getattr(cfg, mode).BACKBONE

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        num_features    = backbone_cfg.NUM_FEATURES         if cfg is not None else 768 # default 768
        patch_size      = backbone_cfg.PATCH_SIZE           if cfg is not None else 16  # default 16
        depth           = backbone_cfg.DEPTH                if cfg is not None else 12  # default 12
        drop_path       = backbone_cfg.DROP_PATH            if cfg is not None else 16  # default 16
        use_cls_token   = backbone_cfg.USE_CLS_TOKEN        if hasattr(backbone_cfg, 'USE_CLS_TOKEN') else True
        pos_embed_learnable = backbone_cfg.POS_EMBD_LEARNABLE if hasattr(backbone_cfg, 'POS_EMBD_LEARNABLE') else True
        init_approach = backbone_cfg.INIT_APPROACH if hasattr(backbone_cfg, 'INIT_APPROACH') else 'trunc_normal_'
        if hasattr(backbone_cfg, "TUBELET_SIZE"):
            tubelet_size = backbone_cfg.TUBELET_SIZE         if cfg is not None else 2
        else:
            tubelet_size = 1

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.use_cls_token = use_cls_token
        self.pos_embed_learnable = pos_embed_learnable
        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size
        self.patch_h, self.patch_w = image_size // patch_size, image_size // patch_size
        self.patch_t = num_frames // tubelet_size

        # constructs the tokenization module.
        self.stem = STEM_REGISTRY.get(backbone_cfg.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)

        # both spatial and temporal embeddings/cls_token needs to be constructed
        # for the factorized transformer video model 
        if use_cls_token:
            assert pos_embed_learnable
            self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches_per_frame + 1, num_features))
            self.temp_embd          = nn.Parameter(torch.zeros(1, num_frames // tubelet_size + 1, num_features))
            self.cls_token          = nn.Parameter(torch.randn(1, 1, num_features))
            self.cls_token_out = nn.Parameter(torch.randn(1, 1, num_features))
        else:
            if pos_embed_learnable:
                self.pos_embed           = nn.Parameter(torch.zeros(1, self.num_patches, num_features))
            else:
                self.pos_embed           = nn.Parameter(get_sinusoid_encoding_table(self.num_patches, num_features), requires_grad=False)

        # construct spatial transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        if type(self.pos_embed) is nn.Parameter and pos_embed_learnable:
            trunc_normal_(self.pos_embed, std=1e-5)
        if use_cls_token:
            trunc_normal_(self.cls_token, std=1e-5)
        if init_approach == 'trunc_normal_':
            self.apply(_init_transformer_weights)
        elif init_approach == 'xavier':
            self.apply(_init_transformer_weights_xavier)

    def forward(self, x, mask=None):
        if isinstance(x, dict):
            x = x["video"]

        h, w = x.shape[-2:]

        actual_num_patches_per_frame = (h // self.patch_size) * (w // self.patch_size)
        x = self.stem(x) # b, n, c
        patch_embed = x
        if self.use_cls_token:
            cls_token = self.cls_token.repeat((x.shape[0],1,1))
            x = torch.cat((cls_token, x), dim = 1)

        # to make the input video size changable
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            raise NotImplementedError("Different input not supported")
        else:
            if self.pos_embed_learnable:
                x += self.pos_embed
            else:
                x += self.pos_embed.clone().detach().expand_as(x)
        if isinstance(mask, torch.Tensor):
            return self.forward_pretrain(x, mask), patch_embed
        elif isinstance(mask, int):
            return self.forward_finetune(x), [self.patch_t, self.patch_h, self.patch_w]
        else:
            feat = self.forward_finetune(x)
            if self.use_cls_token:
                feat = feat[:, 0]
            else:
                feat = feat.mean(dim=1)
            return feat

    def forward_finetune(self, x):
        x = self.layers(x)
        return self.norm(x)

    def forward_pretrain(self, x, mask):
        # b, t, p = mask.size()
        b, n, c = x.size()
        mask = mask.flatten(1)
        x = x[mask, :].reshape(b, -1, c)
        x = self.layers(x)
        x = self.norm(x)
        return x

    def get_num_layers(self):
        return len(self.layers), 0


@BACKBONE_REGISTRY.register()
class JointTransformerWithMask(JointTransformer):
    """
    The factorized transformer. 

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer. 
    """
    def __init__(
        self,
        cfg,
        mode="video"
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__(cfg, mode)
        self.mask_ratio = cfg.VIDEO.BACKBONE.MASK_RATIO
        self.mask_stride = cfg.VIDEO.BACKBONE.MASK_STRIDE if hasattr(cfg.VIDEO.BACKBONE, 'MASK_STRIDE') else [1, 1, 1]
        self.patch_size = cfg.VIDEO.BACKBONE.PATCH_SIZE
        self.tublet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE if hasattr(cfg.VIDEO.BACKBONE, 'TUBELET_SIZE') else 1
        self.input_size = [cfg.DATA.NUM_INPUT_FRAMES, cfg.DATA.TRAIN_CROP_SIZE]
        self.patches_shape = [self.input_size[0]//self.tublet_size, self.input_size[1]//self.patch_size, self.input_size[1]//self.patch_size]
        try:
            self.repeat_mask_num = cfg.AGENT.REPEAT_SAMPLE_NUM if hasattr(cfg.AGENT, 'REPEAT_SAMPLE_NUM') else 1
        except:
            self.repeat_mask_num = 1

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def random_masking_stride(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        Ls = L //self.mask_stride[0] // self.mask_stride[1] // self.mask_stride[2]
        len_keep = int(Ls * (1 - mask_ratio))
        noise = torch.rand(N, Ls, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones([N, Ls], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = 1.0 - mask
        mask = mask.reshape(N, self.patches_shape[0]//self.mask_stride[0], self.patches_shape[1]//self.mask_stride[1], self.patches_shape[2]//self.mask_stride[2])
        mask = mask.repeat_interleave(self.mask_stride[0], dim=1).repeat_interleave(self.mask_stride[1], dim=2).repeat_interleave(self.mask_stride[2], dim=3)
        return mask

    def learning_masking(self, x, mask_ratio, mask):
        if self.repeat_mask_num > 1 and x.size(0) != mask.size(0):
            x = x.repeat_interleave(self.repeat_mask_num, dim=0)
            # mask = mask.flatten(0, 1)
        N, L, D = x.shape  # batch, length, dim
        # len_keep = int(L * (1 - mask_ratio))

        ids_keep = torch.nonzero(mask.flatten(1))[:, 1].reshape(N, -1)
        ids_drop = torch.nonzero(1.0 - mask.flatten(1))[:, 1].reshape(N, -1)
        ids_shuffle = torch.cat([ids_keep, ids_drop], dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep = ids_keep.size(1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x):
        agent_output = None
        if isinstance(x, dict):
            if 'agent_output' in x:
                agent_output = x['agent_output']
            x = x["video"]

        h, w = x.shape[-2:]

        actual_num_patches_per_frame = (h // self.patch_size) * (w // self.patch_size)
        x = self.stem(x) # b, n, c
        assert self.use_cls_token == False
        if self.use_cls_token:
            cls_token = self.cls_token.repeat((x.shape[0],1,1))
            x = torch.cat((cls_token, x), dim = 1)

        # to make the input video size changable
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            raise NotImplementedError("Different input not supported")
        else:
            if self.pos_embed_learnable:
                x += self.pos_embed
            else:
                x += self.pos_embed.clone().detach().expand_as(x)
        if agent_output is not None and len(agent_output) > 0:
            x, mask, ids_restore = self.learning_masking(x, self.mask_ratio, agent_output['mask'])
        else:
            mask = self.random_masking_stride(x, self.mask_ratio)
            # mask = mask.flatten(1).fill_(0)
            # mask[:, :int(mask.size(1) * (1.0-self.mask_ratio))] = 1
            x, mask, ids_restore = self.learning_masking(x, self.mask_ratio, mask)
        feat = self.forward_finetune(x)
        if agent_output is not None and len(agent_output) > 0 and 'indices' in agent_output:
            mask_tokens = feat.new_zeros(feat.shape[0], ids_restore.shape[1] - feat.shape[1], feat.shape[2])
            feat_ = torch.cat([feat[:, :, :], mask_tokens], dim=1)
            feat_ = torch.gather(feat_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, feat.shape[2]))  # unshuffle
            stride_logits = rearrange(feat_, 'b (t s0 h s1 w s2) c -> b (t h w) (s0 s1 s2) c', s0=self.mask_stride[0], 
                                                                                            s1=self.mask_stride[1], 
                                                                                            s2=self.mask_stride[2],
                                                                                            t=self.patches_shape[0]//self.mask_stride[0], 
                                                                                            h=self.patches_shape[1]//self.mask_stride[1], 
                                                                                            w=self.patches_shape[2]//self.mask_stride[2])
            feat = torch.gather(stride_logits, dim=1, index=agent_output['indices'][:, :, None, None].repeat(1, 1, stride_logits.size(2), stride_logits.size(3)))
            feat = feat.mean(dim=2)
        else:
            if self.use_cls_token:
                feat = feat[:, 0]
            else:
                feat = feat.mean(dim=1)
        return feat

    def forward_finetune(self, x):
        x = self.layers(x)
        return self.norm(x)

    def get_num_layers(self):
        return len(self.layers), 0


@BACKBONE_REGISTRY.register()
class VisionTransformer(nn.Module):
    def __init__(
        self,
        cfg,
        mode="video"
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()

        mode = mode.upper()
        backbone_cfg = getattr(cfg, mode).BACKBONE

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        tubelet_size    = backbone_cfg.TUBELET_SIZE
        num_features    = backbone_cfg.NUM_FEATURES         if cfg is not None else 768 # default 768
        patch_size      = backbone_cfg.PATCH_SIZE           if cfg is not None else 16  # default 16
        depth           = backbone_cfg.DEPTH                if cfg is not None else 12  # default 12
        drop_path       = backbone_cfg.DROP_PATH            if cfg is not None else 16  # default 16

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size

        # constructs the tokenization module.
        self.stem = STEM_REGISTRY.get(backbone_cfg.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)

        # both spatial and temporal embeddings/cls_token needs to be constructed
        # for the factorized transformer video model 
        # self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches + 1, num_features))
        self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches_per_frame + 1, num_features))
        self.cls_token          = nn.Parameter(torch.randn(1, 1, num_features))

        # construct spatial transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]

        h, w = x.shape[-2:]

        actual_num_patches_per_frame = (h // self.patch_size) * (w // self.patch_size)
        x = self.stem(x)
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            assert not self.training 
            x = rearrange(x, "b (t n) c -> (b t) n c", n=actual_num_patches_per_frame)
        else:
            x = rearrange(x, "b (t n) c -> (b t) n c", n=self.num_patches_per_frame)

        cls_token = self.cls_token.repeat((x.shape[0],1,1))
        x = torch.cat((cls_token, x), dim = 1)

        x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)[:, 0]

        x = rearrange(x, "(b t) c -> b t c", t=self.num_patches//self.num_patches_per_frame)

        x = x.mean(1)

        return x