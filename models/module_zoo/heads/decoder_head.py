import torch
import torch.nn as nn

from models.base.base_blocks import BaseHead
from models.base.base_blocks import HEAD_REGISTRY
from models.base.base_blocks import (
    STEM_REGISTRY, BRANCH_REGISTRY, HEAD_REGISTRY, DropPath, BaseHead
)
from functools import partial


@HEAD_REGISTRY.register()
class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg.VIDEO.BACKBONE.PATCH_SIZE
        self.tublet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE if hasattr(cfg.VIDEO.BACKBONE, 'TUBELET_SIZE') else 1
        if hasattr(cfg.VIDEO.HEAD.DECODER, 'PREDICT_TYPE') and cfg.VIDEO.HEAD.DECODER.PREDICT_TYPE == 'rgb+flow':
            self.output_dim = 2 * 3 * self.tublet_size * self.patch_size ** 2
        elif 'mvit' in cfg.MODEL.NAME.lower():
            self.output_dim = 2 * 3 * self.patch_size ** 2
        else:
            self.output_dim = 3 * self.tublet_size * self.patch_size ** 2
        self.num_features = cfg.VIDEO.HEAD.DECODER.NUM_FEATURES
        self.num_heads = cfg.VIDEO.HEAD.DECODER.NUM_HEADS
        drop_path       = cfg.VIDEO.HEAD.DECODER.DROP_PATH
        depth       = cfg.VIDEO.HEAD.DECODER.DEPTH

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            BRANCH_REGISTRY.get(cfg.VIDEO.HEAD.DECODER.NAME)(cfg, num_heads_override=self.num_heads, dim_override=self.num_features, drop_path_rate=dpr[i])
            for i in range(depth)])
        self.norm =  nn.LayerNorm(self.num_features, eps=1e-6)
        self.head = nn.Linear(self.num_features, self.output_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.num_features))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.head = nn.Linear(self.num_features, self.output_dim)

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x)) # [B, N, 3*16^2]

        return x


@HEAD_REGISTRY.register()
class TransformerBridgingClsHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg):
        super().__init__()
        self.output_dim = cfg.VIDEO.HEAD.NUM_CLASSES
        self.num_features = cfg.VIDEO.HEAD.CLS_DECODER.NUM_FEATURES
        self.num_heads = cfg.VIDEO.HEAD.CLS_DECODER.NUM_HEADS
        drop_path       = cfg.VIDEO.HEAD.CLS_DECODER.DROP_PATH
        depth       = cfg.VIDEO.HEAD.CLS_DECODER.DEPTH
        dropout_rate = cfg.VIDEO.HEAD.CLS_DECODER.DROP_OUT if hasattr(cfg.VIDEO.HEAD.CLS_DECODER, 'DROP_OUT') else 0.0
        attn_drop_out = cfg.VIDEO.HEAD.CLS_DECODER.ATTN_DROP_OUT if hasattr(cfg.VIDEO.HEAD.CLS_DECODER, 'ATTN_DROP_OUT') else 0.0
        ffn_drop_out = cfg.VIDEO.HEAD.CLS_DECODER.FFN_DROP_OUT if hasattr(cfg.VIDEO.HEAD.CLS_DECODER, 'FFN_DROP_OUT') else 0.0
        multi_layer_forward = cfg.VIDEO.HEAD.CLS_DECODER.MULTI_LAYER_FORWARD if hasattr(cfg.VIDEO.HEAD.CLS_DECODER, 'MULTI_LAYER_FORWARD') else False

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.mask_inv_test = cfg.VIDEO.BACKBONE.MASK_INV_TEST if hasattr(cfg.VIDEO.BACKBONE, 'MASK_INV_TEST') else False
        self.multi_layer_forward = multi_layer_forward
        self.blocks = nn.ModuleList([
            BRANCH_REGISTRY.get(cfg.VIDEO.HEAD.DECODER.NAME)(cfg, num_heads_override=self.num_heads, dim_override=self.num_features,
                                                            drop_path_rate=dpr[i], attn_dropout_override=attn_drop_out, ff_dropout_override=ffn_drop_out)
            for i in range(depth)])
        if dropout_rate > 0.0: 
            self.dropout = nn.Dropout(dropout_rate)
        self.norm =  nn.LayerNorm(self.num_features, eps=1e-6)
        self.head = nn.Linear(self.num_features, self.output_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.num_features))
        self.apply(self._init_weights)
        if hasattr(cfg.VIDEO.HEAD, "INIT_SCALE"):
            self.head.weight.data.mul_(cfg.VIDEO.HEAD.INIT_SCALE)
            self.head.bias.data.mul_(cfg.VIDEO.HEAD.INIT_SCALE)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.head = nn.Linear(self.num_features, self.output_dim)

    def forward(self, x):
        if self.multi_layer_forward:
            preds, feat = self.forward_multi_layer(x)
        else:
            preds, feat = self.forward_single_layer(x)
        if self.mask_inv_test and self.training is False:
            preds = preds.view(preds.size(0)//2, 2, -1).mean(dim=1)
        return preds, feat

    def forward_single_layer(self, x):
        out = x
        for blk in self.blocks:
            out = blk(out)
        feat = out
        out = self.norm(out.mean(dim=1))
        if hasattr(self, "dropout"):
            out = self.dropout(out)
        out = self.head(out)
        return out, feat

    def forward_multi_layer(self, x):
        out = x
        out_list = []
        for blk in self.blocks:
            out = blk(out)
            out_list.append(out)
        feat = out
        out_full = torch.stack(out_list, dim=1).flatten(0, 1)
        out_full = self.norm(out_full.mean(dim=1))
        if hasattr(self, "dropout"):
            out_full = self.dropout(out_full)
        out_full = self.head(out_full)
        out_full = out_full.view(x.size(0), len(out_list), -1).mean(dim=1)
        return out_full, feat

