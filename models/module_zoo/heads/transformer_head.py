#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Transformer heads. """

import torch
import torch.nn as nn

from models.base.base_blocks import BaseHead
from models.base.base_blocks import HEAD_REGISTRY, BRANCH_REGISTRY

from collections import OrderedDict
from models.utils.init_helper import lecun_normal_, trunc_normal_, _init_transformer_weights

@HEAD_REGISTRY.register()
class TransformerHead(BaseHead):
    """
    Construct head for video vision transformers.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        if hasattr(cfg.PRETRAIN, "MAE"):
            self.tem_agg_type = cfg.PRETRAIN.MAE.TEM_AGG_TYPE
        else:
            self.tem_agg_type = 'sum'
        super(TransformerHead, self).__init__(cfg)
        if hasattr(cfg.VIDEO.HEAD, "INIT_STD"):
            trunc_normal_(self.linear.weight, std=cfg.VIDEO.HEAD.INIT_STD)
        else:
            self.apply(_init_transformer_weights)
        if hasattr(cfg.VIDEO.HEAD, "INIT_SCALE"):
            self.linear.weight.data.mul_(cfg.VIDEO.HEAD.INIT_SCALE)
            self.linear.bias.data.mul_(cfg.VIDEO.HEAD.INIT_SCALE)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        if self.cfg.VIDEO.HEAD.PRE_LOGITS:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim, dim)),
                ('act', nn.Tanh())
            ]))
        if self.cfg.VIDEO.HEAD.PRE_BN:
            self.pre_bn = nn.BatchNorm1d(dim, affine=False)
        if self.tem_agg_type == "concat":
            self.linear = nn.Linear(dim*2, num_classes)
        else:
            self.linear = nn.Linear(dim, num_classes)

        if dropout_rate > 0.0: 
            self.dropout = nn.Dropout(dropout_rate)

        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_func == "identity":
            self.activation = nn.Identity()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(activation_func)
            )
    
    def forward(self, x):
        if isinstance(x, dict):
            feat = x["feature"]
        if hasattr(self, "dropout"):
            out = self.dropout(feat)
        else:
            out = feat
        if hasattr(self, "pre_logits"):
            out = self.pre_logits(out)
        if hasattr(self, "pre_bn"):
            out = self.pre_bn(out[:, :, None])[:, :, 0]
        out = self.linear(out)

        if not self.training:
            out = self.activation(out)
        return out, x


@HEAD_REGISTRY.register()
class TransformerHeadNoAct(BaseHead):
    """
    Construct head for video vision transformers.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        if hasattr(cfg.PRETRAIN, "MAE"):
            self.tem_agg_type = cfg.PRETRAIN.MAE.TEM_AGG_TYPE
        else:
            self.tem_agg_type = 'sum'
        super(TransformerHeadNoAct, self).__init__(cfg)
        if hasattr(cfg.VIDEO.HEAD, "INIT_STD"):
            trunc_normal_(self.linear.weight, std=cfg.VIDEO.HEAD.INIT_STD)
        else:
            self.apply(_init_transformer_weights)
        if hasattr(cfg.VIDEO.HEAD, "INIT_SCALE"):
            self.linear.weight.data.mul_(cfg.VIDEO.HEAD.INIT_SCALE)
            self.linear.bias.data.mul_(cfg.VIDEO.HEAD.INIT_SCALE)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        if self.cfg.VIDEO.HEAD.PRE_LOGITS:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim, dim)),
                ('act', nn.Tanh())
            ]))
        if self.cfg.VIDEO.HEAD.PRE_BN:
            self.pre_bn = nn.BatchNorm1d(dim, affine=False)
        if self.tem_agg_type == "concat":
            self.linear = nn.Linear(dim*2, num_classes)
        else:
            self.linear = nn.Linear(dim, num_classes)

        if dropout_rate > 0.0: 
            self.dropout = nn.Dropout(dropout_rate)

        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_func == "identity":
            self.activation = nn.Identity()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(activation_func)
            )
    
    def forward(self, x):
        if isinstance(x, dict):
            feat = x["feature"]
        else:
            feat = x
        if hasattr(self, "dropout"):
            out = self.dropout(feat)
        else:
            out = feat
        if hasattr(self, "pre_logits"):
            out = self.pre_logits(out)
        if hasattr(self, "pre_bn"):
            out = self.pre_bn(out[:, :, None])[:, :, 0]
        out = self.linear(out)

        # if not self.training:
        #     out = self.activation(out)
        return out, x


@HEAD_REGISTRY.register()
class TransformerDecoderPredHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg.VIDEO.BACKBONE.PATCH_SIZE
        self.tublet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE if hasattr(cfg.VIDEO.BACKBONE, 'TUBELET_SIZE') else 1
        self.output_dim = cfg.VIDEO.HEAD.DECODER.NUM_FEATURES
        self.num_features = cfg.VIDEO.HEAD.DECODER.NUM_FEATURES
        self.num_heads = cfg.VIDEO.HEAD.DECODER.NUM_HEADS
        drop_path       = cfg.VIDEO.HEAD.DECODER.DROP_PATH
        depth       = cfg.VIDEO.HEAD.DECODER.DEPTH
        num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            BRANCH_REGISTRY.get(cfg.VIDEO.HEAD.DECODER.NAME)(cfg, num_heads_override=self.num_heads, dim_override=self.num_features, drop_path_rate=dpr[i])
            for i in range(depth)])
        self.encoder_to_decoder = nn.Linear(cfg.VIDEO.BACKBONE.NUM_FEATURES, self.num_features)
        self.norm =  nn.LayerNorm(self.num_features, eps=1e-6)
        self.head = nn.Linear(self.num_features, num_classes)
        self.apply(self._init_weights)
        self.activation = nn.Softmax(dim=-1)

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
        if isinstance(x, dict):
            feat = x["feature"]
        else:
            feat = x
        feat = self.encoder_to_decoder(feat)
        for blk in self.blocks:
            feat = blk(feat)
        feat = self.norm(feat).mean(dim=1)
        out = self.head(feat) # [B, N, 3*16^2]
        if not self.training:
            out = self.activation(out)
        return out, x


@HEAD_REGISTRY.register()
class TransformerDecoderWithPixel(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg.VIDEO.BACKBONE.PATCH_SIZE
        self.tublet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE if hasattr(cfg.VIDEO.BACKBONE, 'TUBELET_SIZE') else 1
        self.output_dim = cfg.VIDEO.HEAD.DECODER.NUM_FEATURES
        self.num_features = cfg.VIDEO.HEAD.DECODER.NUM_FEATURES
        self.num_heads = cfg.VIDEO.HEAD.DECODER.NUM_HEADS
        self.pred_pixel_num = 3 * self.tublet_size * self.patch_size ** 2
        drop_path       = cfg.VIDEO.HEAD.DECODER.DROP_PATH
        depth       = cfg.VIDEO.HEAD.DECODER.DEPTH
        num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            BRANCH_REGISTRY.get(cfg.VIDEO.HEAD.DECODER.NAME)(cfg, num_heads_override=self.num_heads, dim_override=self.num_features, drop_path_rate=dpr[i])
            for i in range(depth)])
        self.encoder_to_decoder = nn.Linear(cfg.VIDEO.BACKBONE.NUM_FEATURES, self.num_features)
        self.norm               = nn.LayerNorm(self.num_features, eps=1e-6)
        self.head               = nn.Linear(cfg.VIDEO.BACKBONE.NUM_FEATURES, num_classes)
        self.pixel_pred_head    = nn.Linear(self.num_features, self.pred_pixel_num)
        self.apply(self._init_weights)
        self.activation         = nn.Softmax(dim=-1)

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
        if isinstance(x, dict):
            feat = x["feature"]
        else:
            feat = x
        feat = self.encoder_to_decoder(feat)
        for blk in self.blocks:
            feat = blk(feat)
        feat = self.norm(feat)
        pixels = self.pixel_pred_head(feat)
        out = self.head(x["cls_feature"])
        if not self.training:
            out = self.activation(out)
        return {'preds': out, 'pixels': pixels}, x


@HEAD_REGISTRY.register()
class TransformerHeadx2(BaseHead):
    """
    The Transformer head for EPIC-KITCHENS dataset.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(TransformerHeadx2, self).__init__(cfg)
        self.apply(_init_transformer_weights)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        if self.cfg.VIDEO.HEAD.PRE_LOGITS:
            self.pre_logits1 = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim, dim)),
                ('act', nn.Tanh())
            ]))
            self.pre_logits2 = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim, dim)),
                ('act', nn.Tanh())
            ]))
        self.linear1 = nn.Linear(dim, num_classes[0], bias=True)
        self.linear2 = nn.Linear(dim, num_classes[1], bias=True)

        if dropout_rate > 0.0: 
            self.dropout = nn.Dropout(dropout_rate)

        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_func == "identity":
            self.activation = nn.Identity()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(activation_func)
            )
    
    def forward(self, x):
        if hasattr(self, "dropout"):
            out1 = self.dropout(x)
            out2 = self.dropout(x)
        else:
            out1 = x
            out2 = x

        if hasattr(self, "pre_logits1"):
            out1 = self.pre_logits1(out1)
            out2 = self.pre_logits2(out2)

        out1 = self.linear1(out1)
        out2 = self.linear2(out2)

        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        return {"verb_class": out1, "noun_class": out2}, x


@HEAD_REGISTRY.register()
class TransformerHeadPool(BaseHead):
    """
    Construct head for video vision transformers.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(TransformerHeadPool, self).__init__(cfg)
        self.apply(_init_transformer_weights)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        if self.cfg.VIDEO.HEAD.PRE_LOGITS:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim, dim)),
                ('act', nn.Tanh())
            ]))
        
        self.linear = nn.Linear(dim, num_classes)
        self.pooler = nn.AdaptiveAvgPool3d(1)
        if dropout_rate > 0.0: 
            self.dropout = nn.Dropout(dropout_rate)

        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_func == "identity":
            self.activation = nn.Identity()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(activation_func)
            )
    
    def forward(self, x):
        x = self.pooler(x).view(x.size(0), -1)
        if hasattr(self, "dropout"):
            out = self.dropout(x)
        else:
            out = x
        if hasattr(self, "pre_logits"):
            out = self.pre_logits(out)
        out = self.linear(out)

        if not self.training:
            out = self.activation(out)
        return out, x
