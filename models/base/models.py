#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

import torch
import torch.nn as nn
from utils.registry import Registry
from einops import rearrange, repeat
from models.base.backbone import BACKBONE_REGISTRY
from models.base.base_blocks import HEAD_REGISTRY

MODEL_REGISTRY = Registry("Model")

class BaseVideoModel(nn.Module):
    """
    Standard video model.
    The model is divided into the backbone and the head, where the backbone
    extracts features and the head performs classification.

    The backbones can be defined in model/base/backbone.py or anywhere else
    as long as the backbone is registered by the BACKBONE_REGISTRY.
    The heads can be defined in model/module_zoo/heads/ or anywhere else
    as long as the head is registered by the HEAD_REGISTRY.

    The registries automatically finds the registered modules and construct 
    the base video model.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseVideoModel, self).__init__()
        self.cfg = cfg
        
        # the backbone is created according to meta-architectures 
        # defined in models/base/backbone.py
        self.backbone = BACKBONE_REGISTRY.get(cfg.VIDEO.BACKBONE.META_ARCH)(cfg=cfg)

        # the head is created according to the heads 
        # defined in models/module_zoo/heads
        self.head = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.NAME)(cfg=cfg)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        super(BaseVideoModel, self).train(mode)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)) and self.cfg.BN.FREEZE:
                module.train(False)
        return self


@MODEL_REGISTRY.register()
class MaeVitVideoModelWithClsVideoMAEBridgingClassifier(nn.Module):
    def __init__(self, cfg):
        super(MaeVitVideoModelWithClsVideoMAEBridgingClassifier, self).__init__()
        from .vit_videomae import PretrainVisionTransformerEncoder, PretrainVisionTransformerDecoder
        self.mean                = nn.Parameter(torch.Tensor(cfg.DATA.MEAN)[None, :, None, None, None], requires_grad=False)
        self.std                 = nn.Parameter(torch.Tensor(cfg.DATA.STD)[None, :, None, None, None], requires_grad=False)
        self.normlize_target     = cfg.TRAIN.MAE.NORMLIZE_TARGET
        self.patch_size          = cfg.VIDEO.BACKBONE.PATCH_SIZE
        self.tubelet_size        = cfg.VIDEO.BACKBONE.TUBELET_SIZE if hasattr(cfg.VIDEO.BACKBONE, 'TUBELET_SIZE') else 1
        self.mask_stride         = cfg.TRAIN.MAE.MASK_STRIDE if hasattr(cfg.TRAIN.MAE, 'MASK_STRIDE') else [1, 1, 1]
        self.input_size          = [cfg.DATA.NUM_INPUT_FRAMES, cfg.DATA.TRAIN_CROP_SIZE]
        self.patches_shape       = [self.input_size[0]//self.tubelet_size, self.input_size[1]//self.patch_size, self.input_size[1]//self.patch_size]
        self.mask_shape          = [(self.patches_shape[0]//self.mask_stride[0]), (self.patches_shape[1]//self.mask_stride[1]), (self.patches_shape[2]//self.mask_stride[2])]

        self.backbone             = PretrainVisionTransformerEncoder(cfg=cfg)
        self.decoder             = PretrainVisionTransformerDecoder(cfg=cfg)
        self.cls_head            = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.CLS_NAME)(cfg=cfg)
        self.mask_token          = nn.Parameter(torch.zeros(1, 1, cfg.VIDEO.HEAD.DECODER.NUM_FEATURES))
        self.encoder_to_decoder  = nn.Linear(cfg.VIDEO.BACKBONE.NUM_FEATURES, cfg.VIDEO.HEAD.DECODER.NUM_FEATURES, bias=False)
        self.encoder_to_cls_decoder  = nn.Linear(cfg.VIDEO.BACKBONE.NUM_FEATURES, cfg.VIDEO.HEAD.CLS_DECODER.NUM_FEATURES, bias=False) if hasattr(cfg.VIDEO.HEAD, 'CLS_DECODER') else lambda x:x.mean(dim=1)
        from .transformer import get_sinusoid_encoding_table
        self.pos_embed           = get_sinusoid_encoding_table(self.backbone.pos_embed.shape[1], cfg.VIDEO.HEAD.DECODER.NUM_FEATURES)
        self.pos_embed           = nn.Parameter(self.pos_embed, requires_grad=False)
        self.fc_norm_mean_pooling = cfg.VIDEO.BACKBONE.FC_NORM_MEAN_POOLING if hasattr(cfg.VIDEO.BACKBONE, "FC_NORM_MEAN_POOLING") else False
        self.masked_patches_type = cfg.VIDEO.BACKBONE.MASKED_PATCHES_TYPE if hasattr(cfg.VIDEO.BACKBONE, 'MASKED_PATCHES_TYPE') else 'none'
        self.pos_embed_for_cls_decoder = cfg.VIDEO.BACKBONE.POS_EMBED_FOR_CLS_DECODER if hasattr(cfg.VIDEO.BACKBONE, 'POS_EMBED_FOR_CLS_DECODER') else False
        self.mask_token_for_cls_decoder = cfg.VIDEO.BACKBONE.MASK_TOKEN_FOR_CLS_DECODER if hasattr(cfg.VIDEO.BACKBONE, 'MASK_TOKEN_FOR_CLS_DECODER') else False
        if self.pos_embed_for_cls_decoder or self.mask_token_for_cls_decoder:
            self.pos_embed_cls           = get_sinusoid_encoding_table(self.backbone.pos_embed.shape[1], cfg.VIDEO.HEAD.CLS_DECODER.NUM_FEATURES)
            self.pos_embed_cls           = nn.Parameter(self.pos_embed_cls, requires_grad=False)
        if self.mask_token_for_cls_decoder:
            self.mask_token_cls          = nn.Parameter(torch.zeros(1, 1, cfg.VIDEO.HEAD.CLS_DECODER.NUM_FEATURES))
        if self.fc_norm_mean_pooling:
            self.fc_norm = nn.LayerNorm(cfg.VIDEO.BACKBONE.NUM_FEATURES, eps=1e-6)

    def forward(self, x):
        x_data = x["video"]
        mask = x["mask"].bool()
        ####################################
        # new_mask = mask.reshape(10, 1, 8, 14, 14).float()
        # new_mask = new_mask.repeat_interleave(2, dim=2).repeat_interleave(16, dim=3).repeat_interleave(16, dim=4)
        # x_data_mask = x_data * new_mask.to(x_data.device)
        ####################################
        if self.training:
            with torch.no_grad():
                # calculate the predict label
                mean = self.mean.data.clone().detach()
                std = self.std.data.clone().detach()
                unnorm_frames = x_data * std + mean
                t, h, w = unnorm_frames.size(2) // self.tubelet_size, unnorm_frames.size(3) // self.patch_size, unnorm_frames.size(4) // self.patch_size
                if self.normlize_target:
                    images_squeeze = rearrange(unnorm_frames, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size)
                    images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                        ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    # we find that the mean is about 0.48 and standard deviation is about 0.08.
                    frames_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
                else:
                    frames_patch = rearrange(unnorm_frames, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size)
                frames_patch = rearrange(frames_patch, 'b (t s0 h s1 w s2) c -> b (t h w) (s0 s1 s2 c)', s0=self.mask_stride[0], 
                                                                                                        s1=self.mask_stride[1], 
                                                                                                        s2=self.mask_stride[2],
                                                                                                        t=t//self.mask_stride[0], 
                                                                                                        h=h//self.mask_stride[1], 
                                                                                                        w=w//self.mask_stride[2])
                B, _, C = frames_patch.shape
                labels = frames_patch[(~mask).flatten(1, 2)].reshape(B, -1, C)
        else:
            B = x_data.size(0)
            labels = None
        full_mask = mask.reshape(B, *self.mask_shape).repeat_interleave(self.mask_stride[0], dim=1).repeat_interleave(self.mask_stride[1], dim=2).repeat_interleave(self.mask_stride[2], dim=3)
        full_mask = full_mask.flatten(2)
        encoder_logits_backbone, patch_embed, x_vis_list = self.backbone(x_data, ~(full_mask.flatten(1)))
        b, t, p = full_mask.size()
        if self.training:
            encoder_logits = self.encoder_to_decoder(encoder_logits_backbone)
            c = encoder_logits.size(-1)
            full_mask = full_mask.flatten(1, 2)
            mask_token = self.mask_token.type_as(encoder_logits).repeat(b, t*p, 1)
            mask_token[full_mask, :] = encoder_logits.flatten(0, 1)
            logits_full = mask_token + self.pos_embed.detach().clone()
            pred_pixels = self.decoder(logits_full, -1)
            pred_pixels = rearrange(pred_pixels, 'b (t s0 h s1 w s2) c -> b (t h w) (s0 s1 s2 c)',s0=self.mask_stride[0], 
                                                                                                s1=self.mask_stride[1], 
                                                                                                s2=self.mask_stride[2],
                                                                                                t=t//self.mask_stride[0], 
                                                                                                h=h//self.mask_stride[1], 
                                                                                                w=w//self.mask_stride[2])
            pred_pixels = pred_pixels[(~mask).flatten(1, 2)].reshape(B, -1, C)
        else:
            pred_pixels = None
        preds_cls = self.cls_head(self.encoder_to_cls_decoder(encoder_logits_backbone))
        output = {"preds_pixel": pred_pixels, "labels_pixel":labels, "preds_cls": preds_cls}
        return output


@MODEL_REGISTRY.register()
class MaeVitVideoModelWithClsVideoMAEBridgingClassifierImageSupervised(MaeVitVideoModelWithClsVideoMAEBridgingClassifier):
    def __init__(self, cfg):
        super(MaeVitVideoModelWithClsVideoMAEBridgingClassifierImageSupervised, self).__init__(cfg)
        from .vit_imagemae_sup4vid import ImageVisionTransformerEncoderSupForVideo, ImageVisionTransformerDecoderSupForVideo
        self.backbone            = ImageVisionTransformerEncoderSupForVideo(cfg=cfg)
        self.decoder             = ImageVisionTransformerDecoderSupForVideo(cfg=cfg)

