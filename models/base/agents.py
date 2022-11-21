#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Maksed agents. """

import torch
import torch.nn as nn
import cv2, math
import random
from models.base.base_blocks import BaseHead
from models.base.base_blocks import HEAD_REGISTRY
import utils.misc as misc
import utils.distributed as du
from einops import rearrange

from utils.registry import Registry
AGENT_REGISTRY = Registry("Agent")
AGENT_NET_REGISTRY = Registry("AgentNet")


@AGENT_NET_REGISTRY.register()
class CellRunningMaskAgent(nn.Module):
    def __init__(self, cfg, patch_num, mask_num):
        super(CellRunningMaskAgent, self).__init__()
        self.patch_num = patch_num
        self.mask_num = mask_num
        self.predictor = nn.Linear(1024 + 256, self.patch_num)
        self.mask_shape = [cfg.DATA.NUM_INPUT_FRAMES // 2, 14, 14]
        self.mask_stride = [1, 2, 2]
        self.spatial_small_patch_num = (self.mask_shape[1] // self.mask_stride[1]) * (self.mask_shape[2] // self.mask_stride[2])
        self.test_mask = torch.zeros(self.mask_shape)
        self.test_mask = rearrange(self.test_mask, '(t s0) (h s1) (w s2) -> t (h w) (s0 s1 s2)', s0=self.mask_stride[0], 
                                                                                                     s1=self.mask_stride[1], 
                                                                                                     s2=self.mask_stride[2])
        mask_per_patch = self.mask_num // (self.test_mask.size(0) * self.test_mask.size(1))
        mask_list = [1 for i in range(mask_per_patch)] + [0 for i in range(self.test_mask.size(2)-mask_per_patch)]
        for t in range(self.test_mask.size(0)):
            offset = t % self.test_mask.size(-1)
            self.test_mask[t, :, :] = torch.Tensor(mask_list[-offset:] + mask_list[:-offset])[None, :]
        self.test_mask = rearrange(self.test_mask, 't (h w) (s0 s1 s2) -> (t s0) (h s1) (w s2)', s0=self.mask_stride[0], 
                                                                                                     s1=self.mask_stride[1], 
                                                                                                     s2=self.mask_stride[2],
                                                                                                     t=self.mask_shape[0]//self.mask_stride[0],
                                                                                                     h=self.mask_shape[1]//self.mask_stride[1],
                                                                                                     w=self.mask_shape[2]//self.mask_stride[2],)
        train_mask_list = []
        for i in range(self.mask_stride[1]*self.mask_stride[2]):
            train_mask = torch.zeros(self.mask_shape[0], self.mask_stride[1]*self.mask_stride[2])
            for t in range(train_mask.size(0)):
                offset = (t+i) % train_mask.size(-1)
                train_mask[t, :] = torch.Tensor(mask_list[-offset:] + mask_list[:-offset])
            train_mask_list.append(train_mask)
        self.train_mask = torch.stack(train_mask_list, dim=0)
        self.temporal_shuffle = cfg.AGENT.MASK_TEMPORAL_SHUFFLE if hasattr(cfg.AGENT, 'MASK_TEMPORAL_SHUFFLE') else False
        self.spatial_repeat = cfg.AGENT.MASK_SPATIAL_REPEAT if hasattr(cfg.AGENT, 'MASK_SPATIAL_REPEAT') else False
        self.test_temporal_shuffle = cfg.AGENT.TEST_MASK_TEMPORAL_SHUFFLE if hasattr(cfg.AGENT, 'TEST_MASK_TEMPORAL_SHUFFLE') else False

    def forward(self, x, mask_shape):
        if isinstance(x, dict):
            x = x['video']
        if self.training:
            if self.spatial_repeat:
                mask_index = torch.randint(self.train_mask.size(0), (x.size(0), 1), device=x.device)
                mask_index = mask_index.repeat(1, self.spatial_small_patch_num).flatten()
            else:
                mask_index = torch.randint(self.train_mask.size(0), (x.size(0), self.spatial_small_patch_num), device=x.device).flatten()
            selected_mask = self.train_mask.to(x.device)[mask_index, ...].view(x.size(0), self.spatial_small_patch_num, self.train_mask.size(1), self.train_mask.size(2))
            selected_mask = selected_mask.permute(0, 2, 1, 3)
            selected_mask = rearrange(selected_mask, 'b t (h w) (s0 s1 s2) -> b (t s0) (h s1) (w s2)', s0=self.mask_stride[0], 
                                                                                                     s1=self.mask_stride[1], 
                                                                                                     s2=self.mask_stride[2],
                                                                                                     t=self.mask_shape[0]//self.mask_stride[0],
                                                                                                     h=self.mask_shape[1]//self.mask_stride[1],
                                                                                                     w=self.mask_shape[2]//self.mask_stride[2],)
            if self.temporal_shuffle:
                temporal_seed = torch.rand(selected_mask.shape[:2], device=selected_mask.device)
                temporal_index = temporal_seed.argsort(dim=-1)
                selected_mask = torch.gather(selected_mask, index=temporal_index[:, :, None, None].expand_as(selected_mask), dim=1)
            selected_mask = selected_mask.flatten(1)
            seq_logits = torch.rand(selected_mask.size(0), selected_mask.size(1), device=x.device)
            values, indices = seq_logits.topk(self.mask_num, dim=1, largest=True, sorted=False)
            seq_logits = seq_logits[:, None, :].repeat(1, self.mask_num, 1)
            output = {"seq_logits": seq_logits.detach(), "indices": indices, "mask": 1.0 - selected_mask}
        else:
            selected_mask = self.test_mask.flatten()[None, ...].to(x.device).repeat(x.size(0), 1)
            if self.test_temporal_shuffle:
                selected_mask = selected_mask.view(x.size(0), mask_shape[0], -1)
                temporal_seed = torch.rand(selected_mask.shape[:2], device=selected_mask.device)
                temporal_index = temporal_seed.argsort(dim=-1)
                selected_mask = torch.gather(selected_mask, index=temporal_index[:, :, None].expand_as(selected_mask), dim=1)
                selected_mask = selected_mask.flatten(1)
            output = {"mask": 1.0 - selected_mask}
        return output


@AGENT_NET_REGISTRY.register()
class CellRunningMaskAgentMultiCell(nn.Module):
    def __init__(self, cfg, patch_num, mask_num):
        super(CellRunningMaskAgentMultiCell, self).__init__()
        self.patch_num = patch_num
        self.mask_num = mask_num
        self.predictor = nn.Linear(1024 + 256, self.patch_num)
        self.mask_shape = [cfg.DATA.NUM_INPUT_FRAMES // 2, 14, 14]
        self.mask_stride = cfg.AGENT.CELL_SHAPE
        self.spatial_small_patch_num = (self.mask_shape[1] // self.mask_stride[1]) * (self.mask_shape[2] // self.mask_stride[2])
        self.test_mask = torch.zeros(self.mask_shape)
        self.test_mask = rearrange(self.test_mask, '(t s0) (h s1) (w s2) -> t (h w) (s0 s1 s2)', s0=self.mask_stride[0], 
                                                                                                     s1=self.mask_stride[1], 
                                                                                                     s2=self.mask_stride[2])
        mask_per_patch = math.ceil(self.mask_num / (self.test_mask.size(0) * self.test_mask.size(1)))
        vis_per_patch = self.test_mask.size(2) - mask_per_patch
        if mask_per_patch <= vis_per_patch:
            stride = math.ceil(self.test_mask.size(2) / mask_per_patch)
            mask_list = [0 for i in range(self.test_mask.size(2))]
            for i in range(mask_per_patch):
                mask_list[i*stride] = 1
        else:
            stride = math.ceil(self.test_mask.size(2) / vis_per_patch)
            mask_list = [1 for i in range(self.test_mask.size(2))]
            for i in range(vis_per_patch):
                mask_list[i*stride] = 0

        assert sum(mask_list) == mask_per_patch
        for t in range(self.test_mask.size(0)):
            offset = t % self.test_mask.size(-1)
            self.test_mask[t, :, :] = torch.Tensor(mask_list[-offset:] + mask_list[:-offset])[None, :]
        self.test_mask = rearrange(self.test_mask, 't (h w) (s0 s1 s2) -> (t s0) (h s1) (w s2)', s0=self.mask_stride[0], 
                                                                                                     s1=self.mask_stride[1], 
                                                                                                     s2=self.mask_stride[2],
                                                                                                     t=self.mask_shape[0]//self.mask_stride[0],
                                                                                                     h=self.mask_shape[1]//self.mask_stride[1],
                                                                                                     w=self.mask_shape[2]//self.mask_stride[2],)
        train_mask_list = []
        for i in range(self.mask_stride[1]*self.mask_stride[2]):
            train_mask = torch.zeros(self.mask_shape[0], self.mask_stride[1]*self.mask_stride[2])
            for t in range(train_mask.size(0)):
                offset = (t+i) % train_mask.size(-1)
                train_mask[t, :] = torch.Tensor(mask_list[-offset:] + mask_list[:-offset])
            train_mask_list.append(train_mask)
        self.train_mask = torch.stack(train_mask_list, dim=0)
        self.temporal_shuffle = cfg.AGENT.MASK_TEMPORAL_SHUFFLE if hasattr(cfg.AGENT, 'MASK_TEMPORAL_SHUFFLE') else False
        self.spatial_repeat = cfg.AGENT.MASK_SPATIAL_REPEAT if hasattr(cfg.AGENT, 'MASK_SPATIAL_REPEAT') else False
        self.test_temporal_shuffle = cfg.AGENT.TEST_MASK_TEMPORAL_SHUFFLE if hasattr(cfg.AGENT, 'TEST_MASK_TEMPORAL_SHUFFLE') else False

    def forward(self, x, mask_shape):
        if isinstance(x, dict):
            x = x['video']
        if self.training:
            if self.spatial_repeat:
                mask_index = torch.randint(self.train_mask.size(0), (x.size(0), 1), device=x.device)
                mask_index = mask_index.repeat(1, self.spatial_small_patch_num).flatten()
            else:
                mask_index = torch.randint(self.train_mask.size(0), (x.size(0), self.spatial_small_patch_num), device=x.device).flatten()
            selected_mask = self.train_mask.to(x.device)[mask_index, ...].view(x.size(0), self.spatial_small_patch_num, self.train_mask.size(1), self.train_mask.size(2))
            selected_mask = selected_mask.permute(0, 2, 1, 3)
            selected_mask = rearrange(selected_mask, 'b t (h w) (s0 s1 s2) -> b (t s0) (h s1) (w s2)', s0=self.mask_stride[0], 
                                                                                                     s1=self.mask_stride[1], 
                                                                                                     s2=self.mask_stride[2],
                                                                                                     t=self.mask_shape[0]//self.mask_stride[0],
                                                                                                     h=self.mask_shape[1]//self.mask_stride[1],
                                                                                                     w=self.mask_shape[2]//self.mask_stride[2],)
            if self.temporal_shuffle:
                temporal_seed = torch.rand(selected_mask.shape[:2], device=selected_mask.device)
                temporal_index = temporal_seed.argsort(dim=-1)
                selected_mask = torch.gather(selected_mask, index=temporal_index[:, :, None, None].expand_as(selected_mask), dim=1)
            selected_mask = selected_mask.flatten(1)
            seq_logits = torch.rand(selected_mask.size(0), selected_mask.size(1), device=x.device)
            values, indices = seq_logits.topk(self.mask_num, dim=1, largest=True, sorted=False)
            seq_logits = seq_logits[:, None, :].repeat(1, self.mask_num, 1)
            output = {"seq_logits": seq_logits.detach(), "indices": indices, "mask": 1.0 - selected_mask}
        else:
            selected_mask = self.test_mask.flatten()[None, ...].to(x.device).repeat(x.size(0), 1)
            if self.test_temporal_shuffle:
                selected_mask = selected_mask.view(x.size(0), mask_shape[0], -1)
                temporal_seed = torch.rand(selected_mask.shape[:2], device=selected_mask.device)
                temporal_index = temporal_seed.argsort(dim=-1)
                selected_mask = torch.gather(selected_mask, index=temporal_index[:, :, None].expand_as(selected_mask), dim=1)
                selected_mask = selected_mask.flatten(1)
            output = {"mask": 1.0 - selected_mask}
        return output


@AGENT_REGISTRY.register()
class MaskAgentSeq(nn.Module):
    def __init__(self, cfg):
        super(MaskAgentSeq, self).__init__()
        self.patch_size = cfg.VIDEO.BACKBONE.PATCH_SIZE
        self.tublet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE if hasattr(cfg.VIDEO.BACKBONE, 'TUBELET_SIZE') else 1
        self.input_size = [cfg.DATA.NUM_INPUT_FRAMES, cfg.DATA.TRAIN_CROP_SIZE]
        self.mask_ratio = cfg.TRAIN.MAE.MASK_RATIO
        self.mask_stride = cfg.TRAIN.MAE.MASK_STRIDE if hasattr(cfg.TRAIN.MAE, 'MASK_STRIDE') else [1, 1, 1]
        self.patches_shape = [self.input_size[0]//self.tublet_size, self.input_size[1]//self.patch_size, self.input_size[1]//self.patch_size]
        self.mask_shape = [(self.patches_shape[0]//self.mask_stride[0]), (self.patches_shape[1]//self.mask_stride[1]), (self.patches_shape[2]//self.mask_stride[2])]
        self.num_patches_per_frame = (self.patches_shape[1]//self.mask_stride[1]) * (self.patches_shape[2]//self.mask_stride[2])
        self.num_patches_per_clip = (self.patches_shape[0]//self.mask_stride[0]) * (self.patches_shape[1]//self.mask_stride[1]) * (self.patches_shape[2]//self.mask_stride[2])
        self.num_mask_per_clip = int(self.mask_ratio * self.num_patches_per_clip)
        self.num_mask_per_frame = int(self.mask_ratio * self.num_patches_per_frame)
        self.agent_net_name = cfg.AGENT.AGENT_NET_NAME if hasattr(cfg.AGENT, 'AGENT_NET_NAME') else "EMAAgent"
        self.network = AGENT_NET_REGISTRY.get(self.agent_net_name)(cfg, self.num_patches_per_clip, self.num_mask_per_clip)

    def forward(self, x):
        rl_pred = self.network(x, self.mask_shape)
        rl_pred['mask'] = rl_pred['mask'].reshape(rl_pred['mask'].size(0), self.mask_shape[0], -1)
        return rl_pred
