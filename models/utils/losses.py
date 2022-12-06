#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Losses. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import Registry
import math
import utils.misc as misc
import utils.distributed as du

from einops import rearrange, repeat
from dataset.utils.mixup import label_smoothing
SSL_LOSSES = Registry("SSL_Losses")

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, reduction='mean'):
        """
        Args:
            reduction: defined for compatibility with other losses.
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss

class FocalCrossEntropy(nn.Module):

    def __init__(self, reduction=None):
        """
        Args:
            reduction: defined for compatibility with other losses.
        """
        super(FocalCrossEntropy, self).__init__()
        self.gamma = 1.5
    
    def forward(self, x, target):
        pred = x.softmax(-1)
        loss = ((1-pred)**self.gamma) * (-torch.log(pred))
        loss = loss[torch.linspace(0, x.shape[0]-1, x.shape[0]).long(), target].mean()
        return loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    "soft_target": SoftTargetCrossEntropy,
    "focal_ce": FocalCrossEntropy
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

def calculate_loss(cfg, inputs, model_preds, logits, labels, cur_epoch):
    """
    Calculates loss according to cfg.
    For pre-training, losses are defined and registered in `SSL_LOSSES`.
    Different losses can be combined by specifying in the `cfg.PRETRAIN.LOSS` by
    connecting different loss names with `+`.
    
    For supervised training, this function supports cross entropy with mixup,
    label smoothing and the plain cross entropy.
    """
    loss_in_parts = {}
    weight = None
    if cfg.PRETRAIN.ENABLE or cfg.MM_RETRIEVAL.ENABLE:
        loss = 0
        loss_parts = cfg.PRETRAIN.LOSS.split('+')
        loss_weights = cfg.PRETRAIN.LOSS_WEIGHTS
        # sum up all loss items
        for loss_idx, loss_item in enumerate(loss_parts):
            loss_cur, weight = SSL_LOSSES.get("Loss_"+loss_item)(cfg, model_preds, logits, labels["self-supervised"], cur_epoch)
            if isinstance(loss_cur, dict):
                for k, v in loss_cur.items():
                    loss_in_parts[k] = v
                    if "debug" not in k and isinstance(v, torch.Tensor):
                        loss += loss_weights[loss_idx]*loss_in_parts[k]
            else:
                loss_in_parts[loss_item] = loss_cur
                loss += loss_weights[loss_idx]*loss_in_parts[loss_item]
        if hasattr(cfg, 'AUTO_AUG') and hasattr(cfg.AUTO_AUG, 'REPORT_IOU') and cfg.AUTO_AUG.REPORT_IOU:
            loss_in_parts['iou'] = model_preds['iou_gt'][:, 0].mean()
            loss_in_parts['dis'] = model_preds['iou_gt'][:, 1].mean()
            loss_in_parts['area1'] = model_preds['iou_gt'][:, 2].mean()
            loss_in_parts['area2'] = model_preds['iou_gt'][:, 3].mean()
            loss_in_parts['tiou'] = model_preds['iou_gt'][:, 4].mean()
            loss_in_parts['tdis'] = model_preds['iou_gt'][:, 5].mean()
    else:
        # Explicitly declare reduction to mean.
        loss_fun = get_loss_func(cfg.TRAIN.LOSS_FUNC)(reduction="mean")
        if isinstance(model_preds, dict):
            preds = model_preds['preds']
        else:
            preds = model_preds
        # Compute the loss.
        if "supervised_mixup" in labels.keys():
            if isinstance(labels["supervised_mixup"], dict):
                loss = 0
                for k, v in labels["supervised_mixup"].items():
                    loss_in_parts["loss_"+k] = loss_fun(preds[k], v)
                    loss += loss_in_parts["loss_"+k]
            else:
                loss = loss_fun(preds, labels["supervised_mixup"])
        else:
            if cfg.AUGMENTATION.LABEL_SMOOTHING > 0.0:
                labels_ = label_smoothing(cfg, labels["supervised"])
            else:
                labels_ = labels["supervised"]
            if isinstance(labels_, dict):
                loss = 0
                for k, v in labels_.items():
                    loss_in_parts["loss_"+k] = loss_fun(preds[k], v)
                    loss += loss_in_parts["loss_"+k]
            else:
                loss = loss_fun(preds, labels_)
        loss_in_parts["ce_loss"] = loss
    return loss, loss_in_parts, weight


def Loss_MaeCls(cfg, preds, logits, labels, cur_epoch=0):
    if 'preds_pixel' in preds:
        loss_mae = nn.MSELoss(reduction='none')(preds['preds_pixel'], preds['labels_pixel'])
    else:
        loss_mae = torch.Tensor([0]).to(preds['preds_pixel'].device())
    if 'supervised_mixup' in labels:
        loss_cls = SoftTargetCrossEntropy(reduction='none')(preds['preds_cls'][0], labels['supervised_mixup'])
    else:
        loss_cls = nn.CrossEntropyLoss(reduction='none')(preds['preds_cls'][0], labels['supervised_mixup'])
    loss_in_parts = {"loss_mae": loss_mae.mean(), "loss_cls": loss_cls.mean()}
    total_loss = loss_mae.mean() * cfg.TRAIN.MAE.MAE_LOSS_WEIGHT + loss_cls.mean()
    return total_loss, loss_in_parts


def construct_logits_with_gradient(cur_logits, all_logits, batch_size_per_gpu, samples):
    """
    Replaces the corresponding parts of the all-gathered logits with the ones generated
    by the local device with gradients.
    
    Args:
        cur_logits (Tensor): the logits generated by the model on the local device.
        all_logits (Tensor): the logits gathered from all the devices.
        batch_size_per_gpu (int): used for calculation of the index of the cur_logits 
            in the all_logits.
        samples (Tensor): for a batch size of N, there can be N*{samples} samples, as 
            the batch size is a indicator of the number of videos, and each video can
            generate multiple samples.
    Returns:
        logits (Tensor): all_logits with gradients.
    """

    num_nodes = du.get_world_size()
    rank = du.get_rank()
    num_samples_per_gpu = batch_size_per_gpu * samples
    if rank == 0:
        logits_post = all_logits[num_samples_per_gpu*(rank+1):, :]
        return torch.cat((cur_logits, logits_post), dim=0)
    elif rank == num_nodes-1:
        logits_prev = all_logits[:num_samples_per_gpu*rank, :]
        return torch.cat((logits_prev, cur_logits), dim=0)
    else:
        logits_prev = all_logits[:num_samples_per_gpu*rank, :]
        logits_post = all_logits[num_samples_per_gpu*(rank+1):, :]
        return torch.cat((logits_prev, cur_logits, logits_post), dim=0)
