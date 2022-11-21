#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# modified from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/optimizer.py

""" Optimizer. """

import torch

import utils.logging as logging
import utils.misc as misc
import models.utils.lr_policy as lr_policy
from models.utils.lars import LARS
import math

logger = logging.get_logger(__name__)


def construct_optimizer(model, cfg):
    """
    Construct an optimizer. 
    Supported optimizers include:
        SGD:    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
        ADAM:   Diederik P.Kingma, and Jimmy Ba. "Adam: A Method for Stochastic Optimization."
        ADAMW:  Ilya Loshchilov, and Frank Hutter. "Decoupled Weight Decay Regularization."
        LARS:   Yang You, Igor Gitman, and Boris Ginsburg. "Large Batch Training of Convolutional Networks."

    Args:
        model (model): model for optimization.
        cfg (Config): Config object that includes hyper-parameters for the optimizers. 
    """
    if hasattr(cfg.TRAIN, "ONLY_PATCH_SELECTOR") and cfg.TRAIN.ONLY_PATCH_SELECTOR:
        params = []
        for name, p in model.named_parameters():
            if "patch_selector" in name:
                params.append(p)
        optim_params = [{"params": params, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY}]
    elif cfg.TRAIN.ONLY_LINEAR:
        # only include linear layers
        params = []
        for name, p in model.named_parameters():
            if "head" in name:
                params.append(p)
        optim_params = [{"params": params, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY}]
    else:
        custom_parameters = []
        custom_bias_parameters = []
        custom_bn_parameters = []
        bn_parameters = []              # Batchnorm parameters.
        head_parameters = []            # Head parameters.
        head_bias_parameters = []       # Head bias parameters.
        non_bn_parameters = []          # Non-batchnorm parameters.
        non_bn_bias_parameters = []     # Non-batchnorm bias parameters.
        no_weight_decay_parameters = [] # No weight decay parameters.
        no_weight_decay_parameters_names = []
        num_skipped_param = 0
        audio_parameters = []
        audio_bn_paramesters = []
        mask_generator_weight_parameters = []
        mask_generator_bias_parameters = []
        paramcrop_parameters = []
        patch_selector_parameters = {'weight': [], 'bias': []}
        layer_decay_lr_parameters = {'weight': {}, 'bias': {}}
        if cfg.OPTIMIZER.LAYER_LRDECAY.ENABLE:
            try:
                num_layers_s, num_layers_t = model.module.backbone.get_num_layers()
            except:
                num_layers_s, num_layers_t = model.backbone.get_num_layers()
            num_layers = num_layers_s + num_layers_t
            assigner = LayerDecayValueAssigner(list(cfg.OPTIMIZER.LAYER_LRDECAY.DECAY_WEIGHT ** (num_layers + 1 - i) for i in range(num_layers + 2)), cfg.MODEL.NAME, cfg.VIDEO.BACKBONE.DEPTH)

        for name, p in model.named_parameters():
            if hasattr(cfg.TRAIN, "FIXED_WEIGHTS") and (
                name.split('.')[1] in cfg.TRAIN.FIXED_WEIGHTS or 
                name.split('.')[2] in cfg.TRAIN.FIXED_WEIGHTS):
                # fixing weights to a certain extent
                logger.info("Fixed weight: {}".format(name))
                num_skipped_param += 1
                continue
            if "patch_selector" in name:
                if 'bias' in name:
                    patch_selector_parameters['bias'].append(p)
                else:
                    patch_selector_parameters['weight'].append(p)
            elif "paramcrop" in name:
                paramcrop_parameters.append(p)
            elif "rf" in name or "avgpool_norm" in name:
                if "bn" in name or "norm" in name or "ln" in name:
                    custom_bn_parameters.append(p)
                elif "bias" in name:
                    custom_bias_parameters.append(p)
                else:
                    custom_parameters.append(p)
            elif "audio_a" in name or "audio_b" in name or "audio_c" in name or "audio_short_cut" in name or "audio_initial" in name:
                if "bn" in name:
                    audio_bn_paramesters.append(p)
                else:
                    audio_parameters.append(p)
            elif "embd" in name or "cls_token" in name or "pos_embed" in name:
                no_weight_decay_parameters_names.append(name)
                no_weight_decay_parameters.append(p)
            elif "mask_generator" in name and 'weight' in name:
                mask_generator_weight_parameters.append(p)
            elif "mask_generator" in name and 'bias' in name:
                mask_generator_bias_parameters.append(p)
            elif "bn" in name or "norm" in name:
                bn_parameters.append(p)
            elif "head" in name:
                if "bias" in name:
                    head_bias_parameters.append(p)
                else:
                    head_parameters.append(p)
            else:
                if cfg.OPTIMIZER.LAYER_LRDECAY.ENABLE:
                    layer_id = assigner.get_layer_id(name, num_layers_s)
                    lr_scale = assigner.get_scale(layer_id)
                    print(layer_id, lr_scale, name)
                    if "bias" in name:
                        if layer_id not in layer_decay_lr_parameters['bias']:
                            layer_decay_lr_parameters['bias'][layer_id] = {"params": [], "weight_decay": 0, "lr_mult": cfg.OPTIMIZER.BIAS_LR_MULT}
                        layer_decay_lr_parameters['bias'][layer_id]["params"].append(p)
                    else:
                        if layer_id not in layer_decay_lr_parameters['weight']:
                            layer_decay_lr_parameters['weight'][layer_id] = {"params": [], "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY, "lr_mult":lr_scale}
                        layer_decay_lr_parameters['weight'][layer_id]["params"].append(p)
                else:
                    if "bias" in name:
                        non_bn_bias_parameters.append(p)
                    else:
                        non_bn_parameters.append(p)
        optim_params = [
            {"params": custom_parameters, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY, "lr_mult": cfg.OPTIMIZER.CUSTOM_LRMULT if (cfg.TRAIN.LR_REDUCE and cfg.TRAIN.FINE_TUNE) else 1},
            {"params": custom_bias_parameters, "weight_decay": 0.0, "lr_mult": cfg.OPTIMIZER.CUSTOM_LRMULT * cfg.OPTIMIZER.BIAS_LR_MULT if (cfg.TRAIN.LR_REDUCE and cfg.TRAIN.FINE_TUNE) else cfg.OPTIMIZER.BIAS_LR_MULT},
            # normal params
            {"params": non_bn_parameters, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY, "lr_mult": 1},
            {"params": non_bn_bias_parameters, "weight_decay": 0.0, "lr_mult": cfg.OPTIMIZER.BIAS_LR_MULT, "lars_exclude": cfg.OPTIMIZER.BIAS_LARS_EXCLUDE if hasattr(cfg.OPTIMIZER, "BIAS_LARS_EXCLUDE") else False},
            # head params
            {"params": head_parameters, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY, "lr_mult": cfg.OPTIMIZER.HEAD_LRMULT if (cfg.TRAIN.LR_REDUCE and cfg.TRAIN.FINE_TUNE) else 1},
            {"params": head_bias_parameters, "weight_decay": 0.0, "lr_mult": cfg.OPTIMIZER.HEAD_LRMULT * cfg.OPTIMIZER.BIAS_LR_MULT if (cfg.TRAIN.LR_REDUCE and cfg.TRAIN.FINE_TUNE) else cfg.OPTIMIZER.BIAS_LR_MULT, "lars_exclude": cfg.OPTIMIZER.BIAS_LARS_EXCLUDE if hasattr(cfg.OPTIMIZER, "BIAS_LARS_EXCLUDE") else False },
            # no weight decay params
            {"params": no_weight_decay_parameters, "weight_decay": 0.0, "lr_mult": 1},
            # audio params
            {"params": audio_parameters, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY, "lr_mult": cfg.TRAIN.AUDIO_REDUCE_RATE}
        ]
        if len(paramcrop_parameters) > 0:
            optim_params.append({"params": paramcrop_parameters, "weight_decay": cfg.AUTO_AUG.WEIGHT_DECAY, "lr_mult": cfg.AUTO_AUG.LR_MULT, "grad_dir": cfg.AUTO_AUG.GRAD_DIRECTION})
        if len(patch_selector_parameters) > 0:
            if hasattr(cfg, "PATCH_SELECTOR_OPTIMIZER"):
                optim_params.append({"params": patch_selector_parameters['weight'], "weight_decay": cfg.PATCH_SELECTOR_OPTIMIZER.WEIGHT_DECAY, "lr_mult": cfg.PATCH_SELECTOR_OPTIMIZER.LR_MULT})
                optim_params.append({"params": patch_selector_parameters['bias'], "weight_decay": 0, "lr_mult": 2 * cfg.PATCH_SELECTOR_OPTIMIZER.LR_MULT if cfg.OPTIMIZER.BIAS_DOUBLE else cfg.PATCH_SELECTOR_OPTIMIZER.LR_MULT})
            else:
                optim_params.append({"params": patch_selector_parameters['weight'], "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY, "lr_mult": 1})
                optim_params.append({"params": patch_selector_parameters['bias'], "weight_decay": 0, "lr_mult": cfg.OPTIMIZER.BIAS_LR_MULT})
        layer_reduce_num = 0
        if cfg.OPTIMIZER.LAYER_LRDECAY.ENABLE:
            for layer_id, p_dict in layer_decay_lr_parameters['weight'].items():
                optim_params.append(p_dict)
                layer_reduce_num += len(p_dict['params'])
            for layer_id, p_dict in layer_decay_lr_parameters['bias'].items():
                optim_params.append(p_dict)
                layer_reduce_num += len(p_dict['params'])
        if not cfg.BN.WB_LOCK:
            optim_params = [
                {"params": bn_parameters, "weight_decay": cfg.BN.WEIGHT_DECAY, "lr_mult": 1, "lars_exclude": cfg.OPTIMIZER.BN_LARS_EXCLUDE if hasattr(cfg.OPTIMIZER, "BN_LARS_EXCLUDE") else False},
                {"params": custom_bn_parameters, "weight_decay": cfg.BN.WEIGHT_DECAY, "lr_mult": cfg.OPTIMIZER.CUSTOM_LRMULT if (cfg.TRAIN.LR_REDUCE and cfg.TRAIN.FINE_TUNE) else 1},
                {"params": audio_bn_paramesters, "weight_decay": cfg.BN.WEIGHT_DECAY, "lr_mult": cfg.TRAIN.AUDIO_REDUCE_RATE}
            ] + optim_params
        else:
            logger.info("Model bn/ln locked (not optimized).")

        if len(mask_generator_weight_parameters) > 0:
            optim_params.append({"params": mask_generator_weight_parameters, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY, "lr_mult": 1, "grad_dir": cfg.PRETRAIN.MAE.GRAD_DIR})
            optim_params.append({"params": mask_generator_bias_parameters, "weight_decay": 0.0, "lr_mult": 10 , "lars_exclude": cfg.OPTIMIZER.BIAS_LARS_EXCLUDE if hasattr(cfg.OPTIMIZER, "BIAS_LARS_EXCLUDE") else False, "grad_dir": cfg.PRETRAIN.MAE.GRAD_DIR})

        # Check all parameters will be passed into optimizer.
        assert len(list(model.parameters())) == len(custom_parameters) + \
            len(custom_bias_parameters) + \
            len(custom_bn_parameters) + \
            len(non_bn_parameters) + \
            len(non_bn_bias_parameters) + \
            len(bn_parameters) + \
            len(head_parameters) + \
            len(head_bias_parameters) + \
            len(no_weight_decay_parameters) + \
            len(audio_parameters) + \
            len(audio_bn_paramesters) + \
            len(mask_generator_weight_parameters) + \
            len(mask_generator_bias_parameters) + \
            len(paramcrop_parameters) + \
            len(patch_selector_parameters['weight']) + \
            len(patch_selector_parameters['bias']) + \
            layer_reduce_num + \
            num_skipped_param, "parameter size does not match: {} + {} != {}".format(len(non_bn_parameters), len(bn_parameters), len(list(model.parameters())))

        logger.info(f"Optimized parameters constructed. Parameters without weight decay: {no_weight_decay_parameters_names}")

    if cfg.OPTIMIZER.OPTIM_METHOD == "sgd":
        if cfg.OPTIMIZER.ADJUST_LR:
            # adjust learning rate for contrastive learning
            # the learning rate calculation is according to SimCLR
            num_clips_per_video = cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO if cfg.PRETRAIN.ENABLE else 1
            cfg.OPTIMIZER.BASE_LR = cfg.OPTIMIZER.BASE_LR*misc.get_num_gpus(cfg)*cfg.TRAIN.BATCH_SIZE*num_clips_per_video/256.
        return torch.optim.SGD(
            optim_params,
            lr=cfg.OPTIMIZER.BASE_LR,
            momentum=cfg.OPTIMIZER.MOMENTUM,
            weight_decay=float(cfg.OPTIMIZER.WEIGHT_DECAY),
            dampening=cfg.OPTIMIZER.DAMPENING,
            nesterov=cfg.OPTIMIZER.NESTEROV,
        )
    elif cfg.OPTIMIZER.OPTIM_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.OPTIMIZER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER.OPTIM_METHOD == "adamw":
        # if len(mask_generator_weight_parameters) > 0:
        #     from .adamw import AdamW
        #     return AdamW(
        #             optim_params,
        #             lr=cfg.OPTIMIZER.BASE_LR,
        #             betas=tuple(cfg.OPTIMIZER.BETAS) if hasattr(cfg.OPTIMIZER, 'BETAS') else (0.9, 0.999),
        #             weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        #         )
        # else:
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.OPTIMIZER.BASE_LR,
            betas=tuple(cfg.OPTIMIZER.BETAS) if hasattr(cfg.OPTIMIZER, 'BETAS') else (0.9, 0.999),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER.OPTIM_METHOD == "lars":
        if cfg.OPTIMIZER.ADJUST_LR:
            # adjust learning rate for contrastive learning
            # the learning rate calculation is according to SimCLR
            num_clips_per_video = cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO if cfg.PRETRAIN.ENABLE else 1
            batch_size = misc.get_num_gpus(cfg)*cfg.TRAIN.BATCH_SIZE*num_clips_per_video
            if hasattr(cfg.OPTIMIZER, 'LR_SCALE_TYPE') and cfg.OPTIMIZER.LR_SCALE_TYPE == 'squareroot':
                cfg.OPTIMIZER.BASE_LR = cfg.OPTIMIZER.BASE_LR*math.sqrt(batch_size)
            else:
                cfg.OPTIMIZER.BASE_LR = cfg.OPTIMIZER.BASE_LR*batch_size/256.
        return LARS(
            optim_params,
            lr=cfg.OPTIMIZER.BASE_LR,
            momentum=cfg.OPTIMIZER.MOMENTUM,
            weight_decay=float(cfg.OPTIMIZER.WEIGHT_DECAY),
            dampening=cfg.OPTIMIZER.DAMPENING,
            nesterov=cfg.OPTIMIZER.NESTEROV,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.OPTIMIZER.OPTIM_METHOD)
        )

def construct_optimizer_agent(model, cfg):
    bias_parameters = []
    weight_parameters = []
    for name, p in model.named_parameters():
        if "bias" in name:
            bias_parameters.append(p)
        else:
            weight_parameters.append(p)
    optim_params = [
            {"params": weight_parameters, "weight_decay": cfg.AGENT.OPTIMIZER.WEIGHT_DECAY, "lr_mult": 1},
            {"params": bias_parameters, "weight_decay": 0.0, "lr_mult": cfg.AGENT.OPTIMIZER.BIAS_LR_MULT}]
    if cfg.AGENT.OPTIMIZER.OPTIM_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.AGENT.OPTIMIZER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.AGENT.OPTIMIZER.WEIGHT_DECAY,
        )
    elif cfg.AGENT.OPTIMIZER.OPTIM_METHOD == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.AGENT.OPTIMIZER.BASE_LR,
            betas=tuple(cfg.AGENT.OPTIMIZER.BETAS) if hasattr(cfg.AGENT.OPTIMIZER, 'BETAS') else (0.9, 0.999),
            weight_decay=cfg.AGENT.OPTIMIZER.WEIGHT_DECAY,
        )

def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cur_epoch (float): current poch id.
        cfg (Config): global config object, including the settings on 
            warm-up epochs, base lr, etc.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def get_epoch_agent_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cur_epoch (float): current poch id.
        cfg (Config): global config object, including the settings on 
            warm-up epochs, base lr, etc.
    """
    return lr_policy.get_agent_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_idx, param_group in enumerate(optimizer.param_groups):
        if "lr_mult" in param_group.keys():
            # reduces the lr by a factor of 10 if specified for lr reduction
            param_group["lr"] = new_lr * param_group["lr_mult"]
        else:
            param_group["lr"] = new_lr


def get_num_layer_for_vit(var_name, num_max_layer, num_layers_s):
    if var_name in ("cls_token", "mask_token", "pos_embd"):
        return 0
    elif var_name.startswith("backbone.stem.") or var_name.startswith("backbone.patch_embed."):
        return 0
    elif var_name.startswith("backbone.layers.") or var_name.startswith("backbone.blocks."):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    elif var_name.startswith("backbone.layers_temporal."):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1 + num_layers_s
    elif var_name.startswith("backbone.conv"):
        layer_id = int(var_name.split('.')[1][-1:])
        return layer_id - 1
    else:
        return num_max_layer - 1


def get_num_layer_for_convnext(var_name, depth):
    """
    Divide [3, 3, 27, 3] layers into 12 groups; each group is three 
    consecutive blocks, including possible neighboring downsample layers;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """
    num_max_layer = 12
    div_val = depth[2] // 9
    if var_name.startswith("backbone.downsample_layers"):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1 or stage_id == 2:
            layer_id = stage_id + 1
        elif stage_id == 3:
            layer_id = 12
        return layer_id
    elif var_name.startswith("backbone.stages"):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0 or stage_id == 1:
            layer_id = stage_id + 1
        elif stage_id == 2:
            layer_id = 3 + block_id // div_val
        elif stage_id == 3:
            layer_id = 12
        return layer_id
    else:
        return num_max_layer + 1


class LayerDecayValueAssigner(object):
    def __init__(self, values, model_name, depth):
        self.values = values
        self.model_name = model_name
        self.depth = depth

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name, num_layers_s):
        if var_name.startswith("module."):
            var_name = var_name.replace("module.", "")
        if "convnext" in self.model_name.lower():
            return get_num_layer_for_convnext(var_name, self.depth)
        else:
            return get_num_layer_for_vit(var_name, len(self.values), num_layers_s)
