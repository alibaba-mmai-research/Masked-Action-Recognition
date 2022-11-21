#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# From https://github.com/facebookresearch/SlowFast/blob/master/slowfast/utils/lr_policy.py

"""Learning rate policy."""

import math


def get_lr_at_epoch(cfg, cur_epoch):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (Config): global config object. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr = get_lr_func(cfg.OPTIMIZER.LR_POLICY)(cfg.OPTIMIZER, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg.OPTIMIZER.WARMUP_EPOCHS:
        lr_start = cfg.OPTIMIZER.WARMUP_START_LR
        lr_end = get_lr_func(cfg.OPTIMIZER.LR_POLICY)(
            cfg.OPTIMIZER, cfg.OPTIMIZER.WARMUP_EPOCHS
        )
        alpha = (lr_end - lr_start) / cfg.OPTIMIZER.WARMUP_EPOCHS
        lr = cur_epoch * alpha + lr_start
    return lr


def get_agent_lr_at_epoch(cfg, cur_epoch):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (Config): global config object. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    if not hasattr(cfg.AGENT.OPTIMIZER, "MIN_LR"):
        cfg.AGENT.OPTIMIZER.MIN_LR = cfg.OPTIMIZER.MIN_LR
    if not hasattr(cfg.AGENT.OPTIMIZER, "MAX_EPOCH"):
        cfg.AGENT.OPTIMIZER.MAX_EPOCH = cfg.OPTIMIZER.MAX_EPOCH
    lr = get_lr_func(cfg.AGENT.OPTIMIZER.LR_POLICY)(cfg.AGENT.OPTIMIZER, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg.AGENT.OPTIMIZER.WARMUP_EPOCHS:
        lr_start = cfg.AGENT.OPTIMIZER.WARMUP_START_LR
        lr_end = get_lr_func(cfg.AGENT.OPTIMIZER.LR_POLICY)(
            cfg.AGENT.OPTIMIZER, cfg.AGENT.OPTIMIZER.WARMUP_EPOCHS
        )
        alpha = (lr_end - lr_start) / cfg.AGENT.OPTIMIZER.WARMUP_EPOCHS
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cfg_opt, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (Config): global config object. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    assert cfg_opt.MIN_LR <= cfg_opt.BASE_LR
    return (
        (cfg_opt.BASE_LR - cfg_opt.MIN_LR)
        * (math.cos(math.pi * cur_epoch / cfg_opt.MAX_EPOCH) + 1.0)
        * 0.5 + cfg_opt.MIN_LR
    )


def lr_func_steps_with_relative_lrs(cfg_opt, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    steps with relative learning rate schedule.
    Args:
        cfg (Config): global config object. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    ind = get_step_index(cfg, cur_epoch)
    return cfg_opt.LRS[ind] * cfg_opt.BASE_LR


def get_step_index(cfg, cur_epoch):
    """
    Retrieves the lr step index for the given epoch.
    Args:
        cfg (Config): global config object. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    steps = cfg.OPTIMIZER.STEPS + [cfg.OPTIMIZER.MAX_EPOCH]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind - 1


def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]
