#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Params. """

import math 

def update_3d_conv_params(cfg, conv, idx):
    """
    Automatically decodes parameters for 3D convolution blocks according to the config and its index in the model.
    Args: 
        cfg (Config):       Config object that contains model parameters such as channel dimensions, whether to downsampling or not, etc.
        conv (BaseBranch):  Branch whose parameters needs to be specified. 
        idx (list):         List containing the index of the current block. ([stage_id, block_id])
    """
    # extract current block location
    stage_id, block_id  = idx
    conv.stage_id       = stage_id
    conv.block_id       = block_id

    # extract basic info
    if block_id == 0:
        conv.dim_in                 = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id-1]
        if hasattr(cfg.VIDEO.BACKBONE, "ADD_FUSION_CHANNEL") and cfg.VIDEO.BACKBONE.ADD_FUSION_CHANNEL:
            conv.dim_in = conv.dim_in * cfg.VIDEO.BACKBONE.SLOWFAST.CONV_CHANNEL_RATIO // cfg.VIDEO.BACKBONE.SLOWFAST.BETA + conv.dim_in
        conv.downsampling           = cfg.VIDEO.BACKBONE.DOWNSAMPLING[stage_id]
        conv.downsampling_temporal  = cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[stage_id]
    else:
        conv.downsampling           = False
        conv.dim_in                 = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id]
    conv.num_filters                = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id]
    conv.bn_mmt                     = cfg.BN.MOMENTUM
    conv.bn_eps                     = cfg.BN.EPS
    conv.kernel_size                = cfg.VIDEO.BACKBONE.KERNEL_SIZE[stage_id]
    conv.expansion_ratio            = cfg.VIDEO.BACKBONE.EXPANSION_RATIO if hasattr(cfg.VIDEO.BACKBONE, "EXPANSION_RATIO") else None

    # configure downsampling
    if conv.downsampling:
        if conv.downsampling_temporal:
            conv.stride = [2, 2, 2]
        else:
            conv.stride = [1, 2, 2]
    else:
        conv.stride = [1, 1, 1]

    # define transformation
    if isinstance(cfg.VIDEO.BACKBONE.DEPTH, str):
        conv.transformation = 'bottleneck'
    elif hasattr(cfg.VIDEO.BACKBONE, 'TRANSFORMATION'):
        conv.transformation = cfg.VIDEO.BACKBONE.TRANSFORMATION
    else:
        if cfg.VIDEO.BACKBONE.DEPTH <= 34:
            conv.transformation = 'simple_block'
        else:
            conv.transformation = 'bottleneck'

    # calculate the input size
    num_downsampling_spatial = sum(
        cfg.VIDEO.BACKBONE.DOWNSAMPLING[:stage_id+(block_id>0)]
    )
    if 'DownSample' in cfg.VIDEO.BACKBONE.STEM.NAME:
        num_downsampling_spatial += 1
    num_downsampling_temporal = sum(
        cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[:stage_id+(block_id>0)]
    )
    conv.h = cfg.DATA.TRAIN_CROP_SIZE // 2**num_downsampling_spatial \
        + (cfg.DATA.TRAIN_CROP_SIZE//2**(num_downsampling_spatial-1))%2
    conv.w = conv.h
    conv.t = cfg.DATA.NUM_INPUT_FRAMES // 2**num_downsampling_temporal


def update_av_conv_params(cfg, conv, idx):
    """
    Automatically decodes parameters for 3D convolution blocks according to the config and its index in the model.
    Args: 
        cfg (Config):       Config object that contains model parameters such as channel dimensions, whether to downsampling or not, etc.
        conv (BaseBranch):  Branch whose parameters needs to be specified. 
        idx (list):         List containing the index of the current block. ([stage_id, block_id])
    """
    # extract current block location
    stage_id, block_id  = idx
    conv.stage_id       = stage_id
    conv.block_id       = block_id

    # extract basic info
    if block_id == 0:
        conv.dim_in                         = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id-1]
        if hasattr(cfg.VIDEO.BACKBONE, "ADD_FUSION_CHANNEL") and cfg.VIDEO.BACKBONE.ADD_FUSION_CHANNEL:
            conv.dim_in = conv.dim_in * cfg.VIDEO.BACKBONE.SLOWFAST.CONV_CHANNEL_RATIO // cfg.VIDEO.BACKBONE.SLOWFAST.BETA + conv.dim_in
        conv.visual_downsampling            = cfg.VIDEO.BACKBONE.DOWNSAMPLING[stage_id]
        conv.visual_downsampling_temporal   = cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[stage_id]
        conv.audio_downsampling_temporal    = cfg.AUDIO.BACKBONE.DOWNSAMPLING_TEMPORAL[stage_id]
        conv.audio_downsampling_mel         = cfg.AUDIO.BACKBONE.DOWNSAMPLING_MEL[stage_id]
        conv.audio_downsampling             = conv.audio_downsampling_mel or conv.audio_downsampling_temporal
    else:
        conv.dim_in                 = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id]
        conv.visual_downsampling    = False
        conv.audio_downsampling     = False
        conv.audio_downsampling_temporal    = False
        conv.audio_downsampling_mel         = False
    conv.num_filters                = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id]
    conv.bn_mmt                     = cfg.BN.MOMENTUM
    conv.bn_eps                     = cfg.BN.EPS
    conv.visual_kernel_size         = cfg.VIDEO.BACKBONE.KERNEL_SIZE[stage_id]
    conv.audio_kernel_size         = cfg.AUDIO.BACKBONE.KERNEL_SIZE[stage_id]
    conv.expansion_ratio            = cfg.VIDEO.BACKBONE.EXPANSION_RATIO if hasattr(cfg.VIDEO.BACKBONE, "EXPANSION_RATIO") else None

    # configure downsampling
    if conv.visual_downsampling:
        if conv.visual_downsampling_temporal:
            conv.visual_stride = [2, 2, 2]
        else:
            conv.visual_stride = [1, 2, 2]
    else:
        conv.visual_stride = [1, 1, 1]
    
    conv.audio_stride = [
        2 if conv.audio_downsampling_temporal else 1,
        2 if conv.audio_downsampling_mel else 1
    ]

    # define transformation
    if isinstance(cfg.VIDEO.BACKBONE.DEPTH, str):
        conv.transformation = 'bottleneck'
    else:
        if cfg.VIDEO.BACKBONE.DEPTH <= 34:
            conv.transformation = 'simple_block'
        else:
            conv.transformation = 'bottleneck'

    num_downsampling_spatial = sum(
        cfg.VIDEO.BACKBONE.DOWNSAMPLING[:stage_id+(block_id>0)]
    )
    if 'DownSample' in cfg.VIDEO.BACKBONE.STEM.NAME:
        num_downsampling_spatial += 1
    
    conv.h = cfg.DATA.TRAIN_CROP_SIZE // 2**num_downsampling_spatial \
        + (cfg.DATA.TRAIN_CROP_SIZE//2**(num_downsampling_spatial-1))%2
    conv.w = conv.h
    
    conv.t = calculate_video_length(cfg, stage_id, block_id)
    conv.audio_t = calculate_audio_length(cfg, stage_id, block_id)

def update_av_conv_params_v2(cfg, conv, idx):
    """
    Automatically decodes parameters for 3D convolution blocks according to the config and its index in the model.
    Args: 
        cfg (Config):       Config object that contains model parameters such as channel dimensions, whether to downsampling or not, etc.
        conv (BaseBranch):  Branch whose parameters needs to be specified. 
        idx (list):         List containing the index of the current block. ([stage_id, block_id])
    """
    # extract current block location
    stage_id, block_id  = idx
    conv.stage_id       = stage_id
    conv.block_id       = block_id

    # extract basic info
    if block_id == 0:
        conv.visual_dim_in                  = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id-1]
        conv.visual_downsampling            = cfg.VIDEO.BACKBONE.DOWNSAMPLING[stage_id]
        conv.visual_downsampling_temporal   = cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[stage_id]

        conv.audio_dim_in                   = cfg.AUDIO.BACKBONE.NUM_FILTERS[stage_id-1]
        conv.audio_downsampling_temporal    = cfg.AUDIO.BACKBONE.DOWNSAMPLING_TEMPORAL[stage_id]
        conv.audio_downsampling_mel         = cfg.AUDIO.BACKBONE.DOWNSAMPLING_MEL[stage_id]
        conv.audio_downsampling             = conv.audio_downsampling_mel or conv.audio_downsampling_temporal
    else:
        conv.visual_dim_in          = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id]
        conv.visual_downsampling    = False
        conv.audio_downsampling     = False

        conv.audio_dim_in           = cfg.AUDIO.BACKBONE.NUM_FILTERS[stage_id]
        conv.audio_downsampling_temporal    = False
        conv.audio_downsampling_mel         = False
        
    conv.visual_num_filters         = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_id]
    conv.audio_num_filters          = cfg.AUDIO.BACKBONE.NUM_FILTERS[stage_id]
    conv.bn_mmt                     = cfg.BN.MOMENTUM
    conv.bn_eps                     = cfg.BN.EPS
    conv.visual_kernel_size         = cfg.VIDEO.BACKBONE.KERNEL_SIZE[stage_id]
    conv.audio_kernel_size         = cfg.AUDIO.BACKBONE.KERNEL_SIZE[stage_id]
    conv.expansion_ratio            = cfg.VIDEO.BACKBONE.EXPANSION_RATIO if hasattr(cfg.VIDEO.BACKBONE, "EXPANSION_RATIO") else None

    # configure downsampling
    if conv.visual_downsampling:
        if conv.visual_downsampling_temporal:
            conv.visual_stride = [2, 2, 2]
        else:
            conv.visual_stride = [1, 2, 2]
    else:
        conv.visual_stride = [1, 1, 1]
    
    conv.audio_stride = [
        2 if conv.audio_downsampling_temporal else 1,
        2 if conv.audio_downsampling_mel else 1
    ]

    # define transformation
    if isinstance(cfg.VIDEO.BACKBONE.DEPTH, str):
        conv.transformation = 'bottleneck'
    else:
        if cfg.VIDEO.BACKBONE.DEPTH <= 34:
            conv.transformation = 'simple_block'
        else:
            conv.transformation = 'bottleneck'

    num_downsampling_spatial = sum(
        cfg.VIDEO.BACKBONE.DOWNSAMPLING[:stage_id+(block_id>0)]
    )
    if 'DownSample' in cfg.VIDEO.BACKBONE.STEM.NAME:
        num_downsampling_spatial += 1
    
    conv.h = cfg.DATA.TRAIN_CROP_SIZE // 2**num_downsampling_spatial \
        + (cfg.DATA.TRAIN_CROP_SIZE//2**(num_downsampling_spatial-1))%2
    conv.w = conv.h
    
    conv.t = calculate_video_length(cfg, stage_id, block_id)
    conv.audio_t = calculate_audio_length(cfg, stage_id, block_id)

def calculate_video_length(cfg, stage_id, block_id):
    num_downsampling_temporal = sum(
        cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[:stage_id+(block_id>0)]
    )
    return cfg.DATA.NUM_INPUT_FRAMES // 2**num_downsampling_temporal


def calculate_audio_length(cfg, stage_id, block_id):
    assert stage_id > 0
    num_downsampling_audio_temporal = sum(
        cfg.AUDIO.BACKBONE.DOWNSAMPLING_TEMPORAL[:stage_id+(block_id>0)]
    ) - 1 * cfg.AUDIO.BACKBONE.DOWNSAMPLING_TEMPORAL[0]
    audio_base_shape = math.floor(
        int(cfg.DATA.NUM_INPUT_FRAMES * cfg.DATA.SAMPLING_RATE 
        / cfg.DATA.TARGET_FPS * cfg.AUDIO.SAMPLE_RATE)//(cfg.AUDIO.BACKBONE.DOWNSAMPLING_TEMPORAL[0]*1+1) 
        / cfg.AUDIO.BACKBONE.STEM.HOP_LENGTH
    )
    return math.ceil(
        audio_base_shape / 2 ** num_downsampling_audio_temporal
    )