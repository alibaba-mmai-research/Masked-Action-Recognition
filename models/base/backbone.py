#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Backbone/Meta architectures. """

import torch
import torch.nn as nn
import torchvision
from utils.registry import Registry
from models.base.base_blocks import (
    Base3DResStage, STEM_REGISTRY, BRANCH_REGISTRY, InceptionBaseConv3D,
)
from models.utils.init_helper import _init_convnet_weights, _init_convnet_weights_v2

BACKBONE_REGISTRY = Registry("Backbone")

_n_conv_resnet = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}

_n_conv_x3d = {
    "S": (3, 5, 11, 7),
    "M": (3, 5, 11, 7),
    "XL": (5, 10, 25, 15),
}
@BACKBONE_REGISTRY.register()
class ResNet3D(nn.Module):
    """
    Meta architecture for 3D ResNet based models. 
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(ResNet3D, self).__init__()
        self._construct_backbone(cfg)

    def _construct_backbone(self, cfg):
        # ------------------- Stem -------------------
        self.conv1 = STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(cfg=cfg)

        (n1, n2, n3, n4) = _n_conv_resnet[cfg.VIDEO.BACKBONE.DEPTH]

        # ------------------- Main arch -------------------
        self.conv2 = Base3DResStage(
            cfg                     = cfg,
            num_blocks              = n1,
            stage_idx               = 1,
        )

        self.conv3 = Base3DResStage(
            cfg                     = cfg,
            num_blocks              = n2,
            stage_idx               = 2,
        )

        self.conv4 = Base3DResStage(
            cfg                     = cfg,
            num_blocks              = n3,
            stage_idx               = 3,
        )

        self.conv5 = Base3DResStage(
            cfg                     = cfg,
            num_blocks              = n4,
            stage_idx               = 4,
        )
        
        # perform initialization
        if cfg.VIDEO.BACKBONE.INITIALIZATION == "kaiming":
            _init_convnet_weights(self)
        elif cfg.VIDEO.BACKBONE.INITIALIZATION == "kaiming_v2":
            _init_convnet_weights_v2(self)
    
    def forward(self, x):
        if type(x) is list:
            x = x[0]
        elif isinstance(x, dict):
            x = x["video"]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def get_num_layers(self):
        return 4, 0

@BACKBONE_REGISTRY.register()
class Inception3D(nn.Module):
    """
    Backbone architecture for I3D/S3DG. 
    Modifed from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(Inception3D, self).__init__()
        _input_channel = cfg.DATA.NUM_INPUT_CHANNELS
        self._construct_backbone(
            cfg,
            _input_channel
        )

    def _construct_backbone(
        self, 
        cfg,
        input_channel
    ):
        # ------------------- Block 1 -------------------
        self.Conv_1a = BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(
            cfg, input_channel, 64, kernel_size=7, stride=2, padding=3
        )

        self.block1 = nn.Sequential(self.Conv_1a) # (64, 32, 112, 112)

        # ------------------- Block 2 -------------------
        self.MaxPool_2a = nn.MaxPool3d(
            kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)
        ) 
        self.Conv_2b = InceptionBaseConv3D(cfg, 64, 64, kernel_size=1, stride=1) 
        self.Conv_2c = BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(
            cfg, 64, 192, kernel_size=3, stride=1, padding=1
        ) 

        self.block2 = nn.Sequential(
            self.MaxPool_2a, # (64, 32, 56, 56)
            self.Conv_2b,    # (64, 32, 56, 56)
            self.Conv_2c)    # (192, 32, 56, 56)

        # ------------------- Block 3 -------------------
        self.MaxPool_3a = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) 
        self.Mixed_3b = InceptionBlock3D(cfg, in_planes=192, out_planes=[64, 96, 128, 16, 32, 32])
        self.Mixed_3c = InceptionBlock3D(cfg, in_planes=256, out_planes=[128, 128, 192, 32, 96, 64])

        self.block3 = nn.Sequential(
            self.MaxPool_3a,    # (192, 32, 28, 28)
            self.Mixed_3b,      # (256, 32, 28, 28)
            self.Mixed_3c)      # (480, 32, 28, 28)

        # ------------------- Block 4 -------------------
        self.MaxPool_4a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.Mixed_4b = InceptionBlock3D(cfg, in_planes=480, out_planes=[192, 96, 208, 16, 48, 64])
        self.Mixed_4c = InceptionBlock3D(cfg, in_planes=512, out_planes=[160, 112, 224, 24, 64, 64])
        self.Mixed_4d = InceptionBlock3D(cfg, in_planes=512, out_planes=[128, 128, 256, 24, 64, 64])
        self.Mixed_4e = InceptionBlock3D(cfg, in_planes=512, out_planes=[112, 144, 288, 32, 64, 64])
        self.Mixed_4f = InceptionBlock3D(cfg, in_planes=528, out_planes=[256, 160, 320, 32, 128, 128])

        self.block4 = nn.Sequential(
            self.MaxPool_4a,  # (480, 16, 14, 14)
            self.Mixed_4b,    # (512, 16, 14, 14)
            self.Mixed_4c,    # (512, 16, 14, 14)
            self.Mixed_4d,    # (512, 16, 14, 14)
            self.Mixed_4e,    # (528, 16, 14, 14)
            self.Mixed_4f)    # (832, 16, 14, 14)

        # ------------------- Block 5 -------------------
        self.MaxPool_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        self.Mixed_5b = InceptionBlock3D(cfg, in_planes=832, out_planes=[256, 160, 320, 32, 128, 128])
        self.Mixed_5c = InceptionBlock3D(cfg, in_planes=832, out_planes=[384, 192, 384, 48, 128, 128])

        self.block5 = nn.Sequential(
            self.MaxPool_5a,  # (832, 8, 7, 7)
            self.Mixed_5b,    # (832, 8, 7, 7)
            self.Mixed_5c)    # (1024, 8, 7, 7)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x 

@BACKBONE_REGISTRY.register()
class TSM(nn.Module):
    """
    Backbone architecture for TSM.

    Ji Lin, Chuang Gan, Song Han.
    TSM: Temporal Shift Module for Efficient Video Understanding
    
    Modified from https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/models.py.
    """
    def __init__(self, cfg,
                 before_softmax=True,
                 dropout=0.8):
        super(TSM, self).__init__()

        self.base_model_name = f"resnet{cfg.VIDEO.BACKBONE.DEPTH}"

        self.num_segments = cfg.DATA.NUM_INPUT_FRAMES
        self.before_softmax = before_softmax
        self.dropout = dropout

        self._prepare_base_model()

    def _prepare_base_model(self):
        self.base_model = getattr(torchvision.models, self.base_model_name)(False)
            
        make_temporal_shift(self.base_model, self.num_segments,
                            n_div=8, place='blockres', temporal_pool=False)

        setattr(self.base_model, "fc", nn.Identity())

        self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input, no_reshape=False):
        if isinstance(input, dict):
            x = input["video"].permute(0,2,1,3,4).contiguous()
        else:
            x = input.permute(0,2,1,3,4).contiguous()

        if not no_reshape:
            sample_len = 3

            x = x.view((-1, sample_len) + x.size()[-2:])

            base_out = self.base_model(x)
        else:
            base_out = self.base_model(x)

        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        output = base_out.mean(dim=1, keepdim=True)
        return output.squeeze(1)

@BACKBONE_REGISTRY.register()
class X3D(nn.Module):
    """
    Meta architecture for 3D ResNet based models. 
    """
    def __init__(self, cfg):
        super(X3D, self).__init__()
        self._construct_backbone(cfg)

    def _construct_backbone(self, cfg):
        # ------------------- Stem -------------------
        self.conv1 = STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(cfg=cfg)

        (n1, n2, n3, n4) = _n_conv_x3d[cfg.VIDEO.BACKBONE.DEPTH]

        # ------------------- Main arch -------------------
        self.conv2 = Base3DResStage(
            cfg                     = cfg,
            num_blocks              = n1,
            stage_idx               = 1,
        )

        self.conv3 = Base3DResStage(
            cfg                     = cfg,
            num_blocks              = n2,
            stage_idx               = 2,
        )

        self.conv4 = Base3DResStage(
            cfg                     = cfg,
            num_blocks              = n3,
            stage_idx               = 3,
        )

        self.conv5 = Base3DResStage(
            cfg                     = cfg,
            num_blocks              = n4,
            stage_idx               = 4,
        )
        self.last_conv = nn.Conv3d(
            in_channels =int(cfg.VIDEO.BACKBONE.NUM_FILTERS[-1]),
            out_channels=int(cfg.VIDEO.BACKBONE.NUM_FILTERS[-1] * cfg.VIDEO.BACKBONE.EXPANSION_RATIO),
            kernel_size =1,
            stride      =1,
            padding     =0,
            bias        =False
        )
        self.last_conv_bn = nn.BatchNorm3d(
            int(cfg.VIDEO.BACKBONE.NUM_FILTERS[-1] * cfg.VIDEO.BACKBONE.EXPANSION_RATIO),
            eps=cfg.BN.EPS,
            momentum=cfg.BN.MOMENTUM
        )
        self.last_conv_relu = nn.ReLU(inplace=True)
        _init_convnet_weights(self)
    
    def forward(self, x):
        if type(x) is list:
            x = x[0]
        elif isinstance(x, dict):
            x = x["video"]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.last_conv(x)
        x = self.last_conv_bn(x)
        x = self.last_conv_relu(x)
        return x

@BACKBONE_REGISTRY.register()
class ResNet2D(nn.Module):
    """
    Meta architecture for 3D ResNet based models. 
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(ResNet2D, self).__init__()
        self._construct_backbone(cfg)

    def _construct_backbone(self, cfg):
        # ------------------- Stem -------------------
        self.conv1 = STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(cfg=cfg)

        (n1, n2, n3, n4) = _n_conv_resnet[cfg.VIDEO.BACKBONE.DEPTH]

        # ------------------- Main arch -------------------
        self.conv2 = Base2DResStage(
            cfg                     = cfg,
            num_blocks              = n1,
            stage_idx               = 1,
        )

        self.conv3 = Base2DResStage(
            cfg                     = cfg,
            num_blocks              = n2,
            stage_idx               = 2,
        )

        self.conv4 = Base2DResStage(
            cfg                     = cfg,
            num_blocks              = n3,
            stage_idx               = 3,
        )

        self.conv5 = Base2DResStage(
            cfg                     = cfg,
            num_blocks              = n4,
            stage_idx               = 4,
        )
        
        # perform initialization
        if cfg.VIDEO.BACKBONE.INITIALIZATION == "kaiming":
            _init_convnet_weights(self)
        elif cfg.VIDEO.BACKBONE.INITIALIZATION == "kaiming_v2":
            _init_convnet_weights_v2(self)
    
    def forward(self, x):
        if type(x) is list:
            x = x[0]
        elif isinstance(x, dict):
            x = x["video"]

        B,C,T,H,W = x.size()
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(-1, C, H, W)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

@BACKBONE_REGISTRY.register()
class AudioVisualResNet(nn.Module):
    """
    Meta architecture for 3D ResNet based models. 
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(AudioVisualResNet, self).__init__()
        self.cfg = cfg
        self._construct_backbone(cfg)

    def _construct_backbone(self, cfg):
        # ------------------- Stem -------------------
        self.conv1 = STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(cfg=cfg)

        (n1, n2, n3, n4) = _n_conv_resnet[cfg.VIDEO.BACKBONE.DEPTH]

        # ------------------- Main arch -------------------
        self.conv2 = BaseAudioVisualStage(
            cfg                     = cfg,
            num_blocks              = n1,
            stage_idx               = 1,
        )

        self.conv3 = BaseAudioVisualStage(
            cfg                     = cfg,
            num_blocks              = n2,
            stage_idx               = 2,
        )

        self.conv4 = BaseAudioVisualStage(
            cfg                     = cfg,
            num_blocks              = n3,
            stage_idx               = 3,
        )

        self.conv5 = BaseAudioVisualStage(
            cfg                     = cfg,
            num_blocks              = n4,
            stage_idx               = 4,
        )
        
        # perform initialization
        if cfg.VIDEO.BACKBONE.INITIALIZATION == "kaiming":
            _init_convnet_weights(self)
        elif cfg.VIDEO.BACKBONE.INITIALIZATION == "kaiming_v2":
            _init_convnet_weights_v2(self)
    
    def forward(self, x):
        assert self.cfg.AUDIO.ENABLE 
        if isinstance(x, dict):
            xv = x["video"]
            xa = x["audio"]

        xv, xa = self.conv1(xv, xa)
        xv, xa = self.conv2(xv, xa)
        xv, xa = self.conv3(xv, xa)
        xv, xa = self.conv4(xv, xa)
        xv, xa = self.conv5(xv, xa)
        return {"video": xv, "audio": xa}

@BACKBONE_REGISTRY.register()
class AudioVisualResNetV2(nn.Module):
    """
    Meta architecture for 3D ResNet based models. 
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(AudioVisualResNetV2, self).__init__()
        self.cfg = cfg
        self._construct_backbone(cfg)

    def _construct_backbone(self, cfg):
        # ------------------- Stem -------------------
        self.conv1 = STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(cfg=cfg)

        (n1, n2, n3, n4) = _n_conv_resnet[cfg.VIDEO.BACKBONE.DEPTH]

        # ------------------- Main arch -------------------
        self.conv2 = BaseAudioVisualStageV2(
            cfg                     = cfg,
            num_blocks              = n1,
            stage_idx               = 1,
        )

        self.conv3 = BaseAudioVisualStageV2(
            cfg                     = cfg,
            num_blocks              = n2,
            stage_idx               = 2,
        )

        self.conv4 = BaseAudioVisualStageV2(
            cfg                     = cfg,
            num_blocks              = n3,
            stage_idx               = 3,
        )

        self.conv5 = BaseAudioVisualStageV2(
            cfg                     = cfg,
            num_blocks              = n4,
            stage_idx               = 4,
        )
        
        # perform initialization
        if cfg.VIDEO.BACKBONE.INITIALIZATION == "kaiming":
            _init_convnet_weights(self)
        elif cfg.VIDEO.BACKBONE.INITIALIZATION == "kaiming_v2":
            _init_convnet_weights_v2(self)
    
    def forward(self, x):
        assert self.cfg.AUDIO.ENABLE 
        if isinstance(x, dict):
            xv = x["video"]
            xa = x["audio"]

        xv, xa = self.conv1(xv, xa)
        xv, xa = self.conv2(xv, xa)
        xv, xa = self.conv3(xv, xa)
        xv, xa = self.conv4(xv, xa)
        xv, xa = self.conv5(xv, xa)
        return {"video": xv, "audio": xa}

@BACKBONE_REGISTRY.register()
class TAda2DEvolving(nn.Module):
    """
    Meta architecture for 3D ResNet based models. 
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(TAda2DEvolving, self).__init__()
        self._construct_backbone(cfg)

    def _construct_backbone(self, cfg):
        # ------------------- Stem -------------------
        self.conv1 = STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(cfg=cfg)

        (n1, n2, n3, n4) = _n_conv_resnet[cfg.VIDEO.BACKBONE.DEPTH]

        # ------------------- Main arch -------------------
        self.conv2 = EvolvingTAdaResStage(
            cfg                     = cfg,
            num_blocks              = n1,
            stage_idx               = 1,
        )

        self.conv3 = EvolvingTAdaResStage(
            cfg                     = cfg,
            num_blocks              = n2,
            stage_idx               = 2,
        )

        self.conv4 = EvolvingTAdaResStage(
            cfg                     = cfg,
            num_blocks              = n3,
            stage_idx               = 3,
        )

        self.conv5 = EvolvingTAdaResStage(
            cfg                     = cfg,
            num_blocks              = n4,
            stage_idx               = 4,
        )
        
        # perform initialization
        if cfg.VIDEO.BACKBONE.INITIALIZATION == "kaiming":
            _init_convnet_weights(self)
        elif cfg.VIDEO.BACKBONE.INITIALIZATION == "kaiming_v2":
            _init_convnet_weights_v2(self)
    
    def forward(self, x):
        if type(x) is list:
            x = x[0]
        elif isinstance(x, dict):
            x = x["video"]

        x = self.conv1(x)
        x, cal = self.conv2(x)
        x, cal = self.conv3(x, cal)
        x, cal = self.conv4(x, cal)
        x, cal = self.conv5(x, cal)
        return x


@BACKBONE_REGISTRY.register()
class ResNet2DPytorch(nn.Module):
    """
    Meta architecture for 3D ResNet based models. 
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(ResNet2DPytorch, self).__init__()
        self._construct_backbone(cfg)

    def _construct_backbone(self, cfg):
        # ------------------- Stem -------------------
        import torchvision
        resnet = torchvision.models.resnet18()
        self.conv1 = nn.Sequential(resnet.conv1,
                                   resnet.bn1,
                                   resnet.relu,
                                   resnet.maxpool)

        # ------------------- Main arch -------------------
        self.conv2 = resnet.layer1

        self.conv3 = resnet.layer2

        self.conv4 = resnet.layer3

        self.conv5 = resnet.layer4
        
        # perform initialization
        if cfg.VIDEO.BACKBONE.INITIALIZATION == "kaiming":
            _init_convnet_weights(self)
        elif cfg.VIDEO.BACKBONE.INITIALIZATION == "kaiming_v2":
            _init_convnet_weights_v2(self)
    
    def forward(self, x):
        if type(x) is list:
            x = x[0]
        elif isinstance(x, dict):
            x = x["video"]

        B,C,T,H,W = x.size()
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(-1, C, H, W)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        BT, D, SH, SW = x.size()
        x = x.view(B, T, D, SH, SW)
        x = x.permute(0, 2, 1, 3, 4)
        return x

