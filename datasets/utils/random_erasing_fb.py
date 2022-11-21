# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/random_erasing.py
pulished under an Apache License 2.0.
COMMENT FROM ORIGINAL:
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng
Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import random
import torch


def _get_pixels(
    per_pixel, rand_color, patch_size, dtype=torch.float32, device="cuda"
):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty(
            (patch_size[0], 1, 1), dtype=dtype, device=device
        ).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasingFB:
    """Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
        self,
        cfg,
        # probability=0.5,
        # min_area=0.02,
        # max_area=1 / 3,
        # min_aspect=0.3,
        # max_aspect=None,
        # mode="const",
        # min_count=1,
        # max_count=None,
        # num_splits=0,
        # device="cuda",
        # cube=True,
    ):
        self.enable                     = cfg.AUGMENTATION.RANDOM_ERASING.ENABLE
        self.probability                = cfg.AUGMENTATION.RANDOM_ERASING.PROB
        self.min_area, self.max_area    = cfg.AUGMENTATION.RANDOM_ERASING.AREA_RANGE
        min_aspect                      = cfg.AUGMENTATION.RANDOM_ERASING.MIN_ASPECT
        max_aspect                      = 1 / min_aspect
        self.log_aspect_ratio           = (math.log(min_aspect), math.log(max_aspect))
        self.min_count, self.max_count  = cfg.AUGMENTATION.RANDOM_ERASING.COUNT
        self.num_splits                 = cfg.AUGMENTATION.RANDOM_ERASING.NUM_SPLITS
        mode                            = cfg.AUGMENTATION.RANDOM_ERASING.MODE.lower()
        self.temporal_persis            = cfg.AUGMENTATION.RANDOM_ERASING.TEMPORAL_PERSIS if hasattr(cfg.AUGMENTATION.RANDOM_ERASING, "TEMPORAL_PERSIS") else False
        self.color_erasing            = cfg.AUGMENTATION.RANDOM_ERASING.COLOR_ERASING if hasattr(cfg.AUGMENTATION.RANDOM_ERASING, "COLOR_ERASING") else False
        self.rand_color = False
        self.per_pixel = False
        self.cube = True
        if mode == "rand":
            self.rand_color = True  # per block random normal
        elif mode == "pixel":
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == "const"

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        for _ in range(count):
            for _ in range(10):
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top : top + h, left : left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=img.device,
                    )
                    break

    def _erase_cube(
        self,
        img,
        batch_start,
        batch_size,
        chan,
        img_h,
        img_w,
        dtype,
    ):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        for _ in range(count):
            for _ in range(100):
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    if self.temporal_persis:
                        rand_pixels = _get_pixels(
                            self.per_pixel,
                            self.rand_color,
                            (chan, h, w),
                            dtype=dtype,
                            device=img.device,
                        )
                    for i in range(batch_start, batch_size):
                        img_instance = img[i]
                        if self.temporal_persis:
                            img_instance[
                                :, top : top + h, left : left + w
                            ] = rand_pixels
                        else:
                            img_instance[
                                :, top : top + h, left : left + w
                            ] = _get_pixels(
                                self.per_pixel,
                                self.rand_color,
                                (chan, h, w),
                                dtype=dtype,
                                device=img.device,
                            )
                    break

    def __call__(self, input):
        if self.enable:
            if len(input.size()) == 3:
                self._erase(input, *input.size(), input.dtype)
            else:
                if self.color_erasing:
                    pass
                else:
                    input = input.permute(1, 0, 2, 3)
                batch_size, chan, img_h, img_w = input.size()
                # skip first slice of batch if num_splits is set (for clean portion of samples)
                batch_start = (
                    batch_size // self.num_splits if self.num_splits > 1 else 0
                )
                if self.cube:
                    self._erase_cube(
                        input,
                        batch_start,
                        batch_size,
                        chan,
                        img_h,
                        img_w,
                        input.dtype,
                    )
                else:
                    for i in range(batch_start, batch_size):
                        self._erase(input[i], chan, img_h, img_w, input.dtype)
                if self.color_erasing:
                    pass
                else:
                    input = input.permute(1, 0, 2, 3)
        return input